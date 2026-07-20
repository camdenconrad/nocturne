//! Turning a free-text request into a pool of real, playable tracks.
//!
//! Three steps, and the middle one is the one that matters:
//!
//! 1. [`llm::suggest_tracks`] names tracks. These are *names* — a model naming music from memory
//!    will occasionally name something that doesn't exist, or exists under a different title.
//! 2. Every name is looked up against Spotify. Whatever doesn't resolve is dropped. This is what
//!    keeps a hallucinated track from ever reaching the queue: nothing is playable until Spotify
//!    has agreed it exists.
//! 3. The survivors are merged with a plain keyword search of the same query, so a request always
//!    returns *something* even when Claude is unavailable, slow, or wrong.
//!
//! Ordering is deliberately not decided here. This module produces a pool; [`crate::Taste::rank`]
//! orders it, because what Camden actually listens to is better evidence than what a model thinks
//! fits the words.

use crate::llm::{self, Suggestion};
use nocturne_api::{Client, Track};
use futures_util::StreamExt;
use std::collections::HashSet;

/// How many lookups run at once. Spotify rate-limits per-app, and a burst of 25 parallel searches
/// is the kind of thing that earns a 429 for the whole session, not just this query.
const RESOLVE_CONCURRENCY: usize = 4;

/// What a single request produced, with enough detail to tell a bad Claude call from a bad query.
pub struct Pool {
    pub tracks: Vec<Track>,
    /// How many names Claude offered.
    pub suggested: usize,
    /// How many of those matched a real track.
    pub resolved: usize,
}

impl Pool {
    /// Share of Claude's names that turned out to exist. `None` when Claude wasn't consulted.
    pub fn resolve_rate(&self) -> Option<f32> {
        (self.suggested > 0).then(|| self.resolved as f32 / self.suggested as f32)
    }
}

/// Match one suggestion against the catalogue.
///
/// Searches by field (`track:… artist:…`) rather than pasting both into a bare query, so "Zero"
/// by "Yeah Yeah Yeahs" can't be answered by a track called "Yeah Yeah Yeahs". A hit still has to
/// look like what was asked for — Spotify always returns *something*, and its something for a
/// track that doesn't exist is an unrelated track, not an empty list.
async fn resolve(api: &Client, s: &Suggestion) -> Option<Track> {
    let query = format!("track:{} artist:{}", s.title, s.artist);
    let hits = api.search_tracks(&query, 5).await.ok()?;

    hits.into_iter()
        .find(|t| is_match(&t.artists, &t.name, &s.artist, &s.title))
}

/// Does a catalogue hit actually correspond to what was asked for?
///
/// Artist is a containment check because Spotify joins collaborators into one field ("Sonic Youth,
/// Kim Gordon") while the model names one. Title is prefix-based because Spotify titles carry
/// suffixes the model won't have predicted — "(Remastered 2011)", "- Live at Wembley" — but a
/// prefix in *either* direction, so an abbreviated guess still matches its fuller real title.
fn is_match(track_artists: &str, track_name: &str, want_artist: &str, want_title: &str) -> bool {
    let (ta, tn) = (norm(track_artists), norm(track_name));
    let (wa, wt) = (norm(want_artist), norm(want_title));
    if ta.is_empty() || tn.is_empty() || wa.is_empty() || wt.is_empty() {
        return false;
    }
    let artist_ok = ta.contains(&wa) || wa.contains(&ta);
    let title_ok = tn.starts_with(&wt) || wt.starts_with(&tn);
    artist_ok && title_ok
}

/// Lowercase, strip punctuation and collapse whitespace, so "Don't Look Back" and "dont look back"
/// compare equal.
///
/// Apostrophes are *deleted* rather than treated as separators — "Don't" has to normalise to
/// "dont", not "don t", or every contraction fails to match. Everything else non-alphanumeric
/// becomes a space, because a comma genuinely does separate two artists.
fn norm(s: &str) -> String {
    const ELIDED: [char; 4] = ['\'', '\u{2019}', '`', '\u{00B4}'];
    let mut out = String::with_capacity(s.len());
    let mut space = false;
    for c in s.chars() {
        if c.is_alphanumeric() {
            space = false;
            out.extend(c.to_lowercase());
        } else if ELIDED.contains(&c) {
            // Skip entirely — no separator.
        } else if !out.is_empty() && !space {
            space = true;
            out.push(' ');
        }
    }
    out.trim_end().to_string()
}

/// Resolve a batch of suggestions into real tracks, dropping whatever doesn't exist.
async fn resolve_all(api: &Client, suggestions: Vec<Suggestion>) -> Vec<Track> {
    futures_util::stream::iter(suggestions)
        .map(|s| async move { resolve(api, &s).await })
        .buffered(RESOLVE_CONCURRENCY)
        .filter_map(|hit| async move { hit })
        .collect()
        .await
}

/// Extend a station from what just played.
///
/// Returns empty when Claude is unavailable or nothing resolves — the caller then falls back to
/// Spotify's own radio, so autoplay never stalls on this. `exclude` is the URIs already in the
/// queue, so a refill can't hand back something queued or just heard.
pub async fn station(
    api: &Client,
    recent: &[(String, String)],
    exclude: &HashSet<String>,
    want: usize,
) -> Vec<Track> {
    if !llm::available() {
        return Vec::new();
    }
    let suggestions = llm::continue_station(recent, want).await;
    let suggested = suggestions.len();
    if suggested == 0 {
        return Vec::new();
    }

    let mut seen = HashSet::new();
    let tracks: Vec<Track> = resolve_all(api, suggestions)
        .await
        .into_iter()
        .filter(|t| !exclude.contains(&t.uri) && seen.insert(t.uri.clone()))
        .collect();

    tracing::info!("claude radio: {}/{suggested} resolved and new", tracks.len());
    tracks
}

/// Build the candidate pool for a request.
///
/// Never fails: with no API key, a Claude error, or nothing resolving, this degrades to exactly
/// the plain keyword search the app did before. A search box that errors is worse than a search
/// box that returns keyword hits.
pub async fn pool(api: &Client, query: &str, want: usize) -> Pool {
    let suggestions = if llm::available() {
        llm::suggest_tracks(query, want).await
    } else {
        Vec::new()
    };
    let suggested = suggestions.len();

    let picked = resolve_all(api, suggestions).await;
    let resolved = picked.len();

    if suggested > 0 {
        tracing::info!(
            "claude: {resolved}/{suggested} suggestions resolved ({:.0}%)",
            (resolved as f32 / suggested as f32) * 100.0
        );
    }

    // Keyword hits always come too — they're the floor when Claude is unavailable, and the
    // backfill when only a few names resolved.
    let keyword = api.search_tracks(query, want).await.unwrap_or_else(|e| {
        tracing::warn!("search: keyword pass failed ({e})");
        Vec::new()
    });

    // Claude's picks first so they survive the truncation below; dedupe by uri.
    let mut seen = HashSet::new();
    let tracks: Vec<Track> = picked
        .into_iter()
        .chain(keyword)
        .filter(|t| seen.insert(t.uri.clone()))
        .take(want)
        .collect();

    Pool { tracks, suggested, resolved }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalises_punctuation_and_case() {
        assert_eq!(norm("Don't Look Back"), "dont look back");
        assert_eq!(norm("  Spaced   Out!  "), "spaced out");
        assert_eq!(norm("Sigur Rós"), "sigur rós");
        // Curly apostrophes too — Spotify uses them, models tend to type straight ones.
        assert_eq!(norm("Don\u{2019}t Look Back"), "dont look back");
        // A comma really does separate two artists, so it must stay a boundary.
        assert_eq!(norm("Sonic Youth, Kim Gordon"), "sonic youth kim gordon");
    }

    #[test]
    fn accepts_a_real_hit() {
        assert!(is_match("Slowdive", "Alison", "Slowdive", "Alison"));
        assert!(is_match("slowdive", "ALISON", "Slowdive", "alison"));
    }

    #[test]
    fn tolerates_spotify_title_suffixes() {
        assert!(is_match("Radiohead", "Creep - Remastered 2011", "Radiohead", "Creep"));
        assert!(is_match("Nirvana", "Come As You Are (Live)", "Nirvana", "Come As You Are"));
    }

    #[test]
    fn tolerates_collaborator_lists() {
        // Spotify joins every credited artist; the model names the primary one.
        assert!(is_match("Sonic Youth, Kim Gordon", "Bull In The Heather", "Sonic Youth", "Bull In The Heather"));
    }

    // The whole point of the resolution pass: a plausible-but-wrong hit must not be accepted.
    #[test]
    fn rejects_a_different_track_by_the_right_artist() {
        assert!(!is_match("Radiohead", "Karma Police", "Radiohead", "Creep"));
    }

    #[test]
    fn rejects_the_right_title_by_the_wrong_artist() {
        assert!(!is_match("Stone Temple Pilots", "Creep", "Radiohead", "Creep"));
    }

    #[test]
    fn rejects_empty_fields() {
        // An empty title would otherwise prefix-match everything.
        assert!(!is_match("Radiohead", "Creep", "Radiohead", ""));
        assert!(!is_match("", "Creep", "Radiohead", "Creep"));
    }
}
