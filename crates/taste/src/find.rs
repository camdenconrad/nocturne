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

    hits.into_iter().find(|t| {
        let artist_ok = norm(&t.artists).contains(&norm(&s.artist))
            || norm(&s.artist).contains(&norm(&t.artists));
        // The title check is prefix-based on purpose: Spotify titles carry suffixes the model
        // won't have predicted — "(Remastered 2011)", "- Live at Wembley".
        let title_ok = norm(&t.name).starts_with(&norm(&s.title))
            || norm(&s.title).starts_with(&norm(&t.name));
        artist_ok && title_ok
    })
}

/// Lowercase, strip punctuation and collapse whitespace, so "Don't Look Back" and "dont look back"
/// compare equal.
fn norm(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut space = false;
    for c in s.chars() {
        if c.is_alphanumeric() {
            space = false;
            out.extend(c.to_lowercase());
        } else if !out.is_empty() && !space {
            space = true;
            out.push(' ');
        }
    }
    out.trim_end().to_string()
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

    // Resolve concurrently, but bounded — see RESOLVE_CONCURRENCY.
    let picked: Vec<Track> = futures_util::stream::iter(suggestions)
        .map(|s| async move { resolve(api, &s).await })
        .buffered(RESOLVE_CONCURRENCY)
        .filter_map(|hit| async move { hit })
        .collect()
        .await;
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
