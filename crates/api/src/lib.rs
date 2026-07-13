//! Spotify Web API — the metadata half of Nocturne.
//!
//! librespot streams audio but is a poor fit for browsing: search, playlists and the saved-tracks
//! library all live in the Web API. Tokens come from the *existing* librespot session
//! (`NocturneHandle::web_token`), so there's no second OAuth flow and nothing extra to cache.
//!
//! Everything here returns plain owned structs — the UI never sees `serde` or `reqwest` types.

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

const API: &str = "https://api.spotify.com/v1";

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    #[error("spotify returned {status}: {body}")]
    Status { status: u16, body: String },
}

/// A track as the UI wants it: already flattened, artists already joined.
/// Serializable so the library can be cached to disk and shown instantly on next launch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Track {
    pub uri: String,
    pub name: String,
    pub artists: String,
    pub album: String,
    pub duration_ms: u32,
    /// Smallest available cover, for the row thumbnail / now-playing art.
    pub art_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Playlist {
    pub id: String,
    pub name: String,
    /// `None` when Spotify omits the count — which it currently does on `/me/playlists`. Don't
    /// coerce that to 0: a confident wrong number is worse than no number.
    pub tracks: Option<u32>,
}

pub struct Client {
    http: reqwest::Client,
    token: String,
}

impl Client {
    pub fn token(&self) -> &str {
        &self.token
    }

    pub fn new(token: String) -> Self {
        Self {
            http: reqwest::Client::new(),
            token,
        }
    }

    async fn get<T: for<'de> Deserialize<'de>>(&self, url: &str) -> Result<T, ApiError> {
        let resp = self
            .http
            .get(url)
            .bearer_auth(&self.token)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(ApiError::Status {
                status: status.as_u16(),
                body,
            });
        }
        Ok(resp.json().await?)
    }

    pub async fn search_tracks(&self, query: &str, limit: u32) -> Result<Vec<Track>, ApiError> {
        let q = urlencode(query);
        let url = format!("{API}/search?q={q}&type=track&limit={limit}");
        let r: SearchResp = self.get(&url).await?;
        Ok(r.tracks.items.into_iter().map(Into::into).collect())
    }

    /// Fetch every page of an offset-paged endpoint **concurrently**.
    ///
    /// Spotify caps `limit` at 50, so a 2000-track library is 40 requests. Walked serially (which
    /// is what `next`-chasing forces) that's 40 round trips end-to-end and the UI sits empty for
    /// all of them. The first response carries `total`, so every remaining offset can be issued at
    /// once — the whole library then costs about as long as its slowest single request.
    async fn get_paged<T: for<'de> Deserialize<'de>>(
        &self,
        base: &str,
        max: usize,
    ) -> Result<Vec<T>, ApiError> {
        const LIMIT: u32 = 50;
        let sep = if base.contains('?') { '&' } else { '?' };

        let first: Page<T> = self.get(&format!("{base}{sep}limit={LIMIT}&offset=0")).await?;
        let total = first.total.unwrap_or(first.items.len() as u32) as usize;
        let want = total.min(max);
        let mut out = first.items;
        if out.len() >= want {
            out.truncate(want);
            return Ok(out);
        }

        let offsets: Vec<u32> = (out.len()..want).step_by(LIMIT as usize).map(|o| o as u32).collect();
        let pages: Vec<Result<Vec<T>, ApiError>> = futures_util::stream::iter(offsets)
            .map(|off| async move {
                let p: Page<T> = self
                    .get(&format!("{base}{sep}limit={LIMIT}&offset={off}"))
                    .await?;
                Ok(p.items)
            })
            .buffered(8)
            .collect()
            .await;

        for page in pages {
            out.extend(page?);
        }
        out.truncate(want);
        Ok(out)
    }

    /// Every saved ("Liked Songs") track.
    pub async fn saved_tracks(&self, max: usize) -> Result<Vec<Track>, ApiError> {
        let items: Vec<SavedTrack> = self.get_paged(&format!("{API}/me/tracks"), max).await?;
        Ok(items.into_iter().filter_map(|s| s.track).map(Into::into).collect())
    }

    /// Every playlist the user owns or follows.
    ///
    /// Spotify sprinkles nulls and partial objects through this page (dead collaborative
    /// playlists, ones with no `tracks` block). One bad entry must not lose the whole library.
    pub async fn playlists(&self, max: usize) -> Result<Vec<Playlist>, ApiError> {
        let items: Vec<Option<RawPlaylist>> = self
            .get_paged(&format!("{API}/me/playlists"), max)
            .await?;
        Ok(items
            .into_iter()
            .flatten()
            .map(|p| Playlist {
                id: p.id,
                name: p.name,
                tracks: p.tracks.map(|t| t.total),
            })
            .collect())
    }

    /// Raw image bytes for a cover URL (art lives on a CDN, not the API host).
    pub async fn fetch_art(&self, url: &str) -> Result<Vec<u8>, ApiError> {
        let resp = self.http.get(url).send().await?;
        Ok(resp.bytes().await?.to_vec())
    }
}

/// Minimal percent-encoding for the query string — avoids a dep for the handful of chars that
/// actually break a Spotify search (`&`, `#`, `+`, spaces…).
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

// ---- wire types ----

#[derive(Deserialize)]
struct SearchResp {
    tracks: Page<RawTrack>,
}

#[derive(Deserialize)]
struct Page<T> {
    items: Vec<T>,
    /// Absolute URL of the next page, or null on the last one. Spotify caps `limit` at 50, so
    /// anything that ignores this silently truncates a real library.
    #[serde(default)]
    next: Option<String>,
    /// Total across all pages — lets us fire every page at once instead of walking them serially.
    #[serde(default)]
    total: Option<u32>,
}

#[derive(Deserialize)]
struct SavedTrack {
    track: Option<RawTrack>,
}

#[derive(Deserialize)]
struct RawPlaylist {
    id: String,
    name: String,
    tracks: Option<TrackCount>,
}

#[derive(Deserialize)]
struct TrackCount {
    total: u32,
}

#[derive(Deserialize)]
struct RawTrack {
    uri: String,
    name: String,
    artists: Vec<Named>,
    album: Album,
    duration_ms: u32,
}

#[derive(Deserialize)]
struct Named {
    name: String,
}

#[derive(Deserialize)]
struct Album {
    name: String,
    images: Vec<Image>,
}

#[derive(Deserialize)]
struct Image {
    url: String,
    width: Option<u32>,
}

impl From<RawTrack> for Track {
    fn from(t: RawTrack) -> Self {
        // Spotify sorts images largest-first; take the smallest for thumbnails.
        let art_url = t
            .album
            .images
            .iter()
            .min_by_key(|i| i.width.unwrap_or(u32::MAX))
            .map(|i| i.url.clone());
        Track {
            uri: t.uri,
            name: t.name,
            artists: t
                .artists
                .iter()
                .map(|a| a.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            album: t.album.name,
            duration_ms: t.duration_ms,
            art_url,
        }
    }
}

pub fn fmt_duration(ms: u32) -> String {
    let total = ms / 1000;
    format!("{}:{:02}", total / 60, total % 60)
}
