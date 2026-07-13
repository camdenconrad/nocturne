//! Spotify Web API — the metadata half of Nocturne.
//!
//! librespot streams audio but is a poor fit for browsing: search, playlists and the saved-tracks
//! library all live in the Web API. Tokens come from the *existing* librespot session
//! (`NocturneHandle::web_token`), so there's no second OAuth flow and nothing extra to cache.
//!
//! Everything here returns plain owned structs — the UI never sees `serde` or `reqwest` types.

use serde::Deserialize;

const API: &str = "https://api.spotify.com/v1";

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    #[error("spotify returned {status}: {body}")]
    Status { status: u16, body: String },
}

/// A track as the UI wants it: already flattened, artists already joined.
#[derive(Debug, Clone)]
pub struct Track {
    pub uri: String,
    pub name: String,
    pub artists: String,
    pub album: String,
    pub duration_ms: u32,
    /// Smallest available cover, for the row thumbnail / now-playing art.
    pub art_url: Option<String>,
}

#[derive(Debug, Clone)]
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

    /// The user's saved ("Liked Songs") tracks — one page.
    pub async fn saved_tracks(&self, limit: u32, offset: u32) -> Result<Vec<Track>, ApiError> {
        let url = format!("{API}/me/tracks?limit={limit}&offset={offset}");
        let r: Page<SavedTrack> = self.get(&url).await?;
        Ok(r.items.into_iter().filter_map(|s| s.track).map(Into::into).collect())
    }

    pub async fn playlists(&self, limit: u32) -> Result<Vec<Playlist>, ApiError> {
        let url = format!("{API}/me/playlists?limit={limit}");
        // Spotify sprinkles nulls and partial objects through this page (dead collaborative
        // playlists, ones with no `tracks` block). One bad entry must not lose the whole library.
        let r: Page<Option<RawPlaylist>> = self.get(&url).await?;
        Ok(r.items
            .into_iter()
            .flatten()
            .map(|p| Playlist {
                id: p.id,
                name: p.name,
                tracks: p.tracks.map(|t| t.total),
            })
            .collect())
    }

    pub async fn playlist_tracks(&self, id: &str, limit: u32) -> Result<Vec<Track>, ApiError> {
        let url = format!("{API}/playlists/{id}/tracks?limit={limit}");
        let r: Page<PlaylistItem> = self.get(&url).await?;
        // Local files and removed tracks come back as null — skip rather than blow up the page.
        Ok(r.items.into_iter().filter_map(|i| i.track).map(Into::into).collect())
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
}

#[derive(Deserialize)]
struct SavedTrack {
    track: Option<RawTrack>,
}

#[derive(Deserialize)]
struct PlaylistItem {
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
