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
    #[error("Spotify rate limit — retry in {}h. Using cached library.", retry_after / 3600)]
    RateLimited { retry_after: u64 },
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
    /// Smallest available cover — for row thumbnails, where a 640px image is a waste.
    pub art_url: Option<String>,
    /// LARGEST available cover (typically 640×640) — for the full-screen view, which was upscaling
    /// the 64px thumbnail and looked exactly as bad as that sounds.
    #[serde(default)]
    pub art_big: Option<String>,
    /// Signals for the taste model. Optional because the two metadata sources (Web API and
    /// librespot) don't always carry them.
    #[serde(default)]
    pub popularity: Option<u32>,
    #[serde(default)]
    pub explicit: Option<bool>,
}

/// Spotify's real audio features — the ones that make a taste model actually work.
///
/// NOT available from the Web API: `/v1/audio-features` is 403 for post-2024 apps. These come from
/// the *internal* `/audio-attributes/v1/audio-features/{id}` service instead (see
/// `nocturne_session::NocturneHandle::audio_features`). Immutable per track, so they cache forever.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFeatures {
    pub danceability: f32,
    pub energy: f32,
    pub valence: f32,
    pub acousticness: f32,
    pub instrumentalness: f32,
    pub speechiness: f32,
    pub liveness: f32,
    /// dB, typically -60..0.
    pub loudness: f32,
    /// BPM.
    pub tempo: f32,
    /// Pitch class 0-11, or -1 when undetected.
    pub key: i32,
    /// 1 = major, 0 = minor.
    pub mode: i32,
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

    /// GET with 429 handling.
    ///
    /// Spotify rate-limits, and it tells you for how long in `Retry-After`. Firing 8 pages at once
    /// (which is how the library used to load) reliably tripped it and surfaced as a bare
    /// "429: Too many requests" in the UI. Honour the header, retry a few times, and only then give
    /// up — a rate limit is a "wait", not an error.
    async fn get<T: for<'de> Deserialize<'de>>(&self, url: &str) -> Result<T, ApiError> {
        for attempt in 0..5u32 {
            let resp = self.http.get(url).bearer_auth(&self.token).send().await?;
            let status = resp.status();

            if status.as_u16() == 429 {
                let wait = resp
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse::<u64>().ok())
                    // No header: back off exponentially rather than guessing small.
                    .unwrap_or(1 << attempt);

                // Spotify's rate limits come in two flavours. A short one (seconds) is a "slow
                // down" and is worth waiting out. A long one — it will happily hand back 80,000+
                // seconds, i.e. a day — is a penalty box, and sleeping through it is not a thing an
                // app can do. Fail immediately so the caller falls back to the disk cache instead
                // of hanging, and say so in a way the user can act on.
                if wait > 60 {
                    tracing::error!(
                        "spotify rate limit: locked out for {wait}s (~{}h) — serving from cache",
                        wait / 3600
                    );
                    return Err(ApiError::RateLimited { retry_after: wait });
                }

                tracing::warn!("spotify rate limit — waiting {wait}s (attempt {})", attempt + 1);
                tokio::time::sleep(std::time::Duration::from_secs(wait)).await;
                continue;
            }

            if !status.is_success() {
                let body = resp.text().await.unwrap_or_default();
                return Err(ApiError::Status {
                    status: status.as_u16(),
                    body,
                });
            }
            return Ok(resp.json().await?);
        }
        Err(ApiError::RateLimited { retry_after: 0 })
    }

    /// Spotify caps search `limit` at **10** for restricted apps like ours — not the 50 the docs
    /// advertise, and anything over 10 is a bare `400 Invalid limit`. So ask for pages of 10 and
    /// fetch them concurrently to still fill a screen.
    pub async fn search_tracks(&self, query: &str, want: usize) -> Result<Vec<Track>, ApiError> {
        const SEARCH_LIMIT: usize = 10;
        let q = urlencode(query);
        let pages = want.div_ceil(SEARCH_LIMIT);
        let offsets: Vec<usize> = (0..pages).map(|p| p * SEARCH_LIMIT).collect();

        let results: Vec<Result<Vec<Track>, ApiError>> = futures_util::stream::iter(offsets)
            .map(|off| {
                let q = q.clone();
                async move {
                    let url = format!(
                        "{API}/search?q={q}&type=track&limit={SEARCH_LIMIT}&offset={off}"
                    );
                    let r: SearchResp = self.get(&url).await?;
                    Ok(r.tracks.items.into_iter().map(Into::into).collect::<Vec<Track>>())
                }
            })
            .buffered(2)
            .collect()
            .await;

        let mut out = Vec::new();
        for page in results {
            match page {
                Ok(t) => out.extend(t),
                // A later page 404ing/erroring shouldn't throw away the hits we already have.
                Err(e) => {
                    tracing::warn!("search page failed: {e}");
                    break;
                }
            }
        }
        out.truncate(want);
        Ok(out)
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
            .buffered(3)
            .collect()
            .await;

        for page in pages {
            match page {
                Ok(items) => out.extend(items),
                // Same rule as search: a later page failing must not throw away everything
                // already fetched. Keep the partial library and say so.
                Err(e) => {
                    tracing::warn!("library page failed — keeping {} items fetched so far: {e}", out.len());
                    break;
                }
            }
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

    /// Create a real Spotify playlist and fill it.
    ///
    /// Untested against a live account: Spotify has us rate-limited for ~21h as I write this, and
    /// library writes (`PUT /me/tracks`) are 403 for restricted apps — playlist writes may well be
    /// too. So this reports its failure honestly instead of pretending; the local copy is always
    /// kept regardless.
    pub async fn create_playlist(
        &self,
        name: &str,
        uris: &[String],
    ) -> Result<String, ApiError> {
        let me: serde_json::Value = self.get(&format!("{API}/me")).await?;
        let uid = me["id"].as_str().unwrap_or_default().to_string();
        if uid.is_empty() {
            return Err(ApiError::Status {
                status: 0,
                body: "could not resolve user id".into(),
            });
        }

        let resp = self
            .http
            .post(format!("{API}/users/{uid}/playlists"))
            .bearer_auth(&self.token)
            .json(&serde_json::json!({
                "name": name,
                "public": false,
                "description": "Created by Nocturne",
            }))
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            return Err(ApiError::Status {
                status: status.as_u16(),
                body: resp.text().await.unwrap_or_default(),
            });
        }
        let created: serde_json::Value = resp.json().await?;
        let pid = created["id"].as_str().unwrap_or_default().to_string();

        // Spotify takes 100 uris per request.
        for chunk in uris.chunks(100) {
            let r = self
                .http
                .post(format!("{API}/playlists/{pid}/tracks"))
                .bearer_auth(&self.token)
                .json(&serde_json::json!({ "uris": chunk }))
                .send()
                .await?;
            if !r.status().is_success() {
                return Err(ApiError::Status {
                    status: r.status().as_u16(),
                    body: r.text().await.unwrap_or_default(),
                });
            }
        }
        Ok(pid)
    }

    /// Raw image bytes for a cover URL (art lives on a CDN, not the API host).
    pub async fn fetch_art(&self, url: &str) -> Result<Vec<u8>, ApiError> {
        // Covers are ~100 KB; cap the download so a hostile/broken URL can't balloon RAM.
        const ART_MAX_BYTES: usize = 20 * 1024 * 1024;
        let resp = self.http.get(url).send().await?;
        let mut out = Vec::new();
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            if out.len() + chunk.len() > ART_MAX_BYTES {
                tracing::warn!("album art from {url} exceeds {ART_MAX_BYTES} bytes — truncating download");
                break;
            }
            out.extend_from_slice(&chunk);
        }
        Ok(out)
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
    #[serde(default)]
    popularity: Option<u32>,
    #[serde(default)]
    explicit: Option<bool>,
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
        let art_big = t
            .album
            .images
            .iter()
            .max_by_key(|i| i.width.unwrap_or(0))
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
            art_big,
            popularity: t.popularity,
            explicit: t.explicit,
        }
    }
}

pub fn fmt_duration(ms: u32) -> String {
    let total = ms / 1000;
    format!("{}:{:02}", total / 60, total % 60)
}
