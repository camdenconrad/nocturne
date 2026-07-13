//! Session layer: owns the librespot session, player, and Connect device.
//! The UI talks to this through `NocturneHandle` — it never touches librespot types.

use librespot_core::authentication::Credentials;
use librespot_core::cache::Cache;
use librespot_core::config::SessionConfig;
use librespot_core::session::Session;
use librespot_core::{SpotifyId, SpotifyUri};
use librespot_playback::config::PlayerConfig;
use librespot_playback::mixer::softmixer::SoftMixer;
use librespot_playback::mixer::{Mixer, MixerConfig};
use librespot_playback::player::Player;

/// Consent is granted once, so ask for everything v1 needs up front — adding a scope later forces
/// the user through the browser again. `streaming` is playback; the rest are the library/search UI.
const OAUTH_SCOPES: &[&str] = &[
    "streaming",
    "user-read-private",
    "user-read-email",
    "user-library-read",
    "playlist-read-private",
    "playlist-read-collaborative",
    "user-top-read",
    "user-read-recently-played",
];

/// Scopes handed to the Web API token (same set, comma-joined as the token endpoint wants).
pub const WEB_API_SCOPES: &str = "user-read-private,user-library-read,playlist-read-private,playlist-read-collaborative,user-top-read,user-read-recently-played";

/// The Spotify app's client id, from the environment or a `.env` beside the binary/repo.
/// No secret is needed anywhere — librespot's OAuth is PKCE.
fn client_id() -> Result<String, SessionError> {
    if let Ok(id) = std::env::var("NOCTURNE_CLIENT_ID") {
        return Ok(id);
    }
    for dir in [".", env!("CARGO_MANIFEST_DIR"), concat!(env!("CARGO_MANIFEST_DIR"), "/../..")] {
        if let Ok(text) = std::fs::read_to_string(format!("{dir}/.env")) {
            if let Some(id) = text
                .lines()
                .filter_map(|l| l.trim().strip_prefix("NOCTURNE_CLIENT_ID="))
                .next()
            {
                return Ok(id.trim().trim_matches(['"', '\'']).to_string());
            }
        }
    }
    Err(SessionError::OAuth(
        "no NOCTURNE_CLIENT_ID — put it in Nocturne/.env (see .env.example) or export it. \
         Get one at https://developer.spotify.com/dashboard with redirect URI \
         http://127.0.0.1:8898/login"
            .into(),
    ))
}

/// Tracks per extended-metadata request. Spotify's own client batches; 100 is comfortably within
/// what the endpoint accepts and keeps a 300-track playlist to three round trips.
const BATCH: usize = 100;

/// librespot's metadata `Track` → the flat shape the UI already renders.
fn to_api_track(t: librespot_metadata::Track) -> nocturne_api::Track {
    let artists = t
        .artists_with_role
        .iter()
        .map(|a| a.name.as_str())
        .collect::<Vec<_>>()
        .join(", ");

    // Covers are file ids, not URLs; the CDN path is the base16 of the id. Pick the smallest
    // for thumbnails, same rule as the Web API path.
    let art_url = t
        .album
        .covers
        .iter()
        .min_by_key(|i| i.width)
        .map(|i| format!("https://i.scdn.co/image/{}", i.id.to_base16()));

    nocturne_api::Track {
        uri: format!("spotify:track:{}", track_id_base62(&t.id)),
        name: t.name,
        artists,
        album: t.album.name,
        duration_ms: t.duration.max(0) as u32,
        art_url,
        popularity: Some(t.popularity.clamp(0, 100) as u32),
        explicit: Some(t.is_explicit),
    }
}

fn track_id_base62(uri: &SpotifyUri) -> String {
    match uri {
        SpotifyUri::Track { id } => id.to_base62(),
        _ => String::new(),
    }
}

fn oauth_client() -> Result<librespot_oauth::OAuthClient, SessionError> {
    librespot_oauth::OAuthClientBuilder::new(
        &client_id()?,
        "http://127.0.0.1:8898/login",
        OAUTH_SCOPES.to_vec(),
    )
    .open_in_browser()
    .build()
    .map_err(|e| SessionError::OAuth(e.to_string()))
}

fn token_path() -> std::path::PathBuf {
    dirs_cache().join("nocturne").join("oauth.json")
}

/// The persisted OAuth token. `expires_at` is stored as a unix timestamp because librespot's
/// `Instant` is process-relative and meaningless across restarts — the bug that made every launch
/// spend a refresh.
#[derive(serde::Serialize, serde::Deserialize)]
struct StoredToken {
    access_token: String,
    refresh_token: String,
    expires_at_unix: u64,
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// True when a previous consent is on disk, so login will be silent (no browser). The UI uses this
/// to sign in on startup instead of making the user click a button that never asks them anything.
pub fn has_cached_login() -> bool {
    load_token().is_some()
}

fn load_token() -> Option<StoredToken> {
    let bytes = std::fs::read(token_path()).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Refresh tokens are long-lived credentials to the account: 0600, and written atomically so a
/// crash mid-write can't leave a truncated token that forces a re-login.
fn save_token(tok: &librespot_oauth::OAuthToken) {
    let stored = StoredToken {
        access_token: tok.access_token.clone(),
        refresh_token: tok.refresh_token.clone(),
        expires_at_unix: now_unix()
            + tok
                .expires_at
                .saturating_duration_since(std::time::Instant::now())
                .as_secs(),
    };
    let path = token_path();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let Ok(bytes) = serde_json::to_vec(&stored) else {
        return;
    };
    let tmp = path.with_extension("tmp");
    if std::fs::write(&tmp, bytes).is_err() {
        return;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600));
    }
    let _ = std::fs::rename(&tmp, &path);
}

/// Get a usable token with the *least* privilege escalation possible, in order:
///
///   1. the stored access token, if it hasn't expired — costs nothing, and crucially does NOT
///      spend a refresh. Refresh tokens rotate on use, so refreshing on every launch was itself
///      what kept invalidating the stored credential and forcing browser logins.
///   2. a refresh, persisting the newly-rotated refresh token immediately.
///   3. the full browser consent flow, only on a cold start or a genuinely dead credential.
fn obtain_token() -> Result<librespot_oauth::OAuthToken, SessionError> {
    if let Some(stored) = load_token() {
        if stored.expires_at_unix > now_unix() + 60 {
            tracing::info!("using cached access token (no refresh spent)");
            return Ok(librespot_oauth::OAuthToken {
                access_token: stored.access_token,
                refresh_token: stored.refresh_token,
                expires_at: std::time::Instant::now()
                    + std::time::Duration::from_secs(stored.expires_at_unix - now_unix()),
                token_type: "Bearer".into(),
                scopes: OAUTH_SCOPES.iter().map(|s| s.to_string()).collect(),
            });
        }

        let client = oauth_client()?;
        match client.refresh_token(&stored.refresh_token) {
            Ok(tok) => {
                tracing::info!("refreshed access token");
                save_token(&tok);
                return Ok(tok);
            }
            Err(e) => tracing::warn!("stored refresh token rejected ({e}) — re-authorizing"),
        }
    }

    let tok = oauth_client()?
        .get_access_token()
        .map_err(|e| SessionError::OAuth(e.to_string()))?;
    save_token(&tok);
    Ok(tok)
}

fn dirs_cache() -> std::path::PathBuf {
    std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(std::env::var("HOME").expect("HOME unset")).join(".cache")
        })
}

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("oauth flow failed: {0}")]
    OAuth(String),
    #[error("session connect failed: {0}")]
    Connect(String),
}

/// Everything the UI needs: spawn once on the tokio runtime, then send commands.
pub struct NocturneHandle {
    session: Session,
    player: std::sync::Arc<Player>,
    mixer: std::sync::Arc<SoftMixer>,
    /// Our own OAuth token — see [`NocturneHandle::web_token`] for why librespot's isn't used.
    oauth: std::sync::Arc<tokio::sync::Mutex<librespot_oauth::OAuthToken>>,
}

impl NocturneHandle {
    /// Connects the session and builds a Player wired to the given sink.
    ///
    /// Auth is driven by our own PKCE token, whose refresh token is cached under
    /// `~/.cache/nocturne` (Spotify gives it a 180-day life). So the browser consent screen
    /// appears once, and later starts silently refresh. Audio files are deliberately *not*
    /// cached — offline is out of scope.
    pub async fn login(
        make_sink: impl FnMut() -> Box<dyn librespot_playback::audio_backend::Sink> + Send + 'static,
    ) -> Result<Self, SessionError> {
        let cache_dir = dirs_cache().join("nocturne");
        let cache = Cache::new(Some(&cache_dir), Some(&cache_dir), None, None)
            .map_err(|e| SessionError::Connect(e.to_string()))?;

        // librespot-oauth is blocking and spins up its own runtime internally; calling it straight
        // from async panics on drop ("cannot drop a runtime in a context where blocking is not
        // allowed"). It has to run on a blocking thread.
        let token = tokio::task::spawn_blocking(obtain_token)
            .await
            .map_err(|e| SessionError::OAuth(e.to_string()))??;
        let credentials = Credentials::with_access_token(token.access_token.clone());

        let session = Session::new(SessionConfig::default(), Some(cache));
        session
            .connect(credentials, true)
            .await
            .map_err(|e| SessionError::Connect(e.to_string()))?;

        let mixer = std::sync::Arc::new(
            SoftMixer::open(MixerConfig::default())
                .map_err(|e| SessionError::Connect(e.to_string()))?,
        );
        let player = Player::new(
            PlayerConfig::default(),
            session.clone(),
            mixer.get_soft_volume(),
            make_sink,
        );

        Ok(Self {
            session,
            player,
            mixer,
            oauth: std::sync::Arc::new(tokio::sync::Mutex::new(token)),
        })
    }

    pub fn play(&self, track: SpotifyId) {
        self.play_uri(SpotifyUri::Track { id: track });
    }

    pub fn play_uri(&self, uri: SpotifyUri) {
        self.player.load(uri, true, 0);
    }

    /// Track/position/state changes, for the UI's now-playing bar.
    pub fn player_events(&self) -> librespot_playback::player::PlayerEventChannel {
        self.player.get_player_event_channel()
    }

    /// Playlist contents, over Spotify's **internal** protocol rather than the Web API.
    ///
    /// This is not a stylistic choice. Spotify 403s `/v1/playlists/{id}/tracks` AND
    /// `/v1/tracks?ids=` for apps registered after their 2024 lockdown — even for playlists you
    /// own, with every scope granted. `/v1/playlists/{id}` still returns 200 but with the track
    /// list stripped out. librespot's metadata layer speaks the protocol the real client uses,
    /// which has no such restriction, so playlists come from here and search stays on the Web API.
    pub async fn playlist_tracks(&self, playlist_id: &str) -> Result<Vec<nocturne_api::Track>, SessionError> {
        use librespot_metadata::{Metadata, Playlist};

        let id = SpotifyId::from_base62(playlist_id)
            .map_err(|e| SessionError::Connect(format!("bad playlist id: {e}")))?;
        let list = Playlist::get(&self.session, &SpotifyUri::Playlist { id, user: None })
            .await
            .map_err(|e| SessionError::Connect(format!("playlist: {e}")))?;

        // Fetch track metadata in BATCHES, not one request per track. librespot's `Track::get`
        // issues a request per id; on a 300-track playlist that stampedes Spotify's extended-
        // metadata service, which rate-limits the whole session ("resource has been exhausted")
        // and then poisons every playlist opened afterwards. The batched entity endpoint is what
        // the real client uses, and it takes the whole chunk in one round trip.
        // Batches run concurrently — a 350-track playlist is 4 requests, and issuing them serially
        // costs 4 round trips for no reason. Bounded so we don't recreate the stampede that got
        // the session rate-limited in the first place.
        use futures_util::StreamExt;
        let uris: Vec<SpotifyUri> = list.tracks().cloned().collect();
        let chunks: Vec<Vec<SpotifyUri>> = uris.chunks(BATCH).map(|c| c.to_vec()).collect();
        let batches: Vec<Result<Vec<nocturne_api::Track>, SessionError>> =
            futures_util::stream::iter(chunks)
                .map(|chunk| async move { self.tracks_batch(&chunk).await })
                .buffered(4)
                .collect()
                .await;

        let mut out = Vec::with_capacity(uris.len());
        for b in batches {
            out.extend(b?);
        }
        Ok(out)
    }

    /// One extended-metadata round trip for up to [`BATCH`] tracks.
    async fn tracks_batch(&self, uris: &[SpotifyUri]) -> Result<Vec<nocturne_api::Track>, SessionError> {
        use librespot_metadata::Metadata;
        use librespot_protocol::extended_metadata::{
            BatchedEntityRequest, EntityRequest, ExtensionQuery,
        };
        use librespot_protocol::extension_kind::ExtensionKind;
        use protobuf::{EnumOrUnknown, Message};

        let req = BatchedEntityRequest {
            entity_request: uris
                .iter()
                .map(|uri| EntityRequest {
                    entity_uri: uri.to_uri(),
                    query: vec![ExtensionQuery {
                        extension_kind: EnumOrUnknown::new(ExtensionKind::TRACK_V4),
                        ..Default::default()
                    }],
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        };

        let res = self
            .session
            .spclient()
            .get_extended_metadata(req)
            .await
            .map_err(|e| SessionError::Connect(format!("metadata: {e}")))?;

        let mut out = Vec::new();
        for array in res.extended_metadata {
            for entry in array.extension_data {
                let Some(any) = entry.extension_data.as_ref() else {
                    continue;
                };
                let Ok(msg) = librespot_protocol::metadata::Track::parse_from_bytes(&any.value)
                else {
                    continue;
                };
                let Ok(uri) = SpotifyUri::from_uri(&entry.entity_uri) else {
                    continue;
                };
                if let Ok(track) = librespot_metadata::Track::parse(&msg, &uri) {
                    out.push(to_api_track(track));
                }
            }
        }
        Ok(out)
    }

    /// Radio: what Spotify would play next once a queue runs dry.
    ///
    /// Uses the internal radio-apollo station service, not the Web API's `/recommendations` —
    /// which is one of the endpoints Spotify 403s for post-2024 apps, same as playlist tracks.
    /// `previous` is fed back so the station doesn't re-suggest what just played.
    pub async fn radio_from(
        &self,
        seed_uri: &SpotifyUri,
        previous: &[SpotifyUri],
        count: usize,
    ) -> Result<Vec<nocturne_api::Track>, SessionError> {
        let prev_ids: Vec<SpotifyId> = previous
            .iter()
            .filter_map(|u| match u {
                SpotifyUri::Track { id } => Some(*id),
                _ => None,
            })
            .collect();

        let bytes = self
            .session
            .spclient()
            .get_apollo_station("tracks", &seed_uri.to_uri(), Some(count), prev_ids, true)
            .await
            .map_err(|e| SessionError::Connect(format!("radio: {e}")))?;

        let json: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| SessionError::Connect(format!("radio json: {e}")))?;

        // The station returns bare track uris; hydrate them through the same batched metadata path
        // the playlists use.
        let uris: Vec<SpotifyUri> = json["tracks"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|t| t["uri"].as_str())
                    .filter_map(|u| SpotifyUri::from_uri(u).ok())
                    .collect()
            })
            .unwrap_or_default();

        if uris.is_empty() {
            return Ok(Vec::new());
        }
        self.tracks_batch(&uris[..uris.len().min(BATCH)]).await
    }

    pub fn seek(&self, position_ms: u32) {
        self.player.seek(position_ms);
    }

    /// 0.0..=1.0. SoftMixer takes a u16 across its own range.
    pub fn set_volume(&self, v: f32) {
        self.mixer.set_volume((v.clamp(0.0, 1.0) * u16::MAX as f32) as u16);
    }

    pub fn stop(&self) {
        self.player.stop();
    }

    /// A bearer token for the Spotify Web API.
    ///
    /// NOT `session.token_provider()` — that mints tokens against *librespot's* built-in client id,
    /// which Spotify 403s ("Invalid request") for Web API scopes. We already hold a PKCE token
    /// issued to *our* app with the scopes we asked for, so reuse it and refresh when it ages out.
    pub async fn web_token(&self) -> Result<String, SessionError> {
        let mut tok = self.oauth.lock().await;
        // Refresh a minute early rather than racing the expiry mid-request.
        if tok.expires_at <= std::time::Instant::now() + std::time::Duration::from_secs(60) {
            tracing::info!("web token expired — refreshing");
            let rt = tok.refresh_token.clone();
            // Blocking, same as the initial flow — keep it off the async worker.
            let fresh = tokio::task::spawn_blocking(move || {
                oauth_client()?
                    .refresh_token(&rt)
                    .map_err(|e| SessionError::OAuth(e.to_string()))
            })
            .await
            .map_err(|e| SessionError::OAuth(e.to_string()))??;
            save_token(&fresh);
            *tok = fresh;
        }
        Ok(tok.access_token.clone())
    }

    pub fn pause(&self) {
        self.player.pause();
    }

    pub fn resume(&self) {
        self.player.play();
    }

    pub fn session(&self) -> &Session {
        &self.session
    }
}
