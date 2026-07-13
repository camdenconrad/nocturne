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

fn refresh_token_path() -> std::path::PathBuf {
    dirs_cache().join("nocturne").join("oauth-refresh")
}

/// True when a previous consent is on disk, so login will be silent (no browser). The UI uses this
/// to sign in on startup instead of making the user click a button that never asks them anything.
pub fn has_cached_login() -> bool {
    std::fs::read_to_string(refresh_token_path())
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false)
}

/// The refresh token is a long-lived credential to the account — treat it like a secret.
fn save_refresh_token(rt: &str) {
    let path = refresh_token_path();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if std::fs::write(&path, rt).is_ok() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
        }
    }
}

/// Silent refresh if we've been here before; browser consent only on a cold start (or if the
/// stored refresh token has been revoked/expired, in which case fall back to the full flow).
fn obtain_token() -> Result<librespot_oauth::OAuthToken, SessionError> {
    let client = oauth_client()?;
    if let Ok(rt) = std::fs::read_to_string(refresh_token_path()) {
        let rt = rt.trim();
        if !rt.is_empty() {
            match client.refresh_token(rt) {
                Ok(tok) => {
                    save_refresh_token(&tok.refresh_token);
                    return Ok(tok);
                }
                Err(e) => tracing::warn!("stored refresh token rejected ({e}) — re-authorizing"),
            }
        }
    }
    let tok = client
        .get_access_token()
        .map_err(|e| SessionError::OAuth(e.to_string()))?;
    save_refresh_token(&tok.refresh_token);
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

        let mixer = SoftMixer::open(MixerConfig::default())
            .map_err(|e| SessionError::Connect(e.to_string()))?;
        let player = Player::new(
            PlayerConfig::default(),
            session.clone(),
            mixer.get_soft_volume(),
            make_sink,
        );

        Ok(Self {
            session,
            player,
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

    pub fn seek(&self, position_ms: u32) {
        self.player.seek(position_ms);
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
            save_refresh_token(&fresh.refresh_token);
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
