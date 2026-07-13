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

const OAUTH_SCOPES: &[&str] = &["streaming"];

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
}

impl NocturneHandle {
    /// Connects the session and builds a Player wired to the given sink.
    ///
    /// Credentials are cached under `~/.cache/nocturne`, so the browser consent screen appears
    /// once and every later start reuses the stored credential — `connect(.., true)` is what
    /// writes it back. Audio files are deliberately *not* cached (offline is out of scope).
    pub async fn login(
        make_sink: impl FnMut() -> Box<dyn librespot_playback::audio_backend::Sink> + Send + 'static,
    ) -> Result<Self, SessionError> {
        let cache_dir = dirs_cache().join("nocturne");
        let cache = Cache::new(Some(&cache_dir), Some(&cache_dir), None, None)
            .map_err(|e| SessionError::Connect(e.to_string()))?;

        let credentials = match cache.credentials() {
            Some(creds) => creds,
            None => {
                let client_id = client_id()?;
                let token = librespot_oauth::OAuthClientBuilder::new(
                    &client_id,
                    "http://127.0.0.1:8898/login",
                    OAUTH_SCOPES.to_vec(),
                )
                .open_in_browser()
                .build()
                .map_err(|e| SessionError::OAuth(e.to_string()))?
                .get_access_token()
                .map_err(|e| SessionError::OAuth(e.to_string()))?;
                Credentials::with_access_token(token.access_token)
            }
        };

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

        Ok(Self { session, player })
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
