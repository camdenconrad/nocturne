//! Session layer: owns the librespot session, player, and Connect device.
//! The UI talks to this through `NocturneHandle` — it never touches librespot types.

use librespot_core::authentication::Credentials;
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
    /// Runs the OAuth flow (opens browser via portal), connects the session,
    /// and builds a Player wired to the given sink.
    pub async fn login(
        make_sink: impl FnMut() -> Box<dyn librespot_playback::audio_backend::Sink> + Send + 'static,
    ) -> Result<Self, SessionError> {
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
        let credentials = Credentials::with_access_token(token.access_token);

        let session = Session::new(SessionConfig::default(), None);
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
        self.player
            .load(SpotifyUri::Track { id: track }, true, 0);
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
