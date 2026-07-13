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

fn client_id() -> Result<String, SessionError> {
    std::env::var("NOCTURNE_CLIENT_ID")
        .map_err(|_| SessionError::OAuth("set NOCTURNE_CLIENT_ID to your Spotify app client id".into()))
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
