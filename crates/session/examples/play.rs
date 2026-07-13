//! End-to-end proof: OAuth → session → Player → runic. Plays a track for 20s.
//!
//!     cargo run -p nocturne-session --example play                  # a default track
//!     cargo run -p nocturne-session --example play <spotify-track-url-or-uri>
//!
//! First run opens a browser for consent; after that the cached credential is reused.

use librespot_core::SpotifyUri;
use librespot_playback::player::PlayerEvent;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info,librespot=info".into()),
        )
        .init();

    // "Never Gonna Give You Up" — a deliberately unmistakable default.
    let arg = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "spotify:track:4PTG3Z6ehGkBFwjybzWkR8".to_string());
    let uri: SpotifyUri = parse_track(&arg)?;

    let sink = nocturne_sink::make_sink()?;
    println!("logging in (browser opens on first run)…");
    let handle = nocturne_session::NocturneHandle::login(sink).await?;
    println!("logged in. loading {arg}");

    let mut events = handle.player_events();
    handle.play_uri(uri);

    let deadline = tokio::time::sleep(std::time::Duration::from_secs(20));
    tokio::pin!(deadline);
    loop {
        tokio::select! {
            _ = &mut deadline => break,
            Some(ev) = events.recv() => match ev {
                PlayerEvent::Playing { position_ms, .. } => println!("▶ playing @ {position_ms}ms"),
                PlayerEvent::Paused { .. } => println!("⏸ paused"),
                PlayerEvent::EndOfTrack { .. } => { println!("⏹ end of track"); break }
                PlayerEvent::Unavailable { .. } => { println!("track unavailable"); break }
                _ => {}
            },
        }
    }
    println!("done — if you heard music, Nocturne plays Spotify through runic.");
    Ok(())
}

/// Accepts `spotify:track:ID`, an open.spotify.com URL, or a bare ID.
fn parse_track(s: &str) -> Result<SpotifyUri, Box<dyn std::error::Error>> {
    let id = if let Some(rest) = s.strip_prefix("spotify:track:") {
        rest.to_string()
    } else if let Some(rest) = s.split("/track/").nth(1) {
        rest.split(['?', '/']).next().unwrap_or_default().to_string()
    } else {
        s.to_string()
    };
    Ok(SpotifyUri::from_uri(&format!("spotify:track:{id}"))?)
}
