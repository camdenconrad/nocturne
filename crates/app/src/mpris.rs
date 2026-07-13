//! MPRIS — so Nocturne is a first-class citizen of the Rune shell.
//!
//! `rtray` scans D-Bus for `org.mpris.MediaPlayer2.*`, reads `PlaybackStatus` and `Metadata`
//! (`xesam:title` / `xesam:artist`), and drives `PlayPause` / `Next` / `Previous`. Own that name
//! and Nocturne becomes the tray's active player, with working media keys, and the track shows in
//! rnotify — no changes needed anywhere in the shell.
//!
//! rtray prefers a player that is actually *Playing* and isn't a browser, so an idle Nocturne won't
//! steal the tray from something that's really making noise.

use crate::backend::{Cmd, Shared};
use std::collections::HashMap;
use tokio::sync::mpsc::UnboundedSender;
use zbus::zvariant::{ObjectPath, Value};
use zbus::{connection, interface};

const BUS_NAME: &str = "org.mpris.MediaPlayer2.nocturne";
const OBJ_PATH: &str = "/org/mpris/MediaPlayer2";

struct Root;

#[interface(name = "org.mpris.MediaPlayer2")]
impl Root {
    #[zbus(property)]
    fn identity(&self) -> String {
        "Nocturne".into()
    }

    #[zbus(property)]
    fn desktop_entry(&self) -> String {
        "nocturne".into()
    }

    #[zbus(property)]
    fn can_quit(&self) -> bool {
        false
    }

    #[zbus(property)]
    fn can_raise(&self) -> bool {
        false
    }

    #[zbus(property)]
    fn has_track_list(&self) -> bool {
        false
    }

    #[zbus(property)]
    fn supported_uri_schemes(&self) -> Vec<String> {
        vec!["spotify".into()]
    }

    #[zbus(property)]
    fn supported_mime_types(&self) -> Vec<String> {
        Vec::new()
    }
}

struct Player {
    state: Shared,
    tx: UnboundedSender<Cmd>,
}

#[interface(name = "org.mpris.MediaPlayer2.Player")]
impl Player {
    fn play_pause(&self) {
        let _ = self.tx.send(Cmd::PlayPause);
    }

    fn play(&self) {
        let s = self.state.lock().unwrap();
        if s.now.as_ref().is_some_and(|n| n.paused) {
            drop(s);
            let _ = self.tx.send(Cmd::PlayPause);
        }
    }

    fn pause(&self) {
        let s = self.state.lock().unwrap();
        if s.now.as_ref().is_some_and(|n| !n.paused) {
            drop(s);
            let _ = self.tx.send(Cmd::PlayPause);
        }
    }

    fn stop(&self) {
        let _ = self.tx.send(Cmd::PlayPause);
    }

    fn next(&self) {
        let _ = self.tx.send(Cmd::Next);
    }

    fn previous(&self) {
        let _ = self.tx.send(Cmd::Prev);
    }

    #[zbus(property)]
    fn playback_status(&self) -> String {
        let s = self.state.lock().unwrap();
        match &s.now {
            Some(n) if !n.paused => "Playing".into(),
            Some(_) => "Paused".into(),
            None => "Stopped".into(),
        }
    }

    /// What the tray actually renders. `xesam:artist` must be an ARRAY of strings — rtray does
    /// `Vec::<String>::try_from(...)` on it, and a bare string silently yields no artist.
    #[zbus(property)]
    fn metadata(&self) -> HashMap<String, Value<'static>> {
        let s = self.state.lock().unwrap();
        let mut md: HashMap<String, Value> = HashMap::new();
        let Some(n) = &s.now else {
            return md;
        };

        // A track id is required by the spec; some consumers drop metadata entirely without it.
        let id = s
            .current_uri
            .as_deref()
            .map(|u| u.replace(':', "/"))
            .unwrap_or_else(|| "/org/mpris/MediaPlayer2/TrackList/NoTrack".into());
        let path = ObjectPath::try_from(format!("/com/coffee/nocturne{id}"))
            .unwrap_or_else(|_| ObjectPath::try_from("/com/coffee/nocturne/track").unwrap());

        md.insert("mpris:trackid".into(), Value::from(path).try_to_owned().unwrap().into());
        md.insert(
            "mpris:length".into(),
            Value::from(n.duration_ms as i64 * 1000),
        );
        md.insert("xesam:title".into(), Value::from(n.name.clone()));
        md.insert(
            "xesam:artist".into(),
            Value::from(
                n.artists
                    .split(", ")
                    .map(|a| a.to_string())
                    .collect::<Vec<_>>(),
            ),
        );
        if let Some(art) = &n.art_url {
            md.insert("mpris:artUrl".into(), Value::from(art.clone()));
        }
        md
    }

    #[zbus(property)]
    fn can_play(&self) -> bool {
        true
    }
    #[zbus(property)]
    fn can_pause(&self) -> bool {
        true
    }
    #[zbus(property)]
    fn can_go_next(&self) -> bool {
        true
    }
    #[zbus(property)]
    fn can_go_previous(&self) -> bool {
        true
    }
    #[zbus(property)]
    fn can_seek(&self) -> bool {
        false
    }
    #[zbus(property)]
    fn can_control(&self) -> bool {
        true
    }

    #[zbus(property)]
    fn position(&self) -> i64 {
        let s = self.state.lock().unwrap();
        s.now.as_ref().map(|n| n.elapsed_ms() as i64 * 1000).unwrap_or(0)
    }

    #[zbus(property)]
    fn volume(&self) -> f64 {
        self.state.lock().unwrap().volume as f64
    }

    #[zbus(property)]
    fn rate(&self) -> f64 {
        1.0
    }
}

/// Publish the MPRIS interfaces and keep the tray's view of the track fresh.
pub fn spawn(state: Shared, tx: UnboundedSender<Cmd>) {
    std::thread::spawn(move || {
        let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
            Ok(rt) => rt,
            Err(e) => {
                tracing::warn!("mpris: no runtime: {e}");
                return;
            }
        };
        rt.block_on(async move {
            let player = Player {
                state: state.clone(),
                tx,
            };
            let conn = match connection::Builder::session()
                .and_then(|b| b.name(BUS_NAME))
                .and_then(|b| b.serve_at(OBJ_PATH, Root))
                .and_then(|b| b.serve_at(OBJ_PATH, player))
                .map(|b| b.build())
            {
                Ok(fut) => match fut.await {
                    Ok(c) => c,
                    Err(e) => {
                        // Not fatal: no D-Bus just means no tray integration.
                        tracing::warn!("mpris: could not take {BUS_NAME}: {e}");
                        return;
                    }
                },
                Err(e) => {
                    tracing::warn!("mpris: setup failed: {e}");
                    return;
                }
            };
            tracing::info!("mpris: serving {BUS_NAME} (rtray will pick this up)");

            // The tray only refreshes when properties change, so tell it when they do. Poll our own
            // state rather than plumbing a signal through the backend — it's one comparison a
            // second and it keeps the player free of D-Bus concerns.
            let iface = match conn
                .object_server()
                .interface::<_, Player>(OBJ_PATH)
                .await
            {
                Ok(i) => i,
                Err(e) => {
                    tracing::warn!("mpris: no interface handle: {e}");
                    return;
                }
            };

            let mut last: Option<(String, bool)> = None;
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(700)).await;
                let now = {
                    let s = state.lock().unwrap();
                    s.now.as_ref().map(|n| (n.name.clone(), n.paused))
                };
                if now != last {
                    last = now;
                    let ctx = iface.signal_emitter();
                    let p = iface.get().await;
                    let _ = p.playback_status_changed(ctx).await;
                    let _ = p.metadata_changed(ctx).await;
                }
            }
        });
    });
}
