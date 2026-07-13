//! The audio/network half of Nocturne, on its own tokio runtime.
//!
//! egui repaints on the UI thread and must never block, so everything that can wait — login, Web
//! API calls, art fetches — happens here and lands in a shared [`State`] the UI just reads. The UI
//! sends [`Cmd`]s; it never awaits anything.

use nocturne_api::{Client, Playlist, Track};
use nocturne_session::NocturneHandle;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::mpsc;

pub enum Cmd {
    Login,
    Search(String),
    LoadSaved,
    LoadPlaylists,
    OpenPlaylist(String),
    Play(String),
    PlayPause,
    Seek(u32),
}

#[derive(Default, Clone)]
pub struct NowPlaying {
    pub name: String,
    pub artists: String,
    pub duration_ms: u32,
    pub art_url: Option<String>,
    pub paused: bool,
    /// Position at the moment of the last player event…
    pub position_ms: u32,
    /// …and when that was, so the UI can interpolate a smooth progress bar between events.
    pub since: Option<Instant>,
}

impl NowPlaying {
    /// Interpolated playhead — librespot only emits events on state *changes*, so a bar driven
    /// straight off `position_ms` would freeze between them.
    pub fn elapsed_ms(&self) -> u32 {
        let base = self.position_ms;
        match (self.paused, self.since) {
            (false, Some(t)) => (base + t.elapsed().as_millis() as u32).min(self.duration_ms),
            _ => base,
        }
    }
}

#[derive(Default)]
pub struct State {
    pub status: String,
    pub logged_in: bool,
    pub busy: bool,
    pub tracks: Vec<Track>,
    pub playlists: Vec<Playlist>,
    pub now: Option<NowPlaying>,
    /// Decoded art bytes keyed by URL; the UI turns these into textures once.
    pub art: std::collections::HashMap<String, Vec<u8>>,
}

pub type Shared = Arc<Mutex<State>>;

/// Spawn the backend thread. Returns the shared state and the command sender.
pub fn spawn(repaint: impl Fn() + Send + Clone + 'static) -> (Shared, mpsc::UnboundedSender<Cmd>) {
    let state: Shared = Arc::new(Mutex::new(State {
        status: "not signed in".into(),
        ..Default::default()
    }));
    let (tx, rx) = mpsc::unbounded_channel();

    let st = state.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        rt.block_on(run(st, rx, repaint));
    });

    (state, tx)
}

async fn run(
    state: Shared,
    mut rx: mpsc::UnboundedReceiver<Cmd>,
    repaint: impl Fn() + Send + Clone + 'static,
) {
    let mut handle: Option<Arc<NocturneHandle>> = None;
    let mut paused = false;

    while let Some(cmd) = rx.recv().await {
        match cmd {
            Cmd::Login => {
                set(&state, &repaint, |s| {
                    s.busy = true;
                    s.status = "signing in — approve in your browser…".into();
                });
                let sink = match nocturne_sink::make_sink() {
                    Ok(s) => s,
                    Err(e) => {
                        fail(&state, &repaint, format!("audio: {e}"));
                        continue;
                    }
                };
                match NocturneHandle::login(sink).await {
                    Ok(h) => {
                        let h = Arc::new(h);
                        spawn_event_pump(h.clone(), state.clone(), erase(&repaint));
                        handle = Some(h);
                        set(&state, &repaint, |s| {
                            s.busy = false;
                            s.logged_in = true;
                            s.status = "signed in".into();
                        });
                    }
                    Err(e) => fail(&state, &repaint, format!("login failed: {e}")),
                }
            }

            Cmd::Search(q) if !q.trim().is_empty() => {
                if let Some(api) = api(&handle, &state, &repaint).await {
                    busy(&state, &repaint, format!("searching “{q}”…"));
                    match api.search_tracks(&q, 50).await {
                        Ok(t) => finish_tracks(&state, &repaint, t, &api).await,
                        Err(e) => fail(&state, &repaint, format!("search: {e}")),
                    }
                }
            }
            Cmd::Search(_) => {}

            Cmd::LoadSaved => {
                if let Some(api) = api(&handle, &state, &repaint).await {
                    busy(&state, &repaint, "loading liked songs…".into());
                    match api.saved_tracks(2000).await {
                        Ok(t) => finish_tracks(&state, &repaint, t, &api).await,
                        Err(e) => fail(&state, &repaint, format!("liked songs: {e}")),
                    }
                }
            }

            Cmd::LoadPlaylists => {
                if let Some(api) = api(&handle, &state, &repaint).await {
                    match api.playlists(500).await {
                        Ok(p) => set(&state, &repaint, |s| s.playlists = p),
                        Err(e) => fail(&state, &repaint, format!("playlists: {e}")),
                    }
                }
            }

            Cmd::OpenPlaylist(id) => {
                if let Some(api) = api(&handle, &state, &repaint).await {
                    busy(&state, &repaint, "loading playlist…".into());
                    match api.playlist_tracks(&id, 2000).await {
                        Ok(t) => finish_tracks(&state, &repaint, t, &api).await,
                        Err(e) => fail(&state, &repaint, format!("playlist: {e}")),
                    }
                }
            }

            Cmd::Play(uri) => {
                if let Some(h) = &handle {
                    let track = state
                        .lock()
                        .unwrap()
                        .tracks
                        .iter()
                        .find(|t| t.uri == uri)
                        .cloned();
                    if let Some(t) = track {
                        set(&state, &repaint, |s| {
                            s.now = Some(NowPlaying {
                                name: t.name.clone(),
                                artists: t.artists.clone(),
                                duration_ms: t.duration_ms,
                                art_url: t.art_url.clone(),
                                paused: false,
                                position_ms: 0,
                                since: Some(Instant::now()),
                            });
                        });
                    }
                    match librespot_uri(&uri) {
                        Ok(u) => {
                            paused = false;
                            h.play_uri(u);
                        }
                        Err(e) => fail(&state, &repaint, e),
                    }
                }
            }

            Cmd::PlayPause => {
                if let Some(h) = &handle {
                    paused = !paused;
                    if paused {
                        h.pause()
                    } else {
                        h.resume()
                    }
                }
            }

            Cmd::Seek(ms) => {
                if let Some(h) = &handle {
                    h.seek(ms);
                    set(&state, &repaint, |s| {
                        if let Some(n) = &mut s.now {
                            n.position_ms = ms;
                            n.since = Some(Instant::now());
                        }
                    });
                }
            }
        }
    }
}

/// Player events → now-playing state. Runs for the life of the session.
fn spawn_event_pump(h: Arc<NocturneHandle>, state: Shared, repaint: Box<dyn Fn() + Send>) {
    use librespot_playback::player::PlayerEvent;
    tokio::spawn(async move {
        let mut ev = h.player_events();
        while let Some(e) = ev.recv().await {
            let mut s = state.lock().unwrap();
            if let Some(n) = &mut s.now {
                match e {
                    PlayerEvent::Playing { position_ms, .. } => {
                        n.paused = false;
                        n.position_ms = position_ms;
                        n.since = Some(Instant::now());
                    }
                    PlayerEvent::Paused { position_ms, .. } => {
                        n.paused = true;
                        n.position_ms = position_ms;
                        n.since = None;
                    }
                    PlayerEvent::EndOfTrack { .. } => {
                        n.paused = true;
                        n.position_ms = n.duration_ms;
                        n.since = None;
                    }
                    _ => {}
                }
            }
            drop(s);
            repaint();
        }
    });
}

/// Mint a fresh Web API client. Tokens are short-lived, so this is per-request rather than cached.
async fn api(
    handle: &Option<Arc<NocturneHandle>>,
    state: &Shared,
    repaint: &(impl Fn() + Send),
) -> Option<Client> {
    let h = handle.as_ref()?;
    match h.web_token().await {
        Ok(t) => Some(Client::new(t)),
        Err(e) => {
            fail(state, repaint, format!("token: {e}"));
            None
        }
    }
}

/// Store the results, then pull their cover art in the background so rows can show thumbnails.
async fn finish_tracks(state: &Shared, repaint: &(impl Fn() + Send), tracks: Vec<Track>, api: &Client) {
    let urls: Vec<String> = tracks.iter().filter_map(|t| t.art_url.clone()).collect();
    set(state, repaint, |s| {
        s.busy = false;
        s.status = format!("{} tracks", tracks.len());
        s.tracks = tracks;
    });
    for url in urls {
        if state.lock().unwrap().art.contains_key(&url) {
            continue;
        }
        if let Ok(bytes) = api.fetch_art(&url).await {
            state.lock().unwrap().art.insert(url, bytes);
            repaint();
        }
    }
}

fn librespot_uri(uri: &str) -> Result<librespot_core::SpotifyUri, String> {
    librespot_core::SpotifyUri::from_uri(uri).map_err(|e| format!("bad uri {uri}: {e}"))
}

fn set(state: &Shared, repaint: &(impl Fn() + Send + ?Sized), f: impl FnOnce(&mut State)) {
    f(&mut state.lock().unwrap());
    repaint();
}

fn busy(state: &Shared, repaint: &(impl Fn() + Send), msg: String) {
    set(state, repaint, |s| {
        s.busy = true;
        s.status = msg;
    });
}

fn fail(state: &Shared, repaint: &(impl Fn() + Send + ?Sized), msg: String) {
    tracing::warn!("{msg}");
    set(state, repaint, |s| {
        s.busy = false;
        s.status = msg;
    });
}

/// egui's repaint handle is `Clone + Send`, but the generic closure isn't object-safe where the
/// event pump needs it — box it once.
fn erase(f: &(impl Fn() + Send + Clone + 'static)) -> Box<dyn Fn() + Send> {
    Box::new(f.clone())
}
