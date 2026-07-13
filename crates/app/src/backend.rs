//! The audio/network half of Nocturne, on its own tokio runtime.
//!
//! egui repaints on the UI thread and must never block, so everything that can wait — login, Web
//! API calls, art fetches — happens here and lands in a shared [`State`] the UI just reads. The UI
//! sends [`Cmd`]s; it never awaits anything.

use crate::cache;
use nocturne_taste::Taste;
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
    /// Play this track and queue the rest of the visible list behind it.
    Play(String),
    PlayPause,
    Next,
    Prev,
    Seek(u32),
    Volume(f32),
    /// Turn radio (autoplay past the end of the queue) on or off.
    SetAutoplay(bool),
    /// Rebuild the taste model from cached playlists (fired once after they load).
    TrainTaste,
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

pub struct State {
    pub status: String,
    pub logged_in: bool,
    pub busy: bool,
    pub tracks: Vec<Track>,
    pub playlists: Vec<Playlist>,
    pub now: Option<NowPlaying>,
    /// URI of the playing track, so the list can highlight the current row.
    pub current_uri: Option<String>,
    /// Title of whatever is on screen ("Liked Songs", a playlist name, a search).
    pub view: String,
    pub volume: f32,
    /// When the queue runs dry, keep playing with Spotify's radio for the last track.
    pub autoplay: bool,
    /// Set while the radio is being fetched, so the UI can say so.
    pub radio_loading: bool,
    /// How many sequences the taste model has learned — 0 means it's cold and radio falls back to
    /// Spotify's own ordering.
    pub taste_trained: usize,
    /// Tracks with Spotify's real analysis attached.
    pub taste_features: usize,
    /// Decoded art bytes keyed by URL; the UI turns these into textures once.
    pub art: std::collections::HashMap<String, Vec<u8>>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            status: String::new(),
            logged_in: false,
            busy: false,
            tracks: Vec::new(),
            playlists: Vec::new(),
            now: None,
            current_uri: None,
            view: String::new(),
            volume: 1.0,
            autoplay: true,
            radio_loading: false,
            taste_trained: 0,
            taste_features: 0,
            art: Default::default(),
        }
    }
}

pub type Shared = Arc<Mutex<State>>;

/// Spawn the backend thread. Returns the shared state and the command sender.
pub fn spawn(repaint: impl Fn() + Send + Sync + Clone + 'static) -> (Shared, mpsc::UnboundedSender<Cmd>) {
    let state: Shared = Arc::new(Mutex::new(State {
        status: "not signed in".into(),
        ..Default::default()
    }));
    let (tx, rx) = mpsc::unbounded_channel();

    let st = state.clone();
    let self_tx = tx.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        rt.block_on(run(st, rx, self_tx, repaint));
    });

    (state, tx)
}

async fn run(
    state: Shared,
    mut rx: mpsc::UnboundedReceiver<Cmd>,
    self_tx: mpsc::UnboundedSender<Cmd>,
    repaint: impl Fn() + Send + Sync + Clone + 'static,
) {
    let mut handle: Option<Arc<NocturneHandle>> = None;
    let mut paused = false;
    // The queue is whatever list was on screen when the user hit play, so a track ending
    // advances through the playlist instead of falling silent.
    let mut queue: Vec<Track> = Vec::new();
    let mut qpos: usize = 0;
    // Learned autoplay. Rebuilt from the on-disk playlist cache at startup rather than persisted:
    // the tree has no serialization, and it trains fast enough that the cache IS the model.
    // A saved model skips retraining entirely; a missing or stale one (different embedding layout)
    // is rebuilt from the cache.
    let mut taste = Taste::load(&cache::model_path()).unwrap_or_default();
    // The current listening run, with how much of each track was actually heard. A skip is the
    // only negative signal we get without asking the user anything.
    let mut run: Vec<(Track, f32)> = Vec::new();

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
                        spawn_event_pump(h.clone(), state.clone(), self_tx.clone(), erase(&repaint));
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
                // Every fetch runs as its own task. On the command loop, one slow load (the
                // 2000-track Liked Songs sweep at startup) blocks every later click behind it —
                // which is exactly why opening a playlist "took forever".
                let (h, st, rp) = (handle.clone(), state.clone(), repaint.clone());
                tokio::spawn(async move {
                    if let Some(api) = api(&h, &st, &rp).await {
                        busy(&st, &rp, format!("searching “{q}”…"));
                        set(&st, &rp, |s| s.view = format!("Search: {q}"));
                        match api.search_tracks(&q, 50).await {
                            Ok(t) => finish_tracks(&st, &rp, t, &api).await,
                            Err(e) => fail(&st, &rp, format!("search: {e}")),
                        }
                    }
                });
            }
            Cmd::Search(_) => {}

            Cmd::LoadSaved => {
                let (h, st, rp) = (handle.clone(), state.clone(), repaint.clone());
                tokio::spawn(async move {
                    set(&st, &rp, |s| s.view = "Liked Songs".into());
                    // Paint from disk first; the network refresh lands behind the visible list.
                    let cached: Option<Vec<Track>> = cache::list_get("liked");
                    if let Some(t) = cached {
                        show(&st, &rp, t);
                    } else {
                        busy(&st, &rp, "loading liked songs…".into());
                    }
                    if let Some(api) = api(&h, &st, &rp).await {
                        match api.saved_tracks(5000).await {
                            Ok(t) => {
                                cache::list_put("liked", &t);
                                finish_tracks(&st, &rp, t, &api).await
                            }
                            Err(e) => fail(&st, &rp, format!("liked songs: {e}")),
                        }
                    }
                });
            }

            Cmd::LoadPlaylists => {
                if let Some(p) = cache::list_get::<Vec<Playlist>>("playlists") {
                    set(&state, &repaint, |s| s.playlists = p);
                }
                if let Some(api) = api(&handle, &state, &repaint).await {
                    match api.playlists(500).await {
                        Ok(p) => {
                            tracing::info!("loaded {} playlists", p.len());
                            cache::list_put("playlists", &p);
                            // Train only once the list exists — firing this from startup raced the
                            // load and trained on an empty playlist set.
                            let _ = self_tx.send(Cmd::TrainTaste);
                            // Debug hook: NOCTURNE_OPEN=<name substring> opens one on startup, so
                            // the real in-app path can be exercised without a human clicking.
                            if let Ok(want) = std::env::var("NOCTURNE_OPEN") {
                                if let Some(hit) = p
                                    .iter()
                                    .find(|x| x.name.to_lowercase().contains(&want.to_lowercase()))
                                {
                                    let _ = self_tx.send(Cmd::OpenPlaylist(hit.id.clone()));
                                }
                            }
                            set(&state, &repaint, |s| s.playlists = p);
                        }
                        Err(e) => fail(&state, &repaint, format!("playlists: {e}")),
                    }
                }
            }

            Cmd::OpenPlaylist(id) => {
                let (h, st, rp) = (handle.clone(), state.clone(), repaint.clone());
                tokio::spawn(async move {
                    let name = st
                        .lock()
                        .unwrap()
                        .playlists
                        .iter()
                        .find(|p| p.id == id)
                        .map(|p| p.name.clone())
                        .unwrap_or_default();
                    set(&st, &rp, |s| s.view = name);

                    if let Some(t) = cache::list_get::<Vec<Track>>(&id) {
                        show(&st, &rp, t);
                    } else {
                        busy(&st, &rp, "loading playlist…".into());
                    }

                    // Contents come from librespot's internal protocol, NOT the Web API, which
                    // 403s playlist tracks for post-2024 apps. Art still needs an HTTP fetch.
                    let Some(h) = h else { return };
                    tracing::info!("opening playlist {id}");
                    match h.playlist_tracks(&id).await {
                        Ok(t) => {
                            tracing::info!("playlist {id} → {} tracks", t.len());
                            cache::list_put(&id, &t);
                            if let Some(api) = api(&Some(h), &st, &rp).await {
                                finish_tracks(&st, &rp, t, &api).await;
                            }
                        }
                        Err(e) => fail(&st, &rp, format!("playlist: {e}")),
                    }
                });
            }

            Cmd::Play(uri) => {
                if let Some(h) = &handle {
                    record(&state, &mut run, &mut taste, &repaint);
                    // Whatever list is on screen becomes the queue, starting at the clicked track.
                    queue = state.lock().unwrap().tracks.clone();
                    qpos = queue.iter().position(|t| t.uri == uri).unwrap_or(0);
                    paused = false;
                    if let Some(t) = queue.get(qpos) {
                        taste.observe(t);
                        run.push((t.clone(), 0.0));
                    }
                    start(&state, &repaint, h, &queue, qpos);
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
                    set(&state, &repaint, |s| {
                        if let Some(n) = &mut s.now {
                            n.paused = paused;
                            n.since = if paused { None } else { Some(Instant::now()) };
                        }
                    });
                }
            }

            Cmd::Next => {
                if let Some(h) = &handle {
                    record(&state, &mut run, &mut taste, &repaint);
                    if qpos + 1 < queue.len() {
                        qpos += 1;
                        paused = false;
                        if let Some(t) = queue.get(qpos) {
                            taste.observe(t);
                            run.push((t.clone(), 0.0));
                        }
                        start(&state, &repaint, h, &queue, qpos);
                    } else if state.lock().unwrap().autoplay && !queue.is_empty() {
                        // Queue is dry and radio is on: extend it from the track that just played,
                        // feeding the recent history back so the station doesn't repeat itself.
                        set(&state, &repaint, |s| s.radio_loading = true);
                        let seed = queue[qpos].uri.clone();
                        let prev: Vec<String> = queue
                            .iter()
                            .rev()
                            .take(10)
                            .map(|t| t.uri.clone())
                            .collect();
                        match radio(h, &seed, &prev).await {
                            Ok(more) if !more.is_empty() => {
                                // Spotify picks the candidates; the taste model picks the order.
                                let more = taste.rank(more);
                                tracing::info!("radio: queued {} more tracks", more.len());
                                set(&state, &repaint, |s| {
                                    s.radio_loading = false;
                                    s.status = format!("radio: +{} tracks", more.len());
                                });
                                queue.extend(more);
                                qpos += 1;
                                paused = false;
                                if let Some(t) = queue.get(qpos) {
                                    taste.observe(t);
                                    run.push((t.clone(), 0.0));
                                }
                                start(&state, &repaint, h, &queue, qpos);
                            }
                            Ok(_) => {
                                set(&state, &repaint, |s| s.radio_loading = false);
                                stop_playback(&state, &repaint, h);
                            }
                            Err(e) => {
                                set(&state, &repaint, |s| s.radio_loading = false);
                                fail(&state, &repaint, format!("radio: {e}"));
                                stop_playback(&state, &repaint, h);
                            }
                        }
                    } else {
                        stop_playback(&state, &repaint, h);
                    }
                }
            }

            Cmd::Prev => {
                if let Some(h) = &handle {
                    // Restart the track first, like every other player; only jump back if we're
                    // already near its start.
                    let near_start = state
                        .lock()
                        .unwrap()
                        .now
                        .as_ref()
                        .is_some_and(|n| n.elapsed_ms() < 3000);
                    if near_start && qpos > 0 {
                        qpos -= 1;
                        start(&state, &repaint, h, &queue, qpos);
                    } else {
                        h.seek(0);
                        set(&state, &repaint, |s| {
                            if let Some(n) = &mut s.now {
                                n.position_ms = 0;
                                n.since = Some(Instant::now());
                            }
                        });
                    }
                    paused = false;
                }
            }

            Cmd::Volume(v) => {
                if let Some(h) = &handle {
                    h.set_volume(v);
                    set(&state, &repaint, |s| s.volume = v);
                }
            }

            Cmd::SetAutoplay(on) => {
                set(&state, &repaint, |s| s.autoplay = on);
            }

            Cmd::TrainTaste => {
                let already = taste.trained_sequences();
                // Playlists are curated orderings — a human already decided these tracks belong
                // next to each other. That's the best free training data available, and it's
                // already sitting in the cache, so this costs no network at all.
                let ids: Vec<String> = state
                    .lock()
                    .unwrap()
                    .playlists
                    .iter()
                    .map(|p| p.id.clone())
                    .collect();
                let mut learned = 0usize;
                for id in ids {
                    if taste.has_learned(&id) {
                        continue;
                    }
                    if let Some(tracks) = cache::list_get::<Vec<Track>>(&id) {
                        taste.learn_corpus(&id, &tracks);
                        learned += 1;
                    }
                }
                // Liked Songs: not an ordering he chose, but a strong statement about what he
                // likes at all. Learned as one sequence so its tracks enter the model's space.
                if !taste.has_learned("liked") {
                    if let Some(liked) = cache::list_get::<Vec<Track>>("liked") {
                        taste.learn_corpus("liked", &liked);
                        learned += 1;
                    }
                }

                // Past listening runs, replayed with their outcomes.
                if let Some(history) = cache::list_get::<Vec<Vec<(Track, f32)>>>("history") {
                    for past in history.iter().take(50) {
                        taste.learn_plays(past);
                    }
                }
                let n = taste.trained_sequences();
                tracing::info!(
                    "taste: trained on {learned} cached playlists ({n} sequences, was {already})"
                );

                // Fetch Spotify's REAL audio features for everything we know about — energy,
                // valence, tempo. The Web API 403s these; the internal service serves them. They're
                // immutable per track, so once they're in the model file we never fetch them again.
                if let Some(h) = handle.clone() {
                    let (st, rp) = (state.clone(), repaint.clone());
                    let known: Vec<String> = {
                        let s = st.lock().unwrap();
                        let mut ids: Vec<String> = s
                            .tracks
                            .iter()
                            .map(|t| nocturne_taste::track_id(&t.uri).to_string())
                            .collect();
                        ids.sort();
                        ids.dedup();
                        ids
                    };
                    let missing: Vec<String> = known
                        .into_iter()
                        .filter(|id| !taste.features().contains_key(id))
                        .take(400)
                        .collect();
                    if !missing.is_empty() {
                        tracing::info!("taste: fetching analysis for {} tracks", missing.len());
                        let got = h.audio_features_many(&missing).await;
                        tracing::info!("taste: got analysis for {}/{}", got.len(), missing.len());
                        taste.add_features(got.into_iter().collect());
                        let _ = rp;
                    }
                }

                let n = taste.trained_sequences();
                if let Err(e) = taste.save(&cache::model_path()) {
                    tracing::warn!("taste: could not save model: {e}");
                }
                set(&state, &repaint, |s| {
                    s.taste_trained = n;
                    s.taste_features = taste.feature_count();
                });
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

/// Close out the currently-playing track: how much of it did he actually hear? That fraction is
/// the reward signal — a track skipped at 5% and one played to the end mean opposite things, and
/// it's the only feedback available without ever asking him to rate anything.
///
/// Runs are learned (and persisted) once they're long enough to carry a pattern.
fn record(
    state: &Shared,
    run: &mut Vec<(Track, f32)>,
    taste: &mut Taste,
    repaint: &(impl Fn() + Send),
) {
    let completion = {
        let s = state.lock().unwrap();
        match &s.now {
            Some(n) if n.duration_ms > 0 => {
                (n.elapsed_ms() as f32 / n.duration_ms as f32).clamp(0.0, 1.0)
            }
            _ => return,
        }
    };
    if let Some(last) = run.last_mut() {
        last.1 = completion;
    }

    // Learn in windows rather than waiting for the app to close — a session that's never cleanly
    // exited would otherwise teach the model nothing.
    if run.len() >= 4 {
        taste.learn_plays(run);
        let mut history: Vec<Vec<(Track, f32)>> =
            cache::list_get("history").unwrap_or_default();
        history.push(run.clone());
        // Bound it: this is a training set, not an archive.
        let len = history.len();
        if len > 100 {
            history.drain(..len - 100);
        }
        cache::list_put("history", &history);
        if let Err(e) = taste.save(&cache::model_path()) {
            tracing::warn!("taste: could not save model: {e}");
        }
        let n = taste.trained_sequences();
        set(state, repaint, |s| s.taste_trained = n);
        run.clear();
    }
}

/// Ask Spotify's radio for tracks that follow on from `seed`.
async fn radio(
    h: &Arc<NocturneHandle>,
    seed: &str,
    previous: &[String],
) -> Result<Vec<Track>, String> {
    let seed = librespot_uri(seed)?;
    let prev: Vec<librespot_core::SpotifyUri> =
        previous.iter().filter_map(|u| librespot_uri(u).ok()).collect();
    h.radio_from(&seed, &prev, 30)
        .await
        .map_err(|e| e.to_string())
}

fn stop_playback(state: &Shared, repaint: &(impl Fn() + Send), h: &Arc<NocturneHandle>) {
    h.stop();
    set(state, repaint, |s| {
        if let Some(n) = &mut s.now {
            n.paused = true;
            n.since = None;
        }
    });
}

/// Load queue[i] and reflect it in the UI immediately (don't wait for the player event).
fn start(
    state: &Shared,
    repaint: &(impl Fn() + Send),
    h: &Arc<NocturneHandle>,
    queue: &[Track],
    i: usize,
) {
    let Some(t) = queue.get(i) else { return };
    set(state, repaint, |s| {
        s.current_uri = Some(t.uri.clone());
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
    match librespot_uri(&t.uri) {
        Ok(u) => h.play_uri(u),
        Err(e) => fail(state, repaint, e),
    }
}

/// Player events → now-playing state. Runs for the life of the session.
fn spawn_event_pump(
    h: Arc<NocturneHandle>,
    state: Shared,
    tx: mpsc::UnboundedSender<Cmd>,
    repaint: Box<dyn Fn() + Send>,
) {
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
                        n.position_ms = n.duration_ms;
                        n.since = None;
                        // Roll on to the next queued track rather than going silent.
                        let _ = tx.send(Cmd::Next);
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

/// Paint a list of tracks with no network work at all (used for cache hits).
fn show(state: &Shared, repaint: &(impl Fn() + Send), tracks: Vec<Track>) {
    set(state, repaint, |s| {
        s.busy = false;
        s.status = format!("{} tracks", tracks.len());
        s.tracks = tracks;
    });
}

/// Show the results immediately, then pull cover art in the background.
///
/// Two things this must NOT do, both of which it used to. It must not fetch art serially — a
/// 350-track playlist is 350 round trips, and the list sat empty for all of them. And it must not
/// fetch art *on the command loop* — that blocked every other command (opening another playlist,
/// pressing play) behind hundreds of image downloads. So: hand the tracks to the UI, then spawn the
/// art fetch and return.
async fn finish_tracks(
    state: &Shared,
    repaint: &(impl Fn() + Send + Sync + Clone + 'static),
    tracks: Vec<Track>,
    api: &Client,
) {
    let mut urls: Vec<String> = {
        let seen = state.lock().unwrap();
        let mut urls: Vec<String> = tracks
            .iter()
            .filter_map(|t| t.art_url.clone())
            .filter(|u| !seen.art.contains_key(u))
            .collect();
        urls.sort();
        urls.dedup();
        urls
    };

    // Serve what's already on disk without touching the network.
    {
        let mut hits = Vec::new();
        urls.retain(|u| match cache::art_get(u) {
            Some(bytes) => {
                hits.push((u.clone(), bytes));
                false
            }
            None => true,
        });
        if !hits.is_empty() {
            let mut s = state.lock().unwrap();
            for (u, b) in hits {
                s.art.insert(u, b);
            }
        }
    }

    set(state, repaint, |s| {
        s.busy = false;
        s.status = format!("{} tracks", tracks.len());
        s.tracks = tracks;
    });

    let state = state.clone();
    let repaint = repaint.clone();
    let token = api.token().to_string();
    tokio::spawn(async move {
        use futures_util::StreamExt;
        let art = Arc::new(Client::new(token));
        let mut stream = futures_util::stream::iter(urls)
            .map(|url| {
                let art = art.clone();
                async move { art.fetch_art(&url).await.ok().map(|b| (url, b)) }
            })
            .buffer_unordered(12);
        while let Some(got) = stream.next().await {
            if let Some((url, bytes)) = got {
                cache::art_put(&url, &bytes);
                state.lock().unwrap().art.insert(url, bytes);
                repaint();
            }
        }
    });
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
