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
    /// Start a radio from a mood phrase ("chill winter lofi vibes").
    MoodRadio(String),
    /// Replace the queue with this list and play it from the top.
    PlayQueue(Vec<Track>),
    /// Add/remove a track from the local library. Carries the whole Track, because a locally-liked
    /// song has to be MERGED INTO the Liked Songs list — a bare URI can't be rendered.
    ToggleLike(Track),
    /// Jump to a track already in the queue (clicking Up Next).
    JumpTo(usize),
    /// Resolve the big-cover URL for these queued tracks and stream the art into the cache, so the
    /// UI can upscale covers for tracks that haven't played yet. Carries queue indices, because the
    /// resolved URL has to be written back onto the queue entry itself.
    PrefetchBigArt(Vec<usize>),
    /// Restore the last listening session (queue, track, position) — paused.
    Resume,
    /// Show the list the currently-playing queue came from.
    ShowPlayingList,
    /// Open the current temp radio playlist.
    ShowRadioPlaylist,
    /// Persist the temp radio playlist to Spotify as a real playlist.
    SaveRadioToSpotify,
    /// Add a track to a playlist. Local: Spotify 403s playlist writes for restricted apps too.
    AddToPlaylist(String, Track),
}

#[derive(Default, Clone)]
pub struct NowPlaying {
    pub name: String,
    pub artists: String,
    pub duration_ms: u32,
    pub art_url: Option<String>,
    /// Full-resolution cover for the full-screen view.
    pub art_big: Option<String>,
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
    /// True while the background analysis backfill is still running.
    pub analyzing: bool,
    /// The live queue and where we are in it — drives Up Next / History panels.
    pub queue: Vec<Track>,
    pub qpos: usize,
    /// The current auto-generated radio playlist: a real, named list that lives only on this disk
    /// until the next radio replaces it. `None` until a radio has been built.
    pub radio_playlist: Option<RadioPlaylist>,
    /// The name of the list the QUEUE came from — which is not the same as `view`, because you can
    /// be browsing one playlist while a different one is playing. The back arrow uses this.
    pub queue_view: String,
    /// Locally liked track URIs.
    ///
    /// Spotify blocks library WRITES for restricted apps (`PUT /v1/me/tracks` → 403, with the
    /// user-library-modify scope granted, and no internal endpoint exists either). So likes live
    /// here, on disk, and are merged with Spotify's Liked Songs for display. They do not sync back.
    pub liked: std::collections::HashSet<String>,
    /// Decoded art bytes keyed by URL; the UI turns these into textures once.
    pub art: std::collections::HashMap<String, Vec<u8>>,
    /// Art URLs that came out of [`cache::art_fetch_best`] — i.e. the best the CDN had, master or
    /// otherwise.
    ///
    /// The UI upscales *only* these. A restored session hands us a queue whose `art_big` is still a
    /// 640px URL from an older run, and whose bytes may already be on disk; without this gate the
    /// UI happily burned a GPU pass upscaling the 640 before the master resolved a second later.
    pub art_best: std::collections::HashSet<String>,
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
            analyzing: false,
            queue: Vec::new(),
            qpos: 0,
            queue_view: String::new(),
            radio_playlist: None,
            liked: Default::default(),
            art: Default::default(),
            art_best: Default::default(),
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
    // Behind a lock: the analysis backfill runs in the background while the app is in use. The
    // user shouldn't wait on it, and it shouldn't wait on the user.
    let taste: Arc<Mutex<Taste>> = Arc::new(Mutex::new(
        Taste::load(&cache::model_path()).unwrap_or_default(),
    ));
    let mut backfilling = false;
    // The last radio survives a restart — it's a playlist, not a transient.
    if let Some(pl) = cache::list_get::<RadioPlaylist>("radio-playlist") {
        state.lock().unwrap().radio_playlist = Some(pl);
    }
    // A restored session shows a track but has NOT loaded it into the player. The first play must
    // load it (and seek), not just un-pause. Inferring this from `resume_at > 0` was the bug that
    // made a resumed track silent whenever it was restored at 0:00 — pressing play un-paused a
    // player with nothing in it.
    let mut resume_at: u32 = 0;
    let mut needs_load = false;
    // Likes = Spotify's Liked Songs (seeded when the list loads) PLUS local additions, MINUS local
    // removals. Two overlay sets, because Spotify won't take the write and we still have to
    // represent "he un-liked a track that Spotify thinks he likes".
    let mut local_added: Vec<Track> = cache::list_get("local-likes").unwrap_or_default();
    let mut local_removed: std::collections::HashSet<String> =
        cache::list_get::<Vec<String>>("local-unlikes").unwrap_or_default().into_iter().collect();
    {
        let mut s = state.lock().unwrap();
        s.liked = local_added.iter().map(|t| t.uri.clone()).collect();
    }
    // The current listening run, with how much of each track was actually heard. Only the finishes
    // in it are ever learned — see `nocturne_taste::FINISHED`.
    let mut run: Vec<(Track, f32)> = Vec::new();
    // URIs Nocturne itself chose for him: the mood playlist, plus anything radio autoplay appended.
    // Finishing or liking one of these is feedback on the RECOMMENDER, not just on the song, so it
    // trains at `OURS_WEIGHT`.
    let mut ours: std::collections::HashSet<String> = cache::list_get::<RadioPlaylist>("radio-playlist")
        .map(|p| p.tracks.iter().map(|t| t.uri.clone()).collect())
        .unwrap_or_default();

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
                            Ok(t) => finish_tracks(&st, &rp, t).await,
                            Err(e) => fail(&st, &rp, format!("search: {e}")),
                        }
                    }
                });
            }
            Cmd::Search(_) => {}

            Cmd::LoadSaved => {
                let (h, st, rp) = (handle.clone(), state.clone(), repaint.clone());
                let tx2 = self_tx.clone();
                let removed = local_removed.clone();
                let added = local_added.clone();
                tokio::spawn(async move {
                    set(&st, &rp, |s| s.view = "Liked Songs".into());
                    // Paint from disk first; the network refresh lands behind the visible list.
                    // Liked Songs = Spotify's list, minus local un-likes, plus local likes. Merge
                    // is done here (not just in the heart state) so a track you like actually shows
                    // up in your library.
                    let merge = |mut list: Vec<Track>| -> Vec<Track> {
                        list.retain(|t| !removed.contains(&t.uri));
                        for t in &added {
                            if !list.iter().any(|x| x.uri == t.uri) {
                                list.insert(0, t.clone());
                            }
                        }
                        list
                    };

                    let cached: Option<Vec<Track>> = cache::list_get::<Vec<Track>>("liked").map(merge);
                    let cached_liked = cached.is_some();
                    if let Some(t) = &cached {
                        seed_likes(&st, t, &removed);
                    }
                    if let Some(t) = cached {
                        if std::env::var_os("NOCTURNE_PLAY_FIRST").is_some() {
                            if let Some(first) = t.first() {
                                let _ = tx2.send(Cmd::Play(first.uri.clone()));
                            }
                        }
                        show(&st, &rp, t);
                    } else {
                        busy(&st, &rp, "loading liked songs…".into());
                    }
                    if let Some(api) = api(&h, &st, &rp).await {
                        match api.saved_tracks(5000).await {
                            Ok(t) if t.is_empty() && cached_liked => {
                                // Don't let an empty/failed refresh blank out a good cached list.
                                tracing::warn!("liked songs came back empty — keeping cache");
                            }
                            Ok(t) => {
                                cache::list_put("liked", &t);
                                let t = merge(t);
                                seed_likes(&st, &t, &removed);
                                // Debug hook: play the first track automatically, so time-to-audio
                                // can be measured without a human clicking.
                                if std::env::var_os("NOCTURNE_PLAY_FIRST").is_some() {
                                    if let Some(first) = t.first() {
                                        let _ = tx2.send(Cmd::Play(first.uri.clone()));
                                    }
                                }
                                finish_tracks(&st, &rp, t).await
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
                            // Debug hook: fire a mood radio at startup, so the real in-app path can
                            // be exercised (and compared across runs) without a human clicking.
                            if let Ok(mood) = std::env::var("NOCTURNE_MOOD") {
                                let _ = self_tx.send(Cmd::MoodRadio(mood));
                            }
                            // NOCTURNE_OPEN=<name substring> opens a playlist on startup.
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
                        Ok(mut t) => {
                            tracing::info!("playlist {id} → {} tracks", t.len());
                            // Locally-added tracks live alongside Spotify's, since Spotify won't
                            // take the write.
                            if let Some(extra) = cache::list_get::<Vec<Track>>(&format!("added-{id}")) {
                                for x in extra {
                                    if !t.iter().any(|y| y.uri == x.uri) {
                                        t.push(x);
                                    }
                                }
                            }
                            cache::list_put(&id, &t);
                            // Still gated on being logged in, even though the art fetch no longer
                            // needs the token — the CDN is public.
                            if api(&Some(h), &st, &rp).await.is_some() {
                                finish_tracks(&st, &rp, t).await;
                            }
                        }
                        Err(e) => fail(&st, &rp, format!("playlist: {e}")),
                    }
                });
            }

            Cmd::Play(uri) => {
                if let Some(h) = &handle {
                    needs_load = false;
                    record(&state, &mut run, &ours, &taste, &repaint);
                    // Whatever list is on screen becomes the queue, starting at the clicked track.
                    queue = state.lock().unwrap().tracks.clone();
                    qpos = queue.iter().position(|t| t.uri == uri).unwrap_or(0);
                    paused = false;
                    if let Some(t) = queue.get(qpos) {
                        taste.lock().unwrap().observe(t);
                        run.push((t.clone(), 0.0));
                    }
                    start(&state, &repaint, h, &queue, qpos);
                }
            }

            Cmd::PlayPause => {
                if let Some(h) = &handle {
                    // First play after a restore: the player has nothing loaded, so load the track
                    // and seek to where he stopped rather than restarting it.
                    if needs_load && !queue.is_empty() {
                        let at = resume_at;
                        resume_at = 0;
                        needs_load = false;
                        paused = false;
                        start(&state, &repaint, h, &queue, qpos);
                        if at > 0 {
                            h.seek(at);
                            set(&state, &repaint, |s| {
                                if let Some(n) = &mut s.now {
                                    n.position_ms = at;
                                    n.since = Some(Instant::now());
                                }
                            });
                        }
                        continue;
                    }
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
                    needs_load = false;
                    record(&state, &mut run, &ours, &taste, &repaint);
                    if qpos + 1 < queue.len() {
                        qpos += 1;
                        paused = false;
                        if let Some(t) = queue.get(qpos) {
                            taste.lock().unwrap().observe(t);
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
                                // Spotify picks the candidates; his listening picks the order —
                                // biased toward what he finishes and what he's hearted. Nothing is
                                // filtered out for resembling a skip: that filter existed, and it
                                // was a negative built from ambiguous evidence, which is exactly
                                // what this model refuses to do.
                                let more = taste.lock().unwrap().rank(more);
                                tracing::info!("radio: queued {} more tracks", more.len());
                                // These are OUR picks — finishing or liking one grades the model.
                                ours.extend(more.iter().map(|t| t.uri.clone()));
                                set(&state, &repaint, |s| {
                                    s.radio_loading = false;
                                    s.status = format!("radio: +{} tracks", more.len());
                                });
                                queue.extend(more);
                                qpos += 1;
                                paused = false;
                                if let Some(t) = queue.get(qpos) {
                                    taste.lock().unwrap().observe(t);
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

            Cmd::MoodRadio(phrase) => {
                let (h, st, rp, ta, tx2) = (
                    handle.clone(),
                    state.clone(),
                    repaint.clone(),
                    taste.clone(),
                    self_tx.clone(),
                );
                tokio::spawn(async move {
                    // Haiku reads arbitrary phrasing when a key is set; the word list is the
                    // offline floor. Either way we end up with the same acoustic target.
                    let (m, understood) = nocturne_taste::llm::mood_for(&phrase).await;
                    if !understood {
                        fail(
                            &st,
                            &rp,
                            "don't know that mood — try words like chill, hype, sad, lofi, winter"
                                .into(),
                        );
                        return;
                    }
                    let Some(h) = h else { return };
                    let target = nocturne_taste::mood::acoustic_vec(&m.to_features());

                    busy(&st, &rp, format!("building “{phrase}” radio…"));
                    set(&st, &rp, |s| s.view = format!("Mood: {phrase}"));

                    // His library — the seed pool, and the "already owns this" filter.
                    let mut library: Vec<Track> = cache::list_get("liked").unwrap_or_default();
                    let ids: Vec<String> =
                        st.lock().unwrap().playlists.iter().map(|p| p.id.clone()).collect();
                    for id in ids {
                        if let Some(ts) = cache::list_get::<Vec<Track>>(&id) {
                            library.extend(ts);
                        }
                    }
                    let owned: std::collections::HashSet<String> =
                        library.iter().map(|t| t.uri.clone()).collect();

                    // What he's already played — a discovery radio shouldn't re-serve it.
                    let history: Vec<(Track, f32)> =
                        cache::list_get::<Vec<Vec<(Track, f32)>>>("history")
                            .unwrap_or_default()
                            .into_iter()
                            .flatten()
                            .collect();
                    let heard: std::collections::HashSet<String> =
                        history.iter().map(|(t, _)| t.uri.clone()).collect();

                    // 1. Seeds: library tracks matching the mood AND his taste. Sampled, not
                    //    top-N, so the same phrase doesn't seed the identical station every time.
                    let seed_pool = ta
                        .lock()
                        .unwrap()
                        .nearest_mood_for_me(&library, &target, &history, &phrase, 25);
                    if seed_pool.is_empty() {
                        fail(&st, &rp, "no analyzed tracks yet — still learning".into());
                        return;
                    }
                    let seeds = sample(&seed_pool, 6);

                    // 2. Candidates from OUTSIDE the library: Spotify's station for each seed.
                    let mut candidates: Vec<Track> = Vec::new();
                    for seed in &seeds {
                        if let Ok(u) = librespot_uri(&seed.uri) {
                            match h.radio_from(&u, &[], 40).await {
                                Ok(more) => candidates.extend(more),
                                Err(e) => tracing::warn!("station for {}: {e}", seed.name),
                            }
                        }
                    }
                    candidates.retain(|t| !owned.contains(&t.uri) && !heard.contains(&t.uri));
                    candidates.sort_by(|a, b| a.uri.cmp(&b.uri));
                    candidates.dedup_by(|a, b| a.uri == b.uri);
                    tracing::info!(
                        "mood “{phrase}”: {} NEW candidates from {} seeds",
                        candidates.len(),
                        seeds.len()
                    );

                    // 3. Analyze the new ones — we've never seen them, so we have no features.
                    let need: Vec<String> = {
                        let t = ta.lock().unwrap();
                        candidates
                            .iter()
                            .map(|t| nocturne_taste::track_id(&t.uri).to_string())
                            .filter(|id| !t.features().contains_key(id))
                            .collect()
                    };
                    if !need.is_empty() {
                        let got = h.audio_features_many(&need).await;
                        tracing::info!("mood: analyzed {}/{} new tracks", got.len(), need.len());
                        ta.lock().unwrap().add_features(got.into_iter().collect());
                    }

                    // 4. Rank both pools by mood+taste, then SAMPLE from the top of each — a radio
                    //    that returns the identical order every time isn't a radio.
                    let (new_ranked, fam_ranked) = {
                        let t = ta.lock().unwrap();
                        (
                            t.nearest_mood_for_me(&candidates, &target, &history, &phrase, 90),
                            t.nearest_mood_for_me(&library, &target, &history, &phrase, 40),
                        )
                    };
                    let new_picks = sample(&new_ranked, 45);
                    let familiar = sample(&fam_ranked, 15);

                    // 5. Mostly discovery, with a familiar anchor every fourth track.
                    let mut queue_tracks: Vec<Track> = Vec::new();
                    let mut fam = familiar.into_iter();
                    for (i, t) in new_picks.into_iter().enumerate() {
                        if i % 4 == 3 {
                            if let Some(f) = fam.next() {
                                queue_tracks.push(f);
                            }
                        }
                        queue_tracks.push(t);
                    }
                    // If the station gave us nothing new (rate limit, dead seed), fall back to his
                    // library rather than leaving him with silence.
                    if queue_tracks.is_empty() {
                        tracing::warn!("mood: no new tracks — falling back to library");
                        queue_tracks = sample(&fam_ranked, 40);
                    }
                    if queue_tracks.is_empty() {
                        fail(&st, &rp, "radio came back empty — try another mood".into());
                        return;
                    }

                    let n = queue_tracks.len();
                    let fresh = queue_tracks.iter().filter(|t| !owned.contains(&t.uri)).count();
                    let name = radio_name(&phrase);
                    tracing::info!("mood “{phrase}” → “{name}”: {n} tracks ({fresh} new)");

                    // The radio IS a playlist: named, saved to disk, and kept until the next one
                    // replaces it.
                    let pl = RadioPlaylist {
                        name: name.clone(),
                        phrase: phrase.clone(),
                        tracks: queue_tracks.clone(),
                        spotify_id: None,
                    };
                    cache::list_put("radio-playlist", &pl);

                    set(&st, &rp, |s| {
                        s.busy = false;
                        s.status = format!("{n} tracks · {fresh} new");
                        s.tracks = queue_tracks.clone();
                        s.view = name.clone();
                        s.radio_playlist = Some(pl);
                    });
                    let _ = tx2.send(Cmd::PlayQueue(queue_tracks));
                });
            }

            Cmd::PlayQueue(tracks) => {
                if let Some(h) = &handle {
                    needs_load = false;
                    record(&state, &mut run, &ours, &taste, &repaint);
                    // If this queue IS the radio playlist we built, everything in it is ours.
                    let radio_uris: std::collections::HashSet<String> = state
                        .lock()
                        .unwrap()
                        .radio_playlist
                        .as_ref()
                        .map(|p| p.tracks.iter().map(|t| t.uri.clone()).collect())
                        .unwrap_or_default();
                    ours.extend(tracks.iter().filter(|t| radio_uris.contains(&t.uri)).map(|t| t.uri.clone()));
                    queue = tracks;
                    qpos = 0;
                    paused = false;
                    if let Some(t) = queue.first() {
                        taste.lock().unwrap().observe(t);
                        run.push((t.clone(), 0.0));
                    }
                    start(&state, &repaint, h, &queue, qpos);
                }
            }

            Cmd::ToggleLike(track) => {
                let uri = track.uri.clone();
                let now_liked = {
                    let mut s = state.lock().unwrap();
                    if s.liked.remove(&uri) {
                        false
                    } else {
                        s.liked.insert(uri.clone());
                        true
                    }
                };
                if now_liked {
                    if !local_added.iter().any(|t| t.uri == uri) {
                        local_added.push(track.clone());
                    }
                    local_removed.remove(&uri);
                } else {
                    local_removed.insert(uri.clone());
                    local_added.retain(|t| t.uri != uri);
                }
                cache::list_put("local-likes", &local_added);
                cache::list_put(
                    "local-unlikes",
                    &local_removed.iter().cloned().collect::<Vec<_>>(),
                );

                // A like is the least ambiguous thing he can tell us — nobody hits the heart by
                // accident. So it trains immediately, not at the next window. Hearting a track our
                // OWN radio served him is worth double: that's the model being told it was right.
                //
                // An un-like is NOT a negative. It's him tidying his library, months later, in a
                // different mood — evidence about a list, not about music. We just stop counting
                // the positive, and the model keeps whatever the finishes taught it.
                if now_liked {
                    let w = if ours.contains(&uri) {
                        nocturne_taste::OURS_WEIGHT
                    } else {
                        1.0
                    };
                    let mut t = taste.lock().unwrap();
                    t.learn_like(&track, w);
                    if let Err(e) = t.save(&cache::model_path()) {
                        tracing::warn!("taste: could not save model: {e}");
                    }
                    let n = t.trained_sequences();
                    let endorsed = t.endorsements().len();
                    drop(t);
                    tracing::info!("taste: liked “{}” (weight {w}) — {endorsed} endorsements", track.name);
                    set(&state, &repaint, |s| s.taste_trained = n);
                }

                // If Liked Songs is on screen, reflect it NOW — liking a track and not seeing it
                // appear in your library is indistinguishable from the like not working.
                let showing_liked = state.lock().unwrap().view == "Liked Songs";
                if showing_liked {
                    let _ = self_tx.send(Cmd::LoadSaved);
                }
                repaint();
            }

            Cmd::Resume => {
                let Some(sess) = cache::list_get::<Session>("session") else {
                    continue;
                };
                if sess.queue.is_empty() {
                    continue;
                }
                let Some(t) = sess.queue.get(sess.qpos).cloned() else {
                    continue;
                };
                tracing::info!(
                    "resuming: “{}” at {}s ({} in queue)",
                    t.name,
                    sess.position_ms / 1000,
                    sess.queue.len()
                );
                queue = sess.queue.clone();
                qpos = sess.qpos;
                paused = true;

                // Restore the UI to exactly where he left it — but PAUSED. Reopening an app should
                // not start blasting music at you.
                let art = t
                    .art_url
                    .as_ref()
                    .and_then(|u| cache::art_get(u).map(|b| (u.clone(), b)));
                set(&state, &repaint, |s| {
                    s.queue = sess.queue.clone();
                    s.qpos = sess.qpos;
                    // Restore the QUEUE, not the browsed list. Setting `tracks` here overwrote the
                    // visible list with the queue while keeping the old view's name — which is how
                    // "Liked Songs" ended up showing 56 tracks instead of 2197. What you're
                    // browsing and what's playing are different things; the chevron bridges them.
                    s.queue_view = sess.view.clone();
                    s.current_uri = Some(t.uri.clone());
                    s.now = Some(NowPlaying {
                        name: t.name.clone(),
                        artists: t.artists.clone(),
                        duration_ms: t.duration_ms,
                        art_url: t.art_url.clone(),
                        art_big: t.art_big.clone(),
                        paused: true,
                        position_ms: sess.position_ms,
                        since: None,
                    });
                    if let Some((u, b)) = art {
                        s.art.entry(u).or_insert(b);
                    }
                });
                resume_at = sess.position_ms;
                needs_load = true;

                // Fetch the cover for the restored track. `start()` normally does this, but a
                // resume deliberately doesn't start playback — so the art has to be pulled here or
                // the full-screen view opens on a grey square.
                if let Some(h) = handle.clone() {
                    let (st, rp) = (state.clone(), repaint.clone());
                    let (uri, known) = (t.uri.clone(), t.art_big.clone());
                    tokio::spawn(async move {
                        let Some(big) = (match known {
                            Some(b) => Some(b),
                            None => h.big_cover(&uri).await,
                        }) else {
                            return;
                        };
                        // The master, not the 640 — and `big` becomes whichever URL actually served.
                        if let Some((big, bytes)) = cache::art_fetch_best(&big).await {
                            let mut s = st.lock().unwrap();
                            s.art_best.insert(big.clone());
                            s.art.insert(big.clone(), bytes);
                            if let Some(n) = &mut s.now {
                                n.art_big = Some(big);
                            }
                            drop(s);
                            rp();
                        }
                    });
                }
            }

            Cmd::ShowPlayingList => {
                set(&state, &repaint, |s| {
                    if !s.queue.is_empty() {
                        s.tracks = s.queue.clone();
                        s.view = s.queue_view.clone();
                        s.status = format!("{} tracks", s.queue.len());
                    }
                });
            }

            Cmd::ShowRadioPlaylist => {
                set(&state, &repaint, |s| {
                    if let Some(pl) = s.radio_playlist.clone() {
                        s.tracks = pl.tracks;
                        s.view = pl.name;
                        s.status = format!("{} tracks", s.tracks.len());
                    }
                });
            }

            Cmd::SaveRadioToSpotify => {
                let Some(pl) = state.lock().unwrap().radio_playlist.clone() else {
                    continue;
                };
                if pl.spotify_id.is_some() {
                    set(&state, &repaint, |s| {
                        s.status = format!("“{}” is already on Spotify", pl.name)
                    });
                    continue;
                }
                let (h, st, rp) = (handle.clone(), state.clone(), repaint.clone());
                tokio::spawn(async move {
                    let Some(api) = api(&h, &st, &rp).await else { return };
                    busy(&st, &rp, format!("saving “{}” to Spotify…", pl.name));
                    let uris: Vec<String> = pl.tracks.iter().map(|t| t.uri.clone()).collect();
                    match api.create_playlist(&pl.name, &uris).await {
                        Ok(id) => {
                            tracing::info!("saved “{}” to Spotify as {id}", pl.name);
                            let mut saved = pl.clone();
                            saved.spotify_id = Some(id);
                            cache::list_put("radio-playlist", &saved);
                            set(&st, &rp, |s| {
                                s.busy = false;
                                s.status = format!("“{}” saved to Spotify", saved.name);
                                s.radio_playlist = Some(saved);
                            });
                            // It's a real playlist now — pull the list again so it appears.
                            let _ = api.playlists(500).await.map(|p| {
                                cache::list_put("playlists", &p);
                                set(&st, &rp, |s| s.playlists = p);
                            });
                        }
                        Err(e) => {
                            // Honest: the local copy is untouched and still works.
                            fail(
                                &st,
                                &rp,
                                format!("Spotify refused the playlist ({e}) — kept locally"),
                            );
                        }
                    }
                });
            }

            Cmd::AddToPlaylist(id, track) => {
                // Local, like likes: Spotify 403s POST /v1/playlists/{id}/tracks for our app just
                // as it does library writes. Additions are merged in when the playlist is opened.
                let key = format!("added-{id}");
                let mut added: Vec<Track> = cache::list_get(&key).unwrap_or_default();
                if !added.iter().any(|t| t.uri == track.uri) {
                    added.push(track.clone());
                    cache::list_put(&key, &added);
                }
                let name = state
                    .lock()
                    .unwrap()
                    .playlists
                    .iter()
                    .find(|p| p.id == id)
                    .map(|p| p.name.clone())
                    .unwrap_or_default();
                set(&state, &repaint, |s| {
                    s.status = format!("added “{}” to {name}", track.name);
                });
            }

            // The UI keeps 8× covers for a window around the playing track, so it needs the big
            // cover for tracks it hasn't played yet. Resolving one costs an API call, so the UI
            // asks only for the handful in its window, and only once each.
            Cmd::PrefetchBigArt(idxs) => {
                if let Some(h) = handle.clone() {
                    for i in idxs {
                        let Some(t) = queue.get(i).cloned() else { continue };
                        let (st, rp) = (state.clone(), repaint.clone());
                        let h = h.clone();
                        tokio::spawn(async move {
                            let big = match t.art_big.clone() {
                                Some(b) => Some(b),
                                None => h.big_cover(&t.uri).await,
                            };
                            let Some(big) = big else { return };

                            let Some((big, bytes)) = cache::art_fetch_best(&big).await else {
                                return;
                            };

                            // Write the resolved URL back onto the queue entry: it is the key the
                            // UI's resident window is built from, so without this the cover is
                            // invisible to the upscaler no matter how many times we fetch its bytes.
                            let mut s = st.lock().unwrap();
                            if let Some(q) = s.queue.get_mut(i) {
                                if q.uri == t.uri {
                                    q.art_big = Some(big.clone());
                                }
                            }
                            s.art_best.insert(big.clone());
                            s.art.entry(big).or_insert(bytes);
                            drop(s);
                            rp();
                        });
                    }
                }
            }

            Cmd::JumpTo(i) => {
                if let Some(h) = &handle {
                    needs_load = false;
                    if i < queue.len() {
                        record(&state, &mut run, &ours, &taste, &repaint);
                        qpos = i;
                        paused = false;
                        taste.lock().unwrap().observe(&queue[i]);
                        run.push((queue[i].clone(), 0.0));
                        start(&state, &repaint, h, &queue, qpos);
                    }
                }
            }

            Cmd::SetAutoplay(on) => {
                set(&state, &repaint, |s| s.autoplay = on);
            }

            Cmd::TrainTaste => {
                let mut t = taste.lock().unwrap();
                let already = t.trained_sequences();
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
                    if t.has_learned(&id) {
                        continue;
                    }
                    if let Some(tracks) = cache::list_get::<Vec<Track>>(&id) {
                        t.learn_corpus(&id, &tracks);
                        learned += 1;
                    }
                }
                // Liked Songs: not an ordering he chose, but the clearest statement he's ever made
                // about what he likes at all. Learned as one sequence so its tracks enter the
                // model's space, AND folded into the positive set — re-seeded every launch, and
                // again after each analysis backfill, because a track liked before its features
                // arrived can only be embedded once they have.
                if let Some(liked) = cache::list_get::<Vec<Track>>("liked") {
                    if !t.has_learned("liked") {
                        t.learn_corpus("liked", &liked);
                        learned += 1;
                    }
                    t.seed_endorsements(&liked);
                }

                // Past listening runs — replayed, finishes only.
                if let Some(history) = cache::list_get::<Vec<Vec<(Track, f32)>>>("history") {
                    for past in history.iter().take(50) {
                        t.learn_finishes(past, 1.0);
                    }
                }
                let n = t.trained_sequences();
                tracing::info!(
                    "taste: trained on {learned} cached playlists ({n} sequences, was {already})"
                );

                // Fetch Spotify's REAL audio features for everything we know about — energy,
                // valence, tempo. The Web API 403s these; the internal service serves them. They're
                // immutable per track, so once they're in the model file we never fetch them again.
                // Backfill runs in the BACKGROUND, for the whole library, while he uses the app —
                // no cap and no waiting at startup. Analysis is immutable per track, so once a
                // track lands in the model file it's never fetched again: this converges and stops.
                if let Some(h) = handle.clone() {
                    if !backfilling {
                        backfilling = true;
                        spawn_backfill(h, taste.clone(), state.clone(), repaint.clone());
                    }
                }

                if let Err(e) = t.save(&cache::model_path()) {
                    tracing::warn!("taste: could not save model: {e}");
                }
                let f = t.feature_count();
                drop(t);
                set(&state, &repaint, |s| {
                    s.taste_trained = n;
                    s.taste_features = f;
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

/// Close out the currently-playing track: how much of it did he actually hear?
///
/// The full completion fraction is *stored* — history is a record, and a later idea about what a
/// 40% play means shouldn't be foreclosed now. But only the finishes are *learned*: the model
/// trains on tracks he sat all the way through, and a skip teaches it nothing at all
/// (`nocturne_taste::FINISHED` explains why).
///
/// Runs are learned (and persisted) once they're long enough to carry a pattern.
fn record(
    state: &Shared,
    run: &mut Vec<(Track, f32)>,
    ours: &std::collections::HashSet<String>,
    taste: &Arc<Mutex<Taste>>,
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

    // A fast skip of a track from the current MOOD radio is evidence about the mood, not the
    // track: a summer lofi song in a "winter chill lofi" station is a fine song in the wrong
    // room. File it under the phrase, where it can only ever affect that station again — the
    // taste model itself never sees it.
    if completion < nocturne_taste::QUICK_SKIP {
        let mood = {
            let s = state.lock().unwrap();
            let uri = s.current_uri.clone();
            s.radio_playlist
                .as_ref()
                .filter(|p| uri.as_ref().is_some_and(|u| p.tracks.iter().any(|t| &t.uri == u)))
                .map(|p| p.phrase.clone())
        };
        if let (Some(phrase), Some((track, _))) = (mood, run.last()) {
            tracing::info!(
                "taste: “{}” skipped at {:.0}% — wrong for “{phrase}”, not wrong for him",
                track.name,
                completion * 100.0
            );
            taste.lock().unwrap().skip_in_mood(&phrase, track);
        }
    }

    // Learn in windows rather than waiting for the app to close — a session that's never cleanly
    // exited would otherwise teach the model nothing.
    if run.len() >= 4 {
        // Did this run happen inside music Nocturne chose? If most of what he finished here came
        // from our own radio, the run is grading the recommender, and it counts double.
        let finished: Vec<&Track> = run
            .iter()
            .filter(|(_, c)| *c >= nocturne_taste::FINISHED)
            .map(|(t, _)| t)
            .collect();
        let from_us = finished.iter().filter(|t| ours.contains(&t.uri)).count();
        let weight = if !finished.is_empty() && from_us * 2 >= finished.len() {
            nocturne_taste::OURS_WEIGHT
        } else {
            1.0
        };

        let mut t = taste.lock().unwrap();
        let learned = t.learn_finishes(run, weight);
        tracing::info!(
            "taste: run of {} → learned {learned} finishes (weight {weight}), {} skips dropped",
            run.len(),
            run.len() - learned
        );
        let mut history: Vec<Vec<(Track, f32)>> =
            cache::list_get("history").unwrap_or_default();
        history.push(run.clone());
        // Bound it: this is a training set, not an archive.
        let len = history.len();
        if len > 100 {
            history.drain(..len - 100);
        }
        cache::list_put("history", &history);
        if let Err(e) = t.save(&cache::model_path()) {
            tracing::warn!("taste: could not save model: {e}");
        }
        let n = t.trained_sequences();
        drop(t);
        set(state, repaint, |s| s.taste_trained = n);
        run.clear();
    }
}

/// Backfill Spotify's real audio analysis for the entire library, in the background.
///
/// Every track in every cached list, in chunks, until nothing is left to fetch. It runs while the
/// app is in use rather than making him wait at startup for a 2000-track sweep — the model simply
/// gets sharper as he listens. Analysis is immutable per track, so once a track is in the model
/// file it's never fetched again: this converges and then stops for good.
///
/// The model is saved after every chunk, so quitting mid-backfill keeps the work done so far.
fn spawn_backfill(
    h: Arc<NocturneHandle>,
    taste: Arc<Mutex<Taste>>,
    state: Shared,
    repaint: impl Fn() + Send + Sync + Clone + 'static,
) {
    tokio::spawn(async move {
        let mut corpora: Vec<String> = vec!["liked".to_string()];
        corpora.extend(state.lock().unwrap().playlists.iter().map(|p| p.id.clone()));

        let mut ids: Vec<String> = Vec::new();
        for key in corpora {
            if let Some(tracks) = cache::list_get::<Vec<Track>>(&key) {
                ids.extend(
                    tracks
                        .iter()
                        .map(|t| nocturne_taste::track_id(&t.uri).to_string()),
                );
            }
        }
        ids.sort();
        ids.dedup();

        let todo: Vec<String> = {
            let t = taste.lock().unwrap();
            ids.into_iter()
                .filter(|id| !t.features().contains_key(id))
                .collect()
        };
        if todo.is_empty() {
            tracing::info!("taste: analysis already complete");
            return;
        }

        let total = todo.len();
        tracing::info!("taste: backfilling analysis for {total} tracks (background)");
        set(&state, &repaint, |s| s.analyzing = true);

        let mut done = 0usize;
        let chunks: Vec<_> = todo.chunks(100).collect();
        let last = chunks.len().saturating_sub(1);
        for (i, chunk) in chunks.into_iter().enumerate() {
            let got = h.audio_features_many(chunk).await;
            done += chunk.len();

            // Checkpoint every 5 chunks (and at the end) rather than every chunk — and never hold
            // the model lock across the disk write. Play needs this lock; a 6MB write under it is
            // exactly why starting a track stalled while the backfill ran.
            let checkpoint = i % 5 == 0 || i == last;
            let (f, bytes) = {
                let mut t = taste.lock().unwrap();
                t.add_features(got.into_iter().collect());
                let bytes = if checkpoint { t.to_bytes().ok() } else { None };
                (t.feature_count(), bytes)
            };
            if let Some(bytes) = bytes {
                let path = cache::model_path();
                // Off the async worker entirely: this is blocking file IO.
                let _ = tokio::task::spawn_blocking(move || {
                    if let Err(e) = Taste::write_bytes(&path, &bytes) {
                        tracing::warn!("taste: could not save model: {e}");
                    }
                })
                .await;
            }
            set(&state, &repaint, |s| {
                s.taste_features = f;
                s.analyzing = done < total;
            });
            tracing::info!("taste: analysis {done}/{total} ({f} known)");
        }
        set(&state, &repaint, |s| s.analyzing = false);
        tracing::info!("taste: analysis backfill COMPLETE ({total} tracks)");
    });
}

/// Take `count` from a ranked list, randomly but rank-biased.
///
/// Strict top-N made every radio identical: same phrase, same station, forever. Pure shuffle throws
/// the ranking away. This samples without replacement with weight `1/(rank+2)`, so the best tracks
/// are still the most likely to appear and the order changes every time you press play.
fn sample(ranked: &[Track], count: usize) -> Vec<Track> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    if ranked.len() <= count {
        return ranked.to_vec();
    }

    // Seed from the clock: a radio started twice in the same second may repeat; anything longer
    // apart won't. No RNG dependency for a handful of draws.
    let mut seed = {
        let mut h = DefaultHasher::new();
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
            .hash(&mut h);
        h.finish()
    };
    let mut next = || {
        // xorshift64 — deterministic, fast, good enough to shuffle a playlist.
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };

    let mut pool: Vec<(f64, Track)> = ranked
        .iter()
        .enumerate()
        .map(|(i, t)| (1.0 / (i as f64 + 2.0), t.clone()))
        .collect();

    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let total: f64 = pool.iter().map(|(w, _)| w).sum();
        if total <= 0.0 || pool.is_empty() {
            break;
        }
        let mut pick = next() * total;
        let mut idx = pool.len() - 1;
        for (i, (w, _)) in pool.iter().enumerate() {
            pick -= w;
            if pick <= 0.0 {
                idx = i;
                break;
            }
        }
        out.push(pool.remove(idx).1);
    }
    out
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

/// Point the now-playing at the big cover once we have it.
fn set_big(state: &Shared, uri: &str, big: &str) {
    let mut s = state.lock().unwrap();
    if s.current_uri.as_deref() == Some(uri) {
        if let Some(n) = &mut s.now {
            n.art_big = Some(big.to_string());
        }
    }
}

/// An auto-generated radio, saved as a playlist on this disk only.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct RadioPlaylist {
    pub name: String,
    pub phrase: String,
    pub tracks: Vec<Track>,
    /// Set once it has been pushed to Spotify as a real playlist.
    #[serde(default)]
    pub spotify_id: Option<String>,
}

/// Turn "chill autumn lofi cozy" into "Chill Autumn Lofi Cozy" — a name that looks like something a
/// human would have typed, not a slug.
fn radio_name(phrase: &str) -> String {
    let words: Vec<String> = phrase
        .split_whitespace()
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .collect();
    if words.is_empty() {
        "Radio".to_string()
    } else {
        words.join(" ")
    }
}

/// The last listening session, so closing and reopening Nocturne picks up where he left off
/// instead of dumping him on an empty screen.
#[derive(serde::Serialize, serde::Deserialize, Default)]
pub struct Session {
    pub queue: Vec<Track>,
    pub qpos: usize,
    pub position_ms: u32,
    pub view: String,
}


fn save_session(state: &Shared, queue: &[Track], qpos: usize) {
    let (position_ms, view) = {
        let s = state.lock().unwrap();
        (
            s.now.as_ref().map(|n| n.elapsed_ms()).unwrap_or(0),
            s.view.clone(),
        )
    };
    cache::list_put(
        "session",
        &Session {
            queue: queue.to_vec(),
            qpos,
            position_ms,
            view,
        },
    );
}

/// Load queue[i] and reflect it in the UI immediately (don't wait for the player event).
///
/// Timing is logged: time-to-audio is the number that decides whether this app feels native or
/// feels like a webapp, so it's measured, not assumed.
fn start(
    state: &Shared,
    repaint: &(impl Fn() + Send + Sync + Clone + 'static),
    h: &Arc<NocturneHandle>,
    queue: &[Track],
    i: usize,
) {
    let Some(t) = queue.get(i) else { return };
    // The full-size cover, for the full-screen view. Fetched per track (not per list) because it's
    // ~100KB and only one is ever on screen. If the track predates `art_big` (anything already in
    // the list cache), ask metadata for it — the big URL cannot be derived from the small one.
    {
        let (st, rp, hh) = (state.clone(), repaint.clone(), h.clone());
        let known = t.art_big.clone();
        let uri = t.uri.clone();
        tokio::spawn(async move {
            let big = match known {
                Some(b) => Some(b),
                None => hh.big_cover(&uri).await,
            };
            let Some(big) = big else { return };

            // `art_fetch_best` serves from the disk cache when it can, so a repeat visit costs a
            // file read, not a request — no need to short-circuit on `art` here.
            if let Some((big, bytes)) = cache::art_fetch_best(&big).await {
                {
                    let mut s = st.lock().unwrap();
                    s.art_best.insert(big.clone());
                    s.art.entry(big.clone()).or_insert(bytes);
                }
                set_big(&st, &uri, &big);
                rp();
            }
        });
    }
    set(state, repaint, |s| {
        s.queue = queue.to_vec();
        s.qpos = i;
        s.current_uri = Some(t.uri.clone());
        s.now = Some(NowPlaying {
            name: t.name.clone(),
            artists: t.artists.clone(),
            duration_ms: t.duration_ms,
            art_url: t.art_url.clone(),
            art_big: t.art_big.clone(),
            paused: false,
            position_ms: 0,
            since: Some(Instant::now()),
        });
    });
    save_session(state, queue, i);
    let t0 = Instant::now();
    match librespot_uri(&t.uri) {
        Ok(u) => {
            h.play_uri(u);
            tracing::info!("play: dispatched in {}ms", t0.elapsed().as_millis());
        }
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
                        tracing::info!("play: AUDIO STARTED (position {position_ms}ms)");
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

/// Every track in his Spotify Liked Songs is, by definition, liked — so the hearts must already be
/// filled when the list loads. Local un-likes are honoured on top.
fn seed_likes(state: &Shared, tracks: &[Track], removed: &std::collections::HashSet<String>) {
    let mut s = state.lock().unwrap();
    for t in tracks {
        if !removed.contains(&t.uri) {
            s.liked.insert(t.uri.clone());
        }
    }
}

/// Paint a list of tracks with no network work at all (used for cache hits).
///
/// This must also pull the covers off the disk cache. It didn't, and the result was that any list
/// served from cache — which is most of them — rendered with grey placeholder squares where every
/// album cover should be, even though the images were sitting right there on disk.
fn show(state: &Shared, repaint: &(impl Fn() + Send), tracks: Vec<Track>) {
    let art: Vec<(String, Vec<u8>)> = tracks
        .iter()
        .filter_map(|t| t.art_url.clone())
        .filter_map(|u| cache::art_get(&u).map(|b| (u, b)))
        .collect();
    set(state, repaint, |s| {
        s.busy = false;
        s.status = format!("{} tracks", tracks.len());
        s.tracks = tracks;
        for (u, b) in art {
            s.art.entry(u).or_insert(b);
        }
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
    tokio::spawn(async move {
        use futures_util::StreamExt;
        // Streamed to disk as they arrive: 12 covers in flight, none of them buffered whole in RAM.
        let mut stream = futures_util::stream::iter(urls)
            .map(|url| async move { cache::art_fetch(&url).await.map(|b| (url, b)) })
            .buffer_unordered(12);
        while let Some(got) = stream.next().await {
            if let Some((url, bytes)) = got {
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

