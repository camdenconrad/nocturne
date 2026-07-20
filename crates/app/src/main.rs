mod backend;
mod cache;
mod emoji;
mod fonts;
mod icons;
mod mpris;
mod upscale;

use backend::{Cmd, LockExt, NowPlaying, Repeat, Shared};
use eframe::egui;
use egui::{Align, Color32, Layout, Margin, RichText, Rounding, Stroke, Vec2};
use livewall_uikit::{chrome, theme};
use icons::Icon;
use nocturne_api::fmt_duration;
use std::collections::HashMap;

thread_local! {
    /// `add_controls` is called deep inside scrolling lists with only `&App`. It drops the request
    /// here; `App::add_dialog` picks it up at top level next frame and opens a real window.
    static ADD_REQUEST: std::cell::RefCell<Option<nocturne_api::Track>> =
        const { std::cell::RefCell::new(None) };
}

const SIDEBAR_W: f32 = 228.0;
const NOWPANE_W: f32 = 320.0;
const ROW_H: f32 = 56.0;
const BAR_H: f32 = 96.0;
const ART: f32 = 40.0;

/// How many entries the decoded-texture caches ([`App::textures`], [`App::blurred`]) may hold.
/// Big enough that a full screen of covers never churns; small enough that a long session doesn't
/// accumulate a texture for every cover ever shown.
const TEX_CACHE_CAP: usize = 200;

/// Mark `url` as most-recently-used in a cache's recency queue.
fn lru_touch(order: &mut std::collections::VecDeque<String>, url: &str) {
    if order.back().is_some_and(|u| u == url) {
        return;
    }
    if let Some(i) = order.iter().position(|u| u == url) {
        if let Some(u) = order.remove(i) {
            order.push_back(u);
        }
    }
}

fn main() -> eframe::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()))
        .init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_decorations(false)
            .with_inner_size([1320.0, 820.0])
            .with_min_inner_size([900.0, 600.0])
            .with_app_id("nocturne")
            .with_icon(load_icon()),
        ..Default::default()
    };

    eframe::run_native(
        "Nocturne",
        options,
        Box::new(|cc| {
            fonts::install(&cc.egui_ctx);
            let ctx = cc.egui_ctx.clone();
            let (state, tx) = backend::spawn(move || ctx.request_repaint());
            mpris::spawn(state.clone(), tx.clone());
            Ok(Box::new(App::new(state, tx)))
        }),
    )
}

/// The window/taskbar icon — the same Rune-family art installed to hicolor.
fn load_icon() -> egui::IconData {
    const PNG: &[u8] = include_bytes!("../../../dist/nocturne-128.png");
    match image::load_from_memory(PNG) {
        Ok(img) => {
            let img = img.to_rgba8();
            let (width, height) = (img.width(), img.height());
            egui::IconData {
                rgba: img.into_raw(),
                width,
                height,
            }
        }
        Err(_) => egui::IconData::default(),
    }
}

struct App {
    state: Shared,
    tx: tokio::sync::mpsc::UnboundedSender<Cmd>,
    /// False until we've told the compositor, once, that we are not a fullscreen window.
    unfullscreened: bool,
    query: String,
    mood: String,
    textures: HashMap<String, egui::TextureHandle>,
    /// Recency order of `textures` keys, least-recent first — the eviction queue that keeps the
    /// texture cache (and its GPU memory) bounded. Touched on every hit so covers on screen this
    /// frame are never the ones evicted.
    textures_order: std::collections::VecDeque<String>,
    emoji: emoji::Emoji,
    loaded: bool,
    /// When the next silent-login attempt is allowed, and how many we've made. A single failure
    /// used to latch the sign-in screen for the whole process — a transient network blip at
    /// startup meant clicking through a browser flow that wasn't actually needed.
    autologin_at: Option<std::time::Instant>,
    autologin_attempts: u32,
    /// Local volume while dragging, so the slider doesn't fight the backend each frame.
    volume: f32,
    show_sidebar: bool,
    show_nowpane: bool,
    /// The now-playing view: the whole WINDOW becomes the album art. Not a compositor fullscreen.
    vibe: bool,
    /// Blurred backdrop textures, keyed by art url — built once, reused. The `f32` is the
    /// backdrop's mean luminance AFTER exposure, which decides how hard it gets scrimmed.
    blurred: HashMap<String, (egui::TextureHandle, f32)>,
    /// Recency order of `blurred` keys — same bounding as `textures_order`.
    blurred_order: std::collections::VecDeque<String>,
    /// Textures evicted from a cache, held until the next frame begins.
    ///
    /// Dropping a `TextureHandle` frees its egui texture id. Do that mid-frame and any paint
    /// command already queued against that id refers to a texture that no longer exists — wgpu
    /// rejects the submission and eframe panics with "Texture ... has been destroyed". Eviction
    /// runs from `art`/`backdrop`, which are called *during* painting, so the old code's
    /// assumption that the least-recent entry can't be on screen fails whenever one frame touches
    /// more covers than the cache holds. Holding the handle until the next frame starts means the
    /// free always lands between frames, where nothing references it.
    retired: Vec<egui::TextureHandle>,
    /// The track whose "add to playlist" dialog is open, if any.
    add_to: Option<nocturne_api::Track>,
    /// ESRGAN-upscaled covers resident in VRAM, keyed by art url.
    ///
    /// Deliberately bounded, and it is the *only* place an upscale lives — nothing is written to
    /// disk. An 8000² RGBA texture is ~256MB, so this map IS the memory budget: only the covers in
    /// [`App::hires_window`] are allowed in, and everything else is evicted the frame it falls out.
    /// Without the eviction, every cover played this session would stay resident and a long listen
    /// would eat VRAM until the GPU gave up.
    hires: HashMap<String, egui::TextureHandle>,
    /// Covers whose upscale failed. Never retried — a missing binary won't appear mid-session.
    hires_failed: std::collections::HashSet<String>,
    /// The one cover being upscaled right now. The GPU serialises these passes anyway, so running
    /// six at once would only delay the one the user is actually looking at.
    hires_inflight: Option<String>,
    /// Queue entries we've already asked the backend to resolve a big cover for.
    hires_asked: std::collections::HashSet<String>,
    /// Workers hand back a decoded image; the UI thread owns the GPU upload.
    hires_tx: std::sync::mpsc::Sender<(String, Option<egui::ColorImage>)>,
    hires_rx: std::sync::mpsc::Receiver<(String, Option<egui::ColorImage>)>,
}

/// The cover's corner radius, and the width of the plate it sits on.
///
/// These are shared by the art, the plate and the shadow on purpose — every "slightly off" square
/// in a UI is two shapes that were rounded and inset independently.
const ART_ROUNDING: f32 = 16.0;
const PLATE: f32 = 10.0;
/// The plate itself: a dark mat, translucent so the blurred backdrop still reads through it.
const PLATE_FILL: Color32 = Color32::from_black_alpha(120);

/// How many played tracks keep their upscaled cover, and how many upcoming ones get it built ahead.
///
/// Six 8000² covers resident is ~1.5GB of VRAM (of 16GB on the 4080) — that is the price of instant
/// scrubbing in both directions, and it is the reason [`App::hires`] evicts rather than grows. A
/// pass costs ~12s, which is why the next tracks are built during the current one and never waited
/// on: until a cover's upscale lands, the view shows Spotify's master, which is sharp already.
const HIRES_BACK: usize = 3;
const HIRES_AHEAD: usize = 2;

impl App {
    fn new(state: Shared, tx: tokio::sync::mpsc::UnboundedSender<Cmd>) -> Self {
        let (tx8, rx8) = std::sync::mpsc::channel();
        Self {
            state,
            tx,
            unfullscreened: false,
            query: String::new(),
            mood: String::new(),
            textures: HashMap::new(),
            textures_order: Default::default(),
            emoji: emoji::Emoji::new(),
            loaded: false,
            autologin_at: Some(std::time::Instant::now()),
            autologin_attempts: 0,
            volume: 1.0,
            // Tucked away: full screen is home, and the library is something you pull out.
            show_sidebar: false,
            show_nowpane: true,
            // Full screen IS the app. Browsing is a detour you take and come back from.
            vibe: true,
            retired: Vec::new(),
            blurred: HashMap::new(),
            blurred_order: Default::default(),
            add_to: None,
            hires: HashMap::new(),
            hires_failed: Default::default(),
            hires_inflight: None,
            hires_asked: Default::default(),
            hires_tx: tx8,
            hires_rx: rx8,
        }
    }

    fn send(&self, cmd: Cmd) {
        let _ = self.tx.send(cmd);
    }

    /// Queue the "add to playlist" dialog. Uses interior state rather than opening a popup at the
    /// call site, because the call sites live inside scrolling lists.
    fn request_add_to(&self, track: nocturne_api::Track) {
        // `add_controls` only has &self, so stash it through the command channel's sibling: a cell.
        ADD_REQUEST.with(|c| *c.borrow_mut() = Some(track));
    }

    /// Draw the add-to-playlist dialog, if one was requested. Called once per frame, at top level,
    /// so it is never clipped by a parent layout.
    fn add_dialog(&mut self, ctx: &egui::Context) {
        if let Some(t) = ADD_REQUEST.with(|c| c.borrow_mut().take()) {
            self.add_to = Some(t);
        }
        let Some(track) = self.add_to.clone() else {
            return;
        };
        let playlists = self.state.lock_ok().playlists.clone();

        let mut open = true;
        egui::Window::new("Add to playlist")
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
            .default_width(320.0)
            .show(ctx, |ui| {
                ui.label(RichText::new(&track.name).strong());
                ui.label(RichText::new(&track.artists).weak().small());
                ui.add_space(6.0);
                ui.label(
                    RichText::new("Saved locally — Spotify blocks playlist writes for this app.")
                        .weak()
                        .small(),
                );
                ui.separator();
                egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
                    for p in &playlists {
                        if ui
                            .add_sized(
                                [ui.available_width(), 28.0],
                                egui::Button::new(&p.name).frame(false),
                            )
                            .clicked()
                        {
                            self.send(Cmd::AddToPlaylist(p.id.clone(), track.clone()));
                            self.add_to = None;
                        }
                    }
                });
            });
        if !open || ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.add_to = None;
        }
    }

    /// Decode art bytes into a texture once; later frames hit the cache.
    fn art(&mut self, ctx: &egui::Context, url: &str) -> Option<egui::TextureHandle> {
        if let Some(t) = self.textures.get(url) {
            let t = t.clone();
            lru_touch(&mut self.textures_order, url);
            return Some(t);
        }
        let bytes = self.state.lock_ok().art.get(url).cloned()?;
        let img = image::load_from_memory(&bytes).ok()?.to_rgba8();
        let size = [img.width() as usize, img.height() as usize];
        let color = egui::ColorImage::from_rgba_unmultiplied(size, img.as_raw());
        let tex = ctx.load_texture(url, color, egui::TextureOptions::LINEAR);
        self.textures.insert(url.to_string(), tex.clone());
        self.textures_order.push_back(url.to_string());
        // Least-recent goes first, but the handle is retired rather than dropped — see `retired`.
        while self.textures_order.len() > TEX_CACHE_CAP {
            if let Some(old) = self.textures_order.pop_front() {
                if let Some(t) = self.textures.remove(&old) {
                    self.retired.push(t);
                }
            }
        }
        Some(tex)
    }

    /// The covers allowed to hold an 8× texture: the playing track, [`HIRES_AHEAD`] ahead of it and
    /// [`HIRES_BACK`] behind, in priority order — current first, then forwards, then backwards.
    ///
    /// Priority is the whole trick. Everything in the window gets built eventually, but the GPU
    /// runs one pass at a time, and the cover on screen must never wait behind a cover for a track
    /// three skips ago.
    fn hires_window(&self) -> Vec<String> {
        let s = self.state.lock_ok();
        let n = s.queue.len();
        if n == 0 {
            // Nothing queued (a bare resume, a single track): the only cover that matters is the
            // one the now-playing pane is showing.
            return s
                .now
                .as_ref()
                .and_then(|t| t.art_big.clone())
                .into_iter()
                .collect();
        }
        let cur = s.qpos.min(n - 1);

        let mut idx = vec![cur];
        idx.extend((1..=HIRES_AHEAD).filter_map(|d| (cur + d < n).then_some(cur + d)));
        idx.extend((1..=HIRES_BACK).filter_map(|d| cur.checked_sub(d)));

        let mut out: Vec<String> = Vec::new();
        for i in idx {
            // An album's tracks share a cover, so the window is often fewer than six distinct URLs
            // — which is a straight VRAM saving, not a bug.
            if let Some(u) = s.queue[i].art_big.clone() {
                if !out.contains(&u) {
                    out.push(u);
                }
            }
        }
        out
    }

    /// Keep the resident 8× set equal to the window: land finished passes, evict what fell out,
    /// and start at most one new pass. Called once per frame.
    fn hires_sync(&mut self, ctx: &egui::Context) {
        // The window decides BOTH what we upload and what we evict, so it is computed once, up
        // front, and the two steps below agree by construction. They must: uploading a texture and
        // freeing it again within one frame is a crash, not a waste. egui hands the upload to the
        // queue as a pending write, but `free_texture` destroys the texture outright — and it runs
        // before the frame is submitted, so the submit trips over a write into destroyed memory:
        //
        //     Error in Queue::submit: Texture with 'egui_texid_Managed(3)' label has been destroyed
        //
        // A pass takes 10–20s. Skip a track while one is running and it lands on a cover that left
        // the window long ago — which is how a routine skip took the whole app down.
        let window = self.hires_window();

        // 1. Land whatever the workers finished. The upload is the UI thread's job — it owns the
        //    egui context — but the decode already happened off-thread, so this is a memcpy.
        while let Ok((url, image)) = self.hires_rx.try_recv() {
            if self.hires_inflight.as_deref() == Some(url.as_str()) {
                self.hires_inflight = None;
            }
            match image {
                // Out of the window already: drop the pixels on the floor rather than upload 256MB
                // we would evict two lines from now. If the cover comes back, so does the pass —
                // it is not recorded as a failure.
                Some(_) if !window.contains(&url) => {}
                Some(image) => {
                    let tex =
                        ctx.load_texture(format!("{url}#8x"), image, egui::TextureOptions::LINEAR);
                    self.hires.insert(url.clone(), tex);
                }
                None => {
                    self.hires_failed.insert(url.clone());
                }
            }
        }

        // 2. Evict. Dropping the handle frees the texture — this is what keeps a long listening
        //    session from growing without bound. Safe against the crash above because nothing
        //    uploaded this frame is outside `window`.
        self.hires.retain(|url, _| window.contains(url));

        // 3. Ask the backend to resolve + stream source art for window tracks that don't have it.
        //    Queued-but-unplayed tracks usually have no big cover URL yet.
        let want: Vec<usize> = {
            let s = self.state.lock_ok();
            let n = s.queue.len();
            let cur = s.qpos.min(n.saturating_sub(1));
            (cur.saturating_sub(HIRES_BACK)..=(cur + HIRES_AHEAD).min(n.saturating_sub(1)))
                .filter(|&i| {
                    s.queue.get(i).is_some_and(|t| {
                        !self.hires_asked.contains(&t.uri)
                            && t.art_big.as_ref().is_none_or(|u| !s.art.contains_key(u))
                    })
                })
                .collect()
        };
        if !want.is_empty() {
            let uris: Vec<String> = {
                let s = self.state.lock_ok();
                want.iter()
                    .filter_map(|&i| s.queue.get(i).map(|t| t.uri.clone()))
                    .collect()
            };
            self.hires_asked.extend(uris);
            self.send(Cmd::PrefetchBigArt(want));
        }

        // 4. Start one pass, highest priority first. The 4× half is cached on disk, so a cover
        //    coming back into the window (skipping backwards) only pays the 2× half.
        if self.hires_inflight.is_none() {
            for url in &window {
                if self.hires.contains_key(url) || self.hires_failed.contains(url) {
                    continue;
                }
                // Only upscale art the backend has confirmed is the best the CDN has. A resumed
                // session's queue still carries 640px URLs whose bytes are already on disk, and
                // upscaling one of those wastes a 12s pass on art we're about to replace.
                let bytes = {
                    let s = self.state.lock_ok();
                    if !s.art_best.contains(url) {
                        continue;
                    }
                    s.art.get(url).cloned()
                };
                let Some(bytes) = bytes else {
                    continue; // source art still in flight; try again next frame
                };

                let art_id = url.rsplit('/').next().unwrap_or("art").to_string();
                let url = url.clone();
                let (tx, ctx2) = (self.hires_tx.clone(), ctx.clone());
                self.hires_inflight = Some(url.clone());
                std::thread::spawn(move || {
                    // Decode off the UI thread and hand over the finished pixels: the 8000² PNG
                    // never crosses a thread boundary, and never lands in shared state.
                    //
                    // The send must happen even if the upscale panics — it is what clears
                    // `hires_inflight`, and a stuck inflight entry blocks every future pass. So a
                    // panic degrades to "this cover failed" rather than freezing the pipeline.
                    let image = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        upscale::upscale_image(&art_id, &bytes).map(|img| {
                            let size = [img.width() as usize, img.height() as usize];
                            egui::ColorImage::from_rgba_unmultiplied(size, img.as_raw())
                        })
                    }))
                    .unwrap_or_else(|_| {
                        tracing::warn!("upscale worker panicked for {art_id}");
                        None
                    });
                    let _ = tx.send((url, image));
                    ctx2.request_repaint();
                });
                break;
            }
        }
    }

    /// The full-screen cover: the 8× if it's resident, the original otherwise.
    ///
    /// Never blocks on the upscale — the view opens on the 640px cover and sharpens a moment later,
    /// which is why [`App::hires_sync`] builds the next tracks' covers before they're needed.
    fn art_hires(&mut self, ctx: &egui::Context, url: &str) -> Option<egui::TextureHandle> {
        if let Some(tex) = self.hires.get(url) {
            return Some(tex.clone());
        }
        self.art(ctx, url)
    }

    /// Shuffle, as a switch. Sends the toggle; the backend owns the order.
    fn shuffle_button(&self, ui: &mut egui::Ui, size: f32) {
        let on = self.state.lock_ok().shuffle;
        let hint = if on {
            "Shuffle is on — click to play in list order (S)"
        } else {
            "Shuffle the rest of the queue (S)"
        };
        if mode_button(ui, Icon::Shuffle, size, on, hint) {
            self.send(Cmd::SetShuffle(!on));
        }
    }

    /// Repeat, cycling off → all → one. One button, three states — the icon says which.
    fn repeat_button(&self, ui: &mut egui::Ui, size: f32) {
        let mode = self.state.lock_ok().repeat;
        let (icon, hint) = match mode {
            Repeat::Off => (Icon::Repeat, "Repeat off — click to repeat the queue (R)"),
            Repeat::All => (Icon::Repeat, "Repeating the queue — click to repeat this track (R)"),
            Repeat::One => (Icon::RepeatOne, "Repeating this track — click for no repeat (R)"),
        };
        if mode_button(ui, icon, size, mode != Repeat::Off, hint) {
            self.send(Cmd::CycleRepeat);
        }
    }

    fn art_at(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        url: Option<&String>,
        size: f32,
        rounding: f32,
    ) {
        match url.and_then(|u| self.art(ctx, u)) {
            Some(t) => {
                ui.add(
                    egui::Image::new(&t)
                        .fit_to_exact_size(Vec2::splat(size))
                        .rounding(Rounding::same(rounding)),
                );
            }
            None => {
                // Reserve the same box so nothing reflows when art lands.
                let (rect, _) = ui.allocate_exact_size(Vec2::splat(size), egui::Sense::hover());
                ui.painter()
                    .rect_filled(rect, Rounding::same(rounding), Color32::from_rgb(38, 35, 42));
            }
        }
    }
}

/// Nocturne's own polish on top of the shared Rune theme.
///
/// uikit gives us the palette and typography; this makes the *controls* modern and, crucially,
/// consistent — one rounding, one set of paddings, one hover treatment, everywhere. Mixed metrics
/// are what make a UI feel amateur even when every individual widget is fine.
fn polish(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    let w = &mut style.visuals.widgets;

    let round = Rounding::same(8.0);
    for s in [&mut w.inactive, &mut w.hovered, &mut w.active, &mut w.noninteractive, &mut w.open] {
        s.rounding = round;
        s.expansion = 0.0;
    }
    w.inactive.bg_fill = Color32::from_rgb(38, 34, 42);
    w.inactive.weak_bg_fill = Color32::from_rgb(38, 34, 42);
    w.inactive.bg_stroke = Stroke::new(1.0, Color32::from_rgb(56, 48, 44));

    w.hovered.bg_fill = Color32::from_rgb(56, 47, 40);
    w.hovered.weak_bg_fill = Color32::from_rgb(56, 47, 40);
    w.hovered.bg_stroke = Stroke::new(1.0, theme::ORANGE);

    w.active.bg_fill = theme::ORANGE;
    w.active.weak_bg_fill = theme::ORANGE;
    w.active.bg_stroke = Stroke::new(1.0, theme::ORANGE_HI);

    // One rhythm for spacing — buttons that all breathe the same amount.
    style.spacing.button_padding = egui::vec2(12.0, 7.0);
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.slider_rail_height = 5.0;
    style.spacing.interact_size.y = 26.0;

    ctx.set_style(style);
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        theme::apply(ctx);
        polish(ctx);
        // Between frames: the previous frame is submitted and the next hasn't queued anything, so
        // this is the one safe moment to actually free what was evicted while painting it.
        self.retired.clear();
        // Keep the resident 8× covers in step with the queue, wherever the user just skipped to.
        self.hires_sync(ctx);

        let (logged_in, busy, status, now, view, current) = {
            let s = self.state.lock_ok();
            (
                s.logged_in,
                s.busy,
                s.status.clone(),
                s.now.clone(),
                s.view.clone(),
                s.current_uri.clone(),
            )
        };

        // The vibe screen is the full-screen view, and full screen means CHROMELESS: the album art
        // goes edge to edge with no title strip cutting the top off. The bar becomes an overlay
        // that wipes in when you reach for the top edge, so close/min/max are still a flick away.
        // Everywhere else (browse, and the sign-in screen) keeps the normal bar, which has to be
        // built here — before any content — because it's a panel and panels claim layout first.
        let chromeless = logged_in && self.vibe;
        if !chromeless {
            chrome::title_bar(ctx, "Nocturne");
        }

        // The vibe view is "full screen" in the APP's language only — it fills Nocturne's window,
        // not your monitor. It used to also drive `ViewportCommand::Fullscreen`, which meant that
        // merely playing a track, hitting shuffle or clicking a cover blew the window up to take
        // over the screen. Deciding to go fullscreen is the window manager's business and yours;
        // it is not something a play button gets to do on your behalf.
        //
        // The cost is that rdock's bottom reveal strip is live again, under the transport row. That
        // is the right trade: a dock that peeks is a nuisance, a window that hijacks the display is
        // a fight.
        //
        // Say it out loud once, on the first frame: KWin remembers the fullscreen state an app last
        // asked for and restores it on the next launch, so merely *not asking* leaves a window that
        // an older Nocturne fullscreened still fullscreen, forever. This is what un-sticks it.
        if !self.unfullscreened {
            self.unfullscreened = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
        }
        // Retry silent login a few times with backoff before falling back to the sign-in screen.
        // Capped at 4 tries so a genuinely dead credential still surfaces the button promptly
        // instead of retrying forever behind a spinner.
        const AUTOLOGIN_TRIES: u32 = 4;
        if !logged_in && !busy && self.autologin_attempts < AUTOLOGIN_TRIES {
            if self.autologin_at.is_some_and(|at| std::time::Instant::now() >= at) {
                if nocturne_session::has_cached_login() {
                    self.autologin_attempts += 1;
                    let backoff = 2u64.pow(self.autologin_attempts); // 2s, 4s, 8s, 16s
                    self.autologin_at =
                        Some(std::time::Instant::now() + std::time::Duration::from_secs(backoff));
                    ctx.request_repaint_after(std::time::Duration::from_secs(backoff));
                    self.send(Cmd::Login);
                } else {
                    self.autologin_attempts = AUTOLOGIN_TRIES; // cold start — go straight to the button
                }
            } else {
                ctx.request_repaint_after(std::time::Duration::from_millis(500));
            }
        }
        if logged_in && !self.loaded {
            self.loaded = true;
            self.send(Cmd::LoadPlaylists);
            self.send(Cmd::LoadSaved);
            // Pick up where he left off — same queue, same track, paused at the same second.
            self.send(Cmd::Resume);
        }

        if !logged_in {
            self.sign_in(ctx, busy, &status);
            return;
        }

        // VIBE is the default and the home. The library slides in OVER it; picking something
        // takes you to the browse view; playing from there drops you straight back here.
        if self.vibe {
            if self.show_sidebar {
                self.sidebar(ctx);
            }
            self.vibe_view(ctx, now.clone());
            self.add_dialog(ctx);
            // After the content — an Area paints in call order, and this one floats on top of it.
            chrome::title_bar_overlay(ctx, "Nocturne");
            if now.is_some_and(|n| !n.paused) {
                ctx.request_repaint_after(std::time::Duration::from_millis(200));
            }
            return;
        }

        // BROWSE: a playlist/album. Only here does the list — and the up-next pane on the right —
        // appear at all.
        self.now_bar(ctx, now.clone());
        if self.show_sidebar {
            self.sidebar(ctx);
        }
        if self.show_nowpane {
            self.now_pane(ctx, now.clone());
        }
        self.track_list(ctx, &view, &status, busy, current.as_deref());

        self.add_dialog(ctx);

        if now.is_some_and(|n| !n.paused) {
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }
    }
}

impl App {
    fn sign_in(&mut self, ctx: &egui::Context, busy: bool, status: &str) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(200.0);
                ui.label(RichText::new("NOCTURNE").size(34.0).strong().color(theme::ORANGE));
                ui.add_space(4.0);
                ui.label(RichText::new("Spotify, native to Rune").weak());
                ui.add_space(24.0);
                if busy {
                    ui.spinner();
                } else if ui
                    .add_sized([200.0, 36.0], egui::Button::new("Sign in with Spotify"))
                    .clicked()
                {
                    self.send(Cmd::Login);
                }
                ui.add_space(10.0);
                ui.label(RichText::new(status).weak().small());
            });
        });
    }

    /// Playlists. No wordmark — the window is already titled, and the space belongs to content.
    fn sidebar(&mut self, ctx: &egui::Context) {
        let ctx2 = ctx.clone();
        egui::SidePanel::left("nav")
            .resizable(false)
            .exact_width(SIDEBAR_W)
            .frame(
                egui::Frame::none()
                    .fill(Color32::from_rgb(16, 15, 19))
                    .inner_margin(Margin::symmetric(10.0, 12.0)),
            )
            .show(ctx, |ui| {
                let view = self.state.lock_ok().view.clone();
                ui.horizontal(|ui| {
                    ui.label(RichText::new("LIBRARY").weak().small());
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if icons::button(ui, Icon::Close, 22.0, false)
                            .on_hover_text("Close (Esc)")
                            .clicked()
                        {
                            self.show_sidebar = false;
                        }
                    });
                });
                ui.add_space(8.0);

                // Clicking the ACTIVE item still navigates: from the vibe screen, "Liked Songs"
                // being highlighted must not mean it's un-clickable — that's how you get back to it.
                if nav_item(ui, "Liked Songs", view == "Liked Songs", Some(Icon::HeartFilled)) {
                    self.send(Cmd::LoadSaved);
                    self.vibe = false;
                    self.show_sidebar = false;
                }
                // The current radio, as a first-class list. It lives on this disk and survives
                // restarts; a new radio replaces it.
                let radio = self.state.lock_ok().radio_playlist.clone();
                if let Some(pl) = radio {
                    ui.add_space(12.0);
                    ui.label(RichText::new("RADIO").weak().small());
                    ui.add_space(4.0);
                    let active = view == pl.name;
                    if nav_item(ui, &pl.name, active, Some(Icon::Radio)) {
                        self.send(Cmd::ShowRadioPlaylist);
                        self.vibe = false;
                        self.show_sidebar = false;
                    }
                }

                ui.add_space(14.0);
                ui.label(RichText::new("PLAYLISTS").weak().small());
                ui.add_space(6.0);

                let playlists = self.state.lock_ok().playlists.clone();
                let mut clicked = None;
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for p in playlists {
                            let active = view == p.name;
                            let (rect, resp) = ui.allocate_exact_size(
                                Vec2::new(ui.available_width(), 32.0),
                                egui::Sense::click(),
                            );
                            if active {
                                ui.painter().rect_filled(rect, Rounding::same(6.0), theme::SEL_HL);
                            } else if resp.hovered() {
                                ui.painter().rect_filled(
                                    rect,
                                    Rounding::same(6.0),
                                    Color32::from_rgb(30, 28, 34),
                                );
                            }
                            let mut row = ui.child_ui(
                                rect.shrink2(Vec2::new(10.0, 0.0)),
                                Layout::left_to_right(Align::Center),
                                None,
                            );
                            let col = active.then_some(theme::ORANGE);
                            self.emoji.label(&mut row, &ctx2, &p.name, 13.5, col, false);
                            if resp.clicked() {
                                clicked = Some(p.id.clone());
                            }
                        }
                    });
                // Same for playlists: re-clicking the one you're already in reopens it.
                if let Some(id) = clicked {
                    self.send(Cmd::OpenPlaylist(id));
                    self.vibe = false;
                    self.show_sidebar = false;
                }
                if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    self.show_sidebar = false;
                }
            });
    }

    /// The Apple-Music-style now-playing pane: BIG album art, then Up Next and History.
    fn now_pane(&mut self, ctx: &egui::Context, now: Option<NowPlaying>) {
        let ctx2 = ctx.clone();
        egui::SidePanel::right("nowpane")
            .resizable(false)
            .exact_width(NOWPANE_W)
            .frame(
                egui::Frame::none()
                    .fill(Color32::from_rgb(16, 15, 19))
                    .inner_margin(Margin::symmetric(16.0, 14.0)),
            )
            .show(ctx, |ui| {
                let art_size = NOWPANE_W - 32.0;
                match &now {
                    Some(n) => {
                        let at = ui.next_widget_position();
                        // The BIG cover. This pane is ~290px wide; the 64px thumbnail rendered here
                        // looked exactly as bad as it sounds.
                        let big = n.art_big.clone().or_else(|| n.art_url.clone());
                        self.art_at(ui, &ctx2, big.as_ref(), art_size, 10.0);
                        let r = egui::Rect::from_min_size(at, Vec2::splat(art_size));
                        let hit = ui.interact(r, egui::Id::new("pane-art"), egui::Sense::click());
                        if hit.hovered() {
                            ui.painter()
                                .rect_filled(r, Rounding::same(10.0), Color32::from_black_alpha(70));
                            let ir = egui::Rect::from_center_size(r.center(), Vec2::splat(34.0));
                            icons::paint(
                                &ui.painter().clone(),
                                ir,
                                Icon::Fullscreen,
                                theme::TEXT,
                            );
                        }
                        if hit.clicked() {
                            self.vibe = true;
                        }
                        ui.add_space(12.0);
                        self.emoji.label(ui, &ctx2, &n.name, 17.0, None, true);
                        ui.label(RichText::new(&n.artists).weak());
                    }
                    None => {
                        self.art_at(ui, &ctx2, None, art_size, 10.0);
                        ui.add_space(12.0);
                        ui.label(RichText::new("Nothing playing").weak());
                    }
                }

                ui.add_space(16.0);
                let (queue, qpos) = {
                    let s = self.state.lock_ok();
                    (s.queue.clone(), s.qpos)
                };

                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        // --- Up next ---
                        ui.label(RichText::new("UP NEXT").weak().small());
                        ui.add_space(4.0);
                        let upcoming: Vec<(usize, _)> = queue
                            .iter()
                            .enumerate()
                            .skip(qpos + 1)
                            .take(20)
                            .map(|(i, t)| (i, t.clone()))
                            .collect();
                        if upcoming.is_empty() {
                            ui.label(RichText::new("end of queue").weak().small());
                        }
                        let mut jump = None;
                        for (i, t) in &upcoming {
                            if mini_row(ui, &ctx2, self, t, false) {
                                jump = Some(*i);
                            }
                        }

                        // --- Previously played ---
                        ui.add_space(14.0);
                        ui.label(RichText::new("PREVIOUSLY").weak().small());
                        ui.add_space(4.0);
                        let past: Vec<(usize, _)> = queue
                            .iter()
                            .enumerate()
                            .take(qpos)
                            .rev()
                            .take(20)
                            .map(|(i, t)| (i, t.clone()))
                            .collect();
                        if past.is_empty() {
                            ui.label(RichText::new("nothing yet").weak().small());
                        }
                        for (i, t) in &past {
                            if mini_row(ui, &ctx2, self, t, true) {
                                jump = Some(*i);
                            }
                        }
                        if let Some(i) = jump {
                            self.send(Cmd::JumpTo(i));
                        }
                    });
            });
    }

    fn track_list(
        &mut self,
        ctx: &egui::Context,
        view: &str,
        status: &str,
        busy: bool,
        current: Option<&str>,
    ) {
        let ctx2 = ctx.clone();
        egui::CentralPanel::default()
            .frame(egui::Frame::none().inner_margin(Margin::symmetric(16.0, 12.0)))
            .show(ctx, |ui| {
                // --- top row: panel toggles, search, radio switch ---
                ui.horizontal(|ui| {
                    if icons::button(ui, Icon::Fullscreen, 30.0, true)
                        .on_hover_text("Back to full screen (Esc)")
                        .clicked()
                        || ui.input(|i| i.key_pressed(egui::Key::Escape))
                    {
                        self.vibe = true;
                    }
                    if icons::button(ui, Icon::Menu, 30.0, true)
                        .on_hover_text("Library")
                        .clicked()
                    {
                        self.show_sidebar = !self.show_sidebar;
                    }
                    ui.add_space(4.0);
                    let field = ui.add(
                        egui::TextEdit::singleline(&mut self.query)
                            .hint_text("Search Spotify…")
                            .desired_width(280.0)
                            .margin(Margin::symmetric(10.0, 7.0)),
                    );
                    let enter = field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if enter || ui.button("Search").clicked() {
                        self.send(Cmd::Search(self.query.clone()));
                    }

                    // Keep a result set. Only offered while looking at one — the button is
                    // meaningless on Liked Songs, which is already a list Spotify holds.
                    let saveable = {
                        let s = self.state.lock_ok();
                        view.starts_with("Search: ") && !s.tracks.is_empty()
                    };
                    if saveable
                        && ui
                            .button("Save as playlist")
                            .on_hover_text("Create a Spotify playlist from these results")
                            .clicked()
                    {
                        let (name, tracks) = {
                            let s = self.state.lock_ok();
                            (
                                view.trim_start_matches("Search: ").to_string(),
                                s.tracks.clone(),
                            )
                        };
                        self.send(Cmd::SaveTracksToSpotify(name, tracks));
                    }

                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if icons::button(ui, Icon::Radio, 30.0, true)
                            .on_hover_text("Show/hide up-next")
                            .clicked()
                        {
                            self.show_nowpane = !self.show_nowpane;
                        }
                        ui.add_space(8.0);

                        let (mut autoplay, radio_loading, analyzing, feats) = {
                            let s = self.state.lock_ok();
                            (s.autoplay, s.radio_loading, s.analyzing, s.taste_features)
                        };
                        if toggle(ui, &mut autoplay, "Radio").on_hover_text(format!(
                            "When the queue runs out, keep playing — picked from your listening, \
                             using {feats} analyzed tracks"
                        )).clicked()
                        {
                            self.send(Cmd::SetAutoplay(autoplay));
                        }
                        ui.add_space(8.0);
                        if busy || radio_loading || analyzing {
                            ui.spinner();
                        }
                        ui.label(RichText::new(status).weak().small());
                    });
                });

                // --- mood radio ---
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    let m = ui.add(
                        egui::TextEdit::singleline(&mut self.mood)
                            .hint_text("Describe a vibe — “chill winter lofi”…")
                            .desired_width(260.0)
                            .margin(Margin::symmetric(10.0, 7.0)),
                    );
                    let go = m.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if (go || labeled_button(ui, Icon::Radio, "Radio")) && !self.mood.trim().is_empty() {
                        self.send(Cmd::MoodRadio(self.mood.clone()));
                        // Land in the new playlist so you can see what it built.
                        self.vibe = false;
                    }
                    ui.add_space(4.0);
                    // Each mood gets an accent dot in a colour that means something: warm amber for
                    // cozy, hot orange for hype, cold blue for sad, indigo for late night, magenta
                    // for party. No emoji — a coloured dot reads at any size and is ours.
                    for (label, phrase, accent) in [
                        ("Cozy lofi", "chill autumn lofi cozy", Color32::from_rgb(214, 138, 62)),
                        ("Hype", "hype energetic workout", Color32::from_rgb(233, 90, 44)),
                        ("Melancholy", "sad melancholy acoustic", Color32::from_rgb(88, 124, 176)),
                        ("Late night", "dark moody night chill", Color32::from_rgb(122, 96, 176)),
                        ("Party", "happy dance party", Color32::from_rgb(196, 78, 130)),
                    ] {
                        if chip(ui, label, accent) {
                            self.mood = phrase.to_string();
                            self.send(Cmd::MoodRadio(phrase.to_string()));
                            self.vibe = false;
                        }
                    }
                });

                ui.add_space(14.0);
                let tracks = self.state.lock_ok().tracks.clone();
                ui.horizontal(|ui| {
                    self.emoji.label(ui, &ctx2, view, 24.0, None, true);
                    ui.add_space(8.0);
                    ui.label(RichText::new(format!("{} tracks", tracks.len())).weak().small());
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if !tracks.is_empty() && labeled_button(ui, Icon::Play, "Play all") {
                            self.send(Cmd::PlayQueue(tracks.clone()));
                            self.vibe = true;
                        }
                        // The "don't start me at track 1" button. Turns shuffle on FIRST, so the
                        // queue this builds is already scrambled — the commands are handled in the
                        // order they're sent.
                        if !tracks.is_empty() {
                            ui.add_space(8.0);
                            if labeled_button(ui, Icon::Shuffle, "Shuffle") {
                                self.send(Cmd::SetShuffle(true));
                                self.send(Cmd::PlayQueue(tracks.clone()));
                                self.vibe = true;
                            }
                        }

                        // Viewing the temp radio? Offer to make it permanent.
                        let radio = self.state.lock_ok().radio_playlist.clone();
                        if let Some(pl) = radio {
                            if pl.name == view {
                                ui.add_space(8.0);
                                match &pl.spotify_id {
                                    Some(_) => {
                                        ui.label(
                                            RichText::new("on Spotify")
                                                .weak()
                                                .small(),
                                        );
                                    }
                                    None => {
                                        if labeled_button(ui, Icon::Plus, "Save to Spotify") {
                                            self.send(Cmd::SaveRadioToSpotify);
                                        }
                                    }
                                }
                            }
                        }
                    });
                });
                ui.add_space(8.0);

                // --- rows ---
                let (liked, playlists) = {
                    let s = self.state.lock_ok();
                    (s.liked.clone(), s.playlists.clone())
                };
                let mut play = None;
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for (i, t) in tracks.iter().enumerate() {
                            let is_current = current == Some(t.uri.as_str());
                            let (rect, resp) = ui.allocate_exact_size(
                                Vec2::new(ui.available_width(), ROW_H),
                                egui::Sense::click(),
                            );
                            if resp.hovered() {
                                ui.painter().rect_filled(
                                    rect,
                                    Rounding::same(6.0),
                                    Color32::from_rgb(30, 28, 34),
                                );
                            }
                            if is_current {
                                ui.painter().rect_filled(rect, Rounding::same(6.0), theme::SEL_HL);
                            }

                            let mut row = ui.child_ui(
                                rect.shrink2(Vec2::new(10.0, 8.0)),
                                Layout::left_to_right(Align::Center),
                                None,
                            );
                            row.allocate_ui_with_layout(
                                Vec2::new(22.0, ART),
                                Layout::centered_and_justified(egui::Direction::TopDown),
                                |ui| {
                                    if is_current {
                                        let (mr, _) = ui.allocate_exact_size(
                                            Vec2::splat(14.0),
                                            egui::Sense::hover(),
                                        );
                                        icons::paint(
                                            &ui.painter().clone(),
                                            mr,
                                            Icon::Play,
                                            theme::ORANGE,
                                        );
                                    } else {
                                        ui.label(RichText::new(format!("{}", i + 1)).weak().small());
                                    }
                                },
                            );
                            row.add_space(6.0);
                            self.art_at(&mut row, &ctx2, t.art_url.as_ref(), ART, 4.0);
                            row.add_space(12.0);

                            let em = &mut self.emoji;
                            row.vertical(|ui| {
                                ui.spacing_mut().item_spacing.y = 1.0;
                                em.label(ui, &ctx2, &t.name, 14.0, is_current.then_some(theme::ORANGE), true);
                                ui.label(RichText::new(&t.artists).weak().small());
                            });

                            row.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                ui.label(RichText::new(fmt_duration(t.duration_ms)).weak().small());
                                ui.add_space(10.0);
                                // Like + add-to-playlist, on every row.
                                add_controls(ui, self, t, liked.contains(&t.uri), &playlists, 15.0);
                                ui.add_space(10.0);
                                if ui.available_width() > 200.0 {
                                    ui.add_sized(
                                        [ui.available_width().min(240.0), ART],
                                        egui::Label::new(RichText::new(&t.album).weak().small())
                                            .truncate(),
                                    );
                                }
                            });

                            if resp.clicked() {
                                play = Some(t.uri.clone());
                            }
                        }
                    });
                if let Some(uri) = play {
                    self.send(Cmd::Play(uri));
                    // Chose a track — back to the vibe.
                    self.vibe = true;
                }

            });
    }

    /// The transport. Controls and scrubber are CENTERED in the window; the track sits left and
    /// volume right, both fixed-width, so the centre stays centred as the window resizes.
    fn now_bar(&mut self, ctx: &egui::Context, now: Option<NowPlaying>) {
        let ctx2 = ctx.clone();
        egui::TopBottomPanel::bottom("now")
            .exact_height(BAR_H)
            .frame(
                egui::Frame::none()
                    .fill(Color32::from_rgb(18, 16, 21))
                    .inner_margin(Margin::symmetric(16.0, 8.0)),
            )
            .show(ctx, |ui| {
                let full = ui.max_rect();
                // Accent hairline along the top edge.
                ui.painter().rect_filled(
                    egui::Rect::from_min_max(
                        egui::pos2(full.min.x - 16.0, full.min.y - 8.0),
                        egui::pos2(full.max.x + 16.0, full.min.y - 6.0),
                    ),
                    Rounding::ZERO,
                    Color32::from_rgb(52, 38, 30),
                );

                const SIDE: f32 = 260.0;
                let n = now.clone();

                // -- left: current track --
                let left = egui::Rect::from_min_size(full.min, Vec2::new(SIDE, full.height()));
                let mut lui = ui.child_ui(left, Layout::left_to_right(Align::Center), None);
                if let Some(n) = &n {
                    let art_start = lui.next_widget_position();
                    self.art_at(&mut lui, &ctx2, n.art_url.as_ref(), 64.0, 6.0);
                    // The cover is the door to the full-screen view.
                    let art_rect = egui::Rect::from_min_size(art_start, Vec2::splat(64.0));
                    let hit = ui.interact(art_rect, egui::Id::new("bar-art"), egui::Sense::click());
                    if hit.hovered() {
                        ui.painter().rect_filled(
                            art_rect,
                            Rounding::same(6.0),
                            Color32::from_black_alpha(90),
                        );
                        let ir = egui::Rect::from_center_size(art_rect.center(), Vec2::splat(22.0));
                        icons::paint(&ui.painter().clone(), ir, Icon::Fullscreen, theme::TEXT);
                    }
                    if hit.clicked() {
                        self.vibe = true;
                    }
                    lui.add_space(10.0);
                    let em = &mut self.emoji;
                    lui.vertical(|ui| {
                        ui.spacing_mut().item_spacing.y = 2.0;
                        ui.add_space(2.0);
                        em.label(ui, &ctx2, &n.name, 14.0, None, true);
                        ui.add_sized(
                            [150.0, 15.0],
                            egui::Label::new(RichText::new(&n.artists).weak().small()).truncate(),
                        );
                    });
                    let (liked, playlists, track) = {
                        let s = self.state.lock_ok();
                        (
                            s.current_uri.as_ref().is_some_and(|u| s.liked.contains(u)),
                            s.playlists.clone(),
                            s.queue.get(s.qpos).cloned(),
                        )
                    };
                    if let Some(track) = track {
                        lui.add_space(6.0);
                        add_controls(&mut lui, self, &track, liked, &playlists, 16.0);
                    }
                }

                // -- right: volume. Icon and bar share one horizontal row, so they're centred on
                //    the same baseline; the bar uses trailing_fill so it fills as it rises. --
                let right = egui::Rect::from_min_size(
                    egui::pos2(full.max.x - SIDE, full.min.y),
                    Vec2::new(SIDE, full.height()),
                );
                let mut rui = ui.child_ui(right, Layout::right_to_left(Align::Center), None);
                let vol = rui.add_sized(
                    [150.0, 18.0],
                    egui::Slider::new(&mut self.volume, 0.0..=1.0)
                        .show_value(false)
                        .trailing_fill(true),
                );
                if vol.changed() {
                    self.send(Cmd::Volume(self.volume));
                }
                rui.add_space(4.0);
                let vi = if self.volume < 0.01 {
                    Icon::VolumeMute
                } else if self.volume < 0.5 {
                    Icon::VolumeLow
                } else {
                    Icon::Volume
                };
                // Same row as the bar, so icon and rail share a centre line.
                let (vr, _) = rui.allocate_exact_size(Vec2::splat(20.0), egui::Sense::hover());
                icons::paint(&rui.painter().clone(), vr, vi, theme::TEXT);

                // -- centre: transport + scrubber --
                let Some(n) = n else { return };
                let cw = 480.0f32.min(full.width() - 2.0 * SIDE - 20.0).max(240.0);
                let centre = egui::Rect::from_center_size(
                    egui::pos2(full.center().x, full.center().y),
                    Vec2::new(cw, full.height()),
                );
                let cui = ui.child_ui(centre, Layout::top_down(Align::Center), None);

                // Same layout rules as the full-screen view: one button baseline, an explicitly
                // sized slider (add_sized does NOT stretch a Slider — spacing.slider_width does),
                // fixed-width times, and the right-hand one counting down.
                let _ = cui;
                const BMODE: f32 = 28.0;
                const BSIDE: f32 = 32.0;
                const BPLAY: f32 = 40.0;
                const BGAP: f32 = 10.0;
                let brow = egui::Rect::from_center_size(
                    egui::pos2(centre.center().x, centre.min.y + 26.0),
                    Vec2::new(BMODE * 2.0 + BSIDE * 2.0 + BPLAY + BGAP * 4.0, BPLAY),
                );
                let mut bui = ui.child_ui(brow, Layout::left_to_right(Align::Center), None);
                bui.spacing_mut().item_spacing.x = BGAP;
                self.shuffle_button(&mut bui, BMODE);
                if icons::button(&mut bui, Icon::Prev, BSIDE, true).clicked() {
                    self.send(Cmd::Prev);
                }
                let pp = if n.paused { Icon::Play } else { Icon::Pause };
                if icons::button(&mut bui, pp, BPLAY, true).clicked() {
                    self.send(Cmd::PlayPause);
                }
                if icons::button(&mut bui, Icon::Next, BSIDE, true).clicked() {
                    self.send(Cmd::Next);
                }
                self.repeat_button(&mut bui, BMODE);

                let elapsed = n.elapsed_ms();
                let remaining = n.duration_ms.saturating_sub(elapsed);
                const TW: f32 = 44.0;
                let srect = egui::Rect::from_center_size(
                    egui::pos2(centre.center().x, centre.min.y + 60.0),
                    Vec2::new(cw, 18.0),
                );
                let mut sui = ui.child_ui(srect, Layout::left_to_right(Align::Center), None);
                sui.spacing_mut().item_spacing.x = 8.0;
                sui.spacing_mut().slider_width = cw - 2.0 * (TW + 8.0);
                sui.add_sized(
                    [TW, 16.0],
                    egui::Label::new(RichText::new(fmt_duration(elapsed)).weak().small()),
                );
                let mut pos = elapsed as f32 / n.duration_ms.max(1) as f32;
                let bar = sui.add(
                    egui::Slider::new(&mut pos, 0.0..=1.0)
                        .show_value(false)
                        .trailing_fill(true),
                );
                if bar.drag_stopped() {
                    self.send(Cmd::Seek((pos * n.duration_ms as f32) as u32));
                }
                sui.add_sized(
                    [TW, 16.0],
                    egui::Label::new(
                        RichText::new(format!("-{}", fmt_duration(remaining))).weak().small(),
                    ),
                );
            });
    }
}

/// Detect a solid matte border (a uniform frame — white label edges, letterboxing, a plain
/// bottom strip) around a cover and return the inner rectangle `(x, y, w, h)` to keep, or `None`
/// if there's no border worth trimming.
///
/// Each side is judged against *its own* edge colour, sampled at the middle of that edge — a
/// one-sided white strip along the bottom has to match the bottom, not the muddy average of all
/// four corners. A row/column counts as border while nearly all of its pixels sit within
/// tolerance of that side's colour. Every side scans inward independently, capped at 30% so a
/// busy edge can't eat the image, and the whole thing bails unless a real border was found.
fn trim_matte(img: &image::RgbaImage) -> Option<(u32, u32, u32, u32)> {
    let (w, h) = img.dimensions();
    if w < 16 || h < 16 {
        return None;
    }
    let at = |x: u32, y: u32| img.get_pixel(x, y);
    // ~14/channel of slack, summed across the three channels so a coloured matte trims as
    // readily as white.
    let close = |p: &image::Rgba<u8>, r: &image::Rgba<u8>| {
        (0..3).map(|c| (p[c] as f32 - r[c] as f32).abs()).sum::<f32>() <= 42.0
    };

    let cap_x = w * 3 / 10;
    let cap_y = h * 3 / 10;
    // A line is border when at least 92% of its pixels match that side's sampled colour.
    let top_c = *at(w / 2, 0);
    let bot_c = *at(w / 2, h - 1);
    let left_c = *at(0, h / 2);
    let right_c = *at(w - 1, h / 2);
    let row_border = |y: u32, r: &image::Rgba<u8>| {
        (0..w).filter(|&x| close(at(x, y), r)).count() as u32 * 100 >= w * 92
    };
    let col_border = |x: u32, r: &image::Rgba<u8>| {
        (0..h).filter(|&y| close(at(x, y), r)).count() as u32 * 100 >= h * 92
    };

    // Scan inward from an edge, tolerating a short run of non-matte lines — a title or logo
    // printed on the frame (like "ANDY LEECH" across the bottom matte) shouldn't halt the trim
    // before the real art begins. We keep going until either a sustained run of non-matte lines
    // (the photo itself) or the cap, and trim to the deepest matte line we saw.
    let scan = |cap: u32, is_border: &dyn Fn(u32) -> bool| -> u32 {
        let tol = (cap / 3).max(3); // an interruption longer than this is content, not a logo
        let (mut depth, mut gap, mut d) = (0u32, 0u32, 0u32);
        while d < cap {
            if is_border(d) {
                depth = d + 1;
                gap = 0;
            } else {
                gap += 1;
                if gap > tol {
                    break;
                }
            }
            d += 1;
        }
        depth
    };

    let top = scan(cap_y, &|d| row_border(d, &top_c));
    let bottom = scan(cap_y, &|d| row_border(h - 1 - d, &bot_c));
    let left = scan(cap_x, &|d| col_border(d, &left_c));
    let right = scan(cap_x, &|d| col_border(w - 1 - d, &right_c));

    // A single stray matching edge line isn't a frame; require a few pixels of real border.
    if top + bottom + left + right < 3 {
        return None;
    }
    Some((left, top, w - left - right, h - top - bottom))
}

impl App {
    /// A big, soft, blurred version of the cover — the backdrop. Built once per cover.
    ///
    /// Blur is done by downscaling hard and letting the GPU's bilinear filter scale it back up:
    /// a 24px image stretched across 1300px IS a blur, for free, with no convolution pass.
    ///
    /// Returns the texture and its mean luminance, because a fixed scrim can't work for every
    /// cover: a night scene averages ~8% luminance, and 130/255 of black on top of that is a
    /// backdrop you can't see. So the downscaled image is *exposed* to a target mean first —
    /// gained up if it's dark, left alone if it's already bright — and the caller scales its
    /// scrim by what came out.
    fn backdrop(&mut self, ctx: &egui::Context, url: &str) -> Option<(egui::TextureHandle, f32)> {
        if let Some(t) = self.blurred.get(url) {
            let t = t.clone();
            lru_touch(&mut self.blurred_order, url);
            return Some(t);
        }
        let bytes = self.state.lock_ok().art.get(url).cloned()?;
        let img = image::load_from_memory(&bytes).ok()?;
        // Some covers ship inside a solid matte frame — a white label border, letterboxing —
        // and a 12px average would carry that frame's colour across the whole backdrop. Trim
        // any near-uniform border first so the backdrop is built from the art, not the frame.
        let full = img.to_rgba8();
        let crop = trim_matte(&full);
        let source = match crop {
            Some((x, y, w, h)) => image::imageops::crop_imm(&full, x, y, w, h).to_image(),
            None => full,
        };
        // 12px, not 24: stretched across ~1300px that's a much softer blur, which is what stops
        // the backdrop competing with the cover in front of it.
        let mut small = image::imageops::resize(
            &source,
            12,
            12,
            image::imageops::FilterType::Triangle,
        );

        let lum = |p: &image::Rgba<u8>| {
            (0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32) / 255.0
        };
        let mean = |px: &image::RgbaImage| {
            px.pixels().map(lum).sum::<f32>() / (px.width() * px.height()) as f32
        };

        // Auto-exposure. Capped at 2.4x so a nearly-black cover lifts into visibility without
        // turning into grey soup, and never darkens a bright one (gain >= 1.0) — the complaint
        // was only ever that dark art vanishes.
        const TARGET: f32 = 0.30;
        let before = mean(&small).max(0.01);
        let gain = (TARGET / before).clamp(1.0, 2.4);
        if gain > 1.0 {
            for p in small.pixels_mut() {
                for c in 0..3 {
                    p[c] = (p[c] as f32 * gain).min(255.0) as u8;
                }
            }
        }
        let exposed = mean(&small);

        let color = egui::ColorImage::from_rgba_unmultiplied([12, 12], small.as_raw());
        let tex = ctx.load_texture(format!("blur-{url}"), color, egui::TextureOptions::LINEAR);
        self.blurred.insert(url.to_string(), (tex.clone(), exposed));
        self.blurred_order.push_back(url.to_string());
        while self.blurred_order.len() > TEX_CACHE_CAP {
            if let Some(old) = self.blurred_order.pop_front() {
                if let Some((t, _)) = self.blurred.remove(&old) {
                    self.retired.push(t);
                }
            }
        }
        Some((tex, exposed))
    }

    /// Full-screen now playing. The art is the interface.
    fn vibe_view(&mut self, ctx: &egui::Context, now: Option<NowPlaying>) {
        let ctx2 = ctx.clone();
        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(Color32::from_rgb(10, 9, 12)))
            .show(ctx, |ui| {
                let full = ui.max_rect();

                // 1. Backdrop: the cover itself, blurred and dimmed, bleeding to the edges.
                if let Some(url) = now
                    .as_ref()
                    .and_then(|n| n.art_big.clone().or_else(|| n.art_url.clone()))
                {
                    if let Some((bg, lum)) = self.backdrop(&ctx2, &url) {
                        egui::Image::new(&bg).paint_at(ui, full);
                        // The scrim exists to keep the text and controls readable — so it should
                        // be as heavy as THIS backdrop needs and no heavier. A bright cover still
                        // gets the full 130; a dark one, which was already readable and just came
                        // out muddy, gets a fraction of it.
                        let k = ((lum - 0.15) / 0.40).clamp(0.0, 1.0);
                        let scrim = (48.0 + k * (130.0 - 48.0)) as u8;
                        let foot = (100.0 + k * (165.0 - 100.0)) as u8;
                        ui.painter()
                            .rect_filled(full, Rounding::ZERO, Color32::from_black_alpha(scrim));
                        // …then a SMOOTH top-to-bottom darkening. This used to be a rect covering
                        // the bottom half, which drew a hard horizontal seam straight across the
                        // middle of the window. A gradient has no edge to see. The foot stays
                        // dark-ish on every cover, because the transport row sits in it.
                        vertical_gradient(
                            ui,
                            full,
                            Color32::from_black_alpha(0),
                            Color32::from_black_alpha(foot),
                        );
                    }
                }

                // Click anywhere (outside the controls) to leave.
                let bg_click = ui.interact(full, egui::Id::new("vibe-bg"), egui::Sense::click());

                let Some(n) = now else {
                    ui.centered_and_justified(|ui| {
                        ui.label(RichText::new("Nothing playing").weak().size(18.0));
                    });
                    if bg_click.clicked() {
                        self.vibe = false;
                    }
                    return;
                };

                // 2. The art, as big as the window allows.
                let art_size = (full.height() - 250.0).min(full.width() - 120.0).max(180.0);
                let art_rect = egui::Rect::from_center_size(
                    egui::pos2(full.center().x, full.min.y + 40.0 + art_size / 2.0),
                    Vec2::splat(art_size),
                );
                // The cover sits on a plate: a mat of even width on all four sides, with the same
                // corner geometry as the art, and a soft shadow cast beneath the whole thing.
                //
                // The plate's radius is the art's plus its own width, because two rounded rects
                // only look concentric if their radii differ by exactly the gap between them —
                // share a radius and the corners visibly drift apart.
                let plate_rect = art_rect.expand(PLATE);
                let plate_rounding = Rounding::same(ART_ROUNDING + PLATE);

                // Shadow: same shape as the plate, blurred rather than merely offset. The old
                // version was a hard rect shoved 14px down, which read as a square that had
                // slipped out from behind the art instead of a shadow under it.
                ui.painter().add(egui::epaint::RectShape {
                    rect: plate_rect.translate(Vec2::new(0.0, 10.0)),
                    rounding: plate_rounding,
                    fill: Color32::from_black_alpha(110),
                    stroke: Stroke::NONE,
                    blur_width: 28.0,
                    fill_texture_id: egui::TextureId::default(),
                    uv: egui::Rect::ZERO,
                });
                ui.painter()
                    .rect_filled(plate_rect, plate_rounding, PLATE_FILL);

                // The big cover: Spotify's original master (1800–2000px), upscaled 4× on the GPU
                // when the pass has landed. Rounded to match the plate — painting it as a hard
                // square inside a rounded plate was half the misalignment.
                let big = n.art_big.clone().or_else(|| n.art_url.clone());
                match big.as_ref().and_then(|u| self.art_hires(&ctx2, u)) {
                    Some(tex) => {
                        ui.painter().add(egui::epaint::RectShape {
                            rect: art_rect,
                            rounding: Rounding::same(ART_ROUNDING),
                            fill: Color32::WHITE, // multiplied with the texture: leaves it as-is
                            stroke: Stroke::NONE,
                            blur_width: 0.0,
                            fill_texture_id: tex.id(),
                            uv: egui::Rect::from_min_max(
                                egui::pos2(0.0, 0.0),
                                egui::pos2(1.0, 1.0),
                            ),
                        });
                    }
                    None => {
                        let mut aui =
                            ui.child_ui(art_rect, Layout::top_down(Align::Center), None);
                        self.art_at(&mut aui, &ctx2, big.as_ref(), art_size, ART_ROUNDING);
                    }
                }

                // 3. Title + artist, centred under the art.
                let text_y = art_rect.max.y + 22.0;
                let text_rect = egui::Rect::from_min_max(
                    egui::pos2(full.min.x, text_y),
                    egui::pos2(full.max.x, text_y + 60.0),
                );
                let mut tui = ui.child_ui(text_rect, Layout::top_down(Align::Center), None);
                tui.label(RichText::new(&n.name).size(26.0).strong().color(theme::TEXT));
                tui.label(RichText::new(&n.artists).size(15.0).color(Color32::from_gray(170)));

                // Like / add-to-playlist, right where the track is.
                //
                // Everything is read under ONE lock. `if let Some(x) = self.state.lock()...` keeps
                // the temporary guard alive for the whole `if let` body, so locking again inside it
                // deadlocks against itself — std::sync::Mutex is not reentrant. That froze the
                // entire app (and MPRIS with it).
                let (liked, playlists, track) = {
                    let s = self.state.lock_ok();
                    let uri = s.current_uri.clone();
                    (
                        uri.as_ref().is_some_and(|u| s.liked.contains(u)),
                        s.playlists.clone(),
                        s.queue.get(s.qpos).cloned(),
                    )
                };
                {
                    if let Some(track) = track {
                        tui.add_space(8.0);
                        tui.horizontal(|ui| {
                            let w = ui.available_width();
                            ui.add_space((w - 70.0).max(0.0) / 2.0);
                            add_controls(ui, self, &track, liked, &playlists, 20.0);
                        });
                    }
                }

                // 4. Scrubber + transport, centred at the bottom.
                let cw = 560.0f32.min(full.width() - 80.0);
                let ctrl = egui::Rect::from_center_size(
                    egui::pos2(full.center().x, full.max.y - 56.0),
                    Vec2::new(cw, 100.0),
                );
                let cui = ui.child_ui(ctrl, Layout::top_down(Align::Center), None);
                let _ = cui;

                // Elapsed on the left counting UP, remaining on the right counting DOWN, with the
                // bar exactly between them. Widths are fixed and the row is laid out in an explicit
                // centred rect, so the bar cannot drift as the timestamps change width.
                let elapsed = n.elapsed_ms();
                let remaining = n.duration_ms.saturating_sub(elapsed);
                const T_W: f32 = 46.0;
                const GAP: f32 = 10.0;
                let bar_w = cw - 2.0 * (T_W + GAP);

                // Rows are placed at EXPLICIT y offsets inside `ctrl`. Deriving both from
                // `next_widget_position()` put them at the same y and the buttons landed on top of
                // the scrubber.
                let srect = egui::Rect::from_center_size(
                    egui::pos2(ctrl.center().x, ctrl.min.y + 14.0),
                    Vec2::new(cw, 20.0),
                );
                let mut sui = ui.child_ui(srect, Layout::left_to_right(Align::Center), None);
                sui.spacing_mut().item_spacing.x = GAP;
                // A Slider takes its width from spacing.slider_width — `add_sized` does NOT stretch
                // it, which is why the bar came out stubby and off-centre.
                sui.spacing_mut().slider_width = bar_w;

                sui.add_sized(
                    [T_W, 16.0],
                    egui::Label::new(RichText::new(fmt_duration(elapsed)).weak().small()),
                );
                let mut pos = elapsed as f32 / n.duration_ms.max(1) as f32;
                let bar = sui.add_sized(
                    [bar_w, 16.0],
                    egui::Slider::new(&mut pos, 0.0..=1.0)
                        .show_value(false)
                        .trailing_fill(true),
                );
                if bar.drag_stopped() {
                    self.send(Cmd::Seek((pos * n.duration_ms as f32) as u32));
                }
                sui.add_sized(
                    [T_W, 16.0],
                    egui::Label::new(
                        RichText::new(format!("-{}", fmt_duration(remaining))).weak().small(),
                    ),
                );

                // All three buttons share ONE height, so ⏮ and ⏭ sit on the play button's centre
                // line instead of riding high.
                // The row must be as tall as its TALLEST child, or a bigger play button cannot be
                // vertically centred in it — it overflows and everything sits off the centre line.
                // Shuffle and repeat flank the transport, unframed: they're modes, not actions, so
                // they read as switches sitting either side of the buttons you press.
                const MODE_BTN: f32 = 34.0;
                const SIDE_BTN: f32 = 42.0;
                const PLAY_BTN: f32 = 52.0;
                const BTN_GAP: f32 = 12.0;
                let row_w = MODE_BTN * 2.0 + SIDE_BTN * 2.0 + PLAY_BTN + BTN_GAP * 4.0;
                let row = egui::Rect::from_center_size(
                    egui::pos2(ctrl.center().x, ctrl.min.y + 62.0),
                    Vec2::new(row_w, PLAY_BTN),
                );
                let mut bui = ui.child_ui(row, Layout::left_to_right(Align::Center), None);
                bui.spacing_mut().item_spacing.x = BTN_GAP;
                self.shuffle_button(&mut bui, MODE_BTN);
                if icons::button(&mut bui, Icon::Prev, SIDE_BTN, true).clicked() {
                    self.send(Cmd::Prev);
                }
                let pp = if n.paused { Icon::Play } else { Icon::Pause };
                if icons::button(&mut bui, pp, PLAY_BTN, true).clicked() {
                    self.send(Cmd::PlayPause);
                }
                if icons::button(&mut bui, Icon::Next, SIDE_BTN, true).clicked() {
                    self.send(Cmd::Next);
                }
                self.repeat_button(&mut bui, MODE_BTN);

                // 5. The library tab. This is the ONLY way out — pop it, pick something, and the
                //    app drops you straight back here.
                let tab = egui::Rect::from_min_size(
                    egui::pos2(full.min.x + 10.0, full.min.y + 10.0),
                    Vec2::new(34.0 + 8.0 + 34.0, 34.0),
                );
                let mut tui2 = ui.child_ui(tab, Layout::left_to_right(Align::Center), None);
                tui2.spacing_mut().item_spacing.x = 8.0;
                if icons::button(&mut tui2, Icon::Menu, 34.0, true)
                    .on_hover_text("Library (L)")
                    .clicked()
                    || ui.input(|i| i.key_pressed(egui::Key::L))
                {
                    self.show_sidebar = true;
                }

                // Straight to the list that is CURRENTLY PLAYING — not "Liked Songs", and not
                // whatever you last browsed. Only shown when something is actually queued.
                let playing_list = {
                    let s = self.state.lock_ok();
                    (!s.queue.is_empty()).then(|| s.queue_view.clone())
                };
                if let Some(name) = playing_list {
                    let hint = if name.is_empty() {
                        "Back to what's playing".to_string()
                    } else {
                        format!("Back to “{name}”")
                    };
                    if icons::button(&mut tui2, Icon::ChevronLeft, 34.0, true)
                        .on_hover_text(hint)
                        .clicked()
                    {
                        self.send(Cmd::ShowPlayingList);
                        self.vibe = false;
                    }
                }
                let _ = bg_click;

                // Space = play/pause, the one shortcut every player has. S and R sit next to it: on
                // the full-screen view there is no text field to steal them.
                if ui.input(|i| i.key_pressed(egui::Key::Space)) {
                    self.send(Cmd::PlayPause);
                }
                if ui.input(|i| i.key_pressed(egui::Key::S)) {
                    let on = self.state.lock_ok().shuffle;
                    self.send(Cmd::SetShuffle(!on));
                }
                if ui.input(|i| i.key_pressed(egui::Key::R)) {
                    self.send(Cmd::CycleRepeat);
                }
            });
    }
}

/// Paint a smooth vertical gradient. egui has no gradient primitive, so this is a two-triangle
/// mesh with per-vertex colours — the GPU interpolates, and there is no visible band or seam.
fn vertical_gradient(ui: &egui::Ui, rect: egui::Rect, top: Color32, bottom: Color32) {
    use egui::epaint::{Mesh, Vertex};
    let mut mesh = Mesh::default();
    let uv = egui::pos2(0.0, 0.0);
    mesh.vertices.push(Vertex { pos: rect.left_top(), uv, color: top });
    mesh.vertices.push(Vertex { pos: rect.right_top(), uv, color: top });
    mesh.vertices.push(Vertex { pos: rect.right_bottom(), uv, color: bottom });
    mesh.vertices.push(Vertex { pos: rect.left_bottom(), uv, color: bottom });
    mesh.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
    ui.painter().add(egui::Shape::mesh(mesh));
}

/// A compact row for the Up Next / Previously lists.
fn mini_row(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    app: &mut App,
    t: &nocturne_api::Track,
    dim: bool,
) -> bool {
    let (rect, resp) = ui.allocate_exact_size(
        Vec2::new(ui.available_width(), 40.0),
        egui::Sense::click(),
    );
    if resp.hovered() {
        ui.painter()
            .rect_filled(rect, Rounding::same(5.0), Color32::from_rgb(30, 28, 34));
    }
    let mut row = ui.child_ui(
        rect.shrink2(Vec2::new(4.0, 2.0)),
        Layout::left_to_right(Align::Center),
        None,
    );
    app.art_at(&mut row, ctx, t.art_url.as_ref(), 30.0, 3.0);
    row.add_space(8.0);
    row.vertical(|ui| {
        ui.spacing_mut().item_spacing.y = 0.0;
        let title = RichText::new(&t.name).size(12.5);
        ui.add(
            egui::Label::new(if dim { title.weak() } else { title })
                .truncate()
                .selectable(false),
        );
        ui.add(
            egui::Label::new(RichText::new(&t.artists).weak().size(11.0))
                .truncate()
                .selectable(false),
        );
    });
    resp.clicked()
}

/// Like + "add to playlist", the pair that belongs next to any track, anywhere.
///
/// Both are LOCAL: Spotify 403s every library and playlist write for a restricted app, so these
/// live on disk and are merged into the display. The tooltip says so rather than pretending.
fn add_controls(
    ui: &mut egui::Ui,
    app: &App,
    track: &nocturne_api::Track,
    liked: bool,
    playlists: &[nocturne_api::Playlist],
    size: f32,
) {
    let (icon, color) = if liked {
        (Icon::HeartFilled, theme::ORANGE)
    } else {
        (Icon::Heart, Color32::from_gray(150))
    };
    if icons::button_colored(ui, icon, size + 8.0, color)
        .on_hover_text(if liked {
            "Remove from your library (local)"
        } else {
            "Add to your library (local)"
        })
        .clicked()
    {
        app.send(Cmd::ToggleLike(track.clone()));
    }

    // A real dialog, opened from anywhere. The previous version put an egui popup INSIDE the
    // track-list ScrollArea, keyed per row — it got laid out, clipped and re-anchored every frame
    // while the list scrolled, which is why it flickered and fought the mouse.
    if icons::button_colored(ui, Icon::Plus, size + 8.0, Color32::from_gray(150))
        .on_hover_text("Add to a playlist (local — Spotify blocks writes for this app)")
        .clicked()
    {
        app.request_add_to(track.clone());
    }
    let _ = playlists;
}

/// A transport MODE button (shuffle, repeat): unframed, orange when active, with a dot beneath it.
///
/// The colour alone isn't enough — an orange glyph on an album cover can read as a hover state, or
/// as nothing at all on warm art. The dot is the unambiguous "this is on", and it's the same
/// language every player uses.
fn mode_button(ui: &mut egui::Ui, icon: Icon, box_size: f32, active: bool, hover: &str) -> bool {
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(box_size), egui::Sense::click());
    let t = ui.ctx().animate_bool(resp.id, active);

    let color = if active {
        theme::ORANGE
    } else if resp.hovered() {
        theme::ORANGE_HI
    } else {
        Color32::from_gray(155)
    };
    icons::paint(
        &ui.painter().clone(),
        rect.shrink(box_size * 0.26),
        icon,
        color,
    );
    if t > 0.0 {
        ui.painter().circle_filled(
            egui::pos2(rect.center().x, rect.max.y - 1.5),
            2.0 * t,
            theme::ORANGE,
        );
    }
    resp.on_hover_text(hover).clicked()
}

/// A modern pill toggle — the kind every current app uses, instead of egui's default checkbox.
fn toggle(ui: &mut egui::Ui, on: &mut bool, label: &str) -> egui::Response {
    let galley = ui.painter().layout_no_wrap(
        label.to_string(),
        egui::FontId::proportional(12.5),
        theme::TEXT,
    );
    let track_w = 34.0f32;
    let h = 20.0f32;
    let size = Vec2::new(track_w + 8.0 + galley.size().x, h.max(22.0));
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());
    if resp.clicked() {
        *on = !*on;
    }

    let track = egui::Rect::from_min_size(
        egui::pos2(rect.min.x, rect.center().y - h / 2.0),
        Vec2::new(track_w, h),
    );
    // Animate the knob so the state change reads as motion, not a jump.
    let t = ui.ctx().animate_bool(resp.id, *on);
    let bg = Color32::from_rgb(
        (46.0 + t * (233.0 - 46.0)) as u8,
        (42.0 + t * (110.0 - 42.0)) as u8,
        (50.0 + t * (44.0 - 50.0)) as u8,
    );
    ui.painter().rect_filled(track, Rounding::same(h / 2.0), bg);
    if !*on {
        ui.painter().rect_stroke(
            track,
            Rounding::same(h / 2.0),
            Stroke::new(1.0, Color32::from_rgb(70, 62, 58)),
        );
    }
    let knob_x = egui::lerp((track.min.x + h / 2.0)..=(track.max.x - h / 2.0), t);
    ui.painter().circle_filled(
        egui::pos2(knob_x, track.center().y),
        h / 2.0 - 3.0,
        Color32::from_rgb(240, 238, 236),
    );

    let text_pos = egui::pos2(track.max.x + 8.0, rect.center().y - galley.size().y / 2.0);
    ui.painter().galley(text_pos, galley, theme::TEXT);
    resp
}

/// A small square icon button (panel toggles).
fn icon_button(ui: &mut egui::Ui, glyph: &str, hover: &str) -> egui::Response {
    ui.add_sized([28.0, 28.0], egui::Button::new(RichText::new(glyph).size(14.0)))
        .on_hover_text(hover)
}

/// A button with a real icon and a label — no glyph fonts.
fn labeled_button(ui: &mut egui::Ui, icon: Icon, label: &str) -> bool {
    let galley = ui.painter().layout_no_wrap(
        label.to_string(),
        egui::FontId::proportional(13.0),
        theme::TEXT,
    );
    let h = 30.0;
    let w = 12.0 + 14.0 + 7.0 + galley.size().x + 12.0;
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(w, h), egui::Sense::click());
    let v = ui.style().interact(&resp);
    ui.painter()
        .rect(rect, Rounding::same(8.0), v.weak_bg_fill, v.bg_stroke);

    let ir = egui::Rect::from_center_size(
        egui::pos2(rect.min.x + 12.0 + 7.0, rect.center().y),
        Vec2::splat(14.0),
    );
    icons::paint(&ui.painter().clone(), ir, icon, v.fg_stroke.color);
    ui.painter().galley(
        egui::pos2(ir.max.x + 7.0, rect.center().y - galley.size().y / 2.0),
        galley,
        v.fg_stroke.color,
    );
    resp.clicked()
}

/// A mood pill.
///
/// Not a button with an emoji glued to it: a proper capsule with an accent dot, a hairline border
/// that warms on hover, and a subtle top-lit gradient so it reads as a physical object rather than
/// a coloured rectangle. Text is small, medium-weight and never emoji — glyph icons in a UI inherit
/// the font's metrics and look like clip art.
fn chip(ui: &mut egui::Ui, label: &str, accent: Color32) -> bool {
    const H: f32 = 30.0;
    const DOT: f32 = 7.0;
    const PAD: f32 = 12.0;

    let galley = ui.painter().layout_no_wrap(
        label.to_string(),
        egui::FontId::proportional(12.5),
        theme::TEXT,
    );
    let w = PAD + DOT + 8.0 + galley.size().x + PAD;
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(w, H), egui::Sense::click());

    let t = ui.ctx().animate_bool(resp.id, resp.hovered());
    let rounding = Rounding::same(H / 2.0);

    // Body: charcoal, lifting toward the accent on hover.
    let base = Color32::from_rgb(30, 27, 34);
    let hot = Color32::from_rgb(
        (30.0 + (accent.r() as f32 - 30.0) * 0.22) as u8,
        (27.0 + (accent.g() as f32 - 27.0) * 0.22) as u8,
        (34.0 + (accent.b() as f32 - 34.0) * 0.22) as u8,
    );
    let body = Color32::from_rgb(
        egui::lerp(base.r() as f32..=hot.r() as f32, t) as u8,
        egui::lerp(base.g() as f32..=hot.g() as f32, t) as u8,
        egui::lerp(base.b() as f32..=hot.b() as f32, t) as u8,
    );
    ui.painter().rect_filled(rect, rounding, body);

    // A whisper of light along the top edge — the thing that separates "capsule" from "rectangle".
    let sheen = egui::Rect::from_min_max(
        rect.min,
        egui::pos2(rect.max.x, rect.min.y + H * 0.5),
    );
    ui.painter().rect_filled(
        sheen,
        Rounding { nw: H / 2.0, ne: H / 2.0, sw: 0.0, se: 0.0 },
        Color32::from_white_alpha(if resp.hovered() { 10 } else { 5 }),
    );

    // Hairline border, warming to the accent on hover.
    let border = Color32::from_rgba_unmultiplied(
        accent.r(),
        accent.g(),
        accent.b(),
        (40.0 + 150.0 * t) as u8,
    );
    ui.painter().rect_stroke(rect, rounding, Stroke::new(1.0, border));

    // Accent dot, with a soft halo when hovered.
    let dot_c = egui::pos2(rect.min.x + PAD + DOT / 2.0, rect.center().y);
    if t > 0.0 {
        ui.painter().circle_filled(
            dot_c,
            DOT / 2.0 + 3.0 * t,
            Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), (60.0 * t) as u8),
        );
    }
    ui.painter().circle_filled(dot_c, DOT / 2.0, accent);

    let text_pos = egui::pos2(
        dot_c.x + DOT / 2.0 + 8.0,
        rect.center().y - galley.size().y / 2.0,
    );
    let text_col = if resp.hovered() { theme::TEXT } else { Color32::from_gray(190) };
    ui.painter().galley(text_pos, galley, text_col);

    resp.clicked()
}

/// A sidebar row that highlights when it's the active view, with an optional leading icon.
fn nav_item(ui: &mut egui::Ui, label: &str, active: bool, icon: Option<Icon>) -> bool {
    let (rect, resp) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), 32.0), egui::Sense::click());
    if active {
        ui.painter().rect_filled(rect, Rounding::same(6.0), theme::SEL_HL);
    } else if resp.hovered() {
        ui.painter()
            .rect_filled(rect, Rounding::same(6.0), Color32::from_rgb(30, 28, 34));
    }
    let color = if active { theme::ORANGE } else { theme::TEXT };

    let mut x = rect.min.x + 10.0;
    if let Some(icon) = icon {
        let ir = egui::Rect::from_center_size(
            egui::pos2(x + 8.0, rect.center().y),
            Vec2::splat(16.0),
        );
        icons::paint(&ui.painter().clone(), ir, icon, color);
        x += 24.0;
    }
    let mut sub = ui.child_ui(
        egui::Rect::from_min_max(egui::pos2(x, rect.min.y), rect.max),
        Layout::left_to_right(Align::Center),
        None,
    );
    sub.add(
        egui::Label::new(RichText::new(label).color(color))
            .truncate()
            .selectable(false),
    );
    resp.clicked()
}
