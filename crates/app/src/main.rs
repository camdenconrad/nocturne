mod backend;
mod cache;
mod emoji;
mod fonts;
mod icons;
mod mpris;

use backend::{Cmd, NowPlaying, Shared};
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
    query: String,
    mood: String,
    textures: HashMap<String, egui::TextureHandle>,
    emoji: emoji::Emoji,
    loaded: bool,
    autologin_tried: bool,
    /// Local volume while dragging, so the slider doesn't fight the backend each frame.
    volume: f32,
    show_sidebar: bool,
    show_nowpane: bool,
    /// Full-screen now-playing. The whole window becomes the album art.
    vibe: bool,
    /// Blurred backdrop textures, keyed by art url — built once, reused.
    blurred: HashMap<String, egui::TextureHandle>,
    /// The track whose "add to playlist" dialog is open, if any.
    add_to: Option<nocturne_api::Track>,
}

impl App {
    fn new(state: Shared, tx: tokio::sync::mpsc::UnboundedSender<Cmd>) -> Self {
        Self {
            state,
            tx,
            query: String::new(),
            mood: String::new(),
            textures: HashMap::new(),
            emoji: emoji::Emoji::new(),
            loaded: false,
            autologin_tried: false,
            volume: 1.0,
            // Tucked away: full screen is home, and the library is something you pull out.
            show_sidebar: false,
            show_nowpane: true,
            // Full screen IS the app. Browsing is a detour you take and come back from.
            vibe: true,
            blurred: HashMap::new(),
            add_to: None,
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
        let playlists = self.state.lock().unwrap().playlists.clone();

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
            return Some(t.clone());
        }
        let bytes = self.state.lock().unwrap().art.get(url).cloned()?;
        let img = image::load_from_memory(&bytes).ok()?.to_rgba8();
        let size = [img.width() as usize, img.height() as usize];
        let color = egui::ColorImage::from_rgba_unmultiplied(size, img.as_raw());
        let tex = ctx.load_texture(url, color, egui::TextureOptions::LINEAR);
        self.textures.insert(url.to_string(), tex.clone());
        Some(tex)
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
        chrome::title_bar(ctx, "Nocturne");

        let (logged_in, busy, status, now, view, current) = {
            let s = self.state.lock().unwrap();
            (
                s.logged_in,
                s.busy,
                s.status.clone(),
                s.now.clone(),
                s.view.clone(),
                s.current_uri.clone(),
            )
        };

        if !logged_in && !self.autologin_tried {
            self.autologin_tried = true;
            if nocturne_session::has_cached_login() {
                self.send(Cmd::Login);
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
                let view = self.state.lock().unwrap().view.clone();
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
                ui.add_space(14.0);
                ui.label(RichText::new("PLAYLISTS").weak().small());
                ui.add_space(6.0);

                let playlists = self.state.lock().unwrap().playlists.clone();
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
                    let s = self.state.lock().unwrap();
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

                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if icons::button(ui, Icon::Radio, 30.0, true)
                            .on_hover_text("Show/hide up-next")
                            .clicked()
                        {
                            self.show_nowpane = !self.show_nowpane;
                        }
                        ui.add_space(8.0);

                        let (mut autoplay, radio_loading, analyzing, feats) = {
                            let s = self.state.lock().unwrap();
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
                        }
                    }
                });

                ui.add_space(14.0);
                let tracks = self.state.lock().unwrap().tracks.clone();
                ui.horizontal(|ui| {
                    self.emoji.label(ui, &ctx2, view, 24.0, None, true);
                    ui.add_space(8.0);
                    ui.label(RichText::new(format!("{} tracks", tracks.len())).weak().small());
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if !tracks.is_empty() && labeled_button(ui, Icon::Play, "Play all") {
                            self.send(Cmd::PlayQueue(tracks.clone()));
                            self.vibe = true;
                        }
                    });
                });
                ui.add_space(8.0);

                // --- rows ---
                let (liked, playlists) = {
                    let s = self.state.lock().unwrap();
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
                        let s = self.state.lock().unwrap();
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
                const BH: f32 = 34.0;
                let brow = egui::Rect::from_center_size(
                    egui::pos2(centre.center().x, centre.min.y + 24.0),
                    Vec2::new(150.0, BH),
                );
                let mut bui = ui.child_ui(brow, Layout::left_to_right(Align::Center), None);
                bui.spacing_mut().item_spacing.x = 8.0;
                if icons::button(&mut bui, Icon::Prev, BH, true).clicked() {
                    self.send(Cmd::Prev);
                }
                let pp = if n.paused { Icon::Play } else { Icon::Pause };
                if icons::button(&mut bui, pp, BH + 6.0, true).clicked() {
                    self.send(Cmd::PlayPause);
                }
                if icons::button(&mut bui, Icon::Next, BH, true).clicked() {
                    self.send(Cmd::Next);
                }

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

impl App {
    /// A big, soft, blurred version of the cover — the backdrop. Built once per cover.
    ///
    /// Blur is done by downscaling hard and letting the GPU's bilinear filter scale it back up:
    /// a 24px image stretched across 1300px IS a blur, for free, with no convolution pass.
    fn backdrop(&mut self, ctx: &egui::Context, url: &str) -> Option<egui::TextureHandle> {
        if let Some(t) = self.blurred.get(url) {
            return Some(t.clone());
        }
        let bytes = self.state.lock().unwrap().art.get(url).cloned()?;
        let img = image::load_from_memory(&bytes).ok()?;
        // 12px, not 24: stretched across ~1300px that's a much softer blur, which is what stops
        // the backdrop competing with the cover in front of it.
        let small = img.resize_exact(12, 12, image::imageops::FilterType::Triangle).to_rgba8();
        let color = egui::ColorImage::from_rgba_unmultiplied([12, 12], small.as_raw());
        let tex = ctx.load_texture(format!("blur-{url}"), color, egui::TextureOptions::LINEAR);
        self.blurred.insert(url.to_string(), tex.clone());
        Some(tex)
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
                    if let Some(bg) = self.backdrop(&ctx2, &url) {
                        egui::Image::new(&bg).paint_at(ui, full);
                        // Flat scrim so anything on top stays readable…
                        ui.painter()
                            .rect_filled(full, Rounding::ZERO, Color32::from_black_alpha(130));
                        // …then a SMOOTH top-to-bottom darkening. This used to be a rect covering
                        // the bottom half, which drew a hard horizontal seam straight across the
                        // middle of the window. A gradient has no edge to see.
                        vertical_gradient(
                            ui,
                            full,
                            Color32::from_black_alpha(0),
                            Color32::from_black_alpha(165),
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
                // Soft shadow under the cover — this is what makes it feel like an object.
                ui.painter().rect_filled(
                    art_rect.translate(Vec2::new(0.0, 14.0)).expand(6.0),
                    Rounding::same(20.0),
                    Color32::from_black_alpha(90),
                );
                let mut aui = ui.child_ui(art_rect, Layout::top_down(Align::Center), None);
                // The big cover — a 640px image, not the 64px thumbnail upscaled.
                let big = n.art_big.clone().or_else(|| n.art_url.clone());
                self.art_at(&mut aui, &ctx2, big.as_ref(), art_size, 16.0);

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
                    let s = self.state.lock().unwrap();
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
                const BTN_H: f32 = 44.0;
                let row_w = 44.0 + 52.0 + 44.0 + 20.0;
                let row = egui::Rect::from_center_size(
                    egui::pos2(ctrl.center().x, ctrl.min.y + 58.0),
                    Vec2::new(row_w, BTN_H),
                );
                let mut bui = ui.child_ui(row, Layout::left_to_right(Align::Center), None);
                bui.spacing_mut().item_spacing.x = 10.0;
                if icons::button(&mut bui, Icon::Prev, BTN_H, true).clicked() {
                    self.send(Cmd::Prev);
                }
                let pp = if n.paused { Icon::Play } else { Icon::Pause };
                if icons::button(&mut bui, pp, BTN_H + 8.0, true).clicked() {
                    self.send(Cmd::PlayPause);
                }
                if icons::button(&mut bui, Icon::Next, BTN_H, true).clicked() {
                    self.send(Cmd::Next);
                }

                // 5. The library tab. This is the ONLY way out — pop it, pick something, and the
                //    app drops you straight back here.
                let tab = egui::Rect::from_min_size(
                    egui::pos2(full.min.x + 8.0, full.min.y + 8.0),
                    Vec2::new(38.0, 34.0),
                );
                let mut tui2 = ui.child_ui(tab, Layout::left_to_right(Align::Center), None);
                if icons::button(&mut tui2, Icon::Menu, 34.0, true)
                    .on_hover_text("Library (L)")
                    .clicked()
                    || ui.input(|i| i.key_pressed(egui::Key::L))
                {
                    self.show_sidebar = true;
                }
                let _ = bg_click;

                // Space = play/pause, the one shortcut every player has.
                if ui.input(|i| i.key_pressed(egui::Key::Space)) {
                    self.send(Cmd::PlayPause);
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
