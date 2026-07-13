mod backend;
mod cache;
mod emoji;
mod fonts;
mod mpris;

use backend::{Cmd, NowPlaying, Shared};
use eframe::egui;
use egui::{Align, Color32, Layout, Margin, RichText, Rounding, Stroke, Vec2};
use livewall_uikit::{chrome, theme};
use nocturne_api::fmt_duration;
use std::collections::HashMap;

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
            show_sidebar: true,
            show_nowpane: true,
            // Full screen IS the app. Browsing is a detour you take and come back from.
            vibe: true,
            blurred: HashMap::new(),
        }
    }

    fn send(&self, cmd: Cmd) {
        let _ = self.tx.send(cmd);
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
                        if ui
                            .add(egui::Button::new(RichText::new("✕").size(13.0)).frame(false))
                            .on_hover_text("Close (Esc)")
                            .clicked()
                        {
                            self.show_sidebar = false;
                        }
                    });
                });
                ui.add_space(8.0);

                if nav_item(ui, "♥   Liked Songs", view == "Liked Songs") {
                    self.send(Cmd::LoadSaved);
                    // Picking a list is what takes you out of full screen.
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
                        self.art_at(ui, &ctx2, n.art_url.as_ref(), art_size, 10.0);
                        let r = egui::Rect::from_min_size(at, Vec2::splat(art_size));
                        let hit = ui.interact(r, egui::Id::new("pane-art"), egui::Sense::click());
                        if hit.hovered() {
                            ui.painter()
                                .rect_filled(r, Rounding::same(10.0), Color32::from_black_alpha(70));
                            ui.put(
                                r,
                                egui::Label::new(RichText::new("⛶  Full screen").size(15.0))
                                    .selectable(false),
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
                    if icon_button(ui, "⛶", "Back to full screen (Esc)").clicked()
                        || ui.input(|i| i.key_pressed(egui::Key::Escape))
                    {
                        self.vibe = true;
                    }
                    if icon_button(ui, "☰", "Library").clicked() {
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
                        if icon_button(ui, "▤", "Show/hide up-next").clicked() {
                            self.show_nowpane = !self.show_nowpane;
                        }
                        ui.add_space(8.0);

                        let (mut autoplay, radio_loading, analyzing, feats) = {
                            let s = self.state.lock().unwrap();
                            (s.autoplay, s.radio_loading, s.analyzing, s.taste_features)
                        };
                        if ui
                            .checkbox(&mut autoplay, "Radio")
                            .on_hover_text(format!(
                                "When the queue runs out, keep playing — picked from your \
                                 listening, using {feats} analyzed tracks"
                            ))
                            .changed()
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
                    if (go || ui.button("▶ Radio").clicked()) && !self.mood.trim().is_empty() {
                        self.send(Cmd::MoodRadio(self.mood.clone()));
                    }
                    ui.add_space(4.0);
                    for (label, phrase) in [
                        ("🍂 cozy lofi", "chill autumn lofi cozy"),
                        ("⚡ hype", "hype energetic workout"),
                        ("🌧 sad", "sad melancholy acoustic"),
                        ("🌙 late night", "dark moody night chill"),
                        ("💃 party", "happy dance party"),
                    ] {
                        if chip(ui, label) {
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
                        if !tracks.is_empty() && ui.button("▶ Play all").clicked() {
                            self.send(Cmd::PlayQueue(tracks.clone()));
                            self.vibe = true;
                        }
                    });
                });
                ui.add_space(8.0);

                // --- rows ---
                let liked = self.state.lock().unwrap().liked.clone();
                let mut play = None;
                let mut like = None;
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
                                        ui.label(RichText::new("▶").small().color(theme::ORANGE));
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
                                // Add/remove from the library.
                                let on = liked.contains(&t.uri);
                                let heart = RichText::new(if on { "♥" } else { "♡" })
                                    .color(if on { theme::ORANGE } else { theme::TEXT });
                                if ui
                                    .add(egui::Button::new(heart).frame(false))
                                    .on_hover_text("Add to / remove from your library (local)")
                                    .clicked()
                                {
                                    like = Some(t.uri.clone());
                                }
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
                if let Some(uri) = like {
                    self.send(Cmd::ToggleLike(uri));
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
                        ui.put(
                            art_rect,
                            egui::Label::new(RichText::new("⛶").size(18.0)).selectable(false),
                        );
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
                rui.add_space(6.0);
                let icon = if self.volume < 0.01 {
                    "🔇"
                } else if self.volume < 0.5 {
                    "🔉"
                } else {
                    "🔊"
                };
                self.emoji.label(&mut rui, &ctx2, icon, 15.0, None, false);

                // -- centre: transport + scrubber --
                let Some(n) = n else { return };
                let cw = 460.0f32.min(full.width() - 2.0 * SIDE - 20.0).max(220.0);
                let centre = egui::Rect::from_center_size(
                    egui::pos2(full.center().x, full.center().y),
                    Vec2::new(cw, full.height()),
                );
                let mut cui = ui.child_ui(centre, Layout::top_down(Align::Center), None);
                cui.add_space(4.0);

                cui.horizontal(|ui| {
                    let w = ui.available_width();
                    ui.add_space((w - 132.0).max(0.0) / 2.0);
                    if ui.add(egui::Button::new(RichText::new("⏮").size(15.0))).clicked() {
                        self.send(Cmd::Prev);
                    }
                    ui.add_space(6.0);
                    let icon = if n.paused { "▶" } else { "⏸" };
                    if ui
                        .add_sized([40.0, 30.0], egui::Button::new(RichText::new(icon).size(17.0)))
                        .clicked()
                    {
                        self.send(Cmd::PlayPause);
                    }
                    ui.add_space(6.0);
                    if ui.add(egui::Button::new(RichText::new("⏭").size(15.0))).clicked() {
                        self.send(Cmd::Next);
                    }
                });

                cui.add_space(3.0);
                let elapsed = n.elapsed_ms();
                cui.horizontal(|ui| {
                    ui.label(RichText::new(fmt_duration(elapsed)).weak().small());
                    let mut pos = elapsed as f32 / n.duration_ms.max(1) as f32;
                    let bar = ui.add_sized(
                        [ui.available_width() - 42.0, 14.0],
                        egui::Slider::new(&mut pos, 0.0..=1.0)
                            .show_value(false)
                            .trailing_fill(true),
                    );
                    // Seek on release only — seeking every frame mid-drag would thrash the player.
                    if bar.drag_stopped() {
                        self.send(Cmd::Seek((pos * n.duration_ms as f32) as u32));
                    }
                    ui.label(RichText::new(fmt_duration(n.duration_ms)).weak().small());
                });
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
        let small = img.resize_exact(24, 24, image::imageops::FilterType::Triangle).to_rgba8();
        let color = egui::ColorImage::from_rgba_unmultiplied([24, 24], small.as_raw());
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
                        // Scrim, or nothing on top of it would be readable.
                        ui.painter().rect_filled(full, Rounding::ZERO, Color32::from_black_alpha(150));
                        // Vignette: darken the bottom so the controls sit on solid ground.
                        let lower = egui::Rect::from_min_max(
                            egui::pos2(full.min.x, full.center().y),
                            full.max,
                        );
                        ui.painter().rect_filled(lower, Rounding::ZERO, Color32::from_black_alpha(90));
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

                // 4. Scrubber + transport, centred at the bottom.
                let cw = 520.0f32.min(full.width() - 80.0);
                let ctrl = egui::Rect::from_center_size(
                    egui::pos2(full.center().x, full.max.y - 62.0),
                    Vec2::new(cw, 96.0),
                );
                let mut cui = ui.child_ui(ctrl, Layout::top_down(Align::Center), None);

                // Fixed-width time labels on both sides, so the bar sits dead centre instead of
                // drifting with the length of the timestamps.
                let elapsed = n.elapsed_ms();
                cui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 8.0;
                    ui.add_sized(
                        [44.0, 16.0],
                        egui::Label::new(RichText::new(fmt_duration(elapsed)).weak().small()),
                    );
                    let mut pos = elapsed as f32 / n.duration_ms.max(1) as f32;
                    let bar = ui.add_sized(
                        [ui.available_width() - 52.0, 16.0],
                        egui::Slider::new(&mut pos, 0.0..=1.0)
                            .show_value(false)
                            .trailing_fill(true),
                    );
                    if bar.drag_stopped() {
                        self.send(Cmd::Seek((pos * n.duration_ms as f32) as u32));
                    }
                    ui.add_sized(
                        [44.0, 16.0],
                        egui::Label::new(
                            RichText::new(fmt_duration(n.duration_ms)).weak().small(),
                        ),
                    );
                });

                cui.add_space(10.0);
                cui.horizontal(|ui| {
                    let w = ui.available_width();
                    ui.add_space((w - 170.0).max(0.0) / 2.0);
                    if ui.add_sized([44.0, 36.0], egui::Button::new(RichText::new("⏮").size(17.0))).clicked() {
                        self.send(Cmd::Prev);
                    }
                    ui.add_space(8.0);
                    let icon = if n.paused { "▶" } else { "⏸" };
                    if ui
                        .add_sized([52.0, 40.0], egui::Button::new(RichText::new(icon).size(20.0)))
                        .clicked()
                    {
                        self.send(Cmd::PlayPause);
                    }
                    ui.add_space(8.0);
                    if ui.add_sized([44.0, 36.0], egui::Button::new(RichText::new("⏭").size(17.0))).clicked() {
                        self.send(Cmd::Next);
                    }
                });

                // 5. The library tab. This is the ONLY way out — pop it, pick something, and the
                //    app drops you straight back here.
                let tab = egui::Rect::from_min_size(
                    egui::pos2(full.min.x + 8.0, full.min.y + 8.0),
                    Vec2::new(38.0, 34.0),
                );
                if ui
                    .put(tab, egui::Button::new(RichText::new("☰").size(16.0)))
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

/// A small square icon button (panel toggles).
fn icon_button(ui: &mut egui::Ui, glyph: &str, hover: &str) -> egui::Response {
    ui.add_sized([28.0, 28.0], egui::Button::new(RichText::new(glyph).size(14.0)))
        .on_hover_text(hover)
}

/// A rounded, glossy pill button — the mood presets.
fn chip(ui: &mut egui::Ui, label: &str) -> bool {
    let galley = ui.painter().layout_no_wrap(
        label.to_string(),
        egui::FontId::proportional(12.0),
        theme::TEXT,
    );
    let size = Vec2::new(galley.size().x + 20.0, 26.0);
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());

    let bg = if resp.hovered() {
        Color32::from_rgb(52, 44, 38)
    } else {
        Color32::from_rgb(34, 31, 38)
    };
    ui.painter().rect_filled(rect, Rounding::same(13.0), bg);
    // The gloss: a lighter top half, which is what makes a flat rect read as a physical pill.
    let top = egui::Rect::from_min_max(rect.min, egui::pos2(rect.max.x, rect.center().y));
    ui.painter().rect_filled(
        top,
        Rounding {
            nw: 13.0,
            ne: 13.0,
            sw: 0.0,
            se: 0.0,
        },
        Color32::from_white_alpha(6),
    );
    ui.painter().rect_stroke(
        rect,
        Rounding::same(13.0),
        Stroke::new(1.0, Color32::from_rgb(70, 58, 48)),
    );
    ui.put(rect, egui::Label::new(RichText::new(label).size(12.0)).selectable(false));
    resp.clicked()
}

/// A sidebar row that highlights when it's the active view.
fn nav_item(ui: &mut egui::Ui, label: &str, active: bool) -> bool {
    let (rect, resp) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), 32.0), egui::Sense::click());
    if active {
        ui.painter().rect_filled(rect, Rounding::same(6.0), theme::SEL_HL);
    } else if resp.hovered() {
        ui.painter()
            .rect_filled(rect, Rounding::same(6.0), Color32::from_rgb(30, 28, 34));
    }
    let text = RichText::new(label).color(if active { theme::ORANGE } else { theme::TEXT });
    ui.child_ui(
        rect.shrink2(Vec2::new(10.0, 0.0)),
        Layout::left_to_right(Align::Center),
        None,
    )
    .add(egui::Label::new(text).truncate().selectable(false));
    resp.clicked()
}
