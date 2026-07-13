mod backend;
mod cache;
mod emoji;
mod fonts;

use backend::{Cmd, NowPlaying, Shared};
use eframe::egui;
use egui::{Align, Color32, Layout, Margin, RichText, Rounding, Stroke, Vec2};
use livewall_uikit::{chrome, theme};
use nocturne_api::fmt_duration;
use std::collections::HashMap;

const SIDEBAR_W: f32 = 232.0;
const ROW_H: f32 = 56.0;
const BAR_H: f32 = 92.0;
const ART: f32 = 40.0;

fn main() -> eframe::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()))
        .init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_decorations(false)
            .with_inner_size([1180.0, 760.0])
            .with_min_inner_size([860.0, 560.0])
            .with_app_id("nocturne"),
        ..Default::default()
    };

    eframe::run_native(
        "Nocturne",
        options,
        Box::new(|cc| {
            fonts::install(&cc.egui_ctx);
            let ctx = cc.egui_ctx.clone();
            let (state, tx) = backend::spawn(move || ctx.request_repaint());
            Ok(Box::new(App::new(state, tx)))
        }),
    )
}

struct App {
    state: Shared,
    tx: tokio::sync::mpsc::UnboundedSender<Cmd>,
    query: String,
    textures: HashMap<String, egui::TextureHandle>,
    emoji: emoji::Emoji,
    loaded: bool,
    autologin_tried: bool,
    /// Local volume while dragging, so the slider doesn't fight the backend each frame.
    volume: f32,
}

impl App {
    fn new(state: Shared, tx: tokio::sync::mpsc::UnboundedSender<Cmd>) -> Self {
        Self {
            state,
            tx,
            query: String::new(),
            textures: HashMap::new(),
            emoji: emoji::Emoji::new(),
            loaded: false,
            autologin_tried: false,
            volume: 1.0,
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

    fn art_or_placeholder(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, url: Option<&String>, size: f32) {
        let tex = url.and_then(|u| self.art(ctx, u));
        match tex {
            Some(t) => {
                ui.add(
                    egui::Image::new(&t)
                        .fit_to_exact_size(Vec2::splat(size))
                        .rounding(Rounding::same(4.0)),
                );
            }
            None => {
                // Reserve the same box so rows don't reflow when art lands.
                let (rect, _) = ui.allocate_exact_size(Vec2::splat(size), egui::Sense::hover());
                ui.painter()
                    .rect_filled(rect, Rounding::same(4.0), Color32::from_rgb(38, 35, 42));
            }
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        theme::apply(ctx);
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
        }

        if !logged_in {
            self.sign_in(ctx, busy, &status);
            return;
        }

        self.sidebar(ctx);
        self.now_bar(ctx, now.clone());
        self.main(ctx, &view, &status, busy, current.as_deref());

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

    fn sidebar(&mut self, ctx: &egui::Context) {
        let ctx = &ctx.clone();
        egui::SidePanel::left("nav")
            .resizable(false)
            .exact_width(SIDEBAR_W)
            .frame(
                egui::Frame::none()
                    .fill(Color32::from_rgb(16, 15, 19))
                    .inner_margin(Margin::symmetric(12.0, 14.0)),
            )
            .show(ctx, |ui| {
                ui.label(RichText::new("NOCTURNE").size(15.0).strong().color(theme::ORANGE));
                ui.add_space(14.0);

                let view = self.state.lock().unwrap().view.clone();
                if nav_item(ui, "♥   Liked Songs", view == "Liked Songs") {
                    self.send(Cmd::LoadSaved);
                }

                ui.add_space(16.0);
                ui.label(RichText::new("PLAYLISTS").weak().small());
                ui.add_space(6.0);

                let playlists = self.state.lock().unwrap().playlists.clone();
                let mut clicked: Option<String> = None;
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for p in playlists {
                            let active = view == p.name;
                            let (rect, resp) = ui.allocate_exact_size(
                                Vec2::new(ui.available_width(), 30.0),
                                egui::Sense::click(),
                            );
                            if active {
                                ui.painter().rect_filled(rect, Rounding::same(5.0), theme::SEL_HL);
                            } else if resp.hovered() {
                                ui.painter().rect_filled(
                                    rect,
                                    Rounding::same(5.0),
                                    Color32::from_rgb(30, 28, 34),
                                );
                            }
                            let mut row = ui.child_ui(
                                rect.shrink2(Vec2::new(9.0, 0.0)),
                                Layout::left_to_right(Align::Center),
                                None,
                            );
                            let col = if active { Some(theme::ORANGE) } else { None };
                            self.emoji.label(&mut row, ctx, &p.name, 14.0, col, false);
                            if resp.clicked() {
                                clicked = Some(p.id.clone());
                            }
                        }
                    });
                if let Some(id) = clicked {
                    self.send(Cmd::OpenPlaylist(id));
                }
            });
    }

    fn main(&mut self, ctx: &egui::Context, view: &str, status: &str, busy: bool, current: Option<&str>) {
        egui::CentralPanel::default()
            .frame(egui::Frame::none().inner_margin(Margin::symmetric(18.0, 12.0)))
            .show(ctx, |ui| {
                // --- search + header ---
                ui.horizontal(|ui| {
                    let field = ui.add(
                        egui::TextEdit::singleline(&mut self.query)
                            .hint_text("Search Spotify…")
                            .desired_width(320.0)
                            .margin(Margin::symmetric(10.0, 7.0)),
                    );
                    let enter = field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if enter || ui.button("Search").clicked() {
                        self.send(Cmd::Search(self.query.clone()));
                    }
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        // Radio: keep playing past the end of the queue with Spotify's station for
                        // the last track. On by default; this is the switch.
                        let (mut autoplay, radio_loading) = {
                            let st = self.state.lock().unwrap();
                            (st.autoplay, st.radio_loading)
                        };
                        if ui
                            .checkbox(&mut autoplay, "Radio")
                            .on_hover_text(
                                "When the queue runs out, keep playing similar tracks (Spotify radio)",
                            )
                            .changed()
                        {
                            self.send(Cmd::SetAutoplay(autoplay));
                        }
                        ui.add_space(10.0);
                        if busy || radio_loading {
                            ui.spinner();
                        }
                        ui.label(RichText::new(status).weak().small());
                    });
                });

                ui.add_space(14.0);
                let tracks = self.state.lock().unwrap().tracks.clone();
                ui.horizontal(|ui| {
                    self.emoji.label(ui, ctx, view, 24.0, None, true);
                    ui.add_space(8.0);
                    ui.label(
                        RichText::new(format!("{} tracks", tracks.len()))
                            .weak()
                            .small(),
                    );
                });
                ui.add_space(10.0);

                // --- track rows ---
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
                                ui.painter().rect_stroke(
                                    rect,
                                    Rounding::same(6.0),
                                    Stroke::new(1.0, theme::ORANGE),
                                );
                            }

                            let mut row = ui.child_ui(
                                rect.shrink2(Vec2::new(10.0, 8.0)),
                                Layout::left_to_right(Align::Center),
                                None,
                            );

                            // index / playing marker
                            row.allocate_ui_with_layout(
                                Vec2::new(24.0, ART),
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
                            self.art_or_placeholder(&mut row, ctx, t.art_url.as_ref(), ART);
                            row.add_space(12.0);

                            let em = &mut self.emoji;
                            row.vertical(|ui| {
                                ui.spacing_mut().item_spacing.y = 1.0;
                                let col = is_current.then_some(theme::ORANGE);
                                em.label(ui, ctx, &t.name, 14.0, col, true);
                                ui.label(RichText::new(&t.artists).weak().small());
                            });

                            row.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                ui.label(
                                    RichText::new(fmt_duration(t.duration_ms)).weak().small(),
                                );
                                ui.add_space(16.0);
                                // Album, but only when there's room — it's the first thing to go.
                                if ui.available_width() > 220.0 {
                                    ui.add_sized(
                                        [ui.available_width().min(260.0), ART],
                                        egui::Label::new(RichText::new(&t.album).weak().small())
                                            .truncate(),
                                    );
                                }
                            });

                            // Single click plays: this is a player, not a file manager.
                            if resp.clicked() {
                                self.send(Cmd::Play(t.uri.clone()));
                            }
                        }
                    });
            });
    }

    fn now_bar(&mut self, ctx: &egui::Context, now: Option<NowPlaying>) {
        egui::TopBottomPanel::bottom("now")
            .exact_height(BAR_H)
            .frame(
                egui::Frame::none()
                    .fill(Color32::from_rgb(16, 15, 19))
                    .inner_margin(Margin::symmetric(18.0, 10.0)),
            )
            .show(ctx, |ui| {
                let Some(n) = now else {
                    ui.centered_and_justified(|ui| {
                        ui.label(RichText::new("nothing playing").weak());
                    });
                    return;
                };

                ui.horizontal(|ui| {
                    // left: art + title
                    self.art_or_placeholder(ui, ctx, n.art_url.as_ref(), 60.0);
                    ui.add_space(12.0);
                    ui.vertical(|ui| {
                        ui.add_space(8.0);
                        ui.spacing_mut().item_spacing.y = 2.0;
                        ui.add_sized(
                            [220.0, 18.0],
                            egui::Label::new(RichText::new(&n.name).strong()).truncate(),
                        );
                        ui.add_sized(
                            [220.0, 16.0],
                            egui::Label::new(RichText::new(&n.artists).weak().small()).truncate(),
                        );
                    });

                    // right: volume (claimed before the centre so it can't be squeezed out)
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.add_space(4.0);
                        let vol = ui.add_sized(
                            [110.0, 18.0],
                            egui::Slider::new(&mut self.volume, 0.0..=1.0).show_value(false),
                        );
                        if vol.changed() {
                            self.send(Cmd::Volume(self.volume));
                        }
                        ui.label(RichText::new("🔊").small());

                        // centre: transport + scrubber, filling what's left
                        ui.with_layout(Layout::top_down(Align::Center), |ui| {
                            ui.add_space(6.0);
                            ui.horizontal(|ui| {
                                ui.with_layout(
                                    Layout::top_down(Align::Center),
                                    |ui| {
                                        ui.horizontal(|ui| {
                                            let w = ui.available_width();
                                            ui.add_space((w - 110.0).max(0.0) / 2.0);
                                            if ui.button("⏮").clicked() {
                                                self.send(Cmd::Prev);
                                            }
                                            let icon = if n.paused { "▶" } else { "⏸" };
                                            if ui
                                                .add_sized([34.0, 26.0], egui::Button::new(
                                                    RichText::new(icon).size(15.0),
                                                ))
                                                .clicked()
                                            {
                                                self.send(Cmd::PlayPause);
                                            }
                                            if ui.button("⏭").clicked() {
                                                self.send(Cmd::Next);
                                            }
                                        });

                                        ui.add_space(2.0);
                                        let elapsed = n.elapsed_ms();
                                        ui.horizontal(|ui| {
                                            ui.label(
                                                RichText::new(fmt_duration(elapsed)).weak().small(),
                                            );
                                            let mut pos =
                                                elapsed as f32 / n.duration_ms.max(1) as f32;
                                            let bar = ui.add_sized(
                                                [ui.available_width() - 44.0, 14.0],
                                                egui::Slider::new(&mut pos, 0.0..=1.0)
                                                    .show_value(false)
                                                    .trailing_fill(true),
                                            );
                                            // Seek on release only — seeking every frame mid-drag
                                            // would thrash the player.
                                            if bar.drag_stopped() {
                                                self.send(Cmd::Seek(
                                                    (pos * n.duration_ms as f32) as u32,
                                                ));
                                            }
                                            ui.label(
                                                RichText::new(fmt_duration(n.duration_ms))
                                                    .weak()
                                                    .small(),
                                            );
                                        });
                                    },
                                );
                            });
                        });
                    });
                });
            });
    }
}

/// A sidebar row that highlights when it's the active view.
fn nav_item(ui: &mut egui::Ui, label: &str, active: bool) -> bool {
    let (rect, resp) = ui.allocate_exact_size(
        Vec2::new(ui.available_width(), 30.0),
        egui::Sense::click(),
    );
    if active {
        ui.painter().rect_filled(rect, Rounding::same(5.0), theme::SEL_HL);
    } else if resp.hovered() {
        ui.painter()
            .rect_filled(rect, Rounding::same(5.0), Color32::from_rgb(30, 28, 34));
    }
    let text = RichText::new(label).color(if active { theme::ORANGE } else { theme::TEXT });
    ui.child_ui(
        rect.shrink2(Vec2::new(9.0, 0.0)),
        Layout::left_to_right(Align::Center),
        None,
    )
    .add(egui::Label::new(text).truncate().selectable(false));
    resp.clicked()
}
