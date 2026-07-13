mod backend;

use backend::{Cmd, Shared};
use eframe::egui;
use livewall_uikit::{chrome, theme};
use nocturne_api::fmt_duration;
use std::collections::HashMap;

fn main() -> eframe::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()))
        .init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_decorations(false)
            .with_inner_size([1080.0, 700.0])
            .with_min_inner_size([720.0, 480.0])
            .with_app_id("nocturne"),
        ..Default::default()
    };

    eframe::run_native(
        "Nocturne",
        options,
        Box::new(|cc| {
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
    /// Art textures, uploaded once per URL and reused across frames.
    textures: HashMap<String, egui::TextureHandle>,
    playlists_loaded: bool,
    autologin_tried: bool,
}

impl App {
    fn new(state: Shared, tx: tokio::sync::mpsc::UnboundedSender<Cmd>) -> Self {
        Self {
            state,
            tx,
            query: String::new(),
            textures: HashMap::new(),
            playlists_loaded: false,
            autologin_tried: false,
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
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        theme::apply(ctx);
        chrome::title_bar(ctx, "Nocturne");

        let (logged_in, busy, status, now) = {
            let s = self.state.lock().unwrap();
            (s.logged_in, s.busy, s.status.clone(), s.now.clone())
        };

        // A cached refresh token means signing in asks the user nothing — so don't make them
        // click a button to be asked nothing. Only a cold start shows the sign-in screen.
        if !logged_in && !self.autologin_tried {
            self.autologin_tried = true;
            if nocturne_session::has_cached_login() {
                self.send(Cmd::Login);
            }
        }

        if logged_in && !self.playlists_loaded {
            self.playlists_loaded = true;
            self.send(Cmd::LoadPlaylists);
            self.send(Cmd::LoadSaved);
        }

        if !logged_in {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(180.0);
                    ui.heading("Nocturne");
                    ui.label("Spotify, native to Rune.");
                    ui.add_space(16.0);
                    ui.add_enabled_ui(!busy, |ui| {
                        if ui.button("  Sign in with Spotify  ").clicked() {
                            self.send(Cmd::Login);
                        }
                    });
                    ui.add_space(8.0);
                    ui.label(egui::RichText::new(&status).weak());
                });
            });
            return;
        }

        self.sidebar(ctx);
        self.now_playing(ctx, now);
        self.track_list(ctx, &status, busy);

        // A playing track needs a moving progress bar; repaint even when idle.
        if self.state.lock().unwrap().now.as_ref().is_some_and(|n| !n.paused) {
            ctx.request_repaint_after(std::time::Duration::from_millis(250));
        }
    }
}

impl App {
    fn sidebar(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("nav")
            .resizable(false)
            .exact_width(220.0)
            .show(ctx, |ui| {
                ui.add_space(8.0);
                if ui.button("♥  Liked Songs").clicked() {
                    self.send(Cmd::LoadSaved);
                }
                ui.add_space(10.0);
                ui.label(egui::RichText::new("PLAYLISTS").weak().small());
                ui.add_space(4.0);

                let playlists = self.state.lock().unwrap().playlists.clone();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for p in playlists {
                        let label = match p.tracks {
                            Some(n) => format!("{}  ({n})", p.name),
                            None => p.name.clone(),
                        };
                        if ui.selectable_label(false, label).clicked() {
                            self.send(Cmd::OpenPlaylist(p.id.clone()));
                        }
                    }
                });
            });
    }

    fn track_list(&mut self, ctx: &egui::Context, status: &str, busy: bool) {
        egui::TopBottomPanel::top("search").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let r = ui.add(
                    egui::TextEdit::singleline(&mut self.query)
                        .hint_text("Search Spotify…")
                        .desired_width(360.0),
                );
                if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    self.send(Cmd::Search(self.query.clone()));
                }
                if ui.button("Search").clicked() {
                    self.send(Cmd::Search(self.query.clone()));
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if busy {
                        ui.spinner();
                    }
                    ui.label(egui::RichText::new(status).weak());
                });
            });
            ui.add_space(4.0);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let tracks = self.state.lock().unwrap().tracks.clone();
            egui::ScrollArea::vertical().show(ui, |ui| {
                for t in &tracks {
                    let resp = ui
                        .horizontal(|ui| {
                            if let Some(tex) = t.art_url.as_ref().and_then(|u| self.art(ctx, u)) {
                                ui.add(egui::Image::new(&tex).fit_to_exact_size(egui::vec2(40.0, 40.0)));
                            } else {
                                ui.allocate_space(egui::vec2(40.0, 40.0));
                            }
                            ui.vertical(|ui| {
                                ui.label(egui::RichText::new(&t.name).strong());
                                ui.label(egui::RichText::new(&t.artists).weak().small());
                            });
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.label(
                                    egui::RichText::new(fmt_duration(t.duration_ms)).weak().small(),
                                );
                            });
                        })
                        .response;

                    // The row itself is the play button — double-click, as in every music app.
                    let hit = ui.interact(
                        resp.rect,
                        egui::Id::new(&t.uri),
                        egui::Sense::click(),
                    );
                    if hit.double_clicked() {
                        self.send(Cmd::Play(t.uri.clone()));
                    }
                    if hit.hovered() {
                        ui.painter().rect_filled(
                            resp.rect,
                            4.0,
                            theme::SEL_HL,
                        );
                    }
                    ui.separator();
                }
            });
        });
    }

    fn now_playing(&mut self, ctx: &egui::Context, now: Option<backend::NowPlaying>) {
        egui::TopBottomPanel::bottom("now").exact_height(76.0).show(ctx, |ui| {
            ui.add_space(6.0);
            let Some(n) = now else {
                ui.centered_and_justified(|ui| {
                    ui.label(egui::RichText::new("nothing playing").weak());
                });
                return;
            };

            ui.horizontal(|ui| {
                if let Some(tex) = n.art_url.as_ref().and_then(|u| self.art(ctx, u)) {
                    ui.add(egui::Image::new(&tex).fit_to_exact_size(egui::vec2(52.0, 52.0)));
                }
                ui.vertical(|ui| {
                    ui.label(egui::RichText::new(&n.name).strong());
                    ui.label(egui::RichText::new(&n.artists).weak().small());
                });

                ui.add_space(12.0);
                if ui.button(if n.paused { "▶" } else { "⏸" }).clicked() {
                    self.send(Cmd::PlayPause);
                }

                let elapsed = n.elapsed_ms();
                ui.label(egui::RichText::new(fmt_duration(elapsed)).weak().small());

                // Seekable: drag the bar, release to seek.
                let mut pos = elapsed as f32 / n.duration_ms.max(1) as f32;
                let bar = ui.add(
                    egui::Slider::new(&mut pos, 0.0..=1.0)
                        .show_value(false)
                        .trailing_fill(true),
                );
                if bar.drag_stopped() || bar.changed() && !bar.dragged() {
                    self.send(Cmd::Seek((pos * n.duration_ms as f32) as u32));
                }
                ui.label(
                    egui::RichText::new(fmt_duration(n.duration_ms)).weak().small(),
                );
            });
        });
    }
}
