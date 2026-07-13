use eframe::egui;
use livewall_uikit::{chrome, theme};

fn main() -> eframe::Result<()> {
    tracing_subscriber::fmt::init();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_decorations(false)
            .with_inner_size([1100.0, 720.0])
            .with_app_id("nocturne"),
        ..Default::default()
    };
    eframe::run_native(
        "Nocturne",
        options,
        Box::new(|_cc| Ok(Box::new(NocturneApp::default()))),
    )
}

#[derive(Default)]
struct NocturneApp {
    search: String,
}

impl eframe::App for NocturneApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        theme::apply(ctx);
        chrome::title_bar(ctx, "Nocturne");

        egui::TopBottomPanel::top("search_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.text_edit_singleline(&mut self.search);
            });
        });
        egui::TopBottomPanel::bottom("now_playing").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("nothing playing");
                let _ = ui.button("⏮");
                let _ = ui.button("⏯");
                let _ = ui.button("⏭");
            });
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("Sign in to Spotify to load your library.");
            let _ = ui.button("Sign in");
        });
    }
}
