//! Font stack: emoji and symbol coverage.
//!
//! egui's bundled fonts cover Latin and a handful of icon glyphs, so real content falls apart —
//! Camden's playlists have 🍁 and 🍂 in their names, and they rendered as blank boxes.
//!
//! The system's only emoji font is NotoColorEmoji, which is a **CBDT bitmap** font: epaint has no
//! COLR/CBDT support and cannot draw a single glyph from it. So the emoji font is the *monochrome*
//! Noto Emoji, vendored in `assets/` (OFL, see assets/OFL.txt) rather than taken from the system,
//! where it isn't packaged at all. Emoji render as clean monochrome glyphs, not color — that's the
//! honest ceiling for egui today, and it beats tofu.
//!
//! Symbol fonts come from the system (DejaVu + Noto Sans Symbols 2) and cover the transport glyphs
//! (⏮ ⏸ ⏭ ♥ 🔊) plus the long tail of arrows/dingbats that turn up in track and playlist names.

use eframe::egui;
use egui::{FontData, FontDefinitions, FontFamily};

const EMOJI: &[u8] = include_bytes!("../assets/NotoEmoji-Regular.ttf");

/// System fonts to append as fallbacks, best-effort and in priority order.
const SYSTEM_FALLBACKS: &[&str] = &[
    "/usr/share/fonts/noto/NotoSansSymbols2-Regular.ttf",
    "/usr/share/fonts/noto/NotoSansSymbols-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
];

pub fn install(ctx: &egui::Context) {
    let mut defs = FontDefinitions::default();

    defs.font_data
        .insert("noto-emoji".into(), FontData::from_static(EMOJI));
    let mut fallbacks = vec!["noto-emoji".to_string()];

    for path in SYSTEM_FALLBACKS {
        // A missing system font is not an error — just one less fallback.
        let Ok(bytes) = std::fs::read(path) else {
            tracing::debug!("font not present: {path}");
            continue;
        };
        let name = std::path::Path::new(path)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| path.to_string());
        defs.font_data
            .insert(name.clone(), FontData::from_owned(bytes));
        fallbacks.push(name);
    }

    // Fallbacks go *after* the default fonts: they fill gaps, they don't replace the UI face.
    for family in [FontFamily::Proportional, FontFamily::Monospace] {
        let list = defs.families.entry(family).or_default();
        for name in &fallbacks {
            list.push(name.clone());
        }
    }

    tracing::info!("fonts: emoji + {} system fallbacks", fallbacks.len() - 1);
    ctx.set_fonts(defs);
}
