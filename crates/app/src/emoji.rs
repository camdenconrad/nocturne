//! Real **color** emoji, drawn as images.
//!
//! egui/epaint cannot render color fonts — no COLR, no CBDT. Handing it NotoColorEmoji produces
//! nothing; handing it the monochrome Noto Emoji produces flat white blobs. Neither is what a 🍁
//! is supposed to look like.
//!
//! So we don't ask the text renderer to do it. NotoColorEmoji is a CBDT font: each glyph is an
//! embedded PNG. `ttf-parser` hands those bitmaps over directly, we decode them once, upload them
//! as textures, and paint them inline with the text. Result: actual color emoji, at the cost of
//! laying out emoji-bearing strings ourselves.
//!
//! Everything is cached per character. A string with no emoji takes the plain-label fast path and
//! never touches any of this.

use eframe::egui;
use egui::{Color32, RichText, TextureHandle, Vec2};
use std::collections::HashMap;

const COLOR_EMOJI: &str = "/usr/share/fonts/noto/NotoColorEmoji.ttf";

pub struct Emoji {
    /// The font file, kept alive for the lifetime of the process so `ttf_parser::Face` can borrow
    /// it. Leaked deliberately — it's one allocation and it lives as long as the app does.
    face: Option<ttf_parser::Face<'static>>,
    cache: HashMap<char, Option<TextureHandle>>,
}

impl Emoji {
    pub fn new() -> Self {
        let face = std::fs::read(COLOR_EMOJI).ok().and_then(|bytes| {
            let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
            match ttf_parser::Face::parse(leaked, 0) {
                Ok(f) => {
                    tracing::info!("color emoji: loaded {COLOR_EMOJI}");
                    Some(f)
                }
                Err(e) => {
                    tracing::warn!("color emoji: cannot parse {COLOR_EMOJI}: {e}");
                    None
                }
            }
        });
        Self {
            face,
            cache: HashMap::new(),
        }
    }

    /// Does this char have a color bitmap? Asking the font what it can draw beats guessing unicode
    /// ranges — but ASCII is excluded first, because NotoColorEmoji *does* carry bitmaps for the
    /// digits and `#`/`*` (they're the bases of keycap sequences like 0️⃣). Without that guard,
    /// "Fall lofi 2026" renders every digit as a separate inline image and comes out as "2 0 2 6".
    fn bitmap(&self, ch: char) -> Option<Vec<u8>> {
        if ch.is_ascii() {
            return None;
        }
        let face = self.face.as_ref()?;
        let gid = face.glyph_index(ch)?;
        // 128px is the strike NotoColorEmoji ships; asking for it avoids a scaled miss.
        let raster = face.glyph_raster_image(gid, 128)?;
        (raster.format == ttf_parser::RasterImageFormat::PNG).then(|| raster.data.to_vec())
    }

    fn texture(&mut self, ctx: &egui::Context, ch: char) -> Option<TextureHandle> {
        if let Some(hit) = self.cache.get(&ch) {
            return hit.clone();
        }
        let tex = self.bitmap(ch).and_then(|png| {
            let img = image::load_from_memory(&png).ok()?.to_rgba8();
            let size = [img.width() as usize, img.height() as usize];
            let color = egui::ColorImage::from_rgba_unmultiplied(size, img.as_raw());
            Some(ctx.load_texture(format!("emoji-{ch}"), color, egui::TextureOptions::LINEAR))
        });
        self.cache.insert(ch, tex.clone());
        tex
    }

    /// True if the string has at least one char the emoji font can draw.
    pub fn has_emoji(&self, s: &str) -> bool {
        s.chars().any(|c| !c.is_ascii() && self.bitmap(c).is_some())
    }

    /// Draw `text` with color emoji inlined. Falls back to a plain label when there's no emoji,
    /// which is the overwhelmingly common case and stays on egui's fast path.
    pub fn label(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        text: &str,
        size: f32,
        color: Option<Color32>,
        strong: bool,
    ) {
        let style = |t: &str| {
            let mut r = RichText::new(t).size(size);
            if strong {
                r = r.strong();
            }
            if let Some(c) = color {
                r = r.color(c);
            }
            r
        };

        if !self.has_emoji(text) {
            ui.add(egui::Label::new(style(text)).truncate().selectable(false));
            return;
        }

        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 1.0;
            let mut run = String::new();
            for ch in text.chars() {
                if self.bitmap(ch).is_some() {
                    if !run.is_empty() {
                        ui.add(egui::Label::new(style(&run)).selectable(false));
                        run.clear();
                    }
                    match self.texture(ui.ctx(), ch) {
                        Some(tex) => {
                            ui.add(egui::Image::new(&tex).fit_to_exact_size(Vec2::splat(size)));
                        }
                        None => {
                            // Unreachable in practice (bitmap() just said yes), but never panic
                            // over a glyph.
                            run.push(ch);
                        }
                    }
                } else {
                    run.push(ch);
                }
            }
            if !run.is_empty() {
                ui.add(egui::Label::new(style(&run)).selectable(false));
            }
            let _ = ctx;
        });
    }
}
