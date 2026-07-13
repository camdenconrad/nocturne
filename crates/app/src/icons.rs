//! Real vector icons, drawn with the painter.
//!
//! The transport and library controls were emoji glyphs (⏮ ⏸ ♥ 🔊). That's a font's idea of a
//! picture: it inherits the font's weight, baseline and metrics, so nothing lines up, the shapes
//! differ per platform, and at 14px they're mush. These are drawn from primitives instead — crisp
//! at any size, one visual weight, and coloured by us.

use eframe::egui;
use egui::{Color32, Painter, Pos2, Rect, Stroke, Vec2};

#[derive(Clone, Copy, PartialEq)]
pub enum Icon {
    Play,
    Pause,
    Prev,
    Next,
    Heart,
    HeartFilled,
    Plus,
    Volume,
    VolumeLow,
    VolumeMute,
    Menu,
    Fullscreen,
    Close,
    Search,
    Radio,
    ChevronLeft,
}

/// Paint `icon` centred in `rect`. `rect` is the *icon box*; the glyph is inset from it.
pub fn paint(p: &Painter, rect: Rect, icon: Icon, color: Color32) {
    let c = rect.center();
    let s = rect.width().min(rect.height());
    let r = s * 0.5;
    // One stroke weight for every outline icon, so they read as a family.
    let w = (s * 0.11).max(1.3);
    let stroke = Stroke::new(w, color);

    match icon {
        Icon::Play => {
            // Span the box symmetrically: half the width either side of centre. The triangle's
            // *centroid* sits a third of the way from its base, so nudge right by w/6 to make the
            // shape look centred rather than merely be centred.
            let hw = r * 0.58;
            let hh = r * 0.66;
            let nudge = hw / 3.0;
            p.add(egui::Shape::convex_polygon(
                vec![
                    Pos2::new(c.x - hw + nudge, c.y - hh),
                    Pos2::new(c.x - hw + nudge, c.y + hh),
                    Pos2::new(c.x + hw + nudge, c.y),
                ],
                color,
                Stroke::NONE,
            ));
        }
        Icon::Pause => {
            let bw = r * 0.28;
            let bh = r * 0.62;
            let gap = r * 0.24;
            for dx in [-gap - bw / 2.0, gap + bw / 2.0] {
                p.rect_filled(
                    Rect::from_center_size(Pos2::new(c.x + dx, c.y), Vec2::new(bw, bh * 2.0)),
                    egui::Rounding::same(w * 0.5),
                    color,
                );
            }
        }
        Icon::Prev | Icon::Next => {
            let dir = if icon == Icon::Next { 1.0 } else { -1.0 };
            let hh = r * 0.58;

            // The glyph is a bar + two triangles. Lay it out on an explicit span from -X..+X so the
            // WHOLE composite is centred on the box — the previous version grew outward from the
            // centre in one direction only, so every prev/next button sat visibly off-centre.
            let x0 = -r * 0.78; // trailing edge (bar side)
            let x1 = r * 0.78; // leading edge (tip of the outer triangle)
            let tri_w = (x1 - x0 - w) / 2.0; // two triangles share the space after the bar

            // Bar, at the trailing edge.
            let bx = c.x + dir * (x0 + w / 2.0);
            p.rect_filled(
                Rect::from_center_size(Pos2::new(bx, c.y), Vec2::new(w, hh * 2.0)),
                egui::Rounding::same(w * 0.5),
                color,
            );

            // Triangles, apex pointing IN the direction of travel (getting this backwards drew
            // next-track as a rewind).
            for k in 0..2 {
                let base = x0 + w + k as f32 * tri_w;
                let tip = base + tri_w;
                p.add(egui::Shape::convex_polygon(
                    vec![
                        Pos2::new(c.x + dir * base, c.y - hh),
                        Pos2::new(c.x + dir * base, c.y + hh),
                        Pos2::new(c.x + dir * tip, c.y),
                    ],
                    color,
                    Stroke::NONE,
                ));
            }
        }
        Icon::Heart | Icon::HeartFilled => {
            // Two lobes and a point. Built as a polygon so it fills cleanly.
            // The classic heart parametric is NOT centred on its own origin — it runs about
            // -17..+12 in y — so centring the curve's origin in the box leaves the glyph sitting
            // low. Compute the curve, then centre its actual bounding box.
            let steps = 48;
            let raw: Vec<(f32, f32)> = (0..=steps)
                .map(|i| {
                    let t = i as f32 / steps as f32 * std::f32::consts::TAU;
                    let x = 16.0 * t.sin().powi(3);
                    let y = 13.0 * t.cos() - 5.0 * (2.0 * t).cos() - 2.0 * (3.0 * t).cos()
                        - (4.0 * t).cos();
                    (x, y)
                })
                .collect();
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for (_, y) in &raw {
                lo = lo.min(*y);
                hi = hi.max(*y);
            }
            let mid = (lo + hi) / 2.0;
            let scale = r * 0.92 / 17.0;
            let pts: Vec<Pos2> = raw
                .iter()
                .map(|(x, y)| Pos2::new(c.x + x * scale, c.y - (y - mid) * scale))
                .collect();
            if icon == Icon::HeartFilled {
                p.add(egui::Shape::convex_polygon(pts, color, Stroke::NONE));
            } else {
                p.add(egui::Shape::closed_line(pts, stroke));
            }
        }
        Icon::Plus => {
            p.line_segment([Pos2::new(c.x - r * 0.62, c.y), Pos2::new(c.x + r * 0.62, c.y)], stroke);
            p.line_segment([Pos2::new(c.x, c.y - r * 0.62), Pos2::new(c.x, c.y + r * 0.62)], stroke);
        }
        Icon::Volume | Icon::VolumeLow | Icon::VolumeMute => {
            // Speaker + waves, centred as a whole: the speaker sits left of centre precisely so the
            // waves to its right balance it.
            let bx = c.x - r * 0.72;
            p.rect_filled(
                Rect::from_min_max(
                    Pos2::new(bx, c.y - r * 0.22),
                    Pos2::new(bx + r * 0.32, c.y + r * 0.22),
                ),
                egui::Rounding::same(1.5),
                color,
            );
            p.add(egui::Shape::convex_polygon(
                vec![
                    Pos2::new(bx + r * 0.3, c.y - r * 0.22),
                    Pos2::new(bx + r * 0.85, c.y - r * 0.6),
                    Pos2::new(bx + r * 0.85, c.y + r * 0.6),
                    Pos2::new(bx + r * 0.3, c.y + r * 0.22),
                ],
                color,
                Stroke::NONE,
            ));
            match icon {
                Icon::VolumeMute => {
                    // A clean X, rather than a crossed-out speaker that reads as noise.
                    let o = r * 0.3;
                    let x0 = c.x + r * 0.4;
                    p.line_segment(
                        [Pos2::new(x0 - o, c.y - o), Pos2::new(x0 + o, c.y + o)],
                        stroke,
                    );
                    p.line_segment(
                        [Pos2::new(x0 + o, c.y - o), Pos2::new(x0 - o, c.y + o)],
                        stroke,
                    );
                }
                _ => {
                    // One arc for low, two for full.
                    let arcs = if icon == Icon::VolumeLow { 1 } else { 2 };
                    for k in 0..arcs {
                        let rad = r * (0.34 + 0.24 * k as f32);
                        let cx = c.x + r * 0.28;
                        let mut pts = Vec::new();
                        for i in 0..=14 {
                            let a = -0.7 + (i as f32 / 14.0) * 1.4;
                            pts.push(Pos2::new(cx + rad * a.cos(), c.y + rad * a.sin()));
                        }
                        p.add(egui::Shape::line(pts, stroke));
                    }
                }
            }
        }
        Icon::Menu => {
            for k in -1..=1 {
                let y = c.y + k as f32 * r * 0.42;
                p.line_segment(
                    [Pos2::new(c.x - r * 0.62, y), Pos2::new(c.x + r * 0.62, y)],
                    stroke,
                );
            }
        }
        Icon::Fullscreen => {
            // Four corner brackets.
            let a = r * 0.62;
            let l = r * 0.32;
            for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
                let px = c.x + sx * a;
                let py = c.y + sy * a;
                p.line_segment([Pos2::new(px, py), Pos2::new(px - sx * l, py)], stroke);
                p.line_segment([Pos2::new(px, py), Pos2::new(px, py - sy * l)], stroke);
            }
        }
        Icon::Close => {
            let a = r * 0.48;
            p.line_segment(
                [Pos2::new(c.x - a, c.y - a), Pos2::new(c.x + a, c.y + a)],
                stroke,
            );
            p.line_segment(
                [Pos2::new(c.x + a, c.y - a), Pos2::new(c.x - a, c.y + a)],
                stroke,
            );
        }
        Icon::ChevronLeft => {
            let a = r * 0.42;
            let stroke = Stroke::new(w * 1.15, color);
            p.add(egui::Shape::line(
                vec![
                    Pos2::new(c.x + a * 0.7, c.y - a),
                    Pos2::new(c.x - a * 0.7, c.y),
                    Pos2::new(c.x + a * 0.7, c.y + a),
                ],
                stroke,
            ));
        }
        Icon::Search => {
            p.circle_stroke(Pos2::new(c.x - r * 0.14, c.y - r * 0.14), r * 0.42, stroke);
            p.line_segment(
                [
                    Pos2::new(c.x + r * 0.18, c.y + r * 0.18),
                    Pos2::new(c.x + r * 0.6, c.y + r * 0.6),
                ],
                stroke,
            );
        }
        Icon::Radio => {
            // A broadcast mark: a dot with two arcs, which is what a station IS.
            p.circle_filled(c, r * 0.16, color);
            for k in 0..2 {
                let rad = r * (0.4 + 0.26 * k as f32);
                for dir in [-1.0f32, 1.0] {
                    let mut pts = Vec::new();
                    for i in 0..=12 {
                        let a = (i as f32 / 12.0 - 0.5) * 1.5;
                        pts.push(Pos2::new(
                            c.x + dir * rad * a.cos(),
                            c.y + rad * a.sin(),
                        ));
                    }
                    p.add(egui::Shape::line(pts, stroke));
                }
            }
        }
    }
}

/// An icon button. `box_size` is the clickable square; the glyph is inset.
pub fn button(ui: &mut egui::Ui, icon: Icon, box_size: f32, frame: bool) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(box_size), egui::Sense::click());
    let v = ui.style().interact(&resp);

    if frame {
        ui.painter().rect(
            rect,
            egui::Rounding::same(8.0),
            v.weak_bg_fill,
            v.bg_stroke,
        );
    }
    let color = if resp.hovered() {
        livewall_uikit::theme::ORANGE_HI
    } else {
        v.fg_stroke.color
    };
    paint(&ui.painter().clone(), rect.shrink(box_size * 0.28), icon, color);
    resp
}

/// Same, but the glyph is drawn in an explicit colour (liked hearts, etc).
pub fn button_colored(
    ui: &mut egui::Ui,
    icon: Icon,
    box_size: f32,
    color: Color32,
) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(box_size), egui::Sense::click());
    let c = if resp.hovered() {
        livewall_uikit::theme::ORANGE_HI
    } else {
        color
    };
    paint(&ui.painter().clone(), rect.shrink(box_size * 0.26), icon, c);
    resp
}
