//! Real-ESRGAN upscaling for the full-screen cover.
//!
//! ## What we start from
//!
//! Not the 640px cover the Web API advertises. Spotify's CDN will hand over the **original master**
//! the label uploaded — usually 1800–2000px — if you rewrite the size prefix in the image id (see
//! [`crate::cache::art_fetch_best`]). That art is already sharper than the full-screen view needs
//! on a 4K panel, and it costs one HTTP GET.
//!
//! So the upscale is no longer rescuing a soft 640px JPEG. It runs on top of the master, purely for
//! headroom: 2000 → **8000×8000** through `realesrgan-x4plus` on the 4080 (Vulkan, not CPU).
//!
//! ## Why 4×, and not the 8× this started as
//!
//! 8× of the master would be 16000², which is ~1GB of VRAM per cover and ~6GB across the resident
//! window — for detail no display can show. 4× of the master (8000²) already beats the old
//! 640→8×→5120 chain on both axes: bigger, and made of real detail rather than invented detail.
//!
//! ## RAM only, deliberately
//!
//! The 8000² result is a ~118MB PNG and 256MB decoded. Caching that on disk would cost tens of GB
//! across a library, so nothing is persisted: the pass runs into RAM, and the resident window in
//! [`crate::App`] *is* the cache. A cover that falls out of the window and comes back is recomputed
//! — and while it recomputes, the view shows the master, which is sharp on its own.
//!
//! Everything here is best-effort: no binary, no GPU, a crash — the UI just keeps the master. An
//! upscale is never allowed to break playback.

use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Album art is illustration as often as photography, and the general model handles both without
/// the plastic look `x4plus-anime` gives a photographic cover.
const MODEL: &str = "realesrgan-x4plus";
const MODELS_DIR: &str = "/usr/share/realesrgan-ncnn-vulkan/models";
const BIN: &str = "realesrgan-ncnn-vulkan";

/// Is upscaling possible on this machine at all?
pub fn available() -> bool {
    which(BIN).is_some() && Path::new(MODELS_DIR).exists()
}

fn which(bin: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|d| d.join(bin))
            .find(|p| p.is_file())
    })
}

/// Upscale a cover 4× and hand back RGBA pixels ready for the GPU.
///
/// Blocking and GPU-bound — ~12s on a 2000px master, so call it from a worker thread, never the UI
/// thread. Returns `None` if anything at all went wrong.
///
/// The tool is file-in/file-out, so this stages both sides on disk and deletes them; the PNG it
/// writes is streamed straight into the decoder rather than read into a `Vec` first, so the
/// compressed bytes and the 256MB of pixels never coexist.
pub fn upscale_image(art_id: &str, src_bytes: &[u8]) -> Option<image::RgbaImage> {
    if !available() {
        return None;
    }
    let dir = crate::cache::scratch_dir();
    std::fs::create_dir_all(&dir).ok()?;

    // Unique per process: two covers upscaling back-to-back must not scribble on each other.
    let stem = format!("{art_id}.{}", std::process::id());
    let in_path = dir.join(format!("{stem}.in"));
    let out_path = dir.join(format!("{stem}.out.png"));
    std::fs::write(&in_path, src_bytes).ok()?;

    let t0 = std::time::Instant::now();
    let ok = run(&in_path, &out_path);
    let _ = std::fs::remove_file(&in_path);

    let decoded = ok
        .then(|| {
            std::fs::File::open(&out_path)
                .ok()
                .and_then(|f| image::load(BufReader::new(f), image::ImageFormat::Png).ok())
                .map(|img| img.into_rgba8())
        })
        .flatten();
    let _ = std::fs::remove_file(&out_path);

    let img = decoded?;
    tracing::info!(
        "esrgan: {art_id} upscaled to {}×{} in {}ms ({}MB in VRAM)",
        img.width(),
        img.height(),
        t0.elapsed().as_millis(),
        (img.width() as usize * img.height() as usize * 4) / 1_000_000,
    );
    Some(img)
}

fn run(in_path: &Path, out_path: &Path) -> bool {
    let (Some(i), Some(o)) = (in_path.to_str(), out_path.to_str()) else {
        return false;
    };
    let status = Command::new(BIN)
        .args([
            "-i", i, "-o", o, "-n", MODEL, "-m", MODELS_DIR, "-s", "4",
            // Explicit format: the tool infers from the extension, and a wrong guess writes garbage.
            "-f", "png",
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();

    match status {
        Ok(s) if s.success() => true,
        Ok(s) => {
            tracing::warn!("esrgan: {BIN} exited {s}");
            false
        }
        Err(e) => {
            tracing::warn!("esrgan: could not run {BIN}: {e}");
            false
        }
    }
}
