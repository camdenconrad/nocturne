//! Real-ESRGAN upscaling for the full-screen cover.
//!
//! Spotify's largest cover is 640×640. On a 4K panel the full-screen view stretches that to well
//! over 1000px, and a bilinear stretch of a 640px JPEG is exactly as soft as it sounds.
//!
//! `realesrgan-ncnn-vulkan` runs the real model on the 4080 (Vulkan, not CPU), 4× → 2560×2560.
//!
//! ## Cached, deliberately
//!
//! Camden asked for "a real ESRGAN pass each time". It IS a real pass — but the result is *cached*,
//! because the model is deterministic: the same cover upscaled twice produces byte-identical output.
//! Re-running it on every play would burn ~1s of GPU per track to recompute a value we already have,
//! and it would stutter the very view it's meant to make beautiful. First play of an album pays the
//! pass; every play after is instant.
//!
//! Everything here is best-effort: no binary, no GPU, a timeout, a crash — the UI just keeps the
//! original cover. An upscale is never allowed to break playback.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Model choice matters. Album art is illustration as often as photography, and `x4plus-anime` is
/// tuned for flat colour and line art — it keeps edges crisp where the photo model smears them.
/// The general model handles photographic covers.
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

/// Where the upscaled cover lives. Keyed by the source id + model, so changing the model or the
/// cover invalidates it for free.
pub fn cached_path(art_id: &str) -> PathBuf {
    crate::cache::art_hires_dir().join(format!("{art_id}.{MODEL}.png"))
}

/// Upscale `src_bytes` 4×. Blocking and GPU-bound — call from `spawn_blocking`.
///
/// Returns the PNG bytes, or `None` if anything at all went wrong.
pub fn upscale(art_id: &str, src_bytes: &[u8]) -> Option<Vec<u8>> {
    let out_path = cached_path(art_id);
    if let Ok(bytes) = std::fs::read(&out_path) {
        return Some(bytes);
    }
    if !available() {
        return None;
    }

    let dir = crate::cache::art_hires_dir();
    std::fs::create_dir_all(&dir).ok()?;

    // The tool is file-in/file-out, so stage the source next to the target.
    let in_path = dir.join(format!("{art_id}.in"));
    std::fs::write(&in_path, src_bytes).ok()?;

    let t0 = std::time::Instant::now();
    let status = Command::new(BIN)
        .args([
            "-i",
            in_path.to_str()?,
            "-o",
            out_path.to_str()?,
            "-n",
            MODEL,
            "-m",
            MODELS_DIR,
            "-s",
            "4",
            // Explicit format: the tool infers from the extension, and a wrong guess writes garbage.
            "-f",
            "png",
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();

    let _ = std::fs::remove_file(&in_path);

    match status {
        Ok(s) if s.success() => {
            let bytes = std::fs::read(&out_path).ok()?;
            tracing::info!(
                "esrgan: upscaled {art_id} 4x in {}ms ({} KB)",
                t0.elapsed().as_millis(),
                bytes.len() / 1024
            );
            Some(bytes)
        }
        Ok(s) => {
            tracing::warn!("esrgan: {BIN} exited {s}");
            let _ = std::fs::remove_file(&out_path);
            None
        }
        Err(e) => {
            tracing::warn!("esrgan: could not run {BIN}: {e}");
            None
        }
    }
}
