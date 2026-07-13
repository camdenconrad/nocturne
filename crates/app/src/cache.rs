//! On-disk cache: album art and library listings.
//!
//! Without this, every launch re-downloads the same few hundred covers and re-walks the whole
//! library before showing a single row. With it, a relaunch paints from disk immediately and the
//! network refresh happens behind the already-visible UI.
//!
//! Everything here is best-effort: a cache miss, a corrupt file, or an unwritable directory just
//! means we fall back to the network. Nothing in here is allowed to fail a user action.

use serde::{de::DeserializeOwned, Serialize};
use std::path::PathBuf;

fn root() -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache"));
    base.join("nocturne")
}

fn art_dir() -> PathBuf {
    root().join("art")
}

fn list_dir() -> PathBuf {
    root().join("lists")
}

/// Spotify art URLs end in a content-addressed id, so the last path segment is a stable, unique
/// filename — no hashing needed, and the file is self-invalidating (new art ⇒ new id ⇒ new file).
fn art_path(url: &str) -> Option<PathBuf> {
    let id = url.rsplit('/').next()?;
    if id.is_empty() || id.contains("..") || id.contains('/') {
        return None;
    }
    Some(art_dir().join(id))
}

pub fn art_get(url: &str) -> Option<Vec<u8>> {
    std::fs::read(art_path(url)?).ok()
}

pub fn art_put(url: &str, bytes: &[u8]) {
    let Some(path) = art_path(url) else { return };
    let _ = std::fs::create_dir_all(art_dir());
    // Write-then-rename: a half-written cover must never be read back as a valid image.
    let tmp = path.with_extension("tmp");
    if std::fs::write(&tmp, bytes).is_ok() {
        let _ = std::fs::rename(&tmp, &path);
    }
}

/// `key` is a caller-chosen slug: "liked", "playlists", or a playlist id.
fn list_path(key: &str) -> PathBuf {
    let safe: String = key
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    list_dir().join(format!("{safe}.json"))
}

pub fn list_get<T: DeserializeOwned>(key: &str) -> Option<T> {
    let bytes = std::fs::read(list_path(key)).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Where the trained taste model lives.
pub fn model_path() -> PathBuf {
    root().join("taste-model.json")
}

pub fn list_put<T: Serialize>(key: &str, value: &T) {
    let _ = std::fs::create_dir_all(list_dir());
    let Ok(bytes) = serde_json::to_vec(value) else {
        return;
    };
    let path = list_path(key);
    let tmp = path.with_extension("tmp");
    if std::fs::write(&tmp, bytes).is_ok() {
        let _ = std::fs::rename(&tmp, &path);
    }
}
