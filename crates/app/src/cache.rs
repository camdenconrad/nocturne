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

/// Staging for the upscaler, which is file-in/file-out. Nothing here outlives a pass: the 8000²
/// result is a RAM-only artifact (see [`crate::upscale`]), so this holds temp files for seconds,
/// not a cache.
pub fn scratch_dir() -> PathBuf {
    root().join("scratch")
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

/// Spotify's CDN encodes the image size in the id prefix: `ab67616d00004851` is 64px,
/// `ab67616d00001e02` is 300px, `ab67616d0000b273` is 640px — and `ab67616d000082c1` is the
/// **original master the label uploaded**, typically 1800–2000px.
///
/// The Web API's `images` array stops at 640, so the master is only reachable by rewriting the
/// prefix ourselves. It is real detail, not an upscale, and it is free — which makes it strictly
/// better than anything we can invent from the 640.
const CDN_640: &str = "ab67616d0000b273";
const CDN_MASTER: &str = "ab67616d000082c1";

fn master_url(url: &str) -> Option<String> {
    url.contains(CDN_640)
        .then(|| url.replace(CDN_640, CDN_MASTER))
}

/// Fetch the best art the CDN will give us: the original master if it has one, else the URL we were
/// handed. Returns the URL actually used — callers key their caches and textures by it, so it has
/// to be the one that produced the bytes.
///
/// Not every album has a master (a few 404), hence the fallback rather than a blind rewrite.
pub async fn art_fetch_best(url: &str) -> Option<(String, Vec<u8>)> {
    if let Some(master) = master_url(url) {
        if let Some(bytes) = art_fetch(&master).await {
            return Some((master, bytes));
        }
    }
    art_fetch(url).await.map(|bytes| (url.to_string(), bytes))
}

/// Fetch a cover, streaming it straight to disk.
///
/// The response body is written chunk-by-chunk into a temp file and renamed into place only once
/// complete, so a cover is never buffered whole in RAM before it lands, and a dropped connection
/// leaves no half-file that a later run would read back as a valid image.
///
/// A disk hit skips the network entirely. Returns the cached bytes — the UI needs them to build a
/// texture, and at 640px that's the one small copy we can't avoid.
pub async fn art_fetch(url: &str) -> Option<Vec<u8>> {
    if let Some(bytes) = art_get(url) {
        return Some(bytes);
    }
    let path = art_path(url)?;
    let _ = std::fs::create_dir_all(art_dir());
    let tmp = path.with_extension("tmp");

    let streamed = async {
        use futures_util::StreamExt;
        use std::io::Write;

        let resp = reqwest::get(url).await.ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let mut file = std::fs::File::create(&tmp).ok()?;
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            file.write_all(&chunk.ok()?).ok()?;
        }
        file.flush().ok()?;
        Some(())
    }
    .await;

    if streamed.is_none() {
        let _ = std::fs::remove_file(&tmp);
        return None;
    }
    std::fs::rename(&tmp, &path).ok()?;
    std::fs::read(&path).ok()
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
