# Nocturne

A native Spotify client for **Rune**, Camden's own Wayland compositor — Rust and wgpu, no Electron, no CEF, no webview.

Spotify's official client is a CEF app with Widevine DRM; on a hand-built compositor that's dead weight. [librespot](https://github.com/librespot-org/librespot) reimplements Spotify's client protocol natively in Rust (Premium required), which makes a genuinely native client possible. Nocturne is that client, plus a taste model that learns what you actually listen to and builds radios from a mood typed in plain English.

## What it does

- **Full-screen album art as the interface** — the cover fills the window over a blurred, gradient-scrimmed backdrop of itself; the library slides in over the top, and playing a track drops you straight back out.
- **Mood radio** — type "chill winter lofi" and get a station: seeded from your library, expanded via Spotify's internal station service, filtered against what you already own, and saved as a real auto-named playlist.
- **Learns from listening** — every play is recorded with how much of it you actually heard; finishes pull the model toward a track, skips push it away.
- **Real ESRGAN upscaling** — Spotify's largest cover art is 640×640; `realesrgan-ncnn-vulkan` runs the actual model on the GPU (640 → 2560 in ~3s, cached after).
- **A Rune citizen** — MPRIS so `rtray` and media keys work, audio delivered straight into `runic`, icon following the Rune palette.

## Architecture

```
crates/
  app/       eframe/wgpu UI, MPRIS, disk cache, vector icons, ESRGAN
  session/   librespot: auth, playback, playlists, radio, audio features
  api/       Spotify Web API: search, liked songs, playlist list
  sink/      audio out — librespot's PulseAudio backend, into runic
  taste/     the learned model: embeddings, ranking, mood, trajectories
```

The UI thread never blocks: a backend thread owns a tokio runtime and the player, the UI sends commands and reads a shared `State`, and every fetch runs as its own task rather than a shared command loop.

`runic` **is** the PulseAudio server, so librespot's stock `pulseaudio-backend` lands directly in it — no custom protocol code, no hand-rolled resampling.

**The Spotify Web API is largely locked down** for apps registered after 2024 — track lists, audio features, recommendations, and library/playlist writes all 403 or 404 even with every scope granted. Nocturne routes around this through librespot's internal spclient protocol wherever an internal path exists; where none does (library and playlist writes), likes and playlist adds are stored **locally on disk** and merged into the display, with the UI honest that they don't sync back to Spotify.

**Taste model:** track selection is nearest-neighbour search over a 56-dimensional acoustic embedding (Spotify's audio features weighted ahead of artist/album identity, key embedded on a circle). Mood *trajectory* uses a `TensorSequenceTree` (from Camden's [WatchTower](https://github.com/camdenconrad/WatchTower)), reserved for session-shaped, time-ordered prediction — held-out evaluation showed it actively hurts accuracy when misapplied to unordered playlists (69% → 0–6%), which is why playlist ranking uses plain acoustic similarity instead.

## Building

Requires a Spotify **Premium** account and a free app registration at the [Spotify developer dashboard](https://developer.spotify.com/dashboard) (redirect URI `http://127.0.0.1:8898/login`, Web API scope). Copy the client ID into `.env` (`cp .env.example .env`) — auth is PKCE, no client secret needed.

```sh
cargo run -p nocturne-app     # or just: nocturne
```

Optionally set `ANTHROPIC_API_KEY` so mood phrases are parsed by Claude Haiku instead of a built-in ~20-word vocabulary.

librespot must build from its `dev` git branch (the crates.io `0.6` release fails to compile against a clean index), pinned in `vendor/librespot` with one local patch so a sink drop failure doesn't kill the whole app mid-stream.

## Status

**Working:** playback, library, playlists, search, mood radio with discovery, learned autoplay, session resume, local likes and playlist adds, MPRIS/rtray, ESRGAN upscaling, audio caching.

**Unverified:** saving a radio to Spotify as a real playlist — never exercised against a live account due to rate limiting during development; falls back to a local-only playlist if Spotify refuses the write.

**Not done:** offline mode, podcasts, lyrics, and Spotify Connect (Nocturne as a Connect target).

## License

MIT
