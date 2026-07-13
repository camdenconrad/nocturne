# Nocturne

A native Spotify client for **Rune** — Camden's own Wayland compositor and shell. No Electron, no
CEF, no webview. Rust, wgpu, and a direct line into `runic`.

It plays Spotify, learns what you actually listen to, and builds radios from a mood you type in
plain English.

```sh
cargo run -p nocturne-app     # or just: nocturne
```

---

## Why this exists

The official Spotify client is a CEF app with Widevine DRM. On a hand-built Wayland compositor that
is dead weight at best. [librespot](https://github.com/librespot-org/librespot) reimplements
Spotify's client protocol natively in Rust (Premium required), which makes a genuinely native client
possible.

Nocturne is that client, plus the parts Spotify won't give you.

---

## What it does

**Full screen is the app.** The album art *is* the interface: the cover fills the window over a
blurred, gradient-scrimmed backdrop of itself, transport centred beneath. The library slides in over
the top (`☰` / `L`); picking a playlist drops you into the list view — the only place a track list
appears at all — and playing anything drops you straight back. A `<` beside the hamburger jumps to
whatever list is *currently playing*, which is not necessarily the one you're looking at.

**Mood radio.** Type "chill winter lofi" and get a station. It seeds from the tracks in your library
that best match the mood *and your taste*, pulls candidates from **outside** your library via
Spotify's station service, analyses them, drops anything you already own or have played, and ranks
what's left. The result is saved as a real, auto-named playlist ("Cozy Winter Lofi") that lives on
disk until the next radio replaces it — and can be pushed to Spotify as a permanent playlist.

**It learns.** Every play is recorded with how much of it you actually heard. Tracks you finish pull
the model toward them; tracks you skip push it away. Radios use both.

**Real ESRGAN upscaling.** Spotify's largest cover is 640×640, which is soft across a 4K panel.
`realesrgan-ncnn-vulkan` runs the real model on the GPU: 640 → 2560 in ~3s, cached after.

**It's a Rune citizen.** MPRIS means `rtray` shows it as the active player and media keys work.
Audio goes straight into `runic`. The icon follows the Rune palette.

---

## Setup

You need a Spotify **Premium** account and a (free) app registration.

1. <https://developer.spotify.com/dashboard> → **Create app**.
2. Redirect URI must be exactly `http://127.0.0.1:8898/login`.
   Use `127.0.0.1`, **not** `localhost` — Spotify rejects `localhost` on new apps, and the failure is
   an unhelpful `INVALID_CLIENT` at *login* rather than at save time.
3. Tick **Web API**, save, copy the **Client ID** from Settings.
4. `cp .env.example .env` and paste it in. No client secret — the OAuth flow is PKCE.

First launch opens a browser once; after that it signs in silently.

**Optional:** set `ANTHROPIC_API_KEY` and mood phrases are parsed by Claude Haiku, which handles
arbitrary phrasing ("music for staring out a train window in November"). Without it, a built-in
~20-word vocabulary covers the common vibes and says so when it doesn't understand.

---

## Architecture

```
crates/
  app/       eframe/wgpu UI, MPRIS, disk cache, vector icons, ESRGAN
  session/   librespot: auth, playback, playlists, radio, audio features
  api/       Spotify Web API: search, liked songs, playlist list
  sink/      audio out — librespot's PulseAudio backend, into runic
  taste/     the learned model: embeddings, ranking, mood, trajectories
```

The UI thread never blocks. A backend thread owns a tokio runtime and the player; the UI sends
commands and reads a shared `State`. Every fetch runs as its own task — putting them on the command
loop meant one slow load (the 2000-track Liked Songs sweep) blocked every later click behind it.

### Audio: why there is no custom protocol code

`runic` **is** the PulseAudio server — `runic-m3` owns `$XDG_RUNTIME_DIR/pulse/native` as well as
`pipewire-0`. So librespot's stock `pulseaudio-backend` lands directly in runic with no protocol code
of our own, and none of the resampling or format-negotiation bugs a hand-rolled sink would earn.
"Runic-native" is satisfied by *runic being the server*, not by inventing a private transport.

Runic's mixer bus is 48 kHz and band-limit-resamples per stream, so we hand it librespot's native
44.1 kHz rather than pre-converting and stacking two resamplers.

---

## The Spotify lockdown, and how we get around it

**This is the most important thing to understand about this codebase.**

Spotify locked down its Web API for apps registered after 2024. For an app like ours these all return
403 (or 404) — *with every scope granted, for your own data*:

| Endpoint | Status | What we do instead |
|---|---|---|
| `/v1/playlists/{id}/tracks` | 403 | librespot's internal protocol |
| `/v1/tracks?ids=` | 403 | batched extended-metadata (internal) |
| `/v1/audio-features` | 403 | `/audio-attributes/v1/audio-features/{id}` (internal) |
| `/v1/audio-analysis` | 403 | same service |
| `/v1/recommendations` | **404** | `radio-apollo` station service (internal) |
| `PUT /v1/me/tracks` | 403 | local likes (below) |
| `/v1/playlists/{id}` | 200 — **track list stripped out** | — |

**The lesson: when the Web API forbids something, check the internal spclient path before giving
up.** Spotify's own client doesn't use the Web API for this, and librespot speaks its protocol.
(One dead end recorded so nobody repeats it: the `AUDIO_ATTRIBUTES` *extension kind* route 404s —
only the REST path works.)

Two things still work on the Web API and are used for exactly that: **search** and **liked songs**.
Search has an undocumented cap of `limit=10` — the docs say 50, and anything over 10 is a bare
`400 Invalid limit`.

**Writes are blocked entirely.** Library *and* playlist writes are 403, and no internal endpoint
exists (probed). So likes and playlist additions are **local**: stored on disk, merged into the
display, and honest about it in the UI. They do not sync back to Spotify.

---

## The taste model

Track selection is nearest-neighbour in acoustic space. Mood *trajectory* is a `TensorSequenceTree`
(WatchTower — Camden's own sequence model).

That split was **measured, not assumed**. Held-out evaluation on the real library
(`crates/taste/examples/diagnose.rs`; chance = 50%):

| ranker | accuracy |
|---|---|
| acoustic similarity to recent listening | **69%** |
| `TensorSequenceTree::predict_next` (all 4 modes) | **0–6%** |

The tree isn't broken — it was asked the wrong question. It predicts the next item in a *sequence*,
and **a playlist is an unordered bag, not a sequence**. Trained on playlist "order" it learns nothing
sequential and falls back to the globally most common state (which lives in the biggest playlist),
making it actively *anti*-correlated. Hence worse than chance.

So the tree does what it is actually built for: a listening *session* genuinely is a time-ordered
path through acoustic space (things build, things wind down), which is exactly its shape. See
`crates/taste/src/mood.rs`.

Also measured: recency-decay weighting on the similarity ranker **hurt** (69% → 56%), because a
weighted max collapses toward "similar to the single most recent track" instead of "fits the run".
It is deliberately absent. Don't re-add it.

**The embedding** puts Spotify's real audio features (energy, valence, danceability, tempo,
acousticness…) at the front of a 56-dim vector, with artist/album identity behind them at 0.35 weight
— so the model can still say "more of this artist" while acoustics carry the similarity. Key is
embedded on a **circle**, because a linear 0–11 would tell the model that B and C are maximally far
apart when they're adjacent in pitch space. Hashing is FNV, never `DefaultHasher` (randomly seeded
per process — it would silently change the embedding space between launches).

**The model file** (`~/.cache/nocturne/taste-model.json`) is versioned against the embedding layout.
A model whose `DIMS`/version don't match is **discarded and retrained** — a stale model silently
reinterprets every stored tensor. Training is idempotent per corpus; without that, every launch
re-learned the same playlists and permanently overweighted them against real listening.

Radios are **sampled**, not top-N: without replacement, weighted `1/(rank+2)`. Strict top-N gives the
same station forever; a pure shuffle throws the ranking away.

---

## Caching

Everything under `~/.cache/nocturne/`:

| path | what |
|---|---|
| `art/` | covers, keyed by content-addressed CDN id (self-invalidating) |
| `art-hires/` | ESRGAN upscales |
| `audio/` | encrypted Ogg, 8 GiB LRU — replays hit disk, not the network |
| `lists/` | library, playlists, listening history, current radio playlist, session |
| `taste-model.json` | the trained model |
| `oauth.json` | the token (0600) |

Cached views paint instantly and refresh behind the visible UI.

---

## Gotchas

Hard-won, mostly the expensive way.

**librespot must come from the `dev` git branch.** The crates.io 0.6 release's build script calls the
vergen 8 API but resolves vergen-lib 9, so it fails to compile outright. `dev` is 0.8 and its API
differs (`Player::load` takes a `SpotifyUri`, `SoftMixer::open` returns a `Result`).

**`default-features = false` on librespot-playback also drops `native-tls`**, which
`librespot-oauth` `compile_error!`s without. Add it back explicitly.

**librespot-oauth is blocking and builds its own tokio runtime.** Called from async it panics on
drop. Wrap in `spawn_blocking`.

**Persist the whole OAuth token, not just the refresh token.** Refresh tokens *rotate on use*, so
refreshing every launch invalidates the stored copy and forces browser logins. Store expiry as a
**unix timestamp** — librespot's `Instant` is process-relative and meaningless across restarts.

**Set the mixer volume explicitly.** librespot's `SoftMixer` does not guarantee a sane starting
level, and a mixer at zero yields a stream that exists, appears in the mixer, reports `Playing` —
and is silent.

**Don't fetch track metadata one id at a time.** `Track::get` per track stampedes a 300-track
playlist into a session-wide rate limit that then poisons every playlist opened afterwards. Use the
batched extended-metadata endpoint (100 uris/request).

**The analysis service throttles.** At 8-way concurrency ~60% of requests silently failed (859 of
2251 tracks). With retry-and-backoff: 2593 of 2599.

**Spotify's rate limits come in two flavours.** A short one (seconds) is a "slow down" worth waiting
out. A long one — it will hand back `Retry-After: 80990`, i.e. **22 hours** — is a penalty box, and
sleeping through it is not something an app can do. Long limits fail fast to the disk cache.

**egui cannot render colour emoji.** epaint has no COLR/CBDT support: `NotoColorEmoji` draws nothing,
monochrome emoji draw as flat white blobs. `crates/app/src/emoji.rs` parses the CBDT PNGs out of the
font with `ttf-parser` and paints them inline as textures. Guard: exclude ASCII first — the font
carries bitmaps for *digits* (keycap bases), so "Fall lofi 2026" renders as "2 0 2 6".

**`std::sync::Mutex` is not reentrant**, and `if let Some(x) = m.lock().unwrap().field()` holds the
guard for the whole body. Locking again inside it deadlocks against itself — and freezes the entire
app, MPRIS included.

**Never hold the model lock across a disk write.** The analysis backfill serialized 6 MB under the
lock playback needed; pressing play stalled for ~10 seconds.

**MPRIS `xesam:artist` must be an array of strings.** `rtray` does `Vec::<String>::try_from`; a bare
string silently yields no artist.

**A `Slider` takes its width from `spacing.slider_width`** — `add_sized` does not stretch it.

**UI icons are drawn vectors, not emoji** (`crates/app/src/icons.rs`). Emoji glyphs inherit the
font's weight, baseline and metrics, differ per platform, and turn to mush at 14px.

---

## Status

**Working:** playback, library, playlists, search, mood radio with discovery, learned autoplay,
session resume, local likes and playlist adds, MPRIS/rtray, ESRGAN upscaling, audio caching.

**Unverified:** *saving a radio to Spotify as a real playlist.* Spotify rate-limited us during
development and this path has never been exercised against a live account. If Spotify refuses the
write (as it does for the library), the local playlist is kept and the UI says so.

**Not done:** offline mode, podcasts, lyrics, Spotify Connect (Nocturne as a Connect *target*).
