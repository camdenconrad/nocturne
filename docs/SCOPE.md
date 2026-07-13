# Nocturne — native Spotify client for Rune

## Goal
A daily-driver Spotify client that is a first-class Rune citizen: eframe/wgpu UI in the
livewall-uikit autumn theme, audio delivered straight to runic, controls surfaced in the
rune shell.

## Why
The official client is a CEF webview; on a hand-built Wayland compositor it's dead weight.
librespot (Premium account, already held) makes a fully native client practical.

## In scope (v1)
- [ ] OAuth login via `librespot-oauth` (`NOCTURNE_CLIENT_ID` env; token cached on disk)
- [ ] Playback: load/play/pause/seek/queue through `librespot-playback`
- [ ] Library + playlists + search UI (Web API / `librespot-metadata`)
- [ ] `nocturne-sink`: librespot `Sink` speaking the PipeWire native protocol runic serves
      (runic has no client crate yet — this crate is the only thing that changes when it does)
- [ ] Spotify Connect device via `librespot-connect` + zeroconf discovery
- [ ] MPRIS interface so rtray/rnotify pick up track info + media keys

## Out of scope (v1)
- Offline caching / downloads
- Podcasts and audiobooks
- Lyrics, friend activity, social features
- Fallback audio backends (ALSA/Pulse) — runic-native only, by decision 2026-07-13

## Owner
Camden (@solo)

## Acceptance
- Sign in once, restart the app, still signed in (cached credentials)
- Search a track, click it, audio plays through runic end-to-end
- Phone Spotify app sees "Nocturne" as a Connect device and can fling audio to it
- rnotify shows track-change popups; media keys work via rune

## Refs
- librespot: https://github.com/librespot-org/librespot
- runic repo: ~/RustroverProjects/runic (M3 responder = the protocol surface the sink targets)
- uikit theme: ~/RustroverProjects/livewall-studio/crates/uikit
