//! Audio out for Nocturne — straight into runic, no PipeWire daemon in the path.
//!
//! The obvious-looking approach is to hand-roll a `Sink` that speaks runic's wire protocol. Don't:
//! runic already *is* the PulseAudio server (runic-m3 owns `$XDG_RUNTIME_DIR/pulse/native` as well
//! as `pipewire-0`), so librespot's stock `pulseaudio-backend` lands in runic with no protocol code
//! of our own and none of the resampling/format-negotiation bugs a fresh implementation would earn.
//! "Runic-native" is satisfied by *runic being the server*, not by inventing a private transport.
//!
//! Rate: runic's mixer bus is 48 kHz and it band-limit-resamples per stream, so handing it
//! librespot's native 44.1 kHz is correct — do not pre-convert here and stack two resamplers.

use librespot_playback::audio_backend::{self, Sink, SinkBuilder};
use librespot_playback::config::AudioFormat;

/// PulseAudio-protocol name librespot connects to. Empty string = "the default server", which
/// resolves through `$PULSE_SERVER` / `$XDG_RUNTIME_DIR/pulse/native` — i.e. runic.
const DEFAULT_DEVICE: Option<String> = None;

/// librespot's `S16` is the format runic's pulse path is happiest with; `F32` is accepted too but
/// buys nothing here since the source is already lossy.
pub const FORMAT: AudioFormat = AudioFormat::S16;

#[derive(Debug, thiserror::Error)]
pub enum SinkError {
    #[error("librespot has no pulseaudio backend compiled in — check the `pulseaudio-backend` feature")]
    BackendMissing,
}

/// librespot's pulse backend reads its stream naming out of the environment and defaults to a bare
/// "stream", which is what shows up in the mixer next to every other app. Name it before the first
/// sink is built, or Nocturne is an anonymous row in rmix.
fn name_stream() {
    if std::env::var_os("PULSE_PROP_application.name").is_none() {
        std::env::set_var("PULSE_PROP_application.name", "Nocturne");
    }
    if std::env::var_os("PULSE_PROP_stream.description").is_none() {
        std::env::set_var("PULSE_PROP_stream.description", "Nocturne");
    }
}

/// Build the sink factory the `Player` wants. Fails loudly at startup rather than silently
/// falling back to a different audio path — Nocturne is runic-only by design.
pub fn runic_sink_builder() -> Result<SinkBuilder, SinkError> {
    name_stream();
    audio_backend::find(Some("pulseaudio".to_string())).ok_or(SinkError::BackendMissing)
}

/// Convenience: the closure `Player::new` takes.
pub fn make_sink() -> Result<impl FnMut() -> Box<dyn Sink> + Send + 'static, SinkError> {
    let builder = runic_sink_builder()?;
    Ok(move || builder(DEFAULT_DEVICE.clone(), FORMAT))
}
