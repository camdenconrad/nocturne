//! Runic-native sink: implements librespot's `Sink` against the audio daemon Rune runs.
//!
//! Runic exposes the PipeWire native protocol (that's what its M3 responder answers),
//! so this crate talks to that socket. When runic grows a first-party client crate,
//! only this crate changes — nothing above the `Sink` boundary moves.

use librespot_playback::audio_backend::{Sink, SinkError, SinkResult};
use librespot_playback::convert::Converter;
use librespot_playback::decoder::AudioPacket;

pub struct RunicSink {
    open: bool,
}

impl RunicSink {
    pub fn new() -> Self {
        Self { open: false }
    }
}

impl Default for RunicSink {
    fn default() -> Self {
        Self::new()
    }
}

impl Sink for RunicSink {
    fn start(&mut self) -> SinkResult<()> {
        // TODO(M1): connect to $XDG_RUNTIME_DIR/pipewire-0 (served by runic),
        // negotiate a 44.1kHz/f32 stream node.
        self.open = true;
        tracing::info!("RunicSink started");
        Ok(())
    }

    fn stop(&mut self) -> SinkResult<()> {
        self.open = false;
        tracing::info!("RunicSink stopped");
        Ok(())
    }

    fn write(&mut self, packet: AudioPacket, converter: &mut Converter) -> SinkResult<()> {
        if !self.open {
            return Err(SinkError::NotConnected("RunicSink not started".into()));
        }
        let samples = packet
            .samples()
            .map_err(|e| SinkError::OnWrite(e.to_string()))?;
        // TODO(M1): hand the converted frames to the stream node instead of dropping them.
        let _pcm = converter.f64_to_f32(samples);
        Ok(())
    }
}
