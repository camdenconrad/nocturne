//! Proves the sink path end-to-end: builds the *same* sink Nocturne's Player uses and pushes a
//! 440 Hz tone through it. If this is audible, librespot → runic works and the rest is UI.
//!
//!     cargo run -p nocturne-sink --example tone

use librespot_playback::audio_backend::Sink;
use librespot_playback::convert::Converter;
use librespot_playback::decoder::AudioPacket;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let mut mk = nocturne_sink::make_sink()?;
    let mut sink: Box<dyn Sink> = mk();
    sink.start()?;

    let mut conv = Converter::new(None);
    // librespot decodes at 44.1k stereo; runic band-limits to its 48k bus itself.
    const RATE: f64 = 44_100.0;
    let mut phase: f64 = 0.0;
    let step = 2.0 * std::f64::consts::PI * 440.0 / RATE;

    for _ in 0..(3 * 44_100 / 1024) {
        let mut buf = Vec::with_capacity(1024 * 2);
        for _ in 0..1024 {
            let s = (phase.sin()) * 0.25;
            phase += step;
            buf.push(s); // L
            buf.push(s); // R
        }
        sink.write(AudioPacket::Samples(buf), &mut conv)?;
    }

    sink.stop()?;
    println!("tone finished — if you heard 3s of 440 Hz, librespot → runic is live");
    Ok(())
}
