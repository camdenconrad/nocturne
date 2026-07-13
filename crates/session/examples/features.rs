//! Proof: real Spotify audio features, off the internal service, feeding the model.
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().with_env_filter("warn").init();
    let h = nocturne_session::NocturneHandle::login(nocturne_sink::make_sink()?).await?;
    let api = nocturne_api::Client::new(h.web_token().await?);

    let tracks = api.search_tracks("bring me the horizon throne", 1).await?;
    let calm = api.search_tracks("kanisan edda", 1).await?;
    for t in tracks.iter().chain(calm.iter()) {
        let id = nocturne_taste::track_id(&t.uri).to_string();
        let f = h.audio_features(&id).await?;
        println!(
            "{:<28} energy={:.2} valence={:.2} dance={:.2} acoustic={:.2} tempo={:.0}bpm key={} {}",
            t.name, f.energy, f.valence, f.danceability, f.acousticness, f.tempo, f.key,
            if f.mode == 1 { "major" } else { "minor" }
        );
    }
    Ok(())
}
