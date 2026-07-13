//! Do playlists open over librespot's internal protocol (where the Web API 403s)?

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().with_env_filter("warn").init();
    let handle = nocturne_session::NocturneHandle::login(nocturne_sink::make_sink()?).await?;
    let api = nocturne_api::Client::new(handle.web_token().await?);

    for p in api.playlists(500).await?.into_iter().take(6) {
        match handle.playlist_tracks(&p.id).await {
            Ok(t) => {
                println!("OK  {:>3} tracks  {}", t.len(), p.name);
                for tr in t.iter().take(2) {
                    println!("        {} — {}  {}", tr.name, tr.artists,
                        if tr.art_url.is_some() { "[art]" } else { "[no art]" });
                }
            }
            Err(e) => println!("FAIL            {}  → {e}", p.name),
        }
    }
    Ok(())
}
