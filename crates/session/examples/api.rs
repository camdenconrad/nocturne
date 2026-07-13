//! Headless check of the Web API half: mints a token from the cached session, then hits search,
//! liked songs and playlists — the three calls the UI depends on.
//!
//!     cargo run -p nocturne-session --example api -- "boards of canada"

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().with_env_filter("warn").init();
    let query = std::env::args().nth(1).unwrap_or_else(|| "aphex twin".into());

    let handle = nocturne_session::NocturneHandle::login(nocturne_sink::make_sink()?).await?;
    let api = nocturne_api::Client::new(handle.web_token().await?);

    let hits = api.search_tracks(&query, 5).await?;
    println!("search “{query}” → {} hits", hits.len());
    for t in &hits {
        println!("  {} — {} [{}]", t.name, t.artists, nocturne_api::fmt_duration(t.duration_ms));
    }

    let liked = api.saved_tracks(5).await?;
    println!("\nliked songs → {} shown", liked.len());
    for t in &liked {
        println!("  {} — {}", t.name, t.artists);
    }

    let pls = api.playlists(500).await?;
    println!("\nplaylists → {}", pls.len());
    for p in &pls {
        println!("  {} ({:?})", p.name, p.tracks);
    }

    if let Some(url) = hits.first().and_then(|t| t.art_url.clone()) {
        let bytes = api.fetch_art(&url).await?;
        println!("\nart fetch → {} bytes", bytes.len());
    }
    Ok(())
}
