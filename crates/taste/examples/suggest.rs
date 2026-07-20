//! Does the Claude request shape actually work? One live call, printing only track names.
//!
//! This exists because the request combines things that are easy to get subtly wrong — structured
//! outputs, adaptive thinking, and an OAuth-vs-API-key credential — and every one of those failure
//! modes is silent: `suggest_tracks` returns an empty vec and the app quietly falls back to
//! keyword search. Without this you cannot tell "no key" from "malformed request".
//!
//!     cargo run -p nocturne-taste --example suggest -- "sad 90s shoegaze for a rainy drive"
//!
//! Prints suggestions only. Nothing here touches Spotify, so nothing is resolved or played.

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let query = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let query = if query.trim().is_empty() {
        "sad 90s shoegaze for a rainy drive".to_string()
    } else {
        query
    };

    match nocturne_taste::auth::credential() {
        // Which source resolved, never the value.
        Some(nocturne_taste::auth::Credential::ApiKey(_)) => println!("auth: ANTHROPIC_API_KEY"),
        Some(nocturne_taste::auth::Credential::Oauth(_)) => println!("auth: OAuth token"),
        None => {
            eprintln!("auth: no credential — set ANTHROPIC_API_KEY, sign in with Claude Code, or `ant auth login`");
            std::process::exit(1);
        }
    }

    println!("query: {query}\n");
    let started = std::time::Instant::now();
    let picks = nocturne_taste::llm::suggest_tracks(&query, 12).await;
    let elapsed = started.elapsed();

    if picks.is_empty() {
        eprintln!("\nNo suggestions — the request failed. Re-run with RUST_LOG=warn for the reason.");
        std::process::exit(1);
    }
    for (i, s) in picks.iter().enumerate() {
        println!("{:>2}. {} — {}", i + 1, s.artist, s.title);
    }
    println!("\n{} suggestions in {:.1}s", picks.len(), elapsed.as_secs_f32());
}
