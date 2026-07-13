//! Does the taste model actually learn? Train it on two very different "listening runs" and check
//! it ranks candidates differently depending on what was just played.

use nocturne_api::Track;
use nocturne_taste::Taste;

fn t(name: &str, artist: &str, album: &str) -> Track {
    Track {
        uri: format!("spotify:track:{name}"),
        name: name.into(),
        artists: artist.into(),
        album: album.into(),
        duration_ms: 200_000,
        art_url: None,
        popularity: Some(50),
        explicit: Some(false),
    }
}

fn trained() -> Taste {
    let lofi: Vec<Track> = (0..8).map(|i| t(&format!("lofi{i}"), "Kanisan", "Edda")).collect();
    let metal: Vec<Track> = (0..8)
        .map(|i| t(&format!("metal{i}"), "Bring Me The Horizon", "Sempiternal"))
        .collect();
    let mut taste = Taste::new();
    for _ in 0..2 {
        taste.learn_sequence(&lofi);
        taste.learn_sequence(&metal);
    }
    taste
}

fn main() {
    let lofi: Vec<Track> = (0..8).map(|i| t(&format!("lofi{i}"), "Kanisan", "Edda")).collect();
    let metal: Vec<Track> = (0..8)
        .map(|i| t(&format!("metal{i}"), "Bring Me The Horizon", "Sempiternal"))
        .collect();

    let candidates = vec![
        t("cand-metal", "Bring Me The Horizon", "Sempiternal"),
        t("cand-lofi", "Kanisan", "Edda"),
        t("cand-other", "Hans Zimmer", "Interstellar"),
    ];

    let mut a = trained();
    println!("trained sequences: {}, warm: {}", a.trained_sequences(), a.is_warm());
    for tr in &lofi[..4] {
        a.observe(tr);
    }
    let ra = a.rank(candidates.clone());
    println!("\nafter listening to LOFI → radio order:");
    for (i, x) in ra.iter().enumerate() {
        println!("  {}. {} — {}", i + 1, x.name, x.artists);
    }

    let mut b = trained();
    for tr in &metal[..4] {
        b.observe(tr);
    }
    let rb = b.rank(candidates.clone());
    println!("\nafter listening to METAL → radio order:");
    for (i, x) in rb.iter().enumerate() {
        println!("  {}. {} — {}", i + 1, x.name, x.artists);
    }

    println!(
        "\nVERDICT: context changed the top pick: {}",
        if ra[0].artists != rb[0].artists { "YES" } else { "NO" }
    );
}
