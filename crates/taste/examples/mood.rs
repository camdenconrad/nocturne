//! Does mood radio actually work on Camden's real library + real Spotify analysis?

use nocturne_api::Track;
use nocturne_taste::{mood, track_id, Taste};

fn root() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/nocturne")
}
fn load(key: &str) -> Option<Vec<Track>> {
    serde_json::from_slice(&std::fs::read(root().join("lists").join(format!("{key}.json"))).ok()?).ok()
}

fn main() {
    let taste = Taste::load(&root().join("taste-model.json")).expect("model");
    let feats = taste.features();

    // The pool: everything we have analysis for.
    let mut pool: Vec<Track> = load("liked").unwrap_or_default();
    let meta: Vec<nocturne_api::Playlist> =
        serde_json::from_slice(&std::fs::read(root().join("lists/playlists.json")).unwrap()).unwrap();
    for p in &meta {
        if let Some(ts) = load(&p.id) {
            pool.extend(ts);
        }
    }
    pool.sort_by(|a, b| a.uri.cmp(&b.uri));
    pool.dedup_by(|a, b| a.uri == b.uri);
    let pool: Vec<Track> = pool
        .into_iter()
        .filter(|t| feats.contains_key(track_id(&t.uri)))
        .collect();
    println!("pool: {} tracks with real analysis\n", pool.len());

    for phrase in [
        "chill winter lofi vibes",
        "hype workout metal",
        "sad rainy acoustic",
        "happy summer dance party",
    ] {
        let (m, understood) = mood::parse(phrase);
        let target = mood::acoustic_vec(&m.to_features());
        let picks = taste.nearest_mood(&pool, &target, 5);
        println!("“{phrase}”  (understood={understood})");
        println!(
            "   target: energy={:.2} valence={:.2} acoustic={:.2} instr={:.2} {:.0}bpm",
            m.energy, m.valence, m.acousticness, m.instrumentalness, m.tempo
        );
        for t in &picks {
            let f = &feats[track_id(&t.uri)];
            println!(
                "   → {:<34} {:<24} e={:.2} v={:.2} a={:.2} {:.0}bpm",
                t.name.chars().take(32).collect::<String>(),
                t.artists.chars().take(22).collect::<String>(),
                f.energy, f.valence, f.acousticness, f.tempo
            );
        }
        println!();
    }
}
