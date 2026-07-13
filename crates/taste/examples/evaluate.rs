//! Held-out evaluation on Camden's REAL library and REAL Spotify analysis.
//!
//! The question that matters: given a listening run from playlist A, does the model rank a
//! genuinely-from-A track above a random track from a different playlist? Chance is 50%.
//!
//! Uses the model file the app already built, so this scores the embedding that actually ships.

use nocturne_api::{AudioFeatures, Track};
use nocturne_taste::{track_id, Taste};
use std::collections::HashMap;

fn cache_root() -> std::path::PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".cache")
        });
    base.join("nocturne")
}

fn load_list(key: &str) -> Option<Vec<Track>> {
    let p = cache_root().join("lists").join(format!("{key}.json"));
    serde_json::from_slice(&std::fs::read(p).ok()?).ok()
}

/// Deterministic shuffle-free picker so the run is reproducible.
fn pick<'a, T>(v: &'a [T], i: usize) -> &'a T {
    &v[i % v.len()]
}

fn main() {
    // Every cached playlist with enough tracks to hold out from.
    let playlists: Vec<nocturne_api::Playlist> = load_list("playlists")
        .map(|_: Vec<Track>| Vec::new())
        .unwrap_or_default();
    let _ = playlists;

    let meta: Vec<nocturne_api::Playlist> = {
        let p = cache_root().join("lists").join("playlists.json");
        serde_json::from_slice(&std::fs::read(p).expect("playlists cache")).expect("parse")
    };

    let mut corpora: Vec<(String, Vec<Track>)> = Vec::new();
    for pl in &meta {
        if let Some(tracks) = load_list(&pl.id) {
            if tracks.len() >= 12 {
                corpora.push((pl.name.clone(), tracks));
            }
        }
    }
    if corpora.len() < 2 {
        eprintln!("need >=2 cached playlists with 12+ tracks; open a few in the app first");
        return;
    }

    // The real model file → real Spotify features.
    let model = Taste::load(&cache_root().join("taste-model.json")).expect("trained model");
    let features: HashMap<String, AudioFeatures> = model.features().clone();
    let with_analysis = |t: &Track| features.contains_key(track_id(&t.uri));

    println!(
        "corpora: {}, tracks with real Spotify analysis: {}\n",
        corpora.len(),
        features.len()
    );

    let mut hits = 0usize;
    let mut trials = 0usize;

    for (i, (name, tracks)) in corpora.iter().enumerate() {
        // Hold out the tail of this playlist; train on everything else.
        let split = tracks.len() * 2 / 3;
        let (train, held) = tracks.split_at(split);
        let held: Vec<&Track> = held.iter().filter(|t| with_analysis(t)).collect();
        if held.is_empty() {
            continue;
        }

        let mut taste = Taste::new();
        taste.add_features(features.clone());
        // Train on every OTHER playlist in full, and this one's head only.
        for (j, (_, other)) in corpora.iter().enumerate() {
            if i == j {
                continue;
            }
            taste.learn_corpus(&format!("c{j}"), other);
        }
        taste.learn_corpus(&format!("c{i}-head"), train);

        // Context: the last few tracks of the training head — i.e. "he's been listening to this".
        for t in train.iter().rev().take(5).rev() {
            taste.observe(t);
        }

        // Candidate pair: one held-out track from THIS playlist, one from a different playlist.
        for (k, correct) in held.iter().enumerate().take(6) {
            let other_idx = (i + 1 + k) % corpora.len();
            if other_idx == i {
                continue;
            }
            let distractors: Vec<&Track> = corpora[other_idx]
                .1
                .iter()
                .filter(|t| with_analysis(t))
                .collect();
            if distractors.is_empty() {
                continue;
            }
            let wrong = pick(&distractors, k);

            let ranked = taste.rank(vec![(*wrong).clone(), (*correct).clone()]);
            let won = ranked[0].uri == correct.uri;
            hits += won as usize;
            trials += 1;
        }
        println!("{:<38} held={:<3} running {}/{}", name, held.len(), hits, trials);
    }

    let pct = 100.0 * hits as f32 / trials.max(1) as f32;
    println!("\n=== {hits}/{trials} correct = {pct:.1}%  (chance = 50%) ===");
}
