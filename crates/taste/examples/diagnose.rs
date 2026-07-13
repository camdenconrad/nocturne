//! Is the EMBEDDING bad, or the TREE's prediction? Measure them separately.

use nocturne_api::Track;
use nocturne_taste::{embed, track_id, Taste};
use std::collections::HashMap;

fn root() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/nocturne")
}
fn load(key: &str) -> Option<Vec<Track>> {
    serde_json::from_slice(&std::fs::read(root().join("lists").join(format!("{key}.json"))).ok()?).ok()
}

fn main() {
    let model = Taste::load(&root().join("taste-model.json")).expect("model");
    let feats = model.features().clone();
    let meta: Vec<nocturne_api::Playlist> =
        serde_json::from_slice(&std::fs::read(root().join("lists/playlists.json")).unwrap()).unwrap();

    let mut corpora: Vec<(String, Vec<Track>)> = Vec::new();
    for p in &meta {
        if let Some(ts) = load(&p.id) {
            let ts: Vec<Track> = ts.into_iter().filter(|t| feats.contains_key(track_id(&t.uri))).collect();
            if ts.len() >= 10 {
                corpora.push((p.name.chars().take(24).collect(), ts));
            }
        }
    }
    println!("corpora with analysis: {}\n", corpora.len());

    let vec_of = |t: &Track| embed(t, feats.get(track_id(&t.uri)));

    // --- A. EMBEDDING: are same-playlist tracks closer than cross-playlist ones? ---
    println!("--- embedding quality (cosine) ---");
    let mut intra_all = 0.0;
    let mut inter_all = 0.0;
    let mut ni = 0;
    let mut ne = 0;
    for (i, (name, ts)) in corpora.iter().enumerate() {
        let vs: Vec<_> = ts.iter().take(30).map(&vec_of).collect();
        let mut intra = 0.0;
        let mut n = 0;
        for a in 0..vs.len() {
            for b in a + 1..vs.len() {
                intra += vs[a].cosine_similarity(&vs[b]);
                n += 1;
            }
        }
        let intra = intra / n.max(1) as f32;

        let other = &corpora[(i + 1) % corpora.len()].1;
        let ovs: Vec<_> = other.iter().take(30).map(&vec_of).collect();
        let mut inter = 0.0;
        let mut m = 0;
        for a in &vs {
            for b in &ovs {
                inter += a.cosine_similarity(b);
                m += 1;
            }
        }
        let inter = inter / m.max(1) as f32;
        intra_all += intra; inter_all += inter; ni += 1; ne += 1;
        println!("  {:<26} intra={:.3}  inter={:.3}  margin={:+.3}", name, intra, inter, intra - inter);
    }
    println!("  MEAN margin (intra - inter) = {:+.3}   (want clearly > 0)",
        intra_all / ni as f32 - inter_all / ne as f32);

    // --- B. NEAREST-NEIGHBOUR baseline: rank by similarity to the context centroid ---
    println!("\n--- baseline: centroid-of-context similarity ---");
    let (mut hits, mut trials) = (0, 0);
    for (i, (_, ts)) in corpora.iter().enumerate() {
        let split = ts.len() * 2 / 3;
        let (train, held) = ts.split_at(split);
        let ctx: Vec<_> = train.iter().rev().take(5).map(&vec_of).collect();
        let other = &corpora[(i + 1) % corpora.len()].1;
        for (k, correct) in held.iter().enumerate().take(6) {
            let wrong = &other[k % other.len()];
            let sc = |t: &Track| {
                let v = vec_of(t);
                ctx.iter().map(|c| v.cosine_similarity(c)).fold(f32::MIN, f32::max)
            };
            if sc(correct) > sc(wrong) { hits += 1; }
            trials += 1;
        }
    }
    println!("  {hits}/{trials} = {:.0}%", 100.0 * hits as f32 / trials.max(1) as f32);

    // --- C. THE SHIPPED ranker ---
    println!("\n--- SHIPPED rank() (recency-weighted similarity) ---");
    let (mut hits, mut trials) = (0, 0);
    for (i, (_, ts)) in corpora.iter().enumerate() {
        let mut taste = Taste::new();
        taste.add_features(feats.clone());
        for (j, (_, o)) in corpora.iter().enumerate() {
            if i != j { taste.learn_corpus(&format!("c{j}"), o); }
        }
        let split = ts.len() * 2 / 3;
        let (train, held) = ts.split_at(split);
        taste.learn_corpus(&format!("c{i}h"), train);
        for t in train.iter().rev().take(5).rev() { taste.observe(t); }

        let other = &corpora[(i + 1) % corpora.len()].1;
        for (k, correct) in held.iter().enumerate().take(6) {
            let wrong = &other[k % other.len()];
            let ranked = taste.rank(vec![wrong.clone(), (*correct).clone()]);
            if ranked[0].uri == correct.uri { hits += 1; }
            trials += 1;
        }
    }
    println!("  {hits}/{trials} = {:.0}%", 100.0 * hits as f32 / trials.max(1) as f32);

    // --- D. every tree prediction mode, scored the same way ---
    println!("\n--- tree prediction modes ---");
    for mode in 0..4u8 {
        let (mut hits, mut trials) = (0, 0);
        for (i, (_, ts)) in corpora.iter().enumerate() {
            let mut taste = Taste::new();
            taste.add_features(feats.clone());
            for (j, (_, o)) in corpora.iter().enumerate() {
                if i != j { taste.learn_corpus(&format!("c{j}"), o); }
            }
            let split = ts.len() * 2 / 3;
            let (train, held) = ts.split_at(split);
            taste.learn_corpus(&format!("c{i}h"), train);
            for t in train.iter().rev().take(5).rev() { taste.observe(t); }

            let preds = taste.predict_variants(mode, 3);
            let other = &corpora[(i + 1) % corpora.len()].1;
            for (k, correct) in held.iter().enumerate().take(6) {
                let wrong = &other[k % other.len()];
                let sc = |t: &Track| {
                    let v = vec_of(t);
                    preds.iter().map(|p| v.cosine_similarity(p)).fold(f32::MIN, f32::max)
                };
                if preds.is_empty() { trials += 1; continue; }
                if sc(correct) > sc(wrong) { hits += 1; }
                trials += 1;
            }
        }
        let name = ["blend+stochastic (shipped)", "plain", "blend only", "get_top_predictions"][mode as usize];
        println!("  {:<28} {hits}/{trials} = {:.0}%", name, 100.0 * hits as f32 / trials.max(1) as f32);
    }
    let _ : HashMap<String,u8> = HashMap::new();
}
