//! Learned autoplay — what to play next, from Camden's actual listening rather than Spotify's guess.
//!
//! Spotify's radio gives us *candidates*; this decides the *order*. A [`TensorSequenceTree`]
//! (WatchTower) learns sequences of dense track vectors and predicts a continuation; we score each
//! candidate by how close it is to that prediction and play the best one.
//!
//! ## Why we embed tracks ourselves
//!
//! The obvious vector is Spotify's audio features (energy, valence, tempo). We can't have it:
//! `/v1/audio-features`, `/v1/audio-analysis` and `/recommendations` are all **403/404** for
//! post-2024 apps like ours. So the embedding is built from metadata librespot *does* give us —
//! artists, album, duration, popularity, explicit — via signed feature hashing. Artist is the
//! dominant signal, which is the right prior for a listening model: taste clusters by artist far
//! more than by tempo.
//!
//! ## Why there's no model file
//!
//! `TensorSequenceTree` has no serialization, and it doesn't need one: it learns fast, and the
//! training data (the playlists, and the play history) is already cached on disk. So the model is
//! rebuilt at startup from that data instead of being persisted. Retraining is cheap; a stale
//! model file that disagrees with the cache would not be.

use nocturne_api::Track;
use serde::{Deserialize, Serialize};
use watchtower::{Tensor, TensorSequenceTree, TensorSequenceTreeConfig};

/// Embedding width. 32 dims of artist, 16 of album, and 8 scalar/pad — wide enough that hash
/// collisions between distinct artists are rare, narrow enough to learn from a few hundred plays.
const ARTIST_DIMS: usize = 32;
const ALBUM_DIMS: usize = 16;
const SCALAR_DIMS: usize = 8;
pub const DIMS: usize = ARTIST_DIMS + ALBUM_DIMS + SCALAR_DIMS;

/// FNV-1a. Any stable hash works; the requirement is only that it's deterministic across runs —
/// `DefaultHasher` is explicitly not (it's randomly seeded), which would silently invalidate the
/// whole embedding space between launches.
fn hash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

/// Signed feature hashing: each token lands in one bucket with a ±1 sign, so unrelated tokens
/// that collide tend to cancel rather than reinforce.
fn hash_into(text: &str, out: &mut [f32]) {
    let n = out.len() as u64;
    for token in text.split(|c: char| !c.is_alphanumeric()).filter(|t| !t.is_empty()) {
        let lower = token.to_lowercase();
        let h = hash(&lower);
        let bucket = (h % n) as usize;
        let sign = if (h >> 63) & 1 == 1 { -1.0 } else { 1.0 };
        out[bucket] += sign;
    }
}

/// A track → a unit vector. Deterministic: the same track always embeds identically, across runs.
pub fn embed(t: &Track) -> Tensor {
    let mut v = vec![0.0f32; DIMS];

    hash_into(&t.artists, &mut v[..ARTIST_DIMS]);
    hash_into(&t.album, &mut v[ARTIST_DIMS..ARTIST_DIMS + ALBUM_DIMS]);

    let s = ARTIST_DIMS + ALBUM_DIMS;
    // Duration in minutes, squashed — a 2-minute interlude and a 9-minute build are different
    // things, but the difference shouldn't dwarf the artist signal.
    v[s] = (t.duration_ms as f32 / 60_000.0).min(10.0) / 10.0;
    v[s + 1] = t.popularity.unwrap_or(50) as f32 / 100.0;
    v[s + 2] = if t.explicit.unwrap_or(false) { 1.0 } else { 0.0 };

    // Unit-normalize: the tree's state equivalence is cosine-based, so magnitude is noise.
    Tensor::from_data(v).normalize()
}

/// One observed play, persisted so the model can be rebuilt next launch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Play {
    pub uri: String,
    /// Fraction of the track actually listened to. A skip at 5% and a full play are very different
    /// training signals, and this is the only reward Nocturne can observe without asking.
    pub completion: f32,
}

pub struct Taste {
    tree: TensorSequenceTree,
    /// Recent plays, newest last — the context handed to the tree at prediction time.
    context: Vec<Tensor>,
    trained_sequences: usize,
}

impl Default for Taste {
    fn default() -> Self {
        Self::new()
    }
}

impl Taste {
    pub fn new() -> Self {
        let config = TensorSequenceTreeConfig {
            // A listening run has short-range structure (this artist, this mood); a huge context
            // window just dilutes it.
            max_context_window: 8,
            // Some exploration, or autoplay becomes a loop of the same five tracks.
            exploration_rate: 0.10,
            enable_experience_replay: true,
            enable_delta_regression: true,
            ..Default::default()
        };

        Self {
            tree: TensorSequenceTree::new(config),
            context: Vec::new(),
            trained_sequences: 0,
        }
    }

    /// Is there enough signal to trust this over Spotify's ordering?
    pub fn is_warm(&self) -> bool {
        self.trained_sequences >= 3
    }

    pub fn trained_sequences(&self) -> usize {
        self.trained_sequences
    }

    /// Pre-train on a curated ordering — a playlist. These are the strongest free training data we
    /// have: a human already decided these tracks belong next to each other.
    pub fn learn_sequence(&mut self, tracks: &[Track]) {
        if tracks.len() < 2 {
            return;
        }
        let seq: Vec<Tensor> = tracks.iter().map(embed).collect();
        self.tree.learn(&seq);
        self.trained_sequences += 1;
    }

    /// Learn from what actually happened: a run of plays, rewarded by how much of each was heard.
    /// A skipped track is a *negative* example — that's the signal a plain playlist can't give.
    pub fn learn_plays(&mut self, plays: &[(Track, f32)]) {
        if plays.len() < 2 {
            return;
        }
        let seq: Vec<Tensor> = plays.iter().map(|(t, _)| embed(t)).collect();
        let outcome = plays.iter().map(|(_, c)| *c).sum::<f32>() / plays.len() as f32;
        self.tree.learn_with_outcome(&seq, outcome);
        self.trained_sequences += 1;
    }

    /// Note a track as it plays, building the live context for the next prediction.
    pub fn observe(&mut self, track: &Track) {
        self.context.push(embed(track));
        let max = 16;
        if self.context.len() > max {
            let drop = self.context.len() - max;
            self.context.drain(..drop);
        }
    }

    /// Rank `candidates` (Spotify's radio) by how well each continues what's being listened to.
    ///
    /// Returns them reordered, best first. If the model is cold or has no opinion, the input order
    /// is preserved — Spotify's ordering is a perfectly good fallback and much better than noise.
    pub fn rank(&mut self, candidates: Vec<Track>) -> Vec<Track> {
        if candidates.is_empty() || !self.is_warm() || self.context.is_empty() {
            return candidates;
        }

        let preds = self.tree.predict_next(&self.context, 3, true, true);
        if preds.is_empty() {
            return candidates;
        }

        let mut scored: Vec<(f32, Track)> = candidates
            .into_iter()
            .map(|t| {
                let v = embed(&t);
                // Best match against any of the top predictions: the tree may see several plausible
                // continuations, and a candidate near *any* of them is a good pick.
                let score = preds
                    .iter()
                    .map(|p| v.cosine_similarity(p))
                    .fold(f32::MIN, f32::max);
                (score, t)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        tracing::info!(
            "taste: reranked {} candidates (top score {:.3}, {} sequences trained)",
            scored.len(),
            scored.first().map(|s| s.0).unwrap_or(0.0),
            self.trained_sequences
        );
        scored.into_iter().map(|(_, t)| t).collect()
    }
}
