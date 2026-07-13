//! Learned autoplay — what to play next, from Camden's actual listening rather than Spotify's guess.
//!
//! Spotify's radio gives us *candidates*; this decides the *order*. A [`TensorSequenceTree`]
//! (WatchTower) learns sequences of dense track vectors and predicts a continuation; we score each
//! candidate by how close it is to that prediction and play the best one.
//!
//! ## The embedding
//!
//! Spotify's **real audio features** — energy, valence, danceability, tempo, acousticness… — are
//! the front of the vector. The Web API 403s them, but the internal `/audio-attributes` service the
//! real client uses hands them over (see `nocturne_session::NocturneHandle::audio_features`), so
//! the model knows a track is 0.94-energy at 113 BPM rather than merely "by an artist whose name
//! hashes here".
//!
//! Identity (artist/album, signed feature hashing) stays *behind* those features and deliberately
//! down-weighted: it keeps the model able to say "more of this artist" while letting acoustics
//! carry the similarity. A track with no analysis — local files, some new releases — still embeds,
//! just with the feature block zeroed and identity doing the work.

use nocturne_api::{AudioFeatures, Track};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use watchtower::{Tensor, TensorSequenceTree, TensorSequenceTreeConfig};

/// Real acoustic features first — they carry the similarity.
const FEATURE_DIMS: usize = 12;
/// Identity behind them, scaled down so "same artist" nudges rather than dominates.
const ARTIST_DIMS: usize = 24;
const ALBUM_DIMS: usize = 12;
const SCALAR_DIMS: usize = 4;
pub const DIMS: usize = FEATURE_DIMS + ARTIST_DIMS + ALBUM_DIMS + SCALAR_DIMS;

/// How much quieter identity is than acoustics. Artist still matters — it's just no longer the
/// only thing the model can see.
const IDENTITY_WEIGHT: f32 = 0.35;

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
///
/// `features` is Spotify's real analysis when we have it. Without it the acoustic block is zeroed
/// and the vector still works — just coarser.
pub fn embed(t: &Track, features: Option<&AudioFeatures>) -> Tensor {
    let mut v = vec![0.0f32; DIMS];

    if let Some(f) = features {
        // All squashed to roughly 0..1 so no single axis dominates the cosine.
        v[0] = f.danceability;
        v[1] = f.energy;
        v[2] = f.valence;
        v[3] = f.acousticness;
        v[4] = f.instrumentalness;
        v[5] = f.speechiness;
        v[6] = f.liveness;
        // Loudness is dB (about -60..0); map to 0..1.
        v[7] = ((f.loudness + 60.0) / 60.0).clamp(0.0, 1.0);
        // Tempo: 0..250 BPM covers everything real.
        v[8] = (f.tempo / 250.0).clamp(0.0, 1.0);
        // Key as a circle, so B(11) and C(0) are neighbours rather than opposites — they are, in
        // pitch space, and a linear 0..11 would tell the model a lie.
        if f.key >= 0 {
            let theta = f.key as f32 / 12.0 * std::f32::consts::TAU;
            v[9] = theta.cos() * 0.5 + 0.5;
            v[10] = theta.sin() * 0.5 + 0.5;
        }
        v[11] = f.mode.clamp(0, 1) as f32;
    }

    let a = FEATURE_DIMS;
    let b = a + ARTIST_DIMS;
    hash_into(&t.artists, &mut v[a..b]);
    hash_into(&t.album, &mut v[b..b + ALBUM_DIMS]);
    for x in &mut v[a..b + ALBUM_DIMS] {
        *x *= IDENTITY_WEIGHT;
    }

    let s = b + ALBUM_DIMS;
    v[s] = (t.duration_ms as f32 / 60_000.0).min(10.0) / 10.0;
    v[s + 1] = t.popularity.unwrap_or(50) as f32 / 100.0;
    v[s + 2] = if t.explicit.unwrap_or(false) { 1.0 } else { 0.0 };

    // Unit-normalize: the tree's state equivalence is cosine-based, so magnitude is noise.
    Tensor::from_data(v).normalize()
}

/// `spotify:track:ID` → `ID`.
pub fn track_id(uri: &str) -> &str {
    uri.rsplit(':').next().unwrap_or(uri)
}

/// One observed play, persisted so the model can be rebuilt next launch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Play {
    pub uri: String,
    /// Fraction of the track actually listened to. A skip at 5% and a full play are very different
    /// training signals, and this is the only reward Nocturne can observe without asking.
    pub completion: f32,
}

/// The persisted model: the trained tree plus the feature store it was trained against.
///
/// Versioned, because the embedding layout is part of the model's meaning: if `DIMS` or the
/// feature order changes, every stored tensor silently means something different. A model whose
/// version doesn't match is discarded and retrained rather than trusted — a wrong model is worse
/// than no model.
#[derive(Serialize, Deserialize)]
struct ModelFile {
    version: u32,
    dims: usize,
    tree: TensorSequenceTree,
    features: HashMap<String, AudioFeatures>,
    trained_sequences: usize,
    /// Which corpora are already in the tree, so a relaunch doesn't learn them again. Without this
    /// every launch re-trained the same playlists, permanently overweighting them against actual
    /// listening — and the count grew without bound.
    #[serde(default)]
    learned: HashSet<String>,
}

/// Bump when the embedding changes meaning.
const MODEL_VERSION: u32 = 1;

pub struct Taste {
    tree: TensorSequenceTree,
    /// Recent plays, newest last — the context handed to the tree at prediction time.
    context: Vec<Tensor>,
    trained_sequences: usize,
    /// track id → Spotify's real analysis. Immutable per track, so this is a pure cache.
    features: HashMap<String, AudioFeatures>,
    /// Keys of corpora already learned (playlist ids, "liked"), so relaunches don't double-count.
    learned: HashSet<String>,
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
            features: HashMap::new(),
            learned: HashSet::new(),
        }
    }

    /// Load a trained model from disk. Returns `None` (and leaves the caller to retrain) if the
    /// file is absent, unreadable, or was written by a different embedding layout.
    pub fn load(path: &std::path::Path) -> Option<Self> {
        let bytes = std::fs::read(path).ok()?;
        let model: ModelFile = match serde_json::from_slice(&bytes) {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!("taste: model file unreadable ({e}) — retraining");
                return None;
            }
        };
        if model.version != MODEL_VERSION || model.dims != DIMS {
            tracing::warn!(
                "taste: model is v{} dims={} but this build is v{MODEL_VERSION} dims={DIMS} — retraining",
                model.version,
                model.dims
            );
            return None;
        }
        tracing::info!(
            "taste: loaded model ({} sequences, {} tracks with analysis)",
            model.trained_sequences,
            model.features.len()
        );
        Some(Self {
            tree: model.tree,
            context: Vec::new(),
            trained_sequences: model.trained_sequences,
            features: model.features,
            learned: model.learned,
        })
    }

    /// Save the trained model. Atomic: a crash mid-write must not leave a truncated model that
    /// fails to parse on next launch.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let model = ModelFile {
            version: MODEL_VERSION,
            dims: DIMS,
            tree: self.tree.clone(),
            features: self.features.clone(),
            trained_sequences: self.trained_sequences,
            learned: self.learned.clone(),
        };
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let bytes = serde_json::to_vec(&model)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, path)?;
        tracing::info!(
            "taste: saved model ({} sequences, {} tracks with analysis)",
            self.trained_sequences,
            self.features.len()
        );
        Ok(())
    }

    /// Feed in analysis fetched from Spotify's internal service.
    pub fn add_features(&mut self, features: HashMap<String, AudioFeatures>) {
        self.features.extend(features);
    }

    pub fn features(&self) -> &HashMap<String, AudioFeatures> {
        &self.features
    }

    /// How many of the tracks we know about have real analysis attached.
    pub fn feature_count(&self) -> usize {
        self.features.len()
    }

    fn vec_of(&self, t: &Track) -> Tensor {
        embed(t, self.features.get(track_id(&t.uri)))
    }

    /// Is there enough signal to trust this over Spotify's ordering?
    pub fn is_warm(&self) -> bool {
        self.trained_sequences >= 3
    }

    pub fn trained_sequences(&self) -> usize {
        self.trained_sequences
    }

    /// Has this corpus (playlist id, "liked") already been learned?
    pub fn has_learned(&self, key: &str) -> bool {
        self.learned.contains(key)
    }

    /// Pre-train on a curated ordering — a playlist. These are the strongest free training data we
    /// have: a human already decided these tracks belong next to each other.
    ///
    /// `key` makes this idempotent across launches. Re-learning the same playlist every start would
    /// stack it in the tree over and over and drown out real listening.
    pub fn learn_corpus(&mut self, key: &str, tracks: &[Track]) {
        if self.learned.contains(key) {
            return;
        }
        self.learned.insert(key.to_string());
        self.learn_sequence(tracks);
    }

    pub fn learn_sequence(&mut self, tracks: &[Track]) {
        if tracks.len() < 2 {
            return;
        }
        let seq: Vec<Tensor> = tracks.iter().map(|t| self.vec_of(t)).collect();
        self.tree.learn(&seq);
        self.trained_sequences += 1;
    }

    /// Learn from what actually happened: a run of plays, rewarded by how much of each was heard.
    /// A skipped track is a *negative* example — that's the signal a plain playlist can't give.
    pub fn learn_plays(&mut self, plays: &[(Track, f32)]) {
        if plays.len() < 2 {
            return;
        }
        let seq: Vec<Tensor> = plays.iter().map(|(t, _)| self.vec_of(t)).collect();
        let outcome = plays.iter().map(|(_, c)| *c).sum::<f32>() / plays.len() as f32;
        self.tree.learn_with_outcome(&seq, outcome);
        self.trained_sequences += 1;
    }

    /// Note a track as it plays, building the live context for the next prediction.
    pub fn observe(&mut self, track: &Track) {
        self.context.push(self.vec_of(track));
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
                let v = self.vec_of(&t);
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
