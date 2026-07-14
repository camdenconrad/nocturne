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

pub mod llm;
pub mod mood;

pub use nocturne_api::{AudioFeatures, Track};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use watchtower::{Tensor, TensorSequenceTree, TensorSequenceTreeConfig};

/// Real acoustic features first — they carry the similarity.
pub(crate) const FEATURE_DIMS: usize = 12;
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
    /// Acoustic vectors of the endorsed tracks — see [`Taste::endorsements`].
    #[serde(default)]
    endorsed: Vec<watchtower::Tensor>,
    /// URIs behind `endorsed`, so the same track can't be counted twice.
    #[serde(default)]
    endorsed_uris: HashSet<String>,
    /// Per-mood misfits — see [`Taste::skip_in_mood`]. Keyed by mood phrase, NOT global.
    #[serde(default)]
    mood_skips: HashMap<String, Vec<watchtower::Tensor>>,
}

/// Bump when the embedding changes meaning.
///
/// v2: positives-only. The model no longer carries skip-derived negatives, so a v1 file's tree is
/// the wrong shape of thing — it was trained against outcomes that included ambiguity.
const MODEL_VERSION: u32 = 2;

/// What counts as a finish.
///
/// # Why Nocturne only ever learns from positives
///
/// Completion is the honest signal: nobody accidentally listens to a whole track. A skip is not
/// its opposite — it's *ambiguous*. He skipped because the phone rang, because it's the wrong
/// hour for it, because he's heard it twice today, because someone walked in. Train on that and
/// the model learns "he hates this" from an event that meant nothing, and there is no later
/// evidence that can undo it: a wrong negative permanently carves a hole in taste space.
///
/// So skips are *dropped*, not learned. Every training example Nocturne accepts is unambiguous —
/// a finish, or a like. It learns strictly slower. It never poisons itself.
///
/// The one exception is [`Taste::skip_in_mood`], and it's an exception that proves the rule: a
/// skip inside a *mood* radio is unambiguous about the mood even when it says nothing about the
/// track. That evidence is kept where it belongs — scoped to the phrase — and never allowed near
/// the taste model.
pub const FINISHED: f32 = 0.85;

/// A skip fast enough to mean "not this, not here" rather than "I drifted off".
pub const QUICK_SKIP: f32 = 0.25;

/// How much more a positive is worth when it happened inside a list *Nocturne itself* built.
///
/// Finishing a track he chose says he likes the track. Finishing one the model served him says
/// the model was right — that's feedback on the recommendation, not just the song, and it's the
/// only signal here that grades the thing we're actually trying to improve.
pub const OURS_WEIGHT: f32 = 2.0;

pub struct Taste {
    tree: TensorSequenceTree,
    /// Recent plays, newest last — the context handed to the tree at prediction time.
    context: Vec<Tensor>,
    trained_sequences: usize,
    /// track id → Spotify's real analysis. Immutable per track, so this is a pure cache.
    features: HashMap<String, AudioFeatures>,
    /// Keys of corpora already learned (playlist ids, "liked"), so relaunches don't double-count.
    learned: HashSet<String>,
    /// Acoustic vectors of tracks he finished or liked — the positive set, used as a prior at
    /// ranking time. Nothing he skipped ever lands here.
    endorsed: Vec<watchtower::Tensor>,
    endorsed_uris: HashSet<String>,
    /// mood phrase → tracks that didn't fit THAT mood. Deliberately not a taste signal.
    mood_skips: HashMap<String, Vec<watchtower::Tensor>>,
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
            endorsed: Vec::new(),
            endorsed_uris: HashSet::new(),
            mood_skips: HashMap::new(),
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
            // The TREE is version-bound — trained against an embedding layout (or, across v1→v2, a
            // reward signal) this build no longer means. Throw it away.
            //
            // The FEATURES are not. They're Spotify's raw analysis of a track, immutable and
            // version-independent, and they cost hours of background backfill to collect. Carrying
            // them across the bump means a retrain is a few seconds of CPU instead of re-fetching
            // the entire library.
            tracing::warn!(
                "taste: model is v{} dims={} but this build is v{MODEL_VERSION} dims={DIMS} — \
                 discarding the tree, keeping {} analyzed tracks",
                model.version,
                model.dims,
                model.features.len()
            );
            return Some(Self {
                features: model.features,
                ..Self::new()
            });
        }
        tracing::info!(
            "taste: loaded model ({} sequences, {} endorsements, {} tracks with analysis)",
            model.trained_sequences,
            model.endorsed.len(),
            model.features.len()
        );
        Some(Self {
            tree: model.tree,
            context: Vec::new(),
            trained_sequences: model.trained_sequences,
            features: model.features,
            learned: model.learned,
            endorsed: model.endorsed,
            endorsed_uris: model.endorsed_uris,
            mood_skips: model.mood_skips,
        })
    }

    /// Serialize the model. Split out from [`Taste::write_bytes`] on purpose: the caller can do
    /// this under the lock (it's pure CPU) and then release the lock *before* touching the disk.
    /// Holding the model lock across a multi-megabyte file write is what made pressing play stall
    /// for seconds while the background analysis backfill was running.
    pub fn to_bytes(&self) -> std::io::Result<Vec<u8>> {
        let model = ModelFile {
            version: MODEL_VERSION,
            dims: DIMS,
            tree: self.tree.clone(),
            features: self.features.clone(),
            trained_sequences: self.trained_sequences,
            learned: self.learned.clone(),
            endorsed: self.endorsed.clone(),
            endorsed_uris: self.endorsed_uris.clone(),
            mood_skips: self.mood_skips.clone(),
        };
        serde_json::to_vec(&model)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Write pre-serialized model bytes. Atomic: a crash mid-write must not leave a truncated
    /// model that fails to parse on next launch.
    pub fn write_bytes(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Convenience for callers not holding a contended lock.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let bytes = self.to_bytes()?;
        Self::write_bytes(path, &bytes)?;
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

    /// Learn from what actually happened — but **only from the finishes**.
    ///
    /// A run of plays comes in; the skips are thrown away and what's left is the sequence of
    /// tracks he sat through, learned as an unambiguous positive (outcome 1.0). See [`FINISHED`]
    /// for why the skips aren't turned into negatives instead.
    ///
    /// Dropping the skips also repairs the *sequence*: the tree is a sequence learner, and a run
    /// with three abandoned tracks in the middle of it was teaching the tree that those tracks
    /// follow each other. They don't. What follows what, for him, is the finishes.
    ///
    /// `weight` is how many times this evidence counts — [`OURS_WEIGHT`] when the run happened
    /// inside a list Nocturne built, 1.0 when he picked the list himself.
    ///
    /// Returns how many finishes were learned (0 if the run was all skips — that's a no-op, not
    /// a failure).
    pub fn learn_finishes(&mut self, plays: &[(Track, f32)], weight: f32) -> usize {
        let finished: Vec<&Track> = plays
            .iter()
            .filter(|(_, c)| *c >= FINISHED)
            .map(|(t, _)| t)
            .collect();

        // Every finish is an endorsement, even a lone one — it still tells the ranker something,
        // it just can't teach the tree anything sequential.
        for t in &finished {
            self.endorse_vec(t);
        }
        if finished.len() < 2 {
            return finished.len();
        }

        let seq: Vec<Tensor> = finished.iter().map(|t| self.vec_of(t)).collect();
        for _ in 0..weight.round().max(1.0) as usize {
            self.tree.learn_with_outcome(&seq, 1.0);
            self.trained_sequences += 1;
        }
        finished.len()
    }

    /// He hit the heart. The single least ambiguous thing he can tell us, so it's learned
    /// immediately rather than waiting for a training window to fill.
    ///
    /// The liked track is learned as the *continuation* of what he was listening to when he liked
    /// it — that's the sequence the tree is built to model, and it's exactly the claim "after
    /// this run of music, this track was the right thing to hear". `weight` should be
    /// [`OURS_WEIGHT`] when the track came from a list Nocturne built: liking something the model
    /// found is the strongest evidence in the whole system that it's working.
    pub fn learn_like(&mut self, track: &Track, weight: f32) {
        self.endorse_vec(track);

        let mut seq: Vec<Tensor> = self.context.iter().rev().take(4).rev().cloned().collect();
        seq.push(self.vec_of(track));
        if seq.len() < 2 {
            return;
        }
        for _ in 0..weight.round().max(1.0) as usize {
            self.tree.learn_with_outcome(&seq, 1.0);
            self.trained_sequences += 1;
        }
    }

    /// Add a track to the positive set (deduped). No-op for tracks with no analysis yet — the
    /// backfill will have their features later, and [`Taste::seed_endorsements`] re-seeds then.
    fn endorse_vec(&mut self, track: &Track) {
        if self.endorsed_uris.contains(&track.uri) {
            return;
        }
        let Some(f) = self.features.get(track_id(&track.uri)) else {
            return;
        };
        let v = mood::acoustic_vec(f);
        self.endorsed_uris.insert(track.uri.clone());
        self.endorsed.push(v);
    }

    /// Fold his existing Liked Songs into the positive set. Idempotent — safe to call every launch
    /// and after every analysis backfill, which is the point: tracks liked before their features
    /// arrived get picked up as soon as they do.
    pub fn seed_endorsements(&mut self, liked: &[Track]) {
        for t in liked {
            self.endorse_vec(t);
        }
    }

    /// Everything he has finished or liked, as acoustic vectors. The positive set — the whole
    /// training signal, and there is deliberately no negative counterpart.
    pub fn endorsements(&self) -> &[watchtower::Tensor] {
        &self.endorsed
    }

    /// He skipped this track inside the "`phrase`" radio, and skipped it fast.
    ///
    /// This is the one negative Nocturne records, and it's a negative about the **mood**, not
    /// about him. A summer lofi track in a "winter chill lofi" station gets skipped in ten
    /// seconds — and that skip means "wrong for this", not "wrong for me". It's a perfectly good
    /// track and he'd happily hear it tomorrow.
    ///
    /// So the evidence is filed under the phrase and used only when that phrase is asked for
    /// again. It never reaches the tree, never becomes an endorsement, never touches the taste
    /// centroid, and never affects any other station. The global model still only learns from
    /// finishes and likes — [`FINISHED`].
    pub fn skip_in_mood(&mut self, phrase: &str, track: &Track) {
        let Some(f) = self.features.get(track_id(&track.uri)) else {
            return;
        };
        let v = mood::acoustic_vec(f);
        let misfits = self.mood_skips.entry(phrase.to_lowercase()).or_default();
        // Bounded, newest-wins: a mood he's been running for months shouldn't accumulate an
        // ever-growing veto list built out of one bad afternoon each.
        misfits.push(v);
        let len = misfits.len();
        if len > 40 {
            misfits.drain(..len - 40);
        }
    }

    /// How badly does this track resemble something he skipped out of *this* mood before?
    fn mood_misfit(&self, phrase: &str, v: &watchtower::Tensor) -> f32 {
        self.mood_skips
            .get(&phrase.to_lowercase())
            .map(|skips| {
                skips
                    .iter()
                    .map(|s| v.cosine_similarity(s))
                    .fold(0.0f32, f32::max)
            })
            .unwrap_or(0.0)
    }

    /// How close is this track to something he has actually endorsed? 0.0 if we can't tell.
    fn endorsement_affinity(&self, track: &Track) -> f32 {
        if self.endorsed.is_empty() {
            return 0.0;
        }
        let Some(f) = self.features.get(track_id(&track.uri)) else {
            return 0.0;
        };
        let v = mood::acoustic_vec(f);
        self.endorsed
            .iter()
            .map(|e| v.cosine_similarity(e))
            .fold(f32::MIN, f32::max)
            .max(0.0)
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

    /// Sweep the tree's prediction modes so we can pick the one that actually works, rather than
    /// assuming. Exposed for `examples/diagnose`.
    pub fn predict_variants(&mut self, mode: u8, n: usize) -> Vec<Tensor> {
        match mode {
            0 => self.tree.predict_next(&self.context, n as i32, true, true),
            1 => self.tree.predict_next(&self.context, n as i32, false, false),
            2 => self.tree.predict_next(&self.context, n as i32, true, false),
            3 => self
                .tree
                .get_top_predictions(&self.context, n)
                .into_iter()
                .map(|(t, _)| t)
                .collect(),
            _ => Vec::new(),
        }
    }

    /// The context itself — the baseline that beat the tree.
    pub fn context(&self) -> &[Tensor] {
        &self.context
    }

    /// Rank `candidates` (Spotify's radio) by how well each continues what's being listened to.
    ///
    /// # Why this is similarity and not the tree's `predict_next`
    ///
    /// Measured on Camden's real library with real Spotify analysis (`examples/diagnose`), against
    /// a 50% chance baseline:
    ///
    /// | ranker                              | accuracy |
    /// |-------------------------------------|----------|
    /// | similarity to recent listening      | **69%**  |
    /// | tree `predict_next` (all 4 modes)   | 0–6%     |
    ///
    /// The tree isn't broken — it was being asked the wrong question. `TensorSequenceTree` predicts
    /// the next item in a *sequence*; a playlist is an unordered bag, not a sequence. Trained on
    /// playlist "order" it learns nothing sequential and falls back to the globally most common
    /// state — which lives in the biggest playlist — making it actively anti-correlated with
    /// whatever else you're listening to. Hence *worse* than chance.
    ///
    /// So ranking uses the acoustic embedding directly, which is where the real signal lives. The
    /// tree keeps learning real *finished* listening runs via [`Taste::learn_finishes`] — the data
    /// it's actually built for — and [`Taste::tree_agrees`] lets it contribute a tiebreak once
    /// enough of those exist to be worth trusting. Re-run the diagnose example to check.
    pub fn rank(&mut self, candidates: Vec<Track>) -> Vec<Track> {
        if candidates.is_empty() || self.context.is_empty() {
            return candidates;
        }

        // Best match against ANY recent track — no recency decay.
        //
        // Decay was tried and measured: weighting the cosine by 0.6^age before taking the max
        // dropped accuracy from 69% to 56%, because a weighted max collapses toward "similar to
        // the single most recent track" instead of "fits the run". Left un-weighted deliberately.
        //
        // Plus a smaller pull toward what he's endorsed: "fits this run" is the brief, "and he has
        // finished or hearted something that sounds like it" is the tiebreak. It's the only other
        // term, because it's the only other thing we actually know.
        let mut scored: Vec<(f32, Track)> = candidates
            .into_iter()
            .map(|t| {
                let v = self.vec_of(&t);
                let fit = self
                    .context
                    .iter()
                    .map(|c| v.cosine_similarity(c))
                    .fold(f32::MIN, f32::max);
                (fit + 0.15 * self.endorsement_affinity(&t), t)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        tracing::info!(
            "taste: reranked {} candidates (top {:.3}, {} sequences learned)",
            scored.len(),
            scored.first().map(|s| s.0).unwrap_or(0.0),
            self.trained_sequences
        );
        scored.into_iter().map(|(_, t)| t).collect()
    }

    /// The centroid of what he ACTUALLY finishes — his taste, as one point in acoustic space.
    ///
    /// Only finishes count, and they count equally: a partial play is not a partial endorsement,
    /// it's an unknown. Anything under [`FINISHED`] contributes nothing at all.
    pub fn taste_centroid(&self, history: &[(Track, f32)]) -> Option<watchtower::Tensor> {
        let mut acc = vec![0.0f32; FEATURE_DIMS];
        let mut total = 0.0f32;
        for (t, completion) in history {
            let Some(f) = self.features.get(track_id(&t.uri)) else {
                continue;
            };
            if *completion < FINISHED {
                continue;
            }
            let w = 1.0;
            let v = mood::acoustic_vec(f);
            for (i, x) in v.data.iter().enumerate().take(FEATURE_DIMS) {
                acc[i] += x * w;
            }
            total += w;
        }
        (total > 0.0).then(|| {
            for x in &mut acc {
                *x /= total;
            }
            watchtower::Tensor::from_data(acc)
        })
    }

    /// Tracks nearest a **mood**, biased toward his taste.
    ///
    /// A pure mood match will happily serve him music that fits "chill lofi" perfectly and that he
    /// would still hate. So the score is the mood match, pulled toward the music he finishes: his
    /// completion-weighted centroid, plus a bonus for resembling something he specifically
    /// endorsed (a finish or a like).
    ///
    /// The only push *away* from anything is mood-local: tracks resembling what he quick-skipped
    /// out of THIS phrase before ([`Taste::skip_in_mood`]). The old global version of that penalty
    /// — subtract `0.35 * worst_skip_similarity` over all history, and hard-filter anything within
    /// 0.97 of a skipped track — is gone. It built a permanent, unfalsifiable negative out of
    /// ambiguous evidence and let one bad afternoon blacklist a whole neighbourhood of his taste.
    /// Scoped to the mood, the same evidence is sound: it only claims the track was wrong *here*.
    pub fn nearest_mood_for_me(
        &self,
        pool: &[Track],
        target: &watchtower::Tensor,
        history: &[(Track, f32)],
        phrase: &str,
        count: usize,
    ) -> Vec<Track> {
        let centroid = self.taste_centroid(history);

        let mut scored: Vec<(f32, &Track)> = pool
            .iter()
            .filter_map(|t| {
                let f = self.features.get(track_id(&t.uri))?;
                let v = mood::acoustic_vec(f);

                // The mood is the brief, so it dominates.
                let mut score = v.cosine_similarity(target);

                // Nudge toward what he actually finishes.
                if let Some(c) = &centroid {
                    score += 0.25 * v.cosine_similarity(c);
                }

                // …and toward the nearest thing he endorsed outright.
                score += 0.20 * self.endorsement_affinity(t);

                // …and away from what didn't belong in THIS mood. A summer track in a winter
                // station is still a good track — it's just not what he asked for.
                score -= 0.30 * self.mood_misfit(phrase, &v);

                Some((score, t))
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut seen = std::collections::HashSet::new();
        scored
            .into_iter()
            .filter(|(_, t)| seen.insert((t.name.to_lowercase(), t.artists.to_lowercase())))
            .take(count)
            .map(|(_, t)| t.clone())
            .collect()
    }

    /// Tracks nearest a **mood**, from a pool (his library). The engine behind mood radio.
    ///
    /// Compares in acoustic space only — artist/album identity is deliberately excluded here,
    /// because "chill winter lofi" is a statement about how music *sounds*, not about who made it.
    pub fn nearest_mood(
        &self,
        pool: &[Track],
        target: &watchtower::Tensor,
        count: usize,
    ) -> Vec<Track> {
        let mut scored: Vec<(f32, &Track)> = pool
            .iter()
            .filter_map(|t| {
                let f = self.features.get(track_id(&t.uri))?;
                let v = mood::acoustic_vec(f);
                Some((v.cosine_similarity(target), t))
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        // Libraries contain the same song twice (liked + in a playlist, or two masters). A radio
        // that plays "Reasons" back to back looks broken, so dedup on title+artist, not uri.
        let mut seen = std::collections::HashSet::new();
        scored
            .into_iter()
            .filter(|(_, t)| seen.insert((t.name.to_lowercase(), t.artists.to_lowercase())))
            .take(count)
            .map(|(_, t)| t.clone())
            .collect()
    }

    /// Rank the whole library against a target vector — the engine behind mood radio.
    pub fn nearest(&self, pool: &[Track], target: &Tensor, count: usize) -> Vec<Track> {
        let mut scored: Vec<(f32, &Track)> = pool
            .iter()
            .map(|t| {
                let v = embed(t, self.features.get(track_id(&t.uri)));
                (v.cosine_similarity(target), t)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(count).map(|(_, t)| t.clone()).collect()
    }
}
