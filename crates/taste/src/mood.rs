//! Moods: turn a phrase like "chill winter lofi vibes" into a point in acoustic space, and predict
//! where a listening session's mood is *heading*.
//!
//! This is where [`watchtower::TensorSequenceTree`] actually earns its keep. Track *selection* is a
//! nearest-neighbour problem and the tree measured terribly at it (see [`crate::Taste::rank`]) —
//! but a listening session genuinely IS a sequence: energy climbs, things wind down. That's a
//! trajectory through a 12-dimensional acoustic space, which is exactly the shape the tree models.
//!
//! So: the tree predicts the *next mood point*, and nearest-neighbour finds tracks near it.

use crate::{AudioFeatures, FEATURE_DIMS};
use watchtower::{Tensor, TensorSequenceTree, TensorSequenceTreeConfig};

/// A target point in acoustic space. Every field is 0..1 except tempo (BPM).
#[derive(Debug, Clone, Copy)]
pub struct Mood {
    pub danceability: f32,
    pub energy: f32,
    pub valence: f32,
    pub acousticness: f32,
    pub instrumentalness: f32,
    pub tempo: f32,
}

impl Default for Mood {
    /// Dead centre — a mood with no opinion.
    fn default() -> Self {
        Self {
            danceability: 0.5,
            energy: 0.5,
            valence: 0.5,
            acousticness: 0.5,
            instrumentalness: 0.3,
            tempo: 110.0,
        }
    }
}

impl Mood {
    /// Build the same 12-dim acoustic block [`crate::embed`] writes, so a mood and a track live in
    /// the same space and can be compared directly.
    pub fn to_features(self) -> AudioFeatures {
        AudioFeatures {
            danceability: self.danceability,
            energy: self.energy,
            valence: self.valence,
            acousticness: self.acousticness,
            instrumentalness: self.instrumentalness,
            speechiness: 0.06,
            liveness: 0.15,
            loudness: -8.0 - (1.0 - self.energy) * 14.0,
            tempo: self.tempo,
            key: -1,
            mode: 1,
        }
    }
}

/// One recognized word and what it does to the target.
struct Word {
    keys: &'static [&'static str],
    apply: fn(&mut Mood),
}

/// The vocabulary. Deliberately small and legible — this is a mood *parser*, not an NLP model, and
/// a wrong guess here is far worse than an ignored word.
const WORDS: &[Word] = &[
    Word { keys: &["chill", "chilled", "relaxed", "calm", "mellow"], apply: |m| { m.energy = 0.25; m.tempo = 88.0; m.danceability = 0.5; } },
    Word { keys: &["lofi", "lo-fi", "study", "focus", "coding", "programming"], apply: |m| { m.instrumentalness = 0.85; m.energy = 0.3; m.acousticness = 0.6; m.tempo = 82.0; } },
    Word { keys: &["hype", "energetic", "energy", "workout", "gym", "intense", "hard"], apply: |m| { m.energy = 0.95; m.danceability = 0.7; m.tempo = 150.0; m.acousticness = 0.05; } },
    Word { keys: &["sad", "melancholy", "melancholic", "down", "depressing", "cry"], apply: |m| { m.valence = 0.12; m.energy = 0.3; m.acousticness = 0.5; } },
    Word { keys: &["happy", "upbeat", "joyful", "sunny", "feelgood"], apply: |m| { m.valence = 0.9; m.energy = 0.75; m.danceability = 0.75; } },
    Word { keys: &["dance", "party", "club", "edm", "rave"], apply: |m| { m.danceability = 0.9; m.energy = 0.9; m.tempo = 126.0; m.acousticness = 0.03; } },
    Word { keys: &["acoustic", "unplugged", "folk"], apply: |m| { m.acousticness = 0.9; m.energy = 0.35; m.instrumentalness = 0.3; } },
    Word { keys: &["instrumental", "ambient", "atmospheric", "soundtrack", "cinematic"], apply: |m| { m.instrumentalness = 0.9; m.energy = 0.35; m.valence = 0.4; } },
    Word { keys: &["dark", "moody", "brooding", "night", "midnight"], apply: |m| { m.valence = 0.2; m.energy = 0.5; } },
    Word { keys: &["driving", "drive", "road"], apply: |m| { m.energy = 0.7; m.tempo = 120.0; m.danceability = 0.6; } },
    Word { keys: &["sleep", "sleepy", "dreamy", "quiet", "soft"], apply: |m| { m.energy = 0.12; m.tempo = 70.0; m.acousticness = 0.8; m.instrumentalness = 0.7; } },
    Word { keys: &["fast", "speed", "rapid"], apply: |m| { m.tempo = 155.0; m.energy = 0.85; } },
    Word { keys: &["slow", "sludge"], apply: |m| { m.tempo = 72.0; m.energy = 0.3; } },
    // Seasons are real listening categories for him — the whole lofi-fall theme exists.
    Word { keys: &["winter", "cold", "snow"], apply: |m| { m.valence = 0.3; m.acousticness = 0.7; m.energy = 0.3; } },
    Word { keys: &["autumn", "fall", "cozy", "warm"], apply: |m| { m.valence = 0.45; m.acousticness = 0.65; m.energy = 0.35; m.tempo = 85.0; } },
    Word { keys: &["summer", "beach", "tropical"], apply: |m| { m.valence = 0.85; m.energy = 0.75; m.danceability = 0.8; } },
    Word { keys: &["spring", "fresh"], apply: |m| { m.valence = 0.75; m.energy = 0.6; } },
    Word { keys: &["metal", "heavy", "aggressive", "rage"], apply: |m| { m.energy = 0.97; m.valence = 0.25; m.acousticness = 0.02; m.tempo = 140.0; } },
    Word { keys: &["jazz", "smooth"], apply: |m| { m.acousticness = 0.7; m.instrumentalness = 0.6; m.energy = 0.35; m.danceability = 0.55; } },
];

/// Parse a phrase into a target mood. Unknown words are ignored rather than guessed at; the
/// returned bool says whether we understood *anything* at all, so the caller can tell the user
/// instead of silently playing a default mood.
pub fn parse(text: &str) -> (Mood, bool) {
    let lower = text.to_lowercase();
    let mut mood = Mood::default();
    let mut understood = false;

    for token in lower.split(|c: char| !c.is_alphanumeric() && c != '-') {
        if token.is_empty() {
            continue;
        }
        for w in WORDS {
            if w.keys.contains(&token) {
                (w.apply)(&mut mood);
                understood = true;
            }
        }
    }
    (mood, understood)
}

/// Every mood word the parser knows — for UI suggestions.
pub fn vocabulary() -> Vec<&'static str> {
    WORDS.iter().filter_map(|w| w.keys.first().copied()).collect()
}

/// The acoustic-only part of a track's embedding, as a standalone vector.
///
/// This is the space moods live in: no artist/album hashing, no duration — just how the music
/// *sounds*. Mood targets and trajectory prediction both operate here.
pub fn acoustic_vec(f: &AudioFeatures) -> Tensor {
    let mut v = vec![0.0f32; FEATURE_DIMS];
    v[0] = f.danceability;
    v[1] = f.energy;
    v[2] = f.valence;
    v[3] = f.acousticness;
    v[4] = f.instrumentalness;
    v[5] = f.speechiness;
    v[6] = f.liveness;
    v[7] = ((f.loudness + 60.0) / 60.0).clamp(0.0, 1.0);
    v[8] = (f.tempo / 250.0).clamp(0.0, 1.0);
    if f.key >= 0 {
        let theta = f.key as f32 / 12.0 * std::f32::consts::TAU;
        v[9] = theta.cos() * 0.5 + 0.5;
        v[10] = theta.sin() * 0.5 + 0.5;
    }
    v[11] = f.mode.clamp(0, 1) as f32;
    Tensor::from_data(v)
}

/// Learns how a session's mood *moves* and predicts where it's going next.
///
/// The tree's home turf: a listening session is a genuine time-ordered sequence in acoustic space
/// (things get more energetic, or wind down), unlike a playlist, which is a bag. Delta regression is
/// on — it's velocity/acceleration extrapolation, which is precisely "the mood is trending calmer".
pub struct Trajectory {
    tree: TensorSequenceTree,
    session: Vec<Tensor>,
    sessions_learned: usize,
}

impl Default for Trajectory {
    fn default() -> Self {
        Self::new()
    }
}

impl Trajectory {
    pub fn new() -> Self {
        let config = TensorSequenceTreeConfig {
            max_context_window: 6,
            // Mood states are continuous and fuzzy; demanding near-identical vectors would make
            // every track its own state and the tree would never generalize.
            similarity_threshold: 0.80,
            retrieval_threshold: 0.75,
            enable_delta_regression: true,
            enable_experience_replay: true,
            ..Default::default()
        };
        Self {
            tree: TensorSequenceTree::new(config),
            session: Vec::new(),
            sessions_learned: 0,
        }
    }

    pub fn sessions_learned(&self) -> usize {
        self.sessions_learned
    }

    /// Note the mood of a track as it plays.
    pub fn observe(&mut self, f: &AudioFeatures) {
        self.session.push(acoustic_vec(f));
        if self.session.len() > 24 {
            self.session.remove(0);
        }
    }

    /// Learn a completed listening run's mood arc.
    pub fn learn_session(&mut self, moods: &[AudioFeatures]) {
        if moods.len() < 3 {
            return;
        }
        let seq: Vec<Tensor> = moods.iter().map(acoustic_vec).collect();
        self.tree.learn(&seq);
        self.sessions_learned += 1;
    }

    /// Where is this session's mood heading? `None` until there's enough of a session to have a
    /// direction at all, or if the tree has no opinion — the caller then just uses the current mood.
    pub fn predict(&mut self) -> Option<Tensor> {
        if self.session.len() < 3 {
            return None;
        }
        let preds = self.tree.predict_next(&self.session, 1, true, false);
        preds.into_iter().next()
    }

    /// The mood right now — the mean of the last few tracks, which is steadier than any single one.
    pub fn current(&self) -> Option<Tensor> {
        let n = self.session.len().min(3);
        if n == 0 {
            return None;
        }
        let mut acc = vec![0.0f32; FEATURE_DIMS];
        for t in self.session.iter().rev().take(n) {
            for (i, x) in t.data.iter().enumerate().take(FEATURE_DIMS) {
                acc[i] += *x / n as f32;
            }
        }
        Some(Tensor::from_data(acc))
    }
}
