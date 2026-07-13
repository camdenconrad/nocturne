//! Optional: let Claude Haiku read the mood phrase.
//!
//! The built-in parser ([`crate::mood::parse`]) is a ~20-word lookup table. It's fast, offline and
//! predictable, but it only knows the words it knows — "music for staring out a train window in
//! November" means nothing to it.
//!
//! Haiku turns any phrase into the same acoustic target. It is strictly an *enhancement*: without
//! `ANTHROPIC_API_KEY` (or on any error, timeout, or malformed reply) we fall back to the word list
//! rather than failing the user's radio. A mood is not worth a hard dependency on the network.

use crate::mood::Mood;
use serde::Deserialize;

const MODEL: &str = "claude-haiku-4-5-20251001";
const ENDPOINT: &str = "https://api.anthropic.com/v1/messages";

const SYSTEM: &str = "\
You convert a music mood phrase into Spotify audio-feature targets. \
Reply with ONLY a JSON object, no prose, no markdown fence, with exactly these keys:
{\"danceability\":0-1,\"energy\":0-1,\"valence\":0-1,\"acousticness\":0-1,\"instrumentalness\":0-1,\"tempo\":40-200}
valence is musical positivity (sad=low, happy=high). energy is intensity. \
acousticness is how acoustic vs produced/electronic. instrumentalness is lack of vocals. \
tempo is BPM. Infer sensibly from any phrasing, including metaphor and imagery.";

#[derive(Deserialize)]
struct ApiResponse {
    content: Vec<Block>,
}

#[derive(Deserialize)]
struct Block {
    #[serde(default)]
    text: String,
}

#[derive(Deserialize)]
struct Target {
    danceability: f32,
    energy: f32,
    valence: f32,
    acousticness: f32,
    instrumentalness: f32,
    tempo: f32,
}

pub fn available() -> bool {
    std::env::var("ANTHROPIC_API_KEY").is_ok_and(|k| !k.trim().is_empty())
}

/// Ask Haiku for the mood. `None` on any problem at all — the caller then uses the word list.
pub async fn parse_mood(phrase: &str) -> Option<Mood> {
    let key = std::env::var("ANTHROPIC_API_KEY").ok()?;
    let body = serde_json::json!({
        "model": MODEL,
        "max_tokens": 200,
        "system": SYSTEM,
        "messages": [{ "role": "user", "content": phrase }],
    });

    let http = reqwest::Client::new();
    let resp = http
        .post(ENDPOINT)
        .header("x-api-key", key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        // A radio must not hang on a mood. If Haiku is slow, the word list answers instead.
        .timeout(std::time::Duration::from_secs(8))
        .json(&body)
        .send()
        .await
        .map_err(|e| tracing::warn!("haiku: {e}"))
        .ok()?;

    if !resp.status().is_success() {
        tracing::warn!("haiku: HTTP {}", resp.status());
        return None;
    }

    let parsed: ApiResponse = resp.json().await.ok()?;
    let text = parsed.content.first()?.text.trim().to_string();
    // Be forgiving about a stray fence, even though the system prompt forbids one.
    let json = text
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    let t: Target = serde_json::from_str(json)
        .map_err(|e| tracing::warn!("haiku: bad JSON ({e}): {json}"))
        .ok()?;

    tracing::info!(
        "haiku: “{phrase}” → energy={:.2} valence={:.2} acoustic={:.2} {:.0}bpm",
        t.energy,
        t.valence,
        t.acousticness,
        t.tempo
    );
    Some(Mood {
        danceability: t.danceability.clamp(0.0, 1.0),
        energy: t.energy.clamp(0.0, 1.0),
        valence: t.valence.clamp(0.0, 1.0),
        acousticness: t.acousticness.clamp(0.0, 1.0),
        instrumentalness: t.instrumentalness.clamp(0.0, 1.0),
        tempo: t.tempo.clamp(40.0, 200.0),
    })
}

/// Haiku first, word list as the floor. Always returns a usable mood.
pub async fn mood_for(phrase: &str) -> (Mood, bool) {
    if available() {
        if let Some(m) = parse_mood(phrase).await {
            return (m, true);
        }
    }
    crate::mood::parse(phrase)
}
