//! Optional: let Claude read the request.
//!
//! Two jobs, two models. [`parse_mood`] turns a phrase into an acoustic target — the built-in
//! parser ([`crate::mood::parse`]) is a ~20-word lookup table, fast and offline, but it only knows
//! the words it knows, and "music for staring out a train window in November" means nothing to it.
//! [`suggest_tracks`] goes further and names actual tracks, which no amount of acoustic targeting
//! can do: "shoegaze that sounds like a hospital at 4am" is a request about music, not about four
//! floats.
//!
//! Both are strictly *enhancements*. Without `ANTHROPIC_API_KEY` — or on any error, timeout, or
//! malformed reply — the caller falls back to the word list and to plain Spotify search rather
//! than failing the user's search or radio. Neither is worth a hard dependency on the network.

use crate::mood::Mood;
use serde::Deserialize;

/// Parsing a phrase into four floats is a small, well-specified job — Haiku does it fast, and a
/// radio must not hang on a mood.
const MOOD_MODEL: &str = "claude-haiku-4-5";
/// Naming tracks is a taste judgement drawn from actual knowledge of music, which is the one thing
/// the small model is bad at. This is the whole point of the feature; don't cheap out on it.
const PICK_MODEL: &str = "claude-opus-4-8";
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
        "model": MOOD_MODEL,
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

/// One track as Claude named it, before it's been matched against the real catalogue.
#[derive(Debug, Clone, Deserialize)]
pub struct Suggestion {
    pub artist: String,
    pub title: String,
}

#[derive(Deserialize)]
struct Suggestions {
    tracks: Vec<Suggestion>,
}

const PICK_SYSTEM: &str = "\
You are a music expert building a listening queue from a natural-language request. \
Name real, specific, commercially released tracks you are confident exist — a request is \
better served by twelve tracks that exist than twenty where half are invented. \
Prefer the original release over a remaster, live cut, or cover unless asked otherwise. \
Give the primary artist only, without featured credits, and the track title without any \
parenthetical suffix. Spread across artists: at most two tracks by any one of them. \
Honour every constraint in the request — era, language, genre, energy, setting.";

/// Ask Claude to name tracks fitting a free-text request.
///
/// These are *names*, not catalogue entries: nothing here is playable until the caller matches
/// each one against Spotify and discards what doesn't resolve. Returns an empty vec on any
/// problem, so the caller falls back to plain search rather than failing the query.
pub async fn suggest_tracks(query: &str, want: usize) -> Vec<Suggestion> {
    let Ok(key) = std::env::var("ANTHROPIC_API_KEY") else {
        return Vec::new();
    };

    // A JSON schema, rather than asking for JSON and hoping: the reply is structurally valid or
    // the request fails, which retires the "strip a stray markdown fence" guesswork below.
    let body = serde_json::json!({
        "model": PICK_MODEL,
        "max_tokens": 4000,
        "system": PICK_SYSTEM,
        // Brief thinking earns its latency on a subtle request ("for a rainy drive"); low effort
        // keeps a plain one from turning the search box into a spinner.
        "thinking": { "type": "adaptive" },
        "output_config": {
            "effort": "low",
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "tracks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "artist": { "type": "string" },
                                    "title": { "type": "string" },
                                },
                                "required": ["artist", "title"],
                                "additionalProperties": false,
                            },
                        },
                    },
                    "required": ["tracks"],
                    "additionalProperties": false,
                },
            },
        },
        "messages": [{
            "role": "user",
            "content": format!("Name up to {want} tracks for this request:\n\n{query}"),
        }],
    });

    let http = reqwest::Client::new();
    let resp = match http
        .post(ENDPOINT)
        .header("x-api-key", key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        // Longer than the mood call: this one thinks, and it's the user's actual query rather
        // than a background refinement.
        .timeout(std::time::Duration::from_secs(30))
        .json(&body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("claude: suggest failed ({e}) — falling back to plain search");
            return Vec::new();
        }
    };

    if !resp.status().is_success() {
        let status = resp.status();
        let detail = resp.text().await.unwrap_or_default();
        tracing::warn!("claude: suggest HTTP {status}: {detail}");
        return Vec::new();
    }

    let Ok(parsed) = resp.json::<ApiResponse>().await else {
        tracing::warn!("claude: suggest reply was not the expected envelope");
        return Vec::new();
    };

    // With adaptive thinking the first block is a thinking block, so take the first block that
    // actually carries text rather than assuming position 0.
    let text = parsed
        .content
        .iter()
        .map(|b| b.text.trim())
        .find(|t| !t.is_empty())
        .unwrap_or_default();

    match serde_json::from_str::<Suggestions>(text) {
        Ok(s) => {
            tracing::info!("claude: “{query}” → {} suggestions", s.tracks.len());
            s.tracks
        }
        Err(e) => {
            tracing::warn!("claude: suggest bad JSON ({e}): {text}");
            Vec::new()
        }
    }
}
