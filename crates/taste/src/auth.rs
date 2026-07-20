//! Finding a credential for the Anthropic API.
//!
//! An unset `ANTHROPIC_API_KEY` does not mean there is no credential. Resolution order:
//!
//! 1. `ANTHROPIC_API_KEY` (env, or `Nocturne/.env`) → sent as `x-api-key`.
//! 2. Claude Code's own sign-in (`~/.claude/.credentials.json` → `claudeAiOauth.accessToken`) →
//!    sent as `Authorization: Bearer` plus the `anthropic-beta: oauth-2025-04-20` header that
//!    OAuth tokens require. Claude Code keeps this fresh; there's no separate login to do.
//! 3. The `ant` CLI's OAuth profile (`ant auth print-credentials --access-token`).
//!
//! Adapted from Scry's `claude.rs`, which solved this first.
//!
//! Tokens live in the request path only. Nothing here is logged, displayed, or written to disk —
//! a credential that turns up in a log is a credential you have to rotate.

/// How to authenticate one request.
pub enum Credential {
    /// A long-lived API key.
    ApiKey(String),
    /// A short-lived OAuth access token.
    Oauth(String),
}

/// `(credentials mtime, parsed token)`. The file is re-parsed only when its mtime moves — that is,
/// on sign-in or token refresh — rather than on every request.
static TOKEN_CACHE: std::sync::Mutex<Option<(std::time::SystemTime, Option<String>)>> =
    std::sync::Mutex::new(None);

fn claude_code_token() -> Option<String> {
    let home = std::env::var_os("HOME")?;
    let creds = std::path::PathBuf::from(&home).join(".claude/.credentials.json");
    let mtime = std::fs::metadata(&creds).ok().and_then(|m| m.modified().ok());

    let mut cache = TOKEN_CACHE.lock().unwrap_or_else(|p| p.into_inner());
    if let (Some(m), Some((cached_m, tok))) = (mtime, cache.as_ref()) {
        if *cached_m == m {
            return tok.clone();
        }
    }

    let tok = std::fs::read_to_string(&creds).ok().and_then(|t| {
        let v: serde_json::Value = serde_json::from_str(&t).ok()?;
        Some(v.get("claudeAiOauth")?.get("accessToken")?.as_str()?.to_string())
    });
    // Only memoize when the file is stat-able; a vanished file drops the memo so a re-created
    // sign-in is picked up immediately rather than after a restart.
    *cache = mtime.map(|m| (m, tok.clone()));
    tok
}

fn ant_token() -> Option<String> {
    let out = std::process::Command::new("ant")
        .args(["auth", "print-credentials", "--access-token"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let token = String::from_utf8(out.stdout).ok()?.trim().to_owned();
    (!token.is_empty()).then_some(token)
}

/// The env key, falling back to `Nocturne/.env` — the same file the Spotify client id lives in, so
/// one place to put secrets rather than two.
fn env_key() -> Option<String> {
    if let Ok(k) = std::env::var("ANTHROPIC_API_KEY") {
        if !k.trim().is_empty() {
            return Some(k);
        }
    }
    let text = std::fs::read_to_string(".env").ok()?;
    text.lines()
        .filter_map(|l| l.trim().strip_prefix("ANTHROPIC_API_KEY="))
        .map(|v| v.trim().trim_matches('"').to_string())
        .find(|v| !v.is_empty())
}

/// Resolve a credential, or `None` when the machine has none — in which case every Claude-backed
/// feature quietly falls back to its non-Claude path.
pub fn credential() -> Option<Credential> {
    if let Some(k) = env_key() {
        return Some(Credential::ApiKey(k));
    }
    if let Some(t) = claude_code_token() {
        return Some(Credential::Oauth(t));
    }
    ant_token().map(Credential::Oauth)
}

/// Is any credential available? Cheap enough to call per keystroke-driven search.
pub fn available() -> bool {
    credential().is_some()
}

/// Attach the credential to a request. OAuth tokens go on `Authorization: Bearer` *and* need the
/// oauth beta header — sending one as `x-api-key` is a 401.
pub fn authorize(req: reqwest::RequestBuilder, cred: &Credential) -> reqwest::RequestBuilder {
    match cred {
        Credential::ApiKey(k) => req.header("x-api-key", k),
        Credential::Oauth(t) => {
            req.bearer_auth(t).header("anthropic-beta", "oauth-2025-04-20")
        }
    }
}
