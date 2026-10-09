//! Shared HTTP plumbing: a per-target reqwest client pinned to validated addresses, and a
//! capped body reader. Used by the fetcher and the search providers.

use std::time::Duration;

use futures::StreamExt;

use crate::ssrf::ValidatedTarget;
use crate::ToolsError;

/// Build a client that connects only to `target.addrs` for `target.host`, never follows
/// redirects, ignores proxy environment variables (a proxy would bypass the pin) and gives up
/// at `timeout`.
pub(crate) fn pinned_client(
    target: &ValidatedTarget,
    timeout: Duration,
    user_agent: &str,
) -> Result<reqwest::Client, ToolsError> {
    let connect_timeout = timeout.min(Duration::from_secs(10));
    reqwest::Client::builder()
        .user_agent(user_agent)
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .timeout(timeout)
        .connect_timeout(connect_timeout)
        .resolve_to_addrs(&target.host, &target.addrs)
        .build()
        .map_err(|e| ToolsError::Fetch(format!("client build failed: {e}")))
}

/// Read at most `cap` bytes of the body. Returns `(bytes, truncated)`.
pub(crate) async fn read_capped(
    resp: reqwest::Response,
    cap: usize,
) -> Result<(Vec<u8>, bool), ToolsError> {
    let mut out: Vec<u8> = Vec::with_capacity(cap.min(64 * 1024));
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| ToolsError::Fetch(format!("body read failed: {e}")))?;
        let room = cap.saturating_sub(out.len());
        if chunk.len() >= room {
            out.extend_from_slice(&chunk[..room]);
            return Ok((out, true));
        }
        out.extend_from_slice(&chunk);
    }
    Ok((out, false))
}

/// Lowercase media type without parameters, e.g. `text/html`.
pub(crate) fn media_type(content_type: &str) -> String {
    content_type
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase()
}
