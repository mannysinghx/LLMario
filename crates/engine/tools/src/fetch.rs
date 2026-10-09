//! `web_fetch`: SSRF-guarded fetching.
//!
//! Controls (after safewebfetch): `http`/`https` only, ports 80/443 unless configured, every
//! resolved address validated and the connection pinned to those addresses, redirects followed
//! manually and re-validated hop by hop (max 5), text content types only, a byte cap (600 KB)
//! and a total deadline (20 s), `robots.txt` honoured for `llmario-engine` (on by default), no
//! proxy environment, no content decoding (so no decompression bombs).
//!
//! The fetcher runs in-process; the OS sandbox for a fetcher subprocess lands with the server
//! integration (see `README.md`).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use tokio::sync::Mutex;

use crate::http::{media_type, pinned_client, read_capped};
use crate::ssrf::{DnsResolver, SsrfConfig, SsrfGuard, ValidatedTarget};
use crate::ToolsError;

/// Product token used to match `robots.txt` user-agent groups.
pub const ROBOTS_AGENT: &str = "llmario-engine";

/// Fetcher configuration.
#[derive(Debug, Clone)]
pub struct FetchConfig {
    /// Body cap in bytes (default 600,000).
    pub max_bytes: usize,
    /// Total deadline including redirects and `robots.txt` (default 20 s).
    pub deadline: Duration,
    /// Maximum redirects (default 5).
    pub max_redirects: usize,
    /// Consult `robots.txt` (default true).
    pub respect_robots: bool,
    /// `User-Agent` header.
    pub user_agent: String,
    /// Allowed media types (exact) besides any `text/*`.
    pub allowed_media_types: Vec<String>,
    /// SSRF guard configuration.
    pub ssrf: SsrfConfig,
}

impl Default for FetchConfig {
    fn default() -> Self {
        Self {
            max_bytes: 600_000,
            deadline: Duration::from_secs(20),
            max_redirects: 5,
            respect_robots: true,
            user_agent: crate::user_agent(),
            allowed_media_types: [
                "application/xhtml+xml",
                "application/xml",
                "application/json",
                "application/ld+json",
                "application/rss+xml",
                "application/atom+xml",
                "application/x-ndjson",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            ssrf: SsrfConfig::default(),
        }
    }
}

impl FetchConfig {
    /// Is this media type fetchable?
    pub fn allows_media_type(&self, media: &str) -> bool {
        media.starts_with("text/") || self.allowed_media_types.iter().any(|m| m == media)
    }
}

/// A fetched text document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchedDocument {
    /// URL as requested.
    pub requested_url: String,
    /// URL after redirects.
    pub final_url: String,
    /// Every redirect target in order (validated before being followed).
    pub redirects: Vec<String>,
    /// HTTP status of the final response.
    pub status: u16,
    /// Media type (lowercase, no parameters).
    pub media_type: String,
    /// Body decoded as UTF-8 (lossy).
    pub body: String,
    /// Bytes received for the body.
    pub bytes: usize,
    /// The body hit the byte cap.
    pub truncated: bool,
    /// Fetch time.
    pub fetched_at: DateTime<Utc>,
}

impl FetchedDocument {
    /// True for HTML-ish media types that go through Readability.
    pub fn is_html(&self) -> bool {
        matches!(
            self.media_type.as_str(),
            "text/html" | "application/xhtml+xml"
        )
    }
}

/// `robots.txt` cache entry per origin.
#[derive(Debug, Clone)]
enum RobotsEntry {
    /// No usable robots file (404, error, non-text): everything allowed.
    Absent,
    /// Raw file bytes.
    Rules(Arc<Vec<u8>>),
}

/// The fetcher.
#[derive(Debug)]
pub struct WebFetcher {
    cfg: FetchConfig,
    guard: SsrfGuard,
    robots: Mutex<HashMap<String, RobotsEntry>>,
}

impl WebFetcher {
    /// Fetcher with the system resolver.
    pub fn new(cfg: FetchConfig) -> Self {
        let guard = SsrfGuard::new(cfg.ssrf.clone());
        Self {
            cfg,
            guard,
            robots: Mutex::new(HashMap::new()),
        }
    }

    /// Fetcher with a custom resolver (tests, DNS-over-HTTPS, ...).
    pub fn with_resolver(cfg: FetchConfig, resolver: Arc<dyn DnsResolver>) -> Self {
        let guard = SsrfGuard::with_resolver(cfg.ssrf.clone(), resolver);
        Self {
            cfg,
            guard,
            robots: Mutex::new(HashMap::new()),
        }
    }

    /// Configuration.
    pub fn config(&self) -> &FetchConfig {
        &self.cfg
    }

    /// The guard (shared with search providers when they reuse the fetcher's settings).
    pub fn guard(&self) -> &SsrfGuard {
        &self.guard
    }

    /// Fetch `url` under every control. Robots denial, SSRF blocks, content-type and size
    /// violations are errors; a capped body is returned with `truncated = true`.
    pub async fn fetch(&self, url: &str) -> Result<FetchedDocument, ToolsError> {
        let start = Instant::now();
        let deadline = self.cfg.deadline;
        let result = tokio::time::timeout(deadline, self.fetch_inner(url, start)).await;
        match result {
            Ok(r) => r,
            Err(_) => Err(ToolsError::Deadline(deadline)),
        }
    }

    fn remaining(&self, start: Instant) -> Result<Duration, ToolsError> {
        let elapsed = start.elapsed();
        if elapsed >= self.cfg.deadline {
            return Err(ToolsError::Deadline(self.cfg.deadline));
        }
        Ok(self.cfg.deadline - elapsed)
    }

    async fn fetch_inner(&self, url: &str, start: Instant) -> Result<FetchedDocument, ToolsError> {
        let mut current = url.trim().to_string();
        let mut redirects = Vec::new();
        for _hop in 0..=self.cfg.max_redirects {
            let target = self.guard.validate(&current).await?;
            if self.cfg.respect_robots && !self.robots_allow(&target, start).await? {
                return Err(ToolsError::RobotsDisallowed(target.url.to_string()));
            }
            let remaining = self.remaining(start)?;
            let client = pinned_client(&target, remaining, &self.cfg.user_agent)?;
            let resp = client
                .get(target.url.clone())
                .header(
                    reqwest::header::ACCEPT,
                    "text/html, application/xhtml+xml, text/plain;q=0.9, application/json;q=0.8, */*;q=0.5",
                )
                .send()
                .await
                .map_err(|e| ToolsError::Fetch(format!("request to {} failed: {e}", target.host)))?;
            let status = resp.status();
            if status.is_redirection() {
                let Some(location) = resp.headers().get(reqwest::header::LOCATION) else {
                    return Err(ToolsError::Fetch(format!(
                        "redirect {status} without Location"
                    )));
                };
                let location = location
                    .to_str()
                    .map_err(|_| ToolsError::Fetch("non-ASCII Location header".into()))?;
                let next = target.url.join(location).map_err(|e| {
                    ToolsError::Url(format!("bad redirect target {location:?}: {e}"))
                })?;
                redirects.push(next.to_string());
                current = next.to_string();
                continue;
            }
            let content_type = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("text/plain")
                .to_string();
            let media = media_type(&content_type);
            if !self.cfg.allows_media_type(&media) {
                return Err(ToolsError::ContentType(media));
            }
            let (bytes, truncated) = read_capped(resp, self.cfg.max_bytes).await?;
            if !status.is_success() {
                return Err(ToolsError::Fetch(format!(
                    "HTTP {} from {}",
                    status.as_u16(),
                    target.host
                )));
            }
            return Ok(FetchedDocument {
                requested_url: url.trim().to_string(),
                final_url: target.url.to_string(),
                redirects,
                status: status.as_u16(),
                media_type: media,
                bytes: bytes.len(),
                body: String::from_utf8_lossy(&bytes).into_owned(),
                truncated,
                fetched_at: Utc::now(),
            });
        }
        Err(ToolsError::TooManyRedirects(self.cfg.max_redirects))
    }

    /// Whether `robots.txt` of the target's origin allows the URL for [`ROBOTS_AGENT`].
    async fn robots_allow(
        &self,
        target: &ValidatedTarget,
        start: Instant,
    ) -> Result<bool, ToolsError> {
        let origin = format!("{}://{}:{}", target.url.scheme(), target.host, target.port);
        let entry = {
            let cached = self.robots.lock().await.get(&origin).cloned();
            match cached {
                Some(e) => e,
                None => {
                    let e = self.load_robots(target, start).await;
                    self.robots.lock().await.insert(origin, e.clone());
                    e
                }
            }
        };
        match entry {
            RobotsEntry::Absent => Ok(true),
            RobotsEntry::Rules(bytes) => match texting_robots::Robot::new(ROBOTS_AGENT, &bytes) {
                Ok(robot) => Ok(robot.allowed(target.url.as_str())),
                Err(_) => Ok(true),
            },
        }
    }

    async fn load_robots(&self, target: &ValidatedTarget, start: Instant) -> RobotsEntry {
        let Ok(remaining) = self.remaining(start) else {
            return RobotsEntry::Absent;
        };
        let mut robots_url = target.url.clone();
        robots_url.set_path("/robots.txt");
        robots_url.set_query(None);
        robots_url.set_fragment(None);
        let robots_target = ValidatedTarget {
            url: robots_url.clone(),
            ..target.clone()
        };
        let Ok(client) = pinned_client(
            &robots_target,
            remaining.min(Duration::from_secs(5)),
            &self.cfg.user_agent,
        ) else {
            return RobotsEntry::Absent;
        };
        let Ok(resp) = client.get(robots_url).send().await else {
            return RobotsEntry::Absent;
        };
        if !resp.status().is_success() {
            return RobotsEntry::Absent;
        }
        let media = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(media_type)
            .unwrap_or_else(|| "text/plain".into());
        if !media.starts_with("text/") {
            return RobotsEntry::Absent;
        }
        match read_capped(resp, 512 * 1024).await {
            Ok((bytes, _)) => RobotsEntry::Rules(Arc::new(bytes)),
            Err(_) => RobotsEntry::Absent,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_types() {
        let c = FetchConfig::default();
        assert!(c.allows_media_type("text/html"));
        assert!(c.allows_media_type("text/plain"));
        assert!(c.allows_media_type("application/json"));
        assert!(!c.allows_media_type("application/pdf"));
        assert!(!c.allows_media_type("application/zip"));
        assert!(!c.allows_media_type("image/png"));
        assert_eq!(c.max_bytes, 600_000);
        assert_eq!(c.deadline, Duration::from_secs(20));
        assert_eq!(c.max_redirects, 5);
        assert!(c.respect_robots);
        assert!(c.user_agent.starts_with("llmario-engine/"));
    }
}
