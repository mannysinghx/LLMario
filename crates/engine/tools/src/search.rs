//! `web_search` providers behind one result schema.
//!
//! * [`SearxngProvider`]: `GET {base}/search?q=…&format=json` (the JSON format must be enabled on
//!   the instance), parsing `results[].{title,url,content,publishedDate}`.
//! * [`GenericJsonEndpointProvider`]: a URL template plus dotted JSON paths, for any other
//!   self-hosted engine (YaCy, Meilisearch, OpenSearch, …).
//!
//! No hosted/paid APIs. The provider URL passes the same SSRF guard as `web_fetch`, except that
//! the search host itself may be private/loopback when `allow_private_search_host` is set
//! (default true) — a self-hosted instance normally lives on the LAN or on localhost.

use std::sync::Arc;
use std::time::Duration;

use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use url::Url;

use crate::http::{media_type, pinned_client, read_capped};
use crate::ssrf::{DnsResolver, SsrfConfig, SsrfGuard};
use crate::ToolsError;

/// Hard cap on results per query.
pub const MAX_RESULTS: usize = 10;

/// One normalised result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchResult {
    /// Title.
    pub title: String,
    /// Absolute URL.
    pub url: String,
    /// Snippet (truncated to the configured length).
    pub snippet: String,
    /// Publication date as the provider gave it, if any.
    pub published: Option<String>,
}

/// A search backend.
pub trait SearchProvider: Send + Sync {
    /// Provider name for provenance.
    fn name(&self) -> &str;
    /// Run a query; at most `max_results` (≤ [`MAX_RESULTS`]) results.
    fn search<'a>(
        &'a self,
        query: &'a str,
        max_results: usize,
    ) -> BoxFuture<'a, Result<Vec<SearchResult>, ToolsError>>;
}

/// Shared provider settings.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Let the search base URL resolve to a private/loopback address (default true).
    pub allow_private_search_host: bool,
    /// Extra port to allow for the search host (the base URL's port is always allowed).
    pub allow_http: bool,
    /// Per-query timeout (default 15 s).
    pub timeout: Duration,
    /// Maximum snippet length in characters (default 1,000).
    pub max_snippet_chars: usize,
    /// Response size cap (default 2 MB).
    pub max_response_bytes: usize,
    /// `User-Agent`.
    pub user_agent: String,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            allow_private_search_host: true,
            allow_http: true,
            timeout: Duration::from_secs(15),
            max_snippet_chars: 1_000,
            max_response_bytes: 2 * 1024 * 1024,
            user_agent: crate::user_agent(),
        }
    }
}

/// HTTP GET of a JSON document through the SSRF guard.
#[derive(Debug)]
struct JsonGetter {
    guard: SsrfGuard,
    cfg: SearchConfig,
}

impl JsonGetter {
    fn new(
        base: &Url,
        cfg: SearchConfig,
        resolver: Option<Arc<dyn DnsResolver>>,
    ) -> Result<Self, ToolsError> {
        let mut ssrf = SsrfConfig {
            allow_http: cfg.allow_http,
            ..SsrfConfig::default()
        };
        if let Some(port) = base.port_or_known_default() {
            ssrf = ssrf.with_port(port);
        }
        if cfg.allow_private_search_host {
            if let Some(host) = base.host_str() {
                ssrf = ssrf.with_exempt_host(host);
            }
        }
        let guard = match resolver {
            Some(r) => SsrfGuard::with_resolver(ssrf, r),
            None => SsrfGuard::new(ssrf),
        };
        Ok(Self { guard, cfg })
    }

    async fn get_json(&self, url: &str) -> Result<Value, ToolsError> {
        let fut = async {
            let target = self.guard.validate(url).await?;
            let client = pinned_client(&target, self.cfg.timeout, &self.cfg.user_agent)?;
            let resp = client
                .get(target.url.clone())
                .header(reqwest::header::ACCEPT, "application/json")
                .send()
                .await
                .map_err(|e| ToolsError::Search(format!("request failed: {e}")))?;
            let status = resp.status();
            let media = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .map(media_type)
                .unwrap_or_default();
            let (bytes, truncated) = read_capped(resp, self.cfg.max_response_bytes).await?;
            if !status.is_success() {
                return Err(ToolsError::Search(format!(
                    "HTTP {} from search provider{}",
                    status.as_u16(),
                    if status.as_u16() == 403 {
                        " (is the JSON format enabled on the instance?)"
                    } else {
                        ""
                    }
                )));
            }
            if truncated {
                return Err(ToolsError::Search("response exceeds size cap".into()));
            }
            if !(media.ends_with("json") || media.starts_with("text/")) {
                return Err(ToolsError::Search(format!(
                    "unexpected content type {media}"
                )));
            }
            serde_json::from_slice(&bytes)
                .map_err(|e| ToolsError::Search(format!("invalid JSON: {e}")))
        };
        tokio::time::timeout(self.cfg.timeout, fut)
            .await
            .unwrap_or(Err(ToolsError::Timeout(self.cfg.timeout)))
    }
}

fn truncate_chars(s: &str, max: usize) -> String {
    let s = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if s.chars().count() <= max {
        return s;
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

fn normalise(
    title: &str,
    url: &str,
    snippet: &str,
    published: Option<&str>,
    max_snippet: usize,
) -> Option<SearchResult> {
    let url = Url::parse(url.trim()).ok()?;
    if !matches!(url.scheme(), "http" | "https") {
        return None;
    }
    let title = truncate_chars(title, 300);
    Some(SearchResult {
        title: if title.is_empty() {
            url.to_string()
        } else {
            title
        },
        url: url.to_string(),
        snippet: truncate_chars(snippet, max_snippet),
        published: published
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty()),
    })
}

/// Self-hosted SearXNG.
#[derive(Debug)]
pub struct SearxngProvider {
    base: Url,
    getter: JsonGetter,
    max_snippet_chars: usize,
}

impl SearxngProvider {
    /// `base_url` is the instance root, e.g. `http://localhost:8080`.
    pub fn new(base_url: &str, cfg: SearchConfig) -> Result<Self, ToolsError> {
        Self::with_resolver(base_url, cfg, None)
    }

    /// With a custom resolver.
    pub fn with_resolver(
        base_url: &str,
        cfg: SearchConfig,
        resolver: Option<Arc<dyn DnsResolver>>,
    ) -> Result<Self, ToolsError> {
        let base = Url::parse(base_url.trim())
            .map_err(|e| ToolsError::Config(format!("searxng base url: {e}")))?;
        if !matches!(base.scheme(), "http" | "https") {
            return Err(ToolsError::Config(
                "searxng base url must be http(s)".into(),
            ));
        }
        let max_snippet_chars = cfg.max_snippet_chars;
        let getter = JsonGetter::new(&base, cfg, resolver)?;
        Ok(Self {
            base,
            getter,
            max_snippet_chars,
        })
    }

    /// The request URL for `query`.
    pub fn query_url(&self, query: &str) -> Result<Url, ToolsError> {
        let mut url = self
            .base
            .join("search")
            .map_err(|e| ToolsError::Config(format!("searxng base url: {e}")))?;
        url.query_pairs_mut()
            .append_pair("q", query)
            .append_pair("format", "json");
        Ok(url)
    }

    /// Parse a SearXNG JSON response.
    pub fn parse_response(
        json: &Value,
        max_results: usize,
        max_snippet_chars: usize,
    ) -> Vec<SearchResult> {
        let max = max_results.clamp(1, MAX_RESULTS);
        json.get("results")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|r| {
                        normalise(
                            r.get("title").and_then(Value::as_str).unwrap_or(""),
                            r.get("url").and_then(Value::as_str)?,
                            r.get("content").and_then(Value::as_str).unwrap_or(""),
                            r.get("publishedDate").and_then(Value::as_str),
                            max_snippet_chars,
                        )
                    })
                    .take(max)
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl SearchProvider for SearxngProvider {
    fn name(&self) -> &str {
        "searxng"
    }

    fn search<'a>(
        &'a self,
        query: &'a str,
        max_results: usize,
    ) -> BoxFuture<'a, Result<Vec<SearchResult>, ToolsError>> {
        Box::pin(async move {
            let url = self.query_url(query)?;
            let json = self.getter.get_json(url.as_str()).await?;
            Ok(Self::parse_response(
                &json,
                max_results,
                self.max_snippet_chars,
            ))
        })
    }
}

/// Dotted JSON paths into a provider's response. Array segments are numeric; an empty path
/// means "this value".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JsonPaths {
    /// Path to the results array (e.g. `data.items`; empty for a root array).
    pub results: String,
    /// Path to the title within one result.
    pub title: String,
    /// Path to the URL within one result.
    pub url: String,
    /// Path to the snippet within one result.
    pub snippet: String,
    /// Optional path to a publication date.
    #[serde(default)]
    pub published: Option<String>,
}

/// Look up a dotted path.
pub fn json_path<'a>(v: &'a Value, path: &str) -> Option<&'a Value> {
    let mut cur = v;
    for seg in path.split('.').filter(|s| !s.is_empty()) {
        cur = match cur {
            Value::Object(m) => m.get(seg)?,
            Value::Array(a) => a.get(seg.parse::<usize>().ok()?)?,
            _ => return None,
        };
    }
    Some(cur)
}

/// Any self-hosted JSON search endpoint.
#[derive(Debug)]
pub struct GenericJsonEndpointProvider {
    name: String,
    template: String,
    paths: JsonPaths,
    getter: JsonGetter,
    max_snippet_chars: usize,
}

impl GenericJsonEndpointProvider {
    /// `template` contains `{query}` and optionally `{max}`; both are percent-encoded on
    /// substitution. The base used for the SSRF exemption is the template with the
    /// placeholders replaced by `x`.
    pub fn new(
        name: &str,
        template: &str,
        paths: JsonPaths,
        cfg: SearchConfig,
    ) -> Result<Self, ToolsError> {
        Self::with_resolver(name, template, paths, cfg, None)
    }

    /// With a custom resolver.
    pub fn with_resolver(
        name: &str,
        template: &str,
        paths: JsonPaths,
        cfg: SearchConfig,
        resolver: Option<Arc<dyn DnsResolver>>,
    ) -> Result<Self, ToolsError> {
        if !template.contains("{query}") {
            return Err(ToolsError::Config(
                "endpoint template must contain {query}".into(),
            ));
        }
        let probe = template.replace("{query}", "x").replace("{max}", "1");
        let base = Url::parse(&probe)
            .map_err(|e| ToolsError::Config(format!("endpoint template: {e}")))?;
        let max_snippet_chars = cfg.max_snippet_chars;
        let getter = JsonGetter::new(&base, cfg, resolver)?;
        Ok(Self {
            name: name.to_string(),
            template: template.to_string(),
            paths,
            getter,
            max_snippet_chars,
        })
    }

    /// The request URL for `query`.
    pub fn query_url(&self, query: &str, max_results: usize) -> String {
        let q: String = url::form_urlencoded::byte_serialize(query.as_bytes()).collect();
        self.template
            .replace("{query}", &q)
            .replace("{max}", &max_results.clamp(1, MAX_RESULTS).to_string())
    }

    /// Parse a response with the configured paths.
    pub fn parse_response(&self, json: &Value, max_results: usize) -> Vec<SearchResult> {
        let max = max_results.clamp(1, MAX_RESULTS);
        let Some(items) = json_path(json, &self.paths.results).and_then(Value::as_array) else {
            return Vec::new();
        };
        items
            .iter()
            .filter_map(|r| {
                let get = |p: &str| json_path(r, p).and_then(Value::as_str);
                normalise(
                    get(&self.paths.title).unwrap_or(""),
                    get(&self.paths.url)?,
                    get(&self.paths.snippet).unwrap_or(""),
                    self.paths.published.as_deref().and_then(get),
                    self.max_snippet_chars,
                )
            })
            .take(max)
            .collect()
    }
}

impl SearchProvider for GenericJsonEndpointProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn search<'a>(
        &'a self,
        query: &'a str,
        max_results: usize,
    ) -> BoxFuture<'a, Result<Vec<SearchResult>, ToolsError>> {
        Box::pin(async move {
            let url = self.query_url(query, max_results);
            let json = self.getter.get_json(&url).await?;
            Ok(self.parse_response(&json, max_results))
        })
    }
}

/// Render results for the model, one numbered block per hit.
pub fn render_results(query: &str, results: &[SearchResult]) -> String {
    if results.is_empty() {
        return format!("No results for: {query}");
    }
    let mut s = format!("# Search results for: {query}\n\n");
    for (i, r) in results.iter().enumerate() {
        let domain = crate::page::domain_of(&r.url);
        s.push_str(&format!("【{i}†{}†{domain}】\n{}\n", r.title, r.url));
        if let Some(p) = &r.published {
            s.push_str(&format!("published: {p}\n"));
        }
        if !r.snippet.is_empty() {
            s.push_str(&r.snippet);
            s.push('\n');
        }
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    pub(crate) const SEARXNG_FIXTURE: &str = r#"{
      "query": "rust ssrf",
      "number_of_results": 3,
      "results": [
        {"title": "Server-side request forgery", "url": "https://en.wikipedia.org/wiki/SSRF", "content": "SSRF is a web vulnerability  where\n an attacker makes the server issue requests.", "publishedDate": "2024-01-02", "engine": "wikipedia"},
        {"title": "", "url": "https://owasp.org/ssrf", "content": "", "publishedDate": null},
        {"title": "bad scheme", "url": "ftp://files.example/x", "content": "ignored"},
        {"title": "no url", "content": "ignored"},
        {"title": "long", "url": "http://example.com/long", "content": "LONGSNIPPET"}
      ]
    }"#;

    #[test]
    fn searxng_parsing_normalises_and_caps() {
        let json: Value = serde_json::from_str(SEARXNG_FIXTURE).unwrap();
        let r = SearxngProvider::parse_response(&json, 10, 1000);
        assert_eq!(r.len(), 3);
        assert_eq!(r[0].title, "Server-side request forgery");
        assert_eq!(r[0].url, "https://en.wikipedia.org/wiki/SSRF");
        assert_eq!(
            r[0].snippet,
            "SSRF is a web vulnerability where an attacker makes the server issue requests."
        );
        assert_eq!(r[0].published.as_deref(), Some("2024-01-02"));
        assert_eq!(
            r[1].title, "https://owasp.org/ssrf",
            "empty title falls back to url"
        );
        assert_eq!(r[1].published, None);
        assert_eq!(r[2].snippet, "LONGSNIPPET");
        let r = SearxngProvider::parse_response(&json, 2, 5);
        assert_eq!(r.len(), 2);
        let r = SearxngProvider::parse_response(&json, 10, 5);
        assert_eq!(r[2].snippet, "LONG…");
        let mut many = json!({"results": []});
        for i in 0..30 {
            many["results"].as_array_mut().unwrap().push(json!({"title": i.to_string(), "url": format!("https://e.com/{i}"), "content": "c"}));
        }
        assert_eq!(
            SearxngProvider::parse_response(&many, 100, 10).len(),
            MAX_RESULTS
        );
        assert!(SearxngProvider::parse_response(&json!({"nope": 1}), 10, 10).is_empty());
    }

    #[test]
    fn searxng_query_url_and_private_host_exemption() {
        let p = SearxngProvider::new("http://localhost:8080/", SearchConfig::default()).unwrap();
        let u = p.query_url("rust ssrf & more").unwrap();
        assert_eq!(
            u.as_str(),
            "http://localhost:8080/search?q=rust+ssrf+%26+more&format=json"
        );
        assert!(p.getter.guard.config().exempt_hosts.contains("localhost"));
        assert!(p.getter.guard.config().allowed_ports.contains(&8080));
        let p = SearxngProvider::new(
            "https://searx.example/searx",
            SearchConfig {
                allow_private_search_host: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(p.getter.guard.config().exempt_hosts.is_empty());
        assert_eq!(p.query_url("x").unwrap().path(), "/search");
        assert!(SearxngProvider::new("ftp://x", SearchConfig::default()).is_err());
    }

    #[test]
    fn generic_provider_paths_and_template() {
        let paths = JsonPaths {
            results: "data.items".into(),
            title: "name".into(),
            url: "link.href".into(),
            snippet: "summary".into(),
            published: Some("meta.date".into()),
        };
        let p = GenericJsonEndpointProvider::new(
            "yacy",
            "http://127.0.0.1:8090/yacysearch.json?query={query}&maximumRecords={max}",
            paths,
            SearchConfig::default(),
        )
        .unwrap();
        assert_eq!(
            p.query_url("a b/c", 50),
            "http://127.0.0.1:8090/yacysearch.json?query=a+b%2Fc&maximumRecords=10"
        );
        let json = json!({"data": {"items": [
            {"name": "One", "link": {"href": "https://a.example/1"}, "summary": "s1", "meta": {"date": "2025"}},
            {"name": "Two", "link": {"href": "/relative"}, "summary": "s2"},
            {"name": "Three", "link": {"href": "https://a.example/3"}}
        ]}});
        let r = p.parse_response(&json, 10);
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].published.as_deref(), Some("2025"));
        assert_eq!(r[1].title, "Three");
        assert_eq!(r[1].snippet, "");
        assert_eq!(json_path(&json, "data.items.1.name").unwrap(), "Two");
        assert!(json_path(&json, "data.items.x").is_none());
        assert!(GenericJsonEndpointProvider::new(
            "n",
            "http://h/?q=",
            JsonPaths {
                results: "".into(),
                title: "t".into(),
                url: "u".into(),
                snippet: "s".into(),
                published: None
            },
            SearchConfig::default()
        )
        .is_err());
    }

    #[test]
    fn render_results_numbers_hits() {
        let r = vec![SearchResult {
            title: "T".into(),
            url: "https://www.x.org/p".into(),
            snippet: "snip".into(),
            published: Some("2020".into()),
        }];
        let s = render_results("q", &r);
        assert!(s.contains("【0†T†x.org】\nhttps://www.x.org/p\npublished: 2020\nsnip\n"));
        assert_eq!(render_results("q", &[]), "No results for: q");
    }
}
