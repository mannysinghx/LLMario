//! Fetcher, SSRF-on-redirect, robots, caps, extraction-through-fetch and search providers,
//! all against a local `std::net::TcpListener` server. No live internet.

mod common;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use common::{spawn, Route, TestServer};
use llmario_engine_tools::builtin::{WebFetchTool, WebSearchTool};
use llmario_engine_tools::search::{GenericJsonEndpointProvider, JsonPaths, SearchConfig};
use llmario_engine_tools::ssrf::StaticResolver;
use llmario_engine_tools::{
    FetchConfig, SearchProvider, SearxngProvider, SsrfConfig, ToolSource, ToolsError, WebFetcher,
};
use serde_json::json;

const PAGE: &str = r#"<html><head><title>Test Page</title></head><body>
<p style="display:none">HIDDEN</p>
<article><h1>Hello</h1>
<p>This is a paragraph of visible text that is long enough for the extractor to keep it as the main content of the page, and it links to <a href="/other">another page</a>.</p>
<p>A second paragraph with more words so Readability is confident about the article body and does not throw it away.</p></article>
</body></html>"#;

fn routes() -> HashMap<String, Route> {
    let mut r = HashMap::new();
    r.insert("/page".into(), Route::html(PAGE));
    r.insert("/other".into(), Route::html("<html><head><title>Other</title></head><body><p>Other page body with enough text to be extracted properly here.</p></body></html>"));
    r.insert("/plain".into(), Route::text("just text\n\n\n\nmore"));
    r.insert(
        "/pdf".into(),
        Route::bytes("application/pdf", b"%PDF-1.4".to_vec()),
    );
    r.insert("/big".into(), Route::text(&"x".repeat(5000)));
    r.insert("/slow".into(), Route::Slow(Duration::from_secs(3)));
    r.insert(
        "/robots.txt".into(),
        Route::text("User-agent: llmario-engine\nDisallow: /private\n\nUser-agent: *\nAllow: /\n"),
    );
    r.insert("/private/x".into(), Route::text("secret"));
    r.insert(
        "/to-metadata".into(),
        Route::Redirect("http://169.254.169.254/latest/meta-data/".into()),
    );
    r.insert(
        "/to-loopback-ip".into(),
        Route::Redirect("http://127.0.0.1:{PORT}/secret".into()),
    );
    r.insert(
        "/to-private-dns".into(),
        Route::Redirect("http://evil.test/".into()),
    );
    r.insert("/to-relative".into(), Route::Redirect("/page".into()));
    r.insert("/loop".into(), Route::Redirect("/loop".into()));
    r.insert(
        "/to-ftp".into(),
        Route::Redirect("ftp://public.test/".into()),
    );
    r.insert(
        "/to-port".into(),
        Route::Redirect("http://public.test:9/".into()),
    );
    r.insert("/secret".into(), Route::text("secret"));
    r.insert("/500".into(), Route::status(500));
    r.insert("/search".into(), Route::json(r#"{"results":[{"title":"A","url":"https://a.example/1","content":"first","publishedDate":"2024"},{"title":"B","url":"https://b.example/2","content":"second"}]}"#));
    r.insert("/custom".into(), Route::json(r#"{"data":{"items":[{"name":"N","link":{"href":"https://n.example/"},"summary":"s"}]}}"#));
    r
}

fn resolver() -> Arc<StaticResolver> {
    Arc::new(
        StaticResolver::new()
            .with("public.test", &["127.0.0.1".parse().unwrap()])
            .with("evil.test", &["10.0.0.1".parse().unwrap()]),
    )
}

fn fetcher(
    server: &TestServer,
    respect_robots: bool,
    deadline: Duration,
    max_bytes: usize,
) -> WebFetcher {
    let cfg = FetchConfig {
        max_bytes,
        deadline,
        respect_robots,
        ssrf: SsrfConfig::default()
            .with_port(server.port())
            .with_exempt_host("public.test"),
        ..FetchConfig::default()
    };
    WebFetcher::with_resolver(cfg, resolver())
}

fn server() -> TestServer {
    spawn(routes())
}

#[tokio::test]
async fn fetches_html_with_pinned_dns_and_user_agent() {
    let s = server();
    let f = fetcher(&s, true, Duration::from_secs(10), 600_000);
    let doc = f.fetch(&s.url("public.test", "/page")).await.unwrap();
    assert_eq!(doc.status, 200);
    assert!(doc.is_html());
    assert!(!doc.truncated);
    assert!(doc.body.contains("Hello"));
    assert_eq!(doc.final_url, s.url("public.test", "/page"));
    let hits = s.hits();
    assert!(hits.iter().any(|h| h.contains("/robots.txt")), "{hits:?}");
    let page_hit = hits.iter().find(|h| h.contains(" /page ")).unwrap();
    assert!(
        page_hit.starts_with(&format!("public.test:{} ", s.port())),
        "host header carries the name: {page_hit}"
    );
    assert!(page_hit.contains("ua=llmario-engine/"), "{page_hit}");
}

#[tokio::test]
async fn redirects_are_revalidated_hop_by_hop() {
    let s = server();
    let f = fetcher(&s, false, Duration::from_secs(10), 600_000);
    let hits_before = s.hits().len();

    let r = f.fetch(&s.url("public.test", "/to-metadata")).await;
    assert!(
        matches!(r, Err(ToolsError::Ssrf(ref m)) if m.contains("169.254.169.254")),
        "{r:?}"
    );

    let r = f.fetch(&s.url("public.test", "/to-loopback-ip")).await;
    assert!(
        matches!(r, Err(ToolsError::Ssrf(ref m)) if m.contains("127.0.0.1")),
        "{r:?}"
    );

    let r = f.fetch(&s.url("public.test", "/to-private-dns")).await;
    assert!(
        matches!(r, Err(ToolsError::Ssrf(ref m)) if m.contains("10.0.0.1")),
        "{r:?}"
    );

    let r = f.fetch(&s.url("public.test", "/to-ftp")).await;
    assert!(
        matches!(r, Err(ToolsError::Ssrf(ref m)) if m.contains("scheme")),
        "{r:?}"
    );

    let r = f.fetch(&s.url("public.test", "/to-port")).await;
    assert!(
        matches!(r, Err(ToolsError::Ssrf(ref m)) if m.contains("port 9")),
        "{r:?}"
    );

    // None of the blocked targets was contacted: only the redirecting paths were hit.
    let hits = s.hits();
    assert_eq!(hits.len() - hits_before, 5, "{hits:?}");
    assert!(!hits.iter().any(|h| h.contains("/secret")), "{hits:?}");

    let doc = f
        .fetch(&s.url("public.test", "/to-relative"))
        .await
        .unwrap();
    assert_eq!(doc.redirects, vec![s.url("public.test", "/page")]);
    assert_eq!(doc.final_url, s.url("public.test", "/page"));

    let r = f.fetch(&s.url("public.test", "/loop")).await;
    assert!(matches!(r, Err(ToolsError::TooManyRedirects(5))), "{r:?}");
}

#[tokio::test]
async fn direct_private_targets_never_connect() {
    let s = server();
    let f = fetcher(&s, false, Duration::from_secs(10), 600_000);
    for u in [
        format!("http://127.0.0.1:{}/page", s.port()),
        format!("http://localhost:{}/page", s.port()),
        format!("http://[::1]:{}/page", s.port()),
        format!("http://2130706433:{}/page", s.port()),
        format!("http://evil.test:{}/page", s.port()),
    ] {
        let r = f.fetch(&u).await;
        assert!(
            matches!(r, Err(ToolsError::Ssrf(_)) | Err(ToolsError::Fetch(_))),
            "{u}: {r:?}"
        );
    }
    assert!(s.hits().is_empty(), "{:?}", s.hits());
}

#[tokio::test]
async fn robots_byte_cap_content_type_and_deadline() {
    let s = server();
    let f = fetcher(&s, true, Duration::from_secs(10), 1000);
    let r = f.fetch(&s.url("public.test", "/private/x")).await;
    assert!(matches!(r, Err(ToolsError::RobotsDisallowed(_))), "{r:?}");
    assert!(
        !s.hits().iter().any(|h| h.contains("/private/x")),
        "robots consulted before the request"
    );

    let doc = f.fetch(&s.url("public.test", "/big")).await.unwrap();
    assert!(doc.truncated);
    assert_eq!(doc.bytes, 1000);
    assert_eq!(doc.body.len(), 1000);

    let r = f.fetch(&s.url("public.test", "/pdf")).await;
    assert!(
        matches!(r, Err(ToolsError::ContentType(ref m)) if m == "application/pdf"),
        "{r:?}"
    );

    let r = f.fetch(&s.url("public.test", "/500")).await;
    assert!(
        matches!(r, Err(ToolsError::Fetch(ref m)) if m.contains("HTTP 500")),
        "{r:?}"
    );

    let doc = f.fetch(&s.url("public.test", "/plain")).await.unwrap();
    assert_eq!(doc.media_type, "text/plain");
    assert!(!doc.is_html());

    let robots_hits = s
        .hits()
        .iter()
        .filter(|h| h.contains("/robots.txt"))
        .count();
    assert_eq!(robots_hits, 1, "robots.txt cached per origin");

    let quick = fetcher(&s, false, Duration::from_millis(500), 1000);
    let r = quick.fetch(&s.url("public.test", "/slow")).await;
    assert!(
        matches!(r, Err(ToolsError::Deadline(_)) | Err(ToolsError::Fetch(_))),
        "{r:?}"
    );

    let ignore_robots = fetcher(&s, false, Duration::from_secs(10), 1000);
    assert_eq!(
        ignore_robots
            .fetch(&s.url("public.test", "/private/x"))
            .await
            .unwrap()
            .body,
        "secret"
    );
}

#[tokio::test]
async fn web_fetch_tool_builds_pages_follows_links_and_caches() {
    let s = server();
    let tool = WebFetchTool::new(Arc::new(fetcher(
        &s,
        false,
        Duration::from_secs(10),
        600_000,
    )));
    let r = tool
        .call("call_1", &json!({"url": s.url("public.test", "/page")}))
        .await
        .unwrap();
    assert_eq!(r.source, ToolSource::Web);
    assert_eq!(
        r.url.as_deref(),
        Some(s.url("public.test", "/page").as_str())
    );
    assert!(
        r.text
            .starts_with("[cursor 0]\nTest Page (public.test)\n**viewing lines [0 - "),
        "{}",
        r.text
    );
    assert!(r.text.contains("L0: "), "{}", r.text);
    assert!(!r.text.contains("HIDDEN"));
    assert!(
        r.text.contains("【0†another page†public.test】"),
        "{}",
        r.text
    );

    let r = tool
        .call("call_2", &json!({"cursor": 0, "find": "paragraph"}))
        .await
        .unwrap();
    assert!(
        r.text.contains("Find results for pattern: `paragraph`"),
        "{}",
        r.text
    );

    let r = tool
        .call("call_3", &json!({"cursor": 0, "link": 0}))
        .await
        .unwrap();
    assert!(
        r.text.starts_with("[cursor 1]\nOther (public.test)"),
        "{}",
        r.text
    );
    assert_eq!(
        r.url.as_deref(),
        Some(s.url("public.test", "/other").as_str())
    );

    let before = s.hits().len();
    let r = tool
        .call(
            "call_4",
            &json!({"url": s.url("public.test", "/page"), "loc": 1, "num_tokens": 64}),
        )
        .await
        .unwrap();
    assert_eq!(s.hits().len(), before, "cache hit: no new request");
    assert!(r.text.contains("**viewing lines [1 - "), "{}", r.text);

    let r = tool
        .call("call_5", &json!({"url": s.url("public.test", "/plain")}))
        .await
        .unwrap();
    assert!(
        r.text.contains("L0: just text\nL1: \nL2: more"),
        "{}",
        r.text
    );

    assert!(matches!(
        tool.call("c", &json!({})).await,
        Err(ToolsError::Arguments(_))
    ));
    assert!(matches!(
        tool.call("c", &json!({"cursor": 99})).await,
        Err(ToolsError::Arguments(_))
    ));
    assert!(matches!(
        tool.call("c", &json!({"cursor": 0, "link": 42})).await,
        Err(ToolsError::Arguments(_))
    ));
    assert!(matches!(
        tool.call("c", &json!({"url": "http://169.254.169.254/"}))
            .await,
        Err(ToolsError::Ssrf(_))
    ));
}

#[tokio::test]
async fn search_providers_over_http() {
    let s = server();
    let base = format!("http://127.0.0.1:{}", s.port());
    let searx = SearxngProvider::new(&base, SearchConfig::default()).unwrap();
    let results = searx.search("rust", 10).await.unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].title, "A");
    assert_eq!(results[0].published.as_deref(), Some("2024"));
    let hit = s
        .hits()
        .iter()
        .find(|h| h.contains("/search?"))
        .cloned()
        .unwrap();
    assert!(hit.contains("q=rust&format=json"), "{hit}");

    let tool = WebSearchTool::new(Box::new(searx));
    let r = tool
        .call("c1", &json!({"query": "rust", "max_results": 1}))
        .await
        .unwrap();
    assert_eq!(r.source, ToolSource::Search);
    assert!(
        r.text
            .starts_with("# Search results for: rust\n\n【0†A†a.example】\n"),
        "{}",
        r.text
    );
    assert!(!r.text.contains("【1†"));
    assert!(matches!(
        tool.call("c", &json!({"query": "  "})).await,
        Err(ToolsError::Arguments(_))
    ));

    // Private hosts are refused when the flag is off.
    let strict = SearxngProvider::new(
        &base,
        SearchConfig {
            allow_private_search_host: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(matches!(
        strict.search("x", 5).await,
        Err(ToolsError::Ssrf(_))
    ));

    let generic = GenericJsonEndpointProvider::new(
        "custom",
        &format!("{base}/custom?q={{query}}&n={{max}}"),
        JsonPaths {
            results: "data.items".into(),
            title: "name".into(),
            url: "link.href".into(),
            snippet: "summary".into(),
            published: None,
        },
        SearchConfig::default(),
    )
    .unwrap();
    let results = generic.search("a b", 3).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].url, "https://n.example/");
    assert!(
        s.hits().iter().any(|h| h.contains("/custom?q=a+b&n=3")),
        "{:?}",
        s.hits()
    );

    // A non-JSON or missing endpoint is a provider error, not a panic.
    let bad = SearxngProvider::new(&format!("{base}/nowhere/"), SearchConfig::default()).unwrap();
    assert!(matches!(
        bad.search("x", 5).await,
        Err(ToolsError::Search(_))
    ));
}
