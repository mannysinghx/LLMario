//! End-to-end tests: real gateway + supervisor + a real engine child process (the mock engine
//! built into the `llmario` binary). No GPU or model weights required.

use futures::StreamExt;
use llmario_core::{Config, ModelFormat, Paths, ProfileKind};
use llmario_registry::{ModelEntry, Registry};
use llmario_supervisor::{EngineAdapter, Supervisor};
use serde_json::{json, Value};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

const EXE: &str = env!("CARGO_BIN_EXE_llmario");

fn hardware() -> llmario_hardware::HardwareReport {
    static HW: OnceLock<llmario_hardware::HardwareReport> = OnceLock::new();
    HW.get_or_init(llmario_hardware::HardwareReport::detect)
        .clone()
}

fn mock_model(id: &str) -> ModelEntry {
    ModelEntry {
        id: id.into(),
        family: Some("mockfam".into()),
        format: ModelFormat::Mock,
        path: "/dev/null".into(),
        managed: false,
        source: None,
        license: Some("test".into()),
        architecture: Some("mock".into()),
        quantization: None,
        shape: None,
        chat_template: true,
        files: vec![],
        size_bytes: 64 * 1024 * 1024,
        added_at: "2026-09-28T00:00:00Z".into(),
    }
}

struct Env {
    _home: tempfile::TempDir,
    sup: Arc<Supervisor>,
    base: String,
    http: reqwest::Client,
}

impl Env {
    async fn new(ids: &[&str], tweak: impl FnOnce(&mut Config)) -> Self {
        let home = tempfile::tempdir().unwrap();
        let paths = Paths::at(home.path());
        paths.ensure().unwrap();
        let mut reg = Registry::load(&paths.registry_file()).unwrap();
        for id in ids {
            reg.insert(mock_model(id)).unwrap();
        }
        reg.save().unwrap();
        let mut cfg = Config::default();
        cfg.runtime.engine_start_timeout_secs = 20;
        cfg.runtime.queue_timeout_secs = 10;
        tweak(&mut cfg);
        let adapters: Vec<Arc<dyn EngineAdapter>> =
            vec![Arc::new(llmario_adapter_mock::MockAdapter {
                program: EXE.into(),
                args_prefix: vec!["mock-engine".into()],
            })];
        let sup = Supervisor::new(cfg, paths, hardware(), adapters).unwrap();
        let (addr, _h) = llmario_api::spawn_ephemeral(sup.clone()).await.unwrap();
        Env {
            _home: home,
            sup,
            base: format!("http://{addr}"),
            http: reqwest::Client::builder().no_proxy().build().unwrap(),
        }
    }

    async fn chat(&self, body: Value) -> reqwest::Response {
        self.http
            .post(format!("{}/v1/chat/completions", self.base))
            .json(&body)
            .send()
            .await
            .unwrap()
    }

    async fn engine_stats(&self) -> Value {
        let e = self.sup.loaded().await;
        let port = e.first().expect("an engine is loaded").port;
        self.http
            .get(format!("http://127.0.0.1:{port}/mock/stats"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap()
    }

    async fn metrics(&self) -> String {
        self.http
            .get(format!("{}/metrics", self.base))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap()
    }
}

// No Drop: when a test's runtime shuts down, engine monitor tasks are dropped and
// `kill_on_drop` terminates every engine child.

async fn sse_events(resp: reqwest::Response) -> Vec<String> {
    let text = resp.text().await.unwrap();
    text.split("\n\n")
        .filter_map(|e| e.strip_prefix("data: ").map(String::from))
        .collect()
}

fn user(content: &str) -> Value {
    json!([{"role": "user", "content": content}])
}

#[tokio::test]
async fn health_and_model_listing() {
    let env = Env::new(&["mock-a"], |_| {}).await;
    let h: Value = env
        .http
        .get(format!("{}/healthz", env.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(h["status"], "ok");
    let m: Value = env
        .http
        .get(format!("{}/v1/models", env.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(m["data"][0]["id"], "mock-a");
    assert_eq!(m["data"][0]["llmario"]["loaded"], false);
    let one = env
        .http
        .get(format!("{}/v1/models/nope", env.base))
        .send()
        .await
        .unwrap();
    assert_eq!(one.status(), 404);
}

#[tokio::test]
async fn non_streaming_rewrites_model_and_allowlists_fields() {
    let env = Env::new(&["mock-a"], |_| {}).await;
    let r = env
        .chat(json!({"model": "mock-a", "messages": user("hi"), "max_tokens": 5, "draft_model": "evil/x", "user": "u1", "temperature": 0}))
        .await;
    assert_eq!(r.status(), 200);
    assert_eq!(r.headers()["x-llmario-model"], "mock-a");
    assert_eq!(r.headers()["x-llmario-backend"], "mock");
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["model"], "mock-a");
    assert_eq!(v["usage"]["completion_tokens"], 5);
    assert_eq!(
        v["choices"][0]["message"]["content"],
        "tok0 tok1 tok2 tok3 tok4 "
    );

    let s = env.engine_stats().await;
    assert_eq!(
        s["last_model"], "mock-upstream-mock-a",
        "engine sees its own model name only"
    );
    let keys: Vec<String> = serde_json::from_value(s["last_keys"].clone()).unwrap();
    assert!(
        !keys.contains(&"draft_model".into()) && !keys.contains(&"user".into()),
        "{keys:?}"
    );
    assert!(keys.contains(&"temperature".into()));
}

#[tokio::test]
async fn streaming_relays_and_hides_unrequested_usage() {
    let env = Env::new(&["mock-a"], |_| {}).await;
    let ev = sse_events(
        env.chat(
            json!({"model": "mockfam", "stream": true, "messages": user("hi"), "max_tokens": 4}),
        )
        .await,
    )
    .await;
    assert_eq!(ev.last().unwrap(), "[DONE]");
    let chunks: Vec<Value> = ev[..ev.len() - 1]
        .iter()
        .map(|e| serde_json::from_str(e).unwrap())
        .collect();
    assert!(
        chunks.iter().all(|c| c["model"] == "mock-a"),
        "family name resolved to concrete id"
    );
    let text: String = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(text, "tok0 tok1 tok2 tok3 ");
    assert!(
        chunks.iter().all(|c| c.get("usage").is_none()),
        "usage not requested → not sent"
    );

    let ev = sse_events(
        env.chat(json!({"model": "mock-a", "stream": true, "stream_options": {"include_usage": true}, "messages": user("hi"), "max_tokens": 3}))
            .await,
    )
    .await;
    let usage: Vec<Value> = ev
        .iter()
        .filter_map(|e| serde_json::from_str::<Value>(e).ok())
        .filter(|c| c.get("usage").is_some())
        .collect();
    assert_eq!(usage.len(), 1);
    assert_eq!(usage[0]["usage"]["completion_tokens"], 3);

    let m = env.metrics().await;
    assert!(
        m.contains("llmario_requests_total{model=\"mock-a\",outcome=\"ok\"} 2"),
        "{m}"
    );
    assert!(
        m.contains("llmario_completion_tokens_total{model=\"mock-a\"} 7"),
        "{m}"
    );
}

#[tokio::test]
async fn client_disconnect_cancels_generation_and_frees_slot() {
    let env = Env::new(&["mock-a"], |_| {}).await;
    let resp = env.chat(json!({"model": "mock-a", "stream": true, "messages": user("__slow__ story"), "max_tokens": 200})).await;
    let mut body = resp.bytes_stream();
    let mut seen = 0;
    while let Some(Ok(_)) = body.next().await {
        seen += 1;
        if seen >= 3 {
            break;
        }
    }
    drop(body); // client goes away mid-stream

    let mut cancelled = false;
    for _ in 0..50 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        if env.engine_stats().await["cancelled"] == 1 {
            cancelled = true;
            break;
        }
    }
    assert!(
        cancelled,
        "engine observed the disconnect and stopped generating"
    );
    assert_eq!(
        env.sup.loaded().await[0].active_requests,
        0,
        "slot released"
    );
    assert!(env.metrics().await.contains("outcome=\"cancelled\"} 1"));

    // With the latency profile's single slot, a follow-up request is served immediately.
    let r = env
        .chat(json!({"model": "mock-a", "messages": user("hi"), "max_tokens": 2}))
        .await;
    assert_eq!(r.status(), 200);
}

#[tokio::test]
async fn engine_crash_is_reported_and_relaunched() {
    let env = Env::new(&["mock-a"], |_| {}).await;
    let r = env
        .chat(json!({"model": "mock-a", "messages": user("hi"), "max_tokens": 1}))
        .await;
    assert_eq!(r.status(), 200);
    let pid1 = env.sup.loaded().await[0].pid;

    let ev = sse_events(env.chat(json!({"model": "mock-a", "stream": true, "messages": user("__crash__"), "max_tokens": 50})).await).await;
    let err = ev
        .iter()
        .filter_map(|e| serde_json::from_str::<Value>(e).ok())
        .find(|v| v.get("error").is_some())
        .expect("error event");
    assert_eq!(err["error"]["code"], "engine_crashed");
    assert_eq!(ev.last().unwrap(), "[DONE]");

    let r = env
        .chat(json!({"model": "mock-a", "messages": user("hi again"), "max_tokens": 2}))
        .await;
    assert_eq!(r.status(), 200, "next request relaunches the engine");
    let pid2 = env.sup.loaded().await[0].pid;
    assert_ne!(pid1, pid2);
    assert!(env
        .metrics()
        .await
        .contains("outcome=\"engine_crashed\"} 1"));
}

#[tokio::test]
async fn structured_errors() {
    let env = Env::new(&["mock-a"], |_| {}).await;
    let r = env
        .chat(json!({"model": "missing", "messages": user("x")}))
        .await;
    assert_eq!(r.status(), 404);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["error"]["code"], "model_not_found");
    assert_eq!(v["error"]["type"], "invalid_request_error");

    let r = env.chat(json!({"model": "mock-a", "messages": user("x"), "tools": [{"type": "function", "function": {"name": "f"}}]})).await;
    assert_eq!(r.status(), 400);
    assert_eq!(
        r.json::<Value>().await.unwrap()["error"]["code"],
        "unsupported_feature"
    );

    let r = env
        .chat(json!({"model": "mock-a", "messages": user(&"word ".repeat(10_000))}))
        .await;
    assert_eq!(r.status(), 400);
    assert_eq!(
        r.json::<Value>().await.unwrap()["error"]["code"],
        "context_length_exceeded"
    );

    let r = env
        .http
        .post(format!("{}/v1/chat/completions", env.base))
        .body("not json")
        .header("content-type", "text/plain")
        .send()
        .await
        .unwrap();
    assert_eq!(
        r.status(),
        400,
        "non-JSON bodies (e.g. cross-site form posts) are rejected"
    );
}

#[tokio::test]
async fn insufficient_memory_is_refused_with_explanation() {
    let env = Env::new(&["mock-a"], |c| c.runtime.memory_limit_gb = Some(0.1)).await;
    let r = env
        .chat(json!({"model": "mock-a", "messages": user("x")}))
        .await;
    assert_eq!(r.status(), 507);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["error"]["code"], "insufficient_memory");
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("does not fit"));
    assert!(env.sup.loaded().await.is_empty(), "nothing was launched");
}

#[tokio::test]
async fn auth_and_dns_rebinding_guard() {
    let env = Env::new(&["mock-a"], |c| c.server.api_key = Some("s3cret".into())).await;
    let r = env
        .chat(json!({"model": "mock-a", "messages": user("x")}))
        .await;
    assert_eq!(r.status(), 401);
    let r = env
        .http
        .get(format!("{}/metrics", env.base))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 401, "metrics need the key too");
    let r = env
        .http
        .get(format!("{}/healthz", env.base))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "health stays open");
    let r = env
        .http
        .post(format!("{}/v1/chat/completions", env.base))
        .bearer_auth("s3cret")
        .json(&json!({"model": "mock-a", "messages": user("x"), "max_tokens": 1}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let r = env
        .http
        .get(format!("{}/healthz", env.base))
        .header("host", "attacker.example")
        .send()
        .await
        .unwrap();
    assert_eq!(
        r.status(),
        401,
        "non-loopback Host header rejected in loopback mode"
    );
}

#[tokio::test]
async fn single_slot_queue_times_out_with_503() {
    let env = Env::new(&["mock-a"], |c| {
        c.runtime.profile = ProfileKind::Latency;
        c.runtime.queue_timeout_secs = 1;
    })
    .await;
    // Load first so the timing below is about the slot, not the launch.
    assert_eq!(
        env.chat(json!({"model": "mock-a", "messages": user("x"), "max_tokens": 1}))
            .await
            .status(),
        200
    );
    let slow = env.chat(
        json!({"model": "mock-a", "stream": true, "messages": user("__slow__"), "max_tokens": 40}),
    );
    let (first, second) = tokio::join!(slow, async {
        tokio::time::sleep(Duration::from_millis(200)).await;
        env.chat(json!({"model": "mock-a", "messages": user("x"), "max_tokens": 1}))
            .await
    });
    assert_eq!(first.status(), 200);
    assert_eq!(second.status(), 503);
    assert!(second.headers().contains_key("retry-after"));
    assert_eq!(
        second.json::<Value>().await.unwrap()["error"]["code"],
        "server_busy"
    );
    drop(first);
}

#[tokio::test]
async fn one_model_resident_lru_swap() {
    let env = Env::new(&["mock-a", "mock-b"], |_| {}).await;
    assert_eq!(
        env.chat(json!({"model": "mock-a", "messages": user("x"), "max_tokens": 1}))
            .await
            .status(),
        200
    );
    let a_pid = env.sup.loaded().await[0].pid;
    assert_eq!(
        env.chat(json!({"model": "mock-b", "messages": user("x"), "max_tokens": 1}))
            .await
            .status(),
        200
    );
    let loaded = env.sup.loaded().await;
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].model, "mock-b");
    // The evicted engine process is really gone.
    assert!(
        !llmario_core::os::pid_alive(a_pid),
        "evicted engine {a_pid} still running"
    );
}

#[tokio::test]
async fn benchmark_harness_against_gateway() {
    let env = Env::new(&["mock-a"], |c| c.runtime.profile = ProfileKind::Balanced).await;
    let suite = llmario_benchmark::Suite::builtin();
    let settings = llmario_benchmark::Settings {
        concurrency: vec![1, 2],
        runs: 1,
        warmup: 1,
        temperature: 0.0,
        seed: Some(1),
        cache: llmario_benchmark::CacheMode::Cold,
        skip_quality: false,
    };
    let t = llmario_benchmark::Target {
        base_url: format!("{}/v1", env.base),
        model: "mock-a".into(),
        api_key: None,
    };
    let (levels, quality, samples) = llmario_benchmark::run(&t, &suite, &settings, None, |_| {})
        .await
        .unwrap();
    assert_eq!(levels.len(), 2);
    assert!(levels.iter().all(|l| l.errors == 0), "{levels:?}");
    assert!(samples.iter().all(|s| s.token_count_source == "usage"));
    assert!(levels[0].ttft_s.is_some() && levels[0].decode_tps.is_some());
    assert_eq!(quality.len(), 3);
    assert!(
        quality.iter().all(|q| !q.passed),
        "mock output cannot pass real checks — the harness does not fake results"
    );
}

#[test]
fn cli_doctor_json_and_config() {
    let home = tempfile::tempdir().unwrap();
    let out = std::process::Command::new(EXE)
        .args(["doctor", "--json"])
        .env("LLMARIO_HOME", home.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(v["hardware"]["total_memory_bytes"].as_u64().unwrap() > 0);
    assert_eq!(v["server"]["host"], "127.0.0.1");

    let out = std::process::Command::new(EXE)
        .args(["serve", "--host", "0.0.0.0"])
        .env("LLMARIO_HOME", home.path())
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "remote bind without --allow-remote is refused"
    );
    assert!(String::from_utf8_lossy(&out.stderr).contains("--allow-remote"));

    let out = std::process::Command::new(EXE)
        .args(["config"])
        .env("LLMARIO_HOME", home.path())
        .env("LLMARIO_API_KEY", "zzz")
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("<redacted>") && !text.contains("zzz"),
        "{text}"
    );
}

#[tokio::test]
async fn runtime_chat_client_streams_cancels_and_reports_errors() {
    use llmario_runtime::chat::{stream_chat, ChatEvent};
    let env = Env::new(&["mock-a"], |_| {}).await;

    // Streams text and reports stats + the concrete model/backend.
    let mut events = Vec::new();
    let stats = stream_chat(
        &env.http,
        &env.base,
        json!({"model": "mockfam", "messages": user("hi"), "max_tokens": 3}),
        std::future::pending::<()>(),
        |e| events.push(e),
    )
    .await
    .unwrap();
    assert_eq!(
        events[0],
        ChatEvent::Start {
            model: "mock-a".into(),
            backend: "mock".into()
        }
    );
    let text: String = events
        .iter()
        .filter_map(|e| match e {
            ChatEvent::Content { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "tok0 tok1 tok2 ");
    assert_eq!(stats.completion_tokens, 3);
    assert!(!stats.cancelled && stats.ttft_ms.is_some());

    // Cancelling mid-stream stops the engine's generation.
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let mut tx = Some(tx);
    let stats = stream_chat(
        &env.http,
        &env.base,
        json!({"model": "mock-a", "messages": user("__slow__"), "max_tokens": 200}),
        async {
            let _ = rx.await;
        },
        |e| {
            if matches!(e, ChatEvent::Content { .. }) {
                if let Some(t) = tx.take() {
                    let _ = t.send(());
                }
            }
        },
    )
    .await
    .unwrap();
    assert!(stats.cancelled);
    let mut cancelled = false;
    for _ in 0..50 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        if env.engine_stats().await["cancelled"] == 1 {
            cancelled = true;
            break;
        }
    }
    assert!(cancelled, "engine saw the cancellation");

    // Gateway errors come back as structured failures.
    let err = stream_chat(
        &env.http,
        &env.base,
        json!({"model": "nope", "messages": user("x")}),
        std::future::pending::<()>(),
        |_| {},
    )
    .await
    .unwrap_err();
    assert_eq!(err.code, "model_not_found");
}
