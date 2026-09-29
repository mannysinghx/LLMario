//! Mock engine for CI: a real child process speaking IPC contract v1 with deterministic output,
//! so the supervisor, gateway, streaming, cancellation and crash paths are tested without GPUs
//! or model weights.
//!
//! Prompt directives (anywhere in the last user message):
//! - `__crash__` → the process exits with code 3 after the first streamed token.
//! - `__slow__`  → 100 ms per token (for cancellation tests).
//!
//! Env: `MOCK_STARTUP_MS` (default 50), `MOCK_TOKEN_DELAY_MS` (default 2).
//! `GET /mock/stats` reports completed/cancelled streams and the last upstream request's
//! `model` and top-level keys, so tests can assert what the gateway forwarded.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::stream;
use llmario_core::{BackendKind, Config, ModelFormat, RuntimeError};
use llmario_hardware::HardwareReport;
use llmario_supervisor::adapter::{BackendStatus, EngineAdapter, LaunchContext, LaunchSpec};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub struct MockAdapter {
    /// Executable that implements `<args_prefix> --port P --model ID` (the llmario binary).
    pub program: PathBuf,
    pub args_prefix: Vec<String>,
}

impl EngineAdapter for MockAdapter {
    fn kind(&self) -> BackendKind {
        BackendKind::Mock
    }
    fn formats(&self) -> &'static [ModelFormat] {
        &[ModelFormat::Mock]
    }
    fn tested_version(&self) -> &'static str {
        "in-tree"
    }
    fn probe(&self, _hw: &HardwareReport, _cfg: &Config) -> BackendStatus {
        BackendStatus {
            kind: BackendKind::Mock,
            available: self.program.is_file(),
            path: Some(self.program.clone()),
            version: Some(llmario_core::VERSION.into()),
            tested_version: "in-tree".into(),
            detail: "mock engine for tests".into(),
            architectures: None,
        }
    }
    fn launch(&self, ctx: &LaunchContext<'_>) -> Result<LaunchSpec, RuntimeError> {
        let mut args = self.args_prefix.clone();
        args.extend([
            "--port".into(),
            ctx.port.to_string(),
            "--model".into(),
            ctx.model.id.clone(),
        ]);
        Ok(LaunchSpec {
            program: self.program.clone(),
            args,
            env: vec![],
            health_path: "/health".into(),
            upstream_model: format!("mock-upstream-{}", ctx.model.id),
            notes: vec![],
        })
    }
}

#[derive(Default)]
struct Stats {
    completed: AtomicU64,
    cancelled: AtomicU64,
    last_model: Mutex<String>,
    last_keys: Mutex<Vec<String>>,
}

struct CancelGuard {
    stats: Arc<Stats>,
    finished: bool,
}
impl Drop for CancelGuard {
    fn drop(&mut self) {
        if self.finished {
            self.stats.completed.fetch_add(1, Ordering::SeqCst);
        } else {
            self.stats.cancelled.fetch_add(1, Ordering::SeqCst);
        }
    }
}

fn env_ms(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

pub async fn run_mock_engine(port: u16, model: String) -> anyhow::Result<()> {
    tokio::time::sleep(Duration::from_millis(env_ms("MOCK_STARTUP_MS", 50))).await;
    let stats = Arc::new(Stats::default());
    let app = Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/mock/stats", get(stats_handler))
        .route("/v1/chat/completions", post(chat))
        .with_state((stats, Arc::new(model)));
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn stats_handler(State((s, _)): State<(Arc<Stats>, Arc<String>)>) -> Json<Value> {
    Json(json!({
        "completed": s.completed.load(Ordering::SeqCst),
        "cancelled": s.cancelled.load(Ordering::SeqCst),
        "last_model": *s.last_model.lock().unwrap(),
        "last_keys": *s.last_keys.lock().unwrap(),
    }))
}

async fn chat(
    State((stats, _model)): State<(Arc<Stats>, Arc<String>)>,
    Json(body): Json<Value>,
) -> Response {
    let req_model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    *stats.last_model.lock().unwrap() = req_model.clone();
    *stats.last_keys.lock().unwrap() = body
        .as_object()
        .map(|o| o.keys().cloned().collect())
        .unwrap_or_default();

    let last = body
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|m| m.last())
        .and_then(|m| m.get("content"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let prompt_chars: usize = body
        .get("messages")
        .and_then(Value::as_array)
        .map(|m| {
            m.iter()
                .filter_map(|x| x.get("content").and_then(Value::as_str))
                .map(str::len)
                .sum()
        })
        .unwrap_or(0);
    let prompt_tokens = (prompt_chars / 4).max(1) as u64;
    let max = body
        .get("max_tokens")
        .or_else(|| body.get("max_completion_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(16);
    let crash = last.contains("__crash__");
    let delay = if last.contains("__slow__") {
        100
    } else {
        env_ms("MOCK_TOKEN_DELAY_MS", 2)
    };
    let stream_mode = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let usage_wanted = body
        .pointer("/stream_options/include_usage")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    if !stream_mode {
        if crash {
            std::process::exit(3);
        }
        tokio::time::sleep(Duration::from_millis(delay * max)).await;
        let text: String = (0..max).map(|i| format!("tok{i} ")).collect();
        stats.completed.fetch_add(1, Ordering::SeqCst);
        return Json(json!({
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 0,
            "model": req_model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": max, "total_tokens": prompt_tokens + max}
        }))
        .into_response();
    }

    let guard = CancelGuard {
        stats,
        finished: false,
    };
    let chunk = move |delta: Value, finish: Option<&str>| {
        json!({"id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": 0, "model": req_model,
               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
    };
    // State: (next index, guard). Index max+0 = finish chunk, max+1 = usage, max+2 = [DONE].
    let s = stream::unfold((0u64, Some(guard)), move |(i, mut guard)| {
        let chunk = chunk.clone();
        async move {
            guard.as_ref()?;
            let data = if i == 0 {
                chunk(json!({"role": "assistant", "content": ""}), None).to_string()
            } else if i <= max {
                tokio::time::sleep(Duration::from_millis(delay)).await;
                if crash && i == 2 {
                    std::process::exit(3);
                }
                chunk(json!({"content": format!("tok{} ", i - 1)}), None).to_string()
            } else if i == max + 1 {
                chunk(json!({}), Some("length")).to_string()
            } else if i == max + 2 && usage_wanted {
                json!({"id": "chatcmpl-mock", "object": "chat.completion.chunk", "choices": [],
                       "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": max, "total_tokens": prompt_tokens + max}})
                .to_string()
            } else {
                if let Some(g) = guard.as_mut() {
                    g.finished = true;
                }
                drop(guard.take());
                return Some((
                    Ok::<_, std::io::Error>(bytes::Bytes::from("data: [DONE]\n\n")),
                    (i + 1, None),
                ));
            };
            Some((
                Ok(bytes::Bytes::from(format!("data: {data}\n\n"))),
                (i + 1, guard),
            ))
        }
    });
    (
        StatusCode::OK,
        [
            ("content-type", "text/event-stream"),
            ("cache-control", "no-cache"),
        ],
        axum::body::Body::from_stream(s),
    )
        .into_response()
}
