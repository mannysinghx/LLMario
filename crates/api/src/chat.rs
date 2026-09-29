//! `POST /v1/chat/completions`: admission → validation → proxy to the engine.
//!
//! Streaming is a line-level SSE relay: each upstream `data:` event is re-emitted with the
//! local model id. The admission [`Lease`] lives inside the response stream, so when a client
//! disconnects the stream is dropped, the upstream connection closes (the engine stops
//! generating), and the slot is released immediately.

use crate::metrics::RequestRecord;
use crate::validate::{self, ValidatedChat};
use crate::{error_body, ApiError, AppState, Metrics};
use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use llmario_core::RuntimeError;
use llmario_supervisor::{Engine, Lease};
use serde_json::Value;
use std::collections::VecDeque;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub async fn chat_completions(
    State(st): State<AppState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    let started = Instant::now();
    let Json(v) = match body {
        Ok(j) => j,
        Err(e) => return ApiError(RuntimeError::InvalidRequest(e.body_text())).into_response(),
    };
    let requested = v
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("?")
        .to_string();
    match handle(&st, v, started).await {
        Ok(r) => r,
        Err(e) => {
            tracing::info!(model = %requested, outcome = e.code(), duration_ms = started.elapsed().as_millis() as u64, "chat request failed");
            st.metrics.record(RequestRecord {
                model: &requested,
                outcome: e.code(),
                ttft_s: None,
                duration_s: started.elapsed().as_secs_f64(),
                prompt_tokens: 0,
                completion_tokens: 0,
            });
            ApiError(e).into_response()
        }
    }
}

async fn handle(st: &AppState, v: Value, started: Instant) -> Result<Response, RuntimeError> {
    let mut req = validate::parse_chat(&v)?;
    let lease = st.sup.acquire(&req.model).await?;
    let engine = lease.engine.clone();
    validate::check_limits(&mut req, &engine.profile)?;

    let mut body = req.upstream_body(&engine.upstream_model);
    if engine.backend == llmario_core::BackendKind::Mlx {
        mlx_keep_batchable(&mut body);
    }
    let resp = st
        .sup
        .http()
        .post(format!("{}/v1/chat/completions", engine.base_url))
        .json(&body)
        .timeout(st.request_timeout)
        .send()
        .await;
    let resp = match resp {
        Ok(r) => r,
        Err(e) => return Err(send_error(&engine, e).await),
    };
    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(upstream_status_error(&engine, status.as_u16(), &text).await);
    }

    let mut out = if req.stream {
        let relay = Relay::new(resp, lease, st.metrics.clone(), &req, started);
        let s = futures::stream::unfold(relay, |mut r| async move {
            r.next().await.map(|b| (Ok::<_, Infallible>(b), r))
        });
        (
            StatusCode::OK,
            [
                ("content-type", "text/event-stream"),
                ("cache-control", "no-cache"),
                ("x-accel-buffering", "no"),
            ],
            axum::body::Body::from_stream(s),
        )
            .into_response()
    } else {
        let mut json: Value = resp.json().await.map_err(|e| {
            if e.is_timeout() {
                RuntimeError::Timeout(format!(
                    "request exceeded {}s",
                    st.request_timeout.as_secs()
                ))
            } else {
                RuntimeError::EngineCrashed(format!("reading engine response: {e}"))
            }
        })?;
        json["model"] = Value::String(engine.model.id.clone());
        let (p, c) = usage_of(&json).unwrap_or((req.approx_prompt_tokens, 0));
        let dur = started.elapsed().as_secs_f64();
        tracing::info!(model = %engine.model.id, backend = %engine.backend, outcome = "ok", duration_ms = (dur * 1000.0) as u64, prompt_tokens = p, completion_tokens = c, "chat request");
        st.metrics.record(RequestRecord {
            model: &engine.model.id,
            outcome: "ok",
            ttft_s: None,
            duration_s: dur,
            prompt_tokens: p,
            completion_tokens: c,
        });
        drop(lease);
        Json(json).into_response()
    };
    let h = out.headers_mut();
    if let Ok(v) = HeaderValue::from_str(&engine.model.id) {
        h.insert("x-llmario-model", v);
    }
    if let Ok(v) = HeaderValue::from_str(&engine.backend.to_string()) {
        h.insert("x-llmario-backend", v);
    }
    Ok(out)
}

/// mlx_lm.server (0.31) only batches requests without a `seed`. With greedy decoding
/// (temperature 0, also its default when unset) the seed cannot change the output, so we drop
/// it and keep continuous batching. A seed with temperature > 0 is kept (reproducibility wins).
fn mlx_keep_batchable(body: &mut Value) {
    let greedy = body
        .get("temperature")
        .and_then(Value::as_f64)
        .is_none_or(|t| t == 0.0);
    if greedy {
        if let Some(o) = body.as_object_mut() {
            o.remove("seed");
        }
    }
}

fn usage_of(v: &Value) -> Option<(u64, u64)> {
    let u = v.get("usage")?;
    Some((
        u.get("prompt_tokens")?.as_u64()?,
        u.get("completion_tokens")?.as_u64()?,
    ))
}

async fn engine_died(engine: &Engine) -> bool {
    // Exit notification can lag the socket error by a few ms.
    for _ in 0..10 {
        if !engine.is_alive() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    false
}

async fn send_error(engine: &Engine, e: reqwest::Error) -> RuntimeError {
    if e.is_timeout() {
        return RuntimeError::Timeout("engine did not respond in time".into());
    }
    if engine_died(engine).await {
        let why = engine.exit_info().map(|i| i.describe()).unwrap_or_default();
        return RuntimeError::EngineCrashed(format!(
            "{} engine for '{}' exited ({why}); it will be relaunched on the next request. Log: {}",
            engine.backend,
            engine.model.id,
            engine.log_path.display()
        ));
    }
    RuntimeError::EngineCrashed(format!("connection to engine failed: {e}"))
}

async fn upstream_status_error(engine: &Engine, status: u16, text: &str) -> RuntimeError {
    let msg = serde_json::from_str::<Value>(text)
        .ok()
        .and_then(|v| {
            v.pointer("/error/message")
                .or_else(|| v.get("error"))
                .and_then(Value::as_str)
                .map(String::from)
        })
        .unwrap_or_else(|| text.chars().take(500).collect());
    let lower = msg.to_ascii_lowercase();
    if status == 400 && (lower.contains("context") || lower.contains("exceed")) {
        return RuntimeError::ContextOverflow(msg);
    }
    if (400..500).contains(&status) {
        return RuntimeError::InvalidRequest(format!("engine rejected the request: {msg}"));
    }
    if engine_died(engine).await {
        return RuntimeError::EngineCrashed(format!(
            "engine exited while handling the request: {msg}"
        ));
    }
    RuntimeError::Other(anyhow::anyhow!("engine returned HTTP {status}: {msg}"))
}

type ByteStream = Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>>;

struct Relay {
    upstream: ByteStream,
    buf: Vec<u8>,
    out: VecDeque<Bytes>,
    lease: Option<Lease>,
    metrics: Arc<Metrics>,
    model_id: Arc<str>,
    wants_usage: bool,
    approx_prompt: u64,
    started: Instant,
    first_token: Option<Instant>,
    content_chunks: u64,
    usage: Option<(u64, u64)>,
    saw_done: bool,
    finished: bool,
    recorded: bool,
}

impl Relay {
    fn new(
        resp: reqwest::Response,
        lease: Lease,
        metrics: Arc<Metrics>,
        req: &ValidatedChat,
        started: Instant,
    ) -> Self {
        Self {
            upstream: Box::pin(resp.bytes_stream()),
            buf: Vec::with_capacity(4096),
            out: VecDeque::new(),
            model_id: lease.engine.model.id.as_str().into(),
            lease: Some(lease),
            metrics,
            wants_usage: req.client_wants_usage,
            approx_prompt: req.approx_prompt_tokens,
            started,
            first_token: None,
            content_chunks: 0,
            usage: None,
            saw_done: false,
            finished: false,
            recorded: false,
        }
    }

    async fn next(&mut self) -> Option<Bytes> {
        loop {
            if let Some(b) = self.out.pop_front() {
                return Some(b);
            }
            if self.finished {
                return None;
            }
            match self.upstream.next().await {
                Some(Ok(chunk)) => {
                    self.buf.extend_from_slice(&chunk);
                    self.drain_lines();
                }
                Some(Err(e)) => {
                    let err = if e.is_timeout() {
                        RuntimeError::Timeout("stream exceeded the request timeout".into())
                    } else {
                        let engine = &self.lease.as_ref().unwrap().engine;
                        send_error(engine, e).await
                    };
                    self.fail(err);
                }
                None => {
                    if !self.buf.is_empty() {
                        self.buf.push(b'\n');
                        self.drain_lines();
                    }
                    if self.saw_done {
                        self.finish("ok");
                    } else {
                        let engine = &self.lease.as_ref().unwrap().engine;
                        let err = if engine_died(engine).await {
                            RuntimeError::EngineCrashed(format!(
                                "engine exited mid-stream ({}); it will be relaunched on the next request",
                                engine.exit_info().map(|i| i.describe()).unwrap_or_default()
                            ))
                        } else {
                            RuntimeError::EngineCrashed(
                                "engine closed the stream without [DONE]".into(),
                            )
                        };
                        self.fail(err);
                    }
                }
            }
        }
    }

    fn fail(&mut self, e: RuntimeError) {
        let payload = error_body(&e);
        self.out
            .push_back(Bytes::from(format!("data: {payload}\n\n")));
        self.out.push_back(Bytes::from_static(b"data: [DONE]\n\n"));
        self.finish(e.code());
    }

    fn drain_lines(&mut self) {
        while let Some(pos) = self.buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=pos).collect();
            let line = String::from_utf8_lossy(&line);
            let line = line.trim_end_matches(['\n', '\r']);
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim_start();
            if data == "[DONE]" {
                self.saw_done = true;
                self.out.push_back(Bytes::from_static(b"data: [DONE]\n\n"));
                continue;
            }
            let Ok(mut v) = serde_json::from_str::<Value>(data) else {
                self.out.push_back(Bytes::from(format!("data: {data}\n\n")));
                continue;
            };
            if let Some(u) = usage_of(&v) {
                self.usage = Some(u);
                if !self.wants_usage {
                    if let Some(o) = v.as_object_mut() {
                        o.remove("usage");
                    }
                }
            }
            let choices_empty = v
                .get("choices")
                .and_then(Value::as_array)
                .is_none_or(|c| c.is_empty());
            if choices_empty && !self.wants_usage && v.get("error").is_none() {
                continue; // usage-only chunk the client did not ask for
            }
            if let Some(delta) = v.pointer("/choices/0/delta") {
                let has_text = ["content", "reasoning_content", "reasoning"]
                    .iter()
                    .any(|k| {
                        delta
                            .get(*k)
                            .and_then(Value::as_str)
                            .is_some_and(|s| !s.is_empty())
                    });
                if has_text {
                    self.content_chunks += 1;
                    if self.first_token.is_none() {
                        self.first_token = Some(Instant::now());
                    }
                }
            }
            v["model"] = Value::String(self.model_id.to_string());
            self.out.push_back(Bytes::from(format!("data: {v}\n\n")));
        }
    }

    fn finish(&mut self, outcome: &str) {
        self.finished = true;
        self.record(outcome);
        self.lease.take(); // release the slot as soon as generation ends
    }

    fn record(&mut self, outcome: &str) {
        if self.recorded {
            return;
        }
        self.recorded = true;
        let (p, c) = self
            .usage
            .unwrap_or((self.approx_prompt, self.content_chunks));
        let ttft = self.first_token.map(|t| (t - self.started).as_secs_f64());
        let dur = self.started.elapsed().as_secs_f64();
        let backend = self
            .lease
            .as_ref()
            .map(|l| l.engine.backend.to_string())
            .unwrap_or_default();
        tracing::info!(
            model = %self.model_id, backend = %backend, outcome, stream = true,
            ttft_ms = ttft.map(|t| (t * 1000.0) as u64), duration_ms = (dur * 1000.0) as u64,
            prompt_tokens = p, completion_tokens = c, "chat request"
        );
        self.metrics.record(RequestRecord {
            model: &self.model_id,
            outcome,
            ttft_s: ttft,
            duration_s: dur,
            prompt_tokens: p,
            completion_tokens: c,
        });
    }
}

impl Drop for Relay {
    fn drop(&mut self) {
        if !self.recorded {
            // The client went away before the stream finished.
            self.record("cancelled");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn mlx_seed_dropped_only_when_greedy() {
        let mut b = json!({"seed": 1, "temperature": 0.0});
        mlx_keep_batchable(&mut b);
        assert!(b.get("seed").is_none());
        let mut b = json!({"seed": 1});
        mlx_keep_batchable(&mut b);
        assert!(b.get("seed").is_none(), "mlx default temperature is 0");
        let mut b = json!({"seed": 1, "temperature": 0.7});
        mlx_keep_batchable(&mut b);
        assert_eq!(b["seed"], 1);
    }
}
