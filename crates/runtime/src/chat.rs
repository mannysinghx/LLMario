//! Streaming chat client for the llmario gateway, used by interactive front-ends.
//!
//! Emits reasoning and answer text as it arrives and returns timing stats. Resolving the
//! `cancel` future drops the HTTP stream, which the gateway turns into engine cancellation.

use futures::StreamExt;
use serde::Serialize;
use serde_json::Value;
use std::future::Future;
use std::time::Instant;

#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ChatEvent {
    /// The gateway accepted the request; names the concrete model and backend serving it.
    Start {
        model: String,
        backend: String,
    },
    Reasoning {
        text: String,
    },
    Content {
        text: String,
    },
}

#[derive(Serialize, Clone, Debug, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ChatStats {
    pub ttft_ms: Option<f64>,
    pub tokens_per_second: Option<f64>,
    pub completion_tokens: u64,
    pub prompt_tokens: Option<u64>,
    pub finish_reason: Option<String>,
    pub cancelled: bool,
}

#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ChatFailure {
    pub code: String,
    pub message: String,
}

impl ChatFailure {
    fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

/// POST a streaming chat completion to `{base}/v1/chat/completions`.
/// `body` must contain `model` and `messages`; streaming fields are set here.
pub async fn stream_chat(
    http: &reqwest::Client,
    base: &str,
    mut body: Value,
    cancel: impl Future<Output = ()>,
    mut on_event: impl FnMut(ChatEvent),
) -> Result<ChatStats, ChatFailure> {
    body["stream"] = Value::Bool(true);
    body["stream_options"] = serde_json::json!({"include_usage": true});
    let start = Instant::now();
    let url = format!("{}/v1/chat/completions", base.trim_end_matches('/'));

    tokio::pin!(cancel);
    let send = http.post(&url).json(&body).send();
    let resp = tokio::select! {
        r = send => r.map_err(|e| ChatFailure::new("connection_failed", e.to_string()))?,
        _ = &mut cancel => return Ok(ChatStats { cancelled: true, ..Default::default() }),
    };
    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let v: Value = resp.json().await.unwrap_or_default();
        return Err(ChatFailure::new(
            v.pointer("/error/code")
                .and_then(Value::as_str)
                .unwrap_or("http_error"),
            v.pointer("/error/message")
                .and_then(Value::as_str)
                .map(String::from)
                .unwrap_or_else(|| format!("HTTP {status}")),
        ));
    }
    let header = |name: &str| {
        resp.headers()
            .get(name)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string()
    };
    on_event(ChatEvent::Start {
        model: header("x-llmario-model"),
        backend: header("x-llmario-backend"),
    });

    let mut stats = ChatStats::default();
    let (mut first, mut last, mut chunks) = (None::<Instant>, None::<Instant>, 0u64);
    let mut buf: Vec<u8> = Vec::new();
    let mut stream = resp.bytes_stream();
    loop {
        let item = tokio::select! {
            i = stream.next() => i,
            _ = &mut cancel => { stats.cancelled = true; break; }
        };
        let Some(item) = item else { break };
        let bytes = item.map_err(|e| ChatFailure::new("stream_failed", e.to_string()))?;
        buf.extend_from_slice(&bytes);
        while let Some(pos) = buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = buf.drain(..=pos).collect();
            let line = String::from_utf8_lossy(&line);
            let Some(data) = line.trim().strip_prefix("data:").map(str::trim) else {
                continue;
            };
            if data == "[DONE]" {
                continue;
            }
            let Ok(v) = serde_json::from_str::<Value>(data) else {
                continue;
            };
            if let Some(err) = v.get("error") {
                return Err(ChatFailure::new(
                    err.get("code")
                        .and_then(Value::as_str)
                        .unwrap_or("engine_error"),
                    err.get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("engine error"),
                ));
            }
            if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
                stats.completion_tokens = u
                    .get("completion_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                stats.prompt_tokens = u.get("prompt_tokens").and_then(Value::as_u64);
            }
            if let Some(f) = v
                .pointer("/choices/0/finish_reason")
                .and_then(Value::as_str)
            {
                stats.finish_reason = Some(f.to_string());
            }
            let Some(delta) = v.pointer("/choices/0/delta") else {
                continue;
            };
            let reasoning = delta
                .get("reasoning_content")
                .or_else(|| delta.get("reasoning"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let content = delta.get("content").and_then(Value::as_str).unwrap_or("");
            if reasoning.is_empty() && content.is_empty() {
                continue;
            }
            let now = Instant::now();
            first.get_or_insert(now);
            last = Some(now);
            chunks += 1;
            if !reasoning.is_empty() {
                on_event(ChatEvent::Reasoning {
                    text: reasoning.to_string(),
                });
            }
            if !content.is_empty() {
                on_event(ChatEvent::Content {
                    text: content.to_string(),
                });
            }
        }
    }
    if stats.completion_tokens == 0 {
        stats.completion_tokens = chunks;
    }
    stats.ttft_ms = first.map(|f| (f - start).as_secs_f64() * 1000.0);
    if let (Some(f), Some(l)) = (first, last) {
        let dt = (l - f).as_secs_f64();
        if stats.completion_tokens > 1 && dt > 0.0 {
            stats.tokens_per_second = Some((stats.completion_tokens - 1) as f64 / dt);
        }
    }
    Ok(stats)
}
