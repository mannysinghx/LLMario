//! OpenAI-compatible request and response types (the subset LLMario's gateway forwards).

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct ChatRequest {
    pub model: Option<String>,
    /// OpenAI-format messages; converted by `toolcall::to_chat_messages`.
    pub messages: Vec<Value>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    pub max_tokens: Option<usize>,
    pub max_completion_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub min_p: Option<f32>,
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Stop,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<usize>,
    pub n: Option<u32>,
    pub tools: Option<Value>,
    pub tool_choice: Option<Value>,
    pub parallel_tool_calls: Option<bool>,
    pub response_format: Option<Value>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u32>,
    pub user: Option<String>,
    /// Qwen-style switch passed to the template.
    pub enable_thinking: Option<bool>,
    /// Extra template variables (llama.cpp `chat_template_kwargs`).
    pub chat_template_kwargs: Option<serde_json::Map<String, Value>>,
}

#[derive(Deserialize, Debug, Clone, Default)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Deserialize, Debug, Clone, Default)]
#[serde(untagged)]
pub enum Stop {
    #[default]
    None,
    One(String),
    Many(Vec<String>),
}

impl Stop {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Stop::None => vec![],
            Stop::One(s) => vec![s],
            Stop::Many(v) => v,
        }
    }
}

#[derive(Serialize, Debug, Clone, Default)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Serialize, Debug, Clone, Default)]
pub struct PromptTokensDetails {
    pub cached_tokens: usize,
}

/// llama-server-style timings (milliseconds), useful to `llmario bench`.
#[derive(Serialize, Debug, Clone, Default)]
pub struct Timings {
    pub prompt_n: usize,
    pub prompt_ms: f64,
    pub predicted_n: usize,
    pub predicted_ms: f64,
}

pub fn completion_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("chatcmpl-{nanos:x}")
}

pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn chunk(id: &str, model: &str, created: u64, delta: Value, finish: Option<&str>) -> Value {
    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    })
}

pub fn error_body(status: u16, kind: &str, message: &str) -> Value {
    serde_json::json!({"error": {"message": message, "type": kind, "code": status}})
}
