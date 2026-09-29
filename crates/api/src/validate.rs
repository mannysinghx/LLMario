//! Chat request validation and upstream body construction (field allowlist).
//!
//! Only documented, implemented fields are forwarded. Features we do not support return a
//! clear 400 instead of being silently ignored; harmless client metadata (e.g. `user`,
//! `metadata`, `store`) is dropped.

use llmario_core::{ResolvedProfile, RuntimeError};
use serde_json::{json, Map, Value};

/// Sampling / control fields forwarded verbatim when present.
const PASS_THROUGH: &[&str] = &[
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "stop",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
];

/// Fields that request a feature this release does not implement.
const UNSUPPORTED: &[(&str, &str)] = &[
    ("tools", "tool/function calling"),
    ("tool_choice", "tool/function calling"),
    ("functions", "tool/function calling"),
    ("function_call", "tool/function calling"),
    ("logprobs", "logprobs"),
    ("top_logprobs", "logprobs"),
    ("logit_bias", "logit_bias"),
    ("audio", "audio output"),
    ("modalities", "non-text modalities"),
    ("prediction", "predicted outputs"),
];

#[derive(Debug)]
pub struct ValidatedChat {
    pub model: String,
    pub stream: bool,
    pub client_wants_usage: bool,
    pub max_tokens: u64,
    pub approx_prompt_tokens: u64,
    body: Map<String, Value>,
}

impl ValidatedChat {
    /// Body for the engine: allowlisted fields, engine-local `model`, usage always requested
    /// on streams (the gateway needs it for metrics and strips it if the client did not ask).
    pub fn upstream_body(&self, upstream_model: &str) -> Value {
        let mut b = self.body.clone();
        b.insert("model".into(), json!(upstream_model));
        b.insert("max_tokens".into(), json!(self.max_tokens));
        b.insert("stream".into(), json!(self.stream));
        if self.stream {
            b.insert("stream_options".into(), json!({"include_usage": true}));
        }
        Value::Object(b)
    }
}

fn invalid(msg: impl Into<String>) -> RuntimeError {
    RuntimeError::InvalidRequest(msg.into())
}

/// Parse and check the client body. Profile-dependent limits are applied by [`check_limits`].
pub fn parse_chat(v: &Value) -> Result<ValidatedChat, RuntimeError> {
    let obj = v
        .as_object()
        .ok_or_else(|| invalid("request body must be a JSON object"))?;
    for (k, what) in UNSUPPORTED {
        if obj
            .get(*k)
            .is_some_and(|x| !x.is_null() && x != &json!([]) && x != &json!(false))
        {
            return Err(RuntimeError::Unsupported(format!(
                "'{k}' is not supported: {what} is not implemented in this release"
            )));
        }
    }
    if let Some(rf) = obj.get("response_format").filter(|x| !x.is_null()) {
        if rf.get("type").and_then(Value::as_str) != Some("text") {
            return Err(RuntimeError::Unsupported(
                "response_format other than {\"type\":\"text\"} (JSON mode / constrained output) is not implemented in this release".into(),
            ));
        }
    }
    if let Some(n) = obj.get("n").and_then(Value::as_u64) {
        if n != 1 {
            return Err(RuntimeError::Unsupported(
                "n > 1 is not supported; send separate requests".into(),
            ));
        }
    }

    let model = obj
        .get("model")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| invalid("'model' is required"))?
        .to_string();
    let stream = obj.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let client_wants_usage = v
        .pointer("/stream_options/include_usage")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let msgs = obj
        .get("messages")
        .and_then(Value::as_array)
        .filter(|m| !m.is_empty())
        .ok_or_else(|| invalid("'messages' must be a non-empty array"))?;
    let mut out_msgs = Vec::with_capacity(msgs.len());
    let mut prompt_chars = 0usize;
    for (i, m) in msgs.iter().enumerate() {
        let role = m
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid(format!("messages[{i}].role is required")))?;
        match role {
            "system" | "developer" | "user" | "assistant" => {}
            "tool" | "function" => {
                return Err(RuntimeError::Unsupported(format!(
                    "messages[{i}]: role '{role}' requires tool calling, which is not implemented"
                )))
            }
            other => return Err(invalid(format!("messages[{i}]: unknown role '{other}'"))),
        }
        let content = match m.get("content") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(parts)) => {
                let mut text = String::new();
                for (j, p) in parts.iter().enumerate() {
                    match p.get("type").and_then(Value::as_str) {
                        Some("text") => text.push_str(p.get("text").and_then(Value::as_str).unwrap_or("")),
                        Some(t) => {
                            return Err(RuntimeError::Unsupported(format!(
                                "messages[{i}].content[{j}]: '{t}' input is not supported (text-only models in this release)"
                            )))
                        }
                        None => return Err(invalid(format!("messages[{i}].content[{j}].type is required"))),
                    }
                }
                text
            }
            Some(Value::Null) | None if role == "assistant" => String::new(),
            _ => {
                return Err(invalid(format!(
                    "messages[{i}].content must be a string or an array of text parts"
                )))
            }
        };
        if m.get("tool_calls").is_some_and(|t| !t.is_null()) {
            return Err(RuntimeError::Unsupported(format!(
                "messages[{i}].tool_calls: tool calling is not implemented"
            )));
        }
        prompt_chars += content.len();
        // `developer` is the newer name for `system`; engines' chat templates expect `system`.
        let role = if role == "developer" { "system" } else { role };
        out_msgs.push(json!({"role": role, "content": content}));
    }

    let max_tokens = obj
        .get("max_completion_tokens")
        .or_else(|| obj.get("max_tokens"))
        .filter(|v| !v.is_null())
        .map(|v| {
            v.as_u64()
                .filter(|n| *n > 0)
                .ok_or_else(|| invalid("max_tokens must be a positive integer"))
        })
        .transpose()?
        .unwrap_or(0);

    let mut body = Map::new();
    body.insert("messages".into(), Value::Array(out_msgs));
    for k in PASS_THROUGH {
        if let Some(v) = obj.get(*k).filter(|v| !v.is_null()) {
            body.insert((*k).into(), v.clone());
        }
    }
    if let Some(t) = body.get("temperature").and_then(Value::as_f64) {
        if !(0.0..=2.0).contains(&t) {
            return Err(invalid("temperature must be between 0 and 2"));
        }
    }
    for k in obj.keys() {
        if k != "model"
            && k != "messages"
            && k != "stream"
            && k != "stream_options"
            && k != "max_tokens"
            && k != "max_completion_tokens"
            && k != "n"
            && k != "response_format"
            && !PASS_THROUGH.contains(&k.as_str())
        {
            tracing::debug!(field = %k, "dropping unrecognised request field");
        }
    }

    Ok(ValidatedChat {
        model,
        stream,
        client_wants_usage,
        max_tokens,
        // ~4 bytes/token for English; an estimate used only to reject clear overflows early.
        approx_prompt_tokens: (prompt_chars / 4) as u64,
        body,
    })
}

/// Apply per-engine limits: default/cap `max_tokens`, reject prompts that cannot fit.
pub fn check_limits(v: &mut ValidatedChat, profile: &ResolvedProfile) -> Result<(), RuntimeError> {
    let ctx = profile.ctx_per_slot as u64;
    if v.approx_prompt_tokens >= ctx {
        return Err(RuntimeError::ContextOverflow(format!(
            "prompt is roughly {} tokens but each request may use {ctx} tokens with the '{}' profile; shorten the prompt or restart with a larger --context",
            v.approx_prompt_tokens, profile.kind
        )));
    }
    if v.max_tokens == 0 {
        v.max_tokens = (profile.default_max_tokens as u64).min(ctx - v.approx_prompt_tokens);
    }
    if v.max_tokens > ctx {
        return Err(RuntimeError::ContextOverflow(format!(
            "max_tokens {} exceeds the per-request context of {ctx} tokens",
            v.max_tokens
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmario_core::ProfileKind;

    fn ok(v: Value) -> ValidatedChat {
        parse_chat(&v).unwrap()
    }
    fn code(v: Value) -> &'static str {
        parse_chat(&v).unwrap_err().code()
    }

    #[test]
    fn minimal_request_and_allowlist() {
        let v = ok(json!({
            "model": "m", "messages": [{"role": "developer", "content": "be brief"}, {"role": "user", "content": "hi"}],
            "temperature": 0.2, "user": "abc", "draft_model": "evil/repo", "adapters": "x", "metadata": {}
        }));
        let up = v.upstream_body("default_model");
        assert_eq!(
            up["model"], "default_model",
            "client model never reaches the engine"
        );
        assert!(
            up.get("draft_model").is_none()
                && up.get("adapters").is_none()
                && up.get("user").is_none()
        );
        assert_eq!(up["temperature"], 0.2);
        assert_eq!(up["messages"][0]["role"], "system");
        assert!(up.get("stream_options").is_none());
    }

    #[test]
    fn text_parts_are_flattened() {
        let v = ok(
            json!({"model": "m", "messages": [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]}),
        );
        assert_eq!(v.upstream_body("x")["messages"][0]["content"], "ab");
    }

    #[test]
    fn unsupported_features_are_explicit() {
        assert_eq!(
            code(
                json!({"model": "m", "messages": [{"role":"user","content":"x"}], "tools": [{"type":"function"}]})
            ),
            "unsupported_feature"
        );
        assert_eq!(
            code(
                json!({"model": "m", "messages": [{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]})
            ),
            "unsupported_feature"
        );
        assert_eq!(
            code(
                json!({"model": "m", "messages": [{"role":"user","content":"x"}], "response_format": {"type":"json_object"}})
            ),
            "unsupported_feature"
        );
        assert_eq!(
            code(json!({"model": "m", "messages": [{"role":"user","content":"x"}], "n": 2})),
            "unsupported_feature"
        );
        assert_eq!(
            code(json!({"model": "m", "messages": [{"role":"tool","content":"x"}]})),
            "unsupported_feature"
        );
        // Empty tools array / text response_format are fine.
        ok(
            json!({"model": "m", "messages": [{"role":"user","content":"x"}], "tools": [], "response_format": {"type":"text"}}),
        );
    }

    #[test]
    fn malformed_requests() {
        assert_eq!(code(json!([])), "invalid_request");
        assert_eq!(
            code(json!({"messages": [{"role":"user","content":"x"}]})),
            "invalid_request"
        );
        assert_eq!(
            code(json!({"model": "m", "messages": []})),
            "invalid_request"
        );
        assert_eq!(
            code(json!({"model": "m", "messages": [{"role":"wizard","content":"x"}]})),
            "invalid_request"
        );
        assert_eq!(
            code(
                json!({"model": "m", "messages": [{"role":"user","content":"x"}], "max_tokens": 0})
            ),
            "invalid_request"
        );
        assert_eq!(
            code(
                json!({"model": "m", "messages": [{"role":"user","content":"x"}], "temperature": 7})
            ),
            "invalid_request"
        );
    }

    #[test]
    fn streaming_always_requests_usage_upstream() {
        let v =
            ok(json!({"model": "m", "stream": true, "messages": [{"role":"user","content":"x"}]}));
        assert!(!v.client_wants_usage);
        assert_eq!(
            v.upstream_body("x")["stream_options"]["include_usage"],
            true
        );
    }

    #[test]
    fn limits() {
        let p = ResolvedProfile::resolve(ProfileKind::Latency, Some(1024));
        let mut v = ok(json!({"model": "m", "messages": [{"role":"user","content":"x"}]}));
        check_limits(&mut v, &p).unwrap();
        assert_eq!(v.max_tokens, 1024, "default capped to remaining context");
        let mut long =
            ok(json!({"model": "m", "messages": [{"role":"user","content":"x".repeat(8000)}]}));
        assert_eq!(
            check_limits(&mut long, &p).unwrap_err().code(),
            "context_length_exceeded"
        );
        let mut big = ok(
            json!({"model": "m", "max_tokens": 5000, "messages": [{"role":"user","content":"x"}]}),
        );
        assert_eq!(
            check_limits(&mut big, &p).unwrap_err().code(),
            "context_length_exceeded"
        );
    }
}
