//! Chat request validation and upstream body construction (field allowlist).
//!
//! Only documented, implemented fields are forwarded. Features we do not support return a
//! clear 400 instead of being silently ignored; harmless client metadata (e.g. `user`,
//! `metadata`, `store`) is dropped.
//!
//! Tool calling and constrained output (`tools`, `tool_choice`, `parallel_tool_calls`,
//! `response_format` other than text, `tool` messages, assistant `tool_calls`) are implemented by
//! LLMario's own engine only. [`parse_chat`] records them in [`ValidatedChat::native_only`] and
//! [`require_native`] refuses them for the other backends once the backend is known.

use llmario_core::{BackendKind, ResolvedProfile, RuntimeError};
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
    ("functions", "legacy function calling (use 'tools')"),
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
    /// Features in this request that only the native engine implements (empty = any backend).
    pub native_only: Vec<&'static str>,
    body: Map<String, Value>,
}

/// Refuse native-only features on the other backends.
pub fn require_native(v: &ValidatedChat, backend: BackendKind) -> Result<(), RuntimeError> {
    if v.native_only.is_empty() || backend == BackendKind::Native || backend == BackendKind::Mock {
        return Ok(());
    }
    Err(RuntimeError::Unsupported(format!(
        "{} is supported only by LLMario's own engine, and this model is served by {backend}; \
         enable it with [backends.native] enabled = true (GGUF models of supported architectures)",
        v.native_only.join(" and ")
    )))
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
    let mut native_only: Vec<&'static str> = Vec::new();
    let mut extended = Map::new();
    if let Some(rf) = obj.get("response_format").filter(|x| !x.is_null()) {
        match rf.get("type").and_then(Value::as_str) {
            Some("text") => {}
            Some("json_object") | Some("json_schema") => {
                native_only.push("JSON mode / constrained output (response_format)");
                extended.insert("response_format".into(), rf.clone());
            }
            _ => {
                return Err(invalid(
                    "response_format.type must be text, json_object or json_schema",
                ))
            }
        }
    }
    if let Some(t) = obj.get("tools").filter(|x| !x.is_null()) {
        let arr = t
            .as_array()
            .ok_or_else(|| invalid("'tools' must be an array"))?;
        for (i, d) in arr.iter().enumerate() {
            if d.get("type").and_then(Value::as_str) != Some("function")
                || d.pointer("/function/name")
                    .and_then(Value::as_str)
                    .is_none()
            {
                return Err(invalid(format!(
                    "tools[{i}] must be {{\"type\":\"function\",\"function\":{{\"name\":...}}}}"
                )));
            }
        }
        if !arr.is_empty() {
            native_only.push("tool calling");
            extended.insert("tools".into(), t.clone());
            for k in ["tool_choice", "parallel_tool_calls"] {
                if let Some(v) = obj.get(k).filter(|v| !v.is_null()) {
                    extended.insert(k.into(), v.clone());
                }
            }
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
            "system" | "developer" | "user" | "assistant" | "tool" => {}
            "function" => {
                return Err(RuntimeError::Unsupported(format!(
                    "messages[{i}]: role 'function' is the legacy function-calling API; use 'tool'"
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
        prompt_chars += content.len();
        // `developer` is the newer name for `system`; engines' chat templates expect `system`.
        let role = if role == "developer" { "system" } else { role };
        let mut out = json!({"role": role, "content": content});
        if role == "tool" {
            let id = m
                .get("tool_call_id")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid(format!("messages[{i}].tool_call_id is required")))?;
            out["tool_call_id"] = json!(id);
            if !native_only.contains(&"tool calling") {
                native_only.push("tool calling");
            }
        }
        if let Some(calls) = m.get("tool_calls").filter(|t| !t.is_null()) {
            let arr = calls
                .as_array()
                .ok_or_else(|| invalid(format!("messages[{i}].tool_calls must be an array")))?;
            for (j, c) in arr.iter().enumerate() {
                let name = c.pointer("/function/name").and_then(Value::as_str);
                let args = c.pointer("/function/arguments");
                if role != "assistant"
                    || name.is_none()
                    || !args.is_some_and(|a| a.is_string() || a.is_object())
                {
                    return Err(invalid(format!(
                        "messages[{i}].tool_calls[{j}] must be an assistant tool call with function.name and function.arguments"
                    )));
                }
                prompt_chars +=
                    name.unwrap_or("").len() + args.map(|a| a.to_string().len()).unwrap_or(0);
            }
            if !arr.is_empty() {
                out["tool_calls"] = calls.clone();
                if !native_only.contains(&"tool calling") {
                    native_only.push("tool calling");
                }
            }
        }
        out_msgs.push(out);
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
    body.extend(extended);
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
            && k != "tools"
            && k != "tool_choice"
            && k != "parallel_tool_calls"
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
        native_only,
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
    fn native_only_features_are_recorded_and_refused_elsewhere() {
        let tools = json!([{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]);
        let v = ok(
            json!({"model": "m", "messages": [{"role":"user","content":"x"}], "tools": tools, "tool_choice": "required"}),
        );
        assert_eq!(v.native_only, vec!["tool calling"]);
        let up = v.upstream_body("e");
        assert_eq!(up["tools"], tools);
        assert_eq!(up["tool_choice"], "required");
        assert!(require_native(&v, BackendKind::Native).is_ok());
        assert_eq!(
            require_native(&v, BackendKind::LlamaCpp)
                .unwrap_err()
                .code(),
            "unsupported_feature"
        );
        assert_eq!(
            require_native(&v, BackendKind::Mlx).unwrap_err().code(),
            "unsupported_feature"
        );

        // Tool round trip: assistant tool_calls and a tool result are kept for the engine.
        let v = ok(json!({"model": "m", "messages": [
            {"role": "user", "content": "w?"},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "18C"}
        ]}));
        let up = v.upstream_body("e");
        assert_eq!(
            up["messages"][1]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert_eq!(up["messages"][2]["tool_call_id"], "c1");
        assert_eq!(v.native_only, vec!["tool calling"]);
        assert_eq!(
            code(json!({"model": "m", "messages": [{"role":"tool","content":"x"}]})),
            "invalid_request",
            "tool_call_id is required"
        );

        let v = ok(
            json!({"model": "m", "messages": [{"role":"user","content":"x"}], "response_format": {"type":"json_object"}}),
        );
        assert_eq!(
            v.upstream_body("e")["response_format"]["type"],
            "json_object"
        );
        assert!(require_native(&v, BackendKind::LlamaCpp).is_err());

        // Plain requests stay valid everywhere.
        let v = ok(
            json!({"model": "m", "messages": [{"role":"user","content":"x"}], "tools": [], "response_format": {"type":"text"}}),
        );
        assert!(v.native_only.is_empty());
        assert!(require_native(&v, BackendKind::LlamaCpp).is_ok());
        assert!(v.upstream_body("e").get("tools").is_none());
    }

    #[test]
    fn unsupported_features_are_explicit() {
        assert_eq!(
            code(
                json!({"model": "m", "messages": [{"role":"user","content":"x"}], "functions": [{"name":"f"}]})
            ),
            "unsupported_feature"
        );
        assert_eq!(
            code(
                json!({"model": "m", "messages": [{"role":"user","content":"x"}], "tools": [{"type":"function"}]})
            ),
            "invalid_request"
        );
        assert_eq!(
            code(
                json!({"model": "m", "messages": [{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]})
            ),
            "unsupported_feature"
        );
        assert_eq!(
            code(json!({"model": "m", "messages": [{"role":"user","content":"x"}], "n": 2})),
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
