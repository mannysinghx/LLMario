//! The server-side agent loop for built-in tools (Architecture §10.5–§10.7).
//!
//! A request may list built-in tools next to its own functions: `{"type": "web_search"}` and
//! `{"type": "web_fetch"}`. The model sees them as ordinary functions. When a generation step
//! ends with calls to built-in tools only, the engine runs them (SSRF-guarded fetch, self-hosted
//! search), appends the call and the provenance-wrapped results to the conversation, and generates
//! again, up to `max_steps` tool rounds; then one last step runs without tools so the model must
//! answer. A step that calls any client-defined function ends the loop and returns those calls to
//! the client exactly as without built-in tools.
//!
//! Policy: web access is off unless the engine was started with it (`--web`, set by
//! `[backends.native] web_access = true`); `web_search` additionally needs a self-hosted search
//! endpoint (`--searxng-url`). Both tools are open-world reads (no state change), so they run
//! without approval; their output is untrusted and is wrapped with a random nonce. Page text is
//! capped before it reaches the prompt.
//!
//! Streaming: reasoning and content deltas of every step are forwarded as they are generated;
//! built-in calls are not sent as `tool_calls` (the client cannot run them) but as a non-standard
//! `llmario_tool_event` delta (`name`, `arguments`, `status`, `url`, `ms`) that OpenAI clients ignore.

use crate::engine::{EngineRuntime, Event, Job};
use crate::toolcall::{self, Assembler, Choice};
use llmario_engine_chat::{ChatTemplate, Content, Message, RenderRequest, ToolCall};
use llmario_engine_decode::SamplingParams;
use llmario_engine_tools::builtin::{
    web_fetch_definition, web_search_definition, WebFetchTool, WebSearchTool, WEB_FETCH, WEB_SEARCH,
};
use llmario_engine_tools::search::SearchConfig;
use llmario_engine_tools::{wrap_untrusted, FetchConfig, SearxngProvider, WebFetcher};
use serde_json::{json, Map, Value};
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

/// Tool rounds before the final no-tools step.
pub const DEFAULT_MAX_STEPS: usize = 4;
/// Characters of one tool result that reach the prompt.
pub const MAX_RESULT_CHARS: usize = 8_000;

/// The built-in web tools this engine process may run.
pub struct WebTools {
    fetcher: Option<Arc<WebFetcher>>,
    search: Option<Arc<WebSearchTool>>,
}

impl WebTools {
    pub fn disabled() -> WebTools {
        WebTools {
            fetcher: None,
            search: None,
        }
    }

    /// `web`: allow `web_fetch` (and `web_search` when `searxng_url` is set).
    pub fn new(web: bool, searxng_url: Option<&str>) -> anyhow::Result<WebTools> {
        if !web {
            return Ok(WebTools::disabled());
        }
        let fetcher = Arc::new(WebFetcher::new(FetchConfig::default()));
        let search = match searxng_url {
            Some(url) => Some(Arc::new(WebSearchTool::new(Box::new(
                SearxngProvider::new(url, SearchConfig::default())
                    .map_err(|e| anyhow::anyhow!("search provider {url}: {e}"))?,
            )))),
            None => None,
        };
        Ok(WebTools {
            fetcher: Some(fetcher),
            search,
        })
    }

    pub fn available(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        if self.search.is_some() {
            v.push(WEB_SEARCH);
        }
        if self.fetcher.is_some() {
            v.push(WEB_FETCH);
        }
        v
    }

    /// Check a request's built-in tool list against what this process allows.
    pub fn check(&self, requested: &[&'static str]) -> Result<(), String> {
        for name in requested {
            let ok = match *name {
                WEB_FETCH => self.fetcher.is_some(),
                WEB_SEARCH => self.search.is_some(),
                _ => false,
            };
            if !ok {
                return Err(if self.fetcher.is_none() {
                    format!(
                        "{name} needs web access, which is off for this engine; enable it with \
                         [backends.native] web_access = true"
                    )
                } else {
                    format!(
                        "{name} needs a self-hosted search endpoint; set [backends.native] \
                         searxng_url (for example http://127.0.0.1:8080)"
                    )
                });
            }
        }
        Ok(())
    }
}

/// Split a request's `tools` into client functions and built-in tool names.
pub fn split_tools(tools: &Value) -> Result<(Vec<Value>, Vec<&'static str>), String> {
    let mut functions = Vec::new();
    let mut builtins = Vec::new();
    for (i, t) in tools.as_array().into_iter().flatten().enumerate() {
        match t.get("type").and_then(Value::as_str) {
            Some("function") => functions.push(t.clone()),
            Some("web_search") => builtins.push(WEB_SEARCH),
            Some("web_fetch") => builtins.push(WEB_FETCH),
            other => {
                return Err(format!(
                    "tools[{i}].type {other:?} is not supported (function, web_search, web_fetch)"
                ))
            }
        }
    }
    builtins.dedup();
    Ok((functions, builtins))
}

/// Everything the loop needs from the request.
pub struct AgentRequest {
    pub messages: Vec<Message>,
    pub functions: Vec<Value>,
    pub builtins: Vec<&'static str>,
    pub enable_thinking: Option<bool>,
    pub extra: Map<String, Value>,
    pub params: SamplingParams,
    pub max_tokens: usize,
    pub stop: Vec<String>,
    pub max_steps: usize,
}

/// What the loop reports.
#[derive(Debug)]
pub enum AgentOut {
    /// A `choices[0].delta` object.
    Delta(Value),
    Done {
        reason: String,
        prompt_tokens: usize,
        cached_tokens: usize,
        completion_tokens: usize,
        prompt_ms: f64,
        decode_ms: f64,
        steps: usize,
    },
    Error(String),
}

fn nonce() -> String {
    let mut h = RandomState::new().build_hasher();
    h.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    );
    format!("{:016x}", h.finish())
}

fn truncate_chars(s: &str, max: usize) -> String {
    match s.char_indices().nth(max) {
        Some((i, _)) => format!("{}\n[… truncated at {max} characters]", &s[..i]),
        None => s.to_string(),
    }
}

/// Run the loop; every outcome is sent on `out`.
pub async fn run(
    runtime: EngineRuntime,
    template: Arc<ChatTemplate>,
    web: Arc<WebTools>,
    req: AgentRequest,
    out: mpsc::Sender<AgentOut>,
) {
    if let Err(e) = run_inner(runtime, template, web, req, &out).await {
        let _ = out.send(AgentOut::Error(e)).await;
    }
}

async fn run_inner(
    runtime: EngineRuntime,
    template: Arc<ChatTemplate>,
    web: Arc<WebTools>,
    req: AgentRequest,
    out: &mpsc::Sender<AgentOut>,
) -> Result<(), String> {
    let family = template.detect_family();
    let mut all_tools: Vec<Value> = req.functions.clone();
    for b in &req.builtins {
        all_tools.push(match *b {
            WEB_SEARCH => web_search_definition().openai_function(),
            _ => web_fetch_definition().openai_function(),
        });
    }
    let all_tools = Value::Array(all_tools);
    let client_names: Vec<String> = req
        .functions
        .iter()
        .filter_map(|f| f.pointer("/function/name").and_then(Value::as_str))
        .map(str::to_string)
        .collect();
    // One page cache per request (session), so `cursor` ids stay valid across steps.
    let fetch_tool = web.fetcher.clone().map(WebFetchTool::new);

    let mut messages = req.messages;
    let mut remaining = req.max_tokens;
    let (mut prompt_tokens, mut cached_tokens, mut completion_tokens) = (0, 0, 0);
    let (mut prompt_ms, mut decode_ms) = (0.0, 0.0);
    let mut step = 0;
    loop {
        let final_step = step >= req.max_steps;
        let tools_now = (!final_step).then(|| all_tools.clone());
        let rr = RenderRequest {
            messages: messages.clone(),
            add_generation_prompt: true,
            tools: tools_now.clone(),
            enable_thinking: req.enable_thinking,
            extra: req.extra.clone(),
            ..Default::default()
        };
        let prompt_text = template
            .render(&rr)
            .map_err(|e| format!("chat template: {e}"))?;
        let reasoning_close =
            toolcall::expected_reasoning_close(family, &prompt_text, req.enable_thinking);
        let (grammar, lazy_trigger) = match &tools_now {
            Some(t) => toolcall::tool_grammar(family, t, &Choice::Auto, reasoning_close)
                .map(|(g, l)| (Some(g), l))
                .unwrap_or((None, None)),
            None => (None, None),
        };
        let mut assembler = Assembler::new(family, tools_now.as_ref(), &prompt_text);
        let tok = runtime.tokenizer.clone();
        let prompt = tok.encode(&prompt_text, tok.add_bos(), true);
        let (tx, mut rx) = mpsc::channel::<Event>(64);
        runtime
            .submit(Job {
                prompt,
                params: req.params.clone(),
                max_tokens: remaining.max(1),
                stop: req.stop.clone(),
                grammar,
                lazy_trigger,
                events: tx,
            })
            .map_err(|e| e.to_string())?;
        let mut finish = None;
        while let Some(ev) = rx.recv().await {
            match ev {
                Event::Text(t) => {
                    for d in assembler.push(&t) {
                        if d.get("tool_calls").is_none()
                            && out.send(AgentOut::Delta(d)).await.is_err()
                        {
                            return Ok(()); // client went away; dropping rx cancels the job
                        }
                    }
                }
                Event::Done(f) => {
                    finish = Some(f);
                    break;
                }
                Event::Error(e) => return Err(e),
            }
        }
        let f = finish.ok_or("generation ended without a result")?;
        for d in assembler.finish() {
            if d.get("tool_calls").is_none() {
                let _ = out.send(AgentOut::Delta(d)).await;
            }
        }
        assembler.prune_incomplete();
        prompt_tokens += f.prompt_tokens;
        cached_tokens += f.cached_tokens;
        completion_tokens += f.completion_tokens;
        prompt_ms += f.prompt_ms;
        decode_ms += f.decode_ms;
        remaining = remaining.saturating_sub(f.completion_tokens);
        step += 1;

        let msg = assembler.message();
        let calls: Vec<Value> = msg
            .get("tool_calls")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let done = |reason: String| AgentOut::Done {
            reason,
            prompt_tokens,
            cached_tokens,
            completion_tokens,
            prompt_ms,
            decode_ms,
            steps: step,
        };
        if calls.is_empty() {
            let _ = out
                .send(done(assembler.finish_reason(f.reason).to_string()))
                .await;
            return Ok(());
        }
        let name_of = |c: &Value| {
            c.pointer("/function/name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        };
        let client_calls: Vec<&Value> = calls
            .iter()
            .filter(|c| client_names.contains(&name_of(c)))
            .collect();
        if !client_calls.is_empty() {
            // The client runs its own functions; built-in calls of the same turn are dropped.
            for (i, c) in client_calls.iter().enumerate() {
                let mut c = (*c).clone();
                c["index"] = json!(i);
                let _ = out.send(AgentOut::Delta(json!({"tool_calls": [c]}))).await;
            }
            let _ = out.send(done("tool_calls".into())).await;
            return Ok(());
        }
        if remaining == 0 {
            let _ = out.send(done("length".into())).await;
            return Ok(());
        }

        // Run the built-in calls and continue the conversation.
        let mut assistant = Message::new(
            "assistant",
            Content::Text(
                msg.get("content")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            ),
        );
        let mut results = Vec::new();
        let mut tc = Vec::new();
        for c in &calls {
            let id = c
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("call")
                .to_string();
            let name = name_of(c);
            let args: Value = c
                .pointer("/function/arguments")
                .and_then(Value::as_str)
                .and_then(|s| serde_json::from_str(s).ok())
                .unwrap_or_else(|| json!({}));
            tc.push(ToolCall {
                id: Some(id.clone()),
                name: name.clone(),
                arguments: args.clone(),
            });
            let _ = out
                .send(AgentOut::Delta(json!({"llmario_tool_event": {
                    "name": name, "arguments": args, "status": "running"}})))
                .await;
            let t0 = Instant::now();
            let r = match name.as_str() {
                WEB_SEARCH => match &web.search {
                    Some(s) => s.call(&id, &args).await,
                    None => Err(llmario_engine_tools::ToolsError::Arguments(
                        "web_search is not available".into(),
                    )),
                },
                _ => match &fetch_tool {
                    Some(f) => f.call(&id, &args).await,
                    None => Err(llmario_engine_tools::ToolsError::Arguments(
                        "web_fetch is not available".into(),
                    )),
                },
            };
            let ms = t0.elapsed().as_millis() as u64;
            let (text, status, url) = match r {
                Ok(mut res) => {
                    res.text = truncate_chars(&res.text, MAX_RESULT_CHARS);
                    let url = res.url.clone();
                    (wrap_untrusted(&res, &nonce()), "done", url)
                }
                Err(e) => (format!("error: {e}"), "error", None),
            };
            let _ = out
                .send(AgentOut::Delta(json!({"llmario_tool_event": {
                    "name": name, "status": status, "url": url, "ms": ms}})))
                .await;
            let mut m = Message::new("tool", Content::Text(text));
            m.tool_call_id = Some(id);
            results.push(m);
        }
        assistant.tool_calls = Some(tc);
        messages.push(assistant);
        messages.extend(results);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_client_functions_and_builtins() {
        let tools = json!([
            {"type": "function", "function": {"name": "f"}},
            {"type": "web_search"},
            {"type": "web_fetch"}
        ]);
        let (f, b) = split_tools(&tools).unwrap();
        assert_eq!(f.len(), 1);
        assert_eq!(b, vec![WEB_SEARCH, WEB_FETCH]);
        assert!(split_tools(&json!([{"type": "computer_use"}])).is_err());
    }

    #[test]
    fn web_tools_policy() {
        let off = WebTools::disabled();
        assert!(off.check(&[WEB_FETCH]).unwrap_err().contains("web_access"));
        let fetch_only = WebTools::new(true, None).unwrap();
        assert!(fetch_only.check(&[WEB_FETCH]).is_ok());
        assert!(fetch_only
            .check(&[WEB_SEARCH])
            .unwrap_err()
            .contains("searxng_url"));
        assert_eq!(fetch_only.available(), vec![WEB_FETCH]);
    }

    #[test]
    fn truncation_is_char_safe() {
        let s = "é".repeat(10);
        let t = truncate_chars(&s, 4);
        assert!(t.starts_with("éééé\n[…"));
        assert_eq!(truncate_chars("abc", 4), "abc");
        assert_ne!(nonce(), nonce());
    }
}
