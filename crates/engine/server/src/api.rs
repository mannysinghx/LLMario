//! HTTP surface: `/health`, `/v1/models`, `/v1/chat/completions`, `/engine/*`.

use crate::agent;
use crate::engine::{EngineRuntime, Event, Job};
use crate::openai::{self, ChatRequest, PromptTokensDetails, Timings, Usage};
use crate::toolcall;
use crate::ServeOptions;
use axum::body::Body;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use llmario_engine_chat::RenderRequest;
use llmario_engine_core::ledger::Ledger;
use llmario_engine_decode::{GrammarSpec, SamplingParams};
use llmario_engine_formats::GgufFile;
use llmario_engine_plan::Plan;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub struct AppState {
    pub runtime: EngineRuntime,
    pub file: Arc<GgufFile>,
    pub opts: ServeOptions,
    pub plan: Plan,
    pub ledger: Arc<Ledger>,
    pub started: Instant,
    pub sleeping: AtomicBool,
    pub shutdown: tokio::sync::Notify,
    pub web: Arc<agent::WebTools>,
}

type Shared = Arc<AppState>;

pub async fn serve_http(
    runtime: EngineRuntime,
    file: Arc<GgufFile>,
    opts: ServeOptions,
    plan: Plan,
    ledger: Arc<Ledger>,
    web: Arc<agent::WebTools>,
) -> anyhow::Result<()> {
    let listen = opts.listen.clone();
    let state: Shared = Arc::new(AppState {
        runtime,
        file,
        opts,
        plan,
        ledger,
        started: Instant::now(),
        sleeping: AtomicBool::new(false),
        shutdown: tokio::sync::Notify::new(),
        web,
    });
    let app = Router::new()
        .route("/health", get(health))
        .route("/healthz", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/engine/plan", get(plan_get))
        .route("/engine/ledger", get(ledger_get))
        .route("/engine/stats", get(stats_get))
        .route("/engine/control", post(control))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind(&listen).await?;
    tracing::info!(%listen, "listening");
    let shutdown = state.clone();
    axum::serve(listener, app)
        .with_graceful_shutdown(async move { shutdown.shutdown.notified().await })
        .await?;
    crate::emit_state("exiting");
    Ok(())
}

async fn health(State(s): State<Shared>) -> Json<Value> {
    Json(
        json!({"status": "ok", "model": s.opts.model_id, "uptime_s": s.started.elapsed().as_secs()}),
    )
}

async fn models(State(s): State<Shared>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": s.opts.model_id,
            "object": "model",
            "owned_by": "llmario",
            "engine": {"backend": "native", "plan_hash": s.plan.hash, "arch": s.plan.arch,
                       "context": s.plan.ctx_per_slot, "kernels": s.runtime.kernels,
                       "kv_type": s.runtime.kv_type.name(), "slots": s.plan.slots,
                       "device": s.runtime.backend,
                       "capabilities": {"tools": true, "json_schema": true, "vision": false, "web_tools": s.web.available()}}
        }]
    }))
}

async fn plan_get(State(s): State<Shared>) -> Json<Value> {
    Json(json!({"plan": s.plan, "rendered": llmario_engine_plan::render(&s.plan)}))
}

async fn ledger_get(State(s): State<Shared>) -> Json<Value> {
    serde_json::to_value(s.ledger.snapshot())
        .map(Json)
        .unwrap_or_else(|e| Json(json!({"error": e.to_string()})))
}

async fn stats_get(State(s): State<Shared>) -> Json<Value> {
    let mut v = s.runtime.stats.snapshot();
    v["threads"] = json!(s.runtime.threads);
    v["kernels"] = json!(s.runtime.kernels);
    v["device"] = json!(s.runtime.backend);
    v["bytes_per_token"] = json!(s.plan.bytes_per_token);
    v["sleeping"] = json!(s.sleeping.load(Ordering::Relaxed));
    Json(v)
}

#[derive(serde::Deserialize)]
struct Control {
    action: String,
}

async fn control(State(s): State<Shared>, Json(c): Json<Control>) -> Response {
    match c.action.as_str() {
        "exit" => {
            s.shutdown.notify_one();
            Json(json!({"ok": true})).into_response()
        }
        "sleep" => {
            s.sleeping.store(true, Ordering::Relaxed);
            Json(json!({"ok": true, "note": "M1: sleep only marks state; KV release lands with the arena"})).into_response()
        }
        "wake" => {
            s.sleeping.store(false, Ordering::Relaxed);
            Json(json!({"ok": true})).into_response()
        }
        other => (
            StatusCode::BAD_REQUEST,
            Json(openai::error_body(
                400,
                "invalid_request_error",
                &format!("unknown action {other}"),
            )),
        )
            .into_response(),
    }
}

fn bad_request(msg: &str) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(openai::error_body(400, "invalid_request_error", msg)),
    )
        .into_response()
}

async fn chat(State(s): State<Shared>, body: axum::body::Bytes) -> Response {
    let req: ChatRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return bad_request(&format!("invalid request: {e}")),
    };
    if req.n.is_some_and(|n| n > 1) {
        return bad_request("n > 1 is not supported");
    }
    // Refused rather than ignored, as the gateway does, so a client never silently gets none.
    if req.logprobs == Some(true) || req.top_logprobs.is_some_and(|k| k > 0) {
        return bad_request("logprobs are not supported yet");
    }
    let Some(template) = s.runtime.template.clone() else {
        return bad_request("this model has no chat template");
    };
    let family = template.detect_family();
    let choice = match toolcall::parse_tool_choice(req.tool_choice.as_ref()) {
        Ok(c) => c,
        Err(e) => return bad_request(&e),
    };
    let (functions, builtins) = match req.tools.as_ref().map(agent::split_tools).transpose() {
        Ok(v) => v.unwrap_or_default(),
        Err(e) => return bad_request(&e),
    };
    let tools = (!functions.is_empty()).then(|| Value::Array(functions.clone()));
    if let (toolcall::Choice::Named(n), Some(t)) = (&choice, &tools) {
        let known = t
            .as_array()
            .unwrap()
            .iter()
            .any(|d| d.pointer("/function/name").and_then(|v| v.as_str()) == Some(n.as_str()));
        if !known {
            return bad_request(&format!("tool_choice names '{n}', which is not in tools"));
        }
    }
    // tool_choice "none": the model is not shown the tools and its output is not parsed for calls.
    let render_tools = if choice == toolcall::Choice::None {
        None
    } else {
        tools.clone()
    };
    let messages = match toolcall::to_chat_messages(&req.messages) {
        Ok(m) => m,
        Err(e) => return bad_request(&e),
    };
    if !builtins.is_empty() && choice != toolcall::Choice::None {
        if let Err(e) = s.web.check(&builtins) {
            return bad_request(&e);
        }
        let areq = agent::AgentRequest {
            messages,
            functions,
            builtins,
            enable_thinking: req.enable_thinking,
            extra: req.chat_template_kwargs.clone().unwrap_or_default(),
            params: sampling_params(&req),
            max_tokens: req
                .max_completion_tokens
                .or(req.max_tokens)
                .unwrap_or(4096)
                .max(1),
            stop: req.stop.clone().into_vec(),
            max_steps: req.max_steps.unwrap_or(agent::DEFAULT_MAX_STEPS).min(16),
        };
        return agent_response(&s, &req, template, areq).await;
    }
    let mut rr = RenderRequest {
        messages,
        add_generation_prompt: true,
        tools: render_tools.clone(),
        enable_thinking: req.enable_thinking,
        ..Default::default()
    };
    if let Some(kw) = &req.chat_template_kwargs {
        rr.extra = kw.clone();
    }
    let prompt_text = match template.render(&rr) {
        Ok(t) => t,
        Err(e) => return bad_request(&format!("chat template: {e}")),
    };
    let reasoning_close =
        toolcall::expected_reasoning_close(family, &prompt_text, req.enable_thinking);
    let response_grammar = match &req.response_format {
        None => None,
        Some(rf) => match rf.get("type").and_then(|t| t.as_str()) {
            None | Some("text") => None,
            Some("json_object") => Some(GrammarSpec::JsonSchema(json!({"type": "object"}))),
            Some("json_schema") => {
                let schema = rf
                    .get("json_schema")
                    .and_then(|j| j.get("schema").cloned().or_else(|| Some(j.clone())))
                    .unwrap_or(json!({"type": "object"}));
                Some(GrammarSpec::JsonSchema(schema))
            }
            Some(other) => {
                return bad_request(&format!("unsupported response_format type {other}"))
            }
        },
    };
    // Tools take precedence over response_format; either starts after the reasoning block when
    // the model is expected to reason first.
    let (grammar, lazy_trigger) = match &render_tools {
        Some(t) => match toolcall::tool_grammar(family, t, &choice, reasoning_close) {
            Some((g, lazy)) => (Some(g), lazy),
            None => (None, None),
        },
        None => match response_grammar {
            Some(g) => (Some(g), reasoning_close.map(str::to_string)),
            None => (None, None),
        },
    };
    let assembler = toolcall::Assembler::new(family, render_tools.as_ref(), &prompt_text);
    let tok = s.runtime.tokenizer.clone();
    let prompt = tok.encode(&prompt_text, tok.add_bos(), true);
    let max_tokens = req
        .max_completion_tokens
        .or(req.max_tokens)
        .unwrap_or(4096)
        .max(1);
    let params = sampling_params(&req);

    let (tx, rx) = tokio::sync::mpsc::channel::<Event>(64);
    let job = Job {
        prompt,
        params,
        max_tokens,
        stop: req.stop.clone().into_vec(),
        grammar,
        lazy_trigger,
        events: tx,
    };
    if let Err(e) = s.runtime.submit(job) {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(openai::error_body(503, "server_error", &e.to_string())),
        )
            .into_response();
    }
    let model = req.model.clone().unwrap_or_else(|| s.opts.model_id.clone());
    let id = openai::completion_id();
    let created = openai::now_secs();
    let include_usage = req.stream_options.as_ref().is_some_and(|o| o.include_usage);

    if !req.stream {
        let mut assembler = assembler;
        let mut finish = None;
        let mut rx = rx;
        while let Some(ev) = rx.recv().await {
            match ev {
                Event::Text(t) => {
                    assembler.push(&t);
                }
                Event::Done(f) => {
                    finish = Some(f);
                    break;
                }
                Event::Error(e) => return bad_request(&e),
            }
        }
        let f = finish.unwrap_or(crate::engine::Finish {
            reason: "stop",
            prompt_tokens: 0,
            cached_tokens: 0,
            completion_tokens: 0,
            prompt_ms: 0.0,
            decode_ms: 0.0,
        });
        assembler.finish();
        assembler.prune_incomplete();
        let body = json!({
            "id": id, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": assembler.message(), "finish_reason": assembler.finish_reason(f.reason)}],
            "usage": usage_of(&f),
            "timings": timings_of(&f),
        });
        return Json(body).into_response();
    }

    let first = openai::chunk(
        &id,
        &model,
        created,
        json!({"role": "assistant", "content": ""}),
        None,
    );
    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    let model2 = model.clone();
    let id2 = id.clone();
    let mut assembler = assembler;
    let body_stream = futures::stream::once(async move { sse(&first) })
        .chain(stream.flat_map(move |ev| {
            let items: Vec<String> = match ev {
                Event::Text(t) => assembler
                    .push(&t)
                    .into_iter()
                    .map(|d| sse(&openai::chunk(&id2, &model2, created, d, None)))
                    .collect(),
                Event::Done(f) => {
                    let mut v: Vec<String> = assembler
                        .finish()
                        .into_iter()
                        .map(|d| sse(&openai::chunk(&id2, &model2, created, d, None)))
                        .collect();
                    let reason = assembler.finish_reason(f.reason);
                    v.push(sse(&openai::chunk(&id2, &model2, created, json!({}), Some(reason))));
                    if include_usage {
                        v.push(sse(&json!({
                            "id": id2, "object": "chat.completion.chunk", "created": created, "model": model2,
                            "choices": [], "usage": usage_of(&f), "timings": timings_of(&f)
                        })));
                    }
                    v.push("data: [DONE]\n\n".to_string());
                    v
                }
                Event::Error(e) => vec![sse(&openai::error_body(500, "server_error", &e))],
            };
            futures::stream::iter(items)
        }))
        .map(|s| Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(s)));
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(body_stream))
        .unwrap()
}

fn sampling_params(req: &ChatRequest) -> SamplingParams {
    let mut params = SamplingParams::default();
    if let Some(t) = req.temperature {
        params.temperature = t;
    }
    if let Some(v) = req.top_p {
        params.top_p = v;
    }
    if let Some(v) = req.top_k {
        params.top_k = v;
    }
    if let Some(v) = req.min_p {
        params.min_p = v;
    }
    if let Some(v) = req.presence_penalty {
        params.presence_penalty = v;
    }
    if let Some(v) = req.frequency_penalty {
        params.frequency_penalty = v;
    }
    if let Some(v) = req.repeat_penalty {
        params.repeat_penalty = v;
    }
    if let Some(v) = req.repeat_last_n {
        params.repeat_last_n = v;
    }
    params.seed = req.seed;
    params
}

/// Run the built-in-tool agent loop and answer in the request's mode (stream or not).
async fn agent_response(
    s: &Shared,
    req: &ChatRequest,
    template: Arc<llmario_engine_chat::ChatTemplate>,
    areq: agent::AgentRequest,
) -> Response {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<agent::AgentOut>(64);
    tokio::spawn(agent::run(
        s.runtime.clone(),
        template,
        s.web.clone(),
        areq,
        tx,
    ));
    let model = req.model.clone().unwrap_or_else(|| s.opts.model_id.clone());
    let id = openai::completion_id();
    let created = openai::now_secs();
    let include_usage = req.stream_options.as_ref().is_some_and(|o| o.include_usage);
    let usage_json = |pt: usize, ct: usize, cached: usize| {
        json!({"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct,
               "prompt_tokens_details": {"cached_tokens": cached}})
    };

    if !req.stream {
        let (mut content, mut reasoning) = (String::new(), String::new());
        let (mut calls, mut events) = (Vec::new(), Vec::new());
        while let Some(o) = rx.recv().await {
            match o {
                agent::AgentOut::Delta(d) => {
                    if let Some(t) = d.get("content").and_then(Value::as_str) {
                        content.push_str(t);
                    }
                    if let Some(t) = d.get("reasoning_content").and_then(Value::as_str) {
                        reasoning.push_str(t);
                    }
                    if let Some(c) = d.get("tool_calls").and_then(Value::as_array) {
                        for c in c {
                            let mut c = c.clone();
                            if let Some(o) = c.as_object_mut() {
                                o.remove("index");
                            }
                            calls.push(c);
                        }
                    }
                    if let Some(e) = d.get("llmario_tool_event") {
                        events.push(e.clone());
                    }
                }
                agent::AgentOut::Done {
                    reason,
                    prompt_tokens,
                    cached_tokens,
                    completion_tokens,
                    prompt_ms,
                    decode_ms,
                    steps,
                } => {
                    let mut msg = json!({"role": "assistant",
                        "content": if content.is_empty() && !calls.is_empty() { Value::Null } else { json!(content) }});
                    if !reasoning.is_empty() {
                        msg["reasoning_content"] = json!(reasoning);
                    }
                    if !calls.is_empty() {
                        msg["tool_calls"] = Value::Array(calls);
                    }
                    return Json(json!({
                        "id": id, "object": "chat.completion", "created": created, "model": model,
                        "choices": [{"index": 0, "message": msg, "finish_reason": reason}],
                        "usage": usage_json(prompt_tokens, completion_tokens, cached_tokens),
                        "timings": {"prompt_n": prompt_tokens - cached_tokens, "prompt_ms": prompt_ms,
                                    "predicted_n": completion_tokens, "predicted_ms": decode_ms},
                        "llmario": {"steps": steps, "tool_events": events},
                    }))
                    .into_response();
                }
                agent::AgentOut::Error(e) => return bad_request(&e),
            }
        }
        return bad_request("agent loop ended without a result");
    }

    let first = openai::chunk(
        &id,
        &model,
        created,
        json!({"role": "assistant", "content": ""}),
        None,
    );
    let body_stream = futures::stream::once(async move { sse(&first) }).chain(
        futures::stream::poll_fn(move |cx| rx.poll_recv(cx)).flat_map(move |o| {
            let items: Vec<String> = match o {
                agent::AgentOut::Delta(d) => vec![sse(&openai::chunk(&id, &model, created, d, None))],
                agent::AgentOut::Done {
                    reason,
                    prompt_tokens,
                    cached_tokens,
                    completion_tokens,
                    prompt_ms,
                    decode_ms,
                    ..
                } => {
                    let mut v = vec![sse(&openai::chunk(&id, &model, created, json!({}), Some(&reason)))];
                    if include_usage {
                        v.push(sse(&json!({
                            "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                            "choices": [], "usage": usage_json(prompt_tokens, completion_tokens, cached_tokens),
                            "timings": {"prompt_n": prompt_tokens - cached_tokens, "prompt_ms": prompt_ms,
                                        "predicted_n": completion_tokens, "predicted_ms": decode_ms}
                        })));
                    }
                    v.push("data: [DONE]\n\n".to_string());
                    v
                }
                agent::AgentOut::Error(e) => vec![
                    sse(&openai::error_body(500, "server_error", &e)),
                    "data: [DONE]\n\n".to_string(),
                ],
            };
            futures::stream::iter(items)
        }),
    )
    .map(|s| Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(s)));
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(body_stream))
        .unwrap()
}

fn sse(v: &Value) -> String {
    format!("data: {v}\n\n")
}

fn usage_of(f: &crate::engine::Finish) -> Usage {
    Usage {
        prompt_tokens: f.prompt_tokens,
        completion_tokens: f.completion_tokens,
        total_tokens: f.prompt_tokens + f.completion_tokens,
        prompt_tokens_details: Some(PromptTokensDetails {
            cached_tokens: f.cached_tokens,
        }),
    }
}

fn timings_of(f: &crate::engine::Finish) -> Timings {
    Timings {
        prompt_n: f.prompt_tokens - f.cached_tokens,
        prompt_ms: f.prompt_ms,
        predicted_n: f.completion_tokens,
        predicted_ms: f.decode_ms,
    }
}
