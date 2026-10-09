//! HTTP surface: `/health`, `/v1/models`, `/v1/chat/completions`, `/engine/*`.

use crate::engine::{EngineRuntime, Event, Job};
use crate::openai::{self, ChatRequest, PromptTokensDetails, Timings, Usage};
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
use llmario_engine_decode::SamplingParams;
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
}

type Shared = Arc<AppState>;

pub async fn serve_http(
    runtime: EngineRuntime,
    file: Arc<GgufFile>,
    opts: ServeOptions,
    plan: Plan,
    ledger: Arc<Ledger>,
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
                       "capabilities": {"tools": false, "json_schema": false, "vision": false}}
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
    if req.tools.is_some() || req.tool_choice.is_some() {
        return bad_request("tools are not supported by the native engine yet (M5)");
    }
    if let Some(rf) = &req.response_format {
        if rf
            .get("type")
            .and_then(|t| t.as_str())
            .is_some_and(|t| t != "text")
        {
            return bad_request("response_format is not supported by the native engine yet (M5)");
        }
    }
    if req.n.is_some_and(|n| n > 1) {
        return bad_request("n > 1 is not supported");
    }
    let Some(template) = s.runtime.template.clone() else {
        return bad_request("this model has no chat template");
    };
    let mut rr = RenderRequest {
        messages: req.messages.clone(),
        add_generation_prompt: true,
        tools: None,
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
    let tok = s.runtime.tokenizer.clone();
    let prompt = tok.encode(&prompt_text, tok.add_bos(), true);
    let max_tokens = req
        .max_completion_tokens
        .or(req.max_tokens)
        .unwrap_or(4096)
        .max(1);
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

    let (tx, rx) = tokio::sync::mpsc::channel::<Event>(64);
    let job = Job {
        prompt,
        params,
        max_tokens,
        stop: req.stop.clone().into_vec(),
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
        let mut text = String::new();
        let mut finish = None;
        let mut rx = rx;
        while let Some(ev) = rx.recv().await {
            match ev {
                Event::Text(t) => text.push_str(&t),
                Event::Done(f) => {
                    finish = Some(f);
                    break;
                }
                Event::Error(e) => return bad_request(&e),
            }
        }
        let f = finish.unwrap_or_else(|| crate::engine::Finish {
            reason: "stop",
            prompt_tokens: 0,
            cached_tokens: 0,
            completion_tokens: 0,
            prompt_ms: 0.0,
            decode_ms: 0.0,
        });
        let body = json!({
            "id": id, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": f.reason}],
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
    let body_stream = futures::stream::once(async move { sse(&first) })
        .chain(stream.flat_map(move |ev| {
            let items: Vec<String> = match ev {
                Event::Text(t) => vec![sse(&openai::chunk(
                    &id2,
                    &model2,
                    created,
                    json!({"content": t}),
                    None,
                ))],
                Event::Done(f) => {
                    let mut v = vec![sse(&openai::chunk(&id2, &model2, created, json!({}), Some(f.reason)))];
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
