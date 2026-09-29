//! Local HTTP gateway.
//!
//! Endpoints: `GET /healthz`, `GET /v1/models`, `GET /v1/models/{id}`,
//! `POST /v1/chat/completions` (streaming and non-streaming), `GET /metrics`.
//!
//! Security defaults: loopback bind (enforced by config validation), optional bearer key
//! (required for remote bind), Host-header check against DNS rebinding in loopback mode,
//! JSON-only bodies (a cross-site form post cannot reach the handlers), request-field
//! allowlist, and no prompt/completion content in logs.

pub mod chat;
pub mod metrics;
pub mod validate;

use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use llmario_core::{config::is_loopback_host, RuntimeError};
use llmario_supervisor::Supervisor;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;

pub use metrics::Metrics;

#[derive(Clone)]
pub struct AppState {
    pub sup: Arc<Supervisor>,
    pub metrics: Arc<Metrics>,
    pub api_key: Option<Arc<str>>,
    pub loopback_only: bool,
    pub request_timeout: Duration,
}

impl AppState {
    pub fn new(sup: Arc<Supervisor>) -> Self {
        let cfg = &sup.cfg;
        Self {
            api_key: cfg
                .server
                .api_key
                .clone()
                .filter(|k| !k.is_empty())
                .map(Into::into),
            loopback_only: is_loopback_host(&cfg.server.host),
            request_timeout: Duration::from_secs(cfg.runtime.request_timeout_secs),
            metrics: Arc::new(Metrics::default()),
            sup,
        }
    }
}

/// OpenAI-style error response.
pub struct ApiError(pub RuntimeError);

impl From<RuntimeError> for ApiError {
    fn from(e: RuntimeError) -> Self {
        Self(e)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status =
            StatusCode::from_u16(self.0.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let body = error_body(&self.0);
        let mut resp = (status, Json(body)).into_response();
        if matches!(self.0, RuntimeError::Busy(_)) {
            resp.headers_mut()
                .insert("retry-after", "5".parse().unwrap());
        }
        resp
    }
}

pub fn error_body(e: &RuntimeError) -> serde_json::Value {
    json!({"error": {"message": e.to_string(), "type": e.kind(), "code": e.code(), "param": null}})
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/models/{id}", get(get_model))
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/metrics", get(metrics_handler))
        .route("/healthz", get(healthz))
        .layer(middleware::from_fn_with_state(state.clone(), guard))
        .with_state(state)
}

/// Host check (DNS rebinding) + bearer auth. `/healthz` skips auth but not the Host check.
async fn guard(State(st): State<AppState>, req: Request, next: Next) -> Response {
    if st.loopback_only {
        let host = req
            .headers()
            .get("host")
            .and_then(|h| h.to_str().ok())
            .unwrap_or("");
        let hostname = strip_port(host);
        if !hostname.is_empty() && !is_loopback_host(hostname) {
            return ApiError(RuntimeError::Unauthorized(format!(
                "Host '{hostname}' rejected: this server only answers loopback hostnames"
            )))
            .into_response();
        }
    }
    if req.uri().path() != "/healthz" {
        if let Some(key) = &st.api_key {
            if !authorized(req.headers(), key) {
                return ApiError(RuntimeError::Unauthorized(
                    "missing or invalid API key (Authorization: Bearer <key>)".into(),
                ))
                .into_response();
            }
        }
    }
    next.run(req).await
}

fn strip_port(host: &str) -> &str {
    if let Some(rest) = host.strip_prefix('[') {
        return rest.split(']').next().unwrap_or("");
    }
    host.rsplit_once(':').map(|(h, _)| h).unwrap_or(host)
}

fn authorized(headers: &HeaderMap, key: &str) -> bool {
    let Some(given) = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
    else {
        return false;
    };
    constant_time_eq(given.trim().as_bytes(), key.as_bytes())
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

async fn healthz(State(st): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "version": llmario_core::VERSION,
        "loaded_models": st.sup.loaded().await.len(),
    }))
}

fn model_json(m: &llmario_registry::ModelEntry, loaded: bool) -> serde_json::Value {
    let created = chrono_like_epoch(&m.added_at);
    json!({
        "id": m.id,
        "object": "model",
        "created": created,
        "owned_by": "local",
        "llmario": {
            "family": m.family,
            "format": m.format,
            "backend": llmario_supervisor::planner::backend_for(m.format),
            "quantization": m.quantization,
            "license": m.license,
            "size_bytes": m.size_bytes,
            "context_max": m.shape.as_ref().and_then(|s| s.context_max),
            "loaded": loaded,
        }
    })
}

fn chrono_like_epoch(rfc3339: &str) -> i64 {
    // Avoid a chrono dependency here: parse only the date part, which is enough for listing.
    let d: Vec<i64> = rfc3339
        .get(..10)
        .unwrap_or("")
        .split('-')
        .filter_map(|p| p.parse().ok())
        .collect();
    if d.len() != 3 {
        return 0;
    }
    let (y, m, day) = (d[0], d[1], d[2]);
    let (y2, m2) = if m <= 2 { (y - 1, m + 12) } else { (y, m) };
    let days = 365 * y2 + y2 / 4 - y2 / 100 + y2 / 400 + (153 * (m2 - 3) + 2) / 5 + day - 719469;
    days * 86400
}

async fn list_models(State(st): State<AppState>) -> Json<serde_json::Value> {
    let _ = st.sup.reload_registry();
    let loaded: Vec<String> = st.sup.loaded().await.into_iter().map(|e| e.model).collect();
    let reg = st.sup.registry();
    let data: Vec<_> = reg
        .models
        .iter()
        .map(|m| model_json(m, loaded.contains(&m.id)))
        .collect();
    Json(json!({"object": "list", "data": data}))
}

async fn get_model(
    State(st): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _ = st.sup.reload_registry();
    let reg = st.sup.registry();
    let m = reg
        .get(&id)
        .ok_or_else(|| RuntimeError::ModelNotFound(id.clone()))?;
    let loaded = st.sup.loaded().await.iter().any(|e| e.model == id);
    Ok(Json(model_json(m, loaded)))
}

async fn metrics_handler(State(st): State<AppState>) -> Response {
    let engines = st.sup.loaded().await;
    let body = st.metrics.render(&engines);
    ([("content-type", "text/plain; version=0.0.4")], body).into_response()
}

/// Serve the gateway on an ephemeral loopback port in the background (used by `run`,
/// `bench`, and integration tests so they exercise the same request path as `serve`).
pub async fn spawn_ephemeral(
    sup: Arc<Supervisor>,
) -> anyhow::Result<(std::net::SocketAddr, tokio::task::JoinHandle<()>)> {
    let mut state = AppState::new(sup);
    state.loopback_only = true;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let app = router(state);
    let handle = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    Ok((addr, handle))
}

/// Bind and serve until Ctrl-C / SIGTERM, then stop all engines.
pub async fn serve(sup: Arc<Supervisor>) -> anyhow::Result<()> {
    let cfg = sup.cfg.clone();
    let state = AppState::new(sup.clone());
    let app = router(state);
    let addr = format!("{}:{}", cfg.server.host, cfg.server.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| anyhow::anyhow!("cannot bind {addr}: {e}"))?;
    let local = listener.local_addr()?;
    if !is_loopback_host(&cfg.server.host) {
        tracing::warn!(%local, "REMOTE MODE: the API is reachable from other machines on this network; an API key is required");
    }
    tracing::info!(%local, "llmario API listening");
    eprintln!("llmario API listening on http://{local}  (OpenAI base URL: http://{local}/v1)");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    eprintln!("shutting down: stopping engines…");
    sup.shutdown().await;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        if let Ok(mut s) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! { _ = ctrl_c => {}, _ = term => {} }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_parsing() {
        assert_eq!(strip_port("127.0.0.1:11500"), "127.0.0.1");
        assert_eq!(strip_port("localhost"), "localhost");
        assert_eq!(strip_port("[::1]:80"), "::1");
        assert_eq!(strip_port("evil.example:11500"), "evil.example");
    }

    #[test]
    fn auth_compare() {
        let mut h = HeaderMap::new();
        assert!(!authorized(&h, "k"));
        h.insert("authorization", "Bearer k".parse().unwrap());
        assert!(authorized(&h, "k"));
        assert!(!authorized(&h, "kk"));
        h.insert("authorization", "Basic k".parse().unwrap());
        assert!(!authorized(&h, "k"));
    }

    #[test]
    fn epoch_from_date() {
        assert_eq!(chrono_like_epoch("1970-01-01T00:00:00Z"), 0);
        assert_eq!(chrono_like_epoch("2026-09-28T12:00:00Z"), 1790553600);
        assert_eq!(chrono_like_epoch("bad"), 0);
    }
}
