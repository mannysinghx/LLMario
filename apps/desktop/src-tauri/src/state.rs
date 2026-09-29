use llmario_core::{Config, Paths, ProfileKind, RuntimeError};
use llmario_supervisor::Supervisor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::oneshot;

/// Error shape returned to the UI.
#[derive(Serialize, Debug)]
pub struct CmdError {
    pub code: String,
    pub message: String,
}

impl CmdError {
    pub fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

impl From<RuntimeError> for CmdError {
    fn from(e: RuntimeError) -> Self {
        Self::new(e.code(), e.to_string())
    }
}

impl From<anyhow::Error> for CmdError {
    fn from(e: anyhow::Error) -> Self {
        Self::new("internal_error", format!("{e:#}"))
    }
}

/// UI-controlled runtime settings (persisted by the UI, applied here).
#[derive(Deserialize, Serialize, Clone, Debug, Default)]
#[serde(rename_all = "camelCase")]
pub struct Settings {
    pub profile: Option<ProfileKind>,
    pub context: Option<u32>,
}

pub struct Runtime {
    pub sup: Arc<Supervisor>,
    /// Private gateway base URL (loopback, ephemeral port).
    pub base: String,
    pub http: reqwest::Client,
    _server: tokio::task::JoinHandle<()>,
}

#[derive(Default)]
pub struct AppState {
    rt: tokio::sync::Mutex<Option<Arc<Runtime>>>,
    settings: std::sync::Mutex<Settings>,
    /// In-flight chats: request id → cancel signal.
    pub chats: std::sync::Mutex<HashMap<String, oneshot::Sender<()>>>,
    /// One download at a time (they write the shared registry file).
    pub pull_lock: tokio::sync::Mutex<()>,
}

impl AppState {
    /// The running runtime, building it on first use.
    pub async fn runtime(&self) -> Result<Arc<Runtime>, CmdError> {
        let mut guard = self.rt.lock().await;
        if let Some(r) = guard.as_ref() {
            return Ok(r.clone());
        }
        let settings = self.settings.lock().unwrap().clone();
        let rt = Arc::new(build(settings).await?);
        *guard = Some(rt.clone());
        Ok(rt)
    }

    /// Apply new settings: stop engines, rebuild the supervisor with the new profile/context.
    pub async fn restart(&self, settings: Settings) -> Result<Arc<Runtime>, CmdError> {
        *self.settings.lock().unwrap() = settings;
        let old = self.rt.lock().await.take();
        if let Some(old) = old {
            old.sup.shutdown().await;
            old._server.abort();
        }
        self.runtime().await
    }

    pub async fn shutdown(&self) {
        if let Some(rt) = self.rt.lock().await.take() {
            rt.sup.shutdown().await;
        }
    }
}

async fn build(settings: Settings) -> Result<Runtime, CmdError> {
    let paths = Paths::from_env()?;
    paths.ensure()?;
    let mut cfg = Config::load(&paths)?;
    if let Some(p) = settings.profile {
        cfg.runtime.profile = p;
    }
    if settings.context.is_some() {
        cfg.runtime.context = settings.context;
    }
    cfg.validate()?;
    let api_key = cfg.server.api_key.clone();
    let sup = llmario_runtime::build_supervisor(cfg, paths, None).await?;
    let (addr, server) = llmario_api::spawn_ephemeral(sup.clone()).await?;

    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(key) = api_key.filter(|k| !k.is_empty()) {
        let v = reqwest::header::HeaderValue::from_str(&format!("Bearer {key}"))
            .map_err(|e| CmdError::new("invalid_api_key", e.to_string()))?;
        headers.insert(reqwest::header::AUTHORIZATION, v);
    }
    let http = reqwest::Client::builder()
        .no_proxy()
        .default_headers(headers)
        .build()
        .map_err(|e| CmdError::new("internal_error", e.to_string()))?;
    Ok(Runtime {
        sup,
        base: format!("http://{addr}"),
        http,
        _server: server,
    })
}
