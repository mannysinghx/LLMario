//! Commands the window calls via Tauri IPC. Long operations (load, pull, chat) are async and
//! stream progress through IPC channels.

use crate::state::{AppState, CmdError, Runtime, Settings};
use llmario_core::{BackendKind, ModelFormat, ResolvedProfile};
use llmario_registry::download::{self, HubClient, ProgressFn, PullOptions, PullProgress};
use llmario_registry::{Catalog, ModelEntry, Registry};
use llmario_runtime::chat::{stream_chat, ChatEvent, ChatStats};
use llmario_runtime::library::CatalogView;
use llmario_supervisor::planner::{self, Selection};
use serde::Serialize;
use serde_json::{json, Value};
use std::sync::Arc;
use tauri::ipc::Channel;
use tauri::State;
use tokio::sync::oneshot;

type CmdResult<T> = Result<T, CmdError>;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BackendView {
    kind: BackendKind,
    available: bool,
    version: Option<String>,
    detail: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Overview {
    version: &'static str,
    cpu: String,
    os: String,
    memory_total_bytes: u64,
    memory_available_bytes: u64,
    gpu: Option<String>,
    gpu_memory_bytes: Option<u64>,
    unified_memory: bool,
    backends: Vec<BackendView>,
    profile: ResolvedProfile,
    home: String,
    loaded: Vec<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelView {
    id: String,
    family: Option<String>,
    format: ModelFormat,
    backend: BackendKind,
    backend_available: bool,
    quantization: Option<String>,
    size_bytes: u64,
    license: Option<String>,
    source: Option<String>,
    context_max: Option<u32>,
    managed: bool,
    fits: bool,
    /// Fits, but above the comfortable target for a machine with 16 GB or less.
    tight: bool,
    /// Decode speed on this computer: measured, or predicted from memory bandwidth.
    speed: Option<llmario_supervisor::speed::SpeedEstimate>,
    needs_bytes: u64,
    budget_bytes: u64,
    loaded: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LoadInfo {
    model: String,
    backend: BackendKind,
    ready_seconds: f64,
    footprint_bytes: Option<u64>,
    estimate_bytes: u64,
    budget_bytes: u64,
    profile: ResolvedProfile,
    notes: Vec<String>,
    /// Set when the other LLMario edition also has models loaded (shown to the user).
    warning: Option<String>,
}

async fn overview_of(rt: &Runtime) -> Overview {
    let sup = &rt.sup;
    let hw = &sup.hw;
    let gpu = hw.primary_gpu();
    Overview {
        version: llmario_core::VERSION,
        cpu: hw.cpu_brand.clone(),
        os: hw.os_version.clone(),
        memory_total_bytes: hw.total_memory_bytes,
        memory_available_bytes: hw.available_memory_bytes,
        gpu: gpu.map(|g| g.name.clone()),
        gpu_memory_bytes: gpu.and_then(|g| g.memory_total_bytes),
        unified_memory: hw.unified_memory,
        // MLX runs only on Apple Silicon Macs; elsewhere it would always show as "not found".
        backends: [BackendKind::LlamaCpp, BackendKind::Mlx, BackendKind::Native]
            .iter()
            .filter(|k| cfg!(target_os = "macos") || **k != BackendKind::Mlx)
            .filter_map(|k| sup.statuses().get(k))
            .map(|s| BackendView {
                kind: s.kind,
                available: s.available,
                version: s.version.clone(),
                detail: s.detail.clone(),
            })
            .collect(),
        profile: ResolvedProfile::resolve(sup.profile, sup.cfg.runtime.context),
        home: sup.paths.home.display().to_string(),
        loaded: sup.loaded().await.into_iter().map(|e| e.model).collect(),
    }
}

fn model_view(rt: &Runtime, m: &ModelEntry, loaded: &[String]) -> ModelView {
    let sup = &rt.sup;
    let backend = planner::backend_for(m.format);
    let profile = planner::effective_profile(sup.profile, sup.cfg.runtime.context, m);
    let sel = Selection {
        model: m.clone(),
        backend,
        profile,
        reason: String::new(),
    };
    let plan = sup.plan_memory(&sel, 0);
    ModelView {
        id: m.id.clone(),
        family: m.family.clone(),
        format: m.format,
        backend,
        backend_available: sup.statuses().get(&backend).is_some_and(|s| s.available),
        quantization: m.quantization.clone(),
        size_bytes: m.size_bytes,
        license: m.license.clone(),
        source: m.source.as_ref().map(|s| s.repo.clone()),
        context_max: m.shape.as_ref().and_then(|s| s.context_max),
        managed: m.managed,
        fits: plan.fits,
        tight: plan.tight,
        speed: sup.speed(&sel, &sup.autotune()),
        needs_bytes: plan.total_bytes,
        budget_bytes: plan.budget_bytes,
        loaded: loaded.contains(&m.id),
    }
}

/// First call from the UI: build the runtime with the UI's saved settings.
#[tauri::command]
pub async fn start(state: State<'_, AppState>, settings: Settings) -> CmdResult<Overview> {
    let rt = state.restart(settings).await?;
    Ok(overview_of(&rt).await)
}

#[tauri::command]
pub async fn overview(state: State<'_, AppState>) -> CmdResult<Overview> {
    let rt = state.runtime().await?;
    Ok(overview_of(&rt).await)
}

/// Change profile/context. Loaded models are stopped; the next message reloads with new values.
#[tauri::command]
pub async fn apply_settings(state: State<'_, AppState>, settings: Settings) -> CmdResult<Overview> {
    let rt = state.restart(settings).await?;
    Ok(overview_of(&rt).await)
}

#[tauri::command]
pub async fn list_models(state: State<'_, AppState>) -> CmdResult<Vec<ModelView>> {
    let rt = state.runtime().await?;
    rt.sup.reload_registry()?;
    let loaded: Vec<String> = rt.sup.loaded().await.into_iter().map(|e| e.model).collect();
    let reg = rt.sup.registry();
    Ok(reg
        .models
        .iter()
        .filter(|m| m.format != ModelFormat::Mock)
        .map(|m| model_view(&rt, m, &loaded))
        .collect())
}

#[tauri::command]
pub async fn list_catalog(state: State<'_, AppState>) -> CmdResult<Vec<CatalogView>> {
    let rt = state.runtime().await?;
    Ok(llmario_runtime::library::catalog_views(&rt.sup))
}

/// Download a catalog model with checksum verification, streaming progress to the UI.
#[tauri::command]
pub async fn pull_model(
    state: State<'_, AppState>,
    id: String,
    on_progress: Channel<PullProgress>,
) -> CmdResult<ModelView> {
    let _one_at_a_time = state.pull_lock.lock().await;
    let rt = state.runtime().await?;
    let cat = Catalog::builtin();
    let entry = cat
        .get(&id)
        .cloned()
        .ok_or_else(|| CmdError::new("model_not_found", format!("'{id}' is not in the catalog")))?;
    let paths = rt.sup.paths.clone();
    let mut reg = Registry::load(&paths.registry_file())?;
    let hub = HubClient::from_env()?;
    let cb: ProgressFn = Arc::new(move |p| {
        let _ = on_progress.send(p);
    });
    let opts = PullOptions {
        force: false,
        show_progress: false,
        use_hf_cache: true,
        on_progress: Some(cb),
    };
    let out = download::pull(&hub, &entry, &paths, &mut reg, &opts).await?;
    rt.sup.reload_registry()?;
    Ok(model_view(&rt, &out.entry, &[]))
}

/// Add a model file or folder the user dropped onto the window. Registered in place: the files
/// are hashed, never copied, and removing the model later never deletes them.
#[tauri::command]
pub async fn add_model(state: State<'_, AppState>, path: String) -> CmdResult<ModelView> {
    let _serialize_registry_writes = state.pull_lock.lock().await;
    let rt = state.runtime().await?;
    let registry_file = rt.sup.paths.registry_file();
    let entry = tokio::task::spawn_blocking(move || {
        let mut reg = Registry::load(&registry_file)?;
        download::add_local_auto(std::path::Path::new(&path), &mut reg)
    })
    .await
    .map_err(|e| CmdError::new("internal_error", e.to_string()))?
    .map_err(|e| CmdError::new("invalid_model", format!("{e:#}")))?;
    rt.sup.reload_registry()?;
    Ok(model_view(&rt, &entry, &[]))
}

/// Remove a model. Downloaded files are deleted; files registered from elsewhere are kept.
#[tauri::command]
pub async fn remove_model(state: State<'_, AppState>, id: String) -> CmdResult<()> {
    let rt = state.runtime().await?;
    rt.sup.unload(&id).await;
    let paths = rt.sup.paths.clone();
    let mut reg = Registry::load(&paths.registry_file())?;
    download::remove(&id, &paths, &mut reg)?;
    rt.sup.reload_registry()?;
    Ok(())
}

/// Load a model (or confirm it is loaded) and report how it went.
#[tauri::command]
pub async fn load_model(state: State<'_, AppState>, name: String) -> CmdResult<LoadInfo> {
    let rt = state.runtime().await?;
    let sel = rt.sup.select(&name)?;
    let plan = rt.sup.plan_memory(&sel, 0);
    if !plan.fits {
        return Err(CmdError::new(
            "insufficient_memory",
            plan.refusal(&sel.model.id, &sel.profile),
        ));
    }
    let lease = rt.sup.acquire(&name).await?;
    let e = &lease.engine;
    let mut notes = plan.notes.clone();
    notes.extend(e.notes.iter().cloned());
    Ok(LoadInfo {
        model: e.model.id.clone(),
        backend: e.backend,
        ready_seconds: e.ready_after().as_secs_f64(),
        footprint_bytes: llmario_hardware::process_memory_bytes(e.pid),
        estimate_bytes: plan.total_bytes,
        budget_bytes: plan.budget_bytes,
        profile: e.profile.clone(),
        notes,
        warning: llmario_supervisor::sibling::sibling_usage(&rt.sup.paths).map(|u| u.warning()),
    })
}

#[tauri::command]
pub async fn unload_model(state: State<'_, AppState>, id: String) -> CmdResult<bool> {
    let rt = state.runtime().await?;
    Ok(rt.sup.unload(&id).await)
}

/// Stream one assistant reply. `messages` are OpenAI-style `{role, content}` objects.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn chat(
    state: State<'_, AppState>,
    request_id: String,
    model: String,
    messages: Vec<Value>,
    max_tokens: Option<u32>,
    temperature: Option<f64>,
    on_event: Channel<ChatEvent>,
) -> CmdResult<ChatStats> {
    let rt = state.runtime().await?;
    let (tx, rx) = oneshot::channel::<()>();
    state.chats.lock().unwrap().insert(request_id.clone(), tx);

    let mut body = json!({"model": model, "messages": messages});
    if let Some(m) = max_tokens {
        body["max_tokens"] = json!(m);
    }
    if let Some(t) = temperature {
        body["temperature"] = json!(t);
    }
    let cancelled = async {
        // A dropped sender (request finished) must not look like a cancellation.
        if rx.await.is_err() {
            std::future::pending::<()>().await;
        }
    };
    let result = stream_chat(&rt.http, &rt.base, body, cancelled, |e| {
        let _ = on_event.send(e);
    })
    .await;
    state.chats.lock().unwrap().remove(&request_id);
    result.map_err(|f| CmdError::new(&f.code, f.message))
}

#[tauri::command]
pub fn cancel_chat(state: State<'_, AppState>, request_id: String) -> bool {
    match state.chats.lock().unwrap().remove(&request_id) {
        Some(tx) => tx.send(()).is_ok(),
        None => false,
    }
}
