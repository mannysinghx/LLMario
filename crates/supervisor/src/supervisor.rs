//! The supervisor: model lifecycle (load / LRU evict / crash relaunch) and request admission.

use crate::adapter::{BackendStatus, EngineAdapter, LaunchContext};
use crate::engine::{self, Engine, EngineInfo};
use crate::memory::{self, MemoryPlan};
use crate::planner::{self, Selection};
use llmario_core::{
    BackendKind, Config, ModelFormat, Paths, ProfileKind, RuntimeError, Speculative,
};
use llmario_hardware::HardwareReport;
use llmario_registry::{ModelEntry, Registry};
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::{Notify, OwnedSemaphorePermit};

const CRASH_WINDOW: Duration = Duration::from_secs(600);

pub struct Supervisor {
    pub cfg: Config,
    pub paths: Paths,
    pub hw: HardwareReport,
    pub profile: ProfileKind,
    adapters: HashMap<BackendKind, Arc<dyn EngineAdapter>>,
    statuses: HashMap<BackendKind, BackendStatus>,
    registry: RwLock<Registry>,
    engines: tokio::sync::Mutex<HashMap<String, Arc<Engine>>>,
    load_lock: tokio::sync::Mutex<()>,
    idle: Arc<Notify>,
    crashes: Mutex<HashMap<String, Vec<Instant>>>,
    http: reqwest::Client,
}

/// Admission ticket for one request. Holds a concurrency slot on the engine until dropped
/// (end of response, client disconnect, or error), which is what makes cancellation free.
pub struct Lease {
    pub engine: Arc<Engine>,
    pub selection_reason: String,
    _permit: OwnedSemaphorePermit,
    idle: Arc<Notify>,
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.engine.active.fetch_sub(1, Ordering::SeqCst);
        *self.engine.last_used.lock().unwrap() = Instant::now();
        self.idle.notify_waiters();
    }
}

impl Supervisor {
    pub fn new(
        cfg: Config,
        paths: Paths,
        hw: HardwareReport,
        adapters: Vec<Arc<dyn EngineAdapter>>,
    ) -> anyhow::Result<Arc<Self>> {
        paths.ensure()?;
        let registry = Registry::load(&paths.registry_file())?;
        let mut statuses = HashMap::new();
        let mut map = HashMap::new();
        for a in adapters {
            let st = a.probe(&hw, &cfg);
            tracing::debug!(backend = %a.kind(), available = st.available, detail = %st.detail, "probed backend");
            statuses.insert(a.kind(), st);
            map.insert(a.kind(), a);
        }
        let http = reqwest::Client::builder()
            .user_agent(llmario_core::user_agent())
            .no_proxy()
            .pool_idle_timeout(Duration::from_secs(30))
            .build()?;
        let profile = cfg.runtime.profile;
        let sup = Arc::new(Self {
            cfg,
            paths,
            hw,
            profile,
            adapters: map,
            statuses,
            registry: RwLock::new(registry),
            engines: tokio::sync::Mutex::new(HashMap::new()),
            load_lock: tokio::sync::Mutex::new(()),
            idle: Arc::new(Notify::new()),
            crashes: Mutex::new(HashMap::new()),
            http,
        });
        if sup.cfg.runtime.idle_unload_secs > 0 {
            let weak = Arc::downgrade(&sup);
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    let Some(s) = weak.upgrade() else { break };
                    s.unload_idle(Duration::from_secs(s.cfg.runtime.idle_unload_secs))
                        .await;
                }
            });
        }
        Ok(sup)
    }

    pub fn statuses(&self) -> &HashMap<BackendKind, BackendStatus> {
        &self.statuses
    }

    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }

    pub fn registry(&self) -> Registry {
        self.registry.read().unwrap().clone()
    }

    /// Re-read the manifest so models pulled by another `llmario` process become visible.
    pub fn reload_registry(&self) -> anyhow::Result<()> {
        let r = Registry::load(&self.paths.registry_file())?;
        *self.registry.write().unwrap() = r;
        Ok(())
    }

    pub fn select(&self, name: &str) -> Result<Selection, RuntimeError> {
        let first = planner::select(
            name,
            &self.registry.read().unwrap(),
            &self.statuses,
            &self.hw,
            &self.cfg,
            self.profile,
        );
        match first {
            Err(RuntimeError::ModelNotFound(_)) => {
                self.reload_registry()?;
                planner::select(
                    name,
                    &self.registry.read().unwrap(),
                    &self.statuses,
                    &self.hw,
                    &self.cfg,
                    self.profile,
                )
            }
            other => other,
        }
    }

    /// Memory plan for a selection, accounting for engines that would stay loaded and for
    /// speculative decoding.
    pub fn plan_memory(&self, sel: &Selection, reserved_by_others: u64) -> MemoryPlan {
        let extra = self
            .adapters
            .get(&sel.backend)
            .map(|a| a.extra_memory_bytes(&sel.model, &sel.profile, &self.hw, &self.cfg))
            .unwrap_or(0);
        let (draft, draft_note) = self.draft_for(sel);
        let (spec, spec_note) = memory::speculative_bytes(
            &sel.model,
            draft.as_ref(),
            &sel.profile,
            sel.backend,
            &self.cfg,
        );
        let mut plan = memory::estimate(
            &sel.model,
            &sel.profile,
            sel.backend,
            &self.hw,
            &self.cfg,
            extra + spec,
            reserved_by_others,
        );
        plan.notes.extend(draft_note.into_iter().chain(spec_note));
        plan
    }

    /// The draft model speculative decoding should use with `sel`, and a note when the
    /// configured one cannot be used (the model then runs without speculation). llama.cpp:
    /// `backends.llamacpp.speculative = "draft"` with `draft_model`. MLX: `backends.mlx.draft_model`,
    /// only with one request at a time, because MLX-LM turns off request batching with a draft.
    pub fn draft_for(&self, sel: &Selection) -> (Option<ModelEntry>, Option<String>) {
        let off = "running without speculation";
        let (id, format) = match sel.backend {
            BackendKind::LlamaCpp
                if self.cfg.backends.llamacpp.speculative == Speculative::Draft =>
            {
                match &self.cfg.backends.llamacpp.draft_model {
                    Some(id) => (id, ModelFormat::Gguf),
                    None => {
                        return (
                            None,
                            Some(format!(
                                "speculative = \"draft\" needs backends.llamacpp.draft_model; {off}"
                            )),
                        )
                    }
                }
            }
            BackendKind::Mlx => match &self.cfg.backends.mlx.draft_model {
                Some(id) => (id, ModelFormat::Mlx),
                None => return (None, None),
            },
            _ => return (None, None),
        };
        if *id == sel.model.id {
            return (
                None,
                Some(format!("draft model {id} is the model itself; {off}")),
            );
        }
        let Some(d) = self.registry.read().unwrap().get(id).cloned() else {
            return (
                None,
                Some(format!("draft model {id} is not installed; {off}")),
            );
        };
        if d.format != format {
            return (
                None,
                Some(format!(
                    "draft model {id} is {}, but {} needs a {format} model; {off}",
                    d.format, sel.backend
                )),
            );
        }
        if sel.backend == BackendKind::Mlx && sel.profile.parallel > 1 {
            return (
                None,
                Some(format!(
                    "MLX uses draft model {id} only with one request at a time (latency profile); {off}"
                )),
            );
        }
        let note = (d.architecture != sel.model.architecture).then(|| {
            format!(
                "draft model {id} ({}) differs in architecture from {} ({}); it must share the tokenizer",
                d.architecture.as_deref().unwrap_or("?"),
                sel.model.id,
                sel.model.architecture.as_deref().unwrap_or("?")
            )
        });
        (Some(d), note)
    }

    pub async fn loaded(&self) -> Vec<EngineInfo> {
        self.engines
            .lock()
            .await
            .values()
            .map(|e| e.info())
            .collect()
    }

    /// Admit a request for `name`, loading the model if needed. Waits up to
    /// `queue_timeout_secs` for a free slot or an idle model to evict.
    pub async fn acquire(&self, name: &str) -> Result<Lease, RuntimeError> {
        let sel = self.select(name)?;
        let id = sel.model.id.clone();
        let deadline =
            tokio::time::Instant::now() + Duration::from_secs(self.cfg.runtime.queue_timeout_secs);

        loop {
            let existing = self.engines.lock().await.get(&id).cloned();
            if let Some(engine) = existing {
                if engine.is_alive() {
                    let permit = tokio::time::timeout_at(deadline, engine.permits.clone().acquire_owned())
                        .await
                        .map_err(|_| RuntimeError::Busy(format!(
                            "all {} slot(s) of '{id}' stayed busy for {}s; retry later or use a profile with more parallel slots",
                            engine.profile.parallel, self.cfg.runtime.queue_timeout_secs
                        )))?
                        .map_err(|_| RuntimeError::EngineCrashed("engine closed".into()))?;
                    if !engine.is_alive() {
                        continue; // died while we waited; reload below
                    }
                    engine.active.fetch_add(1, Ordering::SeqCst);
                    *engine.last_used.lock().unwrap() = Instant::now();
                    return Ok(Lease {
                        engine,
                        selection_reason: sel.reason.clone(),
                        _permit: permit,
                        idle: self.idle.clone(),
                    });
                }
                // Crashed: forget it, count the crash, relaunch.
                let info = engine.exit_info().map(|i| i.describe()).unwrap_or_default();
                self.engines.lock().await.remove(&id);
                self.note_crash(&id, &info, &engine.log_path)?;
            }
            self.ensure_loaded(&sel, deadline).await?;
        }
    }

    fn note_crash(&self, id: &str, info: &str, log: &std::path::Path) -> Result<(), RuntimeError> {
        let mut c = self.crashes.lock().unwrap();
        let list = c.entry(id.to_string()).or_default();
        list.retain(|t| t.elapsed() < CRASH_WINDOW);
        list.push(Instant::now());
        if list.len() as u32 > self.cfg.runtime.max_restarts {
            return Err(RuntimeError::EngineCrashed(format!(
                "'{id}' crashed {} times in 10 minutes (last: {info}); not relaunching. See {}",
                list.len(),
                log.display()
            )));
        }
        tracing::warn!(model = id, last_exit = info, "relaunching crashed engine");
        Ok(())
    }

    async fn ensure_loaded(
        &self,
        sel: &Selection,
        deadline: tokio::time::Instant,
    ) -> Result<(), RuntimeError> {
        let _guard = tokio::time::timeout_at(deadline, self.load_lock.lock())
            .await
            .map_err(|_| {
                RuntimeError::Busy("timed out waiting for another model load to finish".into())
            })?;
        let id = &sel.model.id;
        loop {
            let engines = self.engines.lock().await;
            if engines.get(id).is_some_and(|e| e.is_alive()) {
                return Ok(());
            }
            let others: Vec<Arc<Engine>> =
                engines.values().filter(|e| e.is_alive()).cloned().collect();
            drop(engines);

            let reserved: u64 = others.iter().map(|e| e.memory.total_bytes).sum();
            let plan = self.plan_memory(sel, reserved);
            let over_count = others.len() >= self.cfg.runtime.max_loaded_models;
            if !over_count && plan.fits {
                break;
            }
            if others.is_empty() {
                return Err(RuntimeError::InsufficientMemory(
                    plan.refusal(id, &sel.profile),
                ));
            }
            // Evict the least-recently-used idle engine, or wait for one to become idle.
            let victim = others
                .iter()
                .filter(|e| e.active_requests() == 0)
                .max_by_key(|e| e.idle_for())
                .cloned();
            match victim {
                Some(v) => {
                    tracing::info!(evict = %v.model.id, load = %id, "unloading idle model to make room");
                    self.engines.lock().await.remove(&v.model.id);
                    v.stop().await;
                }
                None => {
                    let notified = self.idle.notified();
                    if tokio::time::timeout_at(deadline, notified).await.is_err() {
                        return Err(RuntimeError::Busy(format!(
                            "cannot load '{id}': the loaded model(s) are serving requests and max_loaded_models = {}; retry shortly",
                            self.cfg.runtime.max_loaded_models
                        )));
                    }
                }
            }
        }

        let engine = self.launch(sel, 0).await?;
        self.engines.lock().await.insert(id.clone(), engine);
        Ok(())
    }

    async fn launch(&self, sel: &Selection, reserved: u64) -> Result<Arc<Engine>, RuntimeError> {
        let adapter = self.adapters.get(&sel.backend).ok_or_else(|| {
            RuntimeError::BackendUnavailable(format!("{} adapter not built in", sel.backend))
        })?;
        let status = &self.statuses[&sel.backend];
        let plan = self.plan_memory(sel, reserved);
        let (draft, _) = self.draft_for(sel);
        let timeout = Duration::from_secs(self.cfg.runtime.engine_start_timeout_secs);
        if let Some(other) = crate::sibling::sibling_usage(&self.paths) {
            tracing::warn!(model = %sel.model.id, "{}", other.warning());
        }

        let mut last_err = None;
        for attempt in 1..=2 {
            let port = engine::free_port().map_err(|e| RuntimeError::Other(e.into()))?;
            let ctx = LaunchContext {
                model: &sel.model,
                profile: &sel.profile,
                hw: &self.hw,
                cfg: &self.cfg,
                memory: &plan,
                status,
                port,
                draft: draft.as_ref(),
            };
            let spec = adapter.launch(&ctx)?;
            tracing::info!(
                model = %sel.model.id, backend = %sel.backend, port, attempt,
                parallel = sel.profile.parallel, ctx_per_slot = sel.profile.ctx_per_slot,
                estimate = %memory::fmt_bytes(plan.total_bytes),
                "launching engine"
            );
            let started = Instant::now();
            let engine = Engine::spawn(
                &spec,
                sel.model.clone(),
                sel.backend,
                port,
                sel.profile.clone(),
                plan.clone(),
                &self.paths.logs_dir(),
                &self.paths.run_dir(),
            )?;
            match engine
                .wait_ready(&self.http, &spec.health_path, timeout, started)
                .await
            {
                Ok(()) => {
                    tracing::info!(model = %sel.model.id, secs = format!("{:.2}", engine.ready_after().as_secs_f64()), "engine ready");
                    return Ok(engine);
                }
                Err(e) => {
                    engine.stop().await;
                    // Retry once only when the engine lost the race for its port; any other
                    // startup failure (bad model file, missing flag) is deterministic.
                    let log = engine::tail(&engine.log_path, 20).to_ascii_lowercase();
                    let port_race =
                        log.contains("address already in use") || log.contains("couldn't bind");
                    last_err = Some(e);
                    if !port_race {
                        break;
                    }
                }
            }
        }
        Err(last_err.unwrap_or_else(|| RuntimeError::EngineStart("unknown".into())))
    }

    pub async fn unload(&self, id: &str) -> bool {
        let e = self.engines.lock().await.remove(id);
        match e {
            Some(e) => {
                e.stop().await;
                self.idle.notify_waiters();
                true
            }
            None => false,
        }
    }

    async fn unload_idle(&self, after: Duration) {
        let idle: Vec<String> = self
            .engines
            .lock()
            .await
            .values()
            .filter(|e| e.active_requests() == 0 && e.idle_for() >= after)
            .map(|e| e.model.id.clone())
            .collect();
        for id in idle {
            tracing::info!(model = %id, "unloading idle model");
            self.unload(&id).await;
        }
    }

    /// Stop every engine. Call on shutdown.
    pub async fn shutdown(&self) {
        let all: Vec<Arc<Engine>> = self.engines.lock().await.drain().map(|(_, e)| e).collect();
        for e in all {
            e.stop().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmario_core::ProfileKind;

    fn entry(id: &str, format: ModelFormat, arch: &str) -> ModelEntry {
        let mut e = crate::memory::tests::model(1.0);
        e.id = id.into();
        e.format = format;
        e.architecture = Some(arch.into());
        e
    }

    fn supervisor(cfg: Config, models: Vec<ModelEntry>) -> (tempfile::TempDir, Arc<Supervisor>) {
        let home = tempfile::tempdir().unwrap();
        let paths = Paths::at(home.path());
        paths.ensure().unwrap();
        let mut reg = Registry::load(&paths.registry_file()).unwrap();
        for m in models {
            reg.insert(m).unwrap();
        }
        reg.save().unwrap();
        let hw = crate::memory::tests::hw(16, None);
        let sup = Supervisor::new(cfg, paths, hw, vec![]).unwrap();
        (home, sup)
    }

    fn select(m: &ModelEntry, backend: BackendKind, profile: ProfileKind) -> Selection {
        Selection {
            model: m.clone(),
            backend,
            profile: llmario_core::ResolvedProfile::resolve(profile, None),
            reason: String::new(),
        }
    }

    /// Only a registered draft of the right format is used; anything else runs without
    /// speculation and says why. MLX drafts need one request at a time.
    #[test]
    fn draft_for_checks_the_configured_draft() {
        let main = entry("big", ModelFormat::Gguf, "qwen35");
        let small = entry("small", ModelFormat::Gguf, "qwen35");
        let mlx_main = entry("big-mlx", ModelFormat::Mlx, "qwen3");
        let mlx_small = entry("small-mlx", ModelFormat::Mlx, "qwen3");
        let other = entry("other", ModelFormat::Gguf, "gemma4");
        let models = vec![
            main.clone(),
            small.clone(),
            mlx_main.clone(),
            mlx_small.clone(),
            other.clone(),
        ];
        let sel = select(&main, BackendKind::LlamaCpp, ProfileKind::Latency);

        let (_h, sup) = supervisor(Config::default(), models.clone());
        assert_eq!(sup.draft_for(&sel), (None, None), "off by default");

        let mut cfg = Config::default();
        cfg.backends.llamacpp.speculative = Speculative::Draft;
        let (_h, sup) = supervisor(cfg.clone(), models.clone());
        assert!(sup
            .draft_for(&sel)
            .1
            .unwrap()
            .contains("needs backends.llamacpp.draft_model"));

        for (id, why) in [
            ("missing", "not installed"),
            ("big", "the model itself"),
            ("small-mlx", "needs a gguf model"),
        ] {
            cfg.backends.llamacpp.draft_model = Some(id.into());
            let (_h, sup) = supervisor(cfg.clone(), models.clone());
            let (d, note) = sup.draft_for(&sel);
            assert!(
                d.is_none() && note.as_deref().unwrap_or("").contains(why),
                "{id}: {note:?}"
            );
        }
        cfg.backends.llamacpp.draft_model = Some("small".into());
        let (_h, sup) = supervisor(cfg.clone(), models.clone());
        assert_eq!(sup.draft_for(&sel), (Some(small.clone()), None));
        cfg.backends.llamacpp.draft_model = Some("other".into());
        let (_h, sup) = supervisor(cfg.clone(), models.clone());
        let (d, note) = sup.draft_for(&sel);
        assert!(d.is_some() && note.unwrap().contains("must share the tokenizer"));

        let mut cfg = Config::default();
        cfg.backends.mlx.draft_model = Some("small-mlx".into());
        let (_h, sup) = supervisor(cfg, models);
        let latency = select(&mlx_main, BackendKind::Mlx, ProfileKind::Latency);
        assert_eq!(sup.draft_for(&latency).0, Some(mlx_small));
        let balanced = select(&mlx_main, BackendKind::Mlx, ProfileKind::Balanced);
        let (d, note) = sup.draft_for(&balanced);
        assert!(d.is_none() && note.unwrap().contains("one request at a time"));
        // The plan includes the draft's memory.
        let plan = sup.plan_memory(&latency, 0);
        assert!(
            plan.notes
                .iter()
                .any(|n| n.contains("draft model small-mlx")),
            "{:?}",
            plan.notes
        );
    }
}
