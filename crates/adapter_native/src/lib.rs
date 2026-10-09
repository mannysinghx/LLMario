//! Adapter for LLMario's own engine, `llmario-engine` (IPC contract v1: OpenAI-compatible HTTP
//! on a loopback port, `GET /health` for readiness).
//!
//! The adapter is a pure description: where the binary is, which architectures it reports
//! (`llmario-engine probe --json`), and the command line for a model and profile. Process
//! management is the supervisor's. The engine is opt-in (`[backends.native] enabled = true`)
//! until its milestones graduate.

use llmario_core::{BackendKind, Config, ModelFormat, RuntimeError};
use llmario_hardware::HardwareReport;
use llmario_supervisor::adapter::{which, BackendStatus, EngineAdapter, LaunchContext, LaunchSpec};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

pub const ENGINE_BIN: &str = "llmario-engine";

pub struct NativeAdapter;

impl NativeAdapter {
    /// Locate the engine: config, then next to the running executable, then `PATH`.
    pub fn locate(cfg: &Config) -> Option<PathBuf> {
        if let Some(p) = &cfg.backends.native.engine_path {
            return p.is_file().then(|| p.clone());
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                let name = if cfg!(windows) {
                    format!("{ENGINE_BIN}.exe")
                } else {
                    ENGINE_BIN.to_string()
                };
                let p = dir.join(name);
                if p.is_file() {
                    return Some(p);
                }
            }
        }
        which(ENGINE_BIN)
    }
}

/// `llmario-engine probe --json` output.
#[derive(serde::Deserialize)]
struct ProbeJson {
    version: String,
    architectures: Vec<String>,
    #[serde(default)]
    kernels: String,
}

impl EngineAdapter for NativeAdapter {
    fn kind(&self) -> BackendKind {
        BackendKind::Native
    }
    fn formats(&self) -> &'static [ModelFormat] {
        &[ModelFormat::Gguf]
    }
    fn tested_version(&self) -> &'static str {
        llmario_core::VERSION
    }
    fn probe(&self, _hw: &HardwareReport, cfg: &Config) -> BackendStatus {
        let unavailable = |detail: String| BackendStatus {
            kind: BackendKind::Native,
            available: false,
            path: None,
            version: None,
            tested_version: llmario_core::VERSION.into(),
            detail,
            architectures: None,
        };
        if cfg!(windows) {
            return unavailable(
                "LLMario's own engine runs on macOS and Linux; on Windows models use llama.cpp"
                    .into(),
            );
        }
        if !cfg.backends.native.enabled {
            return unavailable("disabled ([backends.native] enabled = false)".into());
        }
        let Some(path) = Self::locate(cfg) else {
            return unavailable(format!("{ENGINE_BIN} not found next to llmario or on PATH"));
        };
        let out = llmario_core::os::background_command(&path)
            .args(["probe", "--json"])
            .output();
        match out {
            Ok(o) if o.status.success() => match serde_json::from_slice::<ProbeJson>(&o.stdout) {
                Ok(p) => BackendStatus {
                    kind: BackendKind::Native,
                    available: true,
                    path: Some(path),
                    version: Some(p.version.clone()),
                    tested_version: llmario_core::VERSION.into(),
                    detail: format!(
                        "llmario-engine {} · kernels {} · {} architectures",
                        p.version,
                        p.kernels,
                        p.architectures.len()
                    ),
                    architectures: Some(Arc::new(
                        p.architectures.into_iter().collect::<HashSet<_>>(),
                    )),
                },
                Err(e) => unavailable(format!("{} probe output unreadable: {e}", path.display())),
            },
            Ok(o) => unavailable(format!(
                "{} probe failed: {}",
                path.display(),
                String::from_utf8_lossy(&o.stderr).trim()
            )),
            Err(e) => unavailable(format!("cannot run {}: {e}", path.display())),
        }
    }
    fn launch(&self, ctx: &LaunchContext<'_>) -> Result<LaunchSpec, RuntimeError> {
        let program =
            ctx.status.path.clone().ok_or_else(|| {
                RuntimeError::BackendUnavailable("native engine not probed".into())
            })?;
        let mut args = vec![
            "serve".to_string(),
            "--model".into(),
            ctx.model.path.display().to_string(),
            "--listen".into(),
            format!("127.0.0.1:{}", ctx.port),
            "--ctx".into(),
            ctx.profile.ctx_per_slot.to_string(),
            "--parallel".into(),
            ctx.profile.parallel.to_string(),
            "--batch".into(),
            ctx.profile.ubatch.to_string(),
            "--model-id".into(),
            ctx.model.id.clone(),
        ];
        let threads = ctx
            .cfg
            .backends
            .native
            .threads
            .map(|t| t as usize)
            .or(ctx.hw.performance_cores)
            .unwrap_or(ctx.hw.physical_cores.max(1));
        args.extend(["--threads".into(), threads.to_string()]);
        args.extend(["--memory-limit".into(), ctx.memory.budget_bytes.to_string()]);
        if ctx.cfg.backends.native.web_access {
            args.push("--web".into());
            if let Some(u) = &ctx.cfg.backends.native.searxng_url {
                args.extend(["--searxng-url".into(), u.clone()]);
            }
        }
        args.extend(ctx.cfg.backends.native.extra_args.iter().cloned());
        let mut notes = vec![];
        if ctx.profile.parallel > 1 {
            notes.push(format!(
                "native engine M1 serves requests one at a time (profile asked for {} slots)",
                ctx.profile.parallel
            ));
        }
        if ctx.draft.is_some() {
            notes.push("native engine M1 has no speculative decoding; draft model ignored".into());
        }
        Ok(LaunchSpec {
            program,
            args,
            env: vec![],
            health_path: "/health".into(),
            upstream_model: ctx.model.id.clone(),
            notes,
        })
    }
}
