//! Backend adapter contract (IPC contract v1).
//!
//! An adapter is a pure description of one inference engine: how to detect it, which model
//! formats it accepts, and the exact command line that serves a model with a given profile.
//! The launched process must expose, on `127.0.0.1:<port>`:
//! - `GET <health_path>` → 200 once the model is loaded;
//! - `POST /v1/chat/completions` → OpenAI-compatible, with SSE streaming and
//!   `stream_options.include_usage` support.
//!
//! Process management is shared code in [`crate::engine`].

use crate::memory::MemoryPlan;
use llmario_core::{BackendKind, Config, ModelFormat, ResolvedProfile, RuntimeError};
use llmario_hardware::HardwareReport;
use llmario_registry::ModelEntry;
use serde::Serialize;
use std::path::PathBuf;

pub const CONTRACT_VERSION: u32 = 1;

#[derive(Serialize, Clone, Debug)]
pub struct BackendStatus {
    pub kind: BackendKind,
    pub available: bool,
    /// Executable / interpreter that will be launched.
    pub path: Option<PathBuf>,
    pub version: Option<String>,
    /// Version this adapter was tested against.
    pub tested_version: String,
    /// Human explanation (why unavailable, or what was found).
    pub detail: String,
    /// Architecture names this engine build can load (llama.cpp: GGUF `general.architecture`;
    /// MLX-LM: `model_type`). `None` when they could not be determined.
    #[serde(skip)]
    pub architectures: Option<std::sync::Arc<std::collections::HashSet<String>>>,
}

impl BackendStatus {
    /// `Some(false)` only when we know the engine cannot load this architecture.
    pub fn supports_architecture(&self, arch: &str) -> Option<bool> {
        self.architectures.as_ref().map(|set| set.contains(arch))
    }
}

pub struct LaunchContext<'a> {
    pub model: &'a ModelEntry,
    pub profile: &'a ResolvedProfile,
    pub hw: &'a HardwareReport,
    pub cfg: &'a Config,
    pub memory: &'a MemoryPlan,
    pub status: &'a BackendStatus,
    pub port: u16,
    /// Draft model for speculative decoding, already checked by `Supervisor::draft_for`.
    pub draft: Option<&'a ModelEntry>,
}

#[derive(Clone, Debug)]
pub struct LaunchSpec {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    pub health_path: String,
    /// Value the gateway must put in the upstream request's `model` field.
    pub upstream_model: String,
    /// Profile values this backend cannot honour, or other caveats; shown to the user.
    pub notes: Vec<String>,
}

pub trait EngineAdapter: Send + Sync {
    fn kind(&self) -> BackendKind;
    fn contract_version(&self) -> u32 {
        CONTRACT_VERSION
    }
    fn formats(&self) -> &'static [ModelFormat];
    fn tested_version(&self) -> &'static str;
    /// Detect the backend. May run short subprocesses; called once at startup.
    fn probe(&self, hw: &HardwareReport, cfg: &Config) -> BackendStatus;
    /// Memory the engine uses beyond weights + KV (e.g. host-side prompt caches).
    fn extra_memory_bytes(
        &self,
        _model: &ModelEntry,
        _profile: &ResolvedProfile,
        _hw: &HardwareReport,
        _cfg: &Config,
    ) -> u64 {
        0
    }
    fn launch(&self, ctx: &LaunchContext<'_>) -> Result<LaunchSpec, RuntimeError>;
}

/// Find an executable on `PATH`. On Windows `name` may omit the extension: `llama-server`
/// finds `llama-server.exe` (any `PATHEXT` extension).
pub fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).find_map(|d| executable_in(&d, name))
}

#[cfg(unix)]
fn executable_in(dir: &std::path::Path, name: &str) -> Option<PathBuf> {
    use std::os::unix::fs::PermissionsExt;
    let p = dir.join(name);
    std::fs::metadata(&p)
        .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
        .then_some(p)
}

#[cfg(windows)]
fn executable_in(dir: &std::path::Path, name: &str) -> Option<PathBuf> {
    let exts = std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".into());
    let exts: Vec<String> = exts
        .split(';')
        .filter(|e| !e.is_empty())
        .map(|e| e.to_ascii_lowercase())
        .collect();
    let lower = name.to_ascii_lowercase();
    if exts.iter().any(|e| lower.ends_with(e.as_str())) {
        let p = dir.join(name);
        return p.is_file().then_some(p);
    }
    exts.iter()
        .map(|e| dir.join(format!("{name}{e}")))
        .find(|p| p.is_file())
}
