//! MLX-LM adapter: launches `mlx_lm.server` (MIT, https://github.com/ml-explore/mlx-lm)
//! in an isolated child process. Apple Silicon only.
//!
//! Security: `mlx_lm.server` loads whatever model a request names (including downloading it
//! from the Hub). The gateway therefore always sends `model: "default_model"`, and the engine
//! runs with `HF_HUB_OFFLINE=1` so it cannot fetch anything even if a request slipped through.
//! `--trust-remote-code` is never passed.
//!
//! Profile mapping:
//! | profile value        | flag                     |
//! |----------------------|--------------------------|
//! | parallel             | `--decode-concurrency`   |
//! | prompt_concurrency   | `--prompt-concurrency`   |
//! | prompt_cache_entries | `--prompt-cache-size`    |
//! | default_max_tokens   | `--max-tokens`           |
//! | prompt-cache budget  | `--prompt-cache-bytes`   |
//! | buffer-cache ceiling | `mx.set_cache_limit` (launch shim) |
//! MLX grows its KV cache on demand; `ctx_per_slot` is enforced by the gateway instead.

use llmario_core::{BackendKind, Config, ModelFormat, ResolvedProfile, RuntimeError};
use llmario_hardware::HardwareReport;
use llmario_registry::ModelEntry;
use llmario_supervisor::adapter::{which, BackendStatus, EngineAdapter, LaunchContext, LaunchSpec};
use std::path::{Path, PathBuf};
use std::process::Command;

pub const TESTED_MLX_LM: &str = "0.31.3";
/// Ceiling for MLX's allocator buffer cache (freed Metal buffers MLX keeps for reuse). MLX
/// defaults it to the whole memory limit; in our concurrency-4 benchmark the retained cache
/// was 2.5 GiB and counted against the process footprint, so we cap it explicitly.
pub const MLX_BUFFER_CACHE_LIMIT: u64 = 1024 * 1024 * 1024;
/// Python interpreter, tokenizer and HTTP server resident memory (measured ~0.4 GiB).
const PYTHON_RUNTIME_BYTES: u64 = 512 * 1024 * 1024;

/// Upper bound for MLX's prompt (prefix) cache: every entry can hold one request's full KV
/// cache (per-layer layout when known, see `memory::kv_plan`).
/// Measured: 8 entries of ~4.7k-token prompts held ~3.4 GiB beyond the batch KV.
pub fn prompt_cache_bytes(model: &ModelEntry, profile: &ResolvedProfile, cfg: &Config) -> u64 {
    let one_request = ResolvedProfile {
        parallel: 1,
        ..profile.clone()
    };
    let kv = llmario_supervisor::memory::kv_plan(model, &one_request, BackendKind::Mlx, cfg);
    profile.prompt_cache_entries as u64 * kv.total
}

/// Probe: versions on line 1; on line 2, the `model_type`s this mlx-lm can load (its model
/// modules plus the names it remaps onto them), as JSON.
const PROBE: &str = "import json, os, mlx_lm, mlx.core as mx\n\
print(mlx_lm.__version__, mx.__version__)\n\
d = os.path.join(os.path.dirname(mlx_lm.__file__), 'models')\n\
mods = {f[:-3] for f in os.listdir(d) if f.endswith('.py') and not f.startswith('_')}\n\
try:\n    from mlx_lm.utils import MODEL_REMAPPING as r\nexcept Exception:\n    r = {}\n\
print(json.dumps(sorted(mods | {k for k, v in r.items() if v in mods})))";

/// Launch shim: apply `mx.set_cache_limit` (public MLX API) and then run the stock
/// `mlx_lm.server` entry point with the remaining arguments. `sys.argv[1]` is the limit.
const BOOTSTRAP: &str = "import sys, mlx.core as mx; mx.set_cache_limit(int(sys.argv[1])); \
sys.argv = ['mlx_lm.server'] + sys.argv[2:]; from mlx_lm.server import main; main()";

pub struct MlxAdapter {
    /// `$LLMARIO_HOME/venvs/mlx/bin/python` — created by `scripts/setup-mlx-venv.sh`.
    pub managed_venv_python: PathBuf,
}

impl MlxAdapter {
    pub fn new(home: &Path) -> Self {
        Self {
            managed_venv_python: home.join("venvs/mlx/bin/python"),
        }
    }

    /// Interpreter resolution: config → managed venv → interpreter behind `mlx_lm.server`.
    fn find_python(&self, cfg: &Config) -> Option<(PathBuf, &'static str)> {
        if let Some(p) = &cfg.backends.mlx.python {
            return Some((p.clone(), "backends.mlx.python"));
        }
        if self.managed_venv_python.is_file() {
            return Some((self.managed_venv_python.clone(), "managed venv"));
        }
        let script = which("mlx_lm.server")?;
        let first = std::fs::read_to_string(&script)
            .ok()?
            .lines()
            .next()?
            .to_string();
        let interp = first
            .strip_prefix("#!")?
            .split_whitespace()
            .next()?
            .to_string();
        Some((PathBuf::from(interp), "mlx_lm.server on PATH"))
    }
}

impl EngineAdapter for MlxAdapter {
    fn kind(&self) -> BackendKind {
        BackendKind::Mlx
    }
    fn formats(&self) -> &'static [ModelFormat] {
        &[ModelFormat::Mlx]
    }
    fn tested_version(&self) -> &'static str {
        "mlx-lm 0.31.3 / mlx 0.32.2"
    }

    fn probe(&self, hw: &HardwareReport, cfg: &Config) -> BackendStatus {
        let mut st = BackendStatus {
            kind: self.kind(),
            available: false,
            path: None,
            version: None,
            tested_version: self.tested_version().into(),
            detail: String::new(),
            architectures: None,
        };
        if !hw.apple_silicon {
            st.detail = "MLX requires Apple Silicon macOS".into();
            return st;
        }
        let Some((python, source)) = self.find_python(cfg) else {
            st.detail = "mlx-lm not found (run scripts/setup-mlx-venv.sh, or `pip install mlx-lm`, or set backends.mlx.python)".into();
            return st;
        };
        st.path = Some(python.clone());
        let out = Command::new(&python).args(["-c", PROBE]).output();
        match out {
            Ok(o) if o.status.success() => {
                let text = String::from_utf8_lossy(&o.stdout);
                let mut lines = text.lines();
                let mut parts = lines.next().unwrap_or("").split_whitespace();
                let lm = parts.next().unwrap_or("?").to_string();
                let mx = parts.next().unwrap_or("?").to_string();
                st.available = true;
                st.architectures = lines
                    .next()
                    .and_then(|l| serde_json::from_str::<Vec<String>>(l).ok())
                    .filter(|v| !v.is_empty())
                    .map(|v| std::sync::Arc::new(v.into_iter().collect()));
                st.version = Some(format!("mlx-lm {lm} / mlx {mx}"));
                st.detail = if lm == TESTED_MLX_LM {
                    format!("found via {source}")
                } else {
                    format!("found via {source}; mlx-lm {lm} differs from tested {TESTED_MLX_LM}")
                };
            }
            Ok(o) => {
                st.detail = format!(
                    "{} cannot import mlx_lm: {}",
                    python.display(),
                    String::from_utf8_lossy(&o.stderr)
                        .lines()
                        .last()
                        .unwrap_or("")
                )
            }
            Err(e) => st.detail = format!("cannot run {}: {e}", python.display()),
        }
        st
    }

    fn extra_memory_bytes(
        &self,
        model: &ModelEntry,
        profile: &ResolvedProfile,
        cfg: &Config,
    ) -> u64 {
        prompt_cache_bytes(model, profile, cfg) + MLX_BUFFER_CACHE_LIMIT + PYTHON_RUNTIME_BYTES
    }

    fn launch(&self, ctx: &LaunchContext<'_>) -> Result<LaunchSpec, RuntimeError> {
        let python = ctx
            .status
            .path
            .clone()
            .ok_or_else(|| RuntimeError::BackendUnavailable(ctx.status.detail.clone()))?;
        if !ctx.model.chat_template {
            return Err(RuntimeError::Unsupported(format!(
                "'{}' has no chat template; mlx_lm.server cannot serve chat completions for it",
                ctx.model.id
            )));
        }
        let p = ctx.profile;
        let mut args: Vec<String> = vec![
            "-c".into(),
            BOOTSTRAP.into(),
            MLX_BUFFER_CACHE_LIMIT.to_string(),
            "--model".into(),
            ctx.model.path.display().to_string(),
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            ctx.port.to_string(),
            "--log-level".into(),
            "WARNING".into(),
            "--max-tokens".into(),
            p.default_max_tokens.to_string(),
            "--decode-concurrency".into(),
            p.parallel.to_string(),
            "--prompt-concurrency".into(),
            p.prompt_concurrency.to_string(),
            "--prompt-cache-size".into(),
            p.prompt_cache_entries.to_string(),
            "--prompt-cache-bytes".into(),
            prompt_cache_bytes(ctx.model, p, ctx.cfg).to_string(),
        ];
        args.extend(ctx.cfg.backends.mlx.extra_args.iter().cloned());
        Ok(LaunchSpec {
            program: python,
            args,
            env: vec![
                ("HF_HUB_OFFLINE".into(), "1".into()),
                ("TRANSFORMERS_OFFLINE".into(), "1".into()),
                ("HF_HUB_DISABLE_TELEMETRY".into(), "1".into()),
                ("PYTHONUNBUFFERED".into(), "1".into()),
            ],
            health_path: "/health".into(),
            upstream_model: "default_model".into(),
            notes: vec![
                "MLX allocates KV cache on demand; per-request context is enforced by the gateway"
                    .into(),
            ],
        })
    }
}
