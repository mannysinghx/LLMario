//! llama.cpp adapter: launches `llama-server` (MIT, https://github.com/ggml-org/llama.cpp).
//!
//! Profile mapping (only flags the tested build supports):
//! | profile value        | flag                  |
//! |----------------------|-----------------------|
//! | parallel             | `-np`                 |
//! | ctx_per_slot×parallel| `-c`                  |
//! | batch / ubatch       | `-b` / `-ub`          |
//! | memory plan          | `-ngl` (999 = all layers when the model fits) |
//! | prefix reuse         | `--cache-reuse 256`   |
//! Host-side prompt cache is capped with `--cache-ram` so it is inside the memory estimate.

use llmario_core::{BackendKind, Config, ModelFormat, ResolvedProfile, RuntimeError};
use llmario_hardware::HardwareReport;
use llmario_supervisor::adapter::{which, BackendStatus, EngineAdapter, LaunchContext, LaunchSpec};
use std::path::PathBuf;
use std::process::Command;

pub const TESTED_BUILD: u32 = 11146;
const CACHE_RAM_MIB: u64 = 1024;

pub struct LlamaCppAdapter;

impl LlamaCppAdapter {
    fn find(cfg: &Config) -> Option<PathBuf> {
        cfg.backends
            .llamacpp
            .server_path
            .clone()
            .or_else(|| which("llama-server"))
    }
}

/// Parse `version: 0.5.0 (build 11146, commit 7fe450e19)` or older `version: 4523 (abc)`.
pub fn parse_version(text: &str) -> Option<(String, Option<u32>)> {
    let line = text
        .lines()
        .find(|l| l.trim_start().starts_with("version:"))?;
    let v = line
        .trim_start()
        .trim_start_matches("version:")
        .trim()
        .to_string();
    let build = v
        .split("build ")
        .nth(1)
        .and_then(|s| s.split(|c: char| !c.is_ascii_digit()).next())
        .and_then(|s| s.parse().ok())
        .or_else(|| v.split_whitespace().next().and_then(|s| s.parse().ok()));
    Some((v, build))
}

impl EngineAdapter for LlamaCppAdapter {
    fn kind(&self) -> BackendKind {
        BackendKind::LlamaCpp
    }
    fn formats(&self) -> &'static [ModelFormat] {
        &[ModelFormat::Gguf]
    }
    fn tested_version(&self) -> &'static str {
        "build 11146 (commit 7fe450e19)"
    }

    fn probe(&self, _hw: &HardwareReport, cfg: &Config) -> BackendStatus {
        let mut st = BackendStatus {
            kind: self.kind(),
            available: false,
            path: None,
            version: None,
            tested_version: self.tested_version().into(),
            detail: String::new(),
        };
        let Some(path) = Self::find(cfg) else {
            st.detail = "llama-server not found on PATH (install: `brew install llama.cpp`, or build from github.com/ggml-org/llama.cpp; or set backends.llamacpp.server_path)".into();
            return st;
        };
        st.path = Some(path.clone());
        match Command::new(&path).arg("--version").output() {
            Ok(out) => {
                let text = format!(
                    "{}{}",
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr)
                );
                match parse_version(&text) {
                    Some((v, build)) => {
                        st.available = true;
                        st.detail = match build {
                            Some(b) if b < TESTED_BUILD.saturating_sub(2000) => {
                                format!("found build {b}; older than the tested build {TESTED_BUILD}, some flags may be missing")
                            }
                            Some(b) => format!("found build {b}"),
                            None => "found (build number unknown)".into(),
                        };
                        st.version = Some(v);
                    }
                    None => st.detail = "llama-server --version produced no version line".into(),
                }
            }
            Err(e) => st.detail = format!("cannot run {}: {e}", path.display()),
        }
        st
    }

    fn extra_memory_bytes(&self, _m: &llmario_registry::ModelEntry, _p: &ResolvedProfile) -> u64 {
        CACHE_RAM_MIB * 1024 * 1024
    }

    fn launch(&self, ctx: &LaunchContext<'_>) -> Result<LaunchSpec, RuntimeError> {
        let program = ctx
            .status
            .path
            .clone()
            .ok_or_else(|| RuntimeError::BackendUnavailable(ctx.status.detail.clone()))?;
        let p = ctx.profile;
        let lcfg = &ctx.cfg.backends.llamacpp;
        let gpu_layers: i64 = match (lcfg.gpu_layers, ctx.memory.gpu_layers) {
            (Some(n), _) => n as i64,
            (None, Some(n)) => n as i64,
            (None, None) => 999,
        };
        let mut notes = Vec::new();
        if !ctx.model.chat_template {
            notes.push(
                "model has no embedded chat template; llama-server will fall back to its default"
                    .into(),
            );
        }
        let mut args: Vec<String> = vec![
            "--model".into(),
            ctx.model.path.display().to_string(),
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            ctx.port.to_string(),
            "--alias".into(),
            ctx.model.id.clone(),
            "--ctx-size".into(),
            p.total_ctx().to_string(),
            "--parallel".into(),
            p.parallel.to_string(),
            "--batch-size".into(),
            p.batch.to_string(),
            "--ubatch-size".into(),
            p.ubatch.to_string(),
            "--n-gpu-layers".into(),
            gpu_layers.to_string(),
            "--flash-attn".into(),
            "auto".into(),
            "--jinja".into(),
            "--cache-reuse".into(),
            "256".into(),
            "--cache-ram".into(),
            CACHE_RAM_MIB.to_string(),
            "--threads".into(),
            ctx.hw.recommended_threads().to_string(),
            "--no-webui".into(),
        ];
        args.extend(lcfg.extra_args.iter().cloned());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_versions() {
        let (v, b) = parse_version("0.00.000.260 I srv init\nversion: 0.5.0 (build 11146, commit 7fe450e19)\nbuilt with clang").unwrap();
        assert_eq!(b, Some(11146));
        assert!(v.starts_with("0.5.0"));
        assert_eq!(
            parse_version("version: 4523 (6152129d)").unwrap().1,
            Some(4523)
        );
        assert!(parse_version("garbage").is_none());
    }
}
