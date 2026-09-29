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

use llmario_core::os::background_command;
use llmario_core::{BackendKind, Config, ModelFormat, ResolvedProfile, RuntimeError};
use llmario_hardware::HardwareReport;
use llmario_supervisor::adapter::{which, BackendStatus, EngineAdapter, LaunchContext, LaunchSpec};
use std::path::PathBuf;

pub const TESTED_BUILD: u32 = 11146;
const CACHE_RAM_MIB: u64 = 1024;

/// How to install llama.cpp on this platform (shown when `llama-server` is missing).
pub const INSTALL_HINT: &str = if cfg!(windows) {
    "`winget install ggml.llamacpp`, or unzip a Windows build from github.com/ggml-org/llama.cpp/releases and add it to PATH"
} else if cfg!(target_os = "macos") {
    "`brew install llama.cpp`, or build from github.com/ggml-org/llama.cpp"
} else {
    "build from github.com/ggml-org/llama.cpp, or use your distribution's package"
};

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
            architectures: None,
        };
        let Some(path) = Self::find(cfg) else {
            st.detail = format!(
                "llama-server not found on PATH (install: {INSTALL_HINT}; or set backends.llamacpp.server_path)"
            );
            return st;
        };
        st.path = Some(path.clone());
        match background_command(&path).arg("--version").output() {
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
                        st.architectures = architecture_names(&path).map(std::sync::Arc::new);
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

/// Architecture names compiled into this llama.cpp build. They live in llama.cpp's
/// architecture table as standalone C strings ("qwen3", "gemma4", "gpt-oss", …), in the
/// `llama-server` binary (static builds) or its shared library: `libllama` (Homebrew, Linux) or
/// `llama.dll` next to `llama-server.exe` (Windows release zips and winget).
/// Returns `None` if the table cannot be found, so callers treat support as unknown.
pub fn architecture_names(server: &std::path::Path) -> Option<std::collections::HashSet<String>> {
    let real = std::fs::canonicalize(server).ok()?;
    let mut files = vec![real.clone()];
    if let Some(bin) = real.parent() {
        for dir in [bin.join("../lib"), bin.to_path_buf()] {
            let Ok(rd) = std::fs::read_dir(&dir) else {
                continue;
            };
            for e in rd.flatten() {
                let n = e.file_name().to_string_lossy().into_owned();
                let unix_lib =
                    n.starts_with("libllama.") && (n.contains(".dylib") || n.contains(".so"));
                if unix_lib || n.eq_ignore_ascii_case("llama.dll") {
                    files.push(e.path());
                }
            }
        }
    }
    let mut set = std::collections::HashSet::new();
    for f in files {
        if let Ok(bytes) = std::fs::read(&f) {
            c_strings(&bytes, &mut set);
        }
    }
    // Sanity check: a real architecture table contains these.
    (set.contains("llama") && set.contains("qwen2")).then_some(set)
}

/// Collect short NUL-delimited identifier-like strings (`[a-z0-9_.-]{2,32}`).
fn c_strings(bytes: &[u8], out: &mut std::collections::HashSet<String>) {
    let ok =
        |b: u8| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-' || b == b'.';
    let mut start = 0usize;
    for (i, &b) in bytes.iter().enumerate() {
        if b == 0 {
            let run = &bytes[start..i];
            let standalone = start == 0 || bytes[start - 1] == 0;
            if standalone && (2..=32).contains(&run.len()) && run.iter().all(|&c| ok(c)) {
                out.insert(String::from_utf8_lossy(run).into_owned());
            }
            start = i + 1;
        } else if !ok(b) {
            // Breaks the run. The next run is only "standalone" if it starts right after a NUL,
            // which the check above verifies via `bytes[start - 1]`.
            start = i + 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_standalone_c_strings() {
        let mut set = std::collections::HashSet::new();
        c_strings(
            b"\0qwen3\0gemma4\0gpt-oss\0Hello World\0x\0nemotron_h_moe\0",
            &mut set,
        );
        for a in ["qwen3", "gemma4", "gpt-oss", "nemotron_h_moe"] {
            assert!(set.contains(a), "{a}");
        }
        assert!(!set.contains("x") && !set.contains("hello"));
    }

    /// Runs only where llama.cpp is installed (e.g. `brew install llama.cpp`).
    #[test]
    fn local_build_lists_known_architectures() {
        let Some(server) = llmario_supervisor::adapter::which("llama-server") else {
            return;
        };
        let set = architecture_names(&server).expect("architecture table found");
        for a in ["llama", "qwen2", "qwen3", "gemma3", "phi3"] {
            assert!(set.contains(a), "{a} missing from {} names", set.len());
        }
    }

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
