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
//! Context checkpoints (copies of sliding-window caches and recurrent state, kept so a prompt
//! prefix can be reused) are capped with `--ctx-checkpoints` and counted too.

use llmario_core::os::background_command;
use llmario_core::{BackendKind, Config, ModelFormat, ResolvedProfile, RuntimeError};
use llmario_hardware::HardwareReport;
use llmario_supervisor::adapter::{which, BackendStatus, EngineAdapter, LaunchContext, LaunchSpec};
use std::path::PathBuf;

pub const TESTED_BUILD: u32 = 11146;
const CACHE_RAM_MIB: u64 = 1024;
/// Context checkpoints per slot. llama.cpp keeps up to 32 by default; each holds a copy of the
/// sliding-window caches and recurrent state. Measured on Gemma 4 12B (8k context, a long prompt
/// and two follow-up turns): default peak 9.81 GiB (+2.54 GiB after load), 2 checkpoints
/// 7.94 GiB (+0.66 GiB), and the follow-up turns reused the prompt equally well (17 and 14 new
/// tokens processed in both cases).
const CTX_CHECKPOINTS: u64 = 2;

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

    fn extra_memory_bytes(
        &self,
        m: &llmario_registry::ModelEntry,
        p: &ResolvedProfile,
        cfg: &Config,
    ) -> u64 {
        // Checkpoints copy what does not grow with the context: sliding-window caches at their
        // cap and recurrent state, for every slot.
        let bounded = llmario_supervisor::memory::kv_plan(m, p, BackendKind::LlamaCpp, cfg).bounded;
        CACHE_RAM_MIB * 1024 * 1024 + CTX_CHECKPOINTS * bounded
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
            "--ctx-checkpoints".into(),
            CTX_CHECKPOINTS.to_string(),
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

/// Collect short NUL-delimited identifier-like strings (`[a-z0-9_.-]{2,32}`) and their endings.
///
/// Endings count because linkers may store a string as the tail of a longer one that ends the
/// same way (tail merging): llama.cpp's Windows build keeps "qwen2" only inside "rwkv6qwen2".
fn c_strings(bytes: &[u8], out: &mut std::collections::HashSet<String>) {
    let ok =
        |b: u8| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-' || b == b'.';
    let mut start = 0usize;
    for (i, &b) in bytes.iter().enumerate() {
        if b == 0 {
            let run = &bytes[start..i];
            let standalone = start == 0 || bytes[start - 1] == 0;
            if standalone && (2..=32).contains(&run.len()) && run.iter().all(|&c| ok(c)) {
                for k in 0..=run.len() - 2 {
                    out.insert(String::from_utf8_lossy(&run[k..]).into_owned());
                }
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

    #[test]
    fn finds_tail_merged_strings() {
        // lld-link stores "qwen2" as the last bytes of "rwkv6qwen2" (llama.cpp Windows build).
        let mut set = std::collections::HashSet::new();
        c_strings(b"\0rwkv6qwen2\0llama\0\xb8mistral3\0", &mut set);
        for a in ["rwkv6qwen2", "qwen2", "llama"] {
            assert!(set.contains(a), "{a}");
        }
        // Not after a NUL (here: machine code), so not a string.
        assert!(!set.contains("mistral3"));
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
    fn checkpoints_are_capped_and_counted() {
        use llmario_registry::{FileRecord, KvGroup, ModelEntry, ModelShape};
        let group = |layers, n_kv_heads, head_dim, window| KvGroup {
            layers,
            n_kv_heads,
            head_dim,
            window,
        };
        // Gemma 4 12B layout: the sliding-window cache at its cap is 480 MiB (1536 cells).
        let m = ModelEntry {
            id: "g".into(),
            family: None,
            format: ModelFormat::Gguf,
            path: "/g.gguf".into(),
            managed: false,
            source: None,
            license: None,
            architecture: None,
            quantization: None,
            shape: Some(ModelShape {
                n_layers: 48,
                n_heads: 16,
                n_kv_heads: 8,
                head_dim: 512,
                hidden_size: 3840,
                context_max: None,
                kv_groups: vec![group(8, 1, 512, None), group(40, 8, 256, Some(1024))],
                state_bytes_per_seq: 0,
            }),
            chat_template: true,
            files: vec![FileRecord {
                name: "g.gguf".into(),
                size: 1,
                sha256: None,
            }],
            size_bytes: 1,
            added_at: String::new(),
        };
        let p = ResolvedProfile::resolve(llmario_core::ProfileKind::Latency, None);
        let mib = 1024 * 1024;
        let a = LlamaCppAdapter;
        assert_eq!(
            a.extra_memory_bytes(&m, &p, &Config::default()),
            (1024 + 2 * 480) * mib
        );
        // Plain models have nothing to checkpoint.
        let mut plain = m.clone();
        plain.shape.as_mut().unwrap().kv_groups.clear();
        assert_eq!(
            a.extra_memory_bytes(&plain, &p, &Config::default()),
            1024 * mib
        );
    }

    /// Phase 2 exit check from the catalog alone (before download): on a 16 GB Mac (GPU working
    /// set 2/3 of RAM = 10.67 GiB) the per-layer layouts make Qwen3.5 9B and Gemma 4 12B fit;
    /// gpt-oss-20b's weights alone (11.28 GiB) are over budget either way.
    #[test]
    fn catalog_models_on_a_16_gb_mac() {
        use llmario_hardware::{GpuApi, GpuInfo};
        const GIB: u64 = 1 << 30;
        let hw = HardwareReport {
            os: "macos".into(),
            os_version: "test".into(),
            arch: "aarch64".into(),
            cpu_brand: "Apple M4".into(),
            physical_cores: 10,
            logical_cores: 10,
            performance_cores: None,
            efficiency_cores: None,
            cpu_features: vec![],
            total_memory_bytes: 16 * GIB,
            available_memory_bytes: 8 * GIB,
            apple_silicon: true,
            unified_memory: true,
            gpus: vec![GpuInfo {
                vendor: "apple".into(),
                name: "Apple M4".into(),
                api: GpuApi::Metal,
                memory_total_bytes: Some(16 * GIB * 2 / 3),
                memory_free_bytes: None,
                driver: None,
                cores: None,
            }],
            notes: vec![],
        };
        let catalog = llmario_registry::Catalog::builtin();
        let plan = |id: &str, cfg: &Config| {
            let m = catalog.get(id).unwrap().planning_entry();
            let p = ResolvedProfile::resolve(llmario_core::ProfileKind::Latency, None);
            let extra = LlamaCppAdapter.extra_memory_bytes(&m, &p, cfg);
            llmario_supervisor::memory::estimate(&m, &p, BackendKind::LlamaCpp, &hw, cfg, extra, 0)
        };
        let now = Config::default();
        let mut old = Config::default();
        old.runtime.kv_accounting = llmario_core::KvAccounting::Conservative;
        for id in ["qwen3.5-9b-gguf-q4km", "gemma-4-12b-gguf-q4_0"] {
            let p = plan(id, &now);
            assert!(p.fits, "{id}: {}", p.explain());
            assert!(
                p.notes.iter().any(|n| n.contains("KV cache per layer")),
                "{id}"
            );
        }
        assert!(
            !plan("gemma-4-12b-gguf-q4_0", &old).fits,
            "the old formula refused Gemma 4 12B"
        );
        assert!(!plan("gpt-oss-20b-gguf-mxfp4", &now).fits);
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
