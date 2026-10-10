//! `llmario-engine`: the native engine process.
//!
//! Subcommands grow with the milestones: `inspect` and `raw-run` (M1 development), then `plan`,
//! `serve`, `probe` and `bench`.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use llmario_engine_formats::GgufFile;
use llmario_engine_model::{ArchSpec, CpuBackend, CpuOptions, KvType, ModelBackend};
use llmario_engine_server::Device;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "llmario-engine", version, about)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print a model's metadata, architecture description and tensor type summary.
    Inspect {
        model: PathBuf,
        /// Also list every tensor.
        #[arg(long)]
        tensors: bool,
    },
    /// Serve a model over HTTP (IPC contract v1) for the LLMario supervisor.
    Serve {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "127.0.0.1:0")]
        listen: String,
        #[arg(long, default_value_t = 8192)]
        ctx: u32,
        #[arg(long, default_value_t = 1)]
        parallel: u32,
        #[arg(long, default_value_t = 512)]
        batch: u32,
        #[arg(long)]
        threads: Option<usize>,
        #[arg(long)]
        model_id: Option<String>,
        /// Memory ceiling in bytes for the plan (default: physical memory).
        #[arg(long)]
        memory_limit: Option<u64>,
        /// Backend: `auto` (Metal when a usable GPU is present, else CPU), `cpu` or `metal`.
        #[arg(long, default_value = "auto")]
        device: Device,
        /// Allow the built-in `web_fetch` tool (SSRF-guarded); off by default.
        #[arg(long)]
        web: bool,
        /// Self-hosted SearXNG instance for the built-in `web_search` tool (needs `--web`).
        #[arg(long)]
        searxng_url: Option<String>,
        /// KV cache element type: `auto` (q8_0 when the memory ceiling is 16 GiB or less, else
        /// f16, and the planner may switch to q8_0 before shortening the context), `f16` or
        /// `q8_0`.
        #[arg(long, default_value = "auto", value_parser = parse_kv_type)]
        kv_type: KvArg,
        /// Directory for the KV disk tier: conversations evicted from memory are saved there
        /// (owner-only files) and read back when they continue. Off when not given.
        #[arg(long)]
        kv_cache_dir: Option<PathBuf>,
        /// Disk budget of `--kv-cache-dir` in GiB (least recently used files leave first;
        /// 0 = off).
        #[arg(long, default_value_t = 8.0)]
        kv_cache_gb: f64,
        /// Fewest tokens worth saving or restoring.
        #[arg(long, default_value_t = llmario_engine_server::KV_CACHE_MIN_TOKENS)]
        kv_cache_min_tokens: usize,
    },
    /// Report what this build can run (`--json` for the supervisor).
    Probe {
        #[arg(long)]
        json: bool,
    },
    /// Compute the memory plan for a model without loading it.
    Plan {
        model: PathBuf,
        #[arg(long, default_value_t = 8192)]
        ctx: u32,
        #[arg(long, default_value_t = 1)]
        parallel: u32,
        #[arg(long, default_value_t = 512)]
        batch: u32,
        #[arg(long)]
        memory_limit: Option<u64>,
        #[arg(long)]
        json: bool,
        /// KV cache element type (`auto`, `f16`, `q8_0`).
        #[arg(long, default_value = "auto", value_parser = parse_kv_type)]
        kv_type: KvArg,
    },
    /// Tokenize text with the model's tokenizer (development).
    Tokenize {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        text: String,
    },
    /// Development: run greedy generation from raw token ids (no tokenizer, no template).
    RawRun {
        #[arg(long)]
        model: PathBuf,
        /// Comma-separated prompt token ids.
        #[arg(long)]
        tokens: String,
        /// Tokens to generate.
        #[arg(long, default_value_t = 16)]
        n: usize,
        #[arg(long)]
        threads: Option<usize>,
        /// Context size to reserve.
        #[arg(long, default_value_t = 2048)]
        ctx: usize,
        /// Backend: `auto`, `cpu` or `metal`.
        #[arg(long, default_value = "auto")]
        device: Device,
        /// KV cache element type (`f16` or `q8_0`; `auto` means f16 here).
        #[arg(long, default_value = "f16", value_parser = parse_kv_type)]
        kv_type: KvArg,
    },
}

/// `--kv-type` value: `None` = automatic.
#[derive(Clone, Copy, Debug)]
struct KvArg(Option<KvType>);

fn parse_kv_type(s: &str) -> std::result::Result<KvArg, String> {
    if s.eq_ignore_ascii_case("auto") {
        return Ok(KvArg(None));
    }
    KvType::parse(s)
        .map(|t| KvArg(Some(t)))
        .ok_or_else(|| format!("unknown KV cache type `{s}` (auto|f16|q8_0)"))
}

/// Architectures the CPU forward pass covers in this build (M1 dense families + the Qwen3.5 hybrid).
const ARCHITECTURES: &[&str] = &[
    "llama",
    "mistral3",
    "qwen2",
    "qwen3",
    "smollm3",
    "qwen35",
    "qwen3next",
    "gemma4",
    "qwen3moe",
];

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("LLMARIO_ENGINE_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Inspect { model, tensors } => inspect(&model, tensors),
        Cmd::Probe { json } => probe(json),
        Cmd::Plan {
            model,
            ctx,
            parallel,
            batch,
            memory_limit,
            json,
            kv_type,
        } => {
            let f = GgufFile::open(&model)?;
            let opts = llmario_engine_server::ServeOptions {
                model: model.clone(),
                model_id: model
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                listen: String::new(),
                ctx,
                parallel,
                batch,
                threads: 0,
                memory_limit,
                device: Device::Auto,
                web: false,
                searxng_url: None,
                kv_type: kv_type.0,
                kv_cache_dir: None,
                kv_cache_bytes: 0,
                kv_cache_min_tokens: llmario_engine_server::KV_CACHE_MIN_TOKENS,
            };
            let (plan, _) = llmario_engine_server::plan_for(&f, &opts)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&plan)?);
            } else {
                print!("{}", llmario_engine_plan::render(&plan));
            }
            if !plan.fits {
                std::process::exit(2);
            }
            Ok(())
        }
        Cmd::Tokenize { model, text } => {
            let f = GgufFile::open(&model)?;
            let tok = llmario_engine_tokenizer::Tokenizer::from_gguf(&f)?;
            let ids = tok.encode(&text, tok.add_bos(), true);
            println!(
                "{}",
                ids.iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );
            println!("{:?}", tok.decode(&ids));
            Ok(())
        }
        Cmd::Serve {
            model,
            listen,
            ctx,
            parallel,
            batch,
            threads,
            model_id,
            memory_limit,
            device,
            web,
            searxng_url,
            kv_type,
            kv_cache_dir,
            kv_cache_gb,
            kv_cache_min_tokens,
        } => {
            anyhow::ensure!(
                kv_cache_gb.is_finite() && kv_cache_gb >= 0.0,
                "--kv-cache-gb must be a number of GiB, 0 or more"
            );
            let threads = threads.unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            });
            let opts = llmario_engine_server::ServeOptions {
                model_id: model_id.unwrap_or_else(|| {
                    model
                        .file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default()
                }),
                model,
                listen,
                ctx,
                parallel,
                batch,
                threads,
                memory_limit,
                device,
                web,
                searxng_url,
                kv_type: kv_type.0,
                kv_cache_dir,
                kv_cache_bytes: (kv_cache_gb * (1u64 << 30) as f64) as u64,
                kv_cache_min_tokens,
            };
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()?;
            rt.block_on(llmario_engine_server::serve(opts))
        }
        Cmd::RawRun {
            model,
            tokens,
            n,
            threads,
            ctx,
            device,
            kv_type,
        } => raw_run(
            &model,
            &tokens,
            n,
            threads,
            ctx,
            device,
            kv_type.0.unwrap_or_default(),
        ),
    }
}

fn probe(json: bool) -> Result<()> {
    let kernels = llmario_engine_cpu::simd::kernels().name;
    let metal = metal_info();
    if json {
        let metal_json = metal.as_ref().map(|m| {
            serde_json::json!({
                "name": m.name,
                "family": m.family,
                "unified_memory": m.has_unified_memory,
                "recommended_max_working_set": m.recommended_max_working_set,
                "residency_sets": m.residency_sets,
            })
        });
        println!(
            "{}",
            serde_json::json!({
                "version": env!("CARGO_PKG_VERSION"),
                "architectures": ARCHITECTURES,
                "kernels": kernels,
                "devices": if metal.is_some() { vec!["cpu", "metal"] } else { vec!["cpu"] },
                "metal": metal_json,
                "formats": ["gguf"],
                "threads_available": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
            })
        );
    } else {
        println!("llmario-engine {}", env!("CARGO_PKG_VERSION"));
        println!("kernels: {kernels}");
        println!("architectures: {}", ARCHITECTURES.join(", "));
        match metal {
            Some(m) => println!(
                "metal: {} ({}), unified memory {}, recommendedMaxWorkingSetSize {:.1} GiB, residency sets {}",
                m.name,
                m.family,
                m.has_unified_memory,
                m.recommended_max_working_set as f64 / 1073741824.0,
                m.residency_sets
            ),
            None => println!("metal: not available"),
        }
    }
    Ok(())
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_info() -> Option<llmario_engine_metal::DeviceInfo> {
    llmario_engine_metal::MetalBackend::device_info()
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn metal_info() -> Option<llmario_engine_metal_stub::DeviceInfo> {
    None
}

/// Stand-in so `probe` has one `DeviceInfo` shape on every target.
#[cfg(not(all(feature = "metal", target_os = "macos")))]
mod llmario_engine_metal_stub {
    pub struct DeviceInfo {
        pub name: String,
        pub family: String,
        pub has_unified_memory: bool,
        pub recommended_max_working_set: u64,
        pub residency_sets: bool,
    }
}

fn inspect(path: &std::path::Path, list_tensors: bool) -> Result<()> {
    let f = GgufFile::open(path).with_context(|| format!("open {}", path.display()))?;
    println!("file: {}", path.display());
    println!(
        "gguf version {}, alignment {}, {} tensors, {} metadata keys, {} part(s)",
        f.version,
        f.alignment,
        f.tensors.len(),
        f.metadata.len(),
        f.parts.len()
    );
    println!("--- metadata");
    for (k, v) in &f.metadata {
        if k.starts_with("tokenizer.ggml.")
            && matches!(v, llmario_engine_formats::MetaValue::Array(_))
        {
            println!("{k} = {}", v.summary());
            continue;
        }
        if k == "tokenizer.chat_template" {
            println!("{k} = <{} bytes>", v.as_str().map(|s| s.len()).unwrap_or(0));
            continue;
        }
        println!("{k} = {}", v.summary());
    }
    println!("--- bytes by type");
    let total = f.tensor_bytes_total();
    for (t, b) in f.bytes_by_type() {
        println!(
            "{t:>8}: {:>10.1} MiB ({:.1}%)",
            b as f64 / 1048576.0,
            b as f64 * 100.0 / total as f64
        );
    }
    println!("total weights: {:.2} GiB", total as f64 / 1073741824.0);
    match ArchSpec::from_gguf(&f) {
        Ok(spec) => {
            println!("--- arch");
            println!("{}", serde_json::to_string_pretty(&spec)?);
            println!(
                "kv bytes/token: f16 {} B, f32 {} B",
                spec.kv_bytes_per_token(2.0),
                spec.kv_bytes_per_token(4.0)
            );
        }
        Err(e) => println!("--- arch: not runnable by this build: {e}"),
    }
    if list_tensors {
        println!("--- tensors");
        for t in &f.tensors {
            println!(
                "{:<40} {:>8} {:<24} {:>12}",
                t.name,
                t.dtype,
                t.shape.to_string(),
                t.span.len
            );
        }
    }
    Ok(())
}

fn raw_run(
    path: &std::path::Path,
    tokens: &str,
    n: usize,
    threads: Option<usize>,
    ctx: usize,
    device: Device,
    kv_type: KvType,
) -> Result<()> {
    let prompt: Vec<u32> = tokens
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<u32>())
        .collect::<std::result::Result<_, _>>()
        .context("--tokens must be comma-separated integers")?;
    anyhow::ensure!(!prompt.is_empty(), "empty prompt");
    let t0 = Instant::now();
    let f = GgufFile::open(path)?;
    let threads = threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });
    let use_metal = match device {
        Device::Cpu => false,
        Device::Metal => true,
        Device::Auto => llmario_engine_server::engine::metal_available(),
    };
    let n_batch = prompt.len().max(1);
    // Expert streaming for measurements (the server turns it on when its plan streams
    // experts): LLMARIO_EXPERT_PREFETCH=1.
    let cpu_opts = CpuOptions {
        kv_type,
        stream: llmario_engine_model::StreamOptions {
            prefetch: std::env::var("LLMARIO_EXPERT_PREFETCH").is_ok_and(|v| v == "1"),
            sim_resident: None,
        },
        ..CpuOptions::new(threads, ctx, n_batch)
    };
    let mut backend: Box<dyn ModelBackend> = match (use_metal, device) {
        (false, _) => Box::new(CpuBackend::with_options(&f, cpu_opts)?),
        (true, Device::Metal) => Box::new(open_metal(&f, ctx, n_batch, kv_type)?),
        // `auto`: a model the Metal kernels do not cover runs on the CPU, with the reason logged.
        (true, _) => match open_metal(&f, ctx, n_batch, kv_type) {
            Ok(b) => Box::new(b),
            Err(e) => {
                eprintln!("metal backend unavailable for this model ({e}); using the CPU backend");
                Box::new(CpuBackend::with_options(&f, cpu_opts)?)
            }
        },
    };
    let spec = backend.spec().clone();
    eprintln!(
        "loaded {} ({}) in {:.2}s; backend {}; threads {}; kernels {}; kv {}",
        spec.name.clone().unwrap_or_default(),
        spec.arch,
        t0.elapsed().as_secs_f32(),
        backend.name(),
        threads,
        llmario_engine_cpu::simd::kernels().name,
        backend.kv_type().name()
    );
    let t1 = Instant::now();
    let logits = backend.forward(&prompt);
    let mut next = argmax(logits);
    let prefill = t1.elapsed();
    eprintln!(
        "prefill {} tokens in {:.3}s ({:.1} tok/s)",
        prompt.len(),
        prefill.as_secs_f32(),
        prompt.len() as f32 / prefill.as_secs_f32()
    );
    let mut out = vec![next];
    let t2 = Instant::now();
    for _ in 1..n {
        let logits = backend.forward(&[next]);
        next = argmax(logits);
        out.push(next);
    }
    let decode = t2.elapsed();
    eprintln!(
        "decode {} tokens in {:.3}s ({:.2} tok/s)",
        out.len() - 1,
        decode.as_secs_f32(),
        (out.len() - 1) as f32 / decode.as_secs_f32().max(1e-9)
    );
    println!(
        "{}",
        out.iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    if let Ok(tok) = llmario_engine_tokenizer::Tokenizer::from_gguf(&f) {
        // Stop at the first end-of-generation token, as a chat loop would.
        let upto = out.iter().position(|&t| tok.is_eog(t)).unwrap_or(out.len());
        let text = tok.decode(&out[..upto]);
        eprintln!(
            "decoded: {}",
            serde_json::to_string(&text).unwrap_or_default()
        );
    }
    Ok(())
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best as u32
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn open_metal(
    f: &GgufFile,
    ctx: usize,
    n_batch: usize,
    kv_type: KvType,
) -> Result<llmario_engine_metal::MetalBackend<'_>> {
    Ok(llmario_engine_metal::MetalBackend::with_options(
        f,
        llmario_engine_metal::MetalOptions {
            kv_type,
            ..llmario_engine_metal::MetalOptions::new(ctx, n_batch)
        },
    )?)
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn open_metal(f: &GgufFile, ctx: usize, n_batch: usize, kv_type: KvType) -> Result<CpuBackend<'_>> {
    let _ = (ctx, n_batch, kv_type);
    let _ = f;
    anyhow::bail!("this build has no Metal backend (use --device cpu)")
}
