//! `llmario bench`: reproducible benchmark through llmario or against any OpenAI-compatible
//! server (`--url`), with the environment manifest embedded in the report.

use crate::{util, RuntimeArgs};
use clap::Args;
use llmario_benchmark::report::{Environment, Report};
use llmario_benchmark::{Settings, Suite, Target};
use llmario_supervisor::autotune::{Autotune, Measurement};
use llmario_supervisor::speed;
use std::path::PathBuf;

#[derive(Args)]
pub struct BenchArgs {
    /// Model id or family (llmario mode), or the server-side model name with --url.
    #[arg(long, short)]
    pub model: String,
    /// Benchmark an external OpenAI-compatible base URL instead (e.g. http://127.0.0.1:11434/v1).
    #[arg(long)]
    pub url: Option<String>,
    /// Bearer key for --url targets.
    #[arg(long, env = "BENCH_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,
    /// PID of the external server's inference process, to sample its memory.
    #[arg(long)]
    pub pid: Option<u32>,
    /// Comma-separated concurrency levels.
    #[arg(long, default_value = "1,4", value_delimiter = ',')]
    pub concurrency: Vec<usize>,
    /// Repetitions of each prompt per level.
    #[arg(long, default_value_t = 3)]
    pub runs: usize,
    #[arg(long, default_value_t = 1)]
    pub warmup: usize,
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f64,
    /// Sampling seed (omitted by default: with temperature 0 decoding is greedy, and a seed
    /// disables batching on mlx_lm.server).
    #[arg(long)]
    pub seed: Option<u64>,
    /// Prefix-cache mode: cold (unique prefix per request; measures real prefill) or warm.
    #[arg(long, default_value = "cold")]
    pub cache: llmario_benchmark::CacheMode,
    /// Suite TOML (default: built-in `default` suite).
    #[arg(long)]
    pub suite: Option<PathBuf>,
    #[arg(long)]
    pub no_quality: bool,
    /// Run even when the machine is busy (1-minute load average above half the CPU cores).
    /// The report is then marked as not comparable.
    #[arg(long)]
    pub allow_busy: bool,
    /// Label for the report (default: target + model).
    #[arg(long)]
    pub label: Option<String>,
    /// Output directory (default: $LLMARIO_HOME/bench).
    #[arg(long)]
    pub out: Option<PathBuf>,
    #[command(flatten)]
    pub rt: RuntimeArgs,
}

pub async fn run(a: BenchArgs) -> anyhow::Result<()> {
    let (paths, cfg) = util::load_config(&a.rt)?;
    let suite = match &a.suite {
        Some(p) => Suite::load(p)?,
        None => Suite::builtin(),
    };
    let settings = Settings {
        concurrency: a.concurrency.clone(),
        runs: a.runs.max(1),
        warmup: a.warmup,
        temperature: a.temperature,
        seed: a.seed,
        cache: a.cache,
        skip_quality: a.no_quality,
    };
    let hw = util::detect_hardware().await;
    let mut env = Environment {
        hardware_fingerprint: hw.fingerprint(),
        hardware: hw.summary_line(),
        os: hw.os_version.clone(),
        power_note: util::power_note(),
        model: a.model.clone(),
        load_average_1m: llmario_hardware::HardwareReport::load_average_1m(),
        logical_cores: Some(hw.logical_cores),
        ..Default::default()
    };
    let busy = env.busy();
    if busy {
        let msg = format!(
            "the machine is busy (1-minute load average {:.1} on {} cores); timings would not be comparable. Close other work and retry, or pass --allow-busy to run anyway (the report is then marked busy)",
            env.load_average_1m.unwrap_or_default(),
            hw.logical_cores
        );
        if !a.allow_busy {
            anyhow::bail!(msg);
        }
        eprintln!("warning: {msg}");
    }
    let progress = |m: &str| eprintln!("  · {m}");

    let (levels, quality, samples, cold, estimate, idle, label) = if let Some(url) = &a.url {
        env.target = format!("external {url}");
        env.backend = Some("external".into());
        let t = Target {
            base_url: url.clone(),
            model: a.model.clone(),
            api_key: a.api_key.clone(),
        };
        eprintln!("benchmarking {} at {url}", a.model);
        let (l, q, s) = llmario_benchmark::run(&t, &suite, &settings, a.pid, progress).await?;
        let label = a
            .label
            .clone()
            .unwrap_or_else(|| format!("{} @ {url}", a.model));
        (l, q, s, None, None, None, label)
    } else {
        let sup = util::supervisor(cfg, paths.clone()).await?;
        let res = async {
            let sel = sup.select(&a.model)?;
            let plan = sup.plan_memory(&sel, 0);
            let status = &sup.statuses()[&sel.backend];
            env.target = "llmario gateway (in-process, loopback)".into();
            env.backend = Some(sel.backend.to_string());
            env.backend_version = status.version.clone();
            env.model = sel.model.id.clone();
            env.model_format = Some(sel.model.format.to_string());
            env.model_weight_bytes = Some(sel.model.weight_bytes()).filter(|b| *b > 0);
            env.model_quantization = sel.model.quantization.clone();
            env.model_content_hash = sel.model.content_hash();
            env.model_source = sel
                .model
                .source
                .as_ref()
                .map(|s| format!("{}@{}", s.repo, s.revision));
            env.profile = Some(serde_json::to_value(&sel.profile)?);

            util::load_with_report(&sup, &a.model).await?;
            let lease = sup.acquire(&a.model).await?;
            let cold = lease.engine.ready_after().as_secs_f64();
            let pid = lease.engine.pid;
            let idle = llmario_hardware::process_memory_bytes(pid);
            drop(lease);

            let (addr, _srv) = llmario_api::spawn_ephemeral(sup.clone()).await?;
            let t = Target {
                base_url: format!("http://{addr}/v1"),
                model: sel.model.id.clone(),
                api_key: sup.cfg.server.api_key.clone(),
            };
            eprintln!("benchmarking {} ({})", sel.model.id, sel.backend);
            let (l, q, s) =
                llmario_benchmark::run(&t, &suite, &settings, Some(pid), progress).await?;
            // Record plain one-at-a-time decode speed for the speed planner (never busy runs).
            let one = l.iter().find(|x| x.concurrency == 1 && x.errors == 0);
            if let (false, Some(tps)) = (
                busy,
                one.and_then(|x| x.decode_tps.as_ref()).map(|s| s.mean),
            ) {
                let (draft, _) = sup.draft_for(&sel);
                let bytes =
                    speed::bytes_read_per_token(&sel.model, &sel.profile, sel.backend, &sup.cfg);
                let m = Measurement {
                    hardware: sup.hw.fingerprint(),
                    backend: sel.backend,
                    engine_version: status.version.clone().unwrap_or_default(),
                    model: sel.model.id.clone(),
                    model_hash: sel.model.content_hash(),
                    profile: sel.profile.kind.to_string(),
                    speculative: speed::speculative_label(
                        &sel.model,
                        sel.backend,
                        &sup.cfg,
                        draft.as_ref().map(|d| d.id.as_str()),
                    ),
                    placement: speed::placement_label(&plan, &sup.hw),
                    kv_cache: speed::kv_cache_label(sel.backend, &sup.cfg),
                    decode_tps: tps,
                    bytes_per_token: bytes,
                    effective_gbs: tps * bytes as f64 / 1e9,
                    measured_at: chrono::Utc::now().to_rfc3339(),
                };
                let file = paths.autotune_file();
                let mut tuned = Autotune::load(&file);
                tuned.record(m);
                match tuned.save(&file) {
                    Ok(()) => eprintln!(
                        "recorded {tps:.1} tok/s for the speed planner ({})",
                        file.display()
                    ),
                    Err(e) => eprintln!("warning: could not save {}: {e}", file.display()),
                }
            }
            let label = a.label.clone().unwrap_or_else(|| {
                format!(
                    "{} via llmario/{} ({} profile)",
                    sel.model.id, sel.backend, sel.profile.kind
                )
            });
            anyhow::Ok((l, q, s, Some(cold), Some(plan.total_bytes), idle, label))
        }
        .await;
        sup.shutdown().await;
        res?
    };

    let mut levels = levels;
    if let Some(w) = env.model_weight_bytes {
        llmario_benchmark::add_effective_bandwidth(&mut levels, w);
    }
    let report = Report {
        // 2: environment load/weights, per-level draft acceptance and effective bandwidth.
        schema: 2,
        tool: format!("llmario {}", llmario_core::VERSION),
        timestamp: chrono::Utc::now().to_rfc3339(),
        label,
        environment: env,
        settings,
        suite: suite.name.clone(),
        suite_version: suite.version,
        cold_start_s: cold,
        estimated_memory_bytes: estimate,
        idle_memory_bytes: idle,
        levels,
        quality,
        samples,
    };
    let dir = a.out.unwrap_or_else(|| paths.bench_dir());
    std::fs::create_dir_all(&dir)?;
    let stem = format!(
        "{}-{}",
        chrono::Utc::now().format("%Y%m%dT%H%M%SZ"),
        report.environment.model.replace(
            |c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '.',
            "_"
        )
    );
    let json_path = dir.join(format!("{stem}.json"));
    let md_path = dir.join(format!("{stem}.md"));
    std::fs::write(&json_path, serde_json::to_vec_pretty(&report)?)?;
    let md = report.markdown();
    std::fs::write(&md_path, &md)?;
    println!("{md}");
    eprintln!("saved {} and {}", json_path.display(), md_path.display());
    Ok(())
}
