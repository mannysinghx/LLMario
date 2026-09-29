//! `llmario` command-line interface.

mod bench;
mod doctor;
mod models;
mod run;
mod util;

use clap::{Args, Parser, Subcommand};
use llmario_core::ProfileKind;

#[derive(Parser)]
#[command(name = "llmario", version, about = "Adaptive local LLM runtime", long_about = None)]
struct Cli {
    /// More logging (-v info, -vv debug). Prompts and completions are never logged.
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,
    /// Emit logs as JSON lines.
    #[arg(long, global = true)]
    log_json: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

/// Runtime overrides shared by run / serve / bench (highest precedence).
#[derive(Args, Clone, Default)]
pub struct RuntimeArgs {
    /// Workload profile: latency | balanced | throughput.
    #[arg(long)]
    pub profile: Option<ProfileKind>,
    /// Per-request context tokens (default: the profile's value, capped by the model).
    #[arg(long)]
    pub context: Option<u32>,
    /// Cap on memory llmario may plan to use, in GiB.
    #[arg(long)]
    pub memory_limit_gb: Option<f64>,
    /// Prefer this backend when a family has several installed variants (llamacpp | mlx).
    #[arg(long)]
    pub backend: Option<llmario_core::BackendKind>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Report hardware, backends, installed models and whether they fit.
    Doctor {
        #[arg(long)]
        json: bool,
    },
    /// Manage local models.
    #[command(subcommand)]
    Model(models::ModelCmd),
    /// Chat with a model in the terminal (one-shot if a prompt is given).
    Run {
        /// Model id or family name.
        model: String,
        /// Prompt; omit for an interactive session.
        prompt: Vec<String>,
        #[arg(long)]
        system: Option<String>,
        #[arg(long)]
        max_tokens: Option<u32>,
        #[arg(long)]
        temperature: Option<f64>,
        /// Hide reasoning/thinking output.
        #[arg(long)]
        hide_reasoning: bool,
        #[command(flatten)]
        rt: RuntimeArgs,
    },
    /// Serve the OpenAI-compatible API.
    Serve {
        #[arg(long)]
        host: Option<String>,
        #[arg(long)]
        port: Option<u16>,
        /// Require `Authorization: Bearer <key>` (prefer the LLMARIO_API_KEY env var).
        #[arg(long)]
        api_key: Option<String>,
        /// Allow binding a non-loopback address (also requires an API key).
        #[arg(long)]
        allow_remote: bool,
        /// Load this model at startup instead of on first request.
        #[arg(long)]
        preload: Option<String>,
        #[command(flatten)]
        rt: RuntimeArgs,
    },
    /// Benchmark a model through llmario, or any OpenAI-compatible server with --url.
    Bench(bench::BenchArgs),
    /// Show the effective configuration and file locations.
    Config,
    /// Internal: deterministic mock engine used by tests.
    #[command(hide = true)]
    MockEngine {
        #[arg(long)]
        port: u16,
        #[arg(long)]
        model: String,
    },
}

fn init_logging(verbose: u8, json: bool, default: &str) {
    let level = match verbose {
        0 => default,
        1 => "llmario=info,llmario_supervisor=info,llmario_api=info,warn",
        _ => "debug,hyper=info,reqwest=info,h2=info",
    };
    let filter =
        tracing_subscriber::EnvFilter::try_from_env("LLMARIO_LOG").unwrap_or_else(|_| level.into());
    let b = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(false);
    if json {
        b.json().init();
    } else {
        b.compact().init();
    }
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let default_level = match cli.cmd {
        Cmd::Serve { .. } => "llmario_supervisor=info,llmario_api=info,warn",
        _ => "warn",
    };
    if !matches!(cli.cmd, Cmd::MockEngine { .. }) {
        init_logging(cli.verbose, cli.log_json, default_level);
    }
    let result = match cli.cmd {
        Cmd::Doctor { json } => doctor::run(json).await,
        Cmd::Model(m) => models::run(m).await,
        Cmd::Run {
            model,
            prompt,
            system,
            max_tokens,
            temperature,
            hide_reasoning,
            rt,
        } => {
            run::run(run::RunOpts {
                model,
                prompt: prompt.join(" "),
                system,
                max_tokens,
                temperature,
                hide_reasoning,
                rt,
            })
            .await
        }
        Cmd::Serve {
            host,
            port,
            api_key,
            allow_remote,
            preload,
            rt,
        } => serve(host, port, api_key, allow_remote, preload, rt).await,
        Cmd::Bench(a) => bench::run(a).await,
        Cmd::Config => show_config(),
        Cmd::MockEngine { port, model } => llmario_adapter_mock::run_mock_engine(port, model).await,
    };
    if let Err(e) = result {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

async fn serve(
    host: Option<String>,
    port: Option<u16>,
    api_key: Option<String>,
    allow_remote: bool,
    preload: Option<String>,
    rt: RuntimeArgs,
) -> anyhow::Result<()> {
    let (paths, mut cfg) = util::load_config(&rt)?;
    if let Some(h) = host {
        cfg.server.host = h;
    }
    if let Some(p) = port {
        cfg.server.port = p;
    }
    if api_key.is_some() {
        cfg.server.api_key = api_key;
    }
    cfg.server.allow_remote |= allow_remote;
    cfg.validate()?;
    let sup = util::supervisor(cfg, paths).await?;
    util::print_backends(&sup);
    let p = llmario_core::ResolvedProfile::resolve(sup.profile, sup.cfg.runtime.context);
    eprintln!(
        "profile {}: {} parallel slot(s) × {} context tokens, batch {}/{}, default max_tokens {}",
        p.kind, p.parallel, p.ctx_per_slot, p.batch, p.ubatch, p.default_max_tokens
    );
    if let Some(m) = preload {
        util::load_with_report(&sup, &m).await?;
    }
    llmario_api::serve(sup).await
}

fn show_config() -> anyhow::Result<()> {
    let (paths, mut cfg) = util::load_config(&RuntimeArgs::default())?;
    if cfg.server.api_key.is_some() {
        cfg.server.api_key = Some("<redacted>".into());
    }
    println!("# home:     {}", paths.home.display());
    println!(
        "# config:   {} ({})",
        paths.config_file().display(),
        if paths.config_file().exists() {
            "present"
        } else {
            "absent; defaults in use"
        }
    );
    println!("# registry: {}", paths.registry_file().display());
    println!("# models:   {}", paths.models_dir().display());
    println!("# logs:     {}", paths.logs_dir().display());
    println!("# bench:    {}\n", paths.bench_dir().display());
    print!("{}", cfg.to_toml());
    Ok(())
}
