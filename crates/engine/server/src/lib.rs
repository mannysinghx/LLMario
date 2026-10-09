//! The engine process runtime: one inference thread owning the model, KV cache and scratch,
//! fed by a queue (Architecture §4.2), and the HTTP surface the LLMario gateway relays to
//! (IPC contract v1: OpenAI-compatible chat completions with SSE, `GET /health`), plus the
//! `/engine/*` extension endpoints (plan, ledger, stats, control).

pub mod api;
pub mod engine;
pub mod footprint;
pub mod openai;

use anyhow::{Context, Result};
use llmario_engine_core::ledger::{DeviceId, Ledger};
use llmario_engine_formats::GgufFile;
use llmario_engine_plan::{DeviceBudget, Plan, PlanRequest};
use std::path::PathBuf;
use std::sync::Arc;

/// Which backend runs the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Device {
    /// Metal when a usable GPU is present and the build has the `metal` feature, else CPU.
    #[default]
    Auto,
    Cpu,
    Metal,
}

impl std::str::FromStr for Device {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Device, String> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Device::Auto),
            "cpu" => Ok(Device::Cpu),
            "metal" | "gpu" => Ok(Device::Metal),
            other => Err(format!("unknown device `{other}` (auto|cpu|metal)")),
        }
    }
}

/// Startup options (the native adapter's command line).
#[derive(Clone, Debug)]
pub struct ServeOptions {
    pub model: PathBuf,
    pub model_id: String,
    pub listen: String,
    pub ctx: u32,
    pub parallel: u32,
    pub batch: u32,
    pub threads: usize,
    /// Ceiling for the plan; `None` = physical memory.
    pub memory_limit: Option<u64>,
    pub device: Device,
}

/// Build the plan for `opts` against the memory ceiling (no allocation).
pub fn plan_for(file: &GgufFile, opts: &ServeOptions) -> Result<(Plan, DeviceBudget)> {
    let ceiling = opts.memory_limit.unwrap_or_else(physical_memory);
    let budget = DeviceBudget::with_default_headroom(DeviceId::Host, ceiling);
    let req = PlanRequest {
        slots: opts.parallel.max(1),
        ctx_per_slot: opts.ctx,
        n_batch: opts.batch.max(1),
        threads: opts.threads as u32,
        ..Default::default()
    };
    let plan = llmario_engine_plan::plan(file, &opts.model_id, &req, &budget)?;
    Ok((plan, budget))
}

pub fn physical_memory() -> u64 {
    let mut s = sysinfo::System::new();
    s.refresh_memory();
    s.total_memory()
}

/// Load everything, print the plan, and serve until the process is told to exit.
pub async fn serve(opts: ServeOptions) -> Result<()> {
    emit_state("starting");
    let file = Arc::new(
        GgufFile::open(&opts.model).with_context(|| format!("open {}", opts.model.display()))?,
    );
    emit_state("planning");
    let (plan, _budget) = plan_for(&file, &opts)?;
    eprintln!("{}", llmario_engine_plan::render(&plan));
    if !plan.fits {
        emit_state("refused");
        anyhow::bail!(
            "model does not fit the memory plan: {}",
            plan.refusal.clone().unwrap_or_default()
        );
    }
    let ledger = Arc::new(Ledger::new(vec![DeviceId::Host]));
    ledger.set_planned_peak(plan.planned_peak);
    emit_state("loading");
    let runtime =
        engine::EngineRuntime::start(file.clone(), opts.clone(), plan.clone(), ledger.clone())
            .await
            .context("engine runtime")?;
    emit_state("ready");
    api::serve_http(runtime, file, opts, plan, ledger).await
}

/// One-line JSON state records on stdout for the supervisor and `doctor`.
pub fn emit_state(state: &str) {
    println!(
        "{}",
        serde_json::json!({"llmario_engine": {"state": state, "pid": std::process::id()}})
    );
}
