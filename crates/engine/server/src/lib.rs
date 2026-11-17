//! The engine process runtime: one inference thread owning the model, KV cache and scratch,
//! fed by a queue (Architecture §4.2), and the HTTP surface the LLMario gateway relays to
//! (IPC contract v1: OpenAI-compatible chat completions with SSE, `GET /health`), plus the
//! `/engine/*` extension endpoints (plan, ledger, stats, control).

pub mod agent;
pub mod api;
pub mod engine;
pub mod footprint;
pub mod kvtier;
pub mod openai;
pub mod pressure;
pub mod toolcall;

use anyhow::{Context, Result};
use llmario_engine_core::ledger::{DeviceId, Ledger};
use llmario_engine_formats::GgufFile;
use llmario_engine_model::KvType;
use llmario_engine_plan::{DeviceBudget, Plan, PlanRequest, GIB};
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
    /// Bytes of the ceiling to leave free; `None` = the default rule (max(1 GiB, 10 %)). A
    /// supervisor that already took its headroom out of `memory_limit` passes 0, so the same
    /// reserve is not subtracted twice.
    pub memory_headroom: Option<u64>,
    pub device: Device,
    /// Allow the built-in `web_fetch` tool (and `web_search` with `searxng_url`). Off by default.
    pub web: bool,
    /// Self-hosted SearXNG instance for `web_search`.
    pub searxng_url: Option<String>,
    /// KV cache element type; `None` = automatic (see [`kv_choice`]).
    pub kv_type: Option<KvType>,
    /// Directory of the KV disk tier (conversations evicted from memory are saved there and
    /// read back when they continue); `None` = off.
    pub kv_cache_dir: Option<PathBuf>,
    /// Byte budget of that directory (0 = off).
    pub kv_cache_bytes: u64,
    /// Fewest tokens worth a disk write or a restore.
    pub kv_cache_min_tokens: usize,
    /// Free idle slots' KV caches when the system reports memory pressure (see `pressure.rs`).
    pub pressure_relief: bool,
    /// Seconds a cache must have been idle before a pressure warning frees it.
    pub pressure_idle_secs: u64,
    /// Start reading the whole model file into the page cache at load, in the background.
    pub readahead: bool,
}

/// Default idle time before a pressure warning frees a cache: long enough that a conversation
/// in full flow keeps its cache in memory, short enough that one left alone gives it back.
pub const PRESSURE_IDLE_SECS: u64 = 30;

/// Default budget of the KV disk tier.
pub const KV_CACHE_DEFAULT_BYTES: u64 = 8 * GIB;
/// Default fewest tokens worth a disk write or a restore (below this, recomputing is cheap).
pub const KV_CACHE_MIN_TOKENS: usize = 256;

/// Machines (or memory budgets) at or below this size get the q8_0 KV cache by default
/// (Architecture §8.3: q8_0 is the default on 8 and 16 GB machines).
pub const SMALL_MACHINE: u64 = 16 * GIB;

/// The KV type to plan with and whether the planner may change it: an explicit type is kept;
/// automatic picks q8_0 when the memory ceiling is at most [`SMALL_MACHINE`], f16 above, and lets
/// the ladder switch f16 to q8_0 before it shortens the context.
pub fn kv_choice(opts: &ServeOptions) -> (KvType, bool) {
    match opts.kv_type {
        Some(t) => (t, false),
        None => {
            let ceiling = opts.memory_limit.unwrap_or_else(physical_memory);
            let t = if ceiling <= SMALL_MACHINE {
                KvType::Q8_0
            } else {
                KvType::F16
            };
            (t, true)
        }
    }
}

/// Build the plan for `opts` against the memory ceiling (no allocation).
pub fn plan_for(file: &GgufFile, opts: &ServeOptions) -> Result<(Plan, DeviceBudget)> {
    let ceiling = opts.memory_limit.unwrap_or_else(physical_memory);
    let budget = match opts.memory_headroom {
        Some(headroom) => DeviceBudget {
            device: DeviceId::Host,
            ceiling,
            headroom: headroom.min(ceiling),
        },
        None => DeviceBudget::with_default_headroom(DeviceId::Host, ceiling),
    };
    let (kv_type, kv_auto) = kv_choice(opts);
    let req = PlanRequest {
        slots: opts.parallel.max(1),
        ctx_per_slot: opts.ctx,
        n_batch: opts.batch.max(1),
        threads: opts.threads as u32,
        kv_type,
        kv_auto,
        ..Default::default()
    };
    let mut plan = llmario_engine_plan::plan(file, &opts.model_id, &req, &budget)?;
    // The ceiling is the machine; what other programs leave free right now can be far less.
    plan.notes.extend(llmario_engine_plan::availability_note(
        &plan,
        available_memory(),
    ));
    Ok((plan, budget))
}

/// Threads the CPU backend uses when none are asked for: the physical performance cores.
///
/// Efficiency cores and SMT siblings slow a decode step down more than they add, because every
/// parallel region waits for its slowest chunk. Measured on an M4 Max (12 performance and 4
/// efficiency cores, Qwen3-1.7B Q4_K_M, 64 tokens after a 512-token prompt, three interleaved
/// rounds): 12 threads 90–99 tok/s, all 16 threads 78–80; prompts ran 7 % faster on 16
/// (257–269 against 239–247 tok/s). Decode is what a reader waits for, token by token.
pub fn default_threads() -> usize {
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    #[cfg(target_os = "macos")]
    {
        let mut v: i32 = 0;
        let mut len = std::mem::size_of::<i32>();
        // SAFETY: `v` and `len` describe a 4-byte buffer for an integer sysctl; the name is a
        // NUL-terminated literal.
        let r = unsafe {
            libc::sysctlbyname(
                c"hw.perflevel0.physicalcpu".as_ptr(),
                &mut v as *mut i32 as *mut libc::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if r == 0 && v > 0 {
            return (v as usize).min(logical);
        }
    }
    // Elsewhere: physical cores (no SMT siblings). A hybrid x86 CPU's efficiency cores are
    // counted, because no portable interface tells them apart.
    sysinfo::System::physical_core_count()
        .filter(|&n| n > 0)
        .map_or(logical, |n| n.min(logical))
}

pub fn physical_memory() -> u64 {
    let mut s = sysinfo::System::new();
    s.refresh_memory();
    s.total_memory()
}

/// Memory the system could hand out right now without pushing other programs to swap (free
/// plus what it can reclaim at once), as the operating system reports it.
pub fn available_memory() -> u64 {
    let mut s = sysinfo::System::new();
    s.refresh_memory();
    s.available_memory()
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
    let web = Arc::new(agent::WebTools::new(opts.web, opts.searxng_url.as_deref())?);
    if !web.available().is_empty() {
        tracing::info!(tools = ?web.available(), "web access on for built-in tools");
    }
    api::serve_http(runtime, file, opts, plan, ledger, web).await
}

/// One-line JSON state records on stdout for the supervisor and `doctor`.
pub fn emit_state(state: &str) {
    println!(
        "{}",
        serde_json::json!({"llmario_engine": {"state": state, "pid": std::process::id()}})
    );
}
