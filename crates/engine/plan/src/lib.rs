//! The memory planner (Architecture §5.3).
//!
//! Given a model's `ArchSpec`, the requested profile and the device budgets, [`plan`] computes
//! every byte the engine process will hold — weights (mapped), KV cache, recurrent state,
//! scratch, caches, the runtime constant — without allocating anything, applies the fixed
//! degradation ladder when the total exceeds the budget, and returns an immutable [`Plan`] with
//! the steps it took and a content hash. The engine charges exactly these numbers to the ledger
//! and verifies the measured peak against `planned_peak` after load.
//!
//! M1 scope: one device (host), f32 KV, dense families. Placement across devices, the three
//! cache classes and KV precision steps plug into the same ladder in M2–M4.

use llmario_engine_core::ledger::DeviceId;
use llmario_engine_formats::GgufFile;
use llmario_engine_model::forward::Scratch;
use llmario_engine_model::{ArchSpec, KvCache};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const MIB: u64 = 1024 * 1024;
pub const GIB: u64 = 1024 * MIB;

/// Fixed cost of the process itself (code, allocator metadata, tokio, tokenizer tables,
/// templates), measured on the M4 Max for the M1 engine and refined by the post-load feedback.
pub const RUNTIME_FIXED_DEFAULT: u64 = 256 * MIB;
/// The ladder never shrinks a context below this (llama.cpp `--fit-ctx` floor).
pub const CONTEXT_FLOOR: u32 = 4096;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct DeviceBudget {
    pub device: DeviceId,
    /// Ceiling the OS or the supervisor gave us for this device.
    pub ceiling: u64,
    /// Bytes kept free for the OS and other applications.
    pub headroom: u64,
}

impl DeviceBudget {
    pub fn usable(&self) -> u64 {
        self.ceiling.saturating_sub(self.headroom)
    }
    /// Default headroom rule: max(1 GiB, 10 % of the ceiling).
    pub fn with_default_headroom(device: DeviceId, ceiling: u64) -> DeviceBudget {
        DeviceBudget {
            device,
            ceiling,
            headroom: (ceiling / 10).max(GIB),
        }
    }
}

/// What the caller asks for; the plan reports what it got.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct PlanRequest {
    pub slots: u32,
    pub ctx_per_slot: u32,
    /// Prefill micro-batch (tokens per forward call).
    pub n_batch: u32,
    pub threads: u32,
    /// Byte limit for the RAM prompt cache (0 = off).
    pub prompt_cache_bytes: u64,
    pub runtime_fixed: u64,
}

impl Default for PlanRequest {
    fn default() -> Self {
        PlanRequest {
            slots: 1,
            ctx_per_slot: 8192,
            n_batch: 512,
            threads: 0,
            prompt_cache_bytes: 0,
            runtime_fixed: RUNTIME_FIXED_DEFAULT,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "step")]
pub enum DegradationStep {
    PromptCacheDropped { before: u64 },
    SlotsReduced { from: u32, to: u32 },
    ContextReduced { from: u32, to: u32 },
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct DeviceTotals {
    pub device: DeviceId,
    pub weights_mapped: u64,
    pub kv_cache: u64,
    pub recurrent_state: u64,
    pub scratch: u64,
    pub prompt_cache: u64,
    pub runtime_fixed: u64,
    pub planned: u64,
    pub budget: u64,
    pub headroom: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub hash: String,
    pub model: String,
    pub arch: String,
    pub requested: PlanRequest,
    pub slots: u32,
    pub ctx_per_slot: u32,
    pub n_batch: u32,
    pub kv_dtype: String,
    pub devices: Vec<DeviceTotals>,
    pub planned_peak: u64,
    pub degradations: Vec<DegradationStep>,
    /// Bytes read per generated token (active weights + KV at full context), for the
    /// speed-of-light estimate.
    pub bytes_per_token: u64,
    pub fits: bool,
    /// Why it does not fit, when `fits` is false.
    pub refusal: Option<String>,
}

#[derive(thiserror::Error, Debug)]
pub enum PlanError {
    #[error("{0}")]
    Model(#[from] llmario_engine_model::ModelError),
}

/// Compute the plan for `file` on `budget`, degrading per the ladder until it fits or every step
/// is exhausted (then `fits == false` with a refusal naming the shortfall).
pub fn plan(
    file: &GgufFile,
    model_name: &str,
    req: &PlanRequest,
    budget: &DeviceBudget,
) -> Result<Plan, PlanError> {
    let spec = ArchSpec::from_gguf(file)?;
    let weights = file.tensor_bytes_total();
    let mut slots = req.slots.max(1);
    let mut ctx = req.ctx_per_slot.max(1).min(spec.context_length.max(1));
    let mut prompt_cache = req.prompt_cache_bytes;
    let n_batch = req.n_batch.max(1);
    let mut steps = Vec::new();
    let usable = budget.usable();

    let recurrent = KvCache::recurrent_bytes(&spec);
    let total = |slots: u32, ctx: u32, prompt_cache: u64| -> (u64, u64, u64) {
        let kv = (KvCache::bytes(&spec, ctx as usize) - recurrent) * slots as u64;
        let scratch = Scratch::bytes(&spec, n_batch as usize) + Scratch::bytes(&spec, 1);
        let planned =
            weights + kv + recurrent * slots as u64 + scratch + prompt_cache + req.runtime_fixed;
        (kv, scratch, planned)
    };

    let (mut kv, mut scratch, mut planned) = total(slots, ctx, prompt_cache);
    // Ladder step 1: caches.
    if planned > usable && prompt_cache > 0 {
        steps.push(DegradationStep::PromptCacheDropped {
            before: prompt_cache,
        });
        prompt_cache = 0;
        (kv, scratch, planned) = total(slots, ctx, prompt_cache);
    }
    // Step 2: slots (16 → 4 → 1).
    while planned > usable && slots > 1 {
        let to = if slots > 4 { 4 } else { 1 };
        steps.push(DegradationStep::SlotsReduced { from: slots, to });
        slots = to;
        (kv, scratch, planned) = total(slots, ctx, prompt_cache);
    }
    // Step 3: context, halving toward the floor.
    while planned > usable && ctx > CONTEXT_FLOOR {
        let to = (ctx / 2).max(CONTEXT_FLOOR);
        steps.push(DegradationStep::ContextReduced { from: ctx, to });
        ctx = to;
        (kv, scratch, planned) = total(slots, ctx, prompt_cache);
    }
    // Steps 4–5 (KV precision, host offload) arrive with the backends that support them.
    let fits = planned <= usable;
    let refusal = (!fits).then(|| {
        format!(
            "needs {} on {} but only {} is usable ({} ceiling − {} headroom); weights alone are {}",
            fmt(planned),
            budget.device,
            fmt(usable),
            fmt(budget.ceiling),
            fmt(budget.headroom),
            fmt(weights)
        )
    });
    let bytes_per_token = weights + spec.kv_bytes_per_token(4.0) * ctx as u64;
    let devices = vec![DeviceTotals {
        device: budget.device,
        weights_mapped: weights,
        kv_cache: kv,
        recurrent_state: recurrent * slots as u64,
        scratch,
        prompt_cache,
        runtime_fixed: req.runtime_fixed,
        planned,
        budget: budget.ceiling,
        headroom: budget.headroom,
    }];
    let mut p = Plan {
        hash: String::new(),
        model: model_name.to_string(),
        arch: spec.arch.clone(),
        requested: req.clone(),
        slots,
        ctx_per_slot: ctx,
        n_batch,
        kv_dtype: "f32".into(),
        devices,
        planned_peak: planned,
        degradations: steps,
        bytes_per_token,
        fits,
        refusal,
    };
    let mut h = Sha256::new();
    h.update(serde_json::to_vec(&p).unwrap());
    p.hash = hex::encode(&h.finalize()[..16]);
    Ok(p)
}

/// Human-readable table for logs and `GET /engine/plan`.
pub fn render(p: &Plan) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "plan {} · {} ({}) · slots {} · ctx {} · batch {} · kv {}\n",
        p.hash, p.model, p.arch, p.slots, p.ctx_per_slot, p.n_batch, p.kv_dtype
    ));
    for d in &p.devices {
        s.push_str(&format!(
            "  {:<6} weights {:>10}  kv {:>10}  state {:>10}  scratch {:>10}  prompt-cache {:>10}  runtime {:>10}\n",
            d.device.to_string(),
            fmt(d.weights_mapped),
            fmt(d.kv_cache),
            fmt(d.recurrent_state),
            fmt(d.scratch),
            fmt(d.prompt_cache),
            fmt(d.runtime_fixed)
        ));
        s.push_str(&format!(
            "         planned {:>10}  usable {:>10}  (ceiling {} − headroom {})\n",
            fmt(d.planned),
            fmt(d.budget.saturating_sub(d.headroom)),
            fmt(d.budget),
            fmt(d.headroom)
        ));
    }
    for st in &p.degradations {
        s.push_str(&format!("  degraded: {st:?}\n"));
    }
    s.push_str(&format!(
        "  bytes/token {} · {}\n",
        fmt(p.bytes_per_token),
        if p.fits { "fits" } else { "DOES NOT FIT" }
    ));
    if let Some(r) = &p.refusal {
        s.push_str(&format!("  {r}\n"));
    }
    s
}

pub fn fmt(b: u64) -> String {
    if b >= GIB {
        format!("{:.2} GiB", b as f64 / GIB as f64)
    } else if b >= MIB {
        format!("{:.1} MiB", b as f64 / MIB as f64)
    } else {
        format!("{b} B")
    }
}

/// Speed-of-light decode rate for a plan on a device with `bandwidth_gbs` GB/s (η = 1).
pub fn speed_of_light(p: &Plan, bandwidth_gbs: f64) -> f64 {
    if p.bytes_per_token == 0 {
        return 0.0;
    }
    bandwidth_gbs * 1e9 / p.bytes_per_token as f64
}
