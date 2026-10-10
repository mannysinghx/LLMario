//! The memory planner (Architecture §5.3).
//!
//! Given a model's `ArchSpec`, the requested profile and the device budgets, [`plan`] computes
//! every byte the engine process will hold — weights (mapped), KV cache, recurrent state,
//! scratch, caches, the runtime constant — without allocating anything, applies the fixed
//! degradation ladder when the total exceeds the budget, and returns an immutable [`Plan`] with
//! the steps it took and a content hash. The engine charges exactly these numbers to the ledger
//! and verifies the measured peak against `planned_peak` after load.
//!
//! The KV numbers come from the same [`KvLayout`] the engine allocates (paged full-attention
//! layers rounded up to whole blocks plus one copy-on-write spare per slot, window rings and
//! recurrent state per slot), so the plan and the cache cannot drift apart. Placement across
//! devices plugs into the same ladder in M4.

use llmario_engine_core::ledger::DeviceId;
use llmario_engine_formats::GgufFile;
use llmario_engine_model::forward::Scratch;
use llmario_engine_model::kv::KvLayout;
use llmario_engine_model::{ArchSpec, KvOptions, KvType};
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
    /// Element type of the KV cache.
    #[serde(default)]
    pub kv_type: KvType,
    /// The ladder may switch an f16 cache to q8_0 (Architecture §8.3: near-lossless) before it
    /// shortens the context.
    #[serde(default)]
    pub kv_auto: bool,
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
            kv_type: KvType::F16,
            kv_auto: false,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "step")]
pub enum DegradationStep {
    PromptCacheDropped { before: u64 },
    SlotsReduced { from: u32, to: u32 },
    KvPrecisionReduced { from: KvType, to: KvType },
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
///
/// Ladder order: prompt cache → slots (16 → 4 → 1) → KV precision (f16 → q8_0, only with
/// `kv_auto` and when every head width is a multiple of 32) → context (halving toward
/// [`CONTEXT_FLOOR`]). The architecture lists precision after context; it runs first here
/// because q8_0 is near-lossless while a shorter context loses the conversation's start.
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
    let mut kv_type = req.kv_type;
    let n_batch = req.n_batch.max(1);
    let mut steps = Vec::new();
    let usable = budget.usable();
    let q8_ok = (0..spec.n_layer as usize).all(|l| {
        let g = spec.attn_geom(l);
        KvType::Q8_0.supports_head_dim(g.head_dim as usize)
            && KvType::Q8_0.supports_head_dim(g.head_dim_v as usize)
    });
    // An automatic q8_0 choice falls back to f16 on models whose heads q8_0 blocks cannot tile;
    // an explicit one is refused below.
    if kv_type == KvType::Q8_0 && !q8_ok && req.kv_auto {
        kv_type = KvType::F16;
    }
    let kv_type_refusal = (kv_type == KvType::Q8_0 && !q8_ok)
        .then(|| "the q8_0 KV cache needs head widths that are multiples of 32".to_string());

    struct Totals {
        kv: u64,
        recurrent: u64,
        scratch: u64,
        planned: u64,
        layout: KvLayout,
    }
    let total = |slots: u32, ctx: u32, prompt_cache: u64, kv_type: KvType| -> Totals {
        let layout = KvLayout::new(
            &spec,
            &KvOptions::new(ctx as usize)
                .seqs(slots as usize)
                .kv_type(kv_type),
        );
        let recurrent = (layout.recurrent_bytes * slots as usize) as u64;
        let kv = layout.reserved_bytes() - recurrent;
        let scratch = Scratch::bytes_with_seqs(&spec, n_batch as usize, slots as usize);
        Totals {
            kv,
            recurrent,
            scratch,
            planned: weights + kv + recurrent + scratch + prompt_cache + req.runtime_fixed,
            layout,
        }
    };

    let mut t = total(slots, ctx, prompt_cache, kv_type);
    // Ladder step 1: caches.
    if t.planned > usable && prompt_cache > 0 {
        steps.push(DegradationStep::PromptCacheDropped {
            before: prompt_cache,
        });
        prompt_cache = 0;
        t = total(slots, ctx, prompt_cache, kv_type);
    }
    // Step 2: slots (16 → 4 → 1).
    while t.planned > usable && slots > 1 {
        let to = if slots > 4 { 4 } else { 1 };
        steps.push(DegradationStep::SlotsReduced { from: slots, to });
        slots = to;
        t = total(slots, ctx, prompt_cache, kv_type);
    }
    // Step 3: KV precision.
    if t.planned > usable && req.kv_auto && kv_type == KvType::F16 && q8_ok {
        steps.push(DegradationStep::KvPrecisionReduced {
            from: kv_type,
            to: KvType::Q8_0,
        });
        kv_type = KvType::Q8_0;
        t = total(slots, ctx, prompt_cache, kv_type);
    }
    // Step 4: context, halving toward the floor.
    while t.planned > usable && ctx > CONTEXT_FLOOR {
        let to = (ctx / 2).max(CONTEXT_FLOOR);
        steps.push(DegradationStep::ContextReduced { from: ctx, to });
        ctx = to;
        t = total(slots, ctx, prompt_cache, kv_type);
    }
    // Step 5 (host offload of experts / layers) arrives with the MoE families.
    let fits = t.planned <= usable && kv_type_refusal.is_none();
    let refusal = kv_type_refusal.or_else(|| {
        (!fits).then(|| {
            format!(
            "needs {} on {} but only {} is usable ({} ceiling − {} headroom); weights alone are {}",
            fmt(t.planned),
            budget.device,
            fmt(usable),
            fmt(budget.ceiling),
            fmt(budget.headroom),
            fmt(weights)
        )
        })
    });
    let bytes_per_token = weights + (t.layout.paged_bytes_per_token() as u64) * ctx as u64;
    let devices = vec![DeviceTotals {
        device: budget.device,
        weights_mapped: weights,
        kv_cache: t.kv,
        recurrent_state: t.recurrent,
        scratch: t.scratch,
        prompt_cache,
        runtime_fixed: req.runtime_fixed,
        planned: t.planned,
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
        kv_dtype: kv_type.name().into(),
        devices,
        planned_peak: t.planned,
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

impl Plan {
    /// The KV element type the plan settled on.
    pub fn kv_type(&self) -> KvType {
        KvType::parse(&self.kv_dtype).unwrap_or_default()
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use llmario_engine_core::GgmlType;
    use llmario_engine_formats::gguf::writer::GgufWriter;
    use llmario_engine_formats::MetaValue;

    /// A small llama-family file (two layers, 32-wide heads so q8_0 applies, 64K context) whose
    /// KV cache dominates the plan.
    fn model(dir: &std::path::Path) -> GgufFile {
        let (d, n_head, hd, n_ff, vocab) = (64u64, 2u64, 32u64, 96u64, 64u64);
        let f32s = |n: u64| -> Vec<u8> { (0..n).flat_map(|_| 0.01f32.to_le_bytes()).collect() };
        let mut w = GgufWriter::new();
        w.meta("general.architecture", MetaValue::Str("llama".into()))
            .meta("llama.block_count", MetaValue::U32(2))
            .meta("llama.embedding_length", MetaValue::U32(d as u32))
            .meta("llama.attention.head_count", MetaValue::U32(n_head as u32))
            .meta(
                "llama.attention.head_count_kv",
                MetaValue::U32(n_head as u32),
            )
            .meta("llama.attention.key_length", MetaValue::U32(hd as u32))
            .meta("llama.attention.value_length", MetaValue::U32(hd as u32))
            .meta("llama.feed_forward_length", MetaValue::U32(n_ff as u32))
            .meta("llama.vocab_size", MetaValue::U32(vocab as u32))
            .meta("llama.context_length", MetaValue::U32(65536));
        w.tensor(
            "token_embd.weight",
            &[d, vocab],
            GgmlType::F32,
            f32s(d * vocab),
        );
        w.tensor("output_norm.weight", &[d], GgmlType::F32, f32s(d));
        for l in 0..2 {
            let p = |s: &str| format!("blk.{l}.{s}");
            w.tensor(&p("attn_norm.weight"), &[d], GgmlType::F32, f32s(d));
            for t in ["attn_q", "attn_k", "attn_v"] {
                w.tensor(
                    &p(&format!("{t}.weight")),
                    &[d, n_head * hd],
                    GgmlType::F32,
                    f32s(d * n_head * hd),
                );
            }
            w.tensor(
                &p("attn_output.weight"),
                &[n_head * hd, d],
                GgmlType::F32,
                f32s(d * n_head * hd),
            );
            w.tensor(&p("ffn_norm.weight"), &[d], GgmlType::F32, f32s(d));
            w.tensor(
                &p("ffn_gate.weight"),
                &[d, n_ff],
                GgmlType::F32,
                f32s(d * n_ff),
            );
            w.tensor(
                &p("ffn_up.weight"),
                &[d, n_ff],
                GgmlType::F32,
                f32s(d * n_ff),
            );
            w.tensor(
                &p("ffn_down.weight"),
                &[n_ff, d],
                GgmlType::F32,
                f32s(n_ff * d),
            );
        }
        let path = dir.join("plan.gguf");
        std::fs::write(&path, w.to_bytes()).unwrap();
        GgufFile::open(&path).unwrap()
    }

    fn req(ctx: u32, kv_auto: bool) -> PlanRequest {
        PlanRequest {
            ctx_per_slot: ctx,
            n_batch: 8,
            runtime_fixed: 0,
            kv_auto,
            ..Default::default()
        }
    }

    fn budget(bytes: u64) -> DeviceBudget {
        DeviceBudget {
            device: DeviceId::Host,
            ceiling: bytes,
            headroom: 0,
        }
    }

    #[test]
    fn kv_numbers_are_the_engines_own_layout() {
        let dir = tempfile::tempdir().unwrap();
        let f = model(dir.path());
        let spec = ArchSpec::from_gguf(&f).unwrap();
        for (slots, ctx, t) in [(1u32, 4096u32, KvType::F16), (4, 1000, KvType::Q8_0)] {
            let r = PlanRequest {
                slots,
                kv_type: t,
                ..req(ctx, false)
            };
            let p = plan(&f, "m", &r, &budget(u64::MAX / 4)).unwrap();
            let layout = KvLayout::new(
                &spec,
                &KvOptions::new(ctx as usize).seqs(slots as usize).kv_type(t),
            );
            assert_eq!(
                p.devices[0].kv_cache + p.devices[0].recurrent_state,
                layout.reserved_bytes()
            );
            assert_eq!(p.kv_type(), t);
            assert_eq!(
                p.devices[0].scratch,
                Scratch::bytes_with_seqs(&spec, 8, slots as usize)
            );
            assert!(p.fits && p.degradations.is_empty());
        }
    }

    #[test]
    fn ladder_prefers_q8_0_over_a_shorter_context() {
        let dir = tempfile::tempdir().unwrap();
        let f = model(dir.path());
        let full = plan(&f, "m", &req(65536, false), &budget(u64::MAX / 4)).unwrap();
        let kv_f16 = full.devices[0].kv_cache;
        // A budget that holds the q8_0 cache (53 % of f16) but not the f16 one.
        let b = budget(full.planned_peak - kv_f16 / 3);
        let p = plan(&f, "m", &req(65536, true), &b).unwrap();
        assert!(p.fits, "{:?}", p.refusal);
        assert_eq!(p.kv_dtype, "q8_0");
        assert_eq!(p.ctx_per_slot, 65536);
        assert_eq!(
            p.degradations,
            vec![DegradationStep::KvPrecisionReduced {
                from: KvType::F16,
                to: KvType::Q8_0
            }]
        );
        // Without permission to change the cache type the context shrinks instead.
        let p = plan(&f, "m", &req(65536, false), &b).unwrap();
        assert!(p.fits);
        assert_eq!(p.kv_dtype, "f16");
        assert_eq!(p.ctx_per_slot, 32768);
        assert!(matches!(
            p.degradations[..],
            [DegradationStep::ContextReduced {
                from: 65536,
                to: 32768
            }]
        ));
    }

    #[test]
    fn refuses_when_weights_alone_do_not_fit() {
        let dir = tempfile::tempdir().unwrap();
        let f = model(dir.path());
        let p = plan(&f, "m", &req(4096, true), &budget(1024)).unwrap();
        assert!(!p.fits);
        assert!(p.refusal.unwrap().contains("weights alone"));
    }
}
