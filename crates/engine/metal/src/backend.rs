//! `MetalBackend`: the forward pass of `llmario_engine_model::forward` executed with the Metal
//! kernels, for the dense families and the routed mixture-of-experts family (`qwen3moe`). The math
//! (pre-norm, QKV projection with optional biases, per-head QK-norm, RoPE, GQA attention with fp32
//! accumulation, SwiGLU or routed SwiGLU experts, residuals, final norm, output head) is the same
//! op for op; only the storage (paged KV cache, shared buffers) and the execution differ.

use crate::device::{groups, Buf, Cmd, Gpu};
use crate::kv::{MetalKv, BLOCK_TOKENS};
use crate::{DeviceInfo, MetalError, MetalOptions, Result};
use llmario_engine_core::GgmlType;
use llmario_engine_cpu::{QMat, RopeKind};
use llmario_engine_formats::GgufFile;
use llmario_engine_model::weights::Weights;
use llmario_engine_model::{ArchSpec, KvFull, KvType, ModelBackend, SeqTokens};
use std::sync::Arc;

/// A weight matrix addressed inside one no-copy view of the mapping.
#[derive(Clone, Copy, Debug)]
struct WRef {
    view: usize,
    off: usize,
    dtype: GgmlType,
    rows: usize,
    cols: usize,
    row_bytes: usize,
}

/// Byte offsets of a layer's small vectors inside the constants buffer.
struct LayerConsts {
    attn_norm: usize,
    ffn_norm: usize,
    q_norm: Option<usize>,
    k_norm: Option<usize>,
    bq: Option<usize>,
    bk: Option<usize>,
    bv: Option<usize>,
}

struct LayerRefs {
    wq: WRef,
    wk: WRef,
    wv: WRef,
    wo: WRef,
    ffn: Ffn,
    c: LayerConsts,
}

enum Ffn {
    Dense {
        gate: WRef,
        up: WRef,
        down: WRef,
    },
    /// Routed experts: `gate` / `up` / `down` address expert 0 and are `rows` of one expert;
    /// expert `e` starts `e · expert_bytes` further into the same view.
    Moe {
        router: WRef,
        gate: WRef,
        up: WRef,
        down: WRef,
        gate_expert_bytes: usize,
        down_expert_bytes: usize,
    },
}

/// Buffers of the routed FFN, sized for `n_batch` rows × `k` experts.
struct MoeScratch {
    /// `[n][n_expert]` router logits.
    router: Buf,
    /// `[n·k]` selected experts (u32) and their weights.
    sel: Buf,
    w: Buf,
    /// `[n·k][n_ff_exp]` gate (then the SwiGLU product) and up rows per pair.
    gate: Buf,
    up: Buf,
    /// `[n·k][d]` expert outputs per pair.
    out: Buf,
    /// `[n_expert]` pairs per expert and `[n_expert][cap]` their pair ids (prefill path).
    counts: Buf,
    ids: Buf,
    cap: usize,
}

/// Activation buffers sized once for `n_batch` tokens.
struct Scratch {
    x: Buf,
    h: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    attn: Buf,
    gate: Buf,
    up: Buf,
    ffn: Buf,
    logits: Buf,
    tokens: Buf,
    /// `(sequence, position)` of each row, `[n_batch][2]` u32.
    tokpos: Buf,
    part: Buf,
    moe: Option<MoeScratch>,
}

/// Threadgroup geometry shared with the shaders.
const GEMV_NSG: usize = 2;
const GEMV_NR: usize = 2;
const GEMV_ROWS_PER_TG: usize = GEMV_NSG * GEMV_NR;
const GEMM_BM: usize = 64;
const GEMM_BN: usize = 32;
const ATTN_TG: usize = 128;
const ATTN_MAX_HD: usize = 256;
/// Query rows per threadgroup of the prefill (flash) attention kernel; q/attn buffers and the KV
/// cache are padded to this so the kernel's whole-tile loads and stores stay in bounds.
const FA_BQ: usize = 32;
/// Prefill attention kernel is used from this many tokens.
const FA_MIN_TOKENS: usize = 8;
const NORM_TG: usize = 256;
/// Partials buffer covers this many (token, head) pairs × splits (split-K only runs below it).
const ATTN_SPLIT_PAIRS: usize = 64;
const ATTN_MAX_SPLIT: usize = 16;
const ATTN_SPLIT_KEYS: usize = 128;
const ELEM_TG: usize = 256;
/// Up to this many rows the routed FFN runs one expert matvec per (row, expert) pair; above it,
/// pairs are grouped by expert and each expert runs one GEMM.
const MOE_GEMV_MAX_ROWS: usize = 4;
const MOE_MAX_EXPERTS: usize = 1024;
const MOE_GROUP_TG: usize = 256;
const ROUTE_TG: usize = 32;

#[repr(C)]
#[derive(Clone, Copy)]
struct EmbedParams {
    cols: u32,
    n_tokens: u32,
    row_bytes: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct GemvParams {
    rows: u32,
    cols: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct GemmParams {
    rows: u32,
    cols: u32,
    n: u32,
    row_bytes: u32,
    accumulate: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct NormParams {
    cols: u32,
    eps: f32,
}
/// Paged-cache addressing of one layer (misc.metal `KvPage`).
#[repr(C)]
#[derive(Clone, Copy)]
struct KvPage {
    shift: u32,
    mask: u32,
    k_base: u32,
    v_base: u32,
    k_row: u32,
    v_row: u32,
    bps: u32,
    pad: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct QkRopeParams {
    n_tokens: u32,
    n_head: u32,
    n_kv_head: u32,
    hd: u32,
    hdv: u32,
    rot_dim: u32,
    mode: u32,
    q_norm: u32,
    k_norm: u32,
    rope: u32,
    eps: f32,
    theta: f32,
    freq_scale: f32,
    attn_factor: f32,
    kv: KvPage,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct AttnParams {
    n_q: u32,
    n_head: u32,
    n_kv_head: u32,
    hd: u32,
    hdv: u32,
    pos0: u32,
    n_split: u32,
    split_len: u32,
    scale: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    kv: KvPage,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct ElemParams {
    n: u32,
    rows: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct MoeRouteParams {
    n_expert: u32,
    k: u32,
    norm: u32,
    pad: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct GemvIdParams {
    rows: u32,
    cols: u32,
    expert_bytes: u32,
    k: u32,
    x_per_pair: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct MoeGroupParams {
    n_pairs: u32,
    n_expert: u32,
    cap: u32,
    pad: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct GemmIdParams {
    rows: u32,
    cols: u32,
    row_bytes: u32,
    expert_bytes: u32,
    k: u32,
    x_per_pair: u32,
    cap: u32,
    pad: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct MoeCombineParams {
    n: u32,
    d: u32,
    k: u32,
    pad: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Embed,
    Gemv,
    GemvAcc,
    GemvGlu,
    Gemm,
    GemvId,
    GemvGluId,
    GemmId,
}

fn kernel_name(kind: Kind, dtype: GgmlType) -> Result<&'static str> {
    use GgmlType as T;
    Ok(match (kind, dtype) {
        (Kind::Embed, T::F32) => "embed_f32",
        (Kind::Embed, T::F16) => "embed_f16",
        (Kind::Embed, T::Q4_0) => "embed_q4_0",
        (Kind::Embed, T::Q8_0) => "embed_q8_0",
        (Kind::Embed, T::Q4_K) => "embed_q4_k",
        (Kind::Embed, T::Q5_K) => "embed_q5_k",
        (Kind::Embed, T::Q6_K) => "embed_q6_k",
        (Kind::Gemv, T::F32) => "gemv_f32",
        (Kind::Gemv, T::F16) => "gemv_f16",
        (Kind::Gemv, T::Q4_0) => "gemv_q4_0",
        (Kind::Gemv, T::Q8_0) => "gemv_q8_0",
        (Kind::Gemv, T::Q4_K) => "gemv_q4_k",
        (Kind::Gemv, T::Q5_K) => "gemv_q5_k",
        (Kind::Gemv, T::Q6_K) => "gemv_q6_k",
        (Kind::GemvAcc, T::F32) => "gemv_acc_f32",
        (Kind::GemvAcc, T::F16) => "gemv_acc_f16",
        (Kind::GemvAcc, T::Q4_0) => "gemv_acc_q4_0",
        (Kind::GemvAcc, T::Q8_0) => "gemv_acc_q8_0",
        (Kind::GemvAcc, T::Q4_K) => "gemv_acc_q4_k",
        (Kind::GemvAcc, T::Q5_K) => "gemv_acc_q5_k",
        (Kind::GemvAcc, T::Q6_K) => "gemv_acc_q6_k",
        (Kind::GemvGlu, T::F32) => "gemv_glu_f32",
        (Kind::GemvGlu, T::F16) => "gemv_glu_f16",
        (Kind::GemvGlu, T::Q4_0) => "gemv_glu_q4_0",
        (Kind::GemvGlu, T::Q8_0) => "gemv_glu_q8_0",
        (Kind::GemvGlu, T::Q4_K) => "gemv_glu_q4_k",
        (Kind::GemvGlu, T::Q5_K) => "gemv_glu_q5_k",
        (Kind::GemvGlu, T::Q6_K) => "gemv_glu_q6_k",
        (Kind::Gemm, T::F32) => "gemm_f32",
        (Kind::Gemm, T::F16) => "gemm_f16",
        (Kind::Gemm, T::Q4_0) => "gemm_q4_0",
        (Kind::Gemm, T::Q8_0) => "gemm_q8_0",
        (Kind::Gemm, T::Q4_K) => "gemm_q4_k",
        (Kind::Gemm, T::Q5_K) => "gemm_q5_k",
        (Kind::Gemm, T::Q6_K) => "gemm_q6_k",
        (Kind::GemvId, T::F32) => "gemv_id_f32",
        (Kind::GemvId, T::F16) => "gemv_id_f16",
        (Kind::GemvId, T::Q4_0) => "gemv_id_q4_0",
        (Kind::GemvId, T::Q8_0) => "gemv_id_q8_0",
        (Kind::GemvId, T::Q4_K) => "gemv_id_q4_k",
        (Kind::GemvId, T::Q5_K) => "gemv_id_q5_k",
        (Kind::GemvId, T::Q6_K) => "gemv_id_q6_k",
        (Kind::GemvGluId, T::F32) => "gemv_glu_id_f32",
        (Kind::GemvGluId, T::F16) => "gemv_glu_id_f16",
        (Kind::GemvGluId, T::Q4_0) => "gemv_glu_id_q4_0",
        (Kind::GemvGluId, T::Q8_0) => "gemv_glu_id_q8_0",
        (Kind::GemvGluId, T::Q4_K) => "gemv_glu_id_q4_k",
        (Kind::GemvGluId, T::Q5_K) => "gemv_glu_id_q5_k",
        (Kind::GemvGluId, T::Q6_K) => "gemv_glu_id_q6_k",
        (Kind::GemmId, T::F32) => "gemm_id_f32",
        (Kind::GemmId, T::F16) => "gemm_id_f16",
        (Kind::GemmId, T::Q4_0) => "gemm_id_q4_0",
        (Kind::GemmId, T::Q8_0) => "gemm_id_q8_0",
        (Kind::GemmId, T::Q4_K) => "gemm_id_q4_k",
        (Kind::GemmId, T::Q5_K) => "gemm_id_q5_k",
        (Kind::GemmId, T::Q6_K) => "gemm_id_q6_k",
        (_, other) => {
            return Err(MetalError::Unsupported(format!(
                "no Metal kernel for tensor type {other}"
            )))
        }
    })
}

/// Page-aligned no-copy views over one mapped part.
struct PartViews {
    base: usize,
    len: usize,
    /// Byte stride between view starts (== view size when there is a single view).
    step: usize,
    first_view: usize,
    n_views: usize,
}

pub struct MetalBackend<'a> {
    gpu: Arc<Gpu>,
    spec: ArchSpec,
    /// Kept for the norm vectors and the shape validation it already did.
    _weights: Weights<'a>,
    views: Vec<Buf>,
    token_embd: WRef,
    output: Option<WRef>,
    output_norm_off: usize,
    layers: Vec<LayerRefs>,
    consts: Buf,
    kv: MetalKv,
    s: Scratch,
    max_ctx: usize,
    n_batch: usize,
    /// Sequences one call may carry (the scratch's logits rows).
    max_seqs: usize,
    /// Decode attention kernel for this head geometry and cache type.
    attn_vec: &'static str,
    /// Prefill attention kernel (`None`: use `attn_vec` for every batch).
    attn_prefill: Option<&'static str>,
    qk_rope_kv: &'static str,
    reserved: u64,
    logits: Vec<f32>,
    /// Owner id of this backend's residency sets.
    residency_owner: u64,
    last_gpu_secs: f64,
}

impl<'a> MetalBackend<'a> {
    pub fn is_available() -> bool {
        crate::device::is_available()
    }

    pub fn device_info() -> Option<DeviceInfo> {
        crate::device::device_info()
    }

    /// A single-slot f16 backend.
    pub fn new(file: &'a GgufFile, max_ctx: usize, n_batch: usize) -> Result<MetalBackend<'a>> {
        Self::with_options(file, MetalOptions::new(max_ctx, n_batch))
    }

    pub fn with_options(file: &'a GgufFile, o: MetalOptions) -> Result<MetalBackend<'a>> {
        let (max_ctx, n_batch, kv_type) = (o.max_ctx, o.n_batch, o.kv_type);
        let gpu = Gpu::get()?;
        let spec = ArchSpec::from_gguf(file)?;
        if let Some(m) = &spec.moe {
            if m.n_expert as usize > MOE_MAX_EXPERTS || m.n_expert_used == 0 {
                return Err(MetalError::Unsupported(format!(
                    "{} experts with {} used per token (the Metal router handles up to {MOE_MAX_EXPERTS})",
                    m.n_expert, m.n_expert_used
                )));
            }
        }
        if spec.gdn.is_some() || spec.n_attn_layers() != spec.n_layer {
            return Err(MetalError::Unsupported(format!(
                "family {:?} (recurrent layers) is CPU-only in this build",
                spec.family
            )));
        }
        let weights = Weights::load(file, &spec)?;
        if weights.hybrid.is_some() || weights.layers.len() != spec.n_layer as usize {
            return Err(MetalError::Unsupported(format!(
                "family {:?} (its layer layout) is CPU-only in this build",
                spec.family
            )));
        }
        let n_batch = n_batch.max(1);
        let max_ctx = max_ctx.max(1);
        let d = spec.d_model as usize;
        let hd = spec.head_dim as usize;
        let hdv = spec.head_dim_v as usize;
        if hd > ATTN_MAX_HD || hdv > ATTN_MAX_HD {
            return Err(MetalError::Unsupported(format!(
                "head_dim {hd}/{hdv} > {ATTN_MAX_HD}"
            )));
        }
        if spec.rope.dim % 2 != 0 || spec.rope.dim as usize > hd {
            return Err(MetalError::Unsupported(format!(
                "rope dim {} for head_dim {hd}",
                spec.rope.dim
            )));
        }
        if d % 16 != 0 {
            return Err(MetalError::Unsupported(format!(
                "d_model {d} is not a multiple of 16"
            )));
        }
        if !kv_type.supports_head_dim(hd) || !kv_type.supports_head_dim(hdv) {
            return Err(MetalError::Unsupported(format!(
                "KV cache type {} with head width {hd}/{hdv} (needs multiples of 32)",
                kv_type.name()
            )));
        }
        let q8 = kv_type == KvType::Q8_0;
        let attn_vec = match ((hd, hdv), q8) {
            ((32, 32), false) => "attn_vec_hd32_f16",
            ((64, 64), false) => "attn_vec_hd64_f16",
            ((128, 128), false) => "attn_vec_hd128_f16",
            ((256, 256), false) => "attn_vec_hd256_f16",
            (_, false) => "attn_vec_generic_f16",
            ((32, 32), true) => "attn_vec_hd32_q8_0",
            ((64, 64), true) => "attn_vec_hd64_q8_0",
            ((128, 128), true) => "attn_vec_hd128_q8_0",
            ((256, 256), true) => "attn_vec_hd256_q8_0",
            (_, true) => "attn_vec_generic_q8_0",
        };
        let attn_prefill = match ((hd, hdv), q8) {
            ((64, 64), false) => Some("attn_prefill_hd64_f16"),
            ((128, 128), false) => Some("attn_prefill_hd128_f16"),
            ((64, 64), true) => Some("attn_prefill_hd64_q8_0"),
            ((128, 128), true) => Some("attn_prefill_hd128_q8_0"),
            _ => None,
        };
        let qk_rope_kv = if q8 {
            "qk_rope_kv_q8_0"
        } else {
            "qk_rope_kv_f16"
        };
        let n_batch_alloc = n_batch.div_ceil(FA_BQ) * FA_BQ;
        let max_seqs = o.n_seqs.clamp(1, n_batch);

        // --- Weight views over the mapping (zero-copy).
        let max_tensor = file.tensors.iter().map(|t| t.span.len).max().unwrap_or(0) as usize;
        let mut views = Vec::new();
        let mut part_views = Vec::new();
        for part in &file.parts {
            let base = part.mmap.as_ptr() as usize;
            let len = part.mmap.len();
            let pv = Self::map_part(&gpu, base, len, max_tensor, &mut views)?;
            part_views.push(pv);
        }
        let locate = |m: &QMat<'a>| -> Result<WRef> {
            let ptr = m.data.as_ptr() as usize;
            let pv = part_views
                .iter()
                .find(|pv| ptr >= pv.base && ptr + m.data.len() <= pv.base + pv.len)
                .ok_or_else(|| MetalError::Device("tensor outside every mapping".into()))?;
            let rel = ptr - pv.base;
            let vi = (rel / pv.step).min(pv.n_views - 1);
            let view = &views[pv.first_view + vi];
            let view_start = vi * pv.step;
            if rel < view_start || rel + m.data.len() > view_start + view.len() {
                return Err(MetalError::Device(
                    "tensor does not fit in one buffer view".into(),
                ));
            }
            if m.cols % 16 != 0 {
                return Err(MetalError::Unsupported(format!(
                    "matrix row length {} is not a multiple of 16",
                    m.cols
                )));
            }
            kernel_name(Kind::Gemv, m.dtype)?;
            Ok(WRef {
                view: pv.first_view + vi,
                off: rel - view_start,
                dtype: m.dtype,
                rows: m.rows,
                cols: m.cols,
                row_bytes: m.row_bytes(),
            })
        };

        // --- Constants (norm weights, biases) in one shared buffer.
        let mut consts: Vec<f32> = Vec::new();
        let mut push = |v: &[f32]| -> usize {
            let off = consts.len() * 4;
            consts.extend_from_slice(v);
            // keep every vector 16-byte aligned
            while consts.len() % 4 != 0 {
                consts.push(0.0);
            }
            off
        };
        let output_norm_off = push(&weights.output_norm);
        let mut layers = Vec::with_capacity(weights.layers.len());
        for l in &weights.layers {
            let c = LayerConsts {
                attn_norm: push(&l.attn_norm),
                ffn_norm: push(&l.ffn_norm),
                q_norm: l.q_norm.as_deref().map(&mut push),
                k_norm: l.k_norm.as_deref().map(&mut push),
                bq: l.bq.as_deref().map(&mut push),
                bk: l.bk.as_deref().map(&mut push),
                bv: l.bv.as_deref().map(&mut push),
            };
            let ffn = match &l.moe {
                Some(mw) => {
                    // Locate each 3-D tensor whole (one view), then address expert 0.
                    let expert0 = |all: WRef, rows: usize| WRef { rows, ..all };
                    let (gb, db) = (mw.gate.expert_bytes(), mw.down.expert_bytes());
                    if gb > u32::MAX as usize || db > u32::MAX as usize {
                        return Err(MetalError::Unsupported("expert matrix over 4 GiB".into()));
                    }
                    Ffn::Moe {
                        router: locate(&mw.gate_inp)?,
                        gate: expert0(locate(&mw.gate.all())?, mw.gate.rows),
                        up: expert0(locate(&mw.up.all())?, mw.up.rows),
                        down: expert0(locate(&mw.down.all())?, mw.down.rows),
                        gate_expert_bytes: gb,
                        down_expert_bytes: db,
                    }
                }
                None => Ffn::Dense {
                    gate: locate(&l.w_gate)?,
                    up: locate(&l.w_up)?,
                    down: locate(&l.w_down)?,
                },
            };
            layers.push(LayerRefs {
                wq: locate(&l.wq)?,
                wk: locate(&l.wk)?,
                wv: locate(&l.wv)?,
                wo: locate(&l.wo)?,
                ffn,
                c,
            });
        }
        let token_embd = locate(&weights.token_embd)?;
        let output = match &weights.output {
            Some(o) => Some(locate(o)?),
            None => None,
        };
        let consts_buf = gpu.alloc(consts.len() * 4)?;
        consts_buf.write_f32(0, &consts);

        // --- Paged KV cache (blocks created on demand) and scratch, allocated once.
        let kv_dim = spec.kv_dim() as usize;
        let v_dim = spec.v_dim() as usize;
        let kv = MetalKv::new(&gpu, &spec, max_ctx, max_seqs, kv_type)?;
        let nb = n_batch_alloc;
        let s = Scratch {
            x: gpu.alloc(nb * d * 4)?,
            h: gpu.alloc(nb * d * 4)?,
            q: gpu.alloc(nb * spec.q_dim() as usize * 4)?,
            k: gpu.alloc(nb * kv_dim * 4)?,
            v: gpu.alloc(nb * v_dim * 4)?,
            attn: gpu.alloc(nb * spec.n_head as usize * hdv * 4)?,
            gate: gpu.alloc(nb * spec.n_ff as usize * 4)?,
            up: gpu.alloc(nb * spec.n_ff as usize * 4)?,
            ffn: gpu.alloc(nb * d * 4)?,
            logits: gpu.alloc(max_seqs * spec.n_vocab as usize * 4)?,
            tokens: gpu.alloc(nb * 4)?,
            tokpos: gpu.alloc(nb * 8)?,
            part: gpu.alloc(ATTN_SPLIT_PAIRS * ATTN_MAX_SPLIT * (hdv + 2) * 4)?,
            moe: match &spec.moe {
                Some(m) => {
                    let (ne, k, ff) = (
                        m.n_expert as usize,
                        m.n_expert_used as usize,
                        m.n_ff_exp as usize,
                    );
                    Some(MoeScratch {
                        router: gpu.alloc(nb * ne * 4)?,
                        sel: gpu.alloc(nb * k * 4)?,
                        w: gpu.alloc(nb * k * 4)?,
                        gate: gpu.alloc(nb * k * ff * 4)?,
                        up: gpu.alloc(nb * k * ff * 4)?,
                        out: gpu.alloc(nb * k * d * 4)?,
                        counts: gpu.alloc(ne * 4)?,
                        ids: gpu.alloc(ne * nb * 4)?,
                        cap: nb,
                    })
                }
                None => None,
            },
        };
        let moe_bufs: Vec<&Buf> = s
            .moe
            .iter()
            .flat_map(|m| {
                [
                    &m.router, &m.sel, &m.w, &m.gate, &m.up, &m.out, &m.counts, &m.ids,
                ]
            })
            .collect();
        let scratch_bytes: u64 = moe_bufs.iter().map(|b| b.len() as u64).sum::<u64>()
            + [
                &s.x,
                &s.h,
                &s.q,
                &s.k,
                &s.v,
                &s.attn,
                &s.gate,
                &s.up,
                &s.ffn,
                &s.logits,
                &s.tokens,
                &s.tokpos,
                &s.part,
                &consts_buf,
            ]
            .iter()
            .map(|b| b.len() as u64)
            .sum::<u64>();
        let reserved = scratch_bytes + kv.reserved_bytes();

        // --- Residency (macOS 15+).
        let residency_owner = crate::device::residency_owner();
        let weight_refs: Vec<&Buf> = views.iter().collect();
        let wired = gpu.make_resident(residency_owner, "llmario-weights", &weight_refs);
        let mut kv_refs: Vec<&Buf> = vec![
            &kv.table,
            &s.x,
            &s.h,
            &s.q,
            &s.k,
            &s.v,
            &s.attn,
            &s.gate,
            &s.up,
            &s.ffn,
            &s.logits,
            &s.tokens,
            &s.tokpos,
            &s.part,
            &consts_buf,
        ];
        kv_refs.extend(moe_bufs.iter().copied());
        gpu.make_resident(residency_owner, "llmario-scratch", &kv_refs);
        tracing::info!(
            device = %gpu.info.name,
            views = views.len(),
            weight_bytes = file.tensor_bytes_total(),
            kv_type = kv_type.name(),
            kv_reserved_mib = kv.reserved_bytes() / (1024 * 1024),
            scratch_kib = scratch_bytes / 1024,
            residency = wired,
            working_set_mib = gpu.info.recommended_max_working_set / (1024 * 1024),
            "Metal backend ready"
        );

        Ok(MetalBackend {
            logits: vec![0.0; max_seqs * spec.n_vocab as usize],
            gpu,
            spec,
            _weights: weights,
            views,
            token_embd,
            output,
            output_norm_off,
            layers,
            consts: consts_buf,
            kv,
            s,
            max_ctx,
            n_batch,
            max_seqs,
            attn_vec,
            attn_prefill,
            qk_rope_kv,
            reserved,
            residency_owner,
            last_gpu_secs: 0.0,
        })
    }

    /// Wrap one mapped part in page-aligned views (overlapping when it exceeds
    /// `maxBufferLength`, as ggml-metal does, so every tensor lies inside one view).
    fn map_part(
        gpu: &Gpu,
        base: usize,
        len: usize,
        max_tensor: usize,
        views: &mut Vec<Buf>,
    ) -> Result<PartViews> {
        let page = gpu.page_size;
        let aligned = len.div_ceil(page) * page;
        let max_buf = gpu.max_buffer_length / page * page;
        let first_view = views.len();
        if aligned <= max_buf {
            // SAFETY: the mapping covers whole pages, so `aligned` bytes from the page-aligned
            // base are readable; the `GgufFile` outlives the backend (lifetime `'a`).
            views.push(unsafe { gpu.wrap_no_copy(base as *const u8, aligned) }?);
            return Ok(PartViews {
                base,
                len,
                step: aligned,
                first_view,
                n_views: 1,
            });
        }
        let overlap = (max_tensor.div_ceil(page) + 1) * page;
        if overlap >= max_buf {
            return Err(MetalError::Unsupported(format!(
                "a tensor of {max_tensor} bytes exceeds maxBufferLength {}",
                gpu.max_buffer_length
            )));
        }
        let step = max_buf - overlap;
        let mut start = 0;
        let mut n_views = 0;
        while start < aligned {
            let size = (aligned - start).min(max_buf);
            // SAFETY: as above; `start` is a page multiple inside the mapping.
            views.push(unsafe { gpu.wrap_no_copy((base + start) as *const u8, size) }?);
            n_views += 1;
            start += step;
        }
        Ok(PartViews {
            base,
            len,
            step,
            first_view,
            n_views,
        })
    }

    /// Bytes of GPU buffers this backend allocated (KV + scratch + constants).
    pub fn reserved(&self) -> u64 {
        self.reserved
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    // ----- encoding helpers -----

    fn embed(&self, cmd: &Cmd, n: usize) -> Result<()> {
        let w = &self.token_embd;
        let d = self.spec.d_model as usize;
        let p = EmbedParams {
            cols: d as u32,
            n_tokens: n as u32,
            row_bytes: w.row_bytes as u32,
        };
        let threads = n * (d / 16);
        cmd.dispatch(
            kernel_name(Kind::Embed, w.dtype)?,
            &[
                (0, &self.views[w.view], w.off),
                (1, &self.s.tokens, 0),
                (2, &self.s.x, 0),
            ],
            3,
            &p,
            (groups(threads, 64), 1, 1),
            (64, 1, 1),
        )?;
        Ok(())
    }

    /// `y = W x` (or `y += W x` with `acc`) for `n` tokens: GEMV for one token, tiled GEMM
    /// otherwise. The caller places barriers; this encodes exactly one dispatch.
    #[allow(clippy::too_many_arguments)]
    fn project(
        &self,
        cmd: &Cmd,
        w: &WRef,
        x: &Buf,
        x_off: usize,
        n: usize,
        y: &Buf,
        acc: bool,
    ) -> Result<()> {
        let wbuf = &self.views[w.view];
        if n == 1 {
            let p = GemvParams {
                rows: w.rows as u32,
                cols: w.cols as u32,
            };
            let kind = if acc { Kind::GemvAcc } else { Kind::Gemv };
            cmd.dispatch(
                kernel_name(kind, w.dtype)?,
                &[(0, wbuf, w.off), (1, x, x_off), (2, y, 0), (4, wbuf, w.off)],
                3,
                &p,
                (groups(w.rows, GEMV_ROWS_PER_TG), 1, 1),
                (GEMV_NSG * 32, 1, 1),
            )
        } else {
            let p = GemmParams {
                rows: w.rows as u32,
                cols: w.cols as u32,
                n: n as u32,
                row_bytes: w.row_bytes as u32,
                accumulate: acc as u32,
            };
            cmd.dispatch(
                kernel_name(Kind::Gemm, w.dtype)?,
                &[(0, wbuf, w.off), (1, x, x_off), (2, y, 0)],
                3,
                &p,
                (groups(n, GEMM_BN), groups(w.rows, GEMM_BM), 1),
                (128, 1, 1),
            )
        }
    }

    /// `y = silu(W1 x) * (W2 x)` for one token (gate and up fused).
    fn project_glu(&self, cmd: &Cmd, w1: &WRef, w2: &WRef, x: &Buf, y: &Buf) -> Result<()> {
        debug_assert_eq!(w1.dtype, w2.dtype);
        let p = GemvParams {
            rows: w1.rows as u32,
            cols: w1.cols as u32,
        };
        cmd.dispatch(
            kernel_name(Kind::GemvGlu, w1.dtype)?,
            &[
                (0, &self.views[w1.view], w1.off),
                (1, x, 0),
                (2, y, 0),
                (4, &self.views[w2.view], w2.off),
            ],
            3,
            &p,
            (groups(w1.rows, GEMV_ROWS_PER_TG), 1, 1),
            (GEMV_NSG * 32, 1, 1),
        )
    }

    fn add_bias(&self, cmd: &Cmd, y: &Buf, bias: usize, n: usize, rows: usize) -> Result<()> {
        let total = n * rows;
        let p = ElemParams {
            n: total as u32,
            rows: rows as u32,
        };
        cmd.dispatch(
            "add_bias",
            &[(0, y, 0), (1, &self.consts, bias)],
            2,
            &p,
            (groups(total, ELEM_TG), 1, 1),
            (ELEM_TG, 1, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn rms_norm(
        &self,
        cmd: &Cmd,
        x: &Buf,
        x_off: usize,
        w_off: usize,
        y: &Buf,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        self.rms_norm_at(cmd, x, x_off, w_off, y, 0, rows, cols)
    }

    #[allow(clippy::too_many_arguments)]
    fn rms_norm_at(
        &self,
        cmd: &Cmd,
        x: &Buf,
        x_off: usize,
        w_off: usize,
        y: &Buf,
        y_off: usize,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let p = NormParams {
            cols: cols as u32,
            eps: self.spec.rms_eps,
        };
        cmd.dispatch(
            "rms_norm",
            &[(0, x, x_off), (1, &self.consts, w_off), (2, y, y_off)],
            3,
            &p,
            (rows, 1, 1),
            (NORM_TG, 1, 1),
        )
    }

    fn elem(&self, cmd: &Cmd, name: &'static str, a: &Buf, b: &Buf, n: usize) -> Result<()> {
        let p = ElemParams {
            n: n as u32,
            rows: 0,
        };
        cmd.dispatch(
            name,
            &[(0, a, 0), (1, b, 0)],
            2,
            &p,
            (groups(n, ELEM_TG), 1, 1),
            (ELEM_TG, 1, 1),
        )
    }

    /// One attention layer over the stacked rows: QKV, per-head norm + RoPE + paged KV write,
    /// decode attention for the rows of `segs[..n_vec]` (one dispatch), prefill attention per
    /// remaining segment, output projection with the residual.
    fn attention_block(&self, cmd: &Cmd, l: usize, rows: &Rows) -> Result<()> {
        let n = rows.n;
        let spec = &self.spec;
        let layer = &self.layers[l];
        let d = spec.d_model as usize;
        let n_head = spec.n_head as usize;
        let n_kv = spec.n_kv_head as usize;
        let hd = spec.head_dim as usize;
        let hdv = spec.head_dim_v as usize;
        let kv_dim = n_kv * hd;
        let v_dim = n_kv * hdv;
        let s = &self.s;

        self.rms_norm(cmd, &s.x, 0, layer.c.attn_norm, &s.h, n, d)?;
        cmd.barrier();
        self.project(cmd, &layer.wq, &s.h, 0, n, &s.q, false)?;
        self.project(cmd, &layer.wk, &s.h, 0, n, &s.k, false)?;
        self.project(cmd, &layer.wv, &s.h, 0, n, &s.v, false)?;
        cmd.barrier();
        if layer.c.bq.is_some() || layer.c.bk.is_some() || layer.c.bv.is_some() {
            if let Some(b) = layer.c.bq {
                self.add_bias(cmd, &s.q, b, n, n_head * hd)?;
            }
            if let Some(b) = layer.c.bk {
                self.add_bias(cmd, &s.k, b, n, kv_dim)?;
            }
            if let Some(b) = layer.c.bv {
                self.add_bias(cmd, &s.v, b, n, v_dim)?;
            }
            cmd.barrier();
        }
        let page = self.kv_page(l);
        {
            let r = &spec.rope;
            let p = QkRopeParams {
                n_tokens: n as u32,
                n_head: n_head as u32,
                n_kv_head: n_kv as u32,
                hd: hd as u32,
                hdv: hdv as u32,
                rot_dim: r.dim,
                mode: match r.kind {
                    RopeKind::Normal => 0,
                    RopeKind::Neox => 1,
                },
                q_norm: layer.c.q_norm.is_some() as u32,
                k_norm: layer.c.k_norm.is_some() as u32,
                rope: (!spec.nope_layers.contains(&(l as u32))) as u32,
                eps: spec.rms_eps,
                theta: r.theta,
                freq_scale: r.freq_scale,
                attn_factor: r.attn_factor,
                kv: page,
            };
            cmd.dispatch(
                self.qk_rope_kv,
                &[
                    (0, &s.q, 0),
                    (1, &s.k, 0),
                    (2, &s.v, 0),
                    (3, &self.consts, layer.c.q_norm.unwrap_or(0)),
                    (4, &self.consts, layer.c.k_norm.unwrap_or(0)),
                    (5, &self.kv.table, 0),
                    (6, &s.tokpos, 0),
                ],
                7,
                &p,
                (n * (n_head + 2 * n_kv), 1, 1),
                (32, 1, 1),
            )?;
            cmd.barrier();
        }
        let base = AttnParams {
            n_q: 0,
            n_head: n_head as u32,
            n_kv_head: n_kv as u32,
            hd: hd as u32,
            hdv: hdv as u32,
            pos0: 0,
            n_split: 1,
            split_len: 1,
            scale: 1.0 / (hd as f32).sqrt(),
            pad0: 0,
            pad1: 0,
            pad2: 0,
            kv: page,
        };
        // Decode-style rows (single tokens and short runs): one dispatch, each row reads its own
        // sequence and position from `tokpos`.
        if rows.n_vec > 0 {
            let n_vec = rows.n_vec;
            let n_pos_max = rows.max_vec_pos + 1;
            let n_split = if n_vec * n_head >= ATTN_SPLIT_PAIRS {
                1
            } else {
                n_pos_max.div_ceil(ATTN_SPLIT_KEYS).clamp(1, ATTN_MAX_SPLIT)
            };
            let p = AttnParams {
                n_q: n_vec as u32,
                n_split: n_split as u32,
                split_len: n_pos_max.div_ceil(n_split).max(1) as u32,
                ..base
            };
            cmd.dispatch(
                self.attn_vec,
                &[
                    (0, &s.q, 0),
                    (1, &self.kv.table, 0),
                    (2, &s.tokpos, 0),
                    (3, &s.attn, 0),
                    (4, &s.part, 0),
                ],
                5,
                &p,
                (n_vec, n_head, n_split),
                (ATTN_TG, 1, 1),
            )?;
            if n_split > 1 {
                cmd.barrier();
                let threads = n_vec * n_head * hdv;
                cmd.dispatch(
                    "attn_reduce",
                    &[(0, &s.part, 0), (1, &s.attn, 0)],
                    2,
                    &p,
                    (groups(threads, ELEM_TG), 1, 1),
                    (ELEM_TG, 1, 1),
                )?;
            }
        }
        // Prompt segments: the flash kernel, one dispatch per sequence (rows are disjoint).
        for sg in &rows.segs[rows.n_vec_segs..] {
            let p = AttnParams {
                n_q: sg.n as u32,
                pos0: sg.pos0 as u32,
                ..base
            };
            cmd.dispatch(
                self.attn_prefill
                    .expect("prefill segments need the flash kernel"),
                &[
                    (0, &s.q, sg.start * n_head * hd * 4),
                    (
                        1,
                        &self.kv.table,
                        sg.seq * self.kv.layout.blocks_per_seq * 8,
                    ),
                    (3, &s.attn, sg.start * n_head * hdv * 4),
                ],
                5,
                &p,
                (groups(sg.n, FA_BQ), n_head, 1),
                (ATTN_TG, 1, 1),
            )?;
        }
        cmd.barrier();
        // x += Wo attn (residual fused into the projection).
        self.project(cmd, &layer.wo, &s.attn, 0, n, &s.x, true)?;
        cmd.barrier();
        Ok(())
    }

    fn kv_page(&self, l: usize) -> KvPage {
        let lay = &self.kv.layout.layers[l];
        KvPage {
            shift: BLOCK_TOKENS.trailing_zeros(),
            mask: (BLOCK_TOKENS - 1) as u32,
            k_base: lay.k_base() as u32,
            v_base: lay.v_base() as u32,
            k_row: lay.k_row as u32,
            v_row: lay.v_row as u32,
            bps: self.kv.layout.blocks_per_seq as u32,
            pad: 0,
        }
    }

    fn ffn_block(&self, cmd: &Cmd, l: usize, n: usize) -> Result<()> {
        let spec = &self.spec;
        let layer = &self.layers[l];
        let d = spec.d_model as usize;
        let n_ff = spec.n_ff as usize;
        let s = &self.s;
        self.rms_norm(cmd, &s.x, 0, layer.c.ffn_norm, &s.h, n, d)?;
        cmd.barrier();
        let (gate, up, down) = match &layer.ffn {
            Ffn::Dense { gate, up, down } => (gate, up, down),
            Ffn::Moe { .. } => return self.moe_block(cmd, &layer.ffn, n),
        };
        if n == 1 && gate.dtype == up.dtype {
            self.project_glu(cmd, gate, up, &s.h, &s.gate)?;
            cmd.barrier();
        } else {
            self.project(cmd, gate, &s.h, 0, n, &s.gate, false)?;
            self.project(cmd, up, &s.h, 0, n, &s.up, false)?;
            cmd.barrier();
            self.elem(cmd, "swiglu", &s.gate, &s.up, n * n_ff)?;
            cmd.barrier();
        }
        // x += Wd gate
        self.project(cmd, down, &s.gate, 0, n, &s.x, true)?;
        cmd.barrier();
        Ok(())
    }

    /// `x += MoE(h)` for `n` rows (`h` holds the normed residual): router, top-k selection,
    /// expert SwiGLU per (row, expert) pair, weighted sum in selection order.
    fn moe_block(&self, cmd: &Cmd, ffn: &Ffn, n: usize) -> Result<()> {
        let Ffn::Moe {
            router,
            gate,
            up,
            down,
            gate_expert_bytes,
            down_expert_bytes,
        } = ffn
        else {
            unreachable!("moe_block on a dense layer")
        };
        let m = self.spec.moe.as_ref().expect("MoE spec");
        let ms = self.s.moe.as_ref().expect("MoE scratch");
        let s = &self.s;
        let d = self.spec.d_model as usize;
        let (ne, k, ff) = (
            m.n_expert as usize,
            m.n_expert_used as usize,
            m.n_ff_exp as usize,
        );
        let pairs = n * k;

        self.project(cmd, router, &s.h, 0, n, &ms.router, false)?;
        cmd.barrier();
        let rp = MoeRouteParams {
            n_expert: ne as u32,
            k: k as u32,
            norm: m.norm_topk as u32,
            pad: 0,
        };
        cmd.dispatch(
            "moe_route",
            &[(0, &ms.router, 0), (1, &ms.sel, 0), (2, &ms.w, 0)],
            3,
            &rp,
            (n, 1, 1),
            (ROUTE_TG, 1, 1),
        )?;
        cmd.barrier();

        let gv = |w: &WRef, eb: usize, x_per_pair: bool| GemvIdParams {
            rows: w.rows as u32,
            cols: w.cols as u32,
            expert_bytes: eb as u32,
            k: k as u32,
            x_per_pair: x_per_pair as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let gm = |w: &WRef, eb: usize, x_per_pair: bool| GemmIdParams {
            rows: w.rows as u32,
            cols: w.cols as u32,
            row_bytes: w.row_bytes as u32,
            expert_bytes: eb as u32,
            k: k as u32,
            x_per_pair: x_per_pair as u32,
            cap: ms.cap as u32,
            pad: 0,
        };
        let gemv_id = |kind: Kind, w: &WRef, w2: &WRef, eb: usize, x: &Buf, xpp: bool, y: &Buf| {
            cmd.dispatch(
                kernel_name(kind, w.dtype)?,
                &[
                    (0, &self.views[w.view], w.off),
                    (1, x, 0),
                    (2, y, 0),
                    (4, &self.views[w2.view], w2.off),
                    (5, &ms.sel, 0),
                ],
                3,
                &gv(w, eb, xpp),
                (groups(w.rows, GEMV_ROWS_PER_TG), pairs, 1),
                (GEMV_NSG * 32, 1, 1),
            )
        };
        let gemm_id = |w: &WRef, eb: usize, x: &Buf, xpp: bool, y: &Buf| {
            cmd.dispatch(
                kernel_name(Kind::GemmId, w.dtype)?,
                &[
                    (0, &self.views[w.view], w.off),
                    (1, x, 0),
                    (2, y, 0),
                    (4, &ms.counts, 0),
                    (5, &ms.ids, 0),
                ],
                3,
                &gm(w, eb, xpp),
                (groups(n, GEMM_BN), groups(w.rows, GEMM_BM), ne),
                (128, 1, 1),
            )
        };

        if n <= MOE_GEMV_MAX_ROWS {
            if gate.dtype == up.dtype {
                gemv_id(
                    Kind::GemvGluId,
                    gate,
                    up,
                    *gate_expert_bytes,
                    &s.h,
                    false,
                    &ms.gate,
                )?;
            } else {
                gemv_id(
                    Kind::GemvId,
                    gate,
                    gate,
                    *gate_expert_bytes,
                    &s.h,
                    false,
                    &ms.gate,
                )?;
                gemv_id(
                    Kind::GemvId,
                    up,
                    up,
                    *gate_expert_bytes,
                    &s.h,
                    false,
                    &ms.up,
                )?;
                cmd.barrier();
                self.elem(cmd, "swiglu", &ms.gate, &ms.up, pairs * ff)?;
            }
            cmd.barrier();
            gemv_id(
                Kind::GemvId,
                down,
                down,
                *down_expert_bytes,
                &ms.gate,
                true,
                &ms.out,
            )?;
        } else {
            let gp = MoeGroupParams {
                n_pairs: pairs as u32,
                n_expert: ne as u32,
                cap: ms.cap as u32,
                pad: 0,
            };
            cmd.dispatch(
                "moe_group",
                &[(0, &ms.sel, 0), (1, &ms.counts, 0), (2, &ms.ids, 0)],
                3,
                &gp,
                (1, 1, 1),
                (MOE_GROUP_TG, 1, 1),
            )?;
            cmd.barrier();
            gemm_id(gate, *gate_expert_bytes, &s.h, false, &ms.gate)?;
            gemm_id(up, *gate_expert_bytes, &s.h, false, &ms.up)?;
            cmd.barrier();
            self.elem(cmd, "swiglu", &ms.gate, &ms.up, pairs * ff)?;
            cmd.barrier();
            gemm_id(down, *down_expert_bytes, &ms.gate, true, &ms.out)?;
        }
        cmd.barrier();
        let cp = MoeCombineParams {
            n: n as u32,
            d: d as u32,
            k: k as u32,
            pad: 0,
        };
        cmd.dispatch(
            "moe_combine",
            &[(0, &s.x, 0), (1, &ms.out, 0), (2, &ms.w, 0)],
            3,
            &cp,
            (groups(n * d, ELEM_TG), 1, 1),
            (ELEM_TG, 1, 1),
        )?;
        cmd.barrier();
        Ok(())
    }

    /// Encode one forward over the stacked rows into `cmd`, ending with the logits of each
    /// row in `rows.outs` (one per batch entry, in batch order).
    fn encode(&self, cmd: &Cmd, rows: &Rows) -> Result<()> {
        let d = self.spec.d_model as usize;
        let n = rows.n;
        self.embed(cmd, n)?;
        cmd.barrier();
        for l in 0..self.layers.len() {
            self.attention_block(cmd, l, rows)?;
            self.ffn_block(cmd, l, n)?;
        }
        // Final norm + head on each sequence's last row.
        for (i, &row) in rows.outs.iter().enumerate() {
            self.rms_norm_at(
                cmd,
                &self.s.x,
                row * d * 4,
                self.output_norm_off,
                &self.s.h,
                i * d * 4,
                1,
                d,
            )?;
        }
        cmd.barrier();
        let head = self.output.unwrap_or(self.token_embd);
        self.project(
            cmd,
            &head,
            &self.s.h,
            0,
            rows.outs.len(),
            &self.s.logits,
            false,
        )
    }

    /// Lay out a batch: decode-style segments first (so their rows are one contiguous range for
    /// the decode attention dispatch), then prompt segments; upload tokens and (seq, pos).
    fn layout_rows(&self, batch: &[SeqTokens]) -> Rows {
        let flash_ok = self.attn_prefill.is_some();
        let is_vec = |n: usize| !(flash_ok && n >= FA_MIN_TOKENS);
        let mut order: Vec<usize> = (0..batch.len()).collect();
        order.sort_by_key(|&i| !is_vec(batch[i].tokens.len()));
        let mut rows = Rows::default();
        let mut toks: Vec<u8> = Vec::new();
        let mut tokpos: Vec<u8> = Vec::new();
        rows.outs = vec![0; batch.len()];
        for &i in &order {
            let b = &batch[i];
            let pos0 = self.kv.len(b.seq);
            let start = rows.n;
            for (k, t) in b.tokens.iter().enumerate() {
                toks.extend_from_slice(&t.to_le_bytes());
                tokpos.extend_from_slice(&(b.seq as u32).to_le_bytes());
                tokpos.extend_from_slice(&((pos0 + k) as u32).to_le_bytes());
            }
            rows.n += b.tokens.len();
            rows.outs[i] = rows.n - 1;
            if is_vec(b.tokens.len()) {
                rows.n_vec = rows.n;
                rows.n_vec_segs += 1;
                rows.max_vec_pos = rows.max_vec_pos.max(pos0 + b.tokens.len() - 1);
            }
            rows.segs.push(Seg {
                seq: b.seq,
                start,
                n: b.tokens.len(),
                pos0,
            });
        }
        self.s.tokens.write_bytes(0, &toks);
        self.s.tokpos.write_bytes(0, &tokpos);
        rows
    }

    /// Validate `batch` and reserve its cache room (nothing changes when it does not fit).
    fn admit(&mut self, batch: &[SeqTokens]) -> std::result::Result<(), KvFull> {
        let total: usize = batch.iter().map(|b| b.tokens.len()).sum();
        assert!(
            !batch.is_empty() && batch.len() <= self.max_seqs,
            "{} sequences in one call (this backend holds {})",
            batch.len(),
            self.max_seqs
        );
        assert!(
            total <= self.n_batch,
            "batch of {total} tokens (max {})",
            self.n_batch
        );
        for (i, b) in batch.iter().enumerate() {
            assert!(
                !b.tokens.is_empty(),
                "empty token list for sequence {}",
                b.seq
            );
            assert!(
                b.seq < self.kv.n_seqs(),
                "sequence {} of {}",
                b.seq,
                self.kv.n_seqs()
            );
            assert!(
                batch[..i].iter().all(|o| o.seq != b.seq),
                "sequence {} appears twice in one batch",
                b.seq
            );
            assert!(
                self.kv.len(b.seq) + b.tokens.len() <= self.max_ctx,
                "context overflow"
            );
        }
        let need: usize = batch
            .iter()
            .map(|b| {
                self.kv
                    .blocks_needed(b.seq, self.kv.len(b.seq) + b.tokens.len())
            })
            .sum();
        if need > self.kv.free_blocks() {
            return Err(KvFull {
                needed: need,
                free: self.kv.free_blocks(),
            });
        }
        for b in batch {
            let new_len = self.kv.len(b.seq) + b.tokens.len();
            match self.kv.reserve(&self.gpu, b.seq, new_len) {
                Ok(Ok(())) => {}
                Ok(Err(e)) => panic!("Metal KV block allocation failed: {e}"),
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// GPU seconds of the last forward's command buffer.
    pub fn last_gpu_secs(&self) -> f64 {
        self.last_gpu_secs
    }

    /// Run one single-sequence forward (slot 0) with every kernel in its own command buffer and
    /// return the per-dispatch (kernel, GPU seconds) list. Advances the cache like `forward`.
    pub fn profile_forward(&mut self, tokens: &[u32]) -> Result<Vec<(&'static str, f64)>> {
        let batch = [SeqTokens { seq: 0, tokens }];
        self.admit(&batch)
            .map_err(|e| MetalError::Device(e.to_string()))?;
        let rows = self.layout_rows(&batch);
        let indirect = self.kv.prepare();
        let cmd = self.gpu.begin_profiled()?;
        cmd.use_indirect(&indirect);
        self.encode(&cmd, &rows)?;
        let prof = cmd.profile();
        self.last_gpu_secs = cmd.finish()?;
        let v = self.spec.n_vocab as usize;
        self.s.logits.read_f32(0, &mut self.logits[..v]);
        for sg in &rows.segs {
            self.kv.set_len(sg.seq, sg.pos0 + sg.n);
        }
        Ok(prof)
    }

    fn run(&mut self, batch: &[SeqTokens]) -> Result<std::result::Result<usize, KvFull>> {
        if let Err(e) = self.admit(batch) {
            return Ok(Err(e));
        }
        let rows = self.layout_rows(batch);
        let indirect = self.kv.prepare();
        let cmd = self.gpu.begin()?;
        cmd.use_indirect(&indirect);
        self.encode(&cmd, &rows)?;
        self.last_gpu_secs = cmd.finish()?;
        let m = batch.len() * self.spec.n_vocab as usize;
        self.s.logits.read_f32(0, &mut self.logits[..m]);
        for sg in &rows.segs {
            self.kv.set_len(sg.seq, sg.pos0 + sg.n);
        }
        Ok(Ok(m))
    }
}

/// Placement of one forward's stacked rows.
#[derive(Default)]
struct Rows {
    n: usize,
    /// Rows `0..n_vec` take the decode attention kernel; they are `segs[..n_vec_segs]`.
    n_vec: usize,
    n_vec_segs: usize,
    max_vec_pos: usize,
    segs: Vec<Seg>,
    /// Last row of each batch entry, in batch order.
    outs: Vec<usize>,
}

struct Seg {
    seq: usize,
    start: usize,
    n: usize,
    pos0: usize,
}

impl Drop for MetalBackend<'_> {
    fn drop(&mut self) {
        self.gpu.end_residency(self.residency_owner);
    }
}

impl ModelBackend for MetalBackend<'_> {
    fn spec(&self) -> &ArchSpec {
        &self.spec
    }
    fn name(&self) -> &'static str {
        "metal"
    }
    fn max_ctx(&self) -> usize {
        self.max_ctx
    }
    fn n_seqs(&self) -> usize {
        self.kv.n_seqs()
    }
    fn seq_len(&self, s: usize) -> usize {
        self.kv.len(s)
    }
    fn truncate_seq(&mut self, s: usize, n: usize) {
        self.kv.truncate(s, n);
    }
    fn clear_seq(&mut self, s: usize) {
        self.kv.clear(s);
    }
    fn max_batch(&self) -> usize {
        self.n_batch
    }
    fn forward_batch(&mut self, batch: &[SeqTokens]) -> std::result::Result<&[f32], KvFull> {
        match self.run(batch) {
            Ok(Ok(m)) => Ok(&self.logits[..m]),
            Ok(Err(full)) => Err(full),
            // A failed command buffer leaves no usable state behind; surface it loudly rather
            // than return stale logits.
            Err(e) => panic!("Metal forward failed: {e}"),
        }
    }
    fn reserved_bytes(&self) -> u64 {
        self.reserved
    }
    fn kv_in_use_bytes(&self) -> u64 {
        self.kv.in_use_bytes()
    }
    fn kv_reserved_bytes(&self) -> u64 {
        self.kv.reserved_bytes()
    }
    fn kv_type(&self) -> KvType {
        self.kv.layout.kv_type
    }
    fn kv_free_tokens(&self) -> usize {
        self.kv.free_tokens()
    }
    fn kv_fingerprint(&self) -> u64 {
        self.kv.fingerprint()
    }
    fn snapshot_bytes(&self, len: usize) -> usize {
        self.kv.snapshot_bytes(len)
    }
    fn snapshot_trimmable(&self) -> bool {
        true
    }
    fn write_seq(&self, s: usize, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        self.kv.write_seq(s, w)
    }
    fn read_seq(&mut self, s: usize, len: usize, bytes: &[u8]) -> bool {
        matches!(self.kv.read_seq(&self.gpu, s, len, bytes), Ok(true))
    }
}
