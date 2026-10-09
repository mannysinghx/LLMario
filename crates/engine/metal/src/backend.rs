//! `MetalBackend`: the dense forward pass of `llmario_engine_model::forward` executed with the
//! Metal kernels. The math (pre-norm, QKV projection with optional biases, per-head QK-norm, RoPE,
//! GQA attention with fp32 accumulation, SwiGLU, residuals, final norm, output head) is the same
//! op for op; only the storage (f16 KV cache, shared buffers) and the execution differ.

use crate::device::{groups, Buf, Cmd, Gpu};
use crate::{DeviceInfo, MetalError, Result};
use llmario_engine_core::GgmlType;
use llmario_engine_cpu::{QMat, RopeKind};
use llmario_engine_formats::GgufFile;
use llmario_engine_model::weights::Weights;
use llmario_engine_model::{ArchSpec, ModelBackend};
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
    w_gate: WRef,
    w_up: WRef,
    w_down: WRef,
    c: LayerConsts,
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
    part: Buf,
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
    pos0: u32,
    k_off: u32,
    v_off: u32,
    kv_dim: u32,
    v_dim: u32,
    q_norm: u32,
    k_norm: u32,
    rope: u32,
    eps: f32,
    theta: f32,
    freq_scale: f32,
    attn_factor: f32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct AttnParams {
    n_q: u32,
    n_head: u32,
    n_kv_head: u32,
    hd: u32,
    hdv: u32,
    kv_dim: u32,
    v_dim: u32,
    pos0: u32,
    k_off: u32,
    v_off: u32,
    n_split: u32,
    split_len: u32,
    scale: f32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct ElemParams {
    n: u32,
    rows: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Embed,
    Gemv,
    GemvAcc,
    GemvGlu,
    Gemm,
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
    kc: Buf,
    vc: Buf,
    s: Scratch,
    max_ctx: usize,
    /// KV rows allocated per layer (`max_ctx` padded for the prefill kernel's whole-tile loads).
    max_ctx_alloc: usize,
    n_batch: usize,
    kv_len: usize,
    /// Decode attention kernel for this head geometry.
    attn_vec: &'static str,
    /// Prefill attention kernel (`None`: use `attn_vec` for every batch).
    attn_prefill: Option<&'static str>,
    reserved: u64,
    logits: Vec<f32>,
    residency_from: usize,
    last_gpu_secs: f64,
}

impl<'a> MetalBackend<'a> {
    pub fn is_available() -> bool {
        crate::device::is_available()
    }

    pub fn device_info() -> Option<DeviceInfo> {
        crate::device::device_info()
    }

    pub fn new(file: &'a GgufFile, max_ctx: usize, n_batch: usize) -> Result<MetalBackend<'a>> {
        let gpu = Gpu::get()?;
        let spec = ArchSpec::from_gguf(file)?;
        if spec.gdn.is_some() || spec.n_attn_layers() != spec.n_layer {
            return Err(MetalError::Unsupported(format!(
                "family {:?} (recurrent layers) is CPU-only in this build",
                spec.family
            )));
        }
        let weights = Weights::load(file, &spec)?;
        if weights.hybrid.is_some() || weights.layers.len() != spec.n_layer as usize {
            return Err(MetalError::Unsupported("hybrid weight layout".into()));
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
        let attn_vec = match (hd, hdv) {
            (32, 32) => "attn_vec_hd32",
            (64, 64) => "attn_vec_hd64",
            (128, 128) => "attn_vec_hd128",
            (256, 256) => "attn_vec_hd256",
            _ => "attn_vec_generic",
        };
        let attn_prefill = match (hd, hdv) {
            (64, 64) => Some("attn_prefill_hd64"),
            (128, 128) => Some("attn_prefill_hd128"),
            _ => None,
        };
        let n_batch_alloc = n_batch.div_ceil(FA_BQ) * FA_BQ;
        let max_ctx_alloc = max_ctx.div_ceil(FA_BQ) * FA_BQ + FA_BQ;

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
            layers.push(LayerRefs {
                wq: locate(&l.wq)?,
                wk: locate(&l.wk)?,
                wv: locate(&l.wv)?,
                wo: locate(&l.wo)?,
                w_gate: locate(&l.w_gate)?,
                w_up: locate(&l.w_up)?,
                w_down: locate(&l.w_down)?,
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

        // --- KV cache (f16) and scratch, allocated once.
        let n_layer = spec.n_layer as usize;
        let kv_dim = spec.kv_dim() as usize;
        let v_dim = spec.v_dim() as usize;
        let k_bytes = n_layer * max_ctx_alloc * kv_dim * 2;
        let v_bytes = n_layer * max_ctx_alloc * v_dim * 2;
        let kc = gpu.alloc(k_bytes)?;
        let vc = gpu.alloc(v_bytes)?;
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
            logits: gpu.alloc(spec.n_vocab as usize * 4)?,
            tokens: gpu.alloc(nb * 4)?,
            part: gpu.alloc(ATTN_SPLIT_PAIRS * ATTN_MAX_SPLIT * (hdv + 2) * 4)?,
        };
        let reserved = [
            &kc,
            &vc,
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
            &s.part,
            &consts_buf,
        ]
        .iter()
        .map(|b| b.len() as u64)
        .sum();

        // --- Residency (macOS 15+).
        let residency_from = gpu.residency_count();
        let weight_refs: Vec<&Buf> = views.iter().collect();
        let wired = gpu.make_resident("llmario-weights", &weight_refs);
        let kv_refs: Vec<&Buf> = vec![
            &kc,
            &vc,
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
            &s.part,
            &consts_buf,
        ];
        gpu.make_resident("llmario-kv-scratch", &kv_refs);
        tracing::info!(
            device = %gpu.info.name,
            views = views.len(),
            weight_bytes = file.tensor_bytes_total(),
            kv_mib = (k_bytes + v_bytes) / (1024 * 1024),
            scratch_kib = (reserved - (k_bytes + v_bytes) as u64) / 1024,
            residency = wired,
            working_set_mib = gpu.info.recommended_max_working_set / (1024 * 1024),
            "Metal backend ready"
        );

        Ok(MetalBackend {
            logits: vec![0.0; spec.n_vocab as usize],
            gpu,
            spec,
            _weights: weights,
            views,
            token_embd,
            output,
            output_norm_off,
            layers,
            consts: consts_buf,
            kc,
            vc,
            s,
            max_ctx,
            max_ctx_alloc,
            n_batch,
            kv_len: 0,
            attn_vec,
            attn_prefill,
            reserved,
            residency_from,
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
        let p = NormParams {
            cols: cols as u32,
            eps: self.spec.rms_eps,
        };
        cmd.dispatch(
            "rms_norm",
            &[(0, x, x_off), (1, &self.consts, w_off), (2, y, 0)],
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

    fn attention_block(&self, cmd: &Cmd, l: usize, n: usize, pos0: usize) -> Result<()> {
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
        let k_off = l * self.max_ctx_alloc * kv_dim;
        let v_off = l * self.max_ctx_alloc * v_dim;
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
                pos0: pos0 as u32,
                k_off: k_off as u32,
                v_off: v_off as u32,
                kv_dim: kv_dim as u32,
                v_dim: v_dim as u32,
                q_norm: layer.c.q_norm.is_some() as u32,
                k_norm: layer.c.k_norm.is_some() as u32,
                rope: (!spec.nope_layers.contains(&(l as u32))) as u32,
                eps: spec.rms_eps,
                theta: r.theta,
                freq_scale: r.freq_scale,
                attn_factor: r.attn_factor,
            };
            cmd.dispatch(
                "qk_rope_kv",
                &[
                    (0, &s.q, 0),
                    (1, &s.k, 0),
                    (2, &s.v, 0),
                    (3, &self.consts, layer.c.q_norm.unwrap_or(0)),
                    (4, &self.consts, layer.c.k_norm.unwrap_or(0)),
                    (5, &self.kc, 0),
                    (6, &self.vc, 0),
                ],
                7,
                &p,
                (n * (n_head + 2 * n_kv), 1, 1),
                (32, 1, 1),
            )?;
            cmd.barrier();
        }
        {
            let n_pos_max = pos0 + n;
            let flash = n >= FA_MIN_TOKENS && self.attn_prefill.is_some();
            let n_split = if flash || n * n_head >= ATTN_SPLIT_PAIRS {
                1
            } else {
                n_pos_max.div_ceil(ATTN_SPLIT_KEYS).clamp(1, ATTN_MAX_SPLIT)
            };
            let split_len = n_pos_max.div_ceil(n_split).max(1);
            let p = AttnParams {
                n_q: n as u32,
                n_head: n_head as u32,
                n_kv_head: n_kv as u32,
                hd: hd as u32,
                hdv: hdv as u32,
                kv_dim: kv_dim as u32,
                v_dim: v_dim as u32,
                pos0: pos0 as u32,
                k_off: k_off as u32,
                v_off: v_off as u32,
                n_split: n_split as u32,
                split_len: split_len as u32,
                scale: 1.0 / (hd as f32).sqrt(),
            };
            let bufs = [
                (0, &s.q, 0),
                (1, &self.kc, 0),
                (2, &self.vc, 0),
                (3, &s.attn, 0),
                (4, &s.part, 0),
            ];
            if flash {
                cmd.dispatch(
                    self.attn_prefill.unwrap(),
                    &bufs,
                    5,
                    &p,
                    (groups(n, FA_BQ), n_head, 1),
                    (ATTN_TG, 1, 1),
                )?;
            } else {
                cmd.dispatch(
                    self.attn_vec,
                    &bufs,
                    5,
                    &p,
                    (n, n_head, n_split),
                    (ATTN_TG, 1, 1),
                )?;
            }
            cmd.barrier();
            if n_split > 1 {
                let threads = n * n_head * hdv;
                cmd.dispatch(
                    "attn_reduce",
                    &[(0, &s.part, 0), (1, &s.attn, 0)],
                    2,
                    &p,
                    (groups(threads, ELEM_TG), 1, 1),
                    (ELEM_TG, 1, 1),
                )?;
                cmd.barrier();
            }
        }
        // x += Wo attn (residual fused into the projection).
        self.project(cmd, &layer.wo, &s.attn, 0, n, &s.x, true)?;
        cmd.barrier();
        Ok(())
    }

    fn ffn_block(&self, cmd: &Cmd, l: usize, n: usize) -> Result<()> {
        let spec = &self.spec;
        let layer = &self.layers[l];
        let d = spec.d_model as usize;
        let n_ff = spec.n_ff as usize;
        let s = &self.s;
        self.rms_norm(cmd, &s.x, 0, layer.c.ffn_norm, &s.h, n, d)?;
        cmd.barrier();
        if n == 1 && layer.w_gate.dtype == layer.w_up.dtype {
            self.project_glu(cmd, &layer.w_gate, &layer.w_up, &s.h, &s.gate)?;
            cmd.barrier();
        } else {
            self.project(cmd, &layer.w_gate, &s.h, 0, n, &s.gate, false)?;
            self.project(cmd, &layer.w_up, &s.h, 0, n, &s.up, false)?;
            cmd.barrier();
            self.elem(cmd, "swiglu", &s.gate, &s.up, n * n_ff)?;
            cmd.barrier();
        }
        // x += Wd gate
        self.project(cmd, &layer.w_down, &s.gate, 0, n, &s.x, true)?;
        cmd.barrier();
        Ok(())
    }

    /// Encode one forward over `n` tokens at positions `pos0..` into `cmd`.
    fn encode(&self, cmd: &Cmd, n: usize, pos0: usize) -> Result<()> {
        let d = self.spec.d_model as usize;
        self.embed(cmd, n)?;
        cmd.barrier();
        for l in 0..self.layers.len() {
            self.attention_block(cmd, l, n, pos0)?;
            self.ffn_block(cmd, l, n)?;
        }
        // Final norm + head on the last token only.
        self.rms_norm(
            cmd,
            &self.s.x,
            (n - 1) * d * 4,
            self.output_norm_off,
            &self.s.h,
            1,
            d,
        )?;
        cmd.barrier();
        let head = self.output.unwrap_or(self.token_embd);
        self.project(cmd, &head, &self.s.h, 0, 1, &self.s.logits, false)
    }

    fn prepare(&mut self, tokens: &[u32]) -> usize {
        let n = tokens.len();
        assert!(n >= 1 && n <= self.n_batch, "batch of {n} tokens");
        assert!(self.kv_len + n <= self.max_ctx, "context overflow");
        let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        self.s.tokens.write_bytes(0, &bytes);
        n
    }

    /// GPU seconds of the last forward's command buffer.
    pub fn last_gpu_secs(&self) -> f64 {
        self.last_gpu_secs
    }

    /// Run one forward with every kernel in its own command buffer and return the per-dispatch
    /// (kernel, GPU seconds) list. Advances the cache like `forward`.
    pub fn profile_forward(&mut self, tokens: &[u32]) -> Result<Vec<(&'static str, f64)>> {
        let n = self.prepare(tokens);
        let pos0 = self.kv_len;
        let cmd = self.gpu.begin_profiled()?;
        self.encode(&cmd, n, pos0)?;
        let prof = cmd.profile();
        self.last_gpu_secs = cmd.finish()?;
        self.s.logits.read_f32(0, &mut self.logits);
        self.kv_len += n;
        Ok(prof)
    }

    fn run(&mut self, tokens: &[u32]) -> Result<()> {
        let n = self.prepare(tokens);
        let pos0 = self.kv_len;
        let cmd = self.gpu.begin()?;
        self.encode(&cmd, n, pos0)?;
        self.last_gpu_secs = cmd.finish()?;
        self.s.logits.read_f32(0, &mut self.logits);
        self.kv_len += n;
        Ok(())
    }
}

impl Drop for MetalBackend<'_> {
    fn drop(&mut self) {
        self.gpu.end_residency(self.residency_from);
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
    fn kv_len(&self) -> usize {
        self.kv_len
    }
    fn truncate(&mut self, n: usize) {
        self.kv_len = self.kv_len.min(n);
    }
    fn clear(&mut self) {
        self.kv_len = 0;
    }
    fn max_batch(&self) -> usize {
        self.n_batch
    }
    fn forward(&mut self, tokens: &[u32]) -> &[f32] {
        if let Err(e) = self.run(tokens) {
            // A failed command buffer leaves no usable state behind; surface it loudly rather
            // than return stale logits.
            panic!("Metal forward failed: {e}");
        }
        &self.logits
    }
    fn reserved_bytes(&self) -> u64 {
        self.reserved
    }
}
