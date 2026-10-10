//! The Gemma 4 family (Google DeepMind, 2026; GGUF arch `gemma4`): dense decoder with a 5:1
//! sliding-window / global attention pattern, per-layer head geometry, K=V global attention,
//! GeGLU, pre- and post-norms around both sub-blocks, per-layer output scalars and final logit
//! soft-capping (Architecture §6.7, §8.1, Appendix C). Weights views and the per-layer forward
//! pass live here; the attention core, projections and scratch come from `forward.rs`; the
//! sliding-window ring is in `kv.rs`.
//!
//! # Equations (text path, non-MoE, no per-layer embeddings)
//!
//! Sources: llama.cpp `src/models/gemma4.cpp` (`load_arch_hparams`, `load_arch_tensors`, the
//! graph constructor), `src/llama-graph.cpp` (`build_inp_embd`, `build_norm`, `build_attn_mha`,
//! `build_ffn` with `LLM_FFN_GELU`/`LLM_FFN_PAR`), `src/llama-hparams.{h,cpp}` (`is_swa`,
//! `n_head_kv(il)`, `n_embd_head_k(il)`, `n_rot(il)`, `is_masked_swa`), `src/llama-model.cpp`
//! (SWA head size / n_rot defaults, `get_rope_freq_base`, `LLAMA_ROPE_TYPE_NEOX`),
//! `ggml/src/ggml-cpu/ops.cpp` (`ggml_rope_cache_init`, `rotate_pairs`) and `vec.h`
//! (`ggml_vec_geglu_f32`); the converter `conversion/gemma.py` (`Gemma4Model`); HF transformers
//! `models/gemma4/modeling_gemma4.py` (`Gemma4RMSNorm`, `Gemma4TextAttention`,
//! `Gemma4TextDecoderLayer`, `Gemma4TextScaledWordEmbedding`) and `modeling_rope_utils.py`
//! (`_compute_proportional_rope_parameters`).
//!
//! ```text
//! x₀  = sqrt(d_model) · E[tok]                       // build_inp_embd(tok_embd, sqrtf(n_embd)); f32
//! for each layer l (geometry per layer: n_kv(l), hd(l), window(l), RoPE(l)):
//!   h   = rms_norm(x) ⊙ attn_norm
//!   q   = rope_l(rms_norm_head(W_q h) ⊙ attn_q_norm)   // per head
//!   k   = rope_l(rms_norm_head(W_k h) ⊙ attn_k_norm)
//!   v   = rms_norm_head(W_v h)                           // W_v absent on K=V layers: W_v := W_k,
//!                                                        // i.e. v = rms_norm_head(W_k h), no gain
//!   a   = softmax(1.0 · q·Kᵀ + mask_l) · V               // f_attention_scale = 1.0 (not 1/√hd)
//!   x  += rms_norm(W_o a) ⊙ post_attention_norm
//!   h   = rms_norm(x) ⊙ ffn_norm
//!   f   = W_down (gelu_tanh(W_gate h) ⊙ W_up h)         // GeGLU
//!   x  += rms_norm(f) ⊙ post_ffw_norm
//!   x  *= layer_output_scale                             // scalar, when present
//! logits = c · tanh((output_norm(x) · Eᵀ) / c),  c = final_logit_softcapping (30)
//! ```
//!
//! - `rms_norm(x) = x / sqrt(mean(x²) + eps)`. **No `(1 + w)` gain:** `Gemma4RMSNorm` multiplies
//!   by the plain `weight` and the converter's `Gemma4Model.norm_shift` returns 0.0 (unlike
//!   Gemma 2/3, where the converter folds `+1` into the tensor), so the GGUF gains are used as is.
//! - Mask: layer `l` with `window = n_swa` lets a query at position `t` see keys `p` with
//!   `t − p < n_swa` (`is_masked_swa`, `LLAMA_SWA_TYPE_STANDARD`); global layers see all `p ≤ t`.
//!   The pattern is `attention.sliding_window_pattern` (bool per layer), `n_swa` is
//!   `attention.sliding_window` (1,024 on the 12B).
//! - Geometry (12B): 16 query heads everywhere; sliding layers 8 KV heads × 256
//!   (`key_length_swa`), global layers 1 KV head × 512 (`key_length`) with no `attn_v`
//!   (`attention_k_eq_v`); `attention.head_count_kv` is a per-layer array.
//! - RoPE (NeoX pairs `(i, i + n_dims/2)`, angle `pos · base^(−2i/n_dims) / ff[i]`, computed
//!   iteratively as ggml does): sliding layers base `rope.freq_base_swa` (10,000) over
//!   `rope.dimension_count_swa` (256) dims, no factors; global layers base `rope.freq_base`
//!   (1,000,000) over `rope.dimension_count` (512) dims with the converter's `rope_freqs.weight`
//!   factors `[1.0] × 64 ++ [1e30] × 192` — HF "proportional" RoPE with
//!   `partial_rotary_factor` 0.25 rotates pairs `i < 64` with `base^(−2i/512)` and leaves the rest
//!   untouched; a divisor of 1e30 makes the angle ≈ 0 to f32 precision, which is how llama.cpp
//!   expresses it without re-ordering the head.
//! - GELU: ggml-cpu's `ggml_vec_geglu_f32` is built with `GGML_GELU_FP16`: `gelu(x)` for
//!   `−10 < x < 10` is `fp16(gelu_tanh(fp16(x)))` via a 65,536-entry table, `0` below, `x` above.
//!   `geglu_ggml_inplace` builds the same table once (`ggml_table_gelu_f16`), so the GeGLU
//!   output equals llama.cpp's CPU path bit for bit for the same inputs.
//! - RoPE's `(cos θ, sin θ)` are computed once per position and shared by every Q/K head
//!   (`rope_cache` / `rope_apply`) from one `(t.sin(), t.cos())` expression, which LLVM
//!   lowers to a single `sincosf` like clang does for ggml's adjacent `cosf`/`sinf`; separate
//!   `sinf`/`cosf` calls differ in the last bit on ~4 % of the angles, enough to flip a near-tie
//!   greedy token in a 512-token prompt.
//!
//! # Cache
//!
//! Sliding layers keep a ring of `n_swa + 512` positions per sequence in [`KvCache`] (see
//! `kv.rs`), global layers are paged over the whole context. K=V global layers still store K and
//! V separately: the cached K is `attn_k_norm` + RoPE of the shared projection and V its
//! weightless RMS norm, so they differ. 12B at f16: 8 global layers × (512 + 512) × 2 B =
//! 16 KiB per token, plus 40 sliding layers × 8 × (256 + 256) × 2 B = 320 KiB per ring position
//! × 1,536 = 480 MiB per sequence at most (336 KiB per token if the sliding layers were cached
//! over the whole context); the ring is lazily backed, so a conversation shorter than the ring
//! holds only the positions it wrote.
//!
//! The E-series' per-layer embeddings (`embedding_length_per_layer_input`) and cross-layer KV
//! sharing (`attention.shared_kv_layers`) and the 26B-A4B's MoE block are rejected in
//! `ArchSpec::from_gguf` (follow-ups).

use crate::arch::{ArchSpec, Gemma4Spec};
use crate::forward::{attend, per_head_norm, project, Attend, Rows, Scratch};
use crate::kv::KvCache;
use crate::weights::{mat, vec1};
use crate::Result;
use llmario_engine_core::GgmlType;
use llmario_engine_cpu::ops::add_inplace;
use llmario_engine_cpu::{rms_norm, QMat, ThreadPool};
use llmario_engine_formats::GgufFile;

pub struct Gemma4Layer<'a> {
    pub attn_norm: Vec<f32>,
    pub wq: QMat<'a>,
    pub wk: QMat<'a>,
    /// `None` on K=V layers (`attn_v` absent): V is the K projection.
    pub wv: Option<QMat<'a>>,
    pub wo: QMat<'a>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub post_attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub w_gate: QMat<'a>,
    pub w_up: QMat<'a>,
    pub w_down: QMat<'a>,
    pub post_ffn_norm: Vec<f32>,
    /// `layer_output_scale.weight` (one scalar), when present.
    pub out_scale: Option<f32>,
}

pub struct Gemma4Weights<'a> {
    pub layers: Vec<Gemma4Layer<'a>>,
    /// `rope_freqs.weight`: per-pair angle divisors of the global layers' RoPE
    /// (`[head_dim / 2]`), `None` when the file has none (plain NeoX RoPE then).
    pub rope_freqs: Option<Vec<f32>>,
    /// Unit gains for the weightless V norm (`[max head_dim]`).
    ones: Vec<f32>,
}

impl<'a> Gemma4Weights<'a> {
    pub fn load(f: &'a GgufFile, spec: &ArchSpec) -> Result<Gemma4Weights<'a>> {
        let d = spec.d_model;
        let mut layers = Vec::with_capacity(spec.n_layer as usize);
        for l in 0..spec.n_layer as usize {
            let p = |s: &str| format!("blk.{l}.{s}");
            let g = spec.attn_geom(l);
            let wv = if f.tensor(&p("attn_v.weight")).is_some() {
                Some(mat(f, &p("attn_v.weight"), d, g.v_dim())?)
            } else {
                None
            };
            let out_scale = if f.tensor(&p("layer_output_scale.weight")).is_some() {
                Some(vec1(f, &p("layer_output_scale.weight"), 1)?[0])
            } else {
                None
            };
            layers.push(Gemma4Layer {
                attn_norm: vec1(f, &p("attn_norm.weight"), d)?,
                wq: mat(f, &p("attn_q.weight"), d, g.q_dim())?,
                wk: mat(f, &p("attn_k.weight"), d, g.kv_dim())?,
                wv,
                wo: mat(f, &p("attn_output.weight"), g.attn_dim(), d)?,
                q_norm: vec1(f, &p("attn_q_norm.weight"), g.head_dim)?,
                k_norm: vec1(f, &p("attn_k_norm.weight"), g.head_dim)?,
                post_attn_norm: vec1(f, &p("post_attention_norm.weight"), d)?,
                ffn_norm: vec1(f, &p("ffn_norm.weight"), d)?,
                w_gate: mat(f, &p("ffn_gate.weight"), d, spec.n_ff)?,
                w_up: mat(f, &p("ffn_up.weight"), d, spec.n_ff)?,
                w_down: mat(f, &p("ffn_down.weight"), spec.n_ff, d)?,
                post_ffn_norm: vec1(f, &p("post_ffw_norm.weight"), d)?,
                out_scale,
            });
        }
        let rope_freqs = if f.tensor("rope_freqs.weight").is_some() {
            Some(vec1(f, "rope_freqs.weight", spec.head_dim / 2)?)
        } else {
            None
        };
        let max_hd = spec
            .head_dim_v
            .max(spec.gemma4.as_ref().map(|g| g.head_dim_v_swa).unwrap_or(0));
        Ok(Gemma4Weights {
            layers,
            rope_freqs,
            ones: vec![1.0; max_hd as usize],
        })
    }

    pub fn push_dtypes(&self, v: &mut Vec<GgmlType>) {
        for l in &self.layers {
            v.extend([
                l.wq.dtype,
                l.wk.dtype,
                l.wo.dtype,
                l.w_gate.dtype,
                l.w_up.dtype,
                l.w_down.dtype,
            ]);
            if let Some(wv) = &l.wv {
                v.push(wv.dtype);
            }
        }
    }
}

/// Run every layer over the embedded stacked rows in `s.x` (placement in `rows`), K/V appended
/// to `kv` (attention layer `l` for layer `l`: every Gemma 4 layer is an attention layer).
pub(crate) fn forward_layers(
    spec: &ArchSpec,
    w: &Gemma4Weights,
    pool: &ThreadPool,
    kv: &mut KvCache,
    rows: &Rows,
    s: &mut Scratch,
) {
    let g = spec.gemma4.as_ref().expect("Gemma 4 spec");
    let n = rows.n();
    let d = spec.d_model as usize;
    for v in &mut s.x[..n * d] {
        *v *= g.embed_scale;
    }
    for (l, layer) in w.layers.iter().enumerate() {
        attention(spec, g, w, layer, l, pool, kv, rows, s);
        ffn(spec, layer, pool, n, s);
        if let Some(sc) = layer.out_scale {
            for v in &mut s.x[..n * d] {
                *v *= sc;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn attention(
    spec: &ArchSpec,
    g: &Gemma4Spec,
    w: &Gemma4Weights,
    layer: &Gemma4Layer,
    l: usize,
    pool: &ThreadPool,
    kv: &mut KvCache,
    rows: &Rows,
    s: &mut Scratch,
) {
    let n = rows.n();
    let d = spec.d_model as usize;
    let geom = spec.attn_geom(l);
    let n_head = geom.n_head as usize;
    let n_kv = geom.n_kv_head as usize;
    let hd = geom.head_dim as usize;
    let hdv = geom.head_dim_v as usize;
    let q_dim = n_head * hd;
    let kv_dim = n_kv * hd;
    let v_dim = n_kv * hdv;
    let eps = spec.rms_eps;

    for t in 0..n {
        rms_norm(
            &s.x[t * d..(t + 1) * d],
            &layer.attn_norm,
            eps,
            &mut s.h[t * d..(t + 1) * d],
        );
    }
    project(pool, &layer.wq, &s.h, n, d, &mut s.q[..n * q_dim], None);
    project(pool, &layer.wk, &s.h, n, d, &mut s.k[..n * kv_dim], None);
    match &layer.wv {
        Some(wv) => project(pool, wv, &s.h, n, d, &mut s.v[..n * v_dim], None),
        // K=V: the value is the *raw* K projection (before `attn_k_norm` and RoPE).
        None => {
            debug_assert_eq!(kv_dim, v_dim);
            s.v[..n * v_dim].copy_from_slice(&s.k[..n * kv_dim]);
        }
    }

    // Per-layer RoPE: sliding layers use the SWA base over `rope_swa.dim` dims without factors;
    // global layers the main base over `rope.dim` dims with the `rope_freqs` divisors.
    let (theta, n_dims, ff) = match geom.window {
        Some(_) => (g.rope_swa.theta, g.rope_swa.dim as usize, None),
        None => (
            spec.rope.theta,
            spec.rope.dim as usize,
            w.rope_freqs.as_deref(),
        ),
    };
    let mut cs = Vec::with_capacity(n_dims);
    for t in 0..n {
        let (seq, pos_abs) = (rows.seq[t], rows.pos[t]);
        let pos = pos_abs as u32;
        let q = &mut s.q[t * q_dim..(t + 1) * q_dim];
        let k = &mut s.k[t * kv_dim..(t + 1) * kv_dim];
        let v = &mut s.v[t * v_dim..(t + 1) * v_dim];
        per_head_norm(q, hd, &layer.q_norm, eps);
        per_head_norm(k, hd, &layer.k_norm, eps);
        per_head_norm(v, hdv, &w.ones[..hdv], eps);
        rope_cache(&mut cs, n_dims, pos, theta, ff);
        rope_apply(q, n_head, hd, &cs);
        rope_apply(k, n_kv, hd, &cs);
        kv.store_k(seq, l, pos_abs, k);
        kv.store_v(seq, l, pos_abs, v);
    }
    attend(
        pool,
        kv,
        l,
        rows,
        &s.q[..n * q_dim],
        Attend {
            n_head,
            n_kv,
            hd,
            hdv,
            scale: g.attn_scale,
            window: geom.window.map(|w| w as usize),
        },
        &mut s.attn,
    );
    let attn_dim = n_head * hdv;
    project(
        pool,
        &layer.wo,
        &s.attn,
        n,
        attn_dim,
        &mut s.ffn[..n * d],
        None,
    );
    for t in 0..n {
        rms_norm(
            &s.ffn[t * d..(t + 1) * d],
            &layer.post_attn_norm,
            eps,
            &mut s.h[t * d..(t + 1) * d],
        );
        add_inplace(&mut s.x[t * d..(t + 1) * d], &s.h[t * d..(t + 1) * d]);
    }
}

fn ffn(spec: &ArchSpec, layer: &Gemma4Layer, pool: &ThreadPool, n: usize, s: &mut Scratch) {
    let d = spec.d_model as usize;
    let n_ff = spec.n_ff as usize;
    let eps = spec.rms_eps;
    for t in 0..n {
        rms_norm(
            &s.x[t * d..(t + 1) * d],
            &layer.ffn_norm,
            eps,
            &mut s.h[t * d..(t + 1) * d],
        );
    }
    project(
        pool,
        &layer.w_gate,
        &s.h,
        n,
        d,
        &mut s.gate[..n * n_ff],
        None,
    );
    project(pool, &layer.w_up, &s.h, n, d, &mut s.up[..n * n_ff], None);
    geglu_ggml_inplace(&mut s.gate[..n * n_ff], &s.up[..n * n_ff]);
    project(
        pool,
        &layer.w_down,
        &s.gate,
        n,
        n_ff,
        &mut s.ffn[..n * d],
        None,
    );
    for t in 0..n {
        rms_norm(
            &s.ffn[t * d..(t + 1) * d],
            &layer.post_ffn_norm,
            eps,
            &mut s.h[t * d..(t + 1) * d],
        );
        add_inplace(&mut s.x[t * d..(t + 1) * d], &s.h[t * d..(t + 1) * d]);
    }
}

/// NeoX RoPE over the first `n_dims` of each `head_dim`-wide head, exactly as ggml-cpu computes
/// it (`ggml_rope_cache_init` + `rotate_pairs<T>(n_dims, n_dims/2, …)`, `freq_scale` 1,
/// `ext_factor` 0, `attn_factor` 1): pair `i` = `(x[i], x[i + n_dims/2])`, angle
/// `θᵢ = pos · base^(−2/n_dims)^i / ff[i]` with `θ` accumulated by repeated multiplication.
#[cfg(test)]
pub(crate) fn rope_neox_ff(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    n_dims: usize,
    pos: u32,
    base: f32,
    ff: Option<&[f32]>,
) {
    let mut cs = Vec::new();
    rope_cache(&mut cs, n_dims, pos, base, ff);
    rope_apply(x, n_heads, head_dim, &cs);
}

/// The `(cos θᵢ, sin θᵢ)` pairs of one position (ggml's per-row rope cache), shared by every
/// head of Q and K at that position.
pub(crate) fn rope_cache(
    cs: &mut Vec<f32>,
    n_dims: usize,
    pos: u32,
    base: f32,
    ff: Option<&[f32]>,
) {
    debug_assert!(n_dims % 2 == 0);
    let theta_scale = base.powf(-2.0 / n_dims as f32);
    cs.clear();
    let mut theta = pos as f32;
    for i in 0..n_dims / 2 {
        let t = theta / ff.map(|f| f[i]).unwrap_or(1.0);
        // One expression so LLVM emits a single `sincosf` (as clang does for ggml's adjacent
        // `cosf`/`sinf` in `rope_yarn`); separate calls differ in the last bit on ~4 % of angles.
        let (sin, cos) = (t.sin(), t.cos());
        cs.push(cos);
        cs.push(sin);
        theta *= theta_scale;
    }
}

/// Rotate `n_heads` heads of `head_dim` floats with a [`rope_cache`] (NeoX pairing over the first
/// `cs.len()` dims; the rest of each head is left as is).
pub(crate) fn rope_apply(x: &mut [f32], n_heads: usize, head_dim: usize, cs: &[f32]) {
    let half = cs.len() / 2;
    debug_assert!(2 * half <= head_dim);
    for head in x[..n_heads * head_dim].chunks_exact_mut(head_dim) {
        for i in 0..half {
            let (cos, sin) = (cs[2 * i], cs[2 * i + 1]);
            let x0 = head[i];
            let x1 = head[i + half];
            head[i] = x0 * cos - x1 * sin;
            head[i + half] = x0 * sin + x1 * cos;
        }
    }
}

/// `gate[i] = gelu(gate[i]) * up[i]` with ggml-cpu's fp16-table GELU (see the module doc).
pub(crate) fn geglu_ggml_inplace(gate: &mut [f32], up: &[f32]) {
    let table = gelu_table();
    for (g, &u) in gate.iter_mut().zip(up) {
        let x = *g;
        let y = if x <= -10.0 {
            0.0
        } else if x >= 10.0 {
            x
        } else {
            table[f32_to_f16(x) as usize]
        };
        *g = y * u;
    }
}

/// ggml-cpu's `ggml_table_gelu_f16` (built in `ggml_cpu_init`): entry `h` is
/// `fp16(gelu_tanh(f32(h)))` for every binary16 bit pattern `h`, stored widened to f32.
fn gelu_table() -> &'static [f32] {
    static TABLE: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        (0..=u16::MAX)
            .map(|h| f16_round(gelu_tanh(f16_to_f32(h))))
            .collect()
    })
}

/// `ggml_gelu_f32`: `0.5·x·(1 + tanh(√(2/π)·x·(1 + 0.044715·x²)))`.
#[inline]
fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6; // ggml: 0.79788456080286535587989211986876f
    const GELU_COEF_A: f32 = 0.044715;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x)).tanh())
}

/// ggml-cpu `ggml_vec_gelu_f32` / `ggml_vec_geglu_f32` under `GGML_GELU_FP16`: `0` for
/// `x ≤ −10`, `x` for `x ≥ 10`, otherwise `fp16(gelu_tanh(f32(fp16(x))))` (computed directly;
/// [`geglu_ggml_inplace`] uses the equivalent table).
#[cfg(test)]
#[inline]
pub(crate) fn gelu_ggml_cpu(x: f32) -> f32 {
    if x <= -10.0 {
        0.0
    } else if x >= 10.0 {
        x
    } else {
        f16_round(gelu_tanh(f16_round(x)))
    }
}

/// `logits[i] = c · tanh(logits[i] / c)` (llama.cpp: `ggml_scale(1/c)`, `ggml_tanh`,
/// `ggml_scale(c)`); a no-op when `c <= 0`.
pub(crate) fn softcap_inplace(logits: &mut [f32], c: f32) {
    if c <= 0.0 {
        return;
    }
    let inv = 1.0 / c;
    for v in logits.iter_mut() {
        *v = (*v * inv).tanh() * c;
    }
}

/// Round an f32 to the nearest IEEE binary16 value (ties to even) and back, mirroring
/// `GGML_CPU_FP32_TO_FP16` → `GGML_CPU_FP16_TO_FP32`. Values beyond the f16 range become ±inf
/// (not reached by [`gelu_ggml_cpu`], whose inputs are clamped to `(−10, 10)`).
pub(crate) fn f16_round(x: f32) -> f32 {
    f16_to_f32(f32_to_f16(x))
}

fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        // inf / nan
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflow → inf
    }
    if e <= 0 {
        if e < -10 {
            return sign; // underflow → ±0
        }
        // Subnormal half: shift the (implicit-1) mantissa right with round-to-nearest-even.
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let half_bits = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        let round_up = rem > halfway || (rem == halfway && (half_bits & 1) == 1);
        return sign | (half_bits + round_up as u32) as u16;
    }
    let half_bits = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    let round_up = rem > 0x1000 || (rem == 0x1000 && (half_bits & 1) == 1);
    // A mantissa carry on round-up correctly bumps the exponent (and may reach inf).
    sign | (half_bits + round_up as u32) as u16
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal: normalise.
            let mut e = 127 - 15 + 1;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::{Model, Scratch};
    use crate::kv::{KvOptions, SWA_RING_BATCH};
    use crate::weights::Weights;
    use crate::{ArchSpec, BlockKind, Family, KvCache};
    use llmario_engine_formats::gguf::writer::GgufWriter;
    use llmario_engine_formats::MetaValue;

    /// A tiny random `gemma4` model: four layers, pattern [sliding, sliding, global, sliding],
    /// sliding layers 2 KV heads × 8 (window 4), the global layer 1 KV head × 16 with no `attn_v`
    /// (K=V) and the proportional-RoPE `rope_freqs` ([1, 1, 1e30 × 6]: 2 of 8 pairs rotated).
    fn tiny_gemma4_model(dir: &std::path::Path) -> std::path::PathBuf {
        let d = 32u64;
        let n_head = 4u64;
        let n_ff = 48u64;
        let vocab = 64u64;
        let n_layer = 4u64;
        let swa = [true, true, false, true];
        let hd_swa = 8u64;
        let hd_full = 16u64;
        let n_kv_swa = 2u64;
        let n_kv_full = 1u64;
        let n_swa = 4u32;
        let f32s = |n: u64, seed: u64| -> Vec<u8> {
            (0..n)
                .flat_map(|i| {
                    let v = (((i * 2654435761 + seed * 97) % 1000) as f32 / 1000.0 - 0.5) * 0.3;
                    v.to_le_bytes()
                })
                .collect()
        };
        let consts = |n: u64, c: f32| -> Vec<u8> { (0..n).flat_map(|_| c.to_le_bytes()).collect() };
        let mut w = GgufWriter::new();
        w.meta("general.architecture", MetaValue::Str("gemma4".into()))
            .meta("gemma4.block_count", MetaValue::U32(n_layer as u32))
            .meta("gemma4.embedding_length", MetaValue::U32(d as u32))
            .meta("gemma4.attention.head_count", MetaValue::U32(n_head as u32))
            .meta(
                "gemma4.attention.head_count_kv",
                MetaValue::Array(
                    swa.iter()
                        .map(|&s| MetaValue::U32(if s { n_kv_swa } else { n_kv_full } as u32))
                        .collect(),
                ),
            )
            .meta(
                "gemma4.attention.sliding_window_pattern",
                MetaValue::Array(swa.iter().map(|&s| MetaValue::Bool(s)).collect()),
            )
            .meta("gemma4.attention.sliding_window", MetaValue::U32(n_swa))
            .meta(
                "gemma4.attention.key_length",
                MetaValue::U32(hd_full as u32),
            )
            .meta(
                "gemma4.attention.value_length",
                MetaValue::U32(hd_full as u32),
            )
            .meta(
                "gemma4.attention.key_length_swa",
                MetaValue::U32(hd_swa as u32),
            )
            .meta(
                "gemma4.attention.value_length_swa",
                MetaValue::U32(hd_swa as u32),
            )
            .meta("gemma4.attention.shared_kv_layers", MetaValue::U32(0))
            .meta("gemma4.embedding_length_per_layer_input", MetaValue::U32(0))
            .meta("gemma4.feed_forward_length", MetaValue::U32(n_ff as u32))
            .meta("gemma4.vocab_size", MetaValue::U32(vocab as u32))
            .meta("gemma4.context_length", MetaValue::U32(64))
            .meta("gemma4.rope.freq_base", MetaValue::F32(1_000_000.0))
            .meta("gemma4.rope.freq_base_swa", MetaValue::F32(10_000.0))
            .meta(
                "gemma4.rope.dimension_count",
                MetaValue::U32(hd_full as u32),
            )
            .meta(
                "gemma4.rope.dimension_count_swa",
                MetaValue::U32(hd_swa as u32),
            )
            .meta("gemma4.final_logit_softcapping", MetaValue::F32(30.0))
            .meta(
                "gemma4.attention.layer_norm_rms_epsilon",
                MetaValue::F32(1e-6),
            );
        w.tensor(
            "token_embd.weight",
            &[d, vocab],
            GgmlType::F32,
            f32s(d * vocab, 1),
        );
        w.tensor("output_norm.weight", &[d], GgmlType::F32, consts(d, 1.0));
        let mut ff: Vec<u8> = Vec::new();
        for i in 0..hd_full / 2 {
            ff.extend((if i < 2 { 1.0f32 } else { 1e30 }).to_le_bytes());
        }
        w.tensor("rope_freqs.weight", &[hd_full / 2], GgmlType::F32, ff);
        for l in 0..n_layer {
            let p = |s: &str| format!("blk.{l}.{s}");
            let (hd, n_kv) = if swa[l as usize] {
                (hd_swa, n_kv_swa)
            } else {
                (hd_full, n_kv_full)
            };
            w.tensor(&p("attn_norm.weight"), &[d], GgmlType::F32, consts(d, 1.0));
            w.tensor(
                &p("attn_q.weight"),
                &[d, n_head * hd],
                GgmlType::F32,
                f32s(d * n_head * hd, 10 + l),
            );
            w.tensor(
                &p("attn_k.weight"),
                &[d, n_kv * hd],
                GgmlType::F32,
                f32s(d * n_kv * hd, 20 + l),
            );
            if swa[l as usize] {
                w.tensor(
                    &p("attn_v.weight"),
                    &[d, n_kv * hd],
                    GgmlType::F32,
                    f32s(d * n_kv * hd, 30 + l),
                );
            }
            w.tensor(
                &p("attn_q_norm.weight"),
                &[hd],
                GgmlType::F32,
                f32s(hd, 80 + l),
            );
            w.tensor(
                &p("attn_k_norm.weight"),
                &[hd],
                GgmlType::F32,
                f32s(hd, 90 + l),
            );
            w.tensor(
                &p("attn_output.weight"),
                &[n_head * hd, d],
                GgmlType::F32,
                f32s(n_head * hd * d, 40 + l),
            );
            w.tensor(
                &p("post_attention_norm.weight"),
                &[d],
                GgmlType::F32,
                consts(d, 1.0),
            );
            w.tensor(&p("ffn_norm.weight"), &[d], GgmlType::F32, consts(d, 1.0));
            w.tensor(
                &p("ffn_gate.weight"),
                &[d, n_ff],
                GgmlType::F32,
                f32s(d * n_ff, 50 + l),
            );
            w.tensor(
                &p("ffn_up.weight"),
                &[d, n_ff],
                GgmlType::F32,
                f32s(d * n_ff, 60 + l),
            );
            w.tensor(
                &p("ffn_down.weight"),
                &[n_ff, d],
                GgmlType::F32,
                f32s(n_ff * d, 70 + l),
            );
            w.tensor(
                &p("post_ffw_norm.weight"),
                &[d],
                GgmlType::F32,
                consts(d, 1.0),
            );
            w.tensor(
                &p("layer_output_scale.weight"),
                &[1],
                GgmlType::F32,
                consts(1, 0.9),
            );
        }
        let p = dir.join("tiny_gemma4.gguf");
        std::fs::write(&p, w.to_bytes()).unwrap();
        p
    }

    fn run(
        m: &Model,
        pool: &ThreadPool,
        kv: &mut KvCache,
        toks: &[u32],
        n_batch: usize,
    ) -> Vec<f32> {
        let mut s = Scratch::new(&m.spec, n_batch);
        m.forward(pool, kv, toks, &mut s).to_vec()
    }

    #[test]
    fn gemma4_spec_geometry_and_cache_sizing() {
        let dir = tempfile::tempdir().unwrap();
        let f = GgufFile::open(&tiny_gemma4_model(dir.path())).unwrap();
        let spec = ArchSpec::from_gguf(&f).unwrap();
        assert_eq!(spec.family, Family::Gemma4);
        assert_eq!(spec.blocks, vec![BlockKind::Attention; 4]);
        assert!(spec.qk_norm && spec.tied_embeddings && spec.gdn.is_none());
        let g = spec.gemma4.as_ref().unwrap();
        assert_eq!(g.swa_layers, vec![true, true, false, true]);
        assert_eq!(g.n_kv_head_layers, vec![2, 2, 1, 2]);
        assert_eq!((g.n_swa, g.head_dim_swa, g.head_dim_v_swa), (4, 8, 8));
        assert_eq!((g.rope_swa.theta, g.rope_swa.dim), (10_000.0, 8));
        assert_eq!((spec.rope.theta, spec.rope.dim), (1_000_000.0, 16));
        assert_eq!(
            (spec.n_kv_head, spec.head_dim, spec.head_dim_v),
            (1, 16, 16)
        );
        assert_eq!(g.final_logit_softcap, 30.0);
        assert_eq!(g.embed_scale, 32f32.sqrt());
        assert!(spec.has_window_layers());
        // Per-layer geometry and the scratch maxima.
        let s0 = spec.attn_geom(0);
        let s2 = spec.attn_geom(2);
        assert_eq!((s0.q_dim(), s0.kv_dim(), s0.window), (32, 16, Some(4)));
        assert_eq!((s2.q_dim(), s2.kv_dim(), s2.window), (64, 16, None));
        assert_eq!(
            (spec.max_q_dim(), spec.max_kv_dim(), spec.max_attn_dim()),
            (64, 16, 64)
        );

        // Cache: the global layer holds max_ctx positions, each sliding layer a ring of
        // min(max_ctx, n_swa + n_batch).
        let kv = KvCache::with_options(&spec, KvOptions::new(64).ring_batch(4));
        assert_eq!(kv.n_layers(), 4);
        assert_eq!(kv.layer(0).cap, 8);
        assert_eq!(kv.layer(0).window, Some(4));
        assert_eq!(kv.layer(2).cap, 64);
        assert_eq!(kv.layer(2).window, None);
        assert_eq!(kv.slot(0, 13), 5);
        assert_eq!(kv.slot(2, 13), 13);
        assert_eq!(kv.max_batch(), 5);
        // Marginal per-token bytes count the global layer only; the rings are fixed.
        assert_eq!(spec.kv_bytes_per_token(4.0), (16 + 16) * 4);
        assert_eq!(spec.window_bytes(4.0, 64, 4), 3 * (16 + 16) * 4 * 8);
        assert_eq!(kv.layout.paged_bytes_per_token(), (16 + 16) * 2);
        // Paged global layer: 64 positions = 2 blocks of 32, plus the copy-on-write spare;
        // three rings of 8 positions.
        assert_eq!(
            kv.layout.reserved_bytes(),
            (2 + 1) * 32 * (16 + 16) * 2 + 3 * (16 + 16) * 2 * 8
        );
        // Default headroom: 4 + 512 > 64, so the rings do not wrap and no batch limit applies.
        let kv = KvCache::new(&spec, 64);
        assert_eq!(kv.layer(0).cap, 64);
        assert_eq!(kv.max_batch(), usize::MAX);
        assert_eq!(
            KvCache::bytes(&spec, 64),
            (2 + 1) * 32 * (16 + 16) * 2 + 3 * (16 + 16) * 2 * 64
        );
        // Scratch accounting matches the allocation with the per-layer maxima.
        let s = Scratch::new(&spec, 8);
        assert_eq!(
            Scratch::bytes(&spec, 8),
            (s.x.len()
                + s.h.len()
                + s.q.len()
                + s.k.len()
                + s.v.len()
                + s.attn.len()
                + s.gate.len()
                + s.up.len()
                + s.ffn.len()
                + s.last.len()
                + s.logits.len()) as u64
                * 4
        );
    }

    #[test]
    fn gemma4_weights_resolve_k_eq_v_and_rope_freqs() {
        let dir = tempfile::tempdir().unwrap();
        let f = GgufFile::open(&tiny_gemma4_model(dir.path())).unwrap();
        let m = Model::load(&f).unwrap();
        let w = m.weights.gemma4.as_ref().unwrap();
        assert!(w.layers[0].wv.is_some());
        assert!(w.layers[2].wv.is_none());
        assert_eq!(w.layers[2].wq.rows, 64);
        assert_eq!(w.layers[2].wk.rows, 16);
        assert_eq!(w.layers[2].wo.cols, 64);
        assert_eq!(w.layers[0].out_scale, Some(0.9));
        assert_eq!(w.rope_freqs.as_ref().unwrap().len(), 8);
        assert!(m.weights.layers.is_empty() && m.weights.hybrid.is_none());
    }

    /// Prefill of 12 tokens at once (three ring-bounded chunks of ≤ 5) equals token-by-token
    /// decode, across window evictions in the sliding layers.
    #[test]
    fn gemma4_prefill_equals_token_by_token_through_the_window() {
        let dir = tempfile::tempdir().unwrap();
        let f = GgufFile::open(&tiny_gemma4_model(dir.path())).unwrap();
        let m = Model::load(&f).unwrap();
        let pool = ThreadPool::new(3);
        let toks = [3u32, 17, 5, 42, 9, 61, 2, 33, 12, 7, 50, 28];
        let ring4 = KvOptions::new(64).ring_batch(4);
        let mut kv_a = KvCache::with_options(&m.spec, ring4);
        let la = run(&m, &pool, &mut kv_a, &toks, toks.len());
        let mut kv_b = KvCache::with_options(&m.spec, ring4);
        let mut lb = vec![];
        for &t in &toks {
            lb = run(&m, &pool, &mut kv_b, &[t], 1);
        }
        assert_eq!(kv_a.len(0), toks.len());
        assert_eq!(kv_b.len(0), toks.len());
        let max_abs = la.iter().map(|v| v.abs()).fold(0f32, f32::max);
        assert!(
            max_abs > 0.0 && max_abs <= 30.0,
            "soft-capped logits, got {max_abs}"
        );
        for (i, (a, b)) in la.iter().zip(&lb).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * max_abs.max(1.0),
                "logit {i}: {a} vs {b}"
            );
        }
        // A cache whose rings never wrap (default headroom) gives the same answer: the ring
        // layout is transparent.
        let mut kv_c = KvCache::new(&m.spec, 64);
        let lc = run(&m, &pool, &mut kv_c, &toks, toks.len());
        assert_eq!(la, lc);
        // Prefill then a decode step past the window, both layouts.
        let next = [19u32];
        let ld_a = run(&m, &pool, &mut kv_a, &next, 1);
        let ld_c = run(&m, &pool, &mut kv_c, &next, 1);
        assert_eq!(ld_a, ld_c);
    }

    /// The sliding mask matters: the same weights with the window disabled (`n_swa` beyond the
    /// context) produce different logits once the prompt exceeds `n_swa`.
    #[test]
    fn gemma4_window_mask_changes_the_result_past_n_swa() {
        let dir = tempfile::tempdir().unwrap();
        let f = GgufFile::open(&tiny_gemma4_model(dir.path())).unwrap();
        let m = Model::load(&f).unwrap();
        let pool = ThreadPool::new(2);
        let toks = [3u32, 17, 5, 42, 9, 61, 2, 33, 12, 7];
        let mut kv = KvCache::new(&m.spec, 64);
        let with_window = run(&m, &pool, &mut kv, &toks, toks.len());

        let mut spec_global = m.spec.clone();
        spec_global.gemma4.as_mut().unwrap().n_swa = 1 << 20;
        let m2 = Model {
            spec: spec_global,
            weights: Weights::load(&f, &m.spec).unwrap(),
        };
        let mut kv2 = KvCache::new(&m2.spec, 64);
        let without = run(&m2, &pool, &mut kv2, &toks, toks.len());
        let diff = with_window
            .iter()
            .zip(&without)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(diff > 1e-3, "window had no effect (max diff {diff})");

        // Up to n_swa tokens the two agree exactly (nothing is masked yet).
        let mut kv3 = KvCache::new(&m.spec, 64);
        let mut kv4 = KvCache::new(&m2.spec, 64);
        let short = &toks[..4];
        assert_eq!(
            run(&m, &pool, &mut kv3, short, 4),
            run(&m2, &pool, &mut kv4, short, 4)
        );
    }

    /// `attend` with a window: a decode step at position 6 with window 4 sees exactly positions
    /// 3..=6; a key that would dominate the softmax at position 2 (evicted) must be ignored, and
    /// the key at position 3 (the oldest visible one) must count.
    #[test]
    fn attend_window_mask_boundary() {
        let dir = tempfile::tempdir().unwrap();
        let f = GgufFile::open(&tiny_gemma4_model(dir.path())).unwrap();
        let spec = ArchSpec::from_gguf(&f).unwrap();
        let pool = ThreadPool::new(1);
        let geom = spec.attn_geom(0); // sliding layer: 4 heads, 2 KV heads × 8, window 4
        let (n_head, n_kv, hd) = (
            geom.n_head as usize,
            geom.n_kv_head as usize,
            geom.head_dim as usize,
        );
        for n_batch in [4usize, 512] {
            let mut kv = KvCache::with_options(&spec, KvOptions::new(64).ring_batch(n_batch));
            kv.reserve(0, 7).unwrap();
            // Keys: position p has key e_{p mod 8} scaled so q·k is huge where they match; values
            // are the position number broadcast.
            for p in 0..7 {
                let mut k = vec![0f32; kv.layer(0).kv_dim];
                for kvh in 0..n_kv {
                    k[kvh * hd + (p % hd)] = 50.0;
                }
                kv.store_k(0, 0, p, &k);
                kv.store_v(0, 0, p, &vec![p as f32; kv.layer(0).v_dim]);
            }
            kv.seq_mut(0).len = 7;
            let rows = crate::forward::Rows::single(0, 6, 1);
            // Query of position 6 pointing at the evicted position 2 and (less strongly) at the
            // oldest visible position 3.
            let mut q = vec![0f32; n_head * hd];
            for h in 0..n_head {
                q[h * hd + 2] = 1.0;
                q[h * hd + 3] = 0.5;
            }
            let mut out = vec![0f32; n_head * hd];
            attend(
                &pool,
                &kv,
                0,
                &rows,
                &q,
                Attend {
                    n_head,
                    n_kv,
                    hd,
                    hdv: hd,
                    scale: 1.0,
                    window: Some(4),
                },
                &mut out,
            );
            // softmax over positions 3..=6 with score 25 at p=3 and 0 elsewhere → ≈ value 3.
            assert!(
                (out[0] - 3.0).abs() < 1e-6,
                "n_batch {n_batch}: windowed output {} (expected ≈ 3; 2 would mean the evicted \
                 key leaked)",
                out[0]
            );
            if n_batch == 512 {
                // Without the window the evicted position 2 (score 50) dominates instead.
                attend(
                    &pool,
                    &kv,
                    0,
                    &rows,
                    &q,
                    Attend {
                        n_head,
                        n_kv,
                        hd,
                        hdv: hd,
                        scale: 1.0,
                        window: None,
                    },
                    &mut out,
                );
                assert!((out[0] - 2.0).abs() < 1e-6, "unwindowed output {}", out[0]);
            }
        }
    }

    /// Snapshots carry the window rings, including wrapped ones.
    #[test]
    fn gemma4_snapshot_round_trip_with_wrapped_rings() {
        let dir = tempfile::tempdir().unwrap();
        let f = GgufFile::open(&tiny_gemma4_model(dir.path())).unwrap();
        let m = Model::load(&f).unwrap();
        let pool = ThreadPool::new(2);
        let opts = KvOptions::new(64).ring_batch(4);
        let toks = [3u32, 17, 5, 42, 9, 61, 2, 33, 12, 7, 50, 28];
        let mut a = KvCache::with_options(&m.spec, opts);
        run(&m, &pool, &mut a, &toks, toks.len());
        let snap = a.export(0);
        assert!(a.layout.ring_bytes > 0 && !a.snapshot_trimmable());
        let want = run(&m, &pool, &mut a, &[19], 1);
        let mut b = KvCache::with_options(&m.spec, opts);
        b.import(0, &snap).unwrap();
        assert_eq!(run(&m, &pool, &mut b, &[19], 1), want);
    }

    #[test]
    fn truncate_resets_only_when_a_ring_lost_positions() {
        let dir = tempfile::tempdir().unwrap();
        let f = GgufFile::open(&tiny_gemma4_model(dir.path())).unwrap();
        let spec = ArchSpec::from_gguf(&f).unwrap();
        // Ring of 8 with window 4: rewinding by ≤ 5 keeps the needed positions, by 6 does not.
        let mut kv = KvCache::with_options(&spec, KvOptions::new(64).ring_batch(4));
        kv.seq_mut(0).len = 20;
        kv.truncate(0, 15);
        assert_eq!(kv.len(0), 15);
        kv.truncate(0, 9);
        assert_eq!(kv.len(0), 0);
        // Rings that never wrap never lose anything.
        let mut kv = KvCache::new(&spec, 64);
        kv.seq_mut(0).len = 20;
        kv.truncate(0, 1);
        assert_eq!(kv.len(0), 1);
    }

    #[test]
    fn gelu_and_softcap_match_ggml_semantics() {
        // f16 round trip: exact on representable values, rounds to nearest even otherwise.
        assert_eq!(f16_round(1.0), 1.0);
        assert_eq!(f16_round(0.1), 0.099_975_586);
        assert_eq!(f16_round(1.0 + 1.0 / 2048.0), 1.0); // tie → even
        assert_eq!(f16_round(1.0 + 3.0 / 2048.0), 1.0 + 2.0 / 1024.0);
        assert_eq!(f16_round(-2.5), -2.5);
        assert_eq!(f16_round(1e-8), 0.0);
        assert_eq!(f16_round(6.0e-8), 5.960_464_5e-8); // smallest subnormal
        assert_eq!(f16_round(70000.0), f32::INFINITY);
        // Clamps and the table rounding.
        assert_eq!(gelu_ggml_cpu(-10.0), 0.0);
        assert_eq!(gelu_ggml_cpu(12.0), 12.0);
        let x = 0.7f32;
        let exact = llmario_engine_cpu::gelu(x);
        let g = gelu_ggml_cpu(x);
        assert!(
            (g - exact).abs() < 1e-3 && g == f16_round(g),
            "{g} vs {exact}"
        );
        // The table path equals the direct path for every input (sweep across the f16 range,
        // both clamps and values between f16 grid points).
        let xs: Vec<f32> = (0..200_000)
            .map(|i| -12.0 + 24.0 * i as f32 / 200_000.0)
            .chain([-10.0, 10.0, -9.999, 9.999, 0.0, -0.0, 1e-6, -1e-6])
            .collect();
        let mut g = xs.clone();
        geglu_ggml_inplace(&mut g, &vec![1.0; xs.len()]);
        for (x, y) in xs.iter().zip(&g) {
            assert_eq!(y.to_bits(), gelu_ggml_cpu(*x).to_bits(), "x = {x}");
        }
        let mut l = vec![0.0, 30.0, -300.0, 1e9];
        softcap_inplace(&mut l, 30.0);
        assert_eq!(l[0], 0.0);
        assert!((l[1] - 30.0 * 1f32.tanh()).abs() < 1e-6);
        assert!((l[2] + 30.0).abs() < 1e-3, "{}", l[2]); // 30·tanh(−10) rounds to −30 in f32
        assert_eq!(l[3], 30.0);
        let mut l = vec![1.0, 2.0];
        softcap_inplace(&mut l, 0.0);
        assert_eq!(l, vec![1.0, 2.0]);
    }

    /// The ggml-faithful NeoX RoPE equals the CPU crate's `rope` when no factors are involved
    /// (up to the iterative-vs-pow angle rounding), and a 1e30 factor leaves a pair untouched.
    #[test]
    fn rope_neox_ff_matches_reference_and_factors_freeze_pairs() {
        use llmario_engine_cpu::ops::{rope, RopeKind, RopeParams};
        let x: Vec<f32> = (0..16).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut a = x.clone();
        let mut b = x.clone();
        rope_neox_ff(&mut a, 1, 16, 16, 37, 10_000.0, None);
        rope(
            &mut b,
            1,
            37,
            &RopeParams {
                kind: RopeKind::Neox,
                head_dim: 16,
                rot_dim: 16,
                theta: 10_000.0,
                freq_scale: 1.0,
                attn_factor: 1.0,
            },
        );
        for (p, q) in a.iter().zip(&b) {
            assert!((p - q).abs() < 1e-4, "{p} vs {q}");
        }
        let ff = [1.0, 1.0, 1e30, 1e30, 1e30, 1e30, 1e30, 1e30];
        let mut c = x.clone();
        rope_neox_ff(&mut c, 1, 16, 16, 1000, 1_000_000.0, Some(&ff));
        for i in 2..8 {
            assert_eq!(c[i], x[i]);
            assert_eq!(c[i + 8], x[i + 8]);
        }
        assert_ne!(c[0], x[0]);
        assert_ne!(c[8], x[8]);
        // Partial rotation over the first n_dims only.
        let mut d = x.clone();
        rope_neox_ff(&mut d, 1, 16, 8, 5, 10_000.0, None);
        assert_eq!(&d[8..], &x[8..]);
        assert_ne!(&d[..8], &x[..8]);
    }

    /// Real model (gated on `LLMARIO_TEST_GGUF`, comma-separated; only `gemma4` files are used):
    /// prefill vs. token-by-token consistency (a) with the model's own spec on a short prompt
    /// (rings do not wrap, no window masking), and (b) with the same weights but `n_swa` shrunk to
    /// 4 and a ring of 7 positions, so the real geometry (512-wide K=V global heads, 256-wide
    /// sliding heads) runs through ring wrap-around, window masking and chunked prefill.
    #[test]
    fn real_gemma4_prefill_equals_token_by_token() {
        let Ok(models) = std::env::var("LLMARIO_TEST_GGUF") else {
            eprintln!("LLMARIO_TEST_GGUF not set; skipping");
            return;
        };
        let check = |m: &Model, pool: &ThreadPool, toks: &[u32], ring_batch: usize| {
            let opts = KvOptions::new(64).ring_batch(ring_batch);
            let mut kv_a = KvCache::with_options(&m.spec, opts);
            let la = run(m, pool, &mut kv_a, toks, toks.len());
            let mut kv_b = KvCache::with_options(&m.spec, opts);
            let mut lb = vec![];
            for &t in toks {
                lb = run(m, pool, &mut kv_b, &[t], 1);
            }
            let max_abs = la.iter().map(|v| v.abs()).fold(0f32, f32::max);
            for (i, (a, b)) in la.iter().zip(&lb).enumerate() {
                assert!(
                    (a - b).abs() <= 2e-3 * max_abs.max(1.0),
                    "logit {i}: {a} vs {b} (max |logit| {max_abs})"
                );
            }
            (kv_a.max_batch(), la)
        };
        for path in models.split(',').filter(|s| !s.is_empty()) {
            let f = GgufFile::open(std::path::Path::new(path)).unwrap();
            if f.architecture() != Some("gemma4") {
                continue;
            }
            let m = Model::load(&f).unwrap();
            let pool = ThreadPool::new(8);
            // "<bos>The capital of France is Paris. The capital of France" (Gemma 4 tokenizer,
            // ids checked against llama-tokenize).
            let toks = [
                2u32, 818, 5279, 529, 7001, 563, 9079, 236761, 818, 5279, 529, 7001,
            ];
            let (mb, _) = check(&m, &pool, &toks, SWA_RING_BATCH);
            assert_eq!(mb, usize::MAX, "a 64-position cache never wraps");

            let mut spec = m.spec.clone();
            spec.gemma4.as_mut().unwrap().n_swa = 4;
            let small = Model {
                weights: Weights::load(&f, &spec).unwrap(),
                spec,
            };
            let (mb, windowed) = check(&small, &pool, &toks, 3);
            assert_eq!(mb, 4, "ring of 7 with window 4 → chunks of 4");
            // The window changes the answer (12 tokens > 4), so the mask really applied.
            let mut kv = KvCache::new(&m.spec, 64);
            let full = run(&m, &pool, &mut kv, &toks, toks.len());
            let diff = full
                .iter()
                .zip(&windowed)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(diff > 1e-2, "window 4 had no effect (max diff {diff})");
            eprintln!("{path}: prefill == stepwise (full spec and n_swa = 4 ring)");
        }
    }
}
