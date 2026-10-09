//! The Qwen3.5 / Qwen3-Next hybrid family: Gated DeltaNet layers interleaved with gated
//! full-attention layers (Architecture §6.7, §8.1). Weights views and the per-layer forward
//! passes live here; the DeltaNet kernels are in `gdn.rs`; the dense attention core, projections
//! and scratch come from `forward.rs`.
//!
//! Layer structure (llama.cpp `src/models/qwen35.cpp`, graph constructor):
//!
//! ```text
//! h   = rms_norm(x, attn_norm)
//! x  += mixer(h)                       // DeltaNet (gdn.rs) or gated attention (below)
//! h   = rms_norm(x, post_attention_norm)
//! x  += W_down (silu(W_gate h) ⊙ W_up h)
//! ```
//!
//! Gated attention layer (`build_layer_attn`): `attn_q` is `[d_model → n_head · 2 · head_dim]`,
//! viewed per head as `(q[head_dim], gate[head_dim])`; `q` gets the per-head RMSNorm
//! `attn_q_norm`, `k` gets `attn_k_norm`; both get RoPE (NeoX pairs over the first `n_rot` dims,
//! see `arch.rs` for why mRoPE reduces to that for text); softmax attention with scale
//! `1/sqrt(head_dim)`; the result is multiplied by `sigmoid(gate)` element-wise and projected by
//! `attn_output` (`[n_head · head_dim → d_model]`). Verified against HF `Qwen3NextAttention`:
//! `torch.chunk(q_proj(x).view(..., head_dim * 2), 2, dim=-1)` then
//! `attn_output * torch.sigmoid(gate)` before `o_proj`.

use crate::arch::{ArchSpec, BlockKind, GdnSpec};
use crate::forward::{attend, per_head_norm, project, Attend, Scratch};
use crate::gdn::{self, GdnDims};
use crate::kv::KvCache;
use crate::weights::{mat, vec1, vecn};
use crate::Result;
use llmario_engine_core::GgmlType;
use llmario_engine_cpu::ops::{add_inplace, swiglu_inplace, RopeParams};
use llmario_engine_cpu::{rms_norm, rope, QMat, ThreadPool};
use llmario_engine_formats::GgufFile;

pub struct AttnMixer<'a> {
    /// `[d_model → 2 · n_head · head_dim]`, `(q, gate)` interleaved per head.
    pub wq: QMat<'a>,
    pub wk: QMat<'a>,
    pub wv: QMat<'a>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub wo: QMat<'a>,
}

pub struct GdnMixer<'a> {
    /// `[d_model → conv_dim]` = `[q | k | v]`.
    pub wqkv: QMat<'a>,
    /// `[d_model → value_dim]`, the output gate `z`.
    pub wz: QMat<'a>,
    /// `[conv_dim][d_conv]`.
    pub conv: Vec<f32>,
    /// `ssm_a = −exp(A_log)`, one per V head.
    pub a: Vec<f32>,
    /// `ssm_dt.bias`, one per V head.
    pub dt_bias: Vec<f32>,
    /// `[d_model → n_v_heads]` each.
    pub w_alpha: QMat<'a>,
    pub w_beta: QMat<'a>,
    /// `ssm_norm`, `[head_v]`.
    pub norm: Vec<f32>,
    /// `[value_dim → d_model]`.
    pub wout: QMat<'a>,
}

pub enum Mixer<'a> {
    Attention(AttnMixer<'a>),
    DeltaNet(GdnMixer<'a>),
}

pub struct HybridLayer<'a> {
    pub attn_norm: Vec<f32>,
    pub mixer: Mixer<'a>,
    /// `post_attention_norm`: the pre-FFN norm.
    pub post_norm: Vec<f32>,
    pub w_gate: QMat<'a>,
    pub w_up: QMat<'a>,
    pub w_down: QMat<'a>,
}

pub struct HybridWeights<'a> {
    pub layers: Vec<HybridLayer<'a>>,
}

impl<'a> HybridWeights<'a> {
    pub fn load(f: &'a GgufFile, spec: &ArchSpec) -> Result<HybridWeights<'a>> {
        let g = spec.gdn.as_ref().expect("hybrid family carries a GdnSpec");
        let d = spec.d_model;
        let mut layers = Vec::with_capacity(spec.n_layer as usize);
        for (l, kind) in spec.blocks.iter().enumerate() {
            let p = |s: &str| format!("blk.{l}.{s}");
            let mixer = match kind {
                BlockKind::Attention => Mixer::Attention(AttnMixer {
                    wq: mat(f, &p("attn_q.weight"), d, 2 * spec.q_dim())?,
                    wk: mat(f, &p("attn_k.weight"), d, spec.kv_dim())?,
                    wv: mat(f, &p("attn_v.weight"), d, spec.v_dim())?,
                    q_norm: vec1(f, &p("attn_q_norm.weight"), spec.head_dim)?,
                    k_norm: vec1(f, &p("attn_k_norm.weight"), spec.head_dim)?,
                    wo: mat(
                        f,
                        &p("attn_output.weight"),
                        spec.n_head * spec.head_dim_v,
                        d,
                    )?,
                }),
                BlockKind::DeltaNet => Mixer::DeltaNet(GdnMixer {
                    wqkv: mat(f, &p("attn_qkv.weight"), d, g.conv_dim())?,
                    wz: mat(f, &p("attn_gate.weight"), d, g.value_dim())?,
                    conv: vecn(f, &p("ssm_conv1d.weight"), g.conv_dim() * g.d_conv)?,
                    a: vec1(f, &p("ssm_a"), g.n_v_heads)?,
                    dt_bias: vec1(f, &p("ssm_dt.bias"), g.n_v_heads)?,
                    w_alpha: mat(f, &p("ssm_alpha.weight"), d, g.n_v_heads)?,
                    w_beta: mat(f, &p("ssm_beta.weight"), d, g.n_v_heads)?,
                    norm: vec1(f, &p("ssm_norm.weight"), g.head_v())?,
                    wout: mat(f, &p("ssm_out.weight"), g.value_dim(), d)?,
                }),
            };
            layers.push(HybridLayer {
                attn_norm: vec1(f, &p("attn_norm.weight"), d)?,
                mixer,
                post_norm: vec1(f, &p("post_attention_norm.weight"), d)?,
                w_gate: mat(f, &p("ffn_gate.weight"), d, spec.n_ff)?,
                w_up: mat(f, &p("ffn_up.weight"), d, spec.n_ff)?,
                w_down: mat(f, &p("ffn_down.weight"), spec.n_ff, d)?,
            });
        }
        Ok(HybridWeights { layers })
    }

    pub fn push_dtypes(&self, v: &mut Vec<GgmlType>) {
        for l in &self.layers {
            v.extend([l.w_gate.dtype, l.w_up.dtype, l.w_down.dtype]);
            match &l.mixer {
                Mixer::Attention(a) => v.extend([a.wq.dtype, a.wk.dtype, a.wv.dtype, a.wo.dtype]),
                Mixer::DeltaNet(m) => v.extend([
                    m.wqkv.dtype,
                    m.wz.dtype,
                    m.w_alpha.dtype,
                    m.w_beta.dtype,
                    m.wout.dtype,
                ]),
            }
        }
    }
}

/// Extra working buffers the hybrid layers need beyond the dense [`Scratch`] set.
pub struct HybridScratch {
    /// `[n][conv_dim]` fused QKV projection (pre-conv).
    pub qkv: Vec<f32>,
    /// `[n][conv_dim]` after conv + SiLU: `[q | k | v]` per token.
    pub conv_out: Vec<f32>,
    /// `[n][key_dim]` each, the L2-normalised q and k gathered contiguously.
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    /// `[n][value_dim]` gathered v.
    pub v: Vec<f32>,
    /// `[n][value_dim]` output gate.
    pub z: Vec<f32>,
    /// `[n][n_v_heads]` each.
    pub alpha: Vec<f32>,
    pub beta: Vec<f32>,
    /// `[n][value_dim]` scan output, then gated-normed in place.
    pub out: Vec<f32>,
    /// `[n][2 · q_dim]` fused `(q, gate)` projection of the attention layers.
    pub qfull: Vec<f32>,
    /// `[n][q_dim]` the sigmoid gate gathered per head.
    pub gate: Vec<f32>,
}

impl HybridScratch {
    pub fn new(spec: &ArchSpec, g: &GdnSpec, n: usize) -> HybridScratch {
        let cd = g.conv_dim() as usize;
        let kd = g.key_dim() as usize;
        let vd = g.value_dim() as usize;
        let hv = g.n_v_heads as usize;
        let qd = spec.q_dim() as usize;
        HybridScratch {
            qkv: vec![0.0; n * cd],
            conv_out: vec![0.0; n * cd],
            q: vec![0.0; n * kd],
            k: vec![0.0; n * kd],
            v: vec![0.0; n * vd],
            z: vec![0.0; n * vd],
            alpha: vec![0.0; n * hv],
            beta: vec![0.0; n * hv],
            out: vec![0.0; n * vd],
            qfull: vec![0.0; n * 2 * qd],
            gate: vec![0.0; n * qd],
        }
    }

    pub fn bytes(spec: &ArchSpec, g: &GdnSpec, n: usize) -> u64 {
        let n = n as u64;
        let per_tok = 2 * g.conv_dim() as u64
            + 2 * g.key_dim() as u64
            + 3 * g.value_dim() as u64
            + 2 * g.n_v_heads as u64
            + 3 * spec.q_dim() as u64;
        n * per_tok * 4
    }
}

/// Run every layer of the hybrid stack over the `n` tokens in `s.x` (positions `pos0..`).
pub fn forward_layers(
    spec: &ArchSpec,
    w: &HybridWeights,
    pool: &ThreadPool,
    kv: &mut KvCache,
    n: usize,
    pos0: usize,
    s: &mut Scratch,
) {
    let g = spec.gdn.as_ref().expect("hybrid spec");
    let mut attn_idx = 0usize;
    let mut rs_idx = 0usize;
    for layer in &w.layers {
        let d = spec.d_model as usize;
        for t in 0..n {
            rms_norm(
                &s.x[t * d..(t + 1) * d],
                &layer.attn_norm,
                spec.rms_eps,
                &mut s.h[t * d..(t + 1) * d],
            );
        }
        match &layer.mixer {
            Mixer::Attention(a) => {
                attention_mixer(spec, a, pool, kv, attn_idx, n, pos0, s);
                attn_idx += 1;
            }
            Mixer::DeltaNet(m) => {
                deltanet_mixer(spec, g, m, pool, kv, rs_idx, n, s);
                rs_idx += 1;
            }
        }
        ffn(spec, layer, pool, n, s);
    }
}

/// Gated full attention: `s.h` (normed input) → `s.x += W_o (attend(q, K, V) ⊙ sigmoid(gate))`.
#[allow(clippy::too_many_arguments)]
fn attention_mixer(
    spec: &ArchSpec,
    a: &AttnMixer,
    pool: &ThreadPool,
    kv: &mut KvCache,
    al: usize,
    n: usize,
    pos0: usize,
    s: &mut Scratch,
) {
    let hs = s.hybrid.as_mut().expect("hybrid scratch");
    let d = spec.d_model as usize;
    let n_head = spec.n_head as usize;
    let n_kv = spec.n_kv_head as usize;
    let hd = spec.head_dim as usize;
    let hdv = spec.head_dim_v as usize;
    let q_dim = n_head * hd;
    let kv_dim = n_kv * hd;
    let v_dim = n_kv * hdv;

    project(
        pool,
        &a.wq,
        &s.h,
        n,
        d,
        &mut hs.qfull[..n * 2 * q_dim],
        None,
    );
    project(pool, &a.wk, &s.h, n, d, &mut s.k[..n * kv_dim], None);
    project(pool, &a.wv, &s.h, n, d, &mut s.v[..n * v_dim], None);
    // Split (q, gate) per head.
    for t in 0..n {
        for h in 0..n_head {
            let src = &hs.qfull[t * 2 * q_dim + h * 2 * hd..t * 2 * q_dim + (h + 1) * 2 * hd];
            s.q[t * q_dim + h * hd..t * q_dim + (h + 1) * hd].copy_from_slice(&src[..hd]);
            hs.gate[t * q_dim + h * hd..t * q_dim + (h + 1) * hd].copy_from_slice(&src[hd..]);
        }
    }
    let rp = RopeParams {
        kind: spec.rope.kind,
        head_dim: hd,
        rot_dim: spec.rope.dim as usize,
        theta: spec.rope.theta,
        freq_scale: spec.rope.freq_scale,
        attn_factor: spec.rope.attn_factor,
    };
    for t in 0..n {
        let q = &mut s.q[t * q_dim..(t + 1) * q_dim];
        let k = &mut s.k[t * kv_dim..(t + 1) * kv_dim];
        per_head_norm(q, hd, &a.q_norm, spec.rms_eps);
        per_head_norm(k, hd, &a.k_norm, spec.rms_eps);
        rope(q, n_head, (pos0 + t) as u32, &rp);
        rope(k, n_kv, (pos0 + t) as u32, &rp);
        kv.store_k(al, pos0 + t, k);
        kv.store_v(al, pos0 + t, &s.v[t * v_dim..(t + 1) * v_dim]);
    }
    attend(
        pool,
        kv,
        al,
        n,
        pos0,
        &s.q[..n * q_dim],
        Attend {
            n_head,
            n_kv,
            hd,
            hdv,
            scale: 1.0 / (hd as f32).sqrt(),
            window: None,
        },
        &mut s.attn,
    );
    let attn_dim = n_head * hdv;
    for i in 0..n * attn_dim {
        s.attn[i] *= gdn::sigmoid(hs.gate[i]);
    }
    project(pool, &a.wo, &s.attn, n, attn_dim, &mut s.ffn[..n * d], None);
    for t in 0..n {
        add_inplace(&mut s.x[t * d..(t + 1) * d], &s.ffn[t * d..(t + 1) * d]);
    }
}

/// Gated DeltaNet: `s.h` (normed input) → `s.x += W_out gated_norm(scan(conv(qkv)), z)`.
#[allow(clippy::too_many_arguments)]
fn deltanet_mixer(
    spec: &ArchSpec,
    g: &GdnSpec,
    m: &GdnMixer,
    pool: &ThreadPool,
    kv: &mut KvCache,
    ri: usize,
    n: usize,
    s: &mut Scratch,
) {
    let hs = s.hybrid.as_mut().expect("hybrid scratch");
    let d = spec.d_model as usize;
    let cd = g.conv_dim() as usize;
    let kd = g.key_dim() as usize;
    let vd = g.value_dim() as usize;
    let hk = g.n_k_heads as usize;
    let hv = g.n_v_heads as usize;
    let hdim = g.head_k as usize;
    let d_conv = g.d_conv as usize;

    project(pool, &m.wqkv, &s.h, n, d, &mut hs.qkv[..n * cd], None);
    project(pool, &m.wz, &s.h, n, d, &mut hs.z[..n * vd], None);
    project(pool, &m.w_alpha, &s.h, n, d, &mut hs.alpha[..n * hv], None);
    project(pool, &m.w_beta, &s.h, n, d, &mut hs.beta[..n * hv], None);

    let rs = &mut kv.rs[ri];
    gdn::conv_silu(
        &mut rs.conv,
        &hs.qkv[..n * cd],
        &m.conv,
        n,
        cd,
        d_conv,
        &mut hs.conv_out[..n * cd],
    );
    for t in 0..n {
        let row = &hs.conv_out[t * cd..(t + 1) * cd];
        hs.q[t * kd..(t + 1) * kd].copy_from_slice(&row[..kd]);
        hs.k[t * kd..(t + 1) * kd].copy_from_slice(&row[kd..2 * kd]);
        hs.v[t * vd..(t + 1) * vd].copy_from_slice(&row[2 * kd..]);
    }
    gdn::l2_norm_heads(&mut hs.q[..n * kd], hdim, spec.rms_eps);
    gdn::l2_norm_heads(&mut hs.k[..n * kd], hdim, spec.rms_eps);
    // g = A · softplus(alpha + dt_bias), beta = sigmoid(beta): in place.
    for t in 0..n {
        for h in 0..hv {
            let i = t * hv + h;
            hs.alpha[i] = m.a[h] * gdn::softplus(hs.alpha[i] + m.dt_bias[h]);
            hs.beta[i] = gdn::sigmoid(hs.beta[i]);
        }
    }
    gdn::gated_delta_scan(
        pool,
        GdnDims {
            n_k_heads: hk,
            n_v_heads: hv,
            head_dim: hdim,
        },
        &mut rs.state,
        &hs.q[..n * kd],
        &hs.k[..n * kd],
        &hs.v[..n * vd],
        &hs.alpha[..n * hv],
        &hs.beta[..n * hv],
        n,
        &mut hs.out[..n * vd],
    );
    gdn::gated_rms_norm(
        &mut hs.out[..n * vd],
        &hs.z[..n * vd],
        &m.norm,
        hdim,
        spec.rms_eps,
    );
    project(pool, &m.wout, &hs.out, n, vd, &mut s.ffn[..n * d], None);
    for t in 0..n {
        add_inplace(&mut s.x[t * d..(t + 1) * d], &s.ffn[t * d..(t + 1) * d]);
    }
}

fn ffn(spec: &ArchSpec, layer: &HybridLayer, pool: &ThreadPool, n: usize, s: &mut Scratch) {
    let d = spec.d_model as usize;
    let n_ff = spec.n_ff as usize;
    for t in 0..n {
        rms_norm(
            &s.x[t * d..(t + 1) * d],
            &layer.post_norm,
            spec.rms_eps,
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
    swiglu_inplace(&mut s.gate[..n * n_ff], &s.up[..n * n_ff]);
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
        add_inplace(&mut s.x[t * d..(t + 1) * d], &s.ffn[t * d..(t + 1) * d]);
    }
}

#[cfg(test)]
mod tests {
    use crate::forward::{Model, Scratch};
    use crate::{ArchSpec, BlockKind, Family, KvCache};
    use llmario_engine_core::GgmlType;
    use llmario_engine_cpu::ThreadPool;
    use llmario_engine_formats::gguf::writer::GgufWriter;
    use llmario_engine_formats::{GgufFile, MetaValue};

    /// A tiny random `qwen35` model (three DeltaNet layers, one gated-attention layer) written as
    /// a GGUF in memory, so the whole hybrid forward runs in CI without model files.
    fn tiny_hybrid_model(dir: &std::path::Path) -> std::path::PathBuf {
        let d = 32u64;
        let n_head = 2u64;
        let n_kv = 1u64;
        let hd = 8u64;
        let n_ff = 48u64;
        let vocab = 64u64;
        let n_layer = 4u64;
        // DeltaNet geometry: 2 QK heads × 8, 4 V heads × 8, conv kernel 4.
        let head = 8u64;
        let n_k = 2u64;
        let n_v = 4u64;
        let key_dim = n_k * head;
        let value_dim = n_v * head;
        let conv_dim = 2 * key_dim + value_dim;
        let d_conv = 4u64;
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
        w.meta("general.architecture", MetaValue::Str("qwen35".into()))
            .meta("qwen35.block_count", MetaValue::U32(n_layer as u32))
            .meta("qwen35.embedding_length", MetaValue::U32(d as u32))
            .meta("qwen35.attention.head_count", MetaValue::U32(n_head as u32))
            .meta(
                "qwen35.attention.head_count_kv",
                MetaValue::U32(n_kv as u32),
            )
            .meta("qwen35.attention.key_length", MetaValue::U32(hd as u32))
            .meta("qwen35.attention.value_length", MetaValue::U32(hd as u32))
            .meta("qwen35.feed_forward_length", MetaValue::U32(n_ff as u32))
            .meta("qwen35.vocab_size", MetaValue::U32(vocab as u32))
            .meta("qwen35.context_length", MetaValue::U32(64))
            .meta("qwen35.rope.freq_base", MetaValue::F32(10000.0))
            .meta("qwen35.rope.dimension_count", MetaValue::U32(4))
            .meta("qwen35.full_attention_interval", MetaValue::U32(4))
            .meta("qwen35.ssm.conv_kernel", MetaValue::U32(d_conv as u32))
            .meta("qwen35.ssm.state_size", MetaValue::U32(head as u32))
            .meta("qwen35.ssm.group_count", MetaValue::U32(n_k as u32))
            .meta("qwen35.ssm.time_step_rank", MetaValue::U32(n_v as u32))
            .meta("qwen35.ssm.inner_size", MetaValue::U32(value_dim as u32))
            .meta(
                "qwen35.attention.layer_norm_rms_epsilon",
                MetaValue::F32(1e-6),
            );
        w.tensor(
            "token_embd.weight",
            &[d, vocab],
            GgmlType::F32,
            f32s(d * vocab, 1),
        );
        w.tensor("output_norm.weight", &[d], GgmlType::F32, consts(d, 1.0));
        for l in 0..n_layer {
            let p = |s: &str| format!("blk.{l}.{s}");
            w.tensor(&p("attn_norm.weight"), &[d], GgmlType::F32, consts(d, 1.0));
            w.tensor(
                &p("post_attention_norm.weight"),
                &[d],
                GgmlType::F32,
                consts(d, 1.0),
            );
            if (l + 1) % 4 == 0 {
                w.tensor(
                    &p("attn_q.weight"),
                    &[d, 2 * n_head * hd],
                    GgmlType::F32,
                    f32s(d * 2 * n_head * hd, 10 + l),
                );
                w.tensor(
                    &p("attn_k.weight"),
                    &[d, n_kv * hd],
                    GgmlType::F32,
                    f32s(d * n_kv * hd, 20 + l),
                );
                w.tensor(
                    &p("attn_v.weight"),
                    &[d, n_kv * hd],
                    GgmlType::F32,
                    f32s(d * n_kv * hd, 30 + l),
                );
                w.tensor(
                    &p("attn_q_norm.weight"),
                    &[hd],
                    GgmlType::F32,
                    consts(hd, 1.0),
                );
                w.tensor(
                    &p("attn_k_norm.weight"),
                    &[hd],
                    GgmlType::F32,
                    consts(hd, 1.0),
                );
                w.tensor(
                    &p("attn_output.weight"),
                    &[n_head * hd, d],
                    GgmlType::F32,
                    f32s(n_head * hd * d, 40 + l),
                );
            } else {
                w.tensor(
                    &p("attn_qkv.weight"),
                    &[d, conv_dim],
                    GgmlType::F32,
                    f32s(d * conv_dim, 110 + l),
                );
                w.tensor(
                    &p("attn_gate.weight"),
                    &[d, value_dim],
                    GgmlType::F32,
                    f32s(d * value_dim, 120 + l),
                );
                w.tensor(
                    &p("ssm_conv1d.weight"),
                    &[d_conv, conv_dim],
                    GgmlType::F32,
                    f32s(d_conv * conv_dim, 130 + l),
                );
                w.tensor(&p("ssm_a"), &[n_v], GgmlType::F32, consts(n_v, -0.7));
                w.tensor(
                    &p("ssm_alpha.weight"),
                    &[d, n_v],
                    GgmlType::F32,
                    f32s(d * n_v, 140 + l),
                );
                w.tensor(
                    &p("ssm_beta.weight"),
                    &[d, n_v],
                    GgmlType::F32,
                    f32s(d * n_v, 150 + l),
                );
                w.tensor(&p("ssm_dt.bias"), &[n_v], GgmlType::F32, consts(n_v, 0.5));
                w.tensor(
                    &p("ssm_norm.weight"),
                    &[head],
                    GgmlType::F32,
                    consts(head, 1.0),
                );
                w.tensor(
                    &p("ssm_out.weight"),
                    &[value_dim, d],
                    GgmlType::F32,
                    f32s(value_dim * d, 160 + l),
                );
            }
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
        }
        let p = dir.join("tiny_hybrid.gguf");
        std::fs::write(&p, w.to_bytes()).unwrap();
        p
    }

    /// Prefill of `toks` at once vs. one token at a time: same logits, same recurrent state.
    fn check_prefill_vs_stepwise(m: &Model, pool: &ThreadPool, toks: &[u32], tol: f32) {
        let mut kv_a = KvCache::new(&m.spec, 64);
        let mut s_a = Scratch::new(&m.spec, toks.len());
        let la = m.forward(pool, &mut kv_a, toks, &mut s_a).to_vec();
        let mut kv_b = KvCache::new(&m.spec, 64);
        let mut s_b = Scratch::new(&m.spec, 1);
        let mut lb = vec![];
        for &t in toks {
            lb = m.forward(pool, &mut kv_b, &[t], &mut s_b).to_vec();
        }
        assert_eq!(kv_a.len, toks.len());
        assert_eq!(kv_b.len, toks.len());
        let max_abs = la.iter().map(|v| v.abs()).fold(0f32, f32::max);
        for (i, (a, b)) in la.iter().zip(&lb).enumerate() {
            assert!(
                (a - b).abs() <= tol * max_abs.max(1.0),
                "logit {i}: {a} vs {b} (max |logit| {max_abs})"
            );
        }
        for (ra, rb) in kv_a.rs.iter().zip(&kv_b.rs) {
            for (a, b) in ra.state.iter().zip(&rb.state) {
                assert!((a - b).abs() <= tol * 10.0, "recurrent state {a} vs {b}");
            }
            assert_eq!(ra.conv, rb.conv);
        }
    }

    #[test]
    fn hybrid_spec_layout_and_cache_sizing() {
        let dir = tempfile::tempdir().unwrap();
        let f = GgufFile::open(&tiny_hybrid_model(dir.path())).unwrap();
        let spec = ArchSpec::from_gguf(&f).unwrap();
        assert_eq!(spec.family, Family::Qwen35);
        assert_eq!(
            spec.blocks,
            vec![
                BlockKind::DeltaNet,
                BlockKind::DeltaNet,
                BlockKind::DeltaNet,
                BlockKind::Attention
            ]
        );
        assert!(spec.attn_out_gate);
        let g = spec.gdn.as_ref().unwrap();
        assert_eq!(
            (g.head_v(), g.key_dim(), g.value_dim(), g.conv_dim()),
            (8, 16, 32, 64)
        );
        assert_eq!(spec.n_attn_layers(), 1);
        assert_eq!(spec.n_recurrent_layers(), 3);
        // One K/V slab, not four; recurrent state charged on top.
        let kv = KvCache::new(&spec, 16);
        assert_eq!(kv.n_layer, 1);
        assert_eq!(kv.rs.len(), 3);
        assert_eq!(kv.rs[0].conv.len(), 3 * 64);
        assert_eq!(kv.rs[0].state.len(), 4 * 8 * 8);
        assert_eq!(
            KvCache::bytes(&spec, 16),
            (8 + 8) * 2 * 16 + 3 * (3 * 64 + 4 * 8 * 8) * 4
        );
        assert_eq!(spec.kv_bytes_per_token(4.0), 16 * 4);
    }

    #[test]
    fn hybrid_prefill_equals_token_by_token_and_truncate_resets() {
        let dir = tempfile::tempdir().unwrap();
        let f = GgufFile::open(&tiny_hybrid_model(dir.path())).unwrap();
        let m = Model::load(&f).unwrap();
        assert!(m.weights.hybrid.is_some());
        let pool = ThreadPool::new(3);
        let toks = [3u32, 17, 5, 42, 9, 61, 2];
        check_prefill_vs_stepwise(&m, &pool, &toks, 1e-4);

        // Truncating below the cached length resets the sequence (no checkpoints yet);
        // truncating at or above it is a no-op; clear() zeroes the recurrent state.
        let mut kv = KvCache::new(&m.spec, 64);
        let mut s = Scratch::new(&m.spec, 8);
        m.forward(&pool, &mut kv, &toks, &mut s);
        assert!(kv.rs[0].state.iter().any(|v| *v != 0.0));
        kv.truncate(toks.len());
        assert_eq!(kv.len, toks.len());
        kv.truncate(3);
        assert_eq!(kv.len, 0);
        assert!(kv.rs.iter().all(|r| r.state.iter().all(|v| *v == 0.0)));
        assert!(kv.rs.iter().all(|r| r.conv.iter().all(|v| *v == 0.0)));
        // Recomputing from scratch reproduces the same logits (state fully reset).
        let l1 = m.forward(&pool, &mut kv, &toks, &mut s).to_vec();
        kv.clear();
        let l2 = m.forward(&pool, &mut kv, &toks, &mut s).to_vec();
        assert_eq!(l1, l2);
    }

    /// Real model (gated on `LLMARIO_TEST_GGUF`, comma-separated; only `qwen35` files are used):
    /// prefill vs. token-by-token consistency on a short prompt.
    #[test]
    fn real_qwen35_prefill_equals_token_by_token() {
        let Ok(models) = std::env::var("LLMARIO_TEST_GGUF") else {
            eprintln!("LLMARIO_TEST_GGUF not set; skipping");
            return;
        };
        for path in models.split(',').filter(|s| !s.is_empty()) {
            let f = GgufFile::open(std::path::Path::new(path)).unwrap();
            if f.architecture() != Some("qwen35") {
                continue;
            }
            let m = Model::load(&f).unwrap();
            let pool = ThreadPool::new(4);
            // "The capital of France is" in the Qwen3.5 tokenizer, plus a few more ids.
            let toks = [
                760u32, 6511, 314, 9338, 369, 11751, 13, 198, 760, 6511, 314, 9338,
            ];
            check_prefill_vs_stepwise(&m, &pool, &toks, 2e-3);
            eprintln!("{path}: prefill == stepwise");
        }
    }
}
