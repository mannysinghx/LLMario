//! Forward pass for the dense GQA families on the CPU backend.
//!
//! Explicit, layer-by-layer execution with a scratch set sized once from the model shape and the
//! batch size (`Scratch::bytes` is what the plan charges). Prefill processes `n` tokens at once
//! through the matmul path; decode processes one token through the matvec path. Attention runs
//! per head in parallel over the pool with fp32 accumulation.

use crate::arch::{ArchSpec, Family};
use crate::kv::KvCache;
use crate::weights::{LayerWeights, Weights};
use llmario_engine_cpu::ops::{add_inplace, dot, swiglu_inplace, RopeParams};
use llmario_engine_cpu::{dequant_row, matmul, matvec, rms_norm, rope, softmax, ThreadPool};
use llmario_engine_formats::GgufFile;

pub struct Model<'a> {
    pub spec: ArchSpec,
    pub weights: Weights<'a>,
}

/// Working buffers for a batch of up to `n_batch` tokens.
pub struct Scratch {
    n_batch: usize,
    x: Vec<f32>,
    h: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    ffn: Vec<f32>,
    logits: Vec<f32>,
}

impl Scratch {
    pub fn new(spec: &ArchSpec, n_batch: usize) -> Scratch {
        let d = spec.d_model as usize;
        let n = n_batch.max(1);
        Scratch {
            n_batch: n,
            x: vec![0.0; n * d],
            h: vec![0.0; n * d],
            q: vec![0.0; n * spec.q_dim() as usize],
            k: vec![0.0; n * spec.kv_dim() as usize],
            v: vec![0.0; n * spec.v_dim() as usize],
            attn: vec![0.0; n * (spec.n_head * spec.head_dim_v) as usize],
            gate: vec![0.0; n * spec.n_ff as usize],
            up: vec![0.0; n * spec.n_ff as usize],
            ffn: vec![0.0; n * d],
            logits: vec![0.0; spec.n_vocab as usize],
        }
    }

    /// Bytes of scratch for `n_batch` tokens (charged by the plan).
    pub fn bytes(spec: &ArchSpec, n_batch: usize) -> u64 {
        let n = n_batch.max(1) as u64;
        let d = spec.d_model as u64;
        let per_tok = 2 * d
            + spec.q_dim() as u64
            + spec.kv_dim() as u64
            + spec.v_dim() as u64
            + (spec.n_head * spec.head_dim_v) as u64
            + 2 * spec.n_ff as u64
            + d;
        (n * per_tok + spec.n_vocab as u64) * 4
    }
}

impl<'a> Model<'a> {
    pub fn load(f: &'a GgufFile) -> crate::Result<Model<'a>> {
        let spec = ArchSpec::from_gguf(f)?;
        let weights = Weights::load(f, &spec)?;
        Ok(Model { spec, weights })
    }

    /// Run `tokens` (positions `kv.len..kv.len + tokens.len()`), append their K/V to `kv`, and
    /// return the logits of the last token in `scratch.logits`.
    pub fn forward<'s>(
        &self,
        pool: &ThreadPool,
        kv: &mut KvCache,
        tokens: &[u32],
        scratch: &'s mut Scratch,
    ) -> &'s [f32] {
        let n = tokens.len();
        assert!(n >= 1 && n <= scratch.n_batch, "batch of {n} tokens");
        assert!(kv.len + n <= kv.max_ctx, "context overflow");
        let spec = &self.spec;
        let d = spec.d_model as usize;
        let pos0 = kv.len;

        // Embeddings.
        for (t, &tok) in tokens.iter().enumerate() {
            dequant_row(
                &self.weights.token_embd,
                tok as usize,
                &mut scratch.x[t * d..(t + 1) * d],
            );
        }

        for (l, layer) in self.weights.layers.iter().enumerate() {
            self.attention_block(pool, kv, l, layer, n, pos0, scratch);
            self.ffn_block(pool, layer, n, scratch);
        }
        kv.len += n;

        // Final norm + output head on the last token only.
        let last = &scratch.x[(n - 1) * d..n * d];
        let h = &mut scratch.h[..d];
        rms_norm(last, &self.weights.output_norm, spec.rms_eps, h);
        let head = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.token_embd);
        matvec(pool, head, h, &mut scratch.logits);
        &scratch.logits
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_block(
        &self,
        pool: &ThreadPool,
        kv: &mut KvCache,
        l: usize,
        layer: &LayerWeights,
        n: usize,
        pos0: usize,
        s: &mut Scratch,
    ) {
        let spec = &self.spec;
        let d = spec.d_model as usize;
        let n_head = spec.n_head as usize;
        let n_kv = spec.n_kv_head as usize;
        let hd = spec.head_dim as usize;
        let hdv = spec.head_dim_v as usize;
        let q_dim = n_head * hd;
        let kv_dim = n_kv * hd;
        let v_dim = n_kv * hdv;

        // Pre-norm.
        for t in 0..n {
            rms_norm(
                &s.x[t * d..(t + 1) * d],
                &layer.attn_norm,
                spec.rms_eps,
                &mut s.h[t * d..(t + 1) * d],
            );
        }
        // Projections (matmul for n > 1, matvec for decode).
        project(
            pool,
            &layer.wq,
            &s.h,
            n,
            d,
            &mut s.q[..n * q_dim],
            layer.bq.as_deref(),
        );
        project(
            pool,
            &layer.wk,
            &s.h,
            n,
            d,
            &mut s.k[..n * kv_dim],
            layer.bk.as_deref(),
        );
        project(
            pool,
            &layer.wv,
            &s.h,
            n,
            d,
            &mut s.v[..n * v_dim],
            layer.bv.as_deref(),
        );

        // QK-norm (per head) and RoPE.
        let use_rope = !spec.nope_layers.contains(&(l as u32));
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
            if let Some(w) = &layer.q_norm {
                per_head_norm(q, hd, w, spec.rms_eps);
            }
            if let Some(w) = &layer.k_norm {
                per_head_norm(k, hd, w, spec.rms_eps);
            }
            if use_rope {
                rope(q, n_head, (pos0 + t) as u32, &rp);
                rope(k, n_kv, (pos0 + t) as u32, &rp);
            }
            kv.k_row_mut(l, pos0 + t).copy_from_slice(k);
            kv.v_row_mut(l, pos0 + t)
                .copy_from_slice(&s.v[t * v_dim..(t + 1) * v_dim]);
        }

        // Attention: for each token and head, softmax(q·K / sqrt(hd)) · V over positions ≤ t.
        let scale = 1.0 / (hd as f32).sqrt();
        let group = n_head / n_kv;
        let n_ctx = pos0 + n;
        let k_all = kv.k_layer(l, n_ctx);
        let v_all = kv.v_layer(l, n_ctx);
        let attn_dim = n_head * hdv;
        let q_all = &s.q[..n * q_dim];
        let out_ptr = SendPtr(s.attn.as_mut_ptr());
        pool.parallel_for(n * n_head, None, |start, end| {
            let mut scores = vec![0f32; n_ctx];
            for idx in start..end {
                let t = idx / n_head;
                let h = idx % n_head;
                let kvh = h / group;
                let q = &q_all[t * q_dim + h * hd..t * q_dim + (h + 1) * hd];
                let n_pos = pos0 + t + 1;
                for p in 0..n_pos {
                    let k = &k_all[p * kv_dim + kvh * hd..p * kv_dim + (kvh + 1) * hd];
                    scores[p] = dot(q, k) * scale;
                }
                softmax(&mut scores[..n_pos]);
                let mut acc = vec![0f32; hdv];
                for p in 0..n_pos {
                    let v = &v_all[p * v_dim + kvh * hdv..p * v_dim + (kvh + 1) * hdv];
                    let w = scores[p];
                    for i in 0..hdv {
                        acc[i] += w * v[i];
                    }
                }
                // SAFETY: each (t, h) writes a disjoint hdv-wide slice of `attn`.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        acc.as_ptr(),
                        out_ptr.get().add(t * attn_dim + h * hdv),
                        hdv,
                    );
                }
            }
        });

        // Output projection and residual.
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
            add_inplace(&mut s.x[t * d..(t + 1) * d], &s.ffn[t * d..(t + 1) * d]);
        }
    }

    fn ffn_block(&self, pool: &ThreadPool, layer: &LayerWeights, n: usize, s: &mut Scratch) {
        let spec = &self.spec;
        let d = spec.d_model as usize;
        let n_ff = spec.n_ff as usize;
        for t in 0..n {
            rms_norm(
                &s.x[t * d..(t + 1) * d],
                &layer.ffn_norm,
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
        match spec.family {
            Family::Llama | Family::Qwen2 | Family::Qwen3 | Family::SmolLm3 => {
                swiglu_inplace(&mut s.gate[..n * n_ff], &s.up[..n * n_ff]);
            }
        }
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
}

/// `y = W x (+ b)` for `n` tokens of width `cols`.
fn project(
    pool: &ThreadPool,
    w: &llmario_engine_cpu::QMat,
    x: &[f32],
    n: usize,
    cols: usize,
    y: &mut [f32],
    bias: Option<&[f32]>,
) {
    let rows = w.rows;
    if n == 1 {
        matvec(pool, w, &x[..cols], &mut y[..rows]);
    } else {
        matmul(pool, w, &x[..n * cols], n, &mut y[..n * rows]);
    }
    if let Some(b) = bias {
        for t in 0..n {
            add_inplace(&mut y[t * rows..(t + 1) * rows], b);
        }
    }
}

fn per_head_norm(x: &mut [f32], hd: usize, w: &[f32], eps: f32) {
    let mut tmp = vec![0f32; hd];
    for head in x.chunks_exact_mut(hd) {
        rms_norm(head, w, eps, &mut tmp);
        head.copy_from_slice(&tmp);
    }
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
// SAFETY: writes are partitioned by (token, head) into disjoint slices.
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    #[inline]
    fn get(&self) -> *mut f32 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmario_engine_core::{EngineError, GgmlType};
    use llmario_engine_formats::gguf::writer::GgufWriter;
    use llmario_engine_formats::MetaValue;

    /// A tiny random llama-family model written as a GGUF in memory, so the forward pass can be
    /// exercised in CI without model files.
    fn tiny_model(dir: &std::path::Path) -> std::path::PathBuf {
        let d = 32u64;
        let n_head = 4u64;
        let n_kv = 2u64;
        let hd = 8u64;
        let n_ff = 48u64;
        let vocab = 64u64;
        let n_layer = 2u64;
        let f32s = |n: u64, seed: u64| -> Vec<u8> {
            (0..n)
                .flat_map(|i| {
                    let v = (((i * 2654435761 + seed * 97) % 1000) as f32 / 1000.0 - 0.5) * 0.2;
                    v.to_le_bytes()
                })
                .collect()
        };
        let ones = |n: u64| -> Vec<u8> { (0..n).flat_map(|_| 1.0f32.to_le_bytes()).collect() };
        let mut w = GgufWriter::new();
        w.meta("general.architecture", MetaValue::Str("llama".into()))
            .meta("llama.block_count", MetaValue::U32(n_layer as u32))
            .meta("llama.embedding_length", MetaValue::U32(d as u32))
            .meta("llama.attention.head_count", MetaValue::U32(n_head as u32))
            .meta("llama.attention.head_count_kv", MetaValue::U32(n_kv as u32))
            .meta("llama.feed_forward_length", MetaValue::U32(n_ff as u32))
            .meta("llama.vocab_size", MetaValue::U32(vocab as u32))
            .meta("llama.context_length", MetaValue::U32(64))
            .meta("llama.rope.freq_base", MetaValue::F32(10000.0))
            .meta(
                "llama.attention.layer_norm_rms_epsilon",
                MetaValue::F32(1e-5),
            );
        w.tensor(
            "token_embd.weight",
            &[d, vocab],
            GgmlType::F32,
            f32s(d * vocab, 1),
        );
        w.tensor("output_norm.weight", &[d], GgmlType::F32, ones(d));
        w.tensor(
            "output.weight",
            &[d, vocab],
            GgmlType::F32,
            f32s(d * vocab, 2),
        );
        for l in 0..n_layer {
            let p = |s: &str| format!("blk.{l}.{s}");
            w.tensor(&p("attn_norm.weight"), &[d], GgmlType::F32, ones(d));
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
            w.tensor(
                &p("attn_v.weight"),
                &[d, n_kv * hd],
                GgmlType::F32,
                f32s(d * n_kv * hd, 30 + l),
            );
            w.tensor(
                &p("attn_output.weight"),
                &[n_head * hd, d],
                GgmlType::F32,
                f32s(n_head * hd * d, 40 + l),
            );
            w.tensor(&p("ffn_norm.weight"), &[d], GgmlType::F32, ones(d));
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
        let p = dir.join("tiny.gguf");
        std::fs::write(&p, w.to_bytes()).unwrap();
        p
    }

    #[test]
    fn prefill_then_decode_equals_token_by_token() {
        let dir = tempfile::tempdir().unwrap();
        let path = tiny_model(dir.path());
        let f = GgufFile::open(&path).unwrap();
        let m = Model::load(&f).unwrap();
        assert_eq!(m.spec.family, Family::Llama);
        let pool = ThreadPool::new(2);
        let toks = [3u32, 17, 5, 42, 9];

        // Path A: prefill all five at once.
        let mut kv_a = KvCache::new(&m.spec, 16);
        let mut s_a = Scratch::new(&m.spec, 8);
        let la = m.forward(&pool, &mut kv_a, &toks, &mut s_a).to_vec();

        // Path B: one token at a time.
        let mut kv_b = KvCache::new(&m.spec, 16);
        let mut s_b = Scratch::new(&m.spec, 1);
        let mut lb = vec![];
        for &t in &toks {
            lb = m.forward(&pool, &mut kv_b, &[t], &mut s_b).to_vec();
        }
        assert_eq!(kv_a.len, 5);
        assert_eq!(kv_b.len, 5);
        for (a, b) in la.iter().zip(&lb) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
        // Scratch accounting is consistent with the allocation.
        assert_eq!(
            Scratch::bytes(&m.spec, 8),
            (s_a.x.len()
                + s_a.h.len()
                + s_a.q.len()
                + s_a.k.len()
                + s_a.v.len()
                + s_a.attn.len()
                + s_a.gate.len()
                + s_a.up.len()
                + s_a.ffn.len()
                + s_a.logits.len()) as u64
                * 4
        );
        let _ = EngineError::Format(String::new());
    }
}
