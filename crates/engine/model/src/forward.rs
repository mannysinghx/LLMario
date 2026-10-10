//! Forward pass for the dense GQA families on the CPU backend (the hybrid family's layers are in
//! `hybrid.rs`, Gemma 4's in `gemma4.rs`; embeddings, the output head, the batch layout and the
//! scratch set are shared here).
//!
//! One call runs a **batch of sequences**: each [`SeqTokens`] appends its tokens to its own
//! sequence in the [`KvCache`]. The rows of every sequence are stacked, so the projections and
//! the FFN run once over all of them (the matmul path for more than one row, the matvec path for
//! exactly one); RoPE, the K/V writes and attention use each row's own sequence and position
//! ([`Rows`]); recurrent layers scan each sequence's rows with that sequence's state. A batch
//! with one sequence is exactly the single-sequence pass. The scratch set is sized once from the
//! model shape, the token budget and the number of sequences (`Scratch::bytes` is what the plan
//! charges). Attention runs per (token, KV head) in parallel with fp32 accumulation.

use crate::arch::{ArchSpec, Family};
use crate::gemma4;
use crate::hybrid::{self, HybridScratch};
use crate::kv::{KvCache, KvFull};
use crate::moe::{self, MoeScratch};
use crate::weights::{LayerWeights, Weights};
use llmario_engine_cpu::ops::{add_inplace, dot, swiglu_inplace, RopeParams};
use llmario_engine_cpu::{dequant_row, matmul, matvec, rms_norm, rope, softmax, ThreadPool};
use llmario_engine_formats::GgufFile;

pub struct Model<'a> {
    pub spec: ArchSpec,
    pub weights: Weights<'a>,
}

/// One sequence's share of a forward call: `tokens` are appended to sequence `seq` at its
/// current length.
#[derive(Clone, Copy, Debug)]
pub struct SeqTokens<'a> {
    pub seq: usize,
    pub tokens: &'a [u32],
}

/// A contiguous run of rows that belong to one sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Seg {
    pub seq: usize,
    /// First row of the run in the stacked batch.
    pub start: usize,
    pub n: usize,
    /// Position of the first row.
    pub pos0: usize,
}

/// Placement of the stacked rows of one pass through the layers.
#[derive(Clone, Debug, Default)]
pub(crate) struct Rows {
    /// Sequence of row `t`.
    pub seq: Vec<usize>,
    /// Absolute position of row `t` in its sequence.
    pub pos: Vec<usize>,
    /// Index into `segs` of row `t`.
    pub seg_of: Vec<usize>,
    pub segs: Vec<Seg>,
}

impl Rows {
    pub fn n(&self) -> usize {
        self.pos.len()
    }
    fn push(&mut self, seq: usize, pos0: usize, n: usize) {
        let start = self.n();
        let si = self.segs.len();
        self.segs.push(Seg {
            seq,
            start,
            n,
            pos0,
        });
        for i in 0..n {
            self.seq.push(seq);
            self.pos.push(pos0 + i);
            self.seg_of.push(si);
        }
    }
    /// One sequence `seq`, `n` rows from position `pos0`.
    #[cfg(test)]
    pub fn single(seq: usize, pos0: usize, n: usize) -> Rows {
        let mut r = Rows::default();
        r.push(seq, pos0, n);
        r
    }
}

/// Working buffers for up to `n_batch` stacked rows and `max_seqs` sequences per call.
pub struct Scratch {
    pub(crate) n_batch: usize,
    pub(crate) max_seqs: usize,
    pub(crate) x: Vec<f32>,
    pub(crate) h: Vec<f32>,
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    pub(crate) attn: Vec<f32>,
    pub(crate) gate: Vec<f32>,
    pub(crate) up: Vec<f32>,
    pub(crate) ffn: Vec<f32>,
    /// Last hidden row of each sequence of the call (`[max_seqs][d_model]`).
    pub(crate) last: Vec<f32>,
    /// Logits of each sequence's last token (`[max_seqs][n_vocab]`).
    pub(crate) logits: Vec<f32>,
    /// Extra buffers of the hybrid family (`None` for the dense families).
    pub(crate) hybrid: Option<HybridScratch>,
    /// Extra buffers of the MoE families.
    pub(crate) moe: Option<MoeScratch>,
}

impl Scratch {
    /// Scratch for single-sequence calls of up to `n_batch` tokens.
    pub fn new(spec: &ArchSpec, n_batch: usize) -> Scratch {
        Self::with_seqs(spec, n_batch, 1)
    }

    pub fn with_seqs(spec: &ArchSpec, n_batch: usize, max_seqs: usize) -> Scratch {
        let d = spec.d_model as usize;
        let n = n_batch.max(1);
        let m = max_seqs.clamp(1, n);
        // Q/K/V/attention widths are the largest over the layers (they differ per layer only in
        // Gemma 4, whose global layers have wider heads and fewer KV heads than its local ones).
        Scratch {
            n_batch: n,
            max_seqs: m,
            x: vec![0.0; n * d],
            h: vec![0.0; n * d],
            q: vec![0.0; n * spec.max_q_dim() as usize],
            k: vec![0.0; n * spec.max_kv_dim() as usize],
            v: vec![0.0; n * spec.max_v_dim() as usize],
            attn: vec![0.0; n * spec.max_attn_dim() as usize],
            gate: vec![0.0; n * spec.n_ff as usize],
            up: vec![0.0; n * spec.n_ff as usize],
            ffn: vec![0.0; n * d],
            last: vec![0.0; m * d],
            logits: vec![0.0; m * spec.n_vocab as usize],
            hybrid: spec.gdn.as_ref().map(|g| HybridScratch::new(spec, g, n)),
            moe: spec.moe.as_ref().map(|m| MoeScratch::new(spec, m, n)),
        }
    }

    /// Bytes of scratch for single-sequence calls of `n_batch` tokens (charged by the plan).
    pub fn bytes(spec: &ArchSpec, n_batch: usize) -> u64 {
        Self::bytes_with_seqs(spec, n_batch, 1)
    }

    pub fn bytes_with_seqs(spec: &ArchSpec, n_batch: usize, max_seqs: usize) -> u64 {
        let n = n_batch.max(1) as u64;
        let m = max_seqs.clamp(1, n_batch.max(1)) as u64;
        let d = spec.d_model as u64;
        let per_tok = 2 * d
            + spec.max_q_dim() as u64
            + spec.max_kv_dim() as u64
            + spec.max_v_dim() as u64
            + spec.max_attn_dim() as u64
            + 2 * spec.n_ff as u64
            + d;
        let hybrid = spec
            .gdn
            .as_ref()
            .map(|g| HybridScratch::bytes(spec, g, n as usize))
            .unwrap_or(0);
        let moe = spec
            .moe
            .as_ref()
            .map(|mo| MoeScratch::bytes(spec, mo, n as usize))
            .unwrap_or(0);
        (n * per_tok + m * (d + spec.n_vocab as u64)) * 4 + hybrid + moe
    }

    /// Rows one call may stack.
    pub fn n_batch(&self) -> usize {
        self.n_batch
    }
    /// Sequences one call may carry.
    pub fn max_seqs(&self) -> usize {
        self.max_seqs
    }
}

impl<'a> Model<'a> {
    pub fn load(f: &'a GgufFile) -> crate::Result<Model<'a>> {
        let spec = ArchSpec::from_gguf(f)?;
        let weights = Weights::load(f, &spec)?;
        Ok(Model { spec, weights })
    }

    /// Run `tokens` on sequence 0 (positions `kv.len(0)..`), append their K/V, and return the
    /// logits of the last token. Panics when the cache cannot grow (a single-sequence cache is
    /// sized for its whole context, so this means a context overflow).
    pub fn forward<'s>(
        &self,
        pool: &ThreadPool,
        kv: &mut KvCache,
        tokens: &[u32],
        scratch: &'s mut Scratch,
    ) -> &'s [f32] {
        match self.forward_batch(pool, kv, &[SeqTokens { seq: 0, tokens }], scratch) {
            Ok(l) => l,
            Err(e) => panic!("{e}"),
        }
    }

    /// Run a batch: every entry appends its tokens to its own sequence. Returns the logits of
    /// each entry's last token, `[batch.len()][n_vocab]` in batch order. Fails without changing
    /// anything when the cache cannot hold the new tokens.
    pub fn forward_batch<'s>(
        &self,
        pool: &ThreadPool,
        kv: &mut KvCache,
        batch: &[SeqTokens],
        scratch: &'s mut Scratch,
    ) -> Result<&'s [f32], KvFull> {
        let spec = &self.spec;
        let d = spec.d_model as usize;
        let vocab = spec.n_vocab as usize;
        let total: usize = batch.iter().map(|b| b.tokens.len()).sum();
        assert!(
            !batch.is_empty() && batch.len() <= scratch.max_seqs,
            "{} sequences in one call (scratch holds {})",
            batch.len(),
            scratch.max_seqs
        );
        assert!(
            total <= scratch.n_batch,
            "batch of {total} tokens (scratch holds {})",
            scratch.n_batch
        );
        for (i, b) in batch.iter().enumerate() {
            assert!(
                !b.tokens.is_empty(),
                "empty token list for sequence {}",
                b.seq
            );
            assert!(
                batch[..i].iter().all(|o| o.seq != b.seq),
                "sequence {} appears twice in one batch",
                b.seq
            );
            assert!(
                kv.len(b.seq) + b.tokens.len() <= kv.max_ctx(),
                "context overflow"
            );
        }
        // Check every sequence's room first, so a batch that does not fit changes nothing.
        let need: usize = batch
            .iter()
            .map(|b| kv.blocks_needed(b.seq, kv.len(b.seq) + b.tokens.len()))
            .sum();
        if need > kv.pool().free_blocks() {
            return Err(KvFull {
                needed: need,
                free: kv.pool().free_blocks(),
            });
        }
        for b in batch {
            let len = kv.len(b.seq);
            kv.reserve(b.seq, len + b.tokens.len())?;
        }

        // A sliding-window ring bounds how many tokens one pass may append per sequence (see
        // `kv.rs`), so long runs go through the layer stack in consecutive passes; for the other
        // families the bound is `usize::MAX`, i.e. one pass.
        let chunk = kv.max_batch().max(1);
        let mut done = vec![0usize; batch.len()];
        loop {
            let mut rows = Rows::default();
            let mut toks = Vec::new();
            let mut finishing = Vec::new();
            for (i, b) in batch.iter().enumerate() {
                let left = b.tokens.len() - done[i];
                if left == 0 {
                    continue;
                }
                let m = left.min(chunk);
                rows.push(b.seq, kv.len(b.seq), m);
                toks.extend_from_slice(&b.tokens[done[i]..done[i] + m]);
                done[i] += m;
                if done[i] == b.tokens.len() {
                    finishing.push((i, rows.n() - 1));
                }
            }
            if rows.n() == 0 {
                break;
            }
            self.run_layers(pool, kv, &toks, &rows, scratch);
            for (i, row) in finishing {
                scratch.last[i * d..(i + 1) * d]
                    .copy_from_slice(&scratch.x[row * d..(row + 1) * d]);
            }
        }

        // Final norm + output head on each sequence's last row.
        let m = batch.len();
        for i in 0..m {
            rms_norm(
                &scratch.last[i * d..(i + 1) * d],
                &self.weights.output_norm,
                spec.rms_eps,
                &mut scratch.h[i * d..(i + 1) * d],
            );
        }
        let head = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.token_embd);
        project(
            pool,
            head,
            &scratch.h,
            m,
            d,
            &mut scratch.logits[..m * vocab],
            None,
        );
        if let Some(g) = &spec.gemma4 {
            gemma4::softcap_inplace(&mut scratch.logits[..m * vocab], g.final_logit_softcap);
        }
        Ok(&scratch.logits[..m * vocab])
    }

    /// Embed `tokens` into `scratch.x` and run every layer over the stacked rows described by
    /// `rows`, appending their K/V (each sequence's length advances by its row count).
    fn run_layers(
        &self,
        pool: &ThreadPool,
        kv: &mut KvCache,
        tokens: &[u32],
        rows: &Rows,
        scratch: &mut Scratch,
    ) {
        let n = tokens.len();
        debug_assert_eq!(n, rows.n());
        let spec = &self.spec;
        let d = spec.d_model as usize;

        // Embeddings.
        for (t, &tok) in tokens.iter().enumerate() {
            dequant_row(
                &self.weights.token_embd,
                tok as usize,
                &mut scratch.x[t * d..(t + 1) * d],
            );
        }

        if let Some(g) = &self.weights.gemma4 {
            gemma4::forward_layers(spec, g, pool, kv, rows, scratch);
        } else if let Some(h) = &self.weights.hybrid {
            hybrid::forward_layers(spec, h, pool, kv, rows, scratch);
        } else {
            for (l, layer) in self.weights.layers.iter().enumerate() {
                self.attention_block(pool, kv, l, layer, rows, scratch);
                match &layer.moe {
                    Some(mw) => self.moe_block(pool, layer, mw, n, scratch),
                    None => self.ffn_block(pool, layer, n, scratch),
                }
            }
        }
        for sg in &rows.segs {
            kv.seq_mut(sg.seq).len += sg.n;
        }
    }

    fn attention_block(
        &self,
        pool: &ThreadPool,
        kv: &mut KvCache,
        l: usize,
        layer: &LayerWeights,
        rows: &Rows,
        s: &mut Scratch,
    ) {
        let spec = &self.spec;
        let n = rows.n();
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
            let (seq, pos) = (rows.seq[t], rows.pos[t]);
            let q = &mut s.q[t * q_dim..(t + 1) * q_dim];
            let k = &mut s.k[t * kv_dim..(t + 1) * kv_dim];
            if let Some(w) = &layer.q_norm {
                per_head_norm(q, hd, w, spec.rms_eps);
            }
            if let Some(w) = &layer.k_norm {
                per_head_norm(k, hd, w, spec.rms_eps);
            }
            if use_rope {
                rope(q, n_head, pos as u32, &rp);
                rope(k, n_kv, pos as u32, &rp);
            }
            kv.store_k(seq, l, pos, k);
            kv.store_v(seq, l, pos, &s.v[t * v_dim..(t + 1) * v_dim]);
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
                scale: 1.0 / (hd as f32).sqrt(),
                window: None,
            },
            &mut s.attn,
        );

        // Output projection and residual.
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
            Family::Llama
            | Family::Qwen2
            | Family::Qwen3
            | Family::SmolLm3
            | Family::Qwen35
            | Family::Qwen3Moe => {
                swiglu_inplace(&mut s.gate[..n * n_ff], &s.up[..n * n_ff]);
            }
            // Gemma 4 (GeGLU) never reaches this block: its layers run in `gemma4.rs`.
            Family::Gemma4 => unreachable!("Gemma 4 FFN runs in gemma4::forward_layers"),
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

impl Model<'_> {
    /// Pre-norm and the routed experts of a mixture-of-experts layer (see `moe.rs`).
    fn moe_block(
        &self,
        pool: &ThreadPool,
        layer: &LayerWeights,
        mw: &crate::weights::MoeWeights,
        n: usize,
        s: &mut Scratch,
    ) {
        let spec = &self.spec;
        let d = spec.d_model as usize;
        let m = spec.moe.as_ref().expect("MoE spec");
        for t in 0..n {
            rms_norm(
                &s.x[t * d..(t + 1) * d],
                &layer.ffn_norm,
                spec.rms_eps,
                &mut s.h[t * d..(t + 1) * d],
            );
        }
        let ms = s.moe.as_mut().expect("MoE scratch");
        moe::moe_ffn(spec, m, mw, pool, n, &s.h, &mut s.x, ms);
    }
}

/// Geometry, score scale and window of one [`attend`] call.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Attend {
    pub n_head: usize,
    pub n_kv: usize,
    pub hd: usize,
    pub hdv: usize,
    /// Score scale (`1/sqrt(hd)` for the dense families; Gemma 4 passes `f_attention_scale` 1.0).
    pub scale: f32,
    /// Sliding window: a query at position `t` sees keys `p` with `t − p < window` (llama.cpp
    /// `is_masked_swa`, `LLAMA_SWA_TYPE_STANDARD`); `None` = every earlier position.
    pub window: Option<usize>,
}

/// Causal softmax attention of the stacked rows against attention layer `l`: row `t` (sequence
/// `rows.seq[t]`, position `rows.pos[t]`) attends to the visible positions ≤ its own of its own
/// sequence, which the cache must already hold (this pass's rows included): for each row and
/// head, `softmax(scale · q·K) · V`, written to `attn[t][head][hdv]`. Runs per (row, KV head)
/// in parallel over the pool with fp32 accumulation; each cached row is decoded once per task
/// and used by every query head of its group (GQA).
pub(crate) fn attend(
    pool: &ThreadPool,
    kv: &KvCache,
    l: usize,
    rows: &Rows,
    q_all: &[f32],
    a: Attend,
    attn: &mut [f32],
) {
    let Attend {
        n_head,
        n_kv,
        hd,
        hdv,
        scale,
        window,
    } = a;
    let n = rows.n();
    let q_dim = n_head * hd;
    debug_assert_eq!(kv.layer(l).kv_dim, n_kv * hd);
    debug_assert_eq!(kv.layer(l).v_dim, n_kv * hdv);
    let group = n_head / n_kv;
    let readers: Vec<_> = rows.segs.iter().map(|sg| kv.reader(sg.seq, l)).collect();
    let attn_dim = n_head * hdv;
    // Longest score row any query in this batch needs.
    let n_ctx = rows.pos.iter().copied().max().unwrap_or(0) + 1;
    let span = window.map(|w| w.min(n_ctx)).unwrap_or(n_ctx);
    let out_ptr = SendPtr(attn.as_mut_ptr());
    // Few (row, KV head) tasks over a long context — a decode step with 4–8 KV heads — would
    // leave cores idle: split each task's key range across workers (flash-decoding) and merge the
    // partial softmaxes. Aim for two items per thread so the rounds balance (8 tasks on 12
    // threads: 3 splits, 24 items; a plain `threads / tasks` gave 1, leaving 4 threads idle).
    // Prompts have enough tasks already.
    let tasks = n * n_kv;
    let threads = pool.n_threads();
    let splits = if tasks < 2 * threads && span >= 2 * SPLIT_MIN_KEYS {
        (2 * threads)
            .div_ceil(tasks)
            .min(span / SPLIT_MIN_KEYS)
            .max(1)
    } else {
        1
    };
    if splits > 1 {
        attend_split(
            pool, &readers, rows, q_all, a, splits, span, out_ptr, attn_dim,
        );
        return;
    }
    pool.parallel_for(n * n_kv, None, |start, end| {
        let mut scores = vec![0f32; group * span];
        let mut row = vec![0f32; hd.max(hdv)];
        let mut acc = vec![0f32; group * hdv];
        for idx in start..end {
            let t = idx / n_kv;
            let kvh = idx % n_kv;
            let rd = &readers[rows.seg_of[t]];
            let t_abs = rows.pos[t];
            let p_lo = window.map(|w| (t_abs + 1).saturating_sub(w)).unwrap_or(0);
            let n_pos = t_abs + 1 - p_lo;
            for (i, p) in (p_lo..=t_abs).enumerate() {
                rd.load_k(p, kvh * hd, &mut row[..hd]);
                for g in 0..group {
                    let h = kvh * group + g;
                    let q = &q_all[t * q_dim + h * hd..t * q_dim + (h + 1) * hd];
                    scores[g * span + i] = dot(q, &row[..hd]) * scale;
                }
            }
            for g in 0..group {
                softmax(&mut scores[g * span..g * span + n_pos]);
            }
            acc.iter_mut().for_each(|a| *a = 0.0);
            for (i, p) in (p_lo..=t_abs).enumerate() {
                rd.load_v(p, kvh * hdv, &mut row[..hdv]);
                for g in 0..group {
                    let w = scores[g * span + i];
                    let a = &mut acc[g * hdv..(g + 1) * hdv];
                    for j in 0..hdv {
                        a[j] += w * row[j];
                    }
                }
            }
            // SAFETY: each (t, kvh) writes the disjoint hdv-wide slices of its own query heads
            // kvh*group .. (kvh+1)*group in row t of `attn`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    acc.as_ptr(),
                    out_ptr.get().add(t * attn_dim + kvh * group * hdv),
                    group * hdv,
                );
            }
        }
    });
}

/// Keys per split below which splitting a decode task's key range does not pay.
const SPLIT_MIN_KEYS: usize = 128;

/// [`attend`] with each (row, KV head) task's visible keys cut into `splits` contiguous ranges:
/// every range computes, per query head of the group, its running max `m`, its sum `l` of
/// `exp(s − m)` and its unnormalised `Σ exp(s − m)·v`; the ranges then merge exactly
/// (`m* = max m_r`, `out = Σ e^{m_r − m*} o_r / Σ e^{m_r − m*} l_r`), the usual log-sum-exp merge.
#[allow(clippy::too_many_arguments)]
fn attend_split(
    pool: &ThreadPool,
    readers: &[crate::kv::LayerReader],
    rows: &Rows,
    q_all: &[f32],
    a: Attend,
    splits: usize,
    span: usize,
    out_ptr: SendPtr,
    attn_dim: usize,
) {
    let Attend {
        n_head,
        n_kv,
        hd,
        hdv,
        scale,
        window,
    } = a;
    let n = rows.n();
    let q_dim = n_head * hd;
    let group = n_head / n_kv;
    // Partials per (task, split): `group × (hdv + 2)` floats (o, m, l per head).
    let stride = group * (hdv + 2);
    let mut part = vec![0f32; n * n_kv * splits * stride];
    let part_ptr = SendPtr(part.as_mut_ptr());
    let chunk = span.div_ceil(splits);
    pool.parallel_for(n * n_kv * splits, None, |start, end| {
        let mut scores = vec![0f32; group * chunk];
        let mut row = vec![0f32; hd.max(hdv)];
        for idx in start..end {
            let sp = idx % splits;
            let task = idx / splits;
            let t = task / n_kv;
            let kvh = task % n_kv;
            let rd = &readers[rows.seg_of[t]];
            let t_abs = rows.pos[t];
            let p_lo = window.map(|w| (t_abs + 1).saturating_sub(w)).unwrap_or(0);
            let r0 = p_lo + sp * chunk;
            let r1 = (r0 + chunk).min(t_abs + 1);
            // SAFETY: every (task, split) owns its own `stride`-wide slice of `part`.
            let out =
                unsafe { std::slice::from_raw_parts_mut(part_ptr.get().add(idx * stride), stride) };
            if r0 >= r1 {
                for g in 0..group {
                    out[g * (hdv + 2) + hdv] = f32::NEG_INFINITY;
                    out[g * (hdv + 2) + hdv + 1] = 0.0;
                }
                continue;
            }
            let m_pos = r1 - r0;
            for (i, p) in (r0..r1).enumerate() {
                rd.load_k(p, kvh * hd, &mut row[..hd]);
                for g in 0..group {
                    let h = kvh * group + g;
                    let q = &q_all[t * q_dim + h * hd..t * q_dim + (h + 1) * hd];
                    scores[g * chunk + i] = dot(q, &row[..hd]) * scale;
                }
            }
            for g in 0..group {
                let sc = &mut scores[g * chunk..g * chunk + m_pos];
                let m = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut l = 0f32;
                for v in sc.iter_mut() {
                    *v = (*v - m).exp();
                    l += *v;
                }
                let o = &mut out[g * (hdv + 2)..(g + 1) * (hdv + 2)];
                o[..hdv].iter_mut().for_each(|v| *v = 0.0);
                o[hdv] = m;
                o[hdv + 1] = l;
            }
            for (i, p) in (r0..r1).enumerate() {
                rd.load_v(p, kvh * hdv, &mut row[..hdv]);
                for g in 0..group {
                    let w = scores[g * chunk + i];
                    let o = &mut out[g * (hdv + 2)..g * (hdv + 2) + hdv];
                    for j in 0..hdv {
                        o[j] += w * row[j];
                    }
                }
            }
        }
    });
    // Merge the splits of every (task, head).
    let mut merged = vec![0f32; hdv];
    for task in 0..n * n_kv {
        let (t, kvh) = (task / n_kv, task % n_kv);
        for g in 0..group {
            let at = |sp: usize| (task * splits + sp) * stride + g * (hdv + 2);
            let m_star = (0..splits)
                .map(|sp| part[at(sp) + hdv])
                .fold(f32::NEG_INFINITY, f32::max);
            merged.iter_mut().for_each(|v| *v = 0.0);
            let mut l_star = 0f32;
            for sp in 0..splits {
                let o = &part[at(sp)..at(sp) + hdv + 2];
                if o[hdv] == f32::NEG_INFINITY {
                    continue;
                }
                let c = (o[hdv] - m_star).exp();
                l_star += c * o[hdv + 1];
                for j in 0..hdv {
                    merged[j] += c * o[j];
                }
            }
            let inv = if l_star > 0.0 { 1.0 / l_star } else { 0.0 };
            // SAFETY: the parallel region is over; this thread alone writes row t's heads.
            let dst = unsafe {
                std::slice::from_raw_parts_mut(
                    out_ptr.get().add(t * attn_dim + (kvh * group + g) * hdv),
                    hdv,
                )
            };
            for j in 0..hdv {
                dst[j] = merged[j] * inv;
            }
        }
    }
}

/// `y = W x (+ b)` for `n` tokens of width `cols`.
pub(crate) fn project(
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

pub(crate) fn per_head_norm(x: &mut [f32], hd: usize, w: &[f32], eps: f32) {
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
        tiny_model_hd(dir, 8)
    }

    /// The tiny model with `hd`-wide heads (32 for the q8_0 cache, whose blocks span 32 values).
    fn tiny_model_hd(dir: &std::path::Path, hd: u64) -> std::path::PathBuf {
        let d = 32u64;
        let n_head = 4u64;
        let n_kv = 2u64;
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
            .meta("llama.attention.key_length", MetaValue::U32(hd as u32))
            .meta("llama.attention.value_length", MetaValue::U32(hd as u32))
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
        let p = dir.join(format!("tiny_hd{hd}.gguf"));
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
        assert_eq!(kv_a.len(0), 5);
        assert_eq!(kv_b.len(0), 5);
        for (a, b) in la.iter().zip(&lb) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
        // Scratch accounting is consistent with the allocation.
        let floats = |s: &Scratch| {
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
        };
        assert_eq!(Scratch::bytes(&m.spec, 8), floats(&s_a));
        let s3 = Scratch::with_seqs(&m.spec, 8, 3);
        assert_eq!(Scratch::bytes_with_seqs(&m.spec, 8, 3), floats(&s3));
        let _ = EngineError::Format(String::new());
    }

    /// A batch of several sequences gives each sequence exactly the logits it gets alone, and
    /// leaves each sequence's cache as if it had run alone (checked by continuing to decode).
    #[test]
    fn batched_sequences_equal_separate_runs() {
        use crate::kv::{KvOptions, KvType};
        let dir = tempfile::tempdir().unwrap();
        let path = tiny_model_hd(dir.path(), 32);
        let f = GgufFile::open(&path).unwrap();
        let m = Model::load(&f).unwrap();
        let pool = ThreadPool::new(2);
        let prompts: [&[u32]; 3] = [
            &[3, 17, 5, 42, 9],
            &[8, 1],
            &[60, 61, 62, 63, 0, 2, 4, 6, 8],
        ];
        for kv_type in [KvType::F16, KvType::Q8_0] {
            // Alone, each in its own single-sequence cache (block size 4 forces several blocks).
            let opts = KvOptions::new(32).kv_type(kv_type).block_tokens(4);
            let mut alone = Vec::new();
            for p in prompts {
                let mut kv = KvCache::with_options(&m.spec, opts);
                let mut s = Scratch::new(&m.spec, 16);
                let first = m.forward(&pool, &mut kv, p, &mut s).to_vec();
                let next = m.forward(&pool, &mut kv, &[7], &mut s).to_vec();
                alone.push((first, next));
            }
            // Together: one prefill call with all three, then one decode call with all three.
            let mut kv = KvCache::with_options(&m.spec, opts.seqs(3));
            let mut s = Scratch::with_seqs(&m.spec, 16, 3);
            let batch: Vec<SeqTokens> = prompts
                .iter()
                .enumerate()
                .map(|(i, p)| SeqTokens { seq: i, tokens: p })
                .collect();
            let first = m
                .forward_batch(&pool, &mut kv, &batch, &mut s)
                .unwrap()
                .to_vec();
            let dec: Vec<SeqTokens> = (0..3)
                .map(|i| SeqTokens {
                    seq: i,
                    tokens: &[7],
                })
                .collect();
            let next = m
                .forward_batch(&pool, &mut kv, &dec, &mut s)
                .unwrap()
                .to_vec();
            let vocab = m.spec.n_vocab as usize;
            for (i, (fa, na)) in alone.iter().enumerate() {
                for (a, b) in fa.iter().zip(&first[i * vocab..(i + 1) * vocab]) {
                    assert!(
                        (a - b).abs() < 1e-5,
                        "{kv_type:?} prefill seq {i}: {a} vs {b}"
                    );
                }
                for (a, b) in na.iter().zip(&next[i * vocab..(i + 1) * vocab]) {
                    assert!(
                        (a - b).abs() < 1e-5,
                        "{kv_type:?} decode seq {i}: {a} vs {b}"
                    );
                }
                assert_eq!(kv.len(i), prompts[i].len() + 1);
            }
        }
    }

    /// Decode steps over a long context split each (row, KV head) task's keys across workers
    /// when there are fewer tasks than threads; the merged result equals the unsplit one (a
    /// 1-thread pool never splits).
    #[test]
    fn split_key_attention_equals_unsplit() {
        let dir = tempfile::tempdir().unwrap();
        let path = tiny_model(dir.path());
        let f = GgufFile::open(&path).unwrap();
        let mut m = Model::load(&f).unwrap();
        m.spec.context_length = 512;
        let prompt: Vec<u32> = (0..300).map(|i| (i * 13 + 5) % 64).collect();
        let run = |threads: usize| -> Vec<Vec<f32>> {
            let pool = ThreadPool::new(threads);
            let mut kv = KvCache::new(&m.spec, 512);
            let mut s = Scratch::new(&m.spec, 512);
            m.forward(&pool, &mut kv, &prompt, &mut s);
            (0..3)
                .map(|i| {
                    m.forward(&pool, &mut kv, &[(i * 7) as u32], &mut s)
                        .to_vec()
                })
                .collect()
        };
        let a = run(1);
        let b = run(8);
        for (x, y) in a.iter().zip(&b) {
            for (u, v) in x.iter().zip(y) {
                assert!((u - v).abs() < 1e-4, "{u} vs {v}");
            }
        }
    }

    /// A snapshot carries a sequence's whole state: imported into another cache (another slot,
    /// another process) it continues exactly like the original.
    #[test]
    fn snapshot_round_trip_continues_identically() {
        use crate::kv::KvOptions;
        let dir = tempfile::tempdir().unwrap();
        let path = tiny_model(dir.path());
        let f = GgufFile::open(&path).unwrap();
        let m = Model::load(&f).unwrap();
        let pool = ThreadPool::new(2);
        let opts = KvOptions::new(64).block_tokens(4).seqs(2);
        let mut a = KvCache::with_options(&m.spec, opts);
        let mut s = Scratch::with_seqs(&m.spec, 16, 2);
        let prompt: Vec<u32> = (0..11).map(|i| (i * 5 + 3) % 64).collect();
        m.forward(&pool, &mut a, &prompt, &mut s);
        let snap = a.export(0);
        assert_eq!(snap.len, 11);
        assert_eq!(snap.bytes.len(), 3 * a.layout.block_bytes);
        assert!(a.snapshot_trimmable());
        let want = m.forward(&pool, &mut a, &[7], &mut s).to_vec();
        // Into slot 1 of a fresh cache with the same layout.
        let mut b = KvCache::with_options(&m.spec, opts);
        assert_eq!(a.fingerprint(), b.fingerprint());
        b.import(1, &snap).unwrap();
        assert_eq!(b.len(1), 11);
        let got = m
            .forward_batch(
                &pool,
                &mut b,
                &[SeqTokens {
                    seq: 1,
                    tokens: &[7],
                }],
                &mut s,
            )
            .unwrap()
            .to_vec();
        assert_eq!(want, got);
        // A different layout has a different fingerprint.
        let c = KvCache::with_options(&m.spec, opts.block_tokens(8));
        assert_ne!(a.fingerprint(), c.fingerprint());
    }

    /// Blocks are backed only while used: growth maps them, truncation and clearing give them
    /// back, and a full shared pool refuses a batch without changing any sequence.
    #[test]
    fn blocks_follow_use_and_full_pool_refuses_cleanly() {
        use crate::kv::KvOptions;
        let dir = tempfile::tempdir().unwrap();
        let path = tiny_model(dir.path());
        let f = GgufFile::open(&path).unwrap();
        let m = Model::load(&f).unwrap();
        let pool = ThreadPool::new(2);
        // Two sequences of up to 16 positions sharing a pool of 16 positions: 4 blocks of 4,
        // plus one copy-on-write spare per sequence = 6 blocks.
        let opts = KvOptions::new(16).block_tokens(4).seqs(2).pool_tokens(16);
        let mut kv = KvCache::with_options(&m.spec, opts);
        assert_eq!(kv.layout.pool_blocks, 6);
        let mut s = Scratch::with_seqs(&m.spec, 16, 2);
        assert_eq!(
            kv.resident_blocks(),
            0,
            "nothing is backed before the first token"
        );
        m.forward(&pool, &mut kv, &[1, 2, 3, 4, 5], &mut s);
        assert_eq!(kv.resident_blocks(), 2);
        kv.truncate(0, 4);
        assert_eq!(kv.resident_blocks(), 1);
        kv.clear(0);
        assert_eq!(kv.resident_blocks(), 0);
        // Sequence 0 takes the whole context (4 blocks); sequence 1 cannot get 4 more.
        let full: Vec<u32> = (0..16).collect();
        m.forward(&pool, &mut kv, &full, &mut s);
        assert_eq!(kv.pool().free_blocks(), 2);
        let both = [
            SeqTokens {
                seq: 0,
                tokens: &[9],
            },
            SeqTokens {
                seq: 1,
                tokens: &full,
            },
        ];
        let err = m
            .forward_batch(&pool, &mut kv, &both[1..], &mut s)
            .unwrap_err();
        assert_eq!(err.needed, 4);
        assert_eq!(
            (kv.len(0), kv.len(1)),
            (16, 0),
            "a refused batch changes nothing"
        );
        assert_eq!(kv.pool().free_blocks(), 2);
        assert_eq!(kv.resident_blocks(), 4);
        // Half of it fits.
        m.forward_batch(
            &pool,
            &mut kv,
            &[SeqTokens {
                seq: 1,
                tokens: &full[..8],
            }],
            &mut s,
        )
        .unwrap();
        assert_eq!(kv.pool().free_blocks(), 0);
        kv.clear(1);
        assert_eq!(kv.pool().free_blocks(), 2);
        assert_eq!(kv.resident_blocks(), 4);
        let _ = both[0];
    }
}
