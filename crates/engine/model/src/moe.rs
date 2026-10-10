//! Routed mixture-of-experts FFN on the CPU (Architecture §9.4, milestone M4): the Qwen3 MoE
//! family (`qwen3moe`).
//!
//! Semantics follow llama.cpp `llm_graph_context::build_moe_ffn` with the arguments
//! `src/models/qwen3moe.cpp` passes (`LLM_FFN_SILU`, `norm_w = true`,
//! `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX`, no expert biases, no weight scale), per token `x`
//! (the RMS-normed residual):
//!
//! ```text
//! p      = softmax(W_router x)                       // over all n_expert experts, f32
//! S      = top_k(p, n_expert_used)                   // ggml_argsort_top_k: descending
//! w_i    = p[S_i] / max(Σ_j p[S_j], 6.103515625e-5)  // norm_w: renormalised, clamped sum
//! e_i    = W_down[S_i] · (silu(W_gate[S_i] x) ⊙ W_up[S_i] x) · w_i
//! out    = ((e_0 + e_1) + e_2) + …                   // summed in selection order
//! ```
//!
//! Experts are grouped: every expert any token of the batch selected runs once over the rows
//! that chose it (a matmul for a prompt, a matvec for one decode token), so a prompt reads each
//! expert's weights once. Expert matrices are slices of the mapped file: the OS keeps the hot
//! ones resident and may drop cold ones under memory pressure, reading them back from disk when a
//! token selects them again. When the plan streams experts, the selected experts' reads start
//! right after routing (`stream.rs`).

use crate::arch::{ArchSpec, MoeSpec};
use crate::forward::project;
use crate::stream::{self, Reader, Recent, SimCache, StreamOptions};
use crate::weights::MoeWeights;
use llmario_engine_cpu::ops::{add_inplace, swiglu_inplace};
use llmario_engine_cpu::{rows_multi, softmax, RowsJob, ThreadPool};

/// Lower bound of the selected weights' sum (llama.cpp clamps to the smallest normal f16).
const WEIGHT_SUM_MIN: f32 = 6.103_515_6e-5;

/// Working buffers of the MoE FFN for up to `n` rows. A (row `t`, selection slot `j`) pair is
/// `p = t·k + j` everywhere.
pub(crate) struct MoeScratch {
    /// `[n][n_expert]` router logits.
    logits: Vec<f32>,
    /// `[n][k]` selected experts and their weights.
    sel: Vec<u32>,
    w: Vec<f32>,
    /// `[n·k][n_ff_exp]` gate / up activations per pair.
    gate: Vec<f32>,
    up: Vec<f32>,
    /// `[n·k][d]` expert outputs per pair.
    acc: Vec<f32>,
    /// Per-expert pair lists (reused across calls).
    pairs: Vec<Vec<u32>>,
    /// Expert residency (see `stream.rs`): reads started after routing (skipping experts used
    /// recently), and the simulation of a smaller page cache.
    reader: Option<(Reader, Recent)>,
    pub(crate) sim: Option<SimCache>,
}

impl MoeScratch {
    pub fn new(spec: &ArchSpec, m: &MoeSpec, n: usize) -> MoeScratch {
        let d = spec.d_model as usize;
        let k = m.n_expert_used as usize;
        let ff = m.n_ff_exp as usize;
        MoeScratch {
            logits: vec![0.0; n * m.n_expert as usize],
            sel: vec![0; n * k],
            w: vec![0.0; n * k],
            gate: vec![0.0; n * k * ff],
            up: vec![0.0; n * k * ff],
            acc: vec![0.0; n * k * d],
            pairs: vec![Vec::new(); m.n_expert as usize],
            reader: None,
            sim: None,
        }
    }

    /// Apply `o` for the model in `file` with layers `layers`.
    pub(crate) fn set_stream(
        &mut self,
        o: StreamOptions,
        file: &llmario_engine_formats::GgufFile,
        m: &MoeSpec,
        layers: &[crate::weights::LayerWeights],
    ) {
        let ne = m.n_expert as usize;
        let moe_layers = layers.iter().filter(|l| l.moe.is_some()).count();
        self.reader = o.prefetch.then(|| {
            let window = 3 * moe_layers * m.n_expert_used as usize;
            (
                Reader::new(file),
                Recent::new(layers.len() * ne, window as u64),
            )
        });
        self.sim = o.sim_resident.map(|frac| {
            let mut total = 0u64;
            // The simulated cache starts empty, so the real one must too.
            for mw in layers.iter().filter_map(|l| l.moe.as_ref()) {
                total += (expert_bytes(mw) * ne) as u64;
                for e in 0..ne {
                    stream::evict(mw.gate.expert(e).data);
                    stream::evict(mw.up.expert(e).data);
                    stream::evict(mw.down.expert(e).data);
                }
            }
            SimCache::new(
                layers.len() * ne,
                (f64::from(frac.clamp(0.0, 1.0)) * total as f64) as u64,
            )
        });
    }

    /// End of a step (`rows` rows in all): the simulation evicts what its budget does not hold.
    pub(crate) fn end_step(
        &mut self,
        m: &MoeSpec,
        layers: &[crate::weights::LayerWeights],
        rows: usize,
    ) {
        let Some(sim) = self.sim.as_mut() else {
            return;
        };
        let ne = m.n_expert as usize;
        sim.end_step(rows == 1, |slot| {
            if let Some(mw) = &layers[slot / ne].moe {
                let e = slot % ne;
                stream::evict(mw.gate.expert(e).data);
                stream::evict(mw.up.expert(e).data);
                stream::evict(mw.down.expert(e).data);
            }
        });
    }

    pub fn bytes(spec: &ArchSpec, m: &MoeSpec, n: usize) -> u64 {
        let (n, d, k, ff, ne) = (
            n as u64,
            spec.d_model as u64,
            m.n_expert_used as u64,
            m.n_ff_exp as u64,
            m.n_expert as u64,
        );
        n * (ne + 2 * k + 2 * k * ff + k * d) * 4
    }
}

/// Pick the top-`k` experts of one token from its router logits and weight them.
pub(crate) fn route(m: &MoeSpec, logits: &[f32], sel: &mut [u32], w: &mut [f32]) {
    let ne = m.n_expert as usize;
    let k = m.n_expert_used as usize;
    let mut p = logits[..ne].to_vec();
    softmax(&mut p);
    let mut idx: Vec<u32> = (0..ne as u32).collect();
    // Descending probability; ties keep the lower expert id first.
    let cmp = |a: &u32, b: &u32| p[*b as usize].total_cmp(&p[*a as usize]).then(a.cmp(b));
    if k < ne {
        idx.select_nth_unstable_by(k - 1, cmp);
    }
    idx[..k].sort_by(cmp);
    for j in 0..k {
        sel[j] = idx[j];
        w[j] = p[idx[j] as usize];
    }
    if m.norm_topk {
        let sum = w[..k].iter().sum::<f32>().max(WEIGHT_SUM_MIN);
        for v in &mut w[..k] {
            *v /= sum;
        }
    }
}

/// `x[t] += MoE(h[t])` for the `n` stacked rows (`h` is the RMS-normed residual).
///
/// Every expert that any row selected is one product over the rows that chose it, and all of
/// them run in one parallel dispatch for gate and up and a second for down
/// ([`llmario_engine_cpu::rows_multi`]): a decode step reads each selected expert once, a prompt
/// reads each expert once for all the rows that chose it, and neither pays a pool round trip per
/// expert matrix.
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_ffn(
    spec: &ArchSpec,
    m: &MoeSpec,
    mw: &MoeWeights,
    pool: &ThreadPool,
    l: usize,
    n: usize,
    h: &[f32],
    x: &mut [f32],
    s: &mut MoeScratch,
) {
    let d = spec.d_model as usize;
    let ne = m.n_expert as usize;
    let k = m.n_expert_used as usize;
    let ff = m.n_ff_exp as usize;
    let npairs = n * k;

    // Router and selection.
    project(pool, &mw.gate_inp, h, n, d, &mut s.logits[..n * ne], None);
    for pl in &mut s.pairs {
        pl.clear();
    }
    for t in 0..n {
        route(
            m,
            &s.logits[t * ne..(t + 1) * ne],
            &mut s.sel[t * k..(t + 1) * k],
            &mut s.w[t * k..(t + 1) * k],
        );
        for j in 0..k {
            s.pairs[s.sel[t * k + j] as usize].push((t * k + j) as u32);
        }
    }
    let used: Vec<usize> = (0..ne).filter(|&e| !s.pairs[e].is_empty()).collect();
    if let Some(sim) = s.sim.as_mut() {
        for &e in &used {
            sim.touch(l * ne + e, expert_bytes(mw) as u64);
        }
    }
    // Every selected expert's reads start now, all at once, instead of as page faults inside the
    // products below.
    if let Some((r, recent)) = s.reader.as_mut() {
        for &e in &used {
            if recent.use_slot(l * ne + e) {
                r.fetch(mw.gate.expert(e).data);
                r.fetch(mw.up.expert(e).data);
                r.fetch(mw.down.expert(e).data);
            }
        }
        recent.advance(npairs as u64);
    }

    // Gate and up: one product per used expert over its rows, all in one dispatch.
    {
        let mut gate_rows: Vec<Option<&mut [f32]>> =
            s.gate[..npairs * ff].chunks_mut(ff).map(Some).collect();
        let mut up_rows: Vec<Option<&mut [f32]>> =
            s.up[..npairs * ff].chunks_mut(ff).map(Some).collect();
        let mut jobs: Vec<RowsJob> = Vec::with_capacity(2 * used.len());
        for &e in &used {
            let pl = &s.pairs[e];
            let xs: Vec<&[f32]> = pl
                .iter()
                .map(|&p| {
                    let t = p as usize / k;
                    &h[t * d..(t + 1) * d]
                })
                .collect();
            jobs.push(RowsJob {
                w: mw.gate.expert(e),
                xs: xs.clone(),
                ys: pl
                    .iter()
                    .map(|&p| gate_rows[p as usize].take().expect("each pair once"))
                    .collect(),
            });
            jobs.push(RowsJob {
                w: mw.up.expert(e),
                xs,
                ys: pl
                    .iter()
                    .map(|&p| up_rows[p as usize].take().expect("each pair once"))
                    .collect(),
            });
        }
        rows_multi(pool, &mut jobs);
    }
    swiglu_inplace(&mut s.gate[..npairs * ff], &s.up[..npairs * ff]);

    // Down: each used expert over its pairs' hidden rows, written to the pairs' output rows.
    {
        let mut out_rows: Vec<Option<&mut [f32]>> =
            s.acc[..npairs * d].chunks_mut(d).map(Some).collect();
        let mut jobs: Vec<RowsJob> = Vec::with_capacity(used.len());
        for &e in &used {
            let pl = &s.pairs[e];
            jobs.push(RowsJob {
                w: mw.down.expert(e),
                xs: pl
                    .iter()
                    .map(|&p| &s.gate[p as usize * ff..(p as usize + 1) * ff])
                    .collect(),
                ys: pl
                    .iter()
                    .map(|&p| out_rows[p as usize].take().expect("each pair once"))
                    .collect(),
            });
        }
        rows_multi(pool, &mut jobs);
    }

    // Weight each pair's output, sum the slots in selection order, add to the residual.
    let mut sum = vec![0f32; d];
    for t in 0..n {
        for j in 0..k {
            let p = t * k + j;
            let wt = s.w[p];
            for v in &mut s.acc[p * d..(p + 1) * d] {
                *v *= wt;
            }
        }
        sum.copy_from_slice(&s.acc[t * k * d..(t * k + 1) * d]);
        for j in 1..k {
            add_inplace(&mut sum, &s.acc[(t * k + j) * d..(t * k + j + 1) * d]);
        }
        add_inplace(&mut x[t * d..(t + 1) * d], &sum);
    }
}

/// Bytes of one expert's gate, up and down matrices.
fn expert_bytes(mw: &MoeWeights) -> usize {
    mw.gate.expert_bytes() + mw.up.expert_bytes() + mw.down.expert_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::{Model, Scratch, SeqTokens};
    use crate::kv::{KvCache, KvOptions};
    use crate::Family;
    use llmario_engine_core::GgmlType;
    use llmario_engine_formats::gguf::writer::GgufWriter;
    use llmario_engine_formats::{GgufFile, MetaValue};

    const D: u64 = 32;
    const NE: u64 = 4;
    const K: u64 = 2;
    const FF: u64 = 16;

    /// A tiny random `qwen3moe` model: two layers, 4 experts with 2 used per token.
    fn tiny_moe(dir: &std::path::Path) -> std::path::PathBuf {
        let (n_head, n_kv, hd, vocab, n_layer) = (4u64, 2u64, 8u64, 64u64, 2u64);
        let f32s = |n: u64, seed: u64| -> Vec<u8> {
            (0..n)
                .flat_map(|i| {
                    let v = (((i * 2654435761 + seed * 97) % 1000) as f32 / 1000.0 - 0.5) * 0.4;
                    v.to_le_bytes()
                })
                .collect()
        };
        let ones = |n: u64| -> Vec<u8> { (0..n).flat_map(|_| 1.0f32.to_le_bytes()).collect() };
        let mut w = GgufWriter::new();
        w.meta("general.architecture", MetaValue::Str("qwen3moe".into()))
            .meta("qwen3moe.block_count", MetaValue::U32(n_layer as u32))
            .meta("qwen3moe.embedding_length", MetaValue::U32(D as u32))
            .meta(
                "qwen3moe.attention.head_count",
                MetaValue::U32(n_head as u32),
            )
            .meta(
                "qwen3moe.attention.head_count_kv",
                MetaValue::U32(n_kv as u32),
            )
            .meta("qwen3moe.attention.key_length", MetaValue::U32(hd as u32))
            .meta("qwen3moe.attention.value_length", MetaValue::U32(hd as u32))
            .meta("qwen3moe.expert_count", MetaValue::U32(NE as u32))
            .meta("qwen3moe.expert_used_count", MetaValue::U32(K as u32))
            .meta(
                "qwen3moe.expert_feed_forward_length",
                MetaValue::U32(FF as u32),
            )
            .meta("qwen3moe.vocab_size", MetaValue::U32(vocab as u32))
            .meta("qwen3moe.context_length", MetaValue::U32(64))
            .meta("qwen3moe.rope.freq_base", MetaValue::F32(1_000_000.0))
            .meta(
                "qwen3moe.attention.layer_norm_rms_epsilon",
                MetaValue::F32(1e-6),
            );
        w.tensor(
            "token_embd.weight",
            &[D, vocab],
            GgmlType::F32,
            f32s(D * vocab, 1),
        );
        w.tensor("output_norm.weight", &[D], GgmlType::F32, ones(D));
        w.tensor(
            "output.weight",
            &[D, vocab],
            GgmlType::F32,
            f32s(D * vocab, 2),
        );
        for l in 0..n_layer {
            let p = |s: &str| format!("blk.{l}.{s}");
            w.tensor(&p("attn_norm.weight"), &[D], GgmlType::F32, ones(D));
            w.tensor(
                &p("attn_q.weight"),
                &[D, n_head * hd],
                GgmlType::F32,
                f32s(D * n_head * hd, 10 + l),
            );
            w.tensor(
                &p("attn_k.weight"),
                &[D, n_kv * hd],
                GgmlType::F32,
                f32s(D * n_kv * hd, 20 + l),
            );
            w.tensor(
                &p("attn_v.weight"),
                &[D, n_kv * hd],
                GgmlType::F32,
                f32s(D * n_kv * hd, 30 + l),
            );
            w.tensor(
                &p("attn_output.weight"),
                &[n_head * hd, D],
                GgmlType::F32,
                f32s(n_head * hd * D, 40 + l),
            );
            w.tensor(&p("attn_q_norm.weight"), &[hd], GgmlType::F32, ones(hd));
            w.tensor(&p("attn_k_norm.weight"), &[hd], GgmlType::F32, ones(hd));
            w.tensor(&p("ffn_norm.weight"), &[D], GgmlType::F32, ones(D));
            w.tensor(
                &p("ffn_gate_inp.weight"),
                &[D, NE],
                GgmlType::F32,
                f32s(D * NE, 50 + l),
            );
            w.tensor(
                &p("ffn_gate_exps.weight"),
                &[D, FF, NE],
                GgmlType::F32,
                f32s(D * FF * NE, 60 + l),
            );
            w.tensor(
                &p("ffn_up_exps.weight"),
                &[D, FF, NE],
                GgmlType::F32,
                f32s(D * FF * NE, 70 + l),
            );
            w.tensor(
                &p("ffn_down_exps.weight"),
                &[FF, D, NE],
                GgmlType::F32,
                f32s(FF * D * NE, 80 + l),
            );
        }
        let p = dir.join("tiny_moe.gguf");
        std::fs::write(&p, w.to_bytes()).unwrap();
        p
    }

    /// `y = W x` in f64 for an f32 expert matrix.
    fn naive(w: &llmario_engine_cpu::QMat, x: &[f32]) -> Vec<f32> {
        (0..w.rows)
            .map(|r| {
                let row = w.row(r);
                (0..w.cols)
                    .map(|c| {
                        let v = f32::from_le_bytes(row[c * 4..c * 4 + 4].try_into().unwrap());
                        v as f64 * x[c] as f64
                    })
                    .sum::<f64>() as f32
            })
            .collect()
    }

    #[test]
    fn moe_ffn_matches_a_naive_reference() {
        let dir = tempfile::tempdir().unwrap();
        let f = GgufFile::open(&tiny_moe(dir.path())).unwrap();
        let m = Model::load(&f).unwrap();
        assert_eq!(m.spec.family, Family::Qwen3Moe);
        let ms = m.spec.moe.clone().unwrap();
        assert_eq!((ms.n_expert, ms.n_expert_used, ms.n_ff_exp), (4, 2, 16));
        let mw = m.weights.layers[0].moe.as_ref().unwrap();
        let pool = ThreadPool::new(2);
        // A decode-sized batch and a prompt-sized one (experts shared between rows).
        for n in [3usize, 12] {
            let h: Vec<f32> = (0..n * D as usize)
                .map(|i| ((i as f32) * 0.37).sin())
                .collect();
            let x0: Vec<f32> = (0..n * D as usize).map(|i| (i as f32) * 0.01).collect();
            let mut x = x0.clone();
            let mut s = MoeScratch::new(&m.spec, &ms, n);
            moe_ffn(&m.spec, &ms, mw, &pool, 0, n, &h, &mut x, &mut s);
            let d = D as usize;
            for t in 0..n {
                let ht = &h[t * d..(t + 1) * d];
                let logits = naive(&mw.gate_inp, ht);
                let (mut sel, mut wts) = ([0u32; 2], [0f32; 2]);
                route(&ms, &logits, &mut sel, &mut wts);
                let mut want = x0[t * d..(t + 1) * d].to_vec();
                for j in 0..2 {
                    let e = sel[j] as usize;
                    let g = naive(&mw.gate.expert(e), ht);
                    let u = naive(&mw.up.expert(e), ht);
                    let hid: Vec<f32> = g
                        .iter()
                        .zip(&u)
                        .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
                        .collect();
                    let o = naive(&mw.down.expert(e), &hid);
                    for i in 0..d {
                        want[i] += o[i] * wts[j];
                    }
                }
                for i in 0..d {
                    assert!(
                        (x[t * d + i] - want[i]).abs() < 1e-4,
                        "n {n} row {t} dim {i}: {} vs {}",
                        x[t * d + i],
                        want[i]
                    );
                }
            }
        }
    }

    #[test]
    fn moe_prefill_equals_token_by_token_and_batches_like_separate_runs() {
        let dir = tempfile::tempdir().unwrap();
        let f = GgufFile::open(&tiny_moe(dir.path())).unwrap();
        let m = Model::load(&f).unwrap();
        let pool = ThreadPool::new(2);
        let toks = [3u32, 17, 5, 42, 9, 11];
        let mut kv_a = KvCache::new(&m.spec, 32);
        let mut s_a = Scratch::new(&m.spec, 8);
        let la = m.forward(&pool, &mut kv_a, &toks, &mut s_a).to_vec();
        let mut kv_b = KvCache::new(&m.spec, 32);
        let mut s_b = Scratch::new(&m.spec, 1);
        let mut lb = vec![];
        for &t in &toks {
            lb = m.forward(&pool, &mut kv_b, &[t], &mut s_b).to_vec();
        }
        for (a, b) in la.iter().zip(&lb) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
        // Two sequences in one call (their tokens route to different experts in one grouped
        // pass) equal separate runs.
        let p2: [&[u32]; 2] = [&toks, &[60, 1]];
        let mut kv = KvCache::with_options(&m.spec, KvOptions::new(32).seqs(2));
        let mut s = Scratch::with_seqs(&m.spec, 16, 2);
        let batch: Vec<SeqTokens> = p2
            .iter()
            .enumerate()
            .map(|(i, p)| SeqTokens { seq: i, tokens: p })
            .collect();
        let both = m
            .forward_batch(&pool, &mut kv, &batch, &mut s)
            .unwrap()
            .to_vec();
        let mut kv2 = KvCache::new(&m.spec, 32);
        let alone = m.forward(&pool, &mut kv2, p2[1], &mut s_a).to_vec();
        let v = m.spec.n_vocab as usize;
        for (a, b) in la.iter().zip(&both[..v]) {
            assert!((a - b).abs() < 1e-4);
        }
        for (a, b) in alone.iter().zip(&both[v..]) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    fn spec(ne: u32, k: u32, norm: bool) -> MoeSpec {
        MoeSpec {
            n_expert: ne,
            n_expert_used: k,
            n_ff_exp: 4,
            norm_topk: norm,
        }
    }

    #[test]
    fn route_picks_top_k_descending_and_renormalises() {
        let logits = [0.1f32, 2.0, -1.0, 1.5, 0.3];
        let (mut sel, mut w) = ([0u32; 2], [0f32; 2]);
        route(&spec(5, 2, true), &logits, &mut sel, &mut w);
        assert_eq!(sel, [1, 3]);
        assert!((w[0] + w[1] - 1.0).abs() < 1e-6);
        // Ratio equals the ratio of the softmax probabilities.
        assert!((w[0] / w[1] - (2.0f32 - 1.5).exp()).abs() < 1e-5);
        // Without renormalisation the weights are the softmax probabilities themselves.
        route(&spec(5, 2, false), &logits, &mut sel, &mut w);
        let z: f32 = logits.iter().map(|v| v.exp()).sum();
        assert!((w[0] - 2.0f32.exp() / z).abs() < 1e-6);
        // k == n_expert: every expert, still in descending order.
        let (mut sel, mut w) = ([0u32; 5], [0f32; 5]);
        route(&spec(5, 5, true), &logits, &mut sel, &mut w);
        assert_eq!(sel, [1, 3, 4, 0, 2]);
    }

    #[test]
    fn ties_prefer_the_lower_expert() {
        let (mut sel, mut w) = ([0u32; 2], [0f32; 2]);
        route(&spec(4, 2, true), &[1.0, 3.0, 3.0, 3.0], &mut sel, &mut w);
        assert_eq!(sel, [1, 2]);
    }
}
