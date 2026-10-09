//! Gated DeltaNet kernels (Qwen3.5 / Qwen3-Next linear-attention layers), plain Rust, fp32.
//!
//! Every equation below was taken from the primary sources, not from memory:
//!
//! - llama.cpp `src/models/qwen35.cpp` (`build_layer_attn_linear`, `build_norm_gated`),
//!   `src/models/delta-net-base.cpp` (`build_delta_net_autoregressive`, `build_conv_state`),
//!   `src/models/models.h` (`build_gdn_l2_norm`), `ggml/src/ggml-cpu/ops.cpp`
//!   (`ggml_compute_forward_ssm_conv_f32`), `ggml/src/ggml-cpu/unary-ops.cpp` (`op_softplus`),
//!   `conversion/qwen.py` (`Qwen3NextModel.modify_tensors`, `_LinearAttentionVReorderBase`).
//! - HuggingFace `modeling_qwen3_5.py` / `modeling_qwen3_next.py`
//!   (`Qwen3_5GatedDeltaNet.forward`, `torch_recurrent_gated_delta_rule`, `l2norm`,
//!   `Qwen3_5RMSNormGated`, `causal_conv1d_update`).
//!
//! Per DeltaNet layer, with `x` the RMS-normed residual stream of one token
//! (`H_k` QK heads of width `D` = `head_k`, `H_v` V heads of width `D` = `head_v`,
//! `key_dim = H_k·D`, `value_dim = H_v·D`, `conv_dim = 2·key_dim + value_dim`):
//!
//! 1. Projections (all without bias):
//!    `qkv = W_qkv x` (`conv_dim`, laid out `[q (key_dim) | k (key_dim) | v (value_dim)]`),
//!    `z = W_z x` (`value_dim`), `a = W_alpha x` (`H_v`), `b = W_beta x` (`H_v`).
//! 2. Causal depthwise conv over the last `d_conv` inputs of every channel, then SiLU:
//!    `c[t][ch] = silu( Σ_{j=0..d_conv-1} w[ch][j] · u[t − (d_conv − 1) + j][ch] )`
//!    with `u` the *pre-activation* `qkv` stream (zeros before the sequence start). The cache
//!    keeps the last `d_conv − 1` inputs per channel (ggml `ssm_conv` on
//!    `concat(conv_state, qkv)`; HF `causal_conv1d_update`). `ssm_conv1d.weight` is
//!    `[conv_dim][d_conv]` (ggml `ne = [d_conv, conv_dim]`) and has no bias in this family.
//! 3. Split `c` into `q`, `k` (per QK head) and `v` (per V head); L2-normalise each `q`/`k` head:
//!    `q̂ = q / sqrt(Σ q² + eps)` with `eps = rms_eps` (1e-6). llama.cpp's `build_gdn_l2_norm`
//!    is `rms_norm(x, eps/D) · 1/sqrt(D)` = `x / sqrt(D·mean(x²) + eps)`, identical to HF
//!    `l2norm(x, eps=1e-6)`.
//! 4. Gates per V head: `β = sigmoid(b)`, `g = A · softplus(a + dt_bias)` where `A = ssm_a =
//!    −exp(A_log)` (the converter stores `-torch.exp(A_log)`; HF: `g = -A_log.exp() *
//!    softplus(a + dt_bias)`), `softplus(x) = x > 20 ? x : ln(1 + eˣ)` (ggml `op_softplus`).
//! 5. The recurrence, per V head `h` with QK head `h % H_k` (ggml `repeat_4d` tiles the QK heads
//!    over the V heads; the converter re-orders the V heads of the GGUF into that tiled order,
//!    `_LinearAttentionVReorderBase`), state `S ∈ ℝ^{D×D}` indexed `S[i_k][j_v]`, scaled query
//!    `q' = q̂ / sqrt(D)`:
//!    ```text
//!    S      ← S · exp(g)
//!    kv[j]   = Σ_i S[i][j] · k̂[i]
//!    Δ[j]    = (v[j] − kv[j]) · β
//!    S[i][j] ← S[i][j] + k̂[i] · Δ[j]
//!    o[j]    = Σ_i S[i][j] · q'[i]
//!    ```
//!    (llama.cpp `build_delta_net_autoregressive`; HF `torch_recurrent_gated_delta_rule`). Prefill
//!    runs this step token by token (the chunked form is an optimisation, not a different result).
//! 6. Gated RMSNorm per V head: `y = rms_norm(o, w_norm, eps) · silu(z)` (`build_norm_gated`; HF
//!    `Qwen3_5RMSNormGated`: norm, multiply by the raw weight — the converter does not add 1 to
//!    `linear_attn.norm.weight` — then multiply by `silu(gate)`).
//! 7. `out = W_out y`, added to the residual stream.

use llmario_engine_cpu::ops::silu;
use llmario_engine_cpu::{rms_norm, ThreadPool};

/// ggml `op_softplus`: `x > 20 ? x : ln(1 + eˣ)`.
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Causal depthwise conv + SiLU over `n` new tokens with the per-sequence history, updating the
/// history in place (equation 2).
///
/// - `state`: `[conv_dim][d_conv − 1]`, oldest input first; zero at sequence start.
/// - `x`: `[n][conv_dim]` pre-activation inputs; `w`: `[conv_dim][d_conv]`.
/// - `out`: `[n][conv_dim]`.
pub fn conv_silu(
    state: &mut [f32],
    x: &[f32],
    w: &[f32],
    n: usize,
    conv_dim: usize,
    d_conv: usize,
    out: &mut [f32],
) {
    let hist = d_conv - 1;
    debug_assert_eq!(state.len(), conv_dim * hist);
    debug_assert_eq!(x.len(), n * conv_dim);
    debug_assert_eq!(w.len(), conv_dim * d_conv);
    debug_assert_eq!(out.len(), n * conv_dim);
    let mut window = vec![0f32; hist + n];
    for ch in 0..conv_dim {
        let st = &mut state[ch * hist..(ch + 1) * hist];
        window[..hist].copy_from_slice(st);
        for t in 0..n {
            window[hist + t] = x[t * conv_dim + ch];
        }
        let wc = &w[ch * d_conv..(ch + 1) * d_conv];
        for t in 0..n {
            let mut acc = 0f32;
            for j in 0..d_conv {
                acc += window[t + j] * wc[j];
            }
            out[t * conv_dim + ch] = silu(acc);
        }
        // Keep the last `hist` inputs for the next call.
        st.copy_from_slice(&window[n..n + hist]);
    }
}

/// `x / sqrt(Σ x² + eps)` for each `head_dim`-wide head of `x` (equation 3).
pub fn l2_norm_heads(x: &mut [f32], head_dim: usize, eps: f32) {
    for head in x.chunks_exact_mut(head_dim) {
        let mut ss = 0f32;
        for &v in head.iter() {
            ss += v * v;
        }
        let inv = 1.0 / (ss + eps).sqrt();
        for v in head.iter_mut() {
            *v *= inv;
        }
    }
}

/// Geometry of one scan call.
#[derive(Clone, Copy, Debug)]
pub struct GdnDims {
    pub n_k_heads: usize,
    pub n_v_heads: usize,
    /// Head width `D` (key and value heads share it in this family).
    pub head_dim: usize,
}

/// The gated delta rule over `n` tokens, token by token, for every V head in parallel over the
/// pool (equation 5). The state is updated in place.
///
/// - `state`: `[n_v_heads][D][D]` as `S[i_k][j_v]`.
/// - `q`, `k`: `[n][n_k_heads·D]`, already L2-normalised (the `1/sqrt(D)` query scale is applied
///   here, as llama.cpp's `build_delta_net_*` does).
/// - `v`: `[n][n_v_heads·D]`; `g`, `beta`: `[n][n_v_heads]` (`g` is the log decay, ≤ 0).
/// - `out`: `[n][n_v_heads·D]`.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_scan(
    pool: &ThreadPool,
    dims: GdnDims,
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    n: usize,
    out: &mut [f32],
) {
    let d = dims.head_dim;
    let hk = dims.n_k_heads;
    let hv = dims.n_v_heads;
    debug_assert_eq!(state.len(), hv * d * d);
    debug_assert_eq!(q.len(), n * hk * d);
    debug_assert_eq!(k.len(), n * hk * d);
    debug_assert_eq!(v.len(), n * hv * d);
    debug_assert_eq!(g.len(), n * hv);
    debug_assert_eq!(beta.len(), n * hv);
    debug_assert_eq!(out.len(), n * hv * d);
    let scale = 1.0 / (d as f32).sqrt();
    let state_ptr = SendPtr(state.as_mut_ptr());
    let out_ptr = SendPtr(out.as_mut_ptr());
    pool.parallel_for(hv, Some(hv), move |start, end| {
        let mut kv = vec![0f32; d];
        for h in start..end {
            let kh = h % hk;
            // SAFETY: each V head `h` owns the disjoint state block `[h·D·D, (h+1)·D·D)` and the
            // disjoint output columns `[h·D, (h+1)·D)` of every token; no two chunks touch the
            // same element, and both slices outlive the region (`parallel_for` blocks).
            let s =
                unsafe { std::slice::from_raw_parts_mut(state_ptr.get().add(h * d * d), d * d) };
            for t in 0..n {
                let qt = &q[t * hk * d + kh * d..t * hk * d + (kh + 1) * d];
                let kt = &k[t * hk * d + kh * d..t * hk * d + (kh + 1) * d];
                let vt = &v[t * hv * d + h * d..t * hv * d + (h + 1) * d];
                let decay = g[t * hv + h].exp();
                let b = beta[t * hv + h];
                // S ← S·exp(g);  kv = Sᵀ k̂
                kv.iter_mut().for_each(|x| *x = 0.0);
                for i in 0..d {
                    let row = &mut s[i * d..(i + 1) * d];
                    let ki = kt[i];
                    for j in 0..d {
                        row[j] *= decay;
                        kv[j] += row[j] * ki;
                    }
                }
                // Δ = (v − kv)·β;  S ← S + k̂ Δᵀ;  o = Sᵀ q'
                for j in 0..d {
                    kv[j] = (vt[j] - kv[j]) * b;
                }
                let o = unsafe {
                    std::slice::from_raw_parts_mut(out_ptr.get().add(t * hv * d + h * d), d)
                };
                o.iter_mut().for_each(|x| *x = 0.0);
                for i in 0..d {
                    let row = &mut s[i * d..(i + 1) * d];
                    let ki = kt[i];
                    let qi = qt[i] * scale;
                    for j in 0..d {
                        row[j] += ki * kv[j];
                        o[j] += row[j] * qi;
                    }
                }
            }
        }
    });
}

/// Gated RMSNorm per V head, in place (equation 6): `x_h = rms_norm(x_h, w, eps) · silu(z_h)`.
pub fn gated_rms_norm(x: &mut [f32], z: &[f32], w: &[f32], head_dim: usize, eps: f32) {
    debug_assert_eq!(x.len(), z.len());
    debug_assert_eq!(w.len(), head_dim);
    let mut tmp = vec![0f32; head_dim];
    for (head, zh) in x.chunks_exact_mut(head_dim).zip(z.chunks_exact(head_dim)) {
        rms_norm(head, w, eps, &mut tmp);
        for j in 0..head_dim {
            head[j] = tmp[j] * silu(zh[j]);
        }
    }
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
// SAFETY: see the per-use comments; writes are partitioned by V head into disjoint ranges.
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

    /// Straightforward reference of equation 5 (one head at a time, explicit matrices), used to
    /// check the parallel scan.
    #[allow(clippy::too_many_arguments)]
    fn reference_scan(
        dims: GdnDims,
        state: &mut [f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        g: &[f32],
        beta: &[f32],
        n: usize,
    ) -> Vec<f32> {
        let d = dims.head_dim;
        let (hk, hv) = (dims.n_k_heads, dims.n_v_heads);
        let mut out = vec![0f32; n * hv * d];
        for t in 0..n {
            for h in 0..hv {
                let kh = h % hk;
                let s = &mut state[h * d * d..(h + 1) * d * d];
                let qt: Vec<f32> = q[t * hk * d + kh * d..t * hk * d + (kh + 1) * d]
                    .iter()
                    .map(|x| x / (d as f32).sqrt())
                    .collect();
                let kt = &k[t * hk * d + kh * d..t * hk * d + (kh + 1) * d];
                let vt = &v[t * hv * d + h * d..t * hv * d + (h + 1) * d];
                let decay = g[t * hv + h].exp();
                for x in s.iter_mut() {
                    *x *= decay;
                }
                let mut kv = vec![0f32; d];
                for i in 0..d {
                    for j in 0..d {
                        kv[j] += s[i * d + j] * kt[i];
                    }
                }
                let delta: Vec<f32> = (0..d).map(|j| (vt[j] - kv[j]) * beta[t * hv + h]).collect();
                for i in 0..d {
                    for j in 0..d {
                        s[i * d + j] += kt[i] * delta[j];
                    }
                }
                for j in 0..d {
                    let mut o = 0f32;
                    for i in 0..d {
                        o += s[i * d + j] * qt[i];
                    }
                    out[t * hv * d + h * d + j] = o;
                }
            }
        }
        out
    }

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }

    #[test]
    fn delta_rule_hand_computed() {
        // One head, D = 2, zero state. Token 1: k = (1, 0), v = (2, 3), β = ½, g = 0:
        //   kv = 0, Δ = (1, 1.5), S = [[1, 1.5], [0, 0]], o = Sᵀ q/√2 with q = (1, 0) → (1, 1.5)/√2.
        // Token 2: g = ln ½ → S = [[.5, .75], [0, 0]]; k = (0, 1), v = (1, 1), β = 1:
        //   kv = S row 1 = (0, 0), Δ = (1, 1), S = [[.5, .75], [1, 1]];
        //   q = (1, 1) (already unit-free for the test) scaled by 1/√2 → o = (.5+1, .75+1)/√2.
        let pool = ThreadPool::new(1);
        let dims = GdnDims {
            n_k_heads: 1,
            n_v_heads: 1,
            head_dim: 2,
        };
        let mut state = vec![0f32; 4];
        let q = [1.0, 0.0, 1.0, 1.0];
        let k = [1.0, 0.0, 0.0, 1.0];
        let v = [2.0, 3.0, 1.0, 1.0];
        let g = [0.0, (0.5f32).ln()];
        let beta = [0.5, 1.0];
        let mut out = vec![0f32; 4];
        gated_delta_scan(&pool, dims, &mut state, &q, &k, &v, &g, &beta, 2, &mut out);
        let r = std::f32::consts::FRAC_1_SQRT_2;
        let want = [1.0 * r, 1.5 * r, 1.5 * r, 1.75 * r];
        for (a, b) in out.iter().zip(&want) {
            assert!((a - b).abs() < 1e-6, "{out:?} vs {want:?}");
        }
        let want_s = [0.5, 0.75, 1.0, 1.0];
        for (a, b) in state.iter().zip(&want_s) {
            assert!((a - b).abs() < 1e-6, "{state:?}");
        }
    }

    #[test]
    fn delta_rule_matches_reference_and_is_stepwise_consistent() {
        let pool = ThreadPool::new(3);
        let dims = GdnDims {
            n_k_heads: 2,
            n_v_heads: 6,
            head_dim: 8,
        };
        let n = 7;
        let d = dims.head_dim;
        let mut seed = 42u64;
        let mut q: Vec<f32> = (0..n * dims.n_k_heads * d)
            .map(|_| lcg(&mut seed))
            .collect();
        let mut k: Vec<f32> = (0..n * dims.n_k_heads * d)
            .map(|_| lcg(&mut seed))
            .collect();
        l2_norm_heads(&mut q, d, 1e-6);
        l2_norm_heads(&mut k, d, 1e-6);
        let v: Vec<f32> = (0..n * dims.n_v_heads * d)
            .map(|_| lcg(&mut seed))
            .collect();
        let g: Vec<f32> = (0..n * dims.n_v_heads)
            .map(|_| -(lcg(&mut seed).abs() * 2.0))
            .collect();
        let beta: Vec<f32> = (0..n * dims.n_v_heads)
            .map(|_| sigmoid(lcg(&mut seed) * 3.0))
            .collect();
        let s0: Vec<f32> = (0..dims.n_v_heads * d * d)
            .map(|_| lcg(&mut seed) * 0.1)
            .collect();

        // Reference, all tokens.
        let mut s_ref = s0.clone();
        let o_ref = reference_scan(dims, &mut s_ref, &q, &k, &v, &g, &beta, n);
        // Parallel scan, all tokens at once.
        let mut s_a = s0.clone();
        let mut o_a = vec![0f32; n * dims.n_v_heads * d];
        gated_delta_scan(&pool, dims, &mut s_a, &q, &k, &v, &g, &beta, n, &mut o_a);
        // Parallel scan, one token per call (decode path).
        let mut s_b = s0.clone();
        let mut o_b = vec![0f32; n * dims.n_v_heads * d];
        for t in 0..n {
            let hk = dims.n_k_heads * d;
            let hv = dims.n_v_heads * d;
            gated_delta_scan(
                &pool,
                dims,
                &mut s_b,
                &q[t * hk..(t + 1) * hk],
                &k[t * hk..(t + 1) * hk],
                &v[t * hv..(t + 1) * hv],
                &g[t * dims.n_v_heads..(t + 1) * dims.n_v_heads],
                &beta[t * dims.n_v_heads..(t + 1) * dims.n_v_heads],
                1,
                &mut o_b[t * hv..(t + 1) * hv],
            );
        }
        for i in 0..o_ref.len() {
            assert!(
                (o_ref[i] - o_a[i]).abs() < 1e-5,
                "out[{i}]: {} vs {}",
                o_ref[i],
                o_a[i]
            );
            assert!((o_ref[i] - o_b[i]).abs() < 1e-5, "step out[{i}]");
        }
        for i in 0..s_ref.len() {
            assert!((s_ref[i] - s_a[i]).abs() < 1e-5, "state[{i}]");
            assert!((s_ref[i] - s_b[i]).abs() < 1e-5, "step state[{i}]");
        }
    }

    #[test]
    fn conv_matches_zero_padded_full_conv_and_chunking() {
        let conv_dim = 5;
        let d_conv = 4;
        let n = 9;
        let mut seed = 7u64;
        let x: Vec<f32> = (0..n * conv_dim).map(|_| lcg(&mut seed)).collect();
        let w: Vec<f32> = (0..conv_dim * d_conv).map(|_| lcg(&mut seed)).collect();
        // Reference: HF causal_conv1d_fn = conv1d with left zero padding of d_conv-1, then SiLU.
        let mut want = vec![0f32; n * conv_dim];
        for t in 0..n {
            for ch in 0..conv_dim {
                let mut acc = 0f32;
                for j in 0..d_conv {
                    let src = t as isize - (d_conv as isize - 1) + j as isize;
                    if src >= 0 {
                        acc += x[src as usize * conv_dim + ch] * w[ch * d_conv + j];
                    }
                }
                want[t * conv_dim + ch] = silu(acc);
            }
        }
        // All at once.
        let mut st = vec![0f32; conv_dim * (d_conv - 1)];
        let mut out = vec![0f32; n * conv_dim];
        conv_silu(&mut st, &x, &w, n, conv_dim, d_conv, &mut out);
        for i in 0..out.len() {
            assert!((out[i] - want[i]).abs() < 1e-6, "{i}");
        }
        // In chunks of 4, 1, 4 tokens through the state.
        let mut st2 = vec![0f32; conv_dim * (d_conv - 1)];
        let mut out2 = vec![0f32; n * conv_dim];
        let mut t0 = 0;
        for len in [4usize, 1, 4] {
            conv_silu(
                &mut st2,
                &x[t0 * conv_dim..(t0 + len) * conv_dim],
                &w,
                len,
                conv_dim,
                d_conv,
                &mut out2[t0 * conv_dim..(t0 + len) * conv_dim],
            );
            t0 += len;
        }
        for i in 0..out.len() {
            assert!((out2[i] - want[i]).abs() < 1e-6, "chunked {i}");
        }
        assert_eq!(st, st2);
        // The state holds the last d_conv-1 raw inputs per channel, oldest first.
        for ch in 0..conv_dim {
            for j in 0..d_conv - 1 {
                assert_eq!(
                    st[ch * (d_conv - 1) + j],
                    x[(n - (d_conv - 1) + j) * conv_dim + ch]
                );
            }
        }
    }

    #[test]
    fn l2_norm_and_gates() {
        let mut x = vec![3.0, 4.0, 0.0, 0.0];
        l2_norm_heads(&mut x, 2, 1e-6);
        assert!((x[0] - 0.6).abs() < 1e-6 && (x[1] - 0.8).abs() < 1e-6);
        // A zero head stays zero (eps keeps the divisor finite, as HF's l2norm does).
        assert_eq!(x[2], 0.0);
        assert!((softplus(0.0) - (2f32).ln()).abs() < 1e-6);
        assert_eq!(softplus(25.0), 25.0);
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-7);
        // Gated norm: unit weight, zero eps, z = 0 → silu(0) = 0 → all zero.
        let mut y = vec![1.0, 2.0];
        gated_rms_norm(&mut y, &[0.0, 0.0], &[1.0, 1.0], 2, 0.0);
        assert_eq!(y, vec![0.0, 0.0]);
        let mut y = vec![3.0, 4.0];
        gated_rms_norm(&mut y, &[10.0, 10.0], &[1.0, 1.0], 2, 0.0);
        // rms = sqrt(12.5); silu(10) ≈ 10.
        let rms = 12.5f32.sqrt();
        assert!((y[0] - 3.0 / rms * silu(10.0)).abs() < 1e-5);
    }
}
