//! Cross-checks of every kernel set against the dequantising scalar reference, bit-exactness of
//! the SIMD sets against the portable int8 set, edge shapes, and a gated timing test
//! (`LLMARIO_BENCH=1 cargo test -p llmario-engine-cpu --release -- --nocapture bench`).

use super::*;
use crate::{dequant_row, QMat, ThreadPool};
use half::f16;
use llmario_engine_core::GgmlType;
use std::time::Instant;

/// Deterministic xorshift64* so failures reproduce.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 56) as u8
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
    /// Uniform magnitude in [lo, hi) with a random sign.
    fn signed(&mut self, lo: f32, hi: f32) -> f32 {
        let m = lo + (hi - lo) * (self.unit() + 1.0) * 0.5;
        if self.next() & 1 == 0 {
            m
        } else {
            -m
        }
    }
    fn f16(&mut self, lo: f32, hi: f32) -> [u8; 2] {
        f16::from_f32(self.signed(lo, hi)).to_le_bytes()
    }
    fn fill(&mut self, dst: &mut [u8]) {
        for b in dst {
            *b = self.byte();
        }
    }
}

/// Every weight type the int8 kernels cover.
const TYPES: [GgmlType; 11] = [
    GgmlType::Q4_0,
    GgmlType::Q4_1,
    GgmlType::Q5_0,
    GgmlType::Q5_1,
    GgmlType::Q8_0,
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
    GgmlType::F16,
    GgmlType::BF16,
    GgmlType::F32,
];

/// A random, valid `rows × cols` matrix of `t` whose dequantised values are O(1).
fn random_matrix(t: GgmlType, rows: usize, cols: usize, rng: &mut Rng) -> Vec<u8> {
    let rb = t.row_bytes(cols);
    let mut data = vec![0u8; rows * rb];
    let bb = t.block_bytes();
    for blk in data.chunks_exact_mut(bb) {
        match t {
            GgmlType::F32 => blk.copy_from_slice(&rng.unit().to_le_bytes()),
            GgmlType::F16 => blk.copy_from_slice(&f16::from_f32(rng.unit()).to_le_bytes()),
            GgmlType::BF16 => blk.copy_from_slice(&rng.unit().to_le_bytes()[2..4]),
            GgmlType::Q4_0 => {
                blk[0..2].copy_from_slice(&rng.f16(0.05, 0.2));
                rng.fill(&mut blk[2..]);
            }
            GgmlType::Q4_1 => {
                blk[0..2].copy_from_slice(&rng.f16(0.02, 0.1));
                blk[2..4].copy_from_slice(&rng.f16(0.1, 0.6));
                rng.fill(&mut blk[4..]);
            }
            GgmlType::Q5_0 => {
                blk[0..2].copy_from_slice(&rng.f16(0.02, 0.1));
                rng.fill(&mut blk[2..]);
            }
            GgmlType::Q5_1 => {
                blk[0..2].copy_from_slice(&rng.f16(0.01, 0.05));
                blk[2..4].copy_from_slice(&rng.f16(0.1, 0.6));
                rng.fill(&mut blk[4..]);
            }
            GgmlType::Q8_0 => {
                blk[0..2].copy_from_slice(&rng.f16(0.004, 0.012));
                rng.fill(&mut blk[2..]);
            }
            GgmlType::Q4_K => {
                blk[0..2].copy_from_slice(&rng.f16(0.001, 0.004));
                blk[2..4].copy_from_slice(&rng.f16(0.001, 0.004));
                rng.fill(&mut blk[4..]);
            }
            GgmlType::Q5_K => {
                blk[0..2].copy_from_slice(&rng.f16(0.0005, 0.002));
                blk[2..4].copy_from_slice(&rng.f16(0.0005, 0.002));
                rng.fill(&mut blk[4..]);
            }
            GgmlType::Q6_K => {
                rng.fill(&mut blk[0..208]);
                blk[208..210].copy_from_slice(&rng.f16(0.0001, 0.0005));
            }
            other => panic!("no generator for {other}"),
        }
    }
    data
}

fn random_act(n: usize, cols: usize, rng: &mut Rng) -> Vec<f32> {
    (0..n * cols).map(|_| rng.unit()).collect()
}

/// Does `t` go through int8 activation quantisation in the non-scalar sets?
fn quantises_activations(t: GgmlType) -> bool {
    !matches!(t, GgmlType::F16 | GgmlType::BF16 | GgmlType::F32)
}

/// `(W · Q(x), Σ|w_i · Q(x)_i|)` per (token, row) in f64, where `Q` is the activation quantiser
/// the int8 kernels use for `w.dtype` (identity for float weights).
pub(super) fn exact_quantised_reference(w: &QMat, x: &[f32], n: usize) -> (Vec<f64>, Vec<f64>) {
    use super::common::{act_row_bytes, quantize_act, RowKernel};
    use llmario_engine_core::dequant::dequantize_row;
    let cols = w.cols;
    let kind = int8::TABLE
        .lookup(w.dtype)
        .expect("int8 kernel for this type");
    let act_type = match kind {
        RowKernel::Q8K(_) => Some(GgmlType::Q8_K),
        RowKernel::Q80(_) => Some(GgmlType::Q8_0),
        RowKernel::F32(_) => None,
    };
    let mut xq = vec![0f32; n * cols];
    for tkn in 0..n {
        let xt = &x[tkn * cols..(tkn + 1) * cols];
        let dst = &mut xq[tkn * cols..(tkn + 1) * cols];
        match act_type {
            Some(at) => {
                let mut q = vec![0u8; act_row_bytes(&kind, cols)];
                quantize_act(&kind, xt, &mut q);
                dequantize_row(at, &q, dst).unwrap();
            }
            None => dst.copy_from_slice(xt),
        }
    }
    let mut exact = vec![0f64; n * w.rows];
    let mut scale = vec![0f64; n * w.rows];
    let mut wrow = vec![0f32; cols];
    for r in 0..w.rows {
        dequant_row(w, r, &mut wrow);
        for tkn in 0..n {
            let xt = &xq[tkn * cols..(tkn + 1) * cols];
            let mut s = 0f64;
            let mut a = 0f64;
            for (wi, xi) in wrow.iter().zip(xt) {
                let p = *wi as f64 * *xi as f64;
                s += p;
                a += p.abs();
            }
            exact[tkn * w.rows + r] = s;
            scale[tkn * w.rows + r] = a;
        }
    }
    (exact, scale)
}

/// Check one (type, shape, token count) against the scalar reference on every available set,
/// and the SIMD set(s) against `int8` bit for bit.
fn check(t: GgmlType, rows: usize, cols: usize, n: usize, pool: &ThreadPool, rng: &mut Rng) {
    let data = random_matrix(t, rows, cols, rng);
    let w = QMat::new(t, rows, cols, &data);
    let x = random_act(n, cols, rng);
    let mut reference = vec![0f32; n * rows];
    SCALAR.matmul(pool, &w, &x, n, &mut reference);
    // Tolerance scale per (token, row): the l2 norm of the term vector w∘x. Int8 activation
    // quantisation perturbs each x_i by ≤ max|x|/254, which gives a dot error with standard
    // deviation ≈ 0.4 % of that norm; 2.5 % is > 5σ.
    let mut wrow = vec![0f32; cols];
    let mut scale = vec![0f32; n * rows];
    for r in 0..rows {
        dequant_row(&w, r, &mut wrow);
        for tkn in 0..n {
            let xt = &x[tkn * cols..(tkn + 1) * cols];
            let l2: f32 = wrow
                .iter()
                .zip(xt)
                .map(|(a, b)| (a * b) * (a * b))
                .sum::<f32>()
                .sqrt();
            scale[tkn * rows + r] = l2;
        }
    }
    let rel = if quantises_activations(t) {
        2.5e-2
    } else {
        1e-5
    };
    // Strong oracle: the kernels compute exactly W · Q(x) where Q is the activation quantiser,
    // so the f64 dot of the dequantised weights with the dequantised *quantised* activation
    // must match to f32 rounding (the block epilogues are the only f32 arithmetic).
    let (exact, exact_scale) = exact_quantised_reference(&w, &x, n);
    let mut int8_out = vec![0f32; n * rows];
    for set in available().into_iter().skip(1) {
        let mut out = vec![0f32; n * rows];
        if n == 1 {
            set.matvec(pool, &w, &x, &mut out);
        } else {
            set.matmul(pool, &w, &x, n, &mut out);
        }
        for i in 0..n * rows {
            let tol = rel * scale[i] + 1e-6;
            assert!(
                (out[i] - reference[i]).abs() <= tol,
                "{t} {}: rows={rows} cols={cols} n={n} index {i}: got {} want {} (tol {tol})",
                set.name,
                out[i],
                reference[i]
            );
            let tol = 2e-5 * exact_scale[i] + 1e-7;
            assert!(
                (out[i] as f64 - exact[i]).abs() <= tol,
                "{t} {}: rows={rows} cols={cols} n={n} index {i}: got {} but W·Q(x) = {} (tol {tol})",
                set.name,
                out[i],
                exact[i]
            );
        }
        if set.name == "int8" {
            int8_out = out;
        } else {
            for i in 0..n * rows {
                assert_eq!(
                    out[i].to_bits(),
                    int8_out[i].to_bits(),
                    "{t} {}: rows={rows} cols={cols} n={n} index {i}: {} vs int8 {}",
                    set.name,
                    out[i],
                    int8_out[i]
                );
            }
        }
    }
}

#[test]
fn every_type_every_set_matches_the_scalar_reference() {
    let pool = ThreadPool::new(4);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for t in TYPES {
        let blk = t.block_elems().max(32);
        // (rows, cols, tokens): one block, odd rows, a non-multiple of every chunk size, 64 tokens.
        for (rows, cols, n) in [
            (1, blk, 1),
            (5, 2 * blk, 2),
            (17, 4 * blk, 7),
            (33, 2 * blk, 64),
            (70, 3 * blk, 1),
        ] {
            check(t, rows, cols, n, &pool, &mut rng);
        }
    }
}

#[test]
fn float_types_handle_odd_lengths() {
    let pool = ThreadPool::new(2);
    let mut rng = Rng(42);
    for t in [GgmlType::F16, GgmlType::BF16, GgmlType::F32] {
        for cols in [1, 3, 15, 16, 17, 37, 100] {
            check(t, 9, cols, 3, &pool, &mut rng);
        }
    }
}

#[test]
fn block_type_row_lengths_that_leave_a_tail_of_blocks() {
    // 32-element block kernels vectorise four blocks per step: exercise 1, 2, 3 leftover blocks.
    let pool = ThreadPool::new(3);
    let mut rng = Rng(7);
    for t in [
        GgmlType::Q4_0,
        GgmlType::Q4_1,
        GgmlType::Q5_0,
        GgmlType::Q5_1,
        GgmlType::Q8_0,
    ] {
        for nb in [1, 2, 3, 5, 6, 7, 9] {
            check(t, 13, 32 * nb, 2, &pool, &mut rng);
        }
    }
}

#[test]
fn larger_shapes_across_thread_counts() {
    let mut rng = Rng(123);
    for threads in [1, 3, 12] {
        let pool = ThreadPool::new(threads);
        check(GgmlType::Q4_K, 1000, 512, 1, &pool, &mut rng);
        check(GgmlType::Q6_K, 300, 768, 5, &pool, &mut rng);
        check(GgmlType::Q8_0, 257, 1024, 1, &pool, &mut rng);
    }
}

#[test]
fn matvec_and_matmul_with_one_token_agree() {
    let pool = ThreadPool::new(4);
    let mut rng = Rng(99);
    for t in [GgmlType::Q4_K, GgmlType::Q6_K, GgmlType::Q4_0] {
        let data = random_matrix(t, 50, 512, &mut rng);
        let w = QMat::new(t, 50, 512, &data);
        let x = random_act(1, 512, &mut rng);
        for set in available() {
            let mut a = vec![0f32; 50];
            let mut b = vec![0f32; 50];
            set.matvec(&pool, &w, &x, &mut a);
            set.matmul(&pool, &w, &x, 1, &mut b);
            assert_eq!(a, b, "{t} {}", set.name);
        }
    }
}

#[test]
fn unsupported_types_fall_back_to_the_scalar_set() {
    // IQ4_NL has a reference dequantizer but no int8 kernel: every set must still produce the
    // scalar answer.
    let pool = ThreadPool::new(2);
    let mut rng = Rng(5);
    let t = GgmlType::IQ4_NL;
    let mut data = vec![0u8; 7 * t.row_bytes(64)];
    for blk in data.chunks_exact_mut(18) {
        blk[0..2].copy_from_slice(&rng.f16(0.005, 0.01));
        rng.fill(&mut blk[2..]);
    }
    let w = QMat::new(t, 7, 64, &data);
    let x = random_act(3, 64, &mut rng);
    let mut want = vec![0f32; 21];
    SCALAR.matmul(&pool, &w, &x, 3, &mut want);
    for set in available() {
        let mut got = vec![0f32; 21];
        set.matmul(&pool, &w, &x, 3, &mut got);
        assert_eq!(got, want, "{}", set.name);
    }
}

#[test]
fn detect_picks_a_real_set() {
    let k = detect();
    assert!(available().iter().any(|s| s.name == k.name));
    #[cfg(target_arch = "aarch64")]
    if neon::available() {
        assert_eq!(k.name, "neon");
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_f16_conversion_is_exact_for_every_bit_pattern() {
    use std::arch::aarch64::*;
    for base in (0..=0xFFFFu32).step_by(4) {
        let h: [u16; 4] = [
            base as u16,
            base as u16 + 1,
            base as u16 + 2,
            base as u16 + 3,
        ];
        // SAFETY: baseline NEON; reads the 4-element array.
        let got: [f32; 4] = unsafe {
            let v = neon::f16x4_to_f32(vld1_u16(h.as_ptr()));
            let mut out = [0f32; 4];
            vst1q_f32(out.as_mut_ptr(), v);
            out
        };
        for l in 0..4 {
            let want = f16::from_bits(h[l]).to_f32();
            if want.is_nan() {
                assert!(got[l].is_nan(), "{:#06x}", h[l]);
            } else {
                assert_eq!(got[l].to_bits(), want.to_bits(), "{:#06x}", h[l]);
            }
        }
    }
}

/// Every kernel must stay inside the row: put each type's matrix in an allocation that is an
/// exact multiple of the 16 KiB page (large enough that the allocator maps it on its own), so a
/// vector load that runs past the last block faults instead of reading slack.
#[test]
fn kernels_do_not_read_past_the_last_block() {
    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
    const PAGE: usize = 16384;
    let pool = ThreadPool::new(2);
    let mut rng = Rng(77);
    for t in TYPES {
        let cols = t.block_elems().max(32);
        let rb = t.row_bytes(cols);
        let mut rows = PAGE / gcd(rb, PAGE);
        while rows * rb < 4 << 20 {
            rows *= 2;
        }
        let data = random_matrix(t, rows, cols, &mut rng);
        assert_eq!(data.len() % PAGE, 0);
        assert_eq!(data.capacity(), data.len());
        let w = QMat::new(t, rows, cols, &data);
        let x = random_act(2, cols, &mut rng);
        let mut y = vec![0f32; 2 * rows];
        for set in available() {
            set.matvec(&pool, &w, &x[..cols], &mut y[..rows]);
            set.matmul(&pool, &w, &x, 2, &mut y);
        }
    }
}

// ---- timing ----

/// Unified-memory bandwidth of the development machine (Apple M4 Max), for the bound column.
const PEAK_GBPS: f64 = 546.0;

fn bench_enabled() -> bool {
    std::env::var("LLMARIO_BENCH").is_ok()
}

/// Time `f` with a warm-up; returns seconds per call: the fastest of five batches of
/// `iters / 5` calls, which filters interference from other processes on a shared machine.
fn time_calls(iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..iters.div_ceil(10).max(2) {
        f();
    }
    let per_batch = iters.div_ceil(5).max(1);
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t0 = Instant::now();
        for _ in 0..per_batch {
            f();
        }
        best = best.min(t0.elapsed().as_secs_f64() / per_batch as f64);
    }
    best
}

#[test]
fn bench_matvec_shapes() {
    if !bench_enabled() {
        return;
    }
    let set = detect();
    let mut rng = Rng(1);
    println!(
        "\nmatvec, kernel set `{}` (cache-warm: the same matrix every call)",
        set.name
    );
    println!(
        "{:<6} {:>5}x{:<5} {:>7} {:>9} {:>9} {:>10} {:>8}",
        "type", "rows", "cols", "threads", "us/call", "GB/s", "bound(us)", "%peak"
    );
    for t in [GgmlType::Q4_K, GgmlType::Q6_K] {
        for (rows, cols) in [(2048, 2048), (6144, 2048), (2048, 6144)] {
            let data = random_matrix(t, rows, cols, &mut rng);
            let w = QMat::new(t, rows, cols, &data);
            let x = random_act(1, cols, &mut rng);
            let mut y = vec![0f32; rows];
            let bytes = data.len() as f64;
            let bound_us = bytes / (PEAK_GBPS * 1e9) * 1e6;
            for threads in [1usize, 4, 12] {
                let pool = ThreadPool::new(threads);
                let secs = time_calls(300, || set.matvec(&pool, &w, &x, &mut y));
                let gbps = bytes / secs / 1e9;
                println!(
                    "{:<6} {:>5}x{:<5} {:>7} {:>9.1} {:>9.1} {:>10.1} {:>7.0}%",
                    t.name(),
                    rows,
                    cols,
                    threads,
                    secs * 1e6,
                    gbps,
                    bound_us,
                    gbps / PEAK_GBPS * 100.0
                );
            }
        }
    }
}

/// Qwen3-1.7B-shaped decode step: 28 layers of attention + MLP matrices at Q4_K_M's type mix
/// (Q4_K everywhere, Q6_K for `attn_v` and `ffn_down`) plus a Q6_K output head, ≈ 1.1 GB per
/// token, streamed from memory in model order so caches do not help.
#[test]
fn bench_model_decode_step() {
    if !bench_enabled() {
        return;
    }
    let set = detect();
    let mut rng = Rng(2);
    let layer: [(GgmlType, usize, usize); 7] = [
        (GgmlType::Q4_K, 2048, 2048), // attn_q
        (GgmlType::Q4_K, 1024, 2048), // attn_k
        (GgmlType::Q6_K, 1024, 2048), // attn_v
        (GgmlType::Q4_K, 2048, 2048), // attn_output
        (GgmlType::Q4_K, 6144, 2048), // ffn_gate
        (GgmlType::Q4_K, 6144, 2048), // ffn_up
        (GgmlType::Q6_K, 2048, 6144), // ffn_down
    ];
    let mut mats: Vec<(GgmlType, usize, usize, Vec<u8>)> = Vec::new();
    for _ in 0..28 {
        for &(t, rows, cols) in &layer {
            mats.push((t, rows, cols, random_matrix(t, rows, cols, &mut rng)));
        }
    }
    mats.push((
        GgmlType::Q6_K,
        151_936,
        2048,
        random_matrix(GgmlType::Q6_K, 151_936, 2048, &mut rng),
    ));
    let total_bytes: usize = mats.iter().map(|m| m.3.len()).sum();
    let x2048 = random_act(1, 2048, &mut rng);
    let x6144 = random_act(1, 6144, &mut rng);
    let mut y = vec![0f32; 151_936];
    println!(
        "\nmodel-shaped decode step, kernel set `{}`: {} matrices, {:.3} GB per token",
        set.name,
        mats.len(),
        total_bytes as f64 / 1e9
    );
    for threads in [1usize, 4, 8, 12] {
        let pool = ThreadPool::new(threads);
        let secs = time_calls(if threads == 1 { 3 } else { 8 }, || {
            for (t, rows, cols, data) in &mats {
                let w = QMat::new(*t, *rows, *cols, data);
                let x = if *cols == 2048 { &x2048 } else { &x6144 };
                set.matvec(&pool, &w, x, &mut y[..*rows]);
            }
        });
        println!(
            "  threads={:<2} {:>8.2} ms/token  {:>6.1} GB/s  {:>6.1} tok/s-equivalent (bound at {PEAK_GBPS} GB/s: {:.1} tok/s)",
            threads,
            secs * 1e3,
            total_bytes as f64 / secs / 1e9,
            1.0 / secs,
            PEAK_GBPS * 1e9 / total_bytes as f64
        );
    }
}

#[test]
fn bench_matmul_prefill() {
    if !bench_enabled() {
        return;
    }
    let set = detect();
    let mut rng = Rng(3);
    println!("\nmatmul (prefill), kernel set `{}`", set.name);
    for t in [GgmlType::Q4_K, GgmlType::Q6_K] {
        let (rows, cols) = (6144, 2048);
        let data = random_matrix(t, rows, cols, &mut rng);
        let w = QMat::new(t, rows, cols, &data);
        for n in [8usize, 64, 512] {
            let x = random_act(n, cols, &mut rng);
            let mut y = vec![0f32; n * rows];
            for threads in [1usize, 12] {
                let pool = ThreadPool::new(threads);
                let iters = if threads == 1 { 2 } else { 6 };
                let secs = time_calls(iters, || set.matmul(&pool, &w, &x, n, &mut y));
                let macs = (n * rows * cols) as f64;
                println!(
                    "  {:<5} {}x{} n={:<4} threads={:<2} {:>9.2} ms  {:>7.1} GMAC/s  ({:.0} tok/s at 1.7B)",
                    t.name(),
                    rows,
                    cols,
                    n,
                    threads,
                    secs * 1e3,
                    macs / secs / 1e9,
                    macs / secs / 1.7e9
                );
            }
        }
    }
}

/// Calibrations behind the numbers above: achievable CPU read bandwidth, pool round-trip
/// latency, activation quantisation cost and single-core kernel speed on cache-resident rows.
#[test]
fn bench_overheads() {
    if !bench_enabled() {
        return;
    }
    use super::common::Q8_K_BLOCK;
    use llmario_engine_core::dequant::quantize_row_q8_k;
    use std::sync::atomic::{AtomicU64, Ordering};
    println!("\ncalibrations");
    // 1. Plain streaming read of 1 GB with the pool (what any decode kernel is bounded by).
    let buf: Vec<u64> = (0..(1usize << 27)).map(|i| i as u64).collect();
    for threads in [1usize, 4, 8, 12] {
        let pool = ThreadPool::new(threads);
        let sink = AtomicU64::new(0);
        let secs = time_calls(if threads == 1 { 2 } else { 4 }, || {
            pool.parallel_for(buf.len(), Some(threads * 4), |s, e| {
                let mut acc = [0u64; 8];
                for c in buf[s..e].chunks_exact(8) {
                    for (a, v) in acc.iter_mut().zip(c) {
                        *a = a.wrapping_add(*v);
                    }
                }
                sink.fetch_add(
                    acc.iter().fold(0, |x, y| x.wrapping_add(*y)),
                    Ordering::Relaxed,
                );
            });
        });
        println!(
            "  read 1 GB, threads={:<2} {:>6.1} GB/s",
            threads,
            (buf.len() * 8) as f64 / secs / 1e9
        );
    }
    // 2. Pool round trip with an empty body.
    for threads in [4usize, 12] {
        let pool = ThreadPool::new(threads);
        for chunks in [threads, threads * 4] {
            let secs = time_calls(2000, || pool.parallel_for(2048, Some(chunks), |_, _| {}));
            println!(
                "  parallel_for(empty) threads={:<2} chunks={:<3} {:>6.1} us",
                threads,
                chunks,
                secs * 1e6
            );
        }
    }
    // 3. Activation quantisation.
    for cols in [2048usize, 6144] {
        let x = random_act(1, cols, &mut Rng(9));
        let mut q = vec![0u8; cols / 256 * Q8_K_BLOCK];
        let secs = time_calls(2000, || quantize_row_q8_k(&x, &mut q));
        println!("  quantize_row_q8_k({cols}) {:>6.1} us", secs * 1e6);
    }
    // 4. Single-core row-dot speed on L2-resident rows (64 rows x 2048 columns), kernel only:
    //    the activation is quantised once outside the timed loop.
    use super::common::{act_row_bytes, quantize_act, RowKernel};
    let mut rng = Rng(11);
    for t in [
        GgmlType::Q4_K,
        GgmlType::Q6_K,
        GgmlType::Q5_K,
        GgmlType::Q8_0,
        GgmlType::Q4_0,
    ] {
        let (rows, cols) = (64usize, 2048usize);
        let data = random_matrix(t, rows, cols, &mut rng);
        let w = QMat::new(t, rows, cols, &data);
        let x = random_act(1, cols, &mut rng);
        let tables: Vec<(&str, &super::common::DotTable)> = vec![
            ("int8", &int8::TABLE),
            #[cfg(target_arch = "aarch64")]
            ("neon", &neon::TABLE),
        ];
        for (name, table) in tables {
            let kind = table.resolve(t).unwrap();
            let mut act = vec![0u8; act_row_bytes(&kind, cols)];
            quantize_act(&kind, &x, &mut act);
            let mut sink = 0f32;
            let secs = time_calls(400, || {
                for r in 0..rows {
                    sink += match kind {
                        RowKernel::Q8K(f) | RowKernel::Q80(f) => f(w.row(r), &act, cols),
                        RowKernel::F32(f) => f(w.row(r), &x),
                    };
                }
            });
            let blocks = (rows * cols / t.block_elems()) as f64;
            println!(
                "  {:<5} {:<5} 1 thread, cache-resident, kernel only: {:>6.2} ns/block = {:>5.1} cycles at 4.4 GHz ({:>5.1} GB/s per core) [{sink:.0}]",
                t.name(),
                name,
                secs * 1e9 / blocks,
                secs * 4.4e9 / blocks,
                data.len() as f64 / secs / 1e9
            );
            if let Some(f2) = table.lookup2(t) {
                let mut sink = 0f32;
                let secs = time_calls(400, || {
                    for r in (0..rows).step_by(2) {
                        let (a, b) = f2(w.row(r), w.row(r + 1), &act, cols);
                        sink += a + b;
                    }
                });
                println!(
                    "  {:<5} {:<5} 1 thread, cache-resident, two-row kernel: {:>6.2} ns/block = {:>5.1} cycles at 4.4 GHz ({:>5.1} GB/s per core) [{sink:.0}]",
                    t.name(),
                    name,
                    secs * 1e9 / blocks,
                    secs * 4.4e9 / blocks,
                    data.len() as f64 / secs / 1e9
                );
            }
        }
    }
}
