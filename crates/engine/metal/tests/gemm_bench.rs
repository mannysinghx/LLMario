//! Prefill GEMM throughput at Qwen3-1.7B's matrix shapes (prints TFLOPS; never asserts on speed).
//! Run with `cargo test --release -p llmario-engine-metal --test gemm_bench -- --ignored --nocapture`;
//! `GEMM_KERNELS=gemm,gemmf` picks the kernel families compared (default: both).

#![cfg(target_os = "macos")]

use llmario_engine_core::GgmlType;
use llmario_engine_metal::device::{groups, Gpu};

#[repr(C)]
#[derive(Clone, Copy)]
struct GemmParams {
    rows: u32,
    cols: u32,
    n: u32,
    row_bytes: u32,
    accumulate: u32,
}

fn lcg_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as u8
        })
        .collect()
}

/// Random blocks with small finite f16 scales.
fn weights(dtype: GgmlType, rows: usize, cols: usize) -> Vec<u8> {
    let mut b = lcg_bytes(rows * dtype.row_bytes(cols), 7);
    let bb = dtype.block_bytes();
    let h = |v: f32| half::f16::from_f32(v).to_le_bytes();
    for blk in b.chunks_exact_mut(bb) {
        match dtype {
            GgmlType::Q4_K => {
                blk[0..2].copy_from_slice(&h(0.003));
                blk[2..4].copy_from_slice(&h(0.001));
            }
            GgmlType::Q6_K => blk[208..210].copy_from_slice(&h(0.002)),
            _ => unreachable!(),
        }
    }
    b
}

#[test]
#[ignore]
fn gemm_throughput() {
    let Ok(gpu) = Gpu::get() else {
        return;
    };
    let n = 512usize;
    let kernels: Vec<&'static str> = std::env::var("GEMM_KERNELS")
        .map(|s| {
            s.split(',')
                .map(|k| &*Box::leak(k.to_string().into_boxed_str()))
                .collect()
        })
        .unwrap_or_else(|_| vec!["gemm", "gemmf"]);
    for (dtype, rows, cols, label) in [
        (
            GgmlType::Q4_K,
            6144usize,
            2048usize,
            "gate/up q4_k 6144x2048",
        ),
        (GgmlType::Q4_K, 2048, 2048, "q/o q4_k 2048x2048"),
        (GgmlType::Q6_K, 2048, 6144, "down q6_k 2048x6144"),
    ] {
        let w = weights(dtype, rows, cols);
        let wbuf = gpu.alloc(w.len()).unwrap();
        wbuf.write_bytes(0, &w);
        let x: Vec<f32> = (0..n * cols)
            .map(|i| (i * 48271 % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let xbuf = gpu.alloc(x.len() * 4).unwrap();
        xbuf.write_f32(0, &x);
        let ybuf = gpu.alloc(n * rows * 4).unwrap();
        let suffix = if dtype == GgmlType::Q4_K {
            "q4_k"
        } else {
            "q6_k"
        };
        for k in &kernels {
            let name: &'static str = Box::leak(format!("{k}_{suffix}").into_boxed_str());
            let p = GemmParams {
                rows: rows as u32,
                cols: cols as u32,
                n: n as u32,
                row_bytes: dtype.row_bytes(cols) as u32,
                accumulate: 0,
            };
            let (bm, bn) = (64, 32);
            let mut best = f64::MAX;
            for _ in 0..12 {
                let cmd = gpu.begin().unwrap();
                for _ in 0..10 {
                    cmd.dispatch(
                        name,
                        &[(0, &wbuf, 0), (1, &xbuf, 0), (2, &ybuf, 0)],
                        3,
                        &p,
                        (groups(n, bn), groups(rows, bm), 1),
                        (128, 1, 1),
                    )
                    .unwrap();
                }
                best = best.min(cmd.finish().unwrap() / 10.0);
            }
            let tflops = 2.0 * (rows * cols * n) as f64 / best / 1e12;
            eprintln!(
                "{label:28} {name:22} {:7.3} ms  {tflops:5.2} TFLOPS",
                best * 1e3
            );
        }
    }
}
