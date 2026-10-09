//! Kernel dispatch and the scalar fallback.
//!
//! [`kernels`] returns the kernel set for this process, chosen once from the CPU's features.
//! Four sets exist, from slowest to fastest:
//!
//! - [`SCALAR`]: dequantises each weight row to f32 and takes an f32 dot product. Correct for every
//!   type the reference dequantizer supports; the oracle for everything below.
//! - [`INT8`] (`simd/int8.rs`): the reference *quantised-activation* recipe without intrinsics.
//!   The activation row is quantised once per op (Q8_K for 256-element block types, Q8_0 for
//!   32-element ones) and each weight row is consumed in its native block layout with integer dot
//!   products scaled by the block deltas. It defines the exact integer sums and the f32
//!   accumulation order that the SIMD sets reproduce bit for bit; it is also the fallback on CPUs
//!   without the detected features.
//! - [`NEON`] (`simd/neon.rs`, aarch64 with dotprod): `vdotq_s32` kernels for Q4_0, Q4_1, Q5_0,
//!   Q5_1, Q8_0, Q4_K, Q5_K, Q6_K, plus F16/BF16/F32.
//! - [`AVX2`] (`simd/avx2.rs`, x86_64 with AVX2+F16C): `maddubs`/`madd` kernels for Q4_0, Q8_0,
//!   Q4_K, Q6_K; the other types fall back to `INT8`.
//!
//! `LLMARIO_CPU_KERNELS=scalar|int8|neon|avx2` overrides detection (a set that is not available on
//! this CPU falls back to detection with a warning on stderr).

use crate::{QMat, ThreadPool};
use llmario_engine_core::dequant::dequantize_row;
use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
pub mod avx2;
pub mod common;
pub mod int8;
#[cfg(target_arch = "aarch64")]
pub mod neon;
#[cfg(test)]
mod tests;

/// A set of matmul kernels; one per ISA level.
pub struct Kernels {
    pub name: &'static str,
    pub matvec: fn(&ThreadPool, &QMat, &[f32], &mut [f32]),
    pub matmul: fn(&ThreadPool, &QMat, &[f32], usize, &mut [f32]),
}

impl Kernels {
    #[inline]
    pub fn matvec(&self, pool: &ThreadPool, w: &QMat, x: &[f32], y: &mut [f32]) {
        (self.matvec)(pool, w, x, y)
    }
    #[inline]
    pub fn matmul(&self, pool: &ThreadPool, w: &QMat, x: &[f32], n: usize, y: &mut [f32]) {
        (self.matmul)(pool, w, x, n, y)
    }
}

static KERNELS: OnceLock<&'static Kernels> = OnceLock::new();

/// The kernel set for this process (detected once). `LLMARIO_CPU_KERNELS=scalar|int8|neon|avx2`
/// forces a set, which the tests and benchmarks use to cross-check the optimised sets.
pub fn kernels() -> &'static Kernels {
    KERNELS.get_or_init(|| match std::env::var("LLMARIO_CPU_KERNELS") {
        Ok(name) => by_name(&name).unwrap_or_else(|| {
            eprintln!("LLMARIO_CPU_KERNELS={name}: unknown or unavailable on this CPU; detecting");
            detect()
        }),
        Err(_) => detect(),
    })
}

/// The set called `name`, if it exists and its ISA features are present on this CPU.
pub fn by_name(name: &str) -> Option<&'static Kernels> {
    match name {
        "scalar" => Some(&SCALAR),
        "int8" => Some(&INT8),
        #[cfg(target_arch = "aarch64")]
        "neon" if neon::available() => Some(&NEON),
        #[cfg(target_arch = "x86_64")]
        "avx2" if avx2::available() => Some(&AVX2),
        _ => None,
    }
}

/// Every set that can run on this CPU, slowest first (the tests cross-check them pairwise).
pub fn available() -> Vec<&'static Kernels> {
    let mut v = vec![&SCALAR, &INT8];
    #[cfg(target_arch = "aarch64")]
    if neon::available() {
        v.push(&NEON);
    }
    #[cfg(target_arch = "x86_64")]
    if avx2::available() {
        v.push(&AVX2);
    }
    v
}

/// The best set for this CPU (Architecture §7.3).
pub fn detect() -> &'static Kernels {
    #[cfg(target_arch = "aarch64")]
    if neon::available() {
        return &NEON;
    }
    #[cfg(target_arch = "x86_64")]
    if avx2::available() {
        return &AVX2;
    }
    &INT8
}

pub static SCALAR: Kernels = Kernels {
    name: "scalar",
    matvec: scalar_matvec,
    matmul: scalar_matmul,
};

/// Portable quantised-activation kernels (no intrinsics).
pub static INT8: Kernels = Kernels {
    name: "int8",
    matvec: int8::matvec,
    matmul: int8::matmul,
};

/// NEON + dotprod kernels.
#[cfg(target_arch = "aarch64")]
pub static NEON: Kernels = Kernels {
    name: "neon",
    matvec: neon::matvec,
    matmul: neon::matmul,
};

/// AVX2 + F16C kernels.
#[cfg(target_arch = "x86_64")]
pub static AVX2: Kernels = Kernels {
    name: "avx2",
    matvec: avx2::matvec,
    matmul: avx2::matmul,
};

/// Rows handled by one parallel chunk; small enough to balance, large enough to amortise.
const ROWS_PER_CHUNK: usize = 16;

fn scalar_matvec(pool: &ThreadPool, w: &QMat, x: &[f32], y: &mut [f32]) {
    let rows = w.rows;
    let cols = w.cols;
    let y_ptr = SendPtr(y.as_mut_ptr());
    pool.parallel_for(rows, Some(rows.div_ceil(ROWS_PER_CHUNK)), |s, e| {
        let mut tmp = vec![0f32; cols];
        for r in s..e {
            dequantize_row(w.dtype, w.row(r), &mut tmp).expect("supported type");
            let mut acc = 0f32;
            for i in 0..cols {
                acc += tmp[i] * x[i];
            }
            // SAFETY: chunks cover disjoint row ranges, so writes never overlap.
            unsafe { *y_ptr.get().add(r) = acc };
        }
    });
}

fn scalar_matmul(pool: &ThreadPool, w: &QMat, x: &[f32], n: usize, y: &mut [f32]) {
    let rows = w.rows;
    let cols = w.cols;
    let y_ptr = SendPtr(y.as_mut_ptr());
    pool.parallel_for(rows, Some(rows.div_ceil(ROWS_PER_CHUNK)), |s, e| {
        let mut tmp = vec![0f32; cols];
        for r in s..e {
            dequantize_row(w.dtype, w.row(r), &mut tmp).expect("supported type");
            for t in 0..n {
                let xt = &x[t * cols..(t + 1) * cols];
                let mut acc = 0f32;
                for i in 0..cols {
                    acc += tmp[i] * xt[i];
                }
                // SAFETY: disjoint rows per chunk; `t * rows + r` is unique per (t, r).
                unsafe { *y_ptr.get().add(t * rows + r) = acc };
            }
        }
    });
}

/// A raw pointer that may cross threads; the kernels guarantee disjoint writes.
#[derive(Clone, Copy)]
pub(crate) struct SendPtr<T>(pub(crate) *mut T);
// SAFETY: every kernel partitions the output by disjoint index ranges before sharing the pointer.
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    /// Accessor so closures capture the whole wrapper (edition-2021 disjoint capture would
    /// otherwise capture the bare `*mut T`, which is not `Sync`).
    #[inline]
    pub(crate) fn get(&self) -> *mut T {
        self.0
    }
}

#[cfg(test)]
mod scalar_tests {
    use super::*;
    use llmario_engine_core::dequant::quantize_row_q8_0;
    use llmario_engine_core::GgmlType;

    #[test]
    fn scalar_matvec_q8_0_matches_f32_math() {
        let rows = 7;
        let cols = 64;
        let mut wf = vec![0f32; rows * cols];
        for (i, v) in wf.iter_mut().enumerate() {
            *v = ((i * 37 % 97) as f32 - 48.0) / 48.0;
        }
        let mut wq = vec![0u8; rows * GgmlType::Q8_0.row_bytes(cols)];
        for r in 0..rows {
            quantize_row_q8_0(
                &wf[r * cols..(r + 1) * cols],
                &mut wq[r * 34 * 2..(r + 1) * 34 * 2],
            );
        }
        let w = QMat::new(GgmlType::Q8_0, rows, cols, &wq);
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.1).sin()).collect();
        let pool = ThreadPool::new(3);
        let mut y = vec![0f32; rows];
        scalar_matvec(&pool, &w, &x, &mut y);
        for r in 0..rows {
            let exact: f32 = (0..cols).map(|i| wf[r * cols + i] * x[i]).sum();
            assert!((y[r] - exact).abs() < 0.05, "row {r}: {} vs {exact}", y[r]);
        }
        // matmul with n=2 reproduces matvec per token
        let x2: Vec<f32> = x
            .iter()
            .chain(x.iter().map(|v| v * 2.0).collect::<Vec<_>>().iter())
            .copied()
            .collect();
        let mut y2 = vec![0f32; 2 * rows];
        scalar_matmul(&pool, &w, &x2, 2, &mut y2);
        for r in 0..rows {
            assert!((y2[r] - y[r]).abs() < 1e-5);
            assert!((y2[rows + r] - 2.0 * y[r]).abs() < 1e-4);
        }
    }

    #[test]
    fn by_name_knows_the_portable_sets() {
        assert_eq!(by_name("scalar").unwrap().name, "scalar");
        assert_eq!(by_name("int8").unwrap().name, "int8");
        assert!(by_name("bogus").is_none());
        assert!(available().len() >= 2);
    }
}
