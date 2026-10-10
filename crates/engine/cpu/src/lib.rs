//! CPU backend of the native engine.
//!
//! Public API (stable across the SIMD work): the model crate builds its forward pass on these
//! functions and never touches block layouts itself.
//!
//! - [`QMat`]: a quantized weight matrix view (`rows × cols`, ggml row-major: `cols` is the
//!   contiguous dimension, so `y = W x` has `x.len() == cols`, `y.len() == rows`).
//! - [`matvec`] / [`matmul`]: decode (one token) and prefill (many tokens) products.
//! - [`rms_norm`], [`rope`], [`softmax`], [`swiglu`], [`gelu`]: the element-wise primitives.
//! - [`ThreadPool`]: the parked worker pool sized to performance cores (Architecture §4.2).
//!
//! The `ops` module holds the scalar reference implementations; `simd` holds the ISA-specific
//! kernels, each tested against the reference. Dispatch happens once per process in
//! [`Kernels::detect`].

pub mod ops;
pub mod pool;
pub mod simd;

pub use ops::{gelu, rms_norm, rope, softmax, swiglu, RopeKind};
pub use pool::ThreadPool;

use llmario_engine_core::GgmlType;

/// A quantized (or float) weight matrix living in mapped or arena memory.
#[derive(Clone, Copy, Debug)]
pub struct QMat<'a> {
    pub dtype: GgmlType,
    /// Output dimension (number of rows).
    pub rows: usize,
    /// Input dimension (row length in elements; a multiple of the block size).
    pub cols: usize,
    pub data: &'a [u8],
}

impl<'a> QMat<'a> {
    pub fn new(dtype: GgmlType, rows: usize, cols: usize, data: &'a [u8]) -> QMat<'a> {
        debug_assert_eq!(data.len(), dtype.row_bytes(cols) * rows, "QMat byte length");
        QMat {
            dtype,
            rows,
            cols,
            data,
        }
    }
    #[inline]
    pub fn row_bytes(&self) -> usize {
        self.dtype.row_bytes(self.cols)
    }
    #[inline]
    pub fn row(&self, r: usize) -> &'a [u8] {
        let rb = self.row_bytes();
        &self.data[r * rb..(r + 1) * rb]
    }
}

/// `y = W x` for one token. `x.len() == cols`, `y.len() == rows`.
pub fn matvec(pool: &ThreadPool, w: &QMat, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), w.cols);
    assert_eq!(y.len(), w.rows);
    simd::kernels().matvec(pool, w, x, y);
}

/// One product of [`rows_multi`]: `ys[i] = w xs[i]` for each of its rows.
pub struct RowsJob<'a, 'b> {
    pub w: QMat<'a>,
    pub xs: Vec<&'b [f32]>,
    pub ys: Vec<&'b mut [f32]>,
}

/// Several independent products, each over its own few rows, in one parallel dispatch; results
/// equal calling [`matvec`] on every (product, row).
pub fn rows_multi(pool: &ThreadPool, jobs: &mut [RowsJob]) {
    for j in jobs.iter() {
        assert_eq!(j.xs.len(), j.ys.len());
        for (x, y) in j.xs.iter().zip(&j.ys) {
            assert_eq!(x.len(), j.w.cols);
            assert_eq!(y.len(), j.w.rows);
        }
    }
    simd::kernels().rows_multi(pool, jobs);
}

/// `Y = X Wᵀ` for `n` tokens: `x` is `n × cols` (row-major), `y` is `n × rows`.
pub fn matmul(pool: &ThreadPool, w: &QMat, x: &[f32], n: usize, y: &mut [f32]) {
    assert_eq!(x.len(), n * w.cols);
    assert_eq!(y.len(), n * w.rows);
    simd::kernels().matmul(pool, w, x, n, y);
}

/// Dequantise one row of `w` into `out` (embedding lookup).
pub fn dequant_row(w: &QMat, r: usize, out: &mut [f32]) {
    llmario_engine_core::dequant::dequantize_row(w.dtype, w.row(r), out)
        .expect("row dequant: type supported by the loader");
}
