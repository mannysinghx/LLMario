//! Shared plumbing for the quantised-activation kernel sets.
//!
//! A kernel set is a [`DotTable`]: one row-dot function per weight type. The drivers here do the
//! rest, identically for every ISA:
//!
//! - `matvec`: quantise the activation vector once (Q8_K for 256-element block types, Q8_0 for
//!   32-element ones, nothing for float weights), then run the row dots over contiguous row chunks
//!   on the pool. Decode is bandwidth-bound, so each worker streams a contiguous slab of rows.
//! - `matmul`: quantise all `n` activation rows once (in parallel), then for each chunk of weight
//!   rows walk the tokens in blocks of [`TOKEN_BLOCK`]: a token block's quantised activations
//!   (≤ 32 × 2.3 KB for 2048 columns) and one weight row stay in L1, so every weight row is read
//!   from memory once per token block and reused `TOKEN_BLOCK` times from cache.
//!
//! The integer dot of a weight row with a quantised activation row is exact; only the f32
//! epilogue rounds. The epilogue helpers at the bottom fix that rounding order so that the portable
//! `int8` set and the SIMD sets produce bit-identical results.

use super::SendPtr;
use crate::{QMat, ThreadPool};
use llmario_engine_core::dequant::{quantize_row_q8_0, quantize_row_q8_k};
use llmario_engine_core::GgmlType;

/// Bytes per Q8_0 activation block (f16 delta + 32 int8).
pub const Q8_0_BLOCK: usize = 34;
/// Bytes per Q8_K activation block (f32 delta + 256 int8 + 16 i16 group sums).
pub const Q8_K_BLOCK: usize = 292;

/// Dot of one weight row (native block layout, `n` elements) with one quantised activation row
/// (Q8_0 or Q8_K layout, chosen by the weight type).
pub type DotFn = fn(w: &[u8], a: &[u8], n: usize) -> f32;
/// Dot of one float-typed weight row with an f32 activation row.
pub type FDotFn = fn(w: &[u8], x: &[f32]) -> f32;
/// Dots of two weight rows with the same quantised activation row (each result is identical to
/// the corresponding [`DotFn`] call; the pair shares the activation loads).
pub type Dot2Fn = fn(w0: &[u8], w1: &[u8], a: &[u8], n: usize) -> (f32, f32);

/// Which activation representation a row kernel consumes.
#[derive(Clone, Copy)]
pub enum RowKernel {
    Q8K(DotFn),
    Q80(DotFn),
    F32(FDotFn),
}

/// One kernel set's row-dot functions. `None` means "no kernel in this set for that type"; the
/// drivers then fall back to the `int8` set's entry (which covers every type below) and, for types
/// no int8 kernel covers, to the dequantising scalar set.
#[derive(Clone, Copy)]
pub struct DotTable {
    pub q4_0: Option<DotFn>,
    pub q4_1: Option<DotFn>,
    pub q5_0: Option<DotFn>,
    pub q5_1: Option<DotFn>,
    pub q8_0: Option<DotFn>,
    pub q4_k: Option<DotFn>,
    pub q5_k: Option<DotFn>,
    pub q6_k: Option<DotFn>,
    pub f16: Option<FDotFn>,
    pub bf16: Option<FDotFn>,
    pub f32: Option<FDotFn>,
    /// Optional two-row variants (decode-critical types only).
    pub q4_k_x2: Option<Dot2Fn>,
    pub q6_k_x2: Option<Dot2Fn>,
    pub q8_0_x2: Option<Dot2Fn>,
}

impl DotTable {
    pub const EMPTY: DotTable = DotTable {
        q4_0: None,
        q4_1: None,
        q5_0: None,
        q5_1: None,
        q8_0: None,
        q4_k: None,
        q5_k: None,
        q6_k: None,
        f16: None,
        bf16: None,
        f32: None,
        q4_k_x2: None,
        q6_k_x2: None,
        q8_0_x2: None,
    };

    /// The kernel for `t` in this table alone.
    pub fn lookup(&self, t: GgmlType) -> Option<RowKernel> {
        match t {
            GgmlType::Q4_0 => self.q4_0.map(RowKernel::Q80),
            GgmlType::Q4_1 => self.q4_1.map(RowKernel::Q80),
            GgmlType::Q5_0 => self.q5_0.map(RowKernel::Q80),
            GgmlType::Q5_1 => self.q5_1.map(RowKernel::Q80),
            GgmlType::Q8_0 => self.q8_0.map(RowKernel::Q80),
            GgmlType::Q4_K => self.q4_k.map(RowKernel::Q8K),
            GgmlType::Q5_K => self.q5_k.map(RowKernel::Q8K),
            GgmlType::Q6_K => self.q6_k.map(RowKernel::Q8K),
            GgmlType::F16 => self.f16.map(RowKernel::F32),
            GgmlType::BF16 => self.bf16.map(RowKernel::F32),
            GgmlType::F32 => self.f32.map(RowKernel::F32),
            _ => None,
        }
    }

    /// The kernel for `t`, falling back to the portable int8 table.
    pub fn resolve(&self, t: GgmlType) -> Option<RowKernel> {
        self.lookup(t).or_else(|| super::int8::TABLE.lookup(t))
    }

    /// The two-row kernel for `t` in this table, if any. It is only used together with this
    /// table's own single-row kernel (never with a fallback), so the two always agree.
    pub fn lookup2(&self, t: GgmlType) -> Option<Dot2Fn> {
        self.lookup(t)?;
        match t {
            GgmlType::Q4_K => self.q4_k_x2,
            GgmlType::Q6_K => self.q6_k_x2,
            GgmlType::Q8_0 => self.q8_0_x2,
            _ => None,
        }
    }
}

/// Bytes of the quantised activation row for `kind` and `cols` elements (0 for float kernels).
pub fn act_row_bytes(kind: &RowKernel, cols: usize) -> usize {
    match kind {
        RowKernel::Q8K(_) => cols / 256 * Q8_K_BLOCK,
        RowKernel::Q80(_) => cols / 32 * Q8_0_BLOCK,
        RowKernel::F32(_) => 0,
    }
}

/// Quantise one activation row for `kind` into `dst` (no-op for float kernels).
#[inline]
pub fn quantize_act(kind: &RowKernel, x: &[f32], dst: &mut [u8]) {
    match kind {
        RowKernel::Q8K(_) => quantize_row_q8_k(x, dst),
        RowKernel::Q80(_) => quantize_row_q8_0(x, dst),
        RowKernel::F32(_) => {}
    }
}

/// Dot of weight row `w` with token `t` of the activation set.
#[inline(always)]
fn row_dot(
    kind: &RowKernel,
    w: &[u8],
    cols: usize,
    x: &[f32],
    act: &[u8],
    ab: usize,
    t: usize,
) -> f32 {
    match *kind {
        RowKernel::Q8K(f) | RowKernel::Q80(f) => f(w, &act[t * ab..(t + 1) * ab], cols),
        RowKernel::F32(f) => f(w, &x[t * cols..(t + 1) * cols]),
    }
}

/// Number of row chunks for a decode matvec: a few per thread so a descheduled core does not
/// stall the whole op, never fewer than [`MIN_ROWS_PER_CHUNK`] rows each so a worker's weight
/// stream stays long and contiguous.
pub fn matvec_chunks(rows: usize, n_threads: usize) -> usize {
    let per = (rows / (n_threads * CHUNKS_PER_THREAD)).max(MIN_ROWS_PER_CHUNK);
    rows.div_ceil(per)
}
const CHUNKS_PER_THREAD: usize = 4;
const MIN_ROWS_PER_CHUNK: usize = 16;

/// Tokens per block in `matmul`: the block's Q8_K rows at 2048 columns are 75 KB, which stays
/// resident in a 128 KB L1D next to the weight row being streamed.
pub const TOKEN_BLOCK: usize = 32;

/// `y = W x` with the row kernels of `table`.
pub fn matvec(table: &DotTable, pool: &ThreadPool, w: &QMat, x: &[f32], y: &mut [f32]) {
    let Some(kind) = table.resolve(w.dtype) else {
        return super::SCALAR.matvec(pool, w, x, y);
    };
    let cols = w.cols;
    let rows = w.rows;
    let ab = act_row_bytes(&kind, cols);
    let mut act = vec![0u8; ab];
    quantize_act(&kind, x, &mut act);
    let y_ptr = SendPtr(y.as_mut_ptr());
    let dot2 = table.lookup2(w.dtype);
    let chunks = matvec_chunks(rows, pool.n_threads());
    pool.parallel_for(rows, Some(chunks), |s, e| {
        let mut r = s;
        if let Some(f2) = dot2 {
            while r + 1 < e {
                let (v0, v1) = f2(w.row(r), w.row(r + 1), &act, cols);
                // SAFETY: chunks cover disjoint row ranges, so writes never overlap;
                // `r + 1 < e <= rows == y.len()`.
                unsafe {
                    *y_ptr.get().add(r) = v0;
                    *y_ptr.get().add(r + 1) = v1;
                }
                r += 2;
            }
        }
        while r < e {
            let v = row_dot(&kind, w.row(r), cols, x, &act, ab, 0);
            // SAFETY: as above; `r < rows == y.len()`.
            unsafe { *y_ptr.get().add(r) = v };
            r += 1;
        }
    });
}

/// Several independent products in one parallel dispatch, each `y_i = W x_i` for its own list of
/// input / output rows (the routed experts of a mixture-of-experts layer: many small matrices,
/// each applied to the few tokens that chose it, where one pool round trip per product would cost
/// more than the arithmetic). Every input row is quantised once; the weight rows of all products
/// form one index space split into chunks, each weight row is applied to all of its product's
/// input rows while it is in cache, and every dot uses the same row kernel as [`matvec`], so the
/// results are bit-identical to calling it per (product, row).
pub fn rows_multi(table: &DotTable, pool: &ThreadPool, jobs: &mut [crate::RowsJob]) {
    let mut kinds = Vec::with_capacity(jobs.len());
    for j in jobs.iter() {
        match table.resolve(j.w.dtype) {
            Some(k) => kinds.push(k),
            None => {
                // A type without a row kernel: the plain path, row by row.
                for j in jobs.iter_mut() {
                    for (x, y) in j.xs.iter().zip(j.ys.iter_mut()) {
                        matvec(table, pool, &j.w, x, y);
                    }
                }
                return;
            }
        }
    }
    // Quantised inputs, `[product][row]` back to back.
    let mut acts: Vec<Vec<u8>> = Vec::with_capacity(jobs.len());
    let mut starts = Vec::with_capacity(jobs.len() + 1);
    let mut total = 0usize;
    for (j, kind) in jobs.iter().zip(&kinds) {
        let ab = act_row_bytes(kind, j.w.cols);
        let mut a = vec![0u8; ab * j.xs.len()];
        for (i, x) in j.xs.iter().enumerate() {
            quantize_act(kind, x, &mut a[i * ab..(i + 1) * ab]);
        }
        acts.push(a);
        starts.push(total);
        total += j.w.rows;
    }
    starts.push(total);
    let ptrs: Vec<Vec<SendPtr<f32>>> = jobs
        .iter_mut()
        .map(|j| j.ys.iter_mut().map(|y| SendPtr(y.as_mut_ptr())).collect())
        .collect();
    let jobs_ref: &[crate::RowsJob] = jobs;
    let chunks = matvec_chunks(total, pool.n_threads());
    pool.parallel_for(total, Some(chunks), |s, e| {
        // The first product this chunk touches; later ones follow in order.
        let mut ji = starts.partition_point(|&st| st <= s) - 1;
        let mut g = s;
        while g < e {
            while g >= starts[ji + 1] {
                ji += 1;
            }
            let job = &jobs_ref[ji];
            let kind = &kinds[ji];
            let cols = job.w.cols;
            let ab = act_row_bytes(kind, cols);
            let end = e.min(starts[ji + 1]);
            let mut r = g - starts[ji];
            let r_end = end - starts[ji];
            let n = job.xs.len();
            if let Some(f2) = table.lookup2(job.w.dtype) {
                while r + 1 < r_end {
                    for i in 0..n {
                        let (v0, v1) = f2(
                            job.w.row(r),
                            job.w.row(r + 1),
                            &acts[ji][i * ab..(i + 1) * ab],
                            cols,
                        );
                        // SAFETY: global row ranges of the chunks are disjoint, so every
                        // (product, weight row) — hence every output element — is written by
                        // exactly one chunk; `r + 1 < rows == ys[i].len()`.
                        unsafe {
                            *ptrs[ji][i].get().add(r) = v0;
                            *ptrs[ji][i].get().add(r + 1) = v1;
                        }
                    }
                    r += 2;
                }
            }
            while r < r_end {
                for (i, (x, y)) in job.xs.iter().zip(&ptrs[ji]).enumerate() {
                    // `xs[i]` is one row, so the float kernels read it as token 0; the quantised
                    // ones read activation row `i`.
                    let v = match *kind {
                        RowKernel::F32(f) => f(job.w.row(r), x),
                        _ => row_dot(kind, job.w.row(r), cols, x, &acts[ji], ab, i),
                    };
                    // SAFETY: as above.
                    unsafe { *y.get().add(r) = v };
                }
                r += 1;
            }
            g = end;
        }
    });
}

/// `Y = X Wᵀ` for `n` tokens with the row kernels of `table`.
pub fn matmul(table: &DotTable, pool: &ThreadPool, w: &QMat, x: &[f32], n: usize, y: &mut [f32]) {
    if n == 1 {
        return matvec(table, pool, w, x, y);
    }
    let Some(kind) = table.resolve(w.dtype) else {
        return super::SCALAR.matmul(pool, w, x, n, y);
    };
    let cols = w.cols;
    let rows = w.rows;
    let ab = act_row_bytes(&kind, cols);
    let mut act = vec![0u8; ab * n];
    if ab > 0 {
        let act_ptr = SendPtr(act.as_mut_ptr());
        pool.parallel_for(n, Some(n.min(pool.n_threads())), |s, e| {
            for t in s..e {
                // SAFETY: token ranges are disjoint per chunk and `(t + 1) * ab <= act.len()`.
                let dst = unsafe { std::slice::from_raw_parts_mut(act_ptr.get().add(t * ab), ab) };
                quantize_act(&kind, &x[t * cols..(t + 1) * cols], dst);
            }
        });
    }
    let y_ptr = SendPtr(y.as_mut_ptr());
    let dot2 = table.lookup2(w.dtype);
    let chunks = (rows / MIN_ROWS_PER_CHUNK).clamp(1, pool.n_threads() * CHUNKS_PER_THREAD);
    pool.parallel_for(rows, Some(chunks), |s, e| {
        for tb in (0..n).step_by(TOKEN_BLOCK) {
            let te = (tb + TOKEN_BLOCK).min(n);
            let mut r = s;
            if let Some(f2) = dot2 {
                while r + 1 < e {
                    let (w0, w1) = (w.row(r), w.row(r + 1));
                    for t in tb..te {
                        let (v0, v1) = f2(w0, w1, &act[t * ab..(t + 1) * ab], cols);
                        // SAFETY: rows are disjoint per chunk, so `t * rows + r` and `+ r + 1`
                        // are written by exactly one chunk; both are `< n * rows == y.len()`.
                        unsafe {
                            *y_ptr.get().add(t * rows + r) = v0;
                            *y_ptr.get().add(t * rows + r + 1) = v1;
                        }
                    }
                    r += 2;
                }
            }
            while r < e {
                let wr = w.row(r);
                for t in tb..te {
                    let v = row_dot(&kind, wr, cols, x, &act, ab, t);
                    // SAFETY: as above; `t * rows + r < n * rows == y.len()`.
                    unsafe { *y_ptr.get().add(t * rows + r) = v };
                }
                r += 1;
            }
        }
    });
}

// ---- epilogues shared by the int8 and SIMD kernels (same operations, same order) ----

/// Final reduction of the four f32 lane accumulators used by the 32-element block kernels (block
/// `b` accumulates into lane `b % 4`).
#[inline(always)]
pub fn lanes4_sum(acc: [f32; 4]) -> f32 {
    (acc[0] + acc[1]) + (acc[2] + acc[3])
}

/// Q4_K / Q5_K super-block epilogue: `acc + d·d8·Σ(scale·q·a) − dmin·d8·Σ(min·Σa)`.
#[inline(always)]
pub fn kq_epilogue(acc: f32, d: f32, dmin: f32, d8: f32, sumi: i32, summin: i32) -> f32 {
    let acc = acc + (d * d8) * sumi as f32;
    acc - (dmin * d8) * summin as f32
}

/// Q6_K super-block epilogue: `acc + d·d8·Σ(scale·(q−32)·a)`.
#[inline(always)]
pub fn q6k_epilogue(acc: f32, d: f32, d8: f32, sumi: i32) -> f32 {
    acc + (d * d8) * sumi as f32
}

/// Float-weight kernels: 16 elements per step into a 4×4 accumulator grid (`acc[k][l]` takes
/// element `16·j + 4·k + l`), then a fixed-order reduction plus the scalar tail.
#[inline(always)]
pub fn grid4x4_sum(acc: [[f32; 4]; 4], tail: f32) -> f32 {
    let mut v = [0f32; 4];
    for l in 0..4 {
        v[l] = (acc[0][l] + acc[1][l]) + (acc[2][l] + acc[3][l]);
    }
    lanes4_sum(v) + tail
}

/// Little-endian f16 at `off` (exact conversion).
#[inline(always)]
pub fn f16_at(b: &[u8], off: usize) -> f32 {
    half::f16::from_le_bytes([b[off], b[off + 1]]).to_f32()
}

/// Little-endian f32 at `off`.
#[inline(always)]
pub fn f32_at(b: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// Little-endian i16 at `off` widened to i32.
#[inline(always)]
pub fn i16_at(b: &[u8], off: usize) -> i32 {
    i16::from_le_bytes([b[off], b[off + 1]]) as i32
}
