//! Portable quantised-activation kernels (no intrinsics).
//!
//! These are the second-level reference: every SIMD kernel computes the same integer sums (exact,
//! so order is irrelevant) and performs the same f32 operations in the same order (see the
//! epilogue helpers in `common`), so the SIMD results equal these bit for bit. They are also the
//! fallback on CPUs without the detected ISA features, so they avoid gratuitous slowness: the
//! inner loops are plain indexed loops the compiler can autovectorise.
//!
//! Conventions, shared with the SIMD sets:
//! - 32-element block types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0) take Q8_0 activations; block `b`'s
//!   contribution is added to f32 lane `b % 4`, lanes are reduced as `(l0 + l1) + (l2 + l3)`.
//! - 256-element block types (Q4_K, Q5_K, Q6_K) take Q8_K activations and a scalar f32 epilogue
//!   per super-block.
//! - Float weights (F16, BF16, F32) use the 4×4 accumulator grid of [`grid4x4_sum`].
// The kernels index blocks by element position on purpose: the loops mirror the block layout
// arithmetic of the reference dequantizer, which is what a reader checks them against.
#![allow(clippy::needless_range_loop)]

use super::common::*;
use crate::{QMat, ThreadPool};
use llmario_engine_core::dequant::get_scale_min_k4;

pub static TABLE: DotTable = DotTable {
    q4_0: Some(dot_q4_0),
    q4_1: Some(dot_q4_1),
    q5_0: Some(dot_q5_0),
    q5_1: Some(dot_q5_1),
    q8_0: Some(dot_q8_0),
    q4_k: Some(dot_q4_k),
    q5_k: Some(dot_q5_k),
    q6_k: Some(dot_q6_k),
    f16: Some(dot_f16),
    bf16: Some(dot_bf16),
    f32: Some(dot_f32),
    q4_k_x2: None,
    q6_k_x2: None,
    q8_0_x2: None,
};

pub fn matvec(pool: &ThreadPool, w: &QMat, x: &[f32], y: &mut [f32]) {
    super::common::matvec(&TABLE, pool, w, x, y)
}

pub fn matmul(pool: &ThreadPool, w: &QMat, x: &[f32], n: usize, y: &mut [f32]) {
    super::common::matmul(&TABLE, pool, w, x, n, y)
}

// ---- 32-element blocks × Q8_0 ----

/// Signed int8 dot of 32 weight quants with 32 activation quants.
#[inline(always)]
fn dot32(wq: &[u8], aq: &[u8]) -> i32 {
    let mut s = 0i32;
    for j in 0..32 {
        s += (wq[j] as i8 as i32) * (aq[j] as i8 as i32);
    }
    s
}

/// Sum of 32 activation quants (for the `min` term of Q4_1 / Q5_1).
#[inline(always)]
fn sum32(aq: &[u8]) -> i32 {
    let mut s = 0i32;
    for j in 0..32 {
        s += aq[j] as i8 as i32;
    }
    s
}

pub fn dot_q8_0(w: &[u8], a: &[u8], n: usize) -> f32 {
    let nb = n / 32;
    let mut acc = [0f32; 4];
    for b in 0..nb {
        let wb = &w[b * 34..b * 34 + 34];
        let ab = &a[b * Q8_0_BLOCK..(b + 1) * Q8_0_BLOCK];
        let sumi = dot32(&wb[2..34], &ab[2..34]);
        acc[b % 4] += (f16_at(wb, 0) * f16_at(ab, 0)) * sumi as f32;
    }
    lanes4_sum(acc)
}

pub fn dot_q4_0(w: &[u8], a: &[u8], n: usize) -> f32 {
    let nb = n / 32;
    let mut acc = [0f32; 4];
    for b in 0..nb {
        let wb = &w[b * 18..b * 18 + 18];
        let ab = &a[b * Q8_0_BLOCK..(b + 1) * Q8_0_BLOCK];
        let qs = &wb[2..18];
        let aq = &ab[2..34];
        let mut sumi = 0i32;
        for j in 0..16 {
            sumi += ((qs[j] & 0xF) as i32 - 8) * (aq[j] as i8 as i32);
            sumi += ((qs[j] >> 4) as i32 - 8) * (aq[j + 16] as i8 as i32);
        }
        acc[b % 4] += (f16_at(wb, 0) * f16_at(ab, 0)) * sumi as f32;
    }
    lanes4_sum(acc)
}

pub fn dot_q4_1(w: &[u8], a: &[u8], n: usize) -> f32 {
    let nb = n / 32;
    let mut acc = [0f32; 4];
    for b in 0..nb {
        let wb = &w[b * 20..b * 20 + 20];
        let ab = &a[b * Q8_0_BLOCK..(b + 1) * Q8_0_BLOCK];
        let qs = &wb[4..20];
        let aq = &ab[2..34];
        let mut sumi = 0i32;
        for j in 0..16 {
            sumi += (qs[j] & 0xF) as i32 * (aq[j] as i8 as i32);
            sumi += (qs[j] >> 4) as i32 * (aq[j + 16] as i8 as i32);
        }
        let da = f16_at(ab, 0);
        acc[b % 4] += (f16_at(wb, 0) * da) * sumi as f32;
        acc[b % 4] += (f16_at(wb, 2) * da) * sum32(aq) as f32;
    }
    lanes4_sum(acc)
}

pub fn dot_q5_0(w: &[u8], a: &[u8], n: usize) -> f32 {
    let nb = n / 32;
    let mut acc = [0f32; 4];
    for b in 0..nb {
        let wb = &w[b * 22..b * 22 + 22];
        let ab = &a[b * Q8_0_BLOCK..(b + 1) * Q8_0_BLOCK];
        let qh = u32::from_le_bytes([wb[2], wb[3], wb[4], wb[5]]);
        let qs = &wb[6..22];
        let aq = &ab[2..34];
        let mut sumi = 0i32;
        for j in 0..16 {
            let xh0 = ((qh >> j) << 4) & 0x10;
            let xh1 = (qh >> (j + 12)) & 0x10;
            let q0 = ((qs[j] & 0xF) as u32 | xh0) as i32 - 16;
            let q1 = ((qs[j] >> 4) as u32 | xh1) as i32 - 16;
            sumi += q0 * (aq[j] as i8 as i32);
            sumi += q1 * (aq[j + 16] as i8 as i32);
        }
        acc[b % 4] += (f16_at(wb, 0) * f16_at(ab, 0)) * sumi as f32;
    }
    lanes4_sum(acc)
}

pub fn dot_q5_1(w: &[u8], a: &[u8], n: usize) -> f32 {
    let nb = n / 32;
    let mut acc = [0f32; 4];
    for b in 0..nb {
        let wb = &w[b * 24..b * 24 + 24];
        let ab = &a[b * Q8_0_BLOCK..(b + 1) * Q8_0_BLOCK];
        let qh = u32::from_le_bytes([wb[4], wb[5], wb[6], wb[7]]);
        let qs = &wb[8..24];
        let aq = &ab[2..34];
        let mut sumi = 0i32;
        for j in 0..16 {
            let xh0 = ((qh >> j) << 4) & 0x10;
            let xh1 = (qh >> (j + 12)) & 0x10;
            let q0 = ((qs[j] & 0xF) as u32 | xh0) as i32;
            let q1 = ((qs[j] >> 4) as u32 | xh1) as i32;
            sumi += q0 * (aq[j] as i8 as i32);
            sumi += q1 * (aq[j + 16] as i8 as i32);
        }
        let da = f16_at(ab, 0);
        acc[b % 4] += (f16_at(wb, 0) * da) * sumi as f32;
        acc[b % 4] += (f16_at(wb, 2) * da) * sum32(aq) as f32;
    }
    lanes4_sum(acc)
}

// ---- 256-element super-blocks × Q8_K ----

/// Σ_s min_s · (bsum_{2s} + bsum_{2s+1}) for the eight 32-element sub-blocks of a Q4_K / Q5_K
/// super-block (`bsums` are the Q8_K per-16 group sums at byte 260 of the block).
#[inline(always)]
pub fn kq_summin(mins: &[i32; 8], ab: &[u8]) -> i32 {
    let mut summin = 0i32;
    for s in 0..8 {
        let bsum = i16_at(ab, 260 + 4 * s) + i16_at(ab, 262 + 4 * s);
        summin += mins[s] * bsum;
    }
    summin
}

/// Unpack the eight (scale, min) pairs of a Q4_K / Q5_K super-block.
#[inline(always)]
pub fn kq_scales(sc: &[u8]) -> ([i32; 8], [i32; 8]) {
    let mut scales = [0i32; 8];
    let mut mins = [0i32; 8];
    for s in 0..8 {
        let (scale, min) = get_scale_min_k4(s, sc);
        scales[s] = scale as i32;
        mins[s] = min as i32;
    }
    (scales, mins)
}

pub fn dot_q4_k(w: &[u8], a: &[u8], n: usize) -> f32 {
    let nb = n / 256;
    let mut acc = 0f32;
    for b in 0..nb {
        let wb = &w[b * 144..b * 144 + 144];
        let ab = &a[b * Q8_K_BLOCK..(b + 1) * Q8_K_BLOCK];
        let (scales, mins) = kq_scales(&wb[4..16]);
        let qs = &wb[16..144];
        let aq = &ab[4..260];
        let mut sumi = 0i32;
        for s in 0..8 {
            let q = &qs[(s / 2) * 32..(s / 2) * 32 + 32];
            let x = &aq[s * 32..s * 32 + 32];
            let shift = 4 * (s % 2);
            let mut dot = 0i32;
            for l in 0..32 {
                dot += ((q[l] >> shift) & 0xF) as i32 * (x[l] as i8 as i32);
            }
            sumi += scales[s] * dot;
        }
        let summin = kq_summin(&mins, ab);
        acc = kq_epilogue(
            acc,
            f16_at(wb, 0),
            f16_at(wb, 2),
            f32_at(ab, 0),
            sumi,
            summin,
        );
    }
    acc
}

pub fn dot_q5_k(w: &[u8], a: &[u8], n: usize) -> f32 {
    let nb = n / 256;
    let mut acc = 0f32;
    for b in 0..nb {
        let wb = &w[b * 176..b * 176 + 176];
        let ab = &a[b * Q8_K_BLOCK..(b + 1) * Q8_K_BLOCK];
        let (scales, mins) = kq_scales(&wb[4..16]);
        let qh = &wb[16..48];
        let qs = &wb[48..176];
        let aq = &ab[4..260];
        let mut sumi = 0i32;
        for s in 0..8 {
            let q = &qs[(s / 2) * 32..(s / 2) * 32 + 32];
            let x = &aq[s * 32..s * 32 + 32];
            let shift = 4 * (s % 2);
            let mut dot = 0i32;
            for l in 0..32 {
                let lo = (q[l] >> shift) & 0xF;
                let hi = ((qh[l] >> s) & 1) << 4;
                dot += (lo | hi) as i32 * (x[l] as i8 as i32);
            }
            sumi += scales[s] * dot;
        }
        let summin = kq_summin(&mins, ab);
        acc = kq_epilogue(
            acc,
            f16_at(wb, 0),
            f16_at(wb, 2),
            f32_at(ab, 0),
            sumi,
            summin,
        );
    }
    acc
}

pub fn dot_q6_k(w: &[u8], a: &[u8], n: usize) -> f32 {
    let nb = n / 256;
    let mut acc = 0f32;
    for b in 0..nb {
        let wb = &w[b * 210..b * 210 + 210];
        let ab = &a[b * Q8_K_BLOCK..(b + 1) * Q8_K_BLOCK];
        let ql = &wb[0..128];
        let qh = &wb[128..192];
        let sc = &wb[192..208];
        let aq = &ab[4..260];
        let mut sumi = 0i32;
        for h in 0..2 {
            let ql = &ql[h * 64..h * 64 + 64];
            let qh = &qh[h * 32..h * 32 + 32];
            let x = &aq[h * 128..h * 128 + 128];
            let s = &sc[h * 8..h * 8 + 8];
            // Four 32-element segments; segment k uses scales s[2k] (l < 16) and s[2k+1].
            let mut seg = [0i32; 8];
            for l in 0..32 {
                let g = l / 16;
                let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                seg[g] += q1 * (x[l] as i8 as i32);
                seg[2 + g] += q2 * (x[l + 32] as i8 as i32);
                seg[4 + g] += q3 * (x[l + 64] as i8 as i32);
                seg[6 + g] += q4 * (x[l + 96] as i8 as i32);
            }
            for g in 0..8 {
                sumi += (s[g] as i8 as i32) * seg[g];
            }
        }
        acc = q6k_epilogue(acc, f16_at(wb, 208), f32_at(ab, 0), sumi);
    }
    acc
}

// ---- float weights × f32 ----

#[inline(always)]
fn dot_float(x: &[f32], load: impl Fn(usize) -> f32) -> f32 {
    let n = x.len();
    let main = n / 16 * 16;
    let mut acc = [[0f32; 4]; 4];
    let mut i = 0;
    while i < main {
        for k in 0..4 {
            for l in 0..4 {
                let idx = i + 4 * k + l;
                acc[k][l] += load(idx) * x[idx];
            }
        }
        i += 16;
    }
    let mut tail = 0f32;
    for idx in main..n {
        tail += load(idx) * x[idx];
    }
    grid4x4_sum(acc, tail)
}

pub fn dot_f32(w: &[u8], x: &[f32]) -> f32 {
    dot_float(x, |i| f32_at(w, 4 * i))
}

pub fn dot_f16(w: &[u8], x: &[f32]) -> f32 {
    dot_float(x, |i| f16_at(w, 2 * i))
}

pub fn dot_bf16(w: &[u8], x: &[f32]) -> f32 {
    dot_float(x, |i| {
        f32::from_bits((u16::from_le_bytes([w[2 * i], w[2 * i + 1]]) as u32) << 16)
    })
}
