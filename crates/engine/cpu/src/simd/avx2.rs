//! AVX2 + F16C kernels (x86_64): Q4_0, Q8_0, Q4_K, Q6_K. Other types fall back to `int8`.
//!
//! Recipe: `_mm256_maddubs_epi16` (unsigned weight quant × signed activation quant, pairs summed
//! to i16) followed by `_mm256_madd_epi16` against ones or against the per-group scales. Weight
//! quants are kept unsigned (nibbles, 6-bit values) so `maddubs` applies directly; the signed
//! offsets of Q4_0 (−8) and Q6_K (−32) are folded in exactly through the activation sums. For
//! Q8_0 the weight's sign is moved onto the activation with `_mm256_sign_epi8`. No pair sum can
//! saturate: the largest is 2 × 127 × 128 for Q8_0.
//!
//! Like the NEON set, the f32 epilogues reproduce the `int8` reference operation for operation
//! (no FMA; four 32-element blocks per step into lane `b % 4`), so results are intended to be
//! bit-identical. This file is compile-checked only on the aarch64 development machine; it has
//! not been executed there (see the crate README).
//!
//! Safety: the `unsafe` kernels require AVX2 and F16C (checked once in [`available`], which
//! gates the table) and read `n / BLOCK * block_bytes` bytes of `w` and `n / BLOCK * act_bytes`
//! bytes of `a`; the safe wrappers assert those lengths before calling in.
// The kernels index blocks by element position on purpose: the loops mirror the block layout
// arithmetic of the reference dequantizer, which is what a reader checks them against.
#![allow(clippy::needless_range_loop)]

use super::common::*;
use super::int8::{kq_scales, kq_summin};
use crate::{QMat, ThreadPool};
use std::arch::x86_64::*;

/// True when this CPU has the features the kernels are compiled for.
pub fn available() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c")
}

pub static TABLE: DotTable = DotTable {
    q4_0: Some(q4_0),
    q8_0: Some(q8_0),
    q4_k: Some(q4_k),
    q6_k: Some(q6_k),
    ..DotTable::EMPTY
};

pub fn matvec(pool: &ThreadPool, w: &QMat, x: &[f32], y: &mut [f32]) {
    super::common::matvec(&TABLE, pool, w, x, y)
}

pub fn matmul(pool: &ThreadPool, w: &QMat, x: &[f32], n: usize, y: &mut [f32]) {
    super::common::matmul(&TABLE, pool, w, x, n, y)
}

pub fn rows_multi(pool: &ThreadPool, jobs: &mut [crate::RowsJob]) {
    super::common::rows_multi(&TABLE, pool, jobs)
}

macro_rules! wrap_q {
    ($name:ident, $inner:ident, $block:expr, $wbytes:expr, $abytes:expr) => {
        fn $name(w: &[u8], a: &[u8], n: usize) -> f32 {
            let nb = n / $block;
            assert!(
                w.len() >= nb * $wbytes && a.len() >= nb * $abytes,
                "row length"
            );
            // SAFETY: `TABLE` is only reachable when `available()` reported AVX2 + F16C, and
            // the kernel reads at most `nb * $wbytes` bytes of `w` and `nb * $abytes` of `a`,
            // which the assertion above guarantees are in bounds.
            unsafe { $inner(w, a, n) }
        }
    };
}

wrap_q!(q8_0, dot_q8_0, 32, 34, Q8_0_BLOCK);
wrap_q!(q4_0, dot_q4_0, 32, 18, Q8_0_BLOCK);
wrap_q!(q4_k, dot_q4_k, 256, 144, Q8_K_BLOCK);
wrap_q!(q6_k, dot_q6_k, 256, 210, Q8_K_BLOCK);

// ---- helpers ----

/// Sum of the eight i32 lanes.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum_i32(v: __m256i) -> i32 {
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s));
    _mm_cvtsi128_si32(s)
}

/// `[Σp0, Σp1, Σp2, Σp3]` from four 8-lane partial sums.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn sum4(p0: __m256i, p1: __m256i, p2: __m256i, p3: __m256i) -> __m128i {
    let h = _mm256_hadd_epi32(_mm256_hadd_epi32(p0, p1), _mm256_hadd_epi32(p2, p3));
    _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256::<1>(h))
}

/// The f16 at `p` and at three further `stride` steps, as f32 (exact, F16C).
#[inline]
#[target_feature(enable = "avx2,f16c")]
unsafe fn deltas4(p: *const u8, stride: usize) -> __m128 {
    let rd = |o: usize| (p.add(o) as *const i16).read_unaligned();
    _mm_cvtph_ps(_mm_setr_epi16(
        rd(0),
        rd(stride),
        rd(2 * stride),
        rd(3 * stride),
        0,
        0,
        0,
        0,
    ))
}

/// `acc[l] += (dw[l] * da[l]) * sumi[l]` (two roundings, like the reference).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn lanes_mul_add(acc: __m128, dw: __m128, da: __m128, sumi: __m128i) -> __m128 {
    _mm_add_ps(acc, _mm_mul_ps(_mm_mul_ps(dw, da), _mm_cvtepi32_ps(sumi)))
}

// ---- 32-element blocks × Q8_0 ----

/// Σ w·a over the block as 8 i32 partials (Q8_0 weights).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn block_q8_0(wb: *const u8, ab: *const u8) -> __m256i {
    let w = _mm256_loadu_si256(wb.add(2) as *const __m256i);
    let a = _mm256_loadu_si256(ab.add(2) as *const __m256i);
    let aw = _mm256_sign_epi8(w, w);
    let sa = _mm256_sign_epi8(a, w);
    _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sa), _mm256_set1_epi16(1))
}

/// Σ (q − 8)·a over the block as 8 i32 partials (Q4_0 weights): `Σ q·a − 8·Σ a`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn block_q4_0(wb: *const u8, ab: *const u8) -> __m256i {
    let q = _mm_loadu_si128(wb.add(2) as *const __m128i);
    let m4 = _mm_set1_epi8(0x0F);
    let lo = _mm_and_si128(q, m4);
    let hi = _mm_and_si128(_mm_srli_epi16::<4>(q), m4);
    let qq = _mm256_set_m128i(hi, lo);
    let a = _mm256_loadu_si256(ab.add(2) as *const __m256i);
    let ones = _mm256_set1_epi16(1);
    let p = _mm256_madd_epi16(_mm256_maddubs_epi16(qq, a), ones);
    let sa = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi8(1), a), ones);
    _mm256_sub_epi32(p, _mm256_slli_epi32::<3>(sa))
}

macro_rules! dot_d_only {
    ($name:ident, $block:ident, $wbytes:expr) => {
        #[target_feature(enable = "avx2,f16c")]
        unsafe fn $name(w: &[u8], a: &[u8], n: usize) -> f32 {
            let nb = n / 32;
            let wp = w.as_ptr();
            let ap = a.as_ptr();
            let mut acc = _mm_setzero_ps();
            let mut b = 0;
            while b + 4 <= nb {
                let wb = wp.add(b * $wbytes);
                let ab = ap.add(b * Q8_0_BLOCK);
                let sumi = sum4(
                    $block(wb, ab),
                    $block(wb.add($wbytes), ab.add(Q8_0_BLOCK)),
                    $block(wb.add(2 * $wbytes), ab.add(2 * Q8_0_BLOCK)),
                    $block(wb.add(3 * $wbytes), ab.add(3 * Q8_0_BLOCK)),
                );
                acc = lanes_mul_add(acc, deltas4(wb, $wbytes), deltas4(ab, Q8_0_BLOCK), sumi);
                b += 4;
            }
            let mut lanes = [0f32; 4];
            _mm_storeu_ps(lanes.as_mut_ptr(), acc);
            while b < nb {
                let wb = &w[b * $wbytes..];
                let ab = &a[b * Q8_0_BLOCK..];
                let sumi = hsum_i32($block(wb.as_ptr(), ab.as_ptr()));
                lanes[b % 4] += (f16_at(wb, 0) * f16_at(ab, 0)) * sumi as f32;
                b += 1;
            }
            lanes4_sum(lanes)
        }
    };
}

dot_d_only!(dot_q8_0, block_q8_0, 34);
dot_d_only!(dot_q4_0, block_q4_0, 18);

// ---- 256-element super-blocks × Q8_K ----

#[target_feature(enable = "avx2,f16c")]
unsafe fn dot_q4_k(w: &[u8], a: &[u8], n: usize) -> f32 {
    let nb = n / 256;
    let m4 = _mm256_set1_epi8(0x0F);
    let mut acc = 0f32;
    for b in 0..nb {
        let wb = &w[b * 144..b * 144 + 144];
        let ab = &a[b * Q8_K_BLOCK..(b + 1) * Q8_K_BLOCK];
        let (scales, mins) = kq_scales(&wb[4..16]);
        let qs = wb.as_ptr().add(16);
        let aq = ab.as_ptr().add(4);
        let mut isum = _mm256_setzero_si256();
        for c in 0..4 {
            let q = _mm256_loadu_si256(qs.add(c * 32) as *const __m256i);
            let lo = _mm256_and_si256(q, m4);
            let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(q), m4);
            let a_lo = _mm256_loadu_si256(aq.add(2 * c * 32) as *const __m256i);
            let a_hi = _mm256_loadu_si256(aq.add((2 * c + 1) * 32) as *const __m256i);
            isum = _mm256_add_epi32(
                isum,
                _mm256_madd_epi16(
                    _mm256_maddubs_epi16(lo, a_lo),
                    _mm256_set1_epi16(scales[2 * c] as i16),
                ),
            );
            isum = _mm256_add_epi32(
                isum,
                _mm256_madd_epi16(
                    _mm256_maddubs_epi16(hi, a_hi),
                    _mm256_set1_epi16(scales[2 * c + 1] as i16),
                ),
            );
        }
        let sumi = hsum_i32(isum);
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

#[target_feature(enable = "avx2,f16c")]
unsafe fn dot_q6_k(w: &[u8], a: &[u8], n: usize) -> f32 {
    let nb = n / 256;
    let m4 = _mm256_set1_epi8(0x0F);
    let m3 = _mm256_set1_epi8(0x03);
    let m30 = _mm256_set1_epi8(0x30);
    let mut acc = 0f32;
    for b in 0..nb {
        let wb = &w[b * 210..b * 210 + 210];
        let ab = &a[b * Q8_K_BLOCK..(b + 1) * Q8_K_BLOCK];
        let aq = ab.as_ptr().add(4);
        let sc = &wb[192..208];
        let mut isum = _mm256_setzero_si256();
        for h in 0..2 {
            let ql = wb.as_ptr().add(h * 64);
            let ql0 = _mm256_loadu_si256(ql as *const __m256i);
            let ql1 = _mm256_loadu_si256(ql.add(32) as *const __m256i);
            let qh = _mm256_loadu_si256(wb.as_ptr().add(128 + h * 32) as *const __m256i);
            // Segment k = (ql part) | (2 bits of qh) << 4, unsigned 0..63; the −32 offset is
            // applied below through the Q8_K group sums.
            let q1 = _mm256_or_si256(
                _mm256_and_si256(ql0, m4),
                _mm256_slli_epi16::<4>(_mm256_and_si256(qh, m3)),
            );
            let q2 = _mm256_or_si256(
                _mm256_and_si256(ql1, m4),
                _mm256_and_si256(_mm256_slli_epi16::<2>(qh), m30),
            );
            let q3 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16::<4>(ql0), m4),
                _mm256_and_si256(qh, m30),
            );
            let q4 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16::<4>(ql1), m4),
                _mm256_and_si256(_mm256_srli_epi16::<2>(qh), m30),
            );
            let s = &sc[h * 8..h * 8 + 8];
            for (k, q) in [q1, q2, q3, q4].into_iter().enumerate() {
                let x = _mm256_loadu_si256(aq.add(h * 128 + k * 32) as *const __m256i);
                let scale = _mm256_set_m128i(
                    _mm_set1_epi16(s[2 * k + 1] as i8 as i16),
                    _mm_set1_epi16(s[2 * k] as i8 as i16),
                );
                isum = _mm256_add_epi32(isum, _mm256_madd_epi16(_mm256_maddubs_epi16(q, x), scale));
            }
        }
        // Σ scale_j · (q − 32) · a = Σ scale_j · q · a − 32 · Σ_j scale_j · bsum_j
        let s16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(sc.as_ptr() as *const __m128i));
        let bs = _mm256_loadu_si256(ab.as_ptr().add(260) as *const __m256i);
        let corr = hsum_i32(_mm256_madd_epi16(s16, bs));
        let sumi = hsum_i32(isum) - 32 * corr;
        acc = q6k_epilogue(acc, f16_at(wb, 208), f32_at(ab, 0), sumi);
    }
    acc
}
