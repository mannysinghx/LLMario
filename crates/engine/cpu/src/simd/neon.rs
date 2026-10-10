//! NEON + dotprod kernels (aarch64).
//!
//! Every kernel here computes the integer sums of the `int8` reference exactly (integer arithmetic
//! is order-independent) and then performs the same f32 operations in the same order, so results
//! are bit-identical to `int8` on the same inputs. In particular the f32 epilogues use separate
//! multiply and add (no fused multiply-add) because the portable reference cannot assume an FMA
//! unit, and the 32-element block types accumulate four blocks per vector step into lane `b % 4`.
//!
//! Two stable-toolchain workarounds: the `sdot` instruction is emitted through `asm!` because the
//! `vdotq_s32` intrinsic is nightly-only on Rust 1.92 (rust-lang/rust#117224), and f16 block
//! deltas are converted with an exact integer-domain trick (`f16x4_to_f32`) because
//! `vcvt_f32_f16` is nightly-only too; the trick is verified against `half` for all 65536 bit
//! patterns in the tests.
//!
//! Safety: the `unsafe` kernels require `dotprod` (checked once in [`available`], which gates the
//! table) and read `n / BLOCK * block_bytes` bytes of `w` and `n / BLOCK * act_bytes` bytes of
//! `a`; the safe wrappers assert those lengths before calling in.
// The kernels index blocks by element position on purpose: the loops mirror the block layout
// arithmetic of the reference dequantizer, which is what a reader checks them against.
#![allow(clippy::needless_range_loop)]

use super::common::*;
use crate::{QMat, ThreadPool};
use std::arch::aarch64::*;

/// True when this CPU has the features the kernels are compiled for.
pub fn available() -> bool {
    std::arch::is_aarch64_feature_detected!("neon")
        && std::arch::is_aarch64_feature_detected!("dotprod")
}

pub static TABLE: DotTable = DotTable {
    q4_0: Some(q4_0),
    q4_1: Some(q4_1),
    q5_0: Some(q5_0),
    q5_1: Some(q5_1),
    q8_0: Some(q8_0),
    q4_k: Some(q4_k),
    q5_k: Some(q5_k),
    q6_k: Some(q6_k),
    f16: Some(f16),
    bf16: Some(bf16),
    f32: Some(f32),
    q4_k_x2: Some(q4_k_x2),
    q6_k_x2: Some(q6_k_x2),
    q8_0_x2: Some(q8_0_x2),
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

/// Safe wrapper: checks the row lengths, then calls the `target_feature` kernel.
macro_rules! wrap_q {
    ($name:ident, $inner:ident, $block:expr, $wbytes:expr, $abytes:expr) => {
        fn $name(w: &[u8], a: &[u8], n: usize) -> f32 {
            let nb = n / $block;
            assert!(
                w.len() >= nb * $wbytes && a.len() >= nb * $abytes,
                "row length"
            );
            // SAFETY: `TABLE` is only reachable when `available()` reported dotprod, and the
            // kernel reads at most `nb * $wbytes` bytes of `w` and `nb * $abytes` of `a`,
            // which the assertion above guarantees are in bounds.
            unsafe { $inner(w, a, n) }
        }
    };
}

/// Safe wrapper for the two-row kernels.
macro_rules! wrap_q2 {
    ($name:ident, $inner:ident, $block:expr, $wbytes:expr, $abytes:expr) => {
        fn $name(w0: &[u8], w1: &[u8], a: &[u8], n: usize) -> (f32, f32) {
            let nb = n / $block;
            assert!(
                w0.len() >= nb * $wbytes && w1.len() >= nb * $wbytes && a.len() >= nb * $abytes,
                "row length"
            );
            // SAFETY: as for `wrap_q!`, for both weight rows.
            unsafe { $inner(w0, w1, a, n) }
        }
    };
}

wrap_q!(q8_0, dot_q8_0, 32, 34, Q8_0_BLOCK);
wrap_q2!(q8_0_x2, dot2_q8_0, 32, 34, Q8_0_BLOCK);
wrap_q2!(q4_k_x2, dot2_q4_k, 256, 144, Q8_K_BLOCK);
wrap_q2!(q6_k_x2, dot2_q6_k, 256, 210, Q8_K_BLOCK);
wrap_q!(q4_0, dot_q4_0, 32, 18, Q8_0_BLOCK);
wrap_q!(q4_1, dot_q4_1, 32, 20, Q8_0_BLOCK);
wrap_q!(q5_0, dot_q5_0, 32, 22, Q8_0_BLOCK);
wrap_q!(q5_1, dot_q5_1, 32, 24, Q8_0_BLOCK);
wrap_q!(q4_k, dot_q4_k, 256, 144, Q8_K_BLOCK);
wrap_q!(q5_k, dot_q5_k, 256, 176, Q8_K_BLOCK);
wrap_q!(q6_k, dot_q6_k, 256, 210, Q8_K_BLOCK);

fn f16(w: &[u8], x: &[f32]) -> f32 {
    assert!(w.len() >= 2 * x.len(), "row length");
    // SAFETY: baseline NEON; reads `2 * x.len()` bytes of `w`, checked above.
    unsafe { dot_f16(w, x) }
}

fn bf16(w: &[u8], x: &[f32]) -> f32 {
    assert!(w.len() >= 2 * x.len(), "row length");
    // SAFETY: baseline NEON; reads `2 * x.len()` bytes of `w`, checked above.
    unsafe { dot_bf16(w, x) }
}

fn f32(w: &[u8], x: &[f32]) -> f32 {
    assert!(w.len() >= 4 * x.len(), "row length");
    // SAFETY: baseline NEON; reads `4 * x.len()` bytes of `w`, checked above.
    unsafe { dot_f32(w, x) }
}

// ---- helpers ----

/// `acc[l] + Σ_{j<4} a[4l+j]·b[4l+j]` per lane: the FEAT_DotProd `sdot` instruction, via inline
/// asm because the `vdotq_s32` intrinsic is nightly-only on this toolchain.
#[inline(always)]
unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let mut r = acc;
    std::arch::asm!(
        "sdot {r:v}.4s, {a:v}.16b, {b:v}.16b",
        r = inout(vreg) r,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack, preserves_flags)
    );
    r
}

/// Little-endian f32 at `p`.
#[inline(always)]
unsafe fn f32_at_ptr(p: *const u8) -> f32 {
    f32::from_bits((p as *const u32).read_unaligned())
}

/// Exact f16 → f32 for four values, in the integer domain: the sign is moved, the exponent and
/// mantissa bits are placed in the f32 fields and rebased by multiplying with 2^112 (exact for
/// normal and subnormal inputs), and ±inf / NaN get their exponent forced to all-ones.
#[inline(always)]
pub(crate) unsafe fn f16x4_to_f32(h: uint16x4_t) -> float32x4_t {
    let hw = vmovl_u16(h);
    let sign = vshlq_n_u32::<16>(vandq_u32(hw, vdupq_n_u32(0x8000)));
    let mag = vandq_u32(hw, vdupq_n_u32(0x7fff));
    let placed = vreinterpretq_f32_u32(vshlq_n_u32::<13>(mag));
    let rebased = vmulq_f32(placed, vdupq_n_f32(f32::from_bits(0x7780_0000))); // 2^112
    let special = vcgeq_u32(mag, vdupq_n_u32(0x7c00));
    let bits = vorrq_u32(
        vreinterpretq_u32_f32(rebased),
        vandq_u32(special, vdupq_n_u32(0x7f80_0000)),
    );
    vreinterpretq_f32_u32(vorrq_u32(bits, sign))
}

/// The f16 at `p` and at three further `stride` steps, as f32.
#[inline(always)]
unsafe fn deltas4(p: *const u8, stride: usize) -> float32x4_t {
    let h = [
        (p as *const u16).read_unaligned(),
        (p.add(stride) as *const u16).read_unaligned(),
        (p.add(2 * stride) as *const u16).read_unaligned(),
        (p.add(3 * stride) as *const u16).read_unaligned(),
    ];
    f16x4_to_f32(vld1_u16(h.as_ptr()))
}

/// Lane sums of four partial-sum vectors: `[Σp0, Σp1, Σp2, Σp3]`.
#[inline(always)]
unsafe fn sum4(p0: int32x4_t, p1: int32x4_t, p2: int32x4_t, p3: int32x4_t) -> int32x4_t {
    vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3))
}

/// `acc[l] += (dw[l] * da[l]) * sumi[l]` (two roundings, like the reference).
#[inline(always)]
unsafe fn lanes_mul_add(
    acc: float32x4_t,
    dw: float32x4_t,
    da: float32x4_t,
    sumi: int32x4_t,
) -> float32x4_t {
    vaddq_f32(acc, vmulq_f32(vmulq_f32(dw, da), vcvtq_f32_s32(sumi)))
}

/// Lane sum of the 32 int8 activation quants at `aq` (partials; reduce with `sum4`).
#[inline(always)]
unsafe fn act_sum32(aq: *const u8) -> int32x4_t {
    let a0 = vld1q_s8(aq as *const i8);
    let a1 = vld1q_s8(aq.add(16) as *const i8);
    vpaddlq_s16(vaddq_s16(vpaddlq_s8(a0), vpaddlq_s8(a1)))
}

/// Bytes `[0x10 where bit j of qh is set]` for elements 0..16 (`lo`) and 16..32 (`hi`) of a
/// Q5_0 / Q5_1 block.
#[inline(always)]
unsafe fn q5_high_bits(qh: u32) -> (uint8x16_t, uint8x16_t) {
    const IDX_LO: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    const IDX_HI: [u8; 16] = [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3];
    const BITS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let qhv = vreinterpretq_u8_u32(vdupq_n_u32(qh));
    let bits = vld1q_u8(BITS.as_ptr());
    let h4 = vdupq_n_u8(0x10);
    let lo = vandq_u8(
        vtstq_u8(vqtbl1q_u8(qhv, vld1q_u8(IDX_LO.as_ptr())), bits),
        h4,
    );
    let hi = vandq_u8(
        vtstq_u8(vqtbl1q_u8(qhv, vld1q_u8(IDX_HI.as_ptr())), bits),
        h4,
    );
    (lo, hi)
}

// ---- 32-element blocks × Q8_0 ----

#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot32(w0: int8x16_t, w1: int8x16_t, aq: *const u8) -> int32x4_t {
    let a0 = vld1q_s8(aq as *const i8);
    let a1 = vld1q_s8(aq.add(16) as *const i8);
    sdot(sdot(vdupq_n_s32(0), w0, a0), w1, a1)
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn block_q8_0(wb: *const u8, ab: *const u8) -> int32x4_t {
    let w0 = vld1q_s8(wb.add(2) as *const i8);
    let w1 = vld1q_s8(wb.add(18) as *const i8);
    dot32(w0, w1, ab.add(2))
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn block_q4_0(wb: *const u8, ab: *const u8) -> int32x4_t {
    let q = vld1q_u8(wb.add(2));
    let eight = vdupq_n_s8(8);
    let lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, vdupq_n_u8(0x0F))), eight);
    let hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(q)), eight);
    dot32(lo, hi, ab.add(2))
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn block_q4_1(wb: *const u8, ab: *const u8) -> int32x4_t {
    let q = vld1q_u8(wb.add(4));
    let lo = vreinterpretq_s8_u8(vandq_u8(q, vdupq_n_u8(0x0F)));
    let hi = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q));
    dot32(lo, hi, ab.add(2))
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn block_q5_0(wb: *const u8, ab: *const u8) -> int32x4_t {
    let qh = (wb.add(2) as *const u32).read_unaligned();
    let (hlo, hhi) = q5_high_bits(qh);
    let q = vld1q_u8(wb.add(6));
    let sixteen = vdupq_n_s8(16);
    let lo = vsubq_s8(
        vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q, vdupq_n_u8(0x0F)), hlo)),
        sixteen,
    );
    let hi = vsubq_s8(
        vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(q), hhi)),
        sixteen,
    );
    dot32(lo, hi, ab.add(2))
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn block_q5_1(wb: *const u8, ab: *const u8) -> int32x4_t {
    let qh = (wb.add(4) as *const u32).read_unaligned();
    let (hlo, hhi) = q5_high_bits(qh);
    let q = vld1q_u8(wb.add(8));
    let lo = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q, vdupq_n_u8(0x0F)), hlo));
    let hi = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(q), hhi));
    dot32(lo, hi, ab.add(2))
}

/// Shared driver for the `d`-only 32-element types: `acc[b%4] += (d_b * da_b) * sumi_b`.
macro_rules! dot_d_only {
    ($name:ident, $block:ident, $wbytes:expr) => {
        #[target_feature(enable = "neon,dotprod")]
        unsafe fn $name(w: &[u8], a: &[u8], n: usize) -> f32 {
            let nb = n / 32;
            let wp = w.as_ptr();
            let ap = a.as_ptr();
            let mut acc = vdupq_n_f32(0.0);
            let mut b = 0;
            while b + 4 <= nb {
                let wb = wp.add(b * $wbytes);
                let ab = ap.add(b * Q8_0_BLOCK);
                let p0 = $block(wb, ab);
                let p1 = $block(wb.add($wbytes), ab.add(Q8_0_BLOCK));
                let p2 = $block(wb.add(2 * $wbytes), ab.add(2 * Q8_0_BLOCK));
                let p3 = $block(wb.add(3 * $wbytes), ab.add(3 * Q8_0_BLOCK));
                let sumi = sum4(p0, p1, p2, p3);
                acc = lanes_mul_add(acc, deltas4(wb, $wbytes), deltas4(ab, Q8_0_BLOCK), sumi);
                b += 4;
            }
            let mut lanes = [0f32; 4];
            vst1q_f32(lanes.as_mut_ptr(), acc);
            while b < nb {
                let wb = &w[b * $wbytes..];
                let ab = &a[b * Q8_0_BLOCK..];
                let sumi = vaddvq_s32($block(wb.as_ptr(), ab.as_ptr()));
                lanes[b % 4] += (f16_at(wb, 0) * f16_at(ab, 0)) * sumi as f32;
                b += 1;
            }
            lanes4_sum(lanes)
        }
    };
}

dot_d_only!(dot_q8_0, block_q8_0, 34);
dot_d_only!(dot_q4_0, block_q4_0, 18);
dot_d_only!(dot_q5_0, block_q5_0, 22);

/// Shared driver for the `d` + `m` 32-element types:
/// `acc[b%4] += (d_b * da_b) * sumi_b; acc[b%4] += (m_b * da_b) * suma_b`.
macro_rules! dot_d_m {
    ($name:ident, $block:ident, $wbytes:expr) => {
        #[target_feature(enable = "neon,dotprod")]
        unsafe fn $name(w: &[u8], a: &[u8], n: usize) -> f32 {
            let nb = n / 32;
            let wp = w.as_ptr();
            let ap = a.as_ptr();
            let mut acc = vdupq_n_f32(0.0);
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
                let suma = sum4(
                    act_sum32(ab.add(2)),
                    act_sum32(ab.add(Q8_0_BLOCK + 2)),
                    act_sum32(ab.add(2 * Q8_0_BLOCK + 2)),
                    act_sum32(ab.add(3 * Q8_0_BLOCK + 2)),
                );
                let da = deltas4(ab, Q8_0_BLOCK);
                acc = lanes_mul_add(acc, deltas4(wb, $wbytes), da, sumi);
                acc = lanes_mul_add(acc, deltas4(wb.add(2), $wbytes), da, suma);
                b += 4;
            }
            let mut lanes = [0f32; 4];
            vst1q_f32(lanes.as_mut_ptr(), acc);
            while b < nb {
                let wb = &w[b * $wbytes..];
                let ab = &a[b * Q8_0_BLOCK..];
                let sumi = vaddvq_s32($block(wb.as_ptr(), ab.as_ptr()));
                let suma = vaddvq_s32(act_sum32(ab.as_ptr().add(2)));
                let da = f16_at(ab, 0);
                lanes[b % 4] += (f16_at(wb, 0) * da) * sumi as f32;
                lanes[b % 4] += (f16_at(wb, 2) * da) * suma as f32;
                b += 1;
            }
            lanes4_sum(lanes)
        }
    };
}

dot_d_m!(dot_q4_1, block_q4_1, 20);
dot_d_m!(dot_q5_1, block_q5_1, 24);

// ---- 256-element super-blocks × Q8_K ----
//
// Each super-block's eight (Q4_K/Q5_K) or sixteen (Q6_K) group dot products are produced as
// 4-lane partials by `sdot`, pair-added down to one lane per group (`[Σg0, Σg1, Σg2, Σg3]`), and
// multiplied by the group scales as whole vectors: two `mul`/`mla` per super-block instead of a
// lane broadcast per group. The two-row kernels load each activation vector once for both rows.

/// The eight 6-bit (scale, min) pairs of a Q4_K / Q5_K super-block as int16 lanes, decoded with
/// word-wide masks (the same bit assignment as `get_scale_min_k4`).
#[inline(always)]
unsafe fn kq_scales_v(sc: *const u8) -> (int16x8_t, int16x8_t) {
    const K1: u32 = 0x3f3f_3f3f;
    const K2: u32 = 0x0f0f_0f0f;
    const K3: u32 = 0x0303_0303;
    let u0 = (sc as *const u32).read_unaligned();
    let u1 = (sc.add(4) as *const u32).read_unaligned();
    let u2 = (sc.add(8) as *const u32).read_unaligned();
    let s_lo = u0 & K1;
    let s_hi = (u2 & K2) | (((u0 >> 6) & K3) << 4);
    let m_lo = u1 & K1;
    let m_hi = ((u2 >> 4) & K2) | (((u1 >> 6) & K3) << 4);
    let scales = vreinterpretq_s16_u16(vmovl_u8(vcreate_u8(s_lo as u64 | ((s_hi as u64) << 32))));
    let mins = vreinterpretq_s16_u16(vmovl_u8(vcreate_u8(m_lo as u64 | ((m_hi as u64) << 32))));
    (scales, mins)
}

/// Per-32-element activation sums `bsum_{2s} + bsum_{2s+1}` (lane `s`) from the Q8_K group sums
/// at `bsums` (16 × i16).
#[inline(always)]
unsafe fn kq_bsum_pairs(bsums: *const u8) -> int16x8_t {
    let b0 = vld1q_s16(bsums as *const i16);
    let b1 = vld1q_s16(bsums.add(16) as *const i16);
    vpaddq_s16(b0, b1)
}

/// Σ_s min_s · pairs_s.
#[inline(always)]
unsafe fn kq_summin_v(mins: int16x8_t, pairs: int16x8_t) -> i32 {
    let lo = vmull_s16(vget_low_s16(mins), vget_low_s16(pairs));
    let hi = vmull_s16(vget_high_s16(mins), vget_high_s16(pairs));
    vaddvq_s32(vaddq_s32(lo, hi))
}

/// `d` and `dmin` (f16 at bytes 0..4 of the super-block) as f32.
#[inline(always)]
unsafe fn kq_deltas(wb: *const u8) -> (f32, f32) {
    let v = f16x4_to_f32(vld1_u16(wb as *const u16));
    (vgetq_lane_f32::<0>(v), vgetq_lane_f32::<1>(v))
}

/// `[Σp0, Σp1, Σp2, Σp3]` from four 4-lane partials.
#[inline(always)]
unsafe fn fold4(p0: int32x4_t, p1: int32x4_t, p2: int32x4_t, p3: int32x4_t) -> int32x4_t {
    vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3))
}

/// Σ_g scale_g · sum_g over eight groups held as two folded vectors and eight int16 scales.
#[inline(always)]
unsafe fn scaled_sum8(g0123: int32x4_t, g4567: int32x4_t, scales: int16x8_t) -> int32x4_t {
    let sc_lo = vmovl_s16(vget_low_s16(scales));
    let sc_hi = vmovl_s16(vget_high_s16(scales));
    vmlaq_s32(vmulq_s32(g0123, sc_lo), g4567, sc_hi)
}

/// The 16 activation vectors of a Q8_K block.
#[inline(always)]
unsafe fn act16(aq: *const u8) -> [int8x16_t; 16] {
    let mut a = [vdupq_n_s8(0); 16];
    for (k, v) in a.iter_mut().enumerate() {
        *v = vld1q_s8(aq.add(16 * k) as *const i8);
    }
    a
}

// -- Q4_K --

/// Folded sub-block sums `([Σp0..Σp3], [Σp4..Σp7])` of one Q4_K super-block's quants `qs`
/// against the activation vectors `a`.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn q4k_sums(qs: *const u8, a: &[int8x16_t; 16], m4: uint8x16_t) -> (int32x4_t, int32x4_t) {
    let zero = vdupq_n_s32(0);
    let mut p = [zero; 8];
    for c in 0..4 {
        let q0 = vld1q_u8(qs.add(c * 32));
        let q1 = vld1q_u8(qs.add(c * 32 + 16));
        let lo0 = vreinterpretq_s8_u8(vandq_u8(q0, m4));
        let lo1 = vreinterpretq_s8_u8(vandq_u8(q1, m4));
        let hi0 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q0));
        let hi1 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1));
        p[2 * c] = sdot(sdot(zero, lo0, a[4 * c]), lo1, a[4 * c + 1]);
        p[2 * c + 1] = sdot(sdot(zero, hi0, a[4 * c + 2]), hi1, a[4 * c + 3]);
    }
    (fold4(p[0], p[1], p[2], p[3]), fold4(p[4], p[5], p[6], p[7]))
}

/// Q4_K / Q5_K super-block epilogue from the folded sums.
#[inline(always)]
unsafe fn kq_finish(
    acc: f32,
    wb: *const u8,
    d8: f32,
    pairs: int16x8_t,
    g0123: int32x4_t,
    g4567: int32x4_t,
) -> f32 {
    let (scales, mins) = kq_scales_v(wb.add(4));
    let sumi = vaddvq_s32(scaled_sum8(g0123, g4567, scales));
    let summin = kq_summin_v(mins, pairs);
    let (d, dmin) = kq_deltas(wb);
    kq_epilogue(acc, d, dmin, d8, sumi, summin)
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_q4_k(w: &[u8], a: &[u8], n: usize) -> f32 {
    let nb = n / 256;
    let m4 = vdupq_n_u8(0x0F);
    let mut acc = 0f32;
    for b in 0..nb {
        let wb = w.as_ptr().add(b * 144);
        let ab = a.as_ptr().add(b * Q8_K_BLOCK);
        let act = act16(ab.add(4));
        let pairs = kq_bsum_pairs(ab.add(260));
        let d8 = f32_at_ptr(ab);
        let (g0123, g4567) = q4k_sums(wb.add(16), &act, m4);
        acc = kq_finish(acc, wb, d8, pairs, g0123, g4567);
    }
    acc
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn dot2_q4_k(w0: &[u8], w1: &[u8], a: &[u8], n: usize) -> (f32, f32) {
    let nb = n / 256;
    let m4 = vdupq_n_u8(0x0F);
    let mut acc0 = 0f32;
    let mut acc1 = 0f32;
    for b in 0..nb {
        let wb0 = w0.as_ptr().add(b * 144);
        let wb1 = w1.as_ptr().add(b * 144);
        let ab = a.as_ptr().add(b * Q8_K_BLOCK);
        let act = act16(ab.add(4));
        let pairs = kq_bsum_pairs(ab.add(260));
        let d8 = f32_at_ptr(ab);
        let (g0, g1) = q4k_sums(wb0.add(16), &act, m4);
        acc0 = kq_finish(acc0, wb0, d8, pairs, g0, g1);
        let (h0, h1) = q4k_sums(wb1.add(16), &act, m4);
        acc1 = kq_finish(acc1, wb1, d8, pairs, h0, h1);
    }
    (acc0, acc1)
}

// -- Q5_K --

#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_q5_k(w: &[u8], a: &[u8], n: usize) -> f32 {
    let nb = n / 256;
    let m4 = vdupq_n_u8(0x0F);
    let h4 = vdupq_n_u8(0x10);
    let zero = vdupq_n_s32(0);
    let mut acc = 0f32;
    for b in 0..nb {
        let wb = w.as_ptr().add(b * 176);
        let ab = a.as_ptr().add(b * Q8_K_BLOCK);
        let act = act16(ab.add(4));
        let pairs = kq_bsum_pairs(ab.add(260));
        let qh0 = vld1q_u8(wb.add(16));
        let qh1 = vld1q_u8(wb.add(32));
        let qs = wb.add(48);
        let mut p = [zero; 8];
        for c in 0..4 {
            let q0 = vld1q_u8(qs.add(c * 32));
            let q1 = vld1q_u8(qs.add(c * 32 + 16));
            // Sub-block s = 2c uses low nibbles and bit s of qh; s = 2c + 1 the high nibbles and
            // bit s + 1. `vshlq_u8` by (4 - s) moves bit s to bit 4 (negative = right shift).
            let sh_lo = vdupq_n_s8(4 - 2 * c as i8);
            let sh_hi = vdupq_n_s8(3 - 2 * c as i8);
            let lo0 = vorrq_u8(vandq_u8(q0, m4), vandq_u8(vshlq_u8(qh0, sh_lo), h4));
            let lo1 = vorrq_u8(vandq_u8(q1, m4), vandq_u8(vshlq_u8(qh1, sh_lo), h4));
            let hi0 = vorrq_u8(vshrq_n_u8::<4>(q0), vandq_u8(vshlq_u8(qh0, sh_hi), h4));
            let hi1 = vorrq_u8(vshrq_n_u8::<4>(q1), vandq_u8(vshlq_u8(qh1, sh_hi), h4));
            p[2 * c] = sdot(
                sdot(zero, vreinterpretq_s8_u8(lo0), act[4 * c]),
                vreinterpretq_s8_u8(lo1),
                act[4 * c + 1],
            );
            p[2 * c + 1] = sdot(
                sdot(zero, vreinterpretq_s8_u8(hi0), act[4 * c + 2]),
                vreinterpretq_s8_u8(hi1),
                act[4 * c + 3],
            );
        }
        let g0123 = fold4(p[0], p[1], p[2], p[3]);
        let g4567 = fold4(p[4], p[5], p[6], p[7]);
        acc = kq_finish(acc, wb, f32_at_ptr(ab), pairs, g0123, g4567);
    }
    acc
}

// -- Q6_K --

/// Scaled group sums of one half (128 elements) of a Q6_K super-block: `wb` is the block, `h`
/// the half, `a` the half's eight activation vectors.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn q6k_half(wb: *const u8, h: usize, a: &[int8x16_t; 8]) -> int32x4_t {
    let m4 = vdupq_n_u8(0x0F);
    let m3 = vdupq_n_u8(0x03);
    let m30 = vdupq_n_u8(0x30);
    let zero = vdupq_n_s32(0);
    let ql = wb.add(h * 64);
    let qh = wb.add(128 + h * 32);
    let ql0 = vld1q_u8(ql);
    let ql1 = vld1q_u8(ql.add(16));
    let ql2 = vld1q_u8(ql.add(32));
    let ql3 = vld1q_u8(ql.add(48));
    let qh0 = vld1q_u8(qh);
    let qh1 = vld1q_u8(qh.add(16));
    // Segment k (elements 32k..32k+32 of this half) = (ql part) | (2 bits of qh) << 4, unsigned
    // 0..63 (the reference's −32 offset is applied through the group sums in `q6k_finish`).
    let q1a = vorrq_u8(vandq_u8(ql0, m4), vshlq_n_u8::<4>(vandq_u8(qh0, m3)));
    let q1b = vorrq_u8(vandq_u8(ql1, m4), vshlq_n_u8::<4>(vandq_u8(qh1, m3)));
    let q2a = vorrq_u8(vandq_u8(ql2, m4), vandq_u8(vshlq_n_u8::<2>(qh0), m30));
    let q2b = vorrq_u8(vandq_u8(ql3, m4), vandq_u8(vshlq_n_u8::<2>(qh1), m30));
    let q3a = vorrq_u8(vshrq_n_u8::<4>(ql0), vandq_u8(qh0, m30));
    let q3b = vorrq_u8(vshrq_n_u8::<4>(ql1), vandq_u8(qh1, m30));
    let q4a = vorrq_u8(vshrq_n_u8::<4>(ql2), vandq_u8(vshrq_n_u8::<2>(qh0), m30));
    let q4b = vorrq_u8(vshrq_n_u8::<4>(ql3), vandq_u8(vshrq_n_u8::<2>(qh1), m30));
    let g = |q: uint8x16_t, k: usize| -> int32x4_t { sdot(zero, vreinterpretq_s8_u8(q), a[k]) };
    let g0123 = fold4(g(q1a, 0), g(q1b, 1), g(q2a, 2), g(q2b, 3));
    let g4567 = fold4(g(q3a, 4), g(q3b, 5), g(q4a, 6), g(q4b, 7));
    let scales = vmovl_s8(vld1_s8(wb.add(192 + h * 8) as *const i8));
    scaled_sum8(g0123, g4567, scales)
}

/// Q6_K super-block epilogue: subtract the folded `32 · Σ_j scale_j · bsum_j`, scale by `d · d8`.
#[inline(always)]
unsafe fn q6k_finish(acc: f32, wb: *const u8, ab: *const u8, isum: int32x4_t) -> f32 {
    let s0 = vmovl_s8(vld1_s8(wb.add(192) as *const i8));
    let s1 = vmovl_s8(vld1_s8(wb.add(200) as *const i8));
    let b0 = vld1q_s16(ab.add(260) as *const i16);
    let b1 = vld1q_s16(ab.add(276) as *const i16);
    let corr = vaddq_s32(
        vaddq_s32(
            vmull_s16(vget_low_s16(s0), vget_low_s16(b0)),
            vmull_s16(vget_high_s16(s0), vget_high_s16(b0)),
        ),
        vaddq_s32(
            vmull_s16(vget_low_s16(s1), vget_low_s16(b1)),
            vmull_s16(vget_high_s16(s1), vget_high_s16(b1)),
        ),
    );
    let sumi = vaddvq_s32(isum) - 32 * vaddvq_s32(corr);
    // `d` is the last two bytes of the block: a scalar read, never a vector load past it.
    let d_bits = (wb.add(208) as *const u16).read_unaligned();
    let d = vgetq_lane_f32::<0>(f16x4_to_f32(vcreate_u16(d_bits as u64)));
    q6k_epilogue(acc, d, f32_at_ptr(ab), sumi)
}

/// The eight activation vectors of one half of a Q8_K block.
#[inline(always)]
unsafe fn act8(aq: *const u8) -> [int8x16_t; 8] {
    let mut a = [vdupq_n_s8(0); 8];
    for (k, v) in a.iter_mut().enumerate() {
        *v = vld1q_s8(aq.add(16 * k) as *const i8);
    }
    a
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_q6_k(w: &[u8], a: &[u8], n: usize) -> f32 {
    let nb = n / 256;
    let mut acc = 0f32;
    for b in 0..nb {
        let wb = w.as_ptr().add(b * 210);
        let ab = a.as_ptr().add(b * Q8_K_BLOCK);
        let lo = act8(ab.add(4));
        let mut isum = q6k_half(wb, 0, &lo);
        let hi = act8(ab.add(4 + 128));
        isum = vaddq_s32(isum, q6k_half(wb, 1, &hi));
        acc = q6k_finish(acc, wb, ab, isum);
    }
    acc
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn dot2_q6_k(w0: &[u8], w1: &[u8], a: &[u8], n: usize) -> (f32, f32) {
    let nb = n / 256;
    let mut acc0 = 0f32;
    let mut acc1 = 0f32;
    for b in 0..nb {
        let wb0 = w0.as_ptr().add(b * 210);
        let wb1 = w1.as_ptr().add(b * 210);
        let ab = a.as_ptr().add(b * Q8_K_BLOCK);
        let lo = act8(ab.add(4));
        let mut i0 = q6k_half(wb0, 0, &lo);
        let mut i1 = q6k_half(wb1, 0, &lo);
        let hi = act8(ab.add(4 + 128));
        i0 = vaddq_s32(i0, q6k_half(wb0, 1, &hi));
        i1 = vaddq_s32(i1, q6k_half(wb1, 1, &hi));
        acc0 = q6k_finish(acc0, wb0, ab, i0);
        acc1 = q6k_finish(acc1, wb1, ab, i1);
    }
    (acc0, acc1)
}

// -- Q8_0 two-row --

#[target_feature(enable = "neon,dotprod")]
unsafe fn dot2_q8_0(w0: &[u8], w1: &[u8], a: &[u8], n: usize) -> (f32, f32) {
    let nb = n / 32;
    let ap = a.as_ptr();
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let zero = vdupq_n_s32(0);
    let mut b = 0;
    while b + 4 <= nb {
        let ab = ap.add(b * Q8_0_BLOCK);
        let mut act = [vdupq_n_s8(0); 8];
        for k in 0..4 {
            act[2 * k] = vld1q_s8(ab.add(k * Q8_0_BLOCK + 2) as *const i8);
            act[2 * k + 1] = vld1q_s8(ab.add(k * Q8_0_BLOCK + 18) as *const i8);
        }
        let da = deltas4(ab, Q8_0_BLOCK);
        for (wr, acc) in [(w0, &mut acc0), (w1, &mut acc1)] {
            let wb = wr.as_ptr().add(b * 34);
            let mut p = [zero; 4];
            for (k, pk) in p.iter_mut().enumerate() {
                let q0 = vld1q_s8(wb.add(k * 34 + 2) as *const i8);
                let q1 = vld1q_s8(wb.add(k * 34 + 18) as *const i8);
                *pk = sdot(sdot(zero, q0, act[2 * k]), q1, act[2 * k + 1]);
            }
            let sumi = sum4(p[0], p[1], p[2], p[3]);
            *acc = lanes_mul_add(*acc, deltas4(wb, 34), da, sumi);
        }
        b += 4;
    }
    let mut lanes0 = [0f32; 4];
    let mut lanes1 = [0f32; 4];
    vst1q_f32(lanes0.as_mut_ptr(), acc0);
    vst1q_f32(lanes1.as_mut_ptr(), acc1);
    while b < nb {
        let ab = &a[b * Q8_0_BLOCK..];
        for (wr, lanes) in [(w0, &mut lanes0), (w1, &mut lanes1)] {
            let wb = &wr[b * 34..];
            let sumi = vaddvq_s32(block_q8_0(wb.as_ptr(), ab.as_ptr()));
            lanes[b % 4] += (f16_at(wb, 0) * f16_at(ab, 0)) * sumi as f32;
        }
        b += 1;
    }
    (lanes4_sum(lanes0), lanes4_sum(lanes1))
}

// ---- float weights × f32 ----

/// 4×4 accumulator grid over 16-element steps (`load(i)` yields elements `i..i+4` as f32), then
/// the reference reduction order and the scalar tail.
#[inline(always)]
unsafe fn dot_float(
    x: &[f32],
    load4: impl Fn(usize) -> float32x4_t,
    load1: impl Fn(usize) -> f32,
) -> f32 {
    let n = x.len();
    let main = n / 16 * 16;
    let mut acc = [vdupq_n_f32(0.0); 4];
    let mut i = 0;
    while i < main {
        for (k, a) in acc.iter_mut().enumerate() {
            let idx = i + 4 * k;
            *a = vaddq_f32(*a, vmulq_f32(load4(idx), vld1q_f32(x.as_ptr().add(idx))));
        }
        i += 16;
    }
    let mut tail = 0f32;
    for idx in main..n {
        tail += load1(idx) * x[idx];
    }
    let v = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
    let mut lanes = [0f32; 4];
    vst1q_f32(lanes.as_mut_ptr(), v);
    lanes4_sum(lanes) + tail
}

unsafe fn dot_f32(w: &[u8], x: &[f32]) -> f32 {
    let p = w.as_ptr();
    dot_float(
        x,
        |i| vld1q_f32(p.add(4 * i) as *const f32),
        |i| f32_at(w, 4 * i),
    )
}

unsafe fn dot_f16(w: &[u8], x: &[f32]) -> f32 {
    let p = w.as_ptr();
    dot_float(
        x,
        |i| f16x4_to_f32(vld1_u16(p.add(2 * i) as *const u16)),
        |i| f16_at(w, 2 * i),
    )
}

unsafe fn dot_bf16(w: &[u8], x: &[f32]) -> f32 {
    let p = w.as_ptr();
    dot_float(
        x,
        |i| vreinterpretq_f32_u32(vshll_n_u16::<16>(vld1_u16(p.add(2 * i) as *const u16))),
        |i| f32::from_bits((u16::from_le_bytes([w[2 * i], w[2 * i + 1]]) as u32) << 16),
    )
}
