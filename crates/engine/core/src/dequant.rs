//! Scalar reference (de)quantizers.
//!
//! These are the ground truth for every optimised kernel: slow, obvious, and transcribed from
//! ggml's `dequantize_row_*` / gguf-py's `dequantize_blocks` (both MIT). They are also what the
//! engine uses for tensors no fast kernel covers yet, so a model never fails to load because of a
//! missing kernel; it only runs slower (and says so in the plan).
//!
//! Layouts: see `ggml-common.h`. All scales are little-endian f16 unless noted.

use crate::dtype::{GgmlType, K_SCALE_SIZE, QK_K};
use crate::{EngineError, Result};
use half::f16;

#[inline]
fn f16_at(b: &[u8], off: usize) -> f32 {
    f16::from_le_bytes([b[off], b[off + 1]]).to_f32()
}

#[inline]
fn bf16_to_f32(lo: u8, hi: u8) -> f32 {
    f32::from_bits((u16::from_le_bytes([lo, hi]) as u32) << 16)
}

/// IQ4 non-linear 4-bit grid (shared by IQ4_NL and IQ4_XS).
pub const IQ4_KVALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Unpack the 6-bit scales and mins of a Q4_K / Q5_K super-block.
#[inline]
pub fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Dequantise `n` elements (a whole number of blocks) of `dtype` from `src` into `dst`.
pub fn dequantize_row(dtype: GgmlType, src: &[u8], dst: &mut [f32]) -> Result<()> {
    let n = dst.len();
    let be = dtype.block_elems();
    if n % be != 0 {
        return Err(EngineError::InvalidShape(format!(
            "{n} elements is not a multiple of the {dtype} block ({be})"
        )));
    }
    let nb = n / be;
    let bb = dtype.block_bytes();
    if src.len() < nb * bb {
        return Err(EngineError::Format(format!(
            "{dtype} row needs {} bytes, got {}",
            nb * bb,
            src.len()
        )));
    }
    match dtype {
        GgmlType::F32 => {
            for (i, y) in dst.iter_mut().enumerate() {
                *y = f32::from_le_bytes(src[i * 4..i * 4 + 4].try_into().unwrap());
            }
        }
        GgmlType::F16 => {
            for (i, y) in dst.iter_mut().enumerate() {
                *y = f16_at(src, i * 2);
            }
        }
        GgmlType::BF16 => {
            for (i, y) in dst.iter_mut().enumerate() {
                *y = bf16_to_f32(src[i * 2], src[i * 2 + 1]);
            }
        }
        GgmlType::Q4_0 => {
            for (b, y) in dst.chunks_exact_mut(32).enumerate() {
                let blk = &src[b * bb..(b + 1) * bb];
                let d = f16_at(blk, 0);
                let qs = &blk[2..18];
                for j in 0..16 {
                    y[j] = ((qs[j] & 0xF) as i32 - 8) as f32 * d;
                    y[j + 16] = ((qs[j] >> 4) as i32 - 8) as f32 * d;
                }
            }
        }
        GgmlType::Q4_1 => {
            for (b, y) in dst.chunks_exact_mut(32).enumerate() {
                let blk = &src[b * bb..(b + 1) * bb];
                let d = f16_at(blk, 0);
                let m = f16_at(blk, 2);
                let qs = &blk[4..20];
                for j in 0..16 {
                    y[j] = (qs[j] & 0xF) as f32 * d + m;
                    y[j + 16] = (qs[j] >> 4) as f32 * d + m;
                }
            }
        }
        GgmlType::Q5_0 => {
            for (b, y) in dst.chunks_exact_mut(32).enumerate() {
                let blk = &src[b * bb..(b + 1) * bb];
                let d = f16_at(blk, 0);
                let qh = u32::from_le_bytes(blk[2..6].try_into().unwrap());
                let qs = &blk[6..22];
                for j in 0..16 {
                    let xh0 = ((qh >> j) << 4) & 0x10;
                    let xh1 = (qh >> (j + 12)) & 0x10;
                    let x0 = ((qs[j] & 0xF) as u32 | xh0) as i32 - 16;
                    let x1 = ((qs[j] >> 4) as u32 | xh1) as i32 - 16;
                    y[j] = x0 as f32 * d;
                    y[j + 16] = x1 as f32 * d;
                }
            }
        }
        GgmlType::Q5_1 => {
            for (b, y) in dst.chunks_exact_mut(32).enumerate() {
                let blk = &src[b * bb..(b + 1) * bb];
                let d = f16_at(blk, 0);
                let m = f16_at(blk, 2);
                let qh = u32::from_le_bytes(blk[4..8].try_into().unwrap());
                let qs = &blk[8..24];
                for j in 0..16 {
                    let xh0 = ((qh >> j) << 4) & 0x10;
                    let xh1 = (qh >> (j + 12)) & 0x10;
                    let x0 = (qs[j] & 0xF) as u32 | xh0;
                    let x1 = (qs[j] >> 4) as u32 | xh1;
                    y[j] = x0 as f32 * d + m;
                    y[j + 16] = x1 as f32 * d + m;
                }
            }
        }
        GgmlType::Q8_0 => {
            for (b, y) in dst.chunks_exact_mut(32).enumerate() {
                let blk = &src[b * bb..(b + 1) * bb];
                let d = f16_at(blk, 0);
                for j in 0..32 {
                    y[j] = (blk[2 + j] as i8) as f32 * d;
                }
            }
        }
        GgmlType::Q2_K => {
            // scales[16] (low nibble scale, high nibble min), qs[64], d, dmin
            for (b, y) in dst.chunks_exact_mut(QK_K).enumerate() {
                let blk = &src[b * bb..(b + 1) * bb];
                let scales = &blk[0..16];
                let qs = &blk[16..80];
                let d = f16_at(blk, 80);
                let dmin = f16_at(blk, 82);
                for g in 0..16 {
                    let dl = d * (scales[g] & 0xF) as f32;
                    let ml = dmin * (scales[g] >> 4) as f32;
                    let h = g / 8;
                    let shift = 2 * ((g % 8) / 2);
                    let base = h * 32 + (g % 2) * 16;
                    for l in 0..16 {
                        let q = (qs[base + l] >> shift) & 3;
                        y[g * 16 + l] = dl * q as f32 - ml;
                    }
                }
            }
        }
        GgmlType::Q3_K => {
            // hmask[32], qs[64], scales[12], d
            for (b, y) in dst.chunks_exact_mut(QK_K).enumerate() {
                let blk = &src[b * bb..(b + 1) * bb];
                let hmask = &blk[0..32];
                let qs = &blk[32..96];
                let sc = &blk[96..108];
                let d = f16_at(blk, 108);
                let mut scales = [0i32; 16];
                for (j, s) in scales.iter_mut().enumerate() {
                    let lo = (sc[j % 8] >> (4 * (j / 8))) & 0xF;
                    let hi = (sc[8 + j % 4] >> (2 * (j / 4))) & 3;
                    *s = ((lo | (hi << 4)) as i8) as i32 - 32;
                }
                for g in 0..16 {
                    let dl = d * scales[g] as f32;
                    let h = g / 8;
                    let shift = 2 * ((g % 8) / 2);
                    let base = h * 32 + (g % 2) * 16;
                    let hbase = (g % 2) * 16;
                    let bit = g / 2;
                    for l in 0..16 {
                        let ql = ((qs[base + l] >> shift) & 3) as i32;
                        let qh = ((hmask[hbase + l] >> bit) & 1) as i32;
                        // the offset is zero when the high bit is set
                        let q = ql - ((qh ^ 1) << 2);
                        y[g * 16 + l] = dl * q as f32;
                    }
                }
            }
        }
        GgmlType::Q4_K => {
            // d, dmin, scales[12], qs[128]
            for (b, y) in dst.chunks_exact_mut(QK_K).enumerate() {
                let blk = &src[b * bb..(b + 1) * bb];
                let d = f16_at(blk, 0);
                let dmin = f16_at(blk, 2);
                let sc = &blk[4..4 + K_SCALE_SIZE];
                let qs = &blk[16..16 + QK_K / 2];
                for s in 0..8 {
                    let (scale, min) = get_scale_min_k4(s, sc);
                    let dl = d * scale as f32;
                    let ml = dmin * min as f32;
                    let chunk = s / 2;
                    let shift = 4 * (s % 2);
                    for l in 0..32 {
                        let q = (qs[chunk * 32 + l] >> shift) & 0xF;
                        y[s * 32 + l] = dl * q as f32 - ml;
                    }
                }
            }
        }
        GgmlType::Q5_K => {
            // d, dmin, scales[12], qh[32], qs[128]
            for (b, y) in dst.chunks_exact_mut(QK_K).enumerate() {
                let blk = &src[b * bb..(b + 1) * bb];
                let d = f16_at(blk, 0);
                let dmin = f16_at(blk, 2);
                let sc = &blk[4..4 + K_SCALE_SIZE];
                let qh = &blk[16..48];
                let qs = &blk[48..48 + QK_K / 2];
                for s in 0..8 {
                    let (scale, min) = get_scale_min_k4(s, sc);
                    let dl = d * scale as f32;
                    let ml = dmin * min as f32;
                    let chunk = s / 2;
                    let shift = 4 * (s % 2);
                    for l in 0..32 {
                        let lo = (qs[chunk * 32 + l] >> shift) & 0xF;
                        let hi = (qh[l] >> s) & 1;
                        let q = lo | (hi << 4);
                        y[s * 32 + l] = dl * q as f32 - ml;
                    }
                }
            }
        }
        GgmlType::Q6_K => {
            // ql[128], qh[64], scales[16] (i8), d
            for (b, y) in dst.chunks_exact_mut(QK_K).enumerate() {
                let blk = &src[b * bb..(b + 1) * bb];
                let ql = &blk[0..128];
                let qh = &blk[128..192];
                let sc = &blk[192..208];
                let d = f16_at(blk, 208);
                for h in 0..2 {
                    for l in 0..32 {
                        let qlh = &ql[h * 64..h * 64 + 64];
                        let qhh = &qh[h * 32..h * 32 + 32];
                        let q1 = ((qlh[l] & 0xF) | (((qhh[l]) & 3) << 4)) as i32 - 32;
                        let q2 = ((qlh[l + 32] & 0xF) | (((qhh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((qlh[l] >> 4) | (((qhh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((qlh[l + 32] >> 4) | (((qhh[l] >> 6) & 3) << 4)) as i32 - 32;
                        let base = h * 128;
                        let is = l / 16;
                        let s = &sc[h * 8..h * 8 + 8];
                        y[base + l] = d * (s[is] as i8) as f32 * q1 as f32;
                        y[base + l + 32] = d * (s[is + 2] as i8) as f32 * q2 as f32;
                        y[base + l + 64] = d * (s[is + 4] as i8) as f32 * q3 as f32;
                        y[base + l + 96] = d * (s[is + 6] as i8) as f32 * q4 as f32;
                    }
                }
            }
        }
        GgmlType::Q8_K => {
            // d (f32), qs[256] (i8), bsums[16] (i16)
            for (b, y) in dst.chunks_exact_mut(QK_K).enumerate() {
                let blk = &src[b * bb..(b + 1) * bb];
                let d = f32::from_le_bytes(blk[0..4].try_into().unwrap());
                for j in 0..QK_K {
                    y[j] = (blk[4 + j] as i8) as f32 * d;
                }
            }
        }
        GgmlType::IQ4_NL => {
            for (b, y) in dst.chunks_exact_mut(32).enumerate() {
                let blk = &src[b * bb..(b + 1) * bb];
                let d = f16_at(blk, 0);
                let qs = &blk[2..18];
                for j in 0..16 {
                    y[j] = IQ4_KVALUES[(qs[j] & 0xF) as usize] as f32 * d;
                    y[j + 16] = IQ4_KVALUES[(qs[j] >> 4) as usize] as f32 * d;
                }
            }
        }
        GgmlType::IQ4_XS => {
            // d, scales_h (u16), scales_l[4], qs[128]
            for (b, y) in dst.chunks_exact_mut(QK_K).enumerate() {
                let blk = &src[b * bb..(b + 1) * bb];
                let d = f16_at(blk, 0);
                let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
                let scales_l = &blk[4..8];
                let qs = &blk[8..136];
                for ib in 0..8 {
                    let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF)
                        | ((((scales_h >> (2 * ib)) & 3) as u8) << 4);
                    let dl = d * (ls as i32 - 32) as f32;
                    let q = &qs[ib * 16..ib * 16 + 16];
                    for j in 0..16 {
                        y[ib * 32 + j] = dl * IQ4_KVALUES[(q[j] & 0xF) as usize] as f32;
                        y[ib * 32 + 16 + j] = dl * IQ4_KVALUES[(q[j] >> 4) as usize] as f32;
                    }
                }
            }
        }
        other => return Err(EngineError::UnsupportedType(other)),
    }
    Ok(())
}

/// Types [`dequantize_row`] handles in this build.
pub fn dequant_supported(t: GgmlType) -> bool {
    matches!(
        t,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q8_0
            | GgmlType::Q2_K
            | GgmlType::Q3_K
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::Q8_K
            | GgmlType::IQ4_NL
            | GgmlType::IQ4_XS
    )
}

#[inline]
fn nearest_int(x: f32) -> i32 {
    x.round_ties_even() as i32
}

/// Quantise a row of f32 activations to Q8_0 (ggml `quantize_row_q8_0_ref`). `dst` receives
/// `n/32` blocks of 34 bytes.
pub fn quantize_row_q8_0(x: &[f32], dst: &mut [u8]) {
    assert_eq!(x.len() % 32, 0);
    assert!(dst.len() >= x.len() / 32 * 34);
    for (b, xb) in x.chunks_exact(32).enumerate() {
        let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let blk = &mut dst[b * 34..(b + 1) * 34];
        blk[0..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
        for j in 0..32 {
            blk[2 + j] = nearest_int(xb[j] * id) as i8 as u8;
        }
    }
}

/// Quantise a row of f32 activations to Q8_K (ggml `quantize_row_q8_K_ref`): per 256 elements a
/// signed scale chosen so the largest-magnitude value maps to −127, 256 int8 quants and 16 block
/// sums. `dst` receives `n/256` blocks of 292 bytes.
pub fn quantize_row_q8_k(x: &[f32], dst: &mut [u8]) {
    assert_eq!(x.len() % QK_K, 0);
    assert!(dst.len() >= x.len() / QK_K * 292);
    for (b, xb) in x.chunks_exact(QK_K).enumerate() {
        let blk = &mut dst[b * 292..(b + 1) * 292];
        let mut amax = 0f32;
        let mut max = 0f32;
        for &v in xb {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        if amax == 0.0 {
            blk.fill(0);
            continue;
        }
        let iscale = -127.0 / max;
        let mut qs = [0i8; QK_K];
        for j in 0..QK_K {
            qs[j] = nearest_int(iscale * xb[j]).min(127) as i8;
        }
        blk[0..4].copy_from_slice(&(1.0 / iscale).to_le_bytes());
        for j in 0..QK_K {
            blk[4 + j] = qs[j] as u8;
        }
        for g in 0..16 {
            let s: i32 = qs[g * 16..g * 16 + 16].iter().map(|&q| q as i32).sum();
            blk[260 + g * 2..262 + g * 2].copy_from_slice(&(s as i16).to_le_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_round_trips_small_values() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.5).collect();
        let mut q = vec![0u8; 2 * 34];
        quantize_row_q8_0(&x, &mut q);
        let mut y = vec![0f32; 64];
        dequantize_row(GgmlType::Q8_0, &q, &mut y).unwrap();
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 0.2, "{a} vs {b}");
        }
    }

    #[test]
    fn q8_k_round_trips_and_sums() {
        let x: Vec<f32> = (0..256).map(|i| ((i * 7919) % 101) as f32 - 50.0).collect();
        let mut q = vec![0u8; 292];
        quantize_row_q8_k(&x, &mut q);
        let mut y = vec![0f32; 256];
        dequantize_row(GgmlType::Q8_K, &q, &mut y).unwrap();
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 0.3, "{a} vs {b}");
        }
        // block sums agree with the quants
        for g in 0..16 {
            let s = i16::from_le_bytes([q[260 + g * 2], q[261 + g * 2]]) as i32;
            let t: i32 = (0..16).map(|j| q[4 + g * 16 + j] as i8 as i32).sum();
            assert_eq!(s, t);
        }
    }

    #[test]
    fn q4_0_known_block() {
        // d = 1.0, nibbles 0..15 then 15..0
        let mut blk = vec![0u8; 18];
        blk[0..2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        for j in 0..16 {
            blk[2 + j] = (j as u8) | (((15 - j) as u8) << 4);
        }
        let mut y = [0f32; 32];
        dequantize_row(GgmlType::Q4_0, &blk, &mut y).unwrap();
        assert_eq!(y[0], -8.0);
        assert_eq!(y[15], 7.0);
        assert_eq!(y[16], 7.0);
        assert_eq!(y[31], -8.0);
    }

    #[test]
    fn scale_min_unpacking_matches_gguf_py_pattern() {
        // bytes: d[0..4] = EEAAAAAA, m[0..4] = eeaaaaaa, m_d[0..4] = eeeeEEEE
        let q = [
            0b11_000001,
            0b10_000010,
            0b01_000011,
            0b00_000100,
            0b11_000101,
            0b10_000110,
            0b01_000111,
            0b00_001000,
            0b1001_0001,
            0b1010_0010,
            0b1011_0011,
            0b1100_0100,
        ];
        assert_eq!(get_scale_min_k4(0, &q), (1, 5));
        assert_eq!(get_scale_min_k4(3, &q), (4, 8));
        // j=4: sc = (q[8]&0xF) | ((q[0]>>6)<<4) = 1 | (3<<4) = 49; min = (q[8]>>4) | ((q[4]>>6)<<4) = 9 | 48 = 57
        assert_eq!(get_scale_min_k4(4, &q), (49, 57));
        assert_eq!(get_scale_min_k4(7, &q), (4, 12));
    }

    #[test]
    fn rejects_bad_sizes() {
        let mut y = vec![0f32; 33];
        assert!(dequantize_row(GgmlType::Q4_0, &[0; 36], &mut y).is_err());
        let mut y = vec![0f32; 32];
        assert!(dequantize_row(GgmlType::Q4_0, &[0; 10], &mut y).is_err());
        assert!(dequantize_row(GgmlType::IQ2_XXS, &[0; 66], &mut vec![0f32; 256]).is_err());
    }
}
