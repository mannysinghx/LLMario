//! Tensor element types.
//!
//! The ids and block geometry are the ggml ones (`ggml_type` / GGUF tensor type), transcribed from
//! `ggml-common.h` and `gguf-py/gguf/constants.py` (both MIT) as of October 2026. Block bytes are the
//! C `sizeof` values, which are what every GGUF writer uses; a test recomputes each from the struct
//! fields so a transcription slip fails loudly.

use serde::{Deserialize, Serialize};

pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1_M,
    BF16,
    TQ1_0,
    TQ2_0,
    MXFP4,
    NVFP4,
    Q1_0,
    Q2_0,
}

impl GgmlType {
    /// All types, in id order.
    pub const ALL: [GgmlType; 35] = [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::Q4_0,
        GgmlType::Q4_1,
        GgmlType::Q5_0,
        GgmlType::Q5_1,
        GgmlType::Q8_0,
        GgmlType::Q8_1,
        GgmlType::Q2_K,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
        GgmlType::Q8_K,
        GgmlType::IQ2_XXS,
        GgmlType::IQ2_XS,
        GgmlType::IQ3_XXS,
        GgmlType::IQ1_S,
        GgmlType::IQ4_NL,
        GgmlType::IQ3_S,
        GgmlType::IQ2_S,
        GgmlType::IQ4_XS,
        GgmlType::I8,
        GgmlType::I16,
        GgmlType::I32,
        GgmlType::I64,
        GgmlType::F64,
        GgmlType::IQ1_M,
        GgmlType::BF16,
        GgmlType::TQ1_0,
        GgmlType::TQ2_0,
        GgmlType::MXFP4,
        GgmlType::NVFP4,
        GgmlType::Q1_0,
        GgmlType::Q2_0,
    ];

    /// The GGUF / ggml type id.
    pub fn id(self) -> u32 {
        match self {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::Q4_0 => 2,
            GgmlType::Q4_1 => 3,
            GgmlType::Q5_0 => 6,
            GgmlType::Q5_1 => 7,
            GgmlType::Q8_0 => 8,
            GgmlType::Q8_1 => 9,
            GgmlType::Q2_K => 10,
            GgmlType::Q3_K => 11,
            GgmlType::Q4_K => 12,
            GgmlType::Q5_K => 13,
            GgmlType::Q6_K => 14,
            GgmlType::Q8_K => 15,
            GgmlType::IQ2_XXS => 16,
            GgmlType::IQ2_XS => 17,
            GgmlType::IQ3_XXS => 18,
            GgmlType::IQ1_S => 19,
            GgmlType::IQ4_NL => 20,
            GgmlType::IQ3_S => 21,
            GgmlType::IQ2_S => 22,
            GgmlType::IQ4_XS => 23,
            GgmlType::I8 => 24,
            GgmlType::I16 => 25,
            GgmlType::I32 => 26,
            GgmlType::I64 => 27,
            GgmlType::F64 => 28,
            GgmlType::IQ1_M => 29,
            GgmlType::BF16 => 30,
            GgmlType::TQ1_0 => 34,
            GgmlType::TQ2_0 => 35,
            GgmlType::MXFP4 => 39,
            GgmlType::NVFP4 => 40,
            GgmlType::Q1_0 => 41,
            GgmlType::Q2_0 => 42,
        }
    }

    pub fn from_id(id: u32) -> Option<GgmlType> {
        GgmlType::ALL.iter().copied().find(|t| t.id() == id)
    }

    /// ggml's type name (what `llama-gguf` and the catalog print).
    pub fn name(self) -> &'static str {
        match self {
            GgmlType::F32 => "f32",
            GgmlType::F16 => "f16",
            GgmlType::Q4_0 => "q4_0",
            GgmlType::Q4_1 => "q4_1",
            GgmlType::Q5_0 => "q5_0",
            GgmlType::Q5_1 => "q5_1",
            GgmlType::Q8_0 => "q8_0",
            GgmlType::Q8_1 => "q8_1",
            GgmlType::Q2_K => "q2_K",
            GgmlType::Q3_K => "q3_K",
            GgmlType::Q4_K => "q4_K",
            GgmlType::Q5_K => "q5_K",
            GgmlType::Q6_K => "q6_K",
            GgmlType::Q8_K => "q8_K",
            GgmlType::IQ2_XXS => "iq2_xxs",
            GgmlType::IQ2_XS => "iq2_xs",
            GgmlType::IQ3_XXS => "iq3_xxs",
            GgmlType::IQ1_S => "iq1_s",
            GgmlType::IQ4_NL => "iq4_nl",
            GgmlType::IQ3_S => "iq3_s",
            GgmlType::IQ2_S => "iq2_s",
            GgmlType::IQ4_XS => "iq4_xs",
            GgmlType::I8 => "i8",
            GgmlType::I16 => "i16",
            GgmlType::I32 => "i32",
            GgmlType::I64 => "i64",
            GgmlType::F64 => "f64",
            GgmlType::IQ1_M => "iq1_m",
            GgmlType::BF16 => "bf16",
            GgmlType::TQ1_0 => "tq1_0",
            GgmlType::TQ2_0 => "tq2_0",
            GgmlType::MXFP4 => "mxfp4",
            GgmlType::NVFP4 => "nvfp4",
            GgmlType::Q1_0 => "q1_0",
            GgmlType::Q2_0 => "q2_0",
        }
    }

    /// Elements per block (1 for plain types).
    pub fn block_elems(self) -> usize {
        match self {
            GgmlType::F32
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::F64
            | GgmlType::I8
            | GgmlType::I16
            | GgmlType::I32
            | GgmlType::I64 => 1,
            GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q8_0
            | GgmlType::Q8_1
            | GgmlType::IQ4_NL
            | GgmlType::MXFP4 => 32,
            GgmlType::NVFP4 | GgmlType::Q2_0 => 64,
            GgmlType::Q1_0 => 128,
            _ => QK_K,
        }
    }

    /// Bytes per block: the C `sizeof(block_*)`, written out from the struct fields.
    pub fn block_bytes(self) -> usize {
        const HALF: usize = 2;
        match self {
            GgmlType::F32 | GgmlType::I32 => 4,
            GgmlType::F16 | GgmlType::BF16 | GgmlType::I16 => 2,
            GgmlType::F64 | GgmlType::I64 => 8,
            GgmlType::I8 => 1,
            GgmlType::Q4_0 => HALF + 32 / 2,
            GgmlType::Q4_1 => 2 * HALF + 32 / 2,
            GgmlType::Q5_0 => HALF + 4 + 32 / 2,
            GgmlType::Q5_1 => 2 * HALF + 4 + 32 / 2,
            GgmlType::Q8_0 => HALF + 32,
            GgmlType::Q8_1 => 2 * HALF + 32,
            GgmlType::Q2_K => 2 * HALF + QK_K / 16 + QK_K / 4,
            GgmlType::Q3_K => HALF + QK_K / 4 + QK_K / 8 + 12,
            GgmlType::Q4_K => 2 * HALF + K_SCALE_SIZE + QK_K / 2,
            GgmlType::Q5_K => 2 * HALF + K_SCALE_SIZE + QK_K / 2 + QK_K / 8,
            GgmlType::Q6_K => HALF + QK_K / 16 + 3 * QK_K / 4,
            GgmlType::Q8_K => 4 + QK_K + QK_K / 16 * 2,
            GgmlType::IQ2_XXS => HALF + QK_K / 8 * 2,
            GgmlType::IQ2_XS => HALF + QK_K / 8 * 2 + QK_K / 32,
            GgmlType::IQ3_XXS => HALF + 3 * (QK_K / 8),
            GgmlType::IQ1_S => HALF + QK_K / 8 + QK_K / 16,
            GgmlType::IQ4_NL => HALF + 32 / 2,
            GgmlType::IQ3_S => HALF + 13 * (QK_K / 32) + QK_K / 64,
            GgmlType::IQ2_S => HALF + QK_K / 4 + QK_K / 16,
            GgmlType::IQ4_XS => HALF + 2 + QK_K / 64 + QK_K / 2,
            GgmlType::IQ1_M => QK_K / 8 + QK_K / 16 + QK_K / 32,
            GgmlType::TQ1_0 => HALF + QK_K / 64 + (QK_K - 4 * QK_K / 64) / 5,
            GgmlType::TQ2_0 => HALF + QK_K / 4,
            GgmlType::MXFP4 => 1 + 32 / 2,
            GgmlType::NVFP4 => 64 / 16 + 64 / 2,
            GgmlType::Q1_0 => HALF + 128 / 8,
            GgmlType::Q2_0 => HALF + 64 / 4,
        }
    }

    /// True for block-quantised types (anything that is not a plain scalar element type).
    pub fn is_quantized(self) -> bool {
        self.block_elems() > 1
    }

    /// Bytes needed to hold `n` elements of this type (a row). `n` must be a multiple of the block
    /// size for quantised types; callers validate that from the tensor shape.
    pub fn row_bytes(self, n: usize) -> usize {
        n / self.block_elems() * self.block_bytes()
    }

    /// Bits per weight implied by the block geometry.
    pub fn bits_per_weight(self) -> f64 {
        self.block_bytes() as f64 * 8.0 / self.block_elems() as f64
    }
}

impl std::cmp::Ord for GgmlType {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id().cmp(&other.id())
    }
}
impl std::cmp::PartialOrd for GgmlType {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::fmt::Display for GgmlType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gguf-py `GGML_QUANT_SIZES` table (block elements, block bytes) as of 2026-10; it must
    /// agree with the C header transcription above for every type.
    const GGUF_PY: &[(GgmlType, usize, usize)] = &[
        (GgmlType::F32, 1, 4),
        (GgmlType::F16, 1, 2),
        (GgmlType::Q4_0, 32, 18),
        (GgmlType::Q4_1, 32, 20),
        (GgmlType::Q5_0, 32, 22),
        (GgmlType::Q5_1, 32, 24),
        (GgmlType::Q8_0, 32, 34),
        (GgmlType::Q8_1, 32, 36),
        (GgmlType::Q2_K, 256, 84),
        (GgmlType::Q3_K, 256, 110),
        (GgmlType::Q4_K, 256, 144),
        (GgmlType::Q5_K, 256, 176),
        (GgmlType::Q6_K, 256, 210),
        (GgmlType::Q8_K, 256, 292),
        (GgmlType::IQ2_XXS, 256, 66),
        (GgmlType::IQ2_XS, 256, 74),
        (GgmlType::IQ3_XXS, 256, 98),
        (GgmlType::IQ1_S, 256, 50),
        (GgmlType::IQ4_NL, 32, 18),
        (GgmlType::IQ3_S, 256, 110),
        (GgmlType::IQ2_S, 256, 82),
        (GgmlType::IQ4_XS, 256, 136),
        (GgmlType::I8, 1, 1),
        (GgmlType::I16, 1, 2),
        (GgmlType::I32, 1, 4),
        (GgmlType::I64, 1, 8),
        (GgmlType::F64, 1, 8),
        (GgmlType::IQ1_M, 256, 56),
        (GgmlType::BF16, 1, 2),
        (GgmlType::TQ1_0, 256, 54),
        (GgmlType::TQ2_0, 256, 66),
        (GgmlType::MXFP4, 32, 17),
        (GgmlType::NVFP4, 64, 36),
        (GgmlType::Q1_0, 128, 18),
        (GgmlType::Q2_0, 64, 18),
    ];

    #[test]
    fn block_geometry_matches_gguf_py_and_header() {
        assert_eq!(GGUF_PY.len(), GgmlType::ALL.len());
        for &(t, elems, bytes) in GGUF_PY {
            assert_eq!(t.block_elems(), elems, "{t} block elems");
            assert_eq!(t.block_bytes(), bytes, "{t} block bytes");
        }
    }

    #[test]
    fn ids_round_trip() {
        for t in GgmlType::ALL {
            assert_eq!(GgmlType::from_id(t.id()), Some(t));
        }
        assert_eq!(GgmlType::from_id(4), None); // Q4_2 was removed from ggml
        assert_eq!(GgmlType::from_id(5), None);
    }

    #[test]
    fn row_bytes_and_bpw() {
        assert_eq!(GgmlType::Q4_K.row_bytes(4096), 4096 / 256 * 144);
        assert!((GgmlType::Q4_K.bits_per_weight() - 4.5).abs() < 1e-9);
        assert!((GgmlType::Q8_0.bits_per_weight() - 8.5).abs() < 1e-9);
    }
}
