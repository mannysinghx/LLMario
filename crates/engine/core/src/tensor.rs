//! Shapes and zero-copy tensor descriptions.
//!
//! A [`TensorInfo`] describes where a tensor's bytes live (which file or arena, at what offset) and
//! how to interpret them. It owns nothing; the file mapping or device arena owns the bytes.
//! Dimensions follow ggml order: `ne[0]` is the innermost (contiguous, row) dimension, so a matrix
//! that multiplies a `d_model`-vector into `d_out` has shape `[d_model, d_out]`.

use crate::dtype::GgmlType;
use crate::{EngineError, Result};
use serde::{Deserialize, Serialize};

pub const MAX_DIMS: usize = 4;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    pub ne: [u64; MAX_DIMS],
    pub n_dims: u8,
}

impl Shape {
    pub fn new(dims: &[u64]) -> Result<Shape> {
        if dims.is_empty() || dims.len() > MAX_DIMS {
            return Err(EngineError::InvalidShape(format!("{} dims", dims.len())));
        }
        let mut ne = [1u64; MAX_DIMS];
        ne[..dims.len()].copy_from_slice(dims);
        Ok(Shape {
            ne,
            n_dims: dims.len() as u8,
        })
    }

    pub fn dims(&self) -> &[u64] {
        &self.ne[..self.n_dims as usize]
    }

    /// Number of elements.
    pub fn numel(&self) -> u64 {
        self.ne.iter().product()
    }

    /// Innermost (row) length.
    pub fn row_len(&self) -> u64 {
        self.ne[0]
    }

    /// Number of rows (product of all outer dims).
    pub fn rows(&self) -> u64 {
        self.ne[1..].iter().product()
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

/// Where a tensor's bytes are: index into the owner's file/arena list plus an offset.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ByteSpan {
    /// Index of the file (GGUF split part) or arena.
    pub source: u32,
    pub offset: u64,
    pub len: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: GgmlType,
    pub shape: Shape,
    pub span: ByteSpan,
}

impl TensorInfo {
    /// Bytes a contiguous tensor of this type and shape occupies; errors when the row length is
    /// not a whole number of blocks (a corrupt or unsupported file).
    pub fn expected_bytes(dtype: GgmlType, shape: &Shape) -> Result<u64> {
        let row = shape.row_len();
        let be = dtype.block_elems() as u64;
        if row % be != 0 {
            return Err(EngineError::InvalidShape(format!(
                "row length {row} is not a multiple of the {dtype} block ({be})"
            )));
        }
        Ok(row / be * dtype.block_bytes() as u64 * shape.rows())
    }

    pub fn row_bytes(&self) -> u64 {
        self.dtype.row_bytes(self.shape.row_len() as usize) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_basics() {
        let s = Shape::new(&[4096, 151936]).unwrap();
        assert_eq!(s.numel(), 4096 * 151936);
        assert_eq!(s.rows(), 151936);
        assert_eq!(s.to_string(), "[4096, 151936]");
        assert!(Shape::new(&[]).is_err());
    }

    #[test]
    fn expected_bytes_checks_blocks() {
        let s = Shape::new(&[4096, 10]).unwrap();
        assert_eq!(
            TensorInfo::expected_bytes(GgmlType::Q4_K, &s).unwrap(),
            4096 / 256 * 144 * 10
        );
        let bad = Shape::new(&[100, 10]).unwrap();
        assert!(TensorInfo::expected_bytes(GgmlType::Q4_K, &bad).is_err());
        assert_eq!(
            TensorInfo::expected_bytes(GgmlType::F32, &bad).unwrap(),
            4000
        );
    }
}
