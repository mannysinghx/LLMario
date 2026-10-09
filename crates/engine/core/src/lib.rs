//! Core types of the LLMario native inference engine.
//!
//! - [`dtype`]: every ggml tensor type with its block size in elements and bytes, transcribed from
//!   `ggml-common.h` (the writers' source of truth), plus the engine's own types.
//! - [`dequant`]: scalar reference dequantizers. Every SIMD or GPU kernel is tested against these.
//! - [`tensor`]: shapes and zero-copy tensor views over mapped files or arenas.
//! - [`ledger`]: the per-device memory ledger (Architecture §5.1).
//!
//! Nothing here allocates model memory; allocation is the planner's and the backends' job.

pub mod dequant;
pub mod dtype;
pub mod ledger;
pub mod tensor;

pub use dtype::GgmlType;
pub use ledger::{DeviceId, DeviceLedger, Ledger};
pub use tensor::{Shape, TensorInfo};

/// Errors shared by engine crates.
#[derive(thiserror::Error, Debug)]
pub enum EngineError {
    #[error("unsupported tensor type id {0}")]
    UnknownType(u32),
    #[error("tensor type {0:?} is not supported by this build (no kernel)")]
    UnsupportedType(GgmlType),
    #[error("invalid shape: {0}")]
    InvalidShape(String),
    #[error("{0}")]
    Format(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, EngineError>;
