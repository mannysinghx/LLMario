//! Metal GPU backend of the native engine (Architecture §7.4, milestone M2).
//!
//! - Weights stay in the GGUF mapping: each part is wrapped zero-copy in shared-storage
//!   `MTLBuffer`s (page-aligned views, each ≤ `maxBufferLength`, overlapping so every tensor lies
//!   inside one view) and tensors are addressed as (view, offset).
//! - The KV cache (f16 K and V) and the activation scratch are shared buffers allocated once from
//!   the model shape, the admitted context and the batch size; nothing is allocated per token.
//! - Kernels are MSL source embedded in the binary and compiled at runtime by the OS compiler, so
//!   the build never needs Xcode. One command buffer per forward call, kernels encoded back to
//!   back on the inference thread, logits copied out as f32.
//! - Residency: the weight views and the KV/scratch buffers go into `MTLResidencySet`s attached
//!   to the command queue on macOS 15+, so the wired collector does not unwire them after idle.
//!
//! On non-macOS targets the crate compiles to a stub whose `is_available()` is `false`.

#[derive(thiserror::Error, Debug)]
pub enum MetalError {
    #[error("Metal device: {0}")]
    Device(String),
    #[error("Metal shader compilation failed: {0}")]
    Compile(String),
    #[error("Metal buffer allocation of {0} bytes failed")]
    Alloc(usize),
    #[error("GPU execution failed: {0}")]
    Gpu(String),
    #[error("unsupported on the Metal backend: {0}")]
    Unsupported(String),
    #[error("{0}")]
    Model(#[from] llmario_engine_model::ModelError),
}

pub type Result<T> = std::result::Result<T, MetalError>;

/// What `probe` prints about the GPU.
#[derive(Clone, Debug, PartialEq)]
pub struct DeviceInfo {
    pub name: String,
    /// `MTLDevice.recommendedMaxWorkingSetSize`: the budget the planner should use.
    pub recommended_max_working_set: u64,
    pub has_unified_memory: bool,
    /// Highest supported Apple GPU family plus the Metal version family.
    pub family: String,
    /// `MTLResidencySet` is available (macOS 15+).
    pub residency_sets: bool,
}

#[cfg(target_os = "macos")]
mod backend;
#[cfg(target_os = "macos")]
pub mod device;

#[cfg(target_os = "macos")]
pub use backend::MetalBackend;

#[cfg(not(target_os = "macos"))]
mod stub {
    use super::*;
    use llmario_engine_formats::GgufFile;
    use llmario_engine_model::{ArchSpec, ModelBackend};

    /// Stub on non-Apple targets: never available, never constructible.
    #[allow(dead_code)]
    pub struct MetalBackend<'a> {
        _file: &'a GgufFile,
        spec: ArchSpec,
    }

    impl<'a> MetalBackend<'a> {
        pub fn new(_file: &'a GgufFile, _max_ctx: usize, _n_batch: usize) -> Result<Self> {
            Err(MetalError::Device(
                "Metal is only available on macOS".into(),
            ))
        }
        pub fn is_available() -> bool {
            false
        }
        pub fn device_info() -> Option<DeviceInfo> {
            None
        }
    }

    impl ModelBackend for MetalBackend<'_> {
        fn spec(&self) -> &ArchSpec {
            &self.spec
        }
        fn name(&self) -> &'static str {
            "metal"
        }
        fn max_ctx(&self) -> usize {
            0
        }
        fn kv_len(&self) -> usize {
            0
        }
        fn truncate(&mut self, _n: usize) {}
        fn clear(&mut self) {}
        fn max_batch(&self) -> usize {
            0
        }
        fn forward(&mut self, _tokens: &[u32]) -> &[f32] {
            &[]
        }
        fn reserved_bytes(&self) -> u64 {
            0
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub use stub::MetalBackend;
