//! Metal GPU backend of the native engine (Architecture §7.4, milestone M2).
//!
//! - Weights stay in the GGUF mapping: each part is wrapped zero-copy in shared-storage
//!   `MTLBuffer`s (page-aligned views, each ≤ `maxBufferLength`, overlapping so every tensor lies
//!   inside one view) and tensors are addressed as (view, offset).
//! - The KV cache is paged (`kv.rs`): blocks of 32 positions, each a shared buffer created when a
//!   sequence reaches it and released when unused, reached by the kernels through per-sequence
//!   tables of GPU addresses; f16 or q8_0 rows. Several sequences share one pool and run in one
//!   forward call. The activation scratch is allocated once from the model shape and the batch.
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

/// Options of the Metal backend's cache and scratch.
#[derive(Clone, Copy, Debug)]
pub struct MetalOptions {
    /// Positions per slot.
    pub max_ctx: usize,
    /// Tokens per forward call (all slots together).
    pub n_batch: usize,
    /// Slots sharing the KV pool.
    pub n_seqs: usize,
    pub kv_type: llmario_engine_model::KvType,
}

impl MetalOptions {
    pub fn new(max_ctx: usize, n_batch: usize) -> MetalOptions {
        MetalOptions {
            max_ctx,
            n_batch,
            n_seqs: 1,
            kv_type: llmario_engine_model::KvType::F16,
        }
    }
}

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
mod kv;

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
        pub fn with_options(_file: &'a GgufFile, _o: MetalOptions) -> Result<Self> {
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
        fn seq_len(&self, _s: usize) -> usize {
            0
        }
        fn truncate_seq(&mut self, _s: usize, _n: usize) {}
        fn clear_seq(&mut self, _s: usize) {}
        fn max_batch(&self) -> usize {
            0
        }
        fn forward_batch(
            &mut self,
            _batch: &[llmario_engine_model::SeqTokens],
        ) -> std::result::Result<&[f32], llmario_engine_model::KvFull> {
            Ok(&[])
        }
        fn reserved_bytes(&self) -> u64 {
            0
        }
        fn kv_in_use_bytes(&self) -> u64 {
            0
        }
        fn kv_reserved_bytes(&self) -> u64 {
            0
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub use stub::MetalBackend;
