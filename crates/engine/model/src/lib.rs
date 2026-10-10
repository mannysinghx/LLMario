//! Model description layer and forward passes (Architecture §6.7).
//!
//! - [`arch::ArchSpec`] is data parsed from GGUF metadata: layer counts, head geometry, RoPE,
//!   norms, which blocks each layer uses. The planner and the KV cache are built from it without
//!   touching tensor bytes.
//! - [`weights::Weights`] maps tensor names to zero-copy [`llmario_engine_cpu::QMat`] views.
//! - [`forward::Model`] runs the dense transformer families (Llama/Mistral, Qwen2, Qwen3,
//!   SmolLM3-style NoPE layers) on the CPU backend with a per-slot KV cache, and the Qwen3.5 /
//!   Qwen3-Next hybrid family ([`hybrid`]: Gated DeltaNet layers from [`gdn`] plus gated
//!   full-attention layers) with the fp32 recurrent state kept per sequence in the same
//!   [`kv::KvCache`], and the Gemma 4 family ([`gemma4`]: sliding-window / global attention
//!   with per-layer geometry, the window layers cached in rings), and the Qwen3 MoE family
//!   ([`moe`]: the Qwen3 attention block with a routed mixture-of-experts FFN).

pub mod arch;
pub mod backend;
pub mod forward;
pub mod gdn;
pub mod gemma4;
pub mod hybrid;
pub mod kv;
pub mod moe;
pub mod stream;
pub mod weights;

pub use arch::{ArchSpec, AttnGeom, BlockKind, Family, GdnSpec, Gemma4Spec, MoeSpec};
pub use backend::{CpuBackend, CpuOptions, ModelBackend};
pub use forward::{Model, SeqTokens};
pub use kv::{KvCache, KvFull, KvOptions, KvType, SeqSnapshot, StableHasher};
pub use stream::StreamOptions;

#[derive(thiserror::Error, Debug)]
pub enum ModelError {
    #[error("unsupported architecture `{0}`")]
    UnsupportedArch(String),
    #[error("missing metadata key `{0}`")]
    MissingKey(String),
    #[error("missing tensor `{0}`")]
    MissingTensor(String),
    #[error("tensor `{name}` has shape {shape}, expected {expected}")]
    BadShape {
        name: String,
        shape: String,
        expected: String,
    },
    #[error("{0}")]
    Engine(#[from] llmario_engine_core::EngineError),
}

pub type Result<T> = std::result::Result<T, ModelError>;
