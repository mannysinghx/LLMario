//! Model file readers for the native engine.
//!
//! [`gguf`] reads GGUF v3 files (single or `-00001-of-NNNNN` split sets) by memory-mapping them
//! read-only and parsing the header once. Tensors are zero-copy views into the mapping; nothing is
//! read until a backend touches it. Metadata is exposed as typed values with the key names the
//! llama.cpp converter writes.
//!
//! [`safetensors`] reads Hugging Face / MLX model folders (`config.json` + one or more
//! `.safetensors` shards) the same way, and resolves MLX affine-quantised weight triplets
//! (`.weight` / `.scales` / `.biases`) into [`MlxQuantView`]s.

pub mod gguf;
pub mod safetensors;

pub use gguf::{GgufFile, MetaValue};
pub use safetensors::{MlxQuantParams, MlxQuantView, MlxQuantization, SafetensorsFolder};
