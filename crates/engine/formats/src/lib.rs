//! Model file readers for the native engine.
//!
//! [`gguf`] reads GGUF v3 files (single or `-00001-of-NNNNN` split sets) by memory-mapping them
//! read-only and parsing the header once. Tensors are zero-copy views into the mapping; nothing is
//! read until a backend touches it. Metadata is exposed as typed values with the key names the
//! llama.cpp converter writes.

pub mod gguf;

pub use gguf::{GgufFile, MetaValue};
