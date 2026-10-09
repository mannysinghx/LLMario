//! Native engine `decode` crate: what runs after each forward pass.
//!
//! - [`Sampler`] turns one logits vector into a token using llama.cpp's default sampler chain
//!   (penalties → top-k → top-p → min-p → temperature → multinomial draw; argmax when
//!   `temperature <= 0`), with a deterministic PRNG seeded from the request `seed`.
//! - [`LogitProcessor`] is the hook for constrained decoding (grammar masks) and logit bias,
//!   applied before the penalties.
//! - [`StopMatcher`] decides, per streamed text fragment, how much text is safe to emit given the
//!   request's stop strings.
//!
//! See `README.md` next to this crate for the semantics compared with llama.cpp.

pub mod params;
pub mod rng;
pub mod sampler;
pub mod stop;

/// A vocabulary token id.
pub type Token = u32;

pub use params::SamplingParams;
pub use rng::Xoshiro256StarStar;
pub use sampler::{LogitProcessor, Sampler};
pub use stop::{StopMatcher, StopResult};
