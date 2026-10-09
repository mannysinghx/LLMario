//! Sampling parameters, mirroring llama.cpp's `common_params_sampling` defaults where the
//! LLMario API does not override them.

use serde::{Deserialize, Serialize};

/// Per-request sampling configuration.
///
/// Field semantics follow llama.cpp's sampler chain (`common/sampling.cpp`); see the crate README
/// for the exact mapping and the order in which they are applied.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SamplingParams {
    /// Softmax temperature. `<= 0` selects greedy decoding (argmax of the processed logits).
    pub temperature: f32,
    /// Keep only the `top_k` highest logits. `<= 0` disables the filter.
    pub top_k: i32,
    /// Nucleus sampling: keep the smallest prefix of the (descending) candidates whose cumulative
    /// probability reaches `top_p`. `>= 1.0` disables the filter.
    pub top_p: f32,
    /// Drop candidates whose probability is below `min_p * p_max`. `<= 0` disables the filter.
    pub min_p: f32,
    /// Repetition penalty over the last `repeat_last_n` accepted tokens. `1.0` disables it.
    pub repeat_penalty: f32,
    /// Size of the penalty window in tokens. `0` disables all three penalties.
    pub repeat_last_n: usize,
    /// Presence penalty: subtracted once from every token present in the window.
    pub presence_penalty: f32,
    /// Frequency penalty: subtracted once per occurrence in the window.
    pub frequency_penalty: f32,
    /// PRNG seed. `None` draws a fresh seed from OS entropy at construction time.
    pub seed: Option<u64>,
    /// Number of top log-probabilities to report per token (`0` = none). Stored for the server;
    /// the sampler itself exposes [`crate::Sampler::top_logprobs`].
    pub n_probs: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: None,
            n_probs: 0,
        }
    }
}

impl SamplingParams {
    /// Greedy decoding: pick the argmax instead of drawing from the distribution.
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    /// Whether any penalty would change a logit.
    pub fn penalties_active(&self) -> bool {
        self.repeat_last_n > 0
            && (self.repeat_penalty != 1.0
                || self.frequency_penalty != 0.0
                || self.presence_penalty != 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_spec() {
        let p = SamplingParams::default();
        assert_eq!(p.temperature, 1.0);
        assert_eq!(p.top_k, 40);
        assert_eq!(p.top_p, 0.95);
        assert_eq!(p.min_p, 0.05);
        assert_eq!(p.repeat_penalty, 1.0);
        assert_eq!(p.repeat_last_n, 64);
        assert_eq!(p.presence_penalty, 0.0);
        assert_eq!(p.frequency_penalty, 0.0);
        assert_eq!(p.seed, None);
        assert_eq!(p.n_probs, 0);
        assert!(!p.is_greedy());
        assert!(!p.penalties_active());
    }

    #[test]
    fn serde_round_trip_with_partial_input() {
        let p: SamplingParams = serde_json::from_str(r#"{"temperature":0.0,"seed":7}"#).unwrap();
        assert!(p.is_greedy());
        assert_eq!(p.seed, Some(7));
        assert_eq!(p.top_k, 40);
        let s = serde_json::to_string(&p).unwrap();
        let back: SamplingParams = serde_json::from_str(&s).unwrap();
        assert_eq!(back, p);
    }
}
