use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// On-disk model format. Determines which backends can serve a model.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum ModelFormat {
    /// Single-file GGUF (llama.cpp).
    Gguf,
    /// Hugging Face style directory with MLX-loadable safetensors (MLX-LM).
    Mlx,
    /// Synthetic model served by the in-tree mock engine (tests / CI only).
    Mock,
}

impl fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ModelFormat::Gguf => "gguf",
            ModelFormat::Mlx => "mlx",
            ModelFormat::Mock => "mock",
        })
    }
}

impl FromStr for ModelFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "gguf" => Ok(Self::Gguf),
            "mlx" => Ok(Self::Mlx),
            "mock" => Ok(Self::Mock),
            other => Err(format!(
                "unknown model format '{other}' (expected gguf|mlx|mock)"
            )),
        }
    }
}

/// Inference engine family.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    LlamaCpp,
    Mlx,
    /// LLMario's own engine (`llmario-engine`), GGUF models.
    Native,
    Mock,
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BackendKind::LlamaCpp => "llamacpp",
            BackendKind::Mlx => "mlx",
            BackendKind::Native => "native",
            BackendKind::Mock => "mock",
        })
    }
}

impl FromStr for BackendKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().replace(['.', '-', '_'], "").as_str() {
            "llamacpp" | "llama" => Ok(Self::LlamaCpp),
            "mlx" | "mlxlm" => Ok(Self::Mlx),
            "native" | "llmario" | "engine" => Ok(Self::Native),
            "mock" => Ok(Self::Mock),
            other => Err(format!(
                "unknown backend '{other}' (expected llamacpp|mlx|native|mock)"
            )),
        }
    }
}

/// Workload profile. Each resolves to explicit values (see [`ResolvedProfile`]).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "lowercase")]
pub enum ProfileKind {
    /// One interactive user: one slot, smallest KV cache, lowest TTFT.
    #[default]
    Latency,
    /// A few concurrent requests (e.g. an agent plus an editor).
    Balanced,
    /// Many concurrent requests: more slots, larger batches, higher p95 latency.
    Throughput,
}

impl fmt::Display for ProfileKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ProfileKind::Latency => "latency",
            ProfileKind::Balanced => "balanced",
            ProfileKind::Throughput => "throughput",
        })
    }
}

impl FromStr for ProfileKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "latency" | "latency-first" => Ok(Self::Latency),
            "balanced" => Ok(Self::Balanced),
            "throughput" | "throughput-first" => Ok(Self::Throughput),
            other => Err(format!(
                "unknown profile '{other}' (expected latency|balanced|throughput)"
            )),
        }
    }
}

/// Concrete, printable runtime values for a profile. Adapters translate these into flags
/// their backend actually supports; anything a backend cannot honour is reported, not faked.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct ResolvedProfile {
    pub kind: ProfileKind,
    /// Concurrent generations the engine will run (llama.cpp slots / MLX decode batch).
    pub parallel: u32,
    /// Context tokens available to each concurrent request.
    pub ctx_per_slot: u32,
    /// Logical prompt batch size (llama.cpp `-b`).
    pub batch: u32,
    /// Physical micro-batch size (llama.cpp `-ub`).
    pub ubatch: u32,
    /// Prompts processed concurrently during prefill (MLX `--prompt-concurrency`).
    pub prompt_concurrency: u32,
    /// Prompt-prefix cache entries kept for reuse (MLX `--prompt-cache-size`).
    pub prompt_cache_entries: u32,
    /// Default max output tokens when a request does not set one.
    pub default_max_tokens: u32,
}

impl ResolvedProfile {
    /// Resolve a profile. `ctx_override` replaces the per-slot context (the user-visible knob).
    pub fn resolve(kind: ProfileKind, ctx_override: Option<u32>) -> Self {
        let mut p = match kind {
            ProfileKind::Latency => Self {
                kind,
                parallel: 1,
                ctx_per_slot: 8192,
                batch: 2048,
                ubatch: 512,
                prompt_concurrency: 1,
                prompt_cache_entries: 2,
                default_max_tokens: 1024,
            },
            ProfileKind::Balanced => Self {
                kind,
                parallel: 4,
                ctx_per_slot: 8192,
                batch: 2048,
                ubatch: 512,
                prompt_concurrency: 1,
                prompt_cache_entries: 4,
                default_max_tokens: 1024,
            },
            ProfileKind::Throughput => Self {
                kind,
                parallel: 16,
                ctx_per_slot: 4096,
                batch: 4096,
                ubatch: 1024,
                prompt_concurrency: 4,
                prompt_cache_entries: 8,
                default_max_tokens: 512,
            },
        };
        if let Some(ctx) = ctx_override {
            p.ctx_per_slot = ctx.max(256);
        }
        p
    }

    /// Total KV-cache context the engine must allocate for all slots.
    pub fn total_ctx(&self) -> u64 {
        self.ctx_per_slot as u64 * self.parallel as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_round_trip() {
        for k in [
            ProfileKind::Latency,
            ProfileKind::Balanced,
            ProfileKind::Throughput,
        ] {
            assert_eq!(k.to_string().parse::<ProfileKind>().unwrap(), k);
        }
        for b in [
            BackendKind::LlamaCpp,
            BackendKind::Mlx,
            BackendKind::Native,
            BackendKind::Mock,
        ] {
            assert_eq!(b.to_string().parse::<BackendKind>().unwrap(), b);
        }
        assert_eq!(
            "llama.cpp".parse::<BackendKind>().unwrap(),
            BackendKind::LlamaCpp
        );
        assert!("cuda".parse::<BackendKind>().is_err());
    }

    #[test]
    fn profile_ctx_override_and_totals() {
        let p = ResolvedProfile::resolve(ProfileKind::Balanced, Some(2048));
        assert_eq!(p.ctx_per_slot, 2048);
        assert_eq!(p.total_ctx(), 2048 * 4);
        let tiny = ResolvedProfile::resolve(ProfileKind::Latency, Some(10));
        assert_eq!(
            tiny.ctx_per_slot, 256,
            "context is clamped to a usable minimum"
        );
    }
}
