//! `ArchSpec`: everything the planner, the cache and the forward pass need to know about a model,
//! parsed from GGUF metadata (the keys llama.cpp's converter writes).

use crate::{ModelError, Result};
use llmario_engine_cpu::RopeKind;
use llmario_engine_formats::GgufFile;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    /// Llama 2/3, Mistral, Ministral, Devstral, Granite-style dense GQA (`llama`, `mistral3`).
    Llama,
    /// Qwen2 / Qwen2.5 (QKV biases, no QK-norm).
    Qwen2,
    /// Qwen3 dense (QK-norm, no biases).
    Qwen3,
    /// SmolLM3 (Llama block with NoPE on every `no_rope_interval`-th layer).
    SmolLm3,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct RopeSpec {
    pub kind: RopeKind,
    pub theta: f32,
    /// Rotated dimensions (≤ head_dim).
    pub dim: u32,
    pub freq_scale: f32,
    pub attn_factor: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ArchSpec {
    pub arch: String,
    pub family: Family,
    pub name: Option<String>,
    pub n_layer: u32,
    pub d_model: u32,
    pub n_head: u32,
    pub n_kv_head: u32,
    pub head_dim: u32,
    pub head_dim_v: u32,
    pub n_ff: u32,
    pub n_vocab: u32,
    pub context_length: u32,
    pub rms_eps: f32,
    pub rope: RopeSpec,
    /// Per-head RMSNorm on Q and K before RoPE (Qwen3, Olmo-style).
    pub qk_norm: bool,
    /// QKV projections carry biases (Qwen2).
    pub attn_bias: bool,
    /// Layers that apply no RoPE at all (SmolLM3 NoPE layers; Llama 4 iRoPE).
    pub nope_layers: Vec<u32>,
    /// Output head is the embedding matrix (no `output.weight`).
    pub tied_embeddings: bool,
}

impl ArchSpec {
    pub fn from_gguf(f: &GgufFile) -> Result<ArchSpec> {
        let arch = f
            .architecture()
            .ok_or_else(|| ModelError::MissingKey("general.architecture".into()))?
            .to_string();
        let family = match arch.as_str() {
            "llama" | "mistral3" => Family::Llama,
            "qwen2" => Family::Qwen2,
            "qwen3" => Family::Qwen3,
            "smollm3" => Family::SmolLm3,
            other => return Err(ModelError::UnsupportedArch(other.into())),
        };
        let u = |k: &str| -> Result<u32> {
            f.get_arch_u32(k)
                .ok_or_else(|| ModelError::MissingKey(f.arch_key(k)))
        };
        let n_layer = u("block_count")?;
        let d_model = u("embedding_length")?;
        let n_head = u("attention.head_count")?;
        let n_kv_head = f.get_arch_u32("attention.head_count_kv").unwrap_or(n_head);
        let head_dim = f
            .get_arch_u32("attention.key_length")
            .unwrap_or(d_model / n_head.max(1));
        let head_dim_v = f.get_arch_u32("attention.value_length").unwrap_or(head_dim);
        let n_ff = u("feed_forward_length")?;
        let n_vocab = f
            .get_arch_u32("vocab_size")
            .or_else(|| f.get_array("tokenizer.ggml.tokens").map(|a| a.len() as u32))
            .or_else(|| f.tensor("token_embd.weight").map(|t| t.shape.rows() as u32))
            .ok_or_else(|| ModelError::MissingKey("vocab size".into()))?;
        let context_length = f.get_arch_u32("context_length").unwrap_or(4096);
        let rms_eps = f
            .get_arch_f32("attention.layer_norm_rms_epsilon")
            .unwrap_or(1e-5);
        let rope_dim = f.get_arch_u32("rope.dimension_count").unwrap_or(head_dim);
        let theta = f.get_arch_f32("rope.freq_base").unwrap_or(10000.0);
        let rope_kind = match family {
            // llama.cpp: LLAMA_ROPE_TYPE_NORM for llama; NEOX for qwen2/qwen3/smollm3? SmolLM3
            // and Llama use the normal (adjacent-pair) layout; Qwen families use NeoX.
            Family::Llama | Family::SmolLm3 => RopeKind::Normal,
            Family::Qwen2 | Family::Qwen3 => RopeKind::Neox,
        };
        let scaling_type = f.get_arch_str("rope.scaling.type").unwrap_or("none");
        let factor = f.get_arch_f32("rope.scaling.factor").unwrap_or(1.0);
        let freq_scale = match scaling_type {
            "linear" if factor > 0.0 => 1.0 / factor,
            _ => 1.0,
        };
        let nope_layers = match family {
            Family::SmolLm3 => {
                // The converter may write `smollm3.no_rope_layers` (per layer, 1 = use RoPE);
                // when absent, llama.cpp's graph uses NoPE on every fourth layer (il+1 % 4 == 0).
                f.get_arch_array("no_rope_layers")
                    .map(|a| {
                        a.iter()
                            .enumerate()
                            .filter(|(_, v)| v.as_u64() == Some(0))
                            .map(|(i, _)| i as u32)
                            .collect()
                    })
                    .unwrap_or_else(|| (0..n_layer).filter(|l| (l + 1) % 4 == 0).collect())
            }
            _ => vec![],
        };
        Ok(ArchSpec {
            name: f.get_str("general.name").map(|s| s.to_string()),
            arch,
            family,
            n_layer,
            d_model,
            n_head,
            n_kv_head,
            head_dim,
            head_dim_v,
            n_ff,
            n_vocab,
            context_length,
            rms_eps,
            rope: RopeSpec {
                kind: rope_kind,
                theta,
                dim: rope_dim,
                freq_scale,
                attn_factor: 1.0,
            },
            qk_norm: matches!(family, Family::Qwen3),
            attn_bias: matches!(family, Family::Qwen2),
            nope_layers,
            tied_embeddings: f.tensor("output.weight").is_none(),
        })
    }

    /// KV bytes per token per sequence at the given element size (f32 = 4, f16 = 2, q8_0 ≈ 1.06).
    pub fn kv_bytes_per_token(&self, bytes_per_elem: f64) -> u64 {
        let per_layer = (self.n_kv_head as u64) * (self.head_dim as u64 + self.head_dim_v as u64);
        (per_layer as f64 * bytes_per_elem * self.n_layer as f64) as u64
    }

    pub fn q_dim(&self) -> u32 {
        self.n_head * self.head_dim
    }
    pub fn kv_dim(&self) -> u32 {
        self.n_kv_head * self.head_dim
    }
    pub fn v_dim(&self) -> u32 {
        self.n_kv_head * self.head_dim_v
    }
}
