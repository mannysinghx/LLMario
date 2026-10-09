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
    /// Qwen3.5 / Qwen3-Next hybrid: Gated DeltaNet layers with one gated full-attention layer
    /// every `full_attention_interval` (`qwen35`, `qwen3next`). See `gdn.rs` and `hybrid.rs`.
    Qwen35,
    /// Gemma 4 (`gemma4`): 5:1 sliding-window / global attention, per-layer head geometry,
    /// K=V global layers, GeGLU, pre+post norms, logit soft-capping. See `gemma4.rs`.
    Gemma4,
}

/// Gemma 4 specifics beyond the dense fields of [`ArchSpec`] (which hold the *global* layers'
/// geometry and RoPE). Keys are the ones llama.cpp's converter writes (`conversion/gemma.py`,
/// `Gemma4Model.set_gguf_parameters`).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Gemma4Spec {
    /// Sliding-window width of the local layers (`attention.sliding_window`): a query at
    /// position `t` sees keys `p` with `t − p < n_swa`.
    pub n_swa: u32,
    /// Per layer, `true` = sliding (local) attention (`attention.sliding_window_pattern`).
    pub swa_layers: Vec<bool>,
    /// Per layer KV head count (`attention.head_count_kv`, an array for this family).
    pub n_kv_head_layers: Vec<u32>,
    /// Head width of the sliding layers (`attention.key_length_swa` / `value_length_swa`); the
    /// global layers use `ArchSpec::head_dim` (`attention.key_length`).
    pub head_dim_swa: u32,
    pub head_dim_v_swa: u32,
    /// RoPE of the sliding layers (`rope.freq_base_swa`, `rope.dimension_count_swa`); the
    /// global layers use `ArchSpec::rope` plus the `rope_freqs.weight` frequency factors.
    pub rope_swa: RopeSpec,
    /// Embedding multiplier, `sqrt(d_model)` in f32 (llama.cpp `build_inp_embd(tok_embd, sqrtf(n_embd))`).
    pub embed_scale: f32,
    /// Score scale of the attention (`f_attention_scale`; 1.0 for Gemma 4, not `1/sqrt(head_dim)`).
    pub attn_scale: f32,
    /// `final_logit_softcapping`: `logits = c · tanh(logits / c)`; 0 = none.
    pub final_logit_softcap: f32,
}

/// Attention geometry of one layer (differs per layer only for Gemma 4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnGeom {
    pub n_head: u32,
    pub n_kv_head: u32,
    pub head_dim: u32,
    pub head_dim_v: u32,
    /// `Some(n_swa)` for a sliding-window layer.
    pub window: Option<u32>,
}

impl AttnGeom {
    pub fn q_dim(&self) -> u32 {
        self.n_head * self.head_dim
    }
    pub fn kv_dim(&self) -> u32 {
        self.n_kv_head * self.head_dim
    }
    pub fn v_dim(&self) -> u32 {
        self.n_kv_head * self.head_dim_v
    }
    pub fn attn_dim(&self) -> u32 {
        self.n_head * self.head_dim_v
    }
}

/// What a layer's token mixer is.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BlockKind {
    /// Softmax attention over a K/V cache (every layer of the dense families).
    Attention,
    /// Gated DeltaNet linear attention over an fp32 recurrent state.
    DeltaNet,
}

/// Gated DeltaNet geometry (GGUF `ssm.*` keys as llama.cpp's converter writes them for this
/// family: `state_size` is the key/value head width, `group_count` the number of QK heads,
/// `time_step_rank` the number of V heads, `inner_size` = V heads × head width).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct GdnSpec {
    /// Causal conv kernel width (`ssm.conv_kernel`, 4).
    pub d_conv: u32,
    /// Key head width (`ssm.state_size`, 128); the value head width is `d_inner / n_v_heads`.
    pub head_k: u32,
    /// QK heads (`ssm.group_count`).
    pub n_k_heads: u32,
    /// V heads (`ssm.time_step_rank`); a multiple of `n_k_heads`.
    pub n_v_heads: u32,
    /// `ssm.inner_size` = `n_v_heads * head_v`.
    pub d_inner: u32,
}

impl GdnSpec {
    pub fn head_v(&self) -> u32 {
        self.d_inner / self.n_v_heads.max(1)
    }
    /// Width of the Q (and K) projection: `n_k_heads * head_k`.
    pub fn key_dim(&self) -> u32 {
        self.n_k_heads * self.head_k
    }
    /// Width of the V projection (and of the gate `z`): `d_inner`.
    pub fn value_dim(&self) -> u32 {
        self.d_inner
    }
    /// Channels of the fused QKV projection and of the causal conv: `2 * key_dim + value_dim`.
    pub fn conv_dim(&self) -> u32 {
        2 * self.key_dim() + self.value_dim()
    }
    /// f32 elements of conv history per layer per sequence.
    pub fn conv_state_len(&self) -> u32 {
        (self.d_conv.saturating_sub(1)) * self.conv_dim()
    }
    /// f32 elements of delta-rule state per layer per sequence.
    pub fn state_len(&self) -> u32 {
        self.n_v_heads * self.head_k * self.head_v()
    }
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
    /// Token mixer of each layer (`n_layer` entries; all `Attention` for the dense families).
    pub blocks: Vec<BlockKind>,
    /// Gated DeltaNet geometry when any layer is `DeltaNet`.
    pub gdn: Option<GdnSpec>,
    /// Attention output gate: `attn_q` is `[d_model → 2·n_head·head_dim]` with `(q, gate)`
    /// interleaved per head and the attention output is multiplied by `sigmoid(gate)` before
    /// `attn_output` (Qwen3.5 / Qwen3-Next).
    pub attn_out_gate: bool,
    /// Gemma 4 sliding-window / per-layer geometry (`Some` only for [`Family::Gemma4`]).
    pub gemma4: Option<Gemma4Spec>,
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
            "qwen35" | "qwen3next" => Family::Qwen35,
            "gemma4" => Family::Gemma4,
            other => return Err(ModelError::UnsupportedArch(other.into())),
        };
        if f.tensor("blk.0.ffn_gate_inp.weight").is_some() {
            // Qwen3-Next 80B-A3B, the Qwen3.5-35B-A3B MoE and Gemma 4 26B-A4B: expert FFNs
            // arrive in M4.
            return Err(ModelError::UnsupportedArch(format!(
                "{arch} with mixture-of-experts FFN"
            )));
        }
        let u = |k: &str| -> Result<u32> {
            f.get_arch_u32(k)
                .ok_or_else(|| ModelError::MissingKey(f.arch_key(k)))
        };
        // `block_count` includes any trailing multi-token-prediction (NextN / MTP) draft blocks
        // (`nextn_predict_layers`; llama.cpp `n_layer_all` vs `n_layer()`); the main pass ignores
        // them, so they are neither loaded nor counted.
        let n_nextn = f.get_arch_u32("nextn_predict_layers").unwrap_or(0);
        let n_layer = u("block_count")?.saturating_sub(n_nextn);
        let d_model = u("embedding_length")?;
        let n_head = u("attention.head_count")?;
        // `attention.head_count_kv` is a scalar for the dense families and a per-layer array for
        // Gemma 4 (llama.cpp `get_key_or_arr`); the scalar view below is the first global layer's
        // value (or the first layer's when the model has no global layer).
        let n_kv_head_layers: Vec<u32> = match f.get_arch_array("attention.head_count_kv") {
            Some(a) if !a.is_empty() => a
                .iter()
                .map(|v| v.as_u64().unwrap_or(n_head as u64) as u32)
                .collect(),
            _ => {
                vec![f.get_arch_u32("attention.head_count_kv").unwrap_or(n_head); n_layer as usize]
            }
        };
        let swa_layers: Vec<bool> = match family {
            Family::Gemma4 => f
                .get_arch_array("attention.sliding_window_pattern")
                .map(|a| {
                    a.iter()
                        .map(|v| v.as_bool().unwrap_or(v.as_u64() == Some(1)))
                        .collect()
                })
                .ok_or_else(|| {
                    ModelError::MissingKey(f.arch_key("attention.sliding_window_pattern"))
                })?,
            _ => Vec::new(),
        };
        let n_kv_head = (0..n_layer as usize)
            .find(|&l| !swa_layers.get(l).copied().unwrap_or(false))
            .or(Some(0))
            .and_then(|l| n_kv_head_layers.get(l).copied())
            .unwrap_or(n_head);
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
            // Qwen3.5 uses llama.cpp's interleaved mRoPE (`LLAMA_ROPE_TYPE_IMROPE`) with
            // `rope.dimension_sections` [11, 11, 10, 0]. For text tokens llama.cpp feeds the
            // position as [p, p, p, 0] (llama-batch.cpp "expand [p] to [p, p, p, 0]") and the
            // sections only ever select the t/h/w angles (sector = (i/2) % 32 → t when sector%3==0,
            // h when %3==1 and sector<33, w when %3==2 and sector<30; the 4th section is 0), so
            // every pair uses the same angle `p · theta^(-2i/n_rot)` and the rotation is exactly
            // NeoX RoPE over the first `n_rot` = 64 dims of the 256-wide head
            // (`ggml_mrope_cache_init` + `rotate_pairs<T>(n_dims, n_dims/2, …)` in ggml-cpu/ops.cpp).
            // Image/video positions (differing per section) are out of scope for text inference.
            Family::Qwen2 | Family::Qwen3 | Family::Qwen35 => RopeKind::Neox,
            // llama.cpp `llama_model_rope_type`: LLM_ARCH_GEMMA4 → LLAMA_ROPE_TYPE_NEOX.
            Family::Gemma4 => RopeKind::Neox,
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
        let (blocks, gdn) = match family {
            Family::Qwen35 => {
                // llama.cpp (src/models/qwen35.cpp `load_arch_hparams`): layer `i` is recurrent
                // unless `(i + 1) % full_attention_interval == 0`, unless the converter wrote an
                // explicit `attention.recurrent_layers` array (Qwen3-Next `layer_types`).
                let blocks: Vec<BlockKind> = match f.get_arch_array("attention.recurrent_layers") {
                    Some(a) if a.len() >= n_layer as usize => a[..n_layer as usize]
                        .iter()
                        .map(|v| {
                            if v.as_bool().unwrap_or(v.as_u64() == Some(1)) {
                                BlockKind::DeltaNet
                            } else {
                                BlockKind::Attention
                            }
                        })
                        .collect(),
                    _ => {
                        let interval = f
                            .get_arch_u32("full_attention_interval")
                            .unwrap_or(4)
                            .max(1);
                        (0..n_layer)
                            .map(|i| {
                                if (i + 1) % interval == 0 {
                                    BlockKind::Attention
                                } else {
                                    BlockKind::DeltaNet
                                }
                            })
                            .collect()
                    }
                };
                let gdn = GdnSpec {
                    d_conv: u("ssm.conv_kernel")?,
                    head_k: u("ssm.state_size")?,
                    n_k_heads: u("ssm.group_count")?,
                    n_v_heads: u("ssm.time_step_rank")?,
                    d_inner: u("ssm.inner_size")?,
                };
                if gdn.n_k_heads == 0
                    || gdn.n_v_heads % gdn.n_k_heads != 0
                    || gdn.d_inner % gdn.n_v_heads != 0
                    || gdn.head_v() != gdn.head_k
                    || gdn.d_conv < 1
                {
                    return Err(ModelError::UnsupportedArch(format!(
                        "{arch}: Gated DeltaNet geometry {gdn:?} (expected head_v == head_k and \
                         n_v_heads a multiple of n_k_heads)"
                    )));
                }
                (blocks, Some(gdn))
            }
            _ => (vec![BlockKind::Attention; n_layer as usize], None),
        };
        let gemma4 = match family {
            Family::Gemma4 => {
                if swa_layers.len() < n_layer as usize || n_kv_head_layers.len() < n_layer as usize
                {
                    return Err(ModelError::UnsupportedArch(format!(
                        "{arch}: sliding_window_pattern / head_count_kv shorter than block_count"
                    )));
                }
                // The E-series' per-layer embeddings (`embedding_length_per_layer_input` > 0) and
                // cross-layer KV sharing (`shared_kv_layers` > 0) are a follow-up; the 12B / 31B
                // carry neither.
                let n_pl = f
                    .get_arch_u32("embedding_length_per_layer_input")
                    .unwrap_or(0);
                let n_shared = f.get_arch_u32("attention.shared_kv_layers").unwrap_or(0);
                if n_pl > 0 || n_shared > 0 {
                    return Err(ModelError::UnsupportedArch(format!(
                        "{arch} with per-layer embeddings ({n_pl}) / shared KV layers ({n_shared}) \
                         (E-series) is not supported yet"
                    )));
                }
                let n_swa = u("attention.sliding_window")?;
                // llama.cpp defaults (llama-model.cpp "head size and n_rot for SWA layers",
                // llama-hparams.h): the SWA head size / n_rot default to the full-attention
                // values, the SWA base to 10000 and the SWA freq scale to 1.
                let head_dim_swa = f
                    .get_arch_u32("attention.key_length_swa")
                    .unwrap_or(head_dim);
                let head_dim_v_swa = f
                    .get_arch_u32("attention.value_length_swa")
                    .unwrap_or(head_dim_v);
                let rope_dim_swa = f
                    .get_arch_u32("rope.dimension_count_swa")
                    .unwrap_or(rope_dim);
                let theta_swa = f.get_arch_f32("rope.freq_base_swa").unwrap_or(10000.0);
                if head_dim != head_dim_v || head_dim_swa != head_dim_v_swa {
                    return Err(ModelError::UnsupportedArch(format!(
                        "{arch}: K/V head widths differ (llama.cpp requires them equal)"
                    )));
                }
                Some(Gemma4Spec {
                    n_swa,
                    swa_layers: swa_layers[..n_layer as usize].to_vec(),
                    n_kv_head_layers: n_kv_head_layers[..n_layer as usize].to_vec(),
                    head_dim_swa,
                    head_dim_v_swa,
                    rope_swa: RopeSpec {
                        kind: RopeKind::Neox,
                        theta: theta_swa,
                        dim: rope_dim_swa,
                        freq_scale: 1.0,
                        attn_factor: 1.0,
                    },
                    embed_scale: (d_model as f32).sqrt(),
                    attn_scale: 1.0,
                    final_logit_softcap: f.get_arch_f32("final_logit_softcapping").unwrap_or(0.0),
                })
            }
            _ => None,
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
            qk_norm: matches!(family, Family::Qwen3 | Family::Gemma4),
            attn_bias: matches!(family, Family::Qwen2),
            nope_layers,
            tied_embeddings: f.tensor("output.weight").is_none(),
            blocks,
            gdn,
            attn_out_gate: matches!(family, Family::Qwen35),
            gemma4,
        })
    }

    /// KV bytes per token per sequence at the given element size (f32 = 4, f16 = 2, q8_0 ≈ 1.06):
    /// the *marginal* per-token cost, i.e. the full-attention layers only. Sliding-window layers
    /// hold a fixed ring of `n_swa + n_batch` positions per sequence ([`Self::window_bytes`]),
    /// and recurrent layers a fixed state (see [`crate::KvCache::recurrent_bytes`]).
    pub fn kv_bytes_per_token(&self, bytes_per_elem: f64) -> u64 {
        let mut total = 0f64;
        for l in 0..self.n_layer as usize {
            let g = self.attn_geom(l);
            if self.blocks[l] != BlockKind::Attention || g.window.is_some() {
                continue;
            }
            total += (g.kv_dim() + g.v_dim()) as f64 * bytes_per_elem;
        }
        total as u64
    }

    /// Fixed per-sequence K/V bytes of the sliding-window layers: each holds a ring of
    /// `min(max_ctx, n_swa + n_batch)` positions (Architecture §8.1 "Window" class).
    pub fn window_bytes(&self, bytes_per_elem: f64, max_ctx: usize, n_batch: usize) -> u64 {
        let mut total = 0f64;
        for l in 0..self.n_layer as usize {
            let g = self.attn_geom(l);
            let Some(w) = g.window else { continue };
            if self.blocks[l] != BlockKind::Attention {
                continue;
            }
            let cap = max_ctx.min(w as usize + n_batch);
            total += (g.kv_dim() + g.v_dim()) as f64 * bytes_per_elem * cap as f64;
        }
        total as u64
    }

    /// Attention geometry of layer `l` (identical for every layer except in Gemma 4, where the
    /// sliding layers have their own head width and KV head count).
    pub fn attn_geom(&self, l: usize) -> AttnGeom {
        match &self.gemma4 {
            Some(g) if g.swa_layers.get(l).copied().unwrap_or(false) => AttnGeom {
                n_head: self.n_head,
                n_kv_head: g.n_kv_head_layers[l],
                head_dim: g.head_dim_swa,
                head_dim_v: g.head_dim_v_swa,
                window: Some(g.n_swa),
            },
            Some(g) => AttnGeom {
                n_head: self.n_head,
                n_kv_head: g.n_kv_head_layers.get(l).copied().unwrap_or(self.n_kv_head),
                head_dim: self.head_dim,
                head_dim_v: self.head_dim_v,
                window: None,
            },
            None => AttnGeom {
                n_head: self.n_head,
                n_kv_head: self.n_kv_head,
                head_dim: self.head_dim,
                head_dim_v: self.head_dim_v,
                window: None,
            },
        }
    }

    /// Whether any attention layer uses a sliding window.
    pub fn has_window_layers(&self) -> bool {
        (0..self.n_layer as usize).any(|l| self.attn_geom(l).window.is_some())
    }

    /// Largest Q / K / V / attention-output widths over the layers (scratch sizing).
    pub fn max_q_dim(&self) -> u32 {
        (0..self.n_layer as usize)
            .map(|l| self.attn_geom(l).q_dim())
            .max()
            .unwrap_or(self.q_dim())
    }
    pub fn max_kv_dim(&self) -> u32 {
        (0..self.n_layer as usize)
            .map(|l| self.attn_geom(l).kv_dim())
            .max()
            .unwrap_or(self.kv_dim())
    }
    pub fn max_v_dim(&self) -> u32 {
        (0..self.n_layer as usize)
            .map(|l| self.attn_geom(l).v_dim())
            .max()
            .unwrap_or(self.v_dim())
    }
    pub fn max_attn_dim(&self) -> u32 {
        (0..self.n_layer as usize)
            .map(|l| self.attn_geom(l).attn_dim())
            .max()
            .unwrap_or(self.n_head * self.head_dim_v)
    }

    /// Layers whose mixer is softmax attention (all of them for the dense families).
    pub fn n_attn_layers(&self) -> u32 {
        self.blocks
            .iter()
            .filter(|b| **b == BlockKind::Attention)
            .count() as u32
    }
    /// Layers whose mixer is Gated DeltaNet.
    pub fn n_recurrent_layers(&self) -> u32 {
        self.blocks
            .iter()
            .filter(|b| **b == BlockKind::DeltaNet)
            .count() as u32
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
