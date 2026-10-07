//! Per-layer KV-cache layout of a model, read from its GGUF header or MLX `config.json`.
//!
//! A layout is recorded only when it differs from "every layer full attention" and every part of
//! it can be computed: unknown sliding-window patterns or recurrent states return no layout, and
//! the planner then counts every layer as full attention (an overestimate, never an
//! underestimate). Checked against what the engines allocate (8k context, 1 slot):
//! - llama.cpp: Qwen3.5 9B KV 256 MiB + state 50.25 MiB; Gemma 4 12B 128 + 480 MiB;
//!   gpt-oss-20b 192 + 18 MiB.
//! - MLX: Qwen3.8 27B 64 KiB per token + 146.8 MiB state.

use crate::gguf::{GgufMetadata, Value};
use crate::manifest::KvGroup;
use std::collections::BTreeMap;

/// `(groups, fixed state bytes per sequence)`; empty groups = no layout recorded.
pub type Layout = (Vec<KvGroup>, u64);

const NONE: Layout = (Vec::new(), 0);

/// What one layer keeps.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Layer {
    /// KV cache: heads, head size (mean of K and V), sliding window.
    Attention(u32, u32, Option<u32>),
    /// Fixed-size recurrent / linear-attention state.
    Recurrent,
}

/// Group attention layers by shape; `NONE` when the result is plain full attention.
fn finish(layers: &[Layer], state: u64) -> Layout {
    let mut groups: BTreeMap<(Option<u32>, u32, u32), u32> = BTreeMap::new();
    for l in layers {
        if let Layer::Attention(h, d, w) = l {
            *groups.entry((*w, *h, *d)).or_default() += 1;
        }
    }
    let plain = state == 0
        && groups.len() == 1
        && groups.keys().all(|(w, _, _)| w.is_none())
        && !layers.contains(&Layer::Recurrent);
    if plain || groups.is_empty() {
        return NONE;
    }
    let groups = groups
        .into_iter()
        .map(|((window, n_kv_heads, head_dim), layers)| KvGroup {
            layers,
            n_kv_heads,
            head_dim,
            window,
        })
        .collect();
    (groups, state)
}

fn per_layer(md: &GgufMetadata, arch: &str, key: &str, n: usize) -> Option<Vec<i64>> {
    match md.get(&format!("{arch}.{key}"))? {
        Value::Array {
            numbers: Some(v), ..
        } if v.len() == n => Some(v.clone()),
        v => v.as_u64().map(|x| vec![x as i64; n]),
    }
}

/// Layout from a GGUF header (`n_layers`, `default_heads`, `default_dim` from the shape).
pub fn from_gguf(md: &GgufMetadata, n_layers: u32, default_heads: u32, default_dim: u32) -> Layout {
    let Some(arch) = md.architecture() else {
        return NONE;
    };
    let n = n_layers as usize;
    let u = |k: &str| md.arch_u64(k).map(|v| v as u32);
    let heads = per_layer(md, arch, "attention.head_count_kv", n)
        .unwrap_or_else(|| vec![default_heads as i64; n]);
    let mean = |k: Option<u32>, v: Option<u32>| match (k, v) {
        (Some(k), Some(v)) => (k + v).div_ceil(2),
        (Some(k), None) | (None, Some(k)) => k,
        (None, None) => default_dim,
    };
    let dim = mean(u("attention.key_length"), u("attention.value_length"));
    let dim_swa = match (
        u("attention.key_length_swa"),
        u("attention.value_length_swa"),
    ) {
        (None, None) => dim,
        (k, v) => mean(k, v),
    };
    let window = u("attention.sliding_window").filter(|w| *w > 0);
    let pattern = per_layer(md, arch, "attention.sliding_window_pattern", n);
    let interval = u("full_attention_interval").filter(|i| *i > 0);

    let mut layers = Vec::with_capacity(n);
    for (i, &h) in heads.iter().enumerate() {
        let attention = interval.is_none_or(|iv| (i as u32 + 1).is_multiple_of(iv)) && h > 0;
        if !attention {
            layers.push(Layer::Recurrent);
            continue;
        }
        let swa = match (window, &pattern) {
            (None, _) => false,
            (Some(_), Some(p)) => p[i] != 0,
            // llama.cpp hard-codes gpt-oss's pattern (`set_swa_pattern(2)`): even layers are
            // sliding. Measured: 12 of 24 layers, 768 cells.
            (Some(_), None) if arch == "gpt-oss" => i % 2 == 0,
            // A window without a known pattern: stay conservative.
            (Some(_), None) => return NONE,
        };
        layers.push(if swa {
            Layer::Attention(h as u32, dim_swa, window)
        } else {
            Layer::Attention(h as u32, dim, None)
        });
    }

    let recurrent = layers.iter().filter(|l| **l == Layer::Recurrent).count() as u64;
    let state = if recurrent == 0 {
        0
    } else {
        // Mamba-2 style state as llama.cpp stores it (f32): S = inner × state per layer, conv =
        // (kernel − 1) × (inner + 2 × groups × state). Qwen3.5 9B: 24 layers → 50.25 MiB.
        let (Some(inner), Some(st), Some(kernel)) = (
            md.arch_u64("ssm.inner_size"),
            md.arch_u64("ssm.state_size"),
            md.arch_u64("ssm.conv_kernel"),
        ) else {
            return NONE;
        };
        let groups = md.arch_u64("ssm.group_count").unwrap_or(1);
        recurrent * 4 * (inner * st + kernel.saturating_sub(1) * (inner + 2 * groups * st))
    };
    finish(&layers, state)
}

/// Layout from an MLX `config.json` (the text config for multimodal models).
pub fn from_config(tc: &serde_json::Value) -> Layout {
    let num = |k: &str| tc.get(k).and_then(serde_json::Value::as_u64);
    let Some(n) = num("num_hidden_layers") else {
        return NONE;
    };
    let types: Vec<String> = match tc.get("layer_types").and_then(|v| v.as_array()) {
        Some(a) if a.len() as u64 == n => a
            .iter()
            .map(|t| t.as_str().unwrap_or("").to_string())
            .collect(),
        Some(_) => return NONE,
        None => match num("full_attention_interval").filter(|i| *i > 0) {
            Some(iv) => (0..n)
                .map(|i| {
                    if (i + 1).is_multiple_of(iv) {
                        "full_attention".into()
                    } else {
                        "linear_attention".into()
                    }
                })
                .collect(),
            None => return NONE,
        },
    };
    let Some(heads) = num("num_key_value_heads").or(num("num_attention_heads")) else {
        return NONE;
    };
    let dim = num("head_dim").or_else(|| Some(num("hidden_size")? / num("num_attention_heads")?));
    let Some(dim) = dim else {
        return NONE;
    };
    let g_heads = num("num_global_key_value_heads").unwrap_or(heads);
    let g_dim = num("global_head_dim").unwrap_or(dim);
    let window = num("sliding_window").filter(|w| *w > 0);

    let mut layers = Vec::with_capacity(n as usize);
    for t in &types {
        layers.push(match t.as_str() {
            "full_attention" => Layer::Attention(g_heads as u32, g_dim as u32, None),
            "sliding_attention" => match window {
                Some(w) => Layer::Attention(heads as u32, dim as u32, Some(w as u32)),
                None => return NONE,
            },
            "linear_attention" => Layer::Recurrent,
            _ => return NONE,
        });
    }
    let recurrent = layers.iter().filter(|l| **l == Layer::Recurrent).count() as u64;
    let state = if recurrent == 0 {
        0
    } else {
        // Gated DeltaNet as mlx-lm keeps it: f32 state value-heads × key-dim × value-dim, and a
        // bf16 conv state (kernel − 1) × (2 × key-heads × key-dim + value-heads × value-dim).
        // Qwen3.8 27B: 48 layers → 146.8 MiB (measured).
        let (Some(vh), Some(kh), Some(kd), Some(vd), Some(kernel)) = (
            num("linear_num_value_heads"),
            num("linear_num_key_heads"),
            num("linear_key_head_dim"),
            num("linear_value_head_dim"),
            num("linear_conv_kernel_dim"),
        ) else {
            return NONE;
        };
        recurrent * (vh * kd * vd * 4 + kernel.saturating_sub(1) * (2 * kh * kd + vh * vd) * 2)
    };
    finish(&layers, state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn md(arch: &str, kv: &[(&str, Value)]) -> GgufMetadata {
        let mut m = GgufMetadata::default();
        m.kv.insert("general.architecture".into(), Value::Str(arch.into()));
        for (k, v) in kv {
            m.kv.insert(format!("{arch}.{k}"), v.clone());
        }
        m
    }
    fn arr(v: Vec<i64>) -> Value {
        Value::Array {
            len: v.len() as u64,
            numbers: Some(v),
        }
    }
    fn g(layers: u32, n_kv_heads: u32, head_dim: u32, window: Option<u32>) -> KvGroup {
        KvGroup {
            layers,
            n_kv_heads,
            head_dim,
            window,
        }
    }

    #[test]
    fn qwen35_gguf_full_every_fourth_layer_plus_state() {
        let m = md(
            "qwen35",
            &[
                ("attention.head_count_kv", Value::Uint(4)),
                ("attention.key_length", Value::Uint(256)),
                ("attention.value_length", Value::Uint(256)),
                ("full_attention_interval", Value::Uint(4)),
                ("ssm.conv_kernel", Value::Uint(4)),
                ("ssm.state_size", Value::Uint(128)),
                ("ssm.group_count", Value::Uint(16)),
                ("ssm.inner_size", Value::Uint(4096)),
            ],
        );
        assert_eq!(
            from_gguf(&m, 32, 4, 256),
            (vec![g(8, 4, 256, None)], 52_690_944)
        );
    }

    #[test]
    fn gemma4_gguf_pattern_and_per_layer_heads() {
        let pattern: Vec<i64> = (0..48).map(|i| (i % 6 != 5) as i64).collect();
        let heads: Vec<i64> = (0..48).map(|i| if i % 6 == 5 { 1 } else { 8 }).collect();
        let m = md(
            "gemma4",
            &[
                ("attention.head_count_kv", arr(heads)),
                ("attention.key_length", Value::Uint(512)),
                ("attention.value_length", Value::Uint(512)),
                ("attention.key_length_swa", Value::Uint(256)),
                ("attention.value_length_swa", Value::Uint(256)),
                ("attention.sliding_window", Value::Uint(1024)),
                ("attention.sliding_window_pattern", arr(pattern)),
            ],
        );
        assert_eq!(
            from_gguf(&m, 48, 8, 512),
            (vec![g(8, 1, 512, None), g(40, 8, 256, Some(1024))], 0)
        );
    }

    #[test]
    fn gpt_oss_gguf_uses_llama_cpp_pattern() {
        let m = md(
            "gpt-oss",
            &[
                ("attention.head_count_kv", Value::Uint(8)),
                ("attention.key_length", Value::Uint(64)),
                ("attention.value_length", Value::Uint(64)),
                ("attention.sliding_window", Value::Uint(128)),
            ],
        );
        assert_eq!(
            from_gguf(&m, 24, 8, 64),
            (vec![g(12, 8, 64, None), g(12, 8, 64, Some(128))], 0)
        );
    }

    #[test]
    fn unknown_or_plain_gguf_layouts_record_nothing() {
        // Plain full attention.
        let plain = md("qwen3", &[("attention.head_count_kv", Value::Uint(8))]);
        assert_eq!(from_gguf(&plain, 36, 8, 128), NONE);
        // A window without a known pattern.
        let swa = md("olmo2", &[("attention.sliding_window", Value::Uint(4096))]);
        assert_eq!(from_gguf(&swa, 32, 32, 128), NONE);
        // Layers without KV but no way to size their state (LFM2-style short convolutions).
        let conv = md(
            "lfm2",
            &[("attention.head_count_kv", arr(vec![0, 0, 8, 0, 0, 8]))],
        );
        assert_eq!(from_gguf(&conv, 6, 8, 64), NONE);
    }

    #[test]
    fn mlx_configs() {
        let qwen = json!({"num_hidden_layers": 64, "full_attention_interval": 4,
            "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
            "linear_num_value_heads": 48, "linear_num_key_heads": 16, "linear_key_head_dim": 128,
            "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4});
        let (groups, state) = from_config(&qwen);
        assert_eq!(groups, vec![g(16, 4, 256, None)]);
        assert_eq!(state, 153_944_064, "146.8 MiB, as measured on Qwen3.8 27B");

        let types: Vec<&str> = (0..48)
            .map(|i| {
                if i % 6 == 5 {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            })
            .collect();
        let gemma = json!({"num_hidden_layers": 48, "layer_types": types,
            "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256,
            "num_global_key_value_heads": 1, "global_head_dim": 512, "sliding_window": 1024});
        assert_eq!(
            from_config(&gemma),
            (vec![g(8, 1, 512, None), g(40, 8, 256, Some(1024))], 0)
        );

        let plain = json!({"num_hidden_layers": 36, "num_attention_heads": 32,
            "num_key_value_heads": 8, "head_dim": 128});
        assert_eq!(from_config(&plain), NONE);
        let odd = json!({"num_hidden_layers": 2, "layer_types": ["conv", "full_attention"],
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8});
        assert_eq!(from_config(&odd), NONE, "unknown layer type: conservative");
    }
}
