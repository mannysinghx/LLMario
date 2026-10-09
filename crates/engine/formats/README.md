# llmario-engine-formats

Model file readers for the native engine. Both readers memory-map every file read-only, parse the
header once, validate every tensor's byte range before anything is touched, and hand out zero-copy
`TensorInfo` views (`ByteSpan { source, offset, len }`, where `source` is the file index). Nothing
is read from disk until a backend touches a tensor.

| Module | Reads | Status |
|---|---|---|
| `gguf` | GGUF v2/v3, single file or `-00001-of-NNNNN` split set | M1 |
| `safetensors` | Hugging Face / MLX folders (`config.json` + `model.safetensors` or index + shards) | M3 (this crate); model wiring is a follow-up |

## GGUF (`gguf::GgufFile`)

- Header: magic, version 2 or 3, tensor count, metadata KV pairs (all 13 value types, nested
  arrays), tensor infos, padding to `general.alignment` (default 32), then data. Little-endian only.
- Every tensor offset is checked against the part size; a tensor never spans parts; duplicate
  names are refused.
- `tensor(name)`, `tensor_bytes(&TensorInfo)`, `tensor_bytes_total()`, `bytes_by_type()`, typed
  metadata getters (`get_u32`, `get_arch_f32("rope.freq_base")`, ...).
- Block sizes come from `ggml-common.h` via `llmario_engine_core::GgmlType`.
- `gguf::writer::GgufWriter` builds synthetic files for tests and `testkit`.

## Safetensors folders (`safetensors::SafetensorsFolder`)

```rust
let f = SafetensorsFolder::open(Path::new("/models/qwen3-1.7b-mlx-4bit"))?;
f.model_type();            // "qwen3"
f.hidden_size();           // Some(2048)   (falls back to config["text_config"][key])
f.quantization();          // Some(MlxQuantization { default: {bits: 4, group_size: 64, mode: "affine"}, overrides })
let t = f.tensor("model.norm.weight").unwrap();      // BF16, shape [2048]
let v = f.mlx_quant_view("model.layers.0.self_attn.q_proj")?.unwrap();
let mut row = vec![0f32; v.cols as usize];
f.dequantize_mlx_rows(&v, 0, 1, &mut row)?;          // w = scale * q + bias, f32
f.tokenizer_json_path();   // Some(".../tokenizer.json")
f.chat_template();         // chat_template.jinja | chat_template.json | tokenizer_config.json["chat_template"]
```

### File layout and validation

A `.safetensors` file is an 8-byte little-endian `u64` header length `N`, `N` bytes of JSON
(must start with `{`, may be space-padded, at most 100 MB), then the byte buffer. Each entry is
`{"dtype": "BF16", "shape": [2048, 2048], "data_offsets": [begin, end]}` with offsets relative to
the buffer; `__metadata__` is a string map (MLX writes `{"format": "mlx"}`).

The reader refuses a file when: the header length exceeds the file or 100 MB; the JSON is not an
object; a dtype is unknown (sub-byte and FP8 tags included); `begin > end` or `end` is past the
buffer; `end - begin != numel * elem_size`; two non-empty tensors overlap; a name repeats across
shards; `model.safetensors.index.json` names a tensor that is missing or lives in a different
shard than it says.

Not enforced: duplicate keys inside one header (serde_json keeps the last one) and the spec's
"every buffer byte belongs to a tensor" rule (gaps are tolerated).

Dtypes: `F32 F16 BF16 F64 U8 I8 I16 I32 I64 U32`. `U8` and `U32` are engine-private
`GgmlType` variants (ids from `ENGINE_PRIVATE_ID` = 1000 up; `GgmlType::from_id` refuses them so a
GGUF can never carry them).

Shapes are converted to ggml order (`ne[0]` = contiguous dimension): an HF `[out, in]` matrix
becomes `[in, out]`, which is exactly how the same tensor is shaped in a GGUF, so
`model::weights::mat(f, name, d_in, d_out)` style checks carry over unchanged. Scalars become
`[1]`; tensors with more than four dims keep the three innermost and fold the rest into `ne[3]`
(`hf_dims(name)` returns the original dims; the 27B's vision tower has 5-D conv weights).

Which files hold weights: `model.safetensors.index.json` (`weight_map` → shards, sorted; `metadata.total_size` is exposed as `index_total_size`), else `model.safetensors`, else every `*.safetensors` in the folder.

### `config.json` accessors

`model_type`, `architectures`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`,
`num_key_value_heads` (defaults to `num_attention_heads`), `head_dim` (defaults to
`hidden_size / num_attention_heads`), `intermediate_size`, `vocab_size`, `rms_norm_eps`,
`rope_theta` (also `rope_parameters.rope_theta`, transformers v5), `rope_scaling` (the
`rope_scaling` object when non-null, else `rope_parameters`), `tie_word_embeddings`,
`max_position_embeddings`, `bos_token_id`, `eos_token_ids` (config + `generation_config.json`).
Every key is looked up at the top level and then under `text_config` (Qwen3.5 multimodal configs
nest the language model there). `cfg(key)` gives the raw `serde_json::Value`.

## MLX quantisation layout

`mlx.core.quantize(w, group_size, bits, mode="affine")` turns an HF `[rows, cols]` matrix into
three tensors (names are `{module}.weight/.scales/.biases`):

| Tensor | dtype | HF shape | Content |
|---|---|---|---|
| `{module}.weight` | `U32` | `[rows, cols * bits / 32]` | packed `q` |
| `{module}.scales` | model dtype (`BF16` Qwen, `F16` Llama, `F32`) | `[rows, cols / group_size]` | `s` per group |
| `{module}.biases` | same as scales | `[rows, cols / group_size]` | `β` per group |

Dequantisation is `w[i] = s[i / group_size] * q[i] + β[i / group_size]` per row. Packing order
(the MLX docs say "packed in an unsigned 32-bit integer from the lower to upper bits"; the
straddling cases were established by experiment, see below): **the row is one contiguous bit
stream over the little-endian `u32` words, element `i` occupying bits `[i*bits, (i+1)*bits)`.**
For 4 bits a word holds elements `8w..8w+7`, element `8w` in bits 0–3. For 3, 5 and 6 bits
elements straddle words: the 3-bit element 10 has its low two bits in bits 30–31 of word 0 and its
high bit in bit 0 of word 1. `group_size` ∈ {32, 64, 128} and bits ∈ {2, 3, 4, 5, 6, 8}; every
group fills whole words. Bits per weight with f16/bf16 scales: `bits + 32 / group_size`
(4.5 for 4-bit g64).

Worked example (Qwen3-1.7B, 4-bit g64, `q_proj` is `[2048, 2048]`): `weight U32 [2048, 256]`,
`scales BF16 [2048, 32]`, `biases BF16 [2048, 32]`; embeddings and `lm_head` are quantised too.

`config.json` carries `"quantization": {"group_size": 64, "bits": 4, "mode": "affine"}` (and a
duplicate `quantization_config`). Mixed-precision exports (mlx-lm `quantize_model` with a
predicate) add one entry per module path: `false` (left unquantised) or its own
`{"bits", "group_size"}`; `MlxQuantization::params_for(module)` resolves them. An HF
`quantization_config` with `quant_method` (AWQ/GPTQ/compressed-tensors) is never mistaken for MLX;
it is exposed raw by `hf_quantization_config()`. Modes `mxfp4`/`mxfp8`/`nvfp4` are recognised and
refused.

In core: `dequant::mlx_affine_row(bits, group_size, &[u32], scale_dtype, scales, biases, out)`,
`mlx_affine_row_le_bytes` (same over raw bytes — a mapped data region is only byte-aligned) and
`mlx_affine_pack` (the inverse, for tests and synthetic models). Scales/biases are upcast to f32
and the product and sum are computed in f32 with no fused multiply-add, so results are bit-exact
against a numpy f32 reference and within one bf16/f16 ulp of MLX's native-dtype output.

## HF/MLX tensor names → engine (GGUF) names

The engine's `Weights::load` uses the llama.cpp names. Mapping (gguf-py `tensor_mapping.py`,
confirmed against the four local MLX folders), for `model_type` `qwen3`, `qwen2` and `llama`:

| HF / MLX name | Engine name | Notes |
|---|---|---|
| `model.embed_tokens.weight` | `token_embd.weight` | MLX quantises it: `.scales`/`.biases` siblings |
| `model.norm.weight` | `output_norm.weight` | |
| `lm_head.weight` | `output.weight` | absent when `tie_word_embeddings` (Qwen3-1.7B, Llama 3.2 3B): use `token_embd` |
| `model.layers.{i}.input_layernorm.weight` | `blk.{i}.attn_norm.weight` | |
| `model.layers.{i}.post_attention_layernorm.weight` | `blk.{i}.ffn_norm.weight` | |
| `model.layers.{i}.self_attn.q_proj.weight` | `blk.{i}.attn_q.weight` | Llama: see permutation below |
| `model.layers.{i}.self_attn.k_proj.weight` | `blk.{i}.attn_k.weight` | Llama: see permutation below |
| `model.layers.{i}.self_attn.v_proj.weight` | `blk.{i}.attn_v.weight` | |
| `model.layers.{i}.self_attn.o_proj.weight` | `blk.{i}.attn_output.weight` | |
| `model.layers.{i}.self_attn.q_norm.weight` | `blk.{i}.attn_q_norm.weight` | Qwen3 only, `[head_dim]` |
| `model.layers.{i}.self_attn.k_norm.weight` | `blk.{i}.attn_k_norm.weight` | Qwen3 only, `[head_dim]` |
| `model.layers.{i}.self_attn.{q,k,v}_proj.bias` | `blk.{i}.attn_{q,k,v}.bias` | Qwen2 only (`attention_bias`) |
| `model.layers.{i}.mlp.gate_proj.weight` | `blk.{i}.ffn_gate.weight` | |
| `model.layers.{i}.mlp.up_proj.weight` | `blk.{i}.ffn_up.weight` | |
| `model.layers.{i}.mlp.down_proj.weight` | `blk.{i}.ffn_down.weight` | |

Every `*.weight` of a linear or embedding in an MLX folder may come as a `weight/scales/biases`
triplet (`mlx_quant_view(module)` with the module = the name without `.weight`); norms are plain
BF16/F16. Rows of `weight`, `scales` and `biases` correspond one-to-one, so a row permutation of
the HF matrix (below) can be applied to the packed form.

Hyper-parameters: `hidden_size` → `d_model`, `num_hidden_layers` → `n_layer`,
`num_attention_heads` → `n_head`, `num_key_value_heads` → `n_head_kv`, `head_dim`,
`intermediate_size` → `n_ff`, `vocab_size` → `n_vocab`, `rms_norm_eps`, `rope_theta`,
`rope_scaling` (`{"rope_type": "llama3", "factor": 32, "low_freq_factor": 1, "high_freq_factor": 4, "original_max_position_embeddings": 8192}` for Llama 3.2), `tie_word_embeddings`,
`max_position_embeddings` → `n_ctx_train`.

**Llama Q/K permutation.** llama.cpp's converter (`convert_hf_to_gguf.py`, `LlamaModel.permute`)
rewrites `q_proj` and `k_proj` rows so the GGUF Llama graph can use the interleaved ("normal")
RoPE; HF weights expect the half-split ("neox") RoPE. The permutation, per head of
`head_dim` rows: `w.reshape(n_head, 2, head_dim / 2, d_in).swapaxes(1, 2).reshape(w.shape)`
(`n_head` = `num_key_value_heads` for `k_proj`). Reading an HF/MLX Llama folder therefore needs
either that row permutation on `q_proj`/`k_proj` (packed weights, scales and biases alike) or the
neox RoPE for the Llama family. Qwen2/Qwen3 GGUFs are *not* permuted (the converter leaves them in
HF order and the GGUF graph uses neox RoPE), so Qwen folders map one-to-one.

**Qwen3.5 (`qwen3_5` / `qwen3_5_text`, hybrid Gated DeltaNet).** Names observed in the local
27B folder (prefix `language_model.` on every tensor, including `language_model.lm_head.*`):
`model.layers.{i}.linear_attn.{A_log, dt_bias, conv1d.weight [10240,4,1], norm.weight, in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj}` on `linear_attention` layers;
`self_attn.{q_proj [n_head*head_dim*2 rows: Q and the attention gate], k_proj, v_proj, o_proj, q_norm, k_norm}` on `full_attention` layers;
`mlp.{gate,up,down}_proj`; `layer_types` and `full_attention_interval` in `text_config`.
The engine's hybrid names are `blk.{i}.{attn_gate, attn_qkv, ssm_a, ssm_alpha, ssm_beta, ssm_conv1d, ssm_dt.bias, ssm_norm, ssm_out, post_attention_norm}`.
The exact split of `q_proj` into `attn_q`/`attn_gate`, and of `in_proj_qkv`, is **not verified
here** (check llama.cpp's `Qwen3NextModel.modify_tensors` before wiring).

## Tests

`cargo test -p llmario-engine-formats -p llmario-engine-core` is hermetic: synthetic GGUF and
safetensors files (single, sharded with index, malformed headers), MLX views and dequantisation on
synthetic 3/4/8-bit folders with F16/BF16/F32 scales, and in core a table of rows produced by
`mlx.core.quantize` for every bit width (2, 3, 4, 5, 6, 8) that must unpack to known `q`.

Against real models (read-only; nothing is copied):

```sh
# reference values (numpy; mlx optional but used for the cross-check when importable)
python -m venv /tmp/pyref && /tmp/pyref/bin/pip install numpy mlx
/tmp/pyref/bin/python scripts/engine/mlx_ref.py "$MODELS/qwen3-1.7b-mlx-4bit" /tmp/ref/qwen3-1.7b.json
/tmp/pyref/bin/python scripts/engine/mlx_ref.py "$BETA/qwen3.8-27b-mlx-4bit" /tmp/ref/qwen3.8-27b.json   # sharded

# header/index/consistency on the folders, then value-by-value comparison with the JSONs
LLMARIO_TEST_MLX="$MODELS/qwen3-1.7b-mlx-4bit,$BETA/qwen3.8-27b-mlx-4bit" \
LLMARIO_TEST_MLX_REF=/tmp/ref/qwen3-1.7b.json,/tmp/ref/qwen3.8-27b.json \
cargo test -p llmario-engine-formats --test mlx_ref -- --nocapture

# regenerate core's MLX_FIXTURES table
/tmp/pyref/bin/python scripts/engine/mlx_ref.py --fixtures
```

`mlx_ref.py` reads the folder with its own parser (no `safetensors` package), dequantises the
first and last 4 rows of ~12 quantised modules (embeddings, `lm_head`, layers 0 and 3) and three
plain float tensors, and records per module whether `mlx.core.dequantize` agrees (`exact_f32`
with f32-cast scales, and the max difference against MLX's native bf16/f16 output).

Measured on 2026-10-09 (mlx 0.32.3, numpy 2.5.3): Qwen3-1.7B, Qwen3-8B, Llama-3.2-3B (F16
scales) and the sharded Qwen3.5-27B (3 shards, 2180 tensors, 498 quantised modules, 16.05 GB)
all parse and resolve; 57,600–61,440 values per model compared with the reference, all bit-exact;
numpy vs `mlx.core.dequantize` exact on every module with f32 scales and ≤ 4.9e-4 (one bf16 ulp
at the magnitudes involved) against the native bf16 output, ≤ 3.1e-5 against native f16.
The GGUF side is covered the same way by `scripts/engine/dequant_ref.py` and
`tests/dequant_ref.rs` (`LLMARIO_DEQUANT_REF`).
