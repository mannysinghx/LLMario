# llmario-engine-metal

Metal GPU backend of the native engine (Architecture §7.4, milestone M2). It implements
`llmario_engine_model::ModelBackend` for the dense GQA families (`llama`, `mistral3`, `qwen2`,
`qwen3`, `smollm3`), the routed mixture-of-experts family (`qwen3moe`), Gemma 4 (`gemma4`:
per-layer head geometry, sliding-window and K=V global layers, GeGLU, post-norms, logit
soft-capping) and the Qwen3.5 hybrid (`qwen35`: Gated DeltaNet layers with 128-wide heads and a
4-tap conv, gated attention), and is selected by `--device auto|metal` in `llmario-engine serve`
and `raw-run`. Other geometries are refused with `MetalError::Unsupported`; with `--device auto`
the server logs the reason and loads the CPU backend instead. The server also keeps a plan that
streams MoE experts from disk on the CPU, because this backend keeps every weight resident.

Open-source-only: the crate talks to the OS Metal framework through `objc2-metal` /
`objc2-foundation` / `objc2` (Zlib OR Apache-2.0 OR MIT). Kernels are MSL source embedded in the
binary and compiled at runtime with `newLibraryWithSource:options:error:` (Metal language version
3.0 set explicitly), so building never needs Xcode. On non-macOS targets the crate is a stub whose
`is_available()` is `false`.

## Design

| Piece | How |
|---|---|
| Weights | Zero-copy. Each GGUF part's mapping is wrapped with `newBufferWithBytesNoCopy` in `MTLResourceStorageModeShared` (page-aligned base, length rounded up to the page, capped at `maxBufferLength`; a larger mapping gets overlapping page-aligned views sized so every tensor lies inside one view, as ggml-metal does). Tensors are addressed as (view, byte offset) found by pointer arithmetic from the `QMat` slices the model crate already resolved. Nothing is copied. |
| Norm weights and biases | One small shared buffer, uploaded once at load. |
| Recurrent state | Qwen3.5: one buffer per sequence holding every DeltaNet layer's conv history and fp32 state, created (zeroed) on the sequence's first token and freed when it is cleared; kernels find it through a table of GPU addresses. It cannot be rewound, so any cut resets the sequence. Snapshots carry it after the KV blocks. |
| Window layers | Gemma 4's sliding-window layers keep their rows in a second pool of 32-position blocks with its own address table (absolute block numbers). A block is released once no future query can see it, so the window layers hold at most `window + batch` positions and a short conversation only what it wrote (a ring buffer would be charged in full at creation). Snapshots carry the paged blocks plus the window blocks the next query can see. |
| KV cache | Paged (`src/kv.rs`): blocks of 32 positions holding every layer's K and V rows (f16 or q8_0), each a shared `MTLBuffer` created when a sequence first reaches it and released when no sequence uses it, so GPU memory follows the cached tokens (Metal charges a buffer's whole size at creation; the old up-front cache held 3.9 GiB for Qwen3-1.7B at 32K). Kernels find blocks through a table of 64-bit GPU addresses, one row per sequence (Metal 3). Block ids, reference counts, the LRU and copy-on-write follow the CPU cache. Blocks are kept resident by a dynamic residency set on macOS 15+, otherwise declared on each encoder. A sequence's blocks can be written to and read from a file (the server's disk tier). |
| Activations | A fixed set of shared buffers sized from the model shape and `n_batch` (x, h, q, k, v, attn, gate, up, logits, token ids, attention partials). No per-token allocation. Logits are read back as f32 after the command buffer completes. |
| Execution | One command buffer per `forward`, one compute encoder with `MTLDispatchTypeConcurrent`; the backend places `memoryBarrierWithScope(Buffers)` between dependent stages so independent kernels (Q/K/V projections) overlap. The inference thread is the only thread that touches Metal objects; the device, queue and pipelines are process-wide and compiled once. |
| Residency | Metal allows 32 residency sets per command queue and aborts past that, so the device counts them and stops at 28: a backend created past the limit binds weights and scratch directly (resident anyway) and declares its KV buffers on each encoder. On macOS 15+ the weight views and the scratch buffers go into two `MTLResidencySet`s (KV blocks into a third, changed as blocks come and go) (`commit`, `requestResidency`, attached to the queue) so the wired collector does not unwire them after idle; released on drop. The keep-alive heartbeat thread from ggml is not implemented yet. |
| Budget | `recommendedMaxWorkingSetSize`, `hasUnifiedMemory`, the GPU family and residency support are exposed through `MetalBackend::device_info()` (printed by `probe`). No system setting is changed. |

The ledger rows for the Metal device: `DeviceId::Gpu(0)` carries `kv_arena_reserved` and
`scratch_reserved`; the weights stay `weights_mapped` on the host row (unified memory).

## Kernels (`src/shaders/*.metal`)

| Kernel | Purpose |
|---|---|
| `embed_{f32,f16,q4_0,q8_0,q4_k,q5_k,q6_k}` | Embedding gather: dequantise one row per token (16 elements per thread). |
| `gemv_<type>`, `gemv_acc_<type>`, `gemv_glu_<type>` | Decode GEMV `y = W x`, `y += W x` (residual fused into the output projections) and `y = silu(W1 x)·(W2 x)` (gate/up fused). Threadgroup = 2 simdgroups × 2 rows; per-type lane mappings read whole blocks with contiguous aligned loads (Q4_K/Q5_K: four super-blocks in flight, 16 bytes per lane; Q6_K: two super-blocks, 16 elements per lane); `simd_sum` per row. |
| `gemm_<type>` | Prefill GEMM `Y = X Wᵀ`: 64-row × 32-token × 32-deep tiles, dequantised to f16 in threadgroup memory, `simdgroup_matrix` 8×8 multiply-accumulate (f16 inputs, f32 accumulation), transposed store into the token-major layout, optional accumulate. |
| `rms_norm` | RMSNorm, 256 threads per row (two-level `simd_sum`). |
| `qk_rope_kv_{f16,q8_0}` | Fused per-head QK-norm + RoPE (normal / NeoX, partial rotary dim, `freq_scale`, `attn_factor`, NoPE layers) + K/V store into the row's sequence and position in the paged cache; one simdgroup per (token, head). |
| `attn_vec_hd{32,64,128,256}_{f16,q8_0}`, `attn_vec_generic_{f16,q8_0}` | Decode attention: one threadgroup per (query, head, split), each query reading its own sequence's blocks; each simdgroup walks a contiguous key range eight keys at a time with an fp32 online softmax; GQA head mapping; split-K across threadgroups for long contexts with `attn_reduce` merging the partials. |
| `attn_prefill_hd{64,128}_{f16,q8_0}` | Prefill attention with simdgroup matrices, one dispatch per prompt in the batch: 32 queries per threadgroup, 8 per simdgroup; `S = Q·Kᵀ` from the cache (q8_0 tiles dequantised cooperatively into threadgroup memory), causal mask, fp32 online softmax, `O = diag(corr)·O + P·V`. Used from 8 tokens; other head dims use the per-query kernel. |
| `swiglu`, `add`, `add_bias` | Element-wise (the GEMM path still uses `swiglu`; biases for Qwen2). |
| `attn_vec_hd512_*`, window mask | Gemma 4's 512-wide global heads; every decode attention kernel takes a window (`t − p < window`) and the K=V layers store the raw K projection (normalised without a gain) as V. |
| `rms_norm_add`, `geglu`, `gemv_geglu_<type>`, `scale_inplace`, `softcap` | Gemma 4: post-norm + residual in one pass, GeGLU with ggml-cpu's fp16 GELU, embedding and layer-output scales, final logit soft-capping. |
| `gdn_conv_silu_k4`, `gdn_scan_128`, `attn_gate` | Qwen3.5: causal depthwise conv + SiLU with each sequence's history (one thread per channel and sequence); the delta rule with L2-normalised q/k, gates and the gated RMSNorm fused, one threadgroup of 512 threads per (V head, sequence) keeping the 128×128 state in registers; `sigmoid(gate)` on the gated attention output (Q and its gate interleaved per head in one projection). |
| `moe_route` | Routed MoE (`qwen3moe`): one simdgroup per row computes the router softmax and picks the top-k experts in descending order (ties to the lower id), weights renormalised like the CPU path. |
| `gemv_id_<type>`, `gemv_glu_id_<type>` | Expert matvecs for up to 4 rows: one grid row per (row, expert) pair, the expert's matrix found at `expert · expert_bytes` in the 3-D tensor; gate and up fused with SwiGLU. |
| `moe_group`, `gemm_id_<type>` | Prompts: `moe_group` lists each expert's pairs (threadgroup atomics), then one GEMM grid per expert multiplies its matrix once over the gathered rows and scatters the outputs. |
| `moe_combine` | `x += Σ_j w_j · out_j` in selection order. |

Block layouts are written from `crates/engine/core/src/dequant.rs`, the scalar reference every
kernel is tested against.

## Tests

```
cargo test -p llmario-engine-metal
```

All tests skip (pass) without a usable Metal device (GPU family < Apple7 or a paravirtual
device). Without a model they still cover: shader compilation, every GEMV/GEMM/embedding kernel
of every supported type against the scalar dequantizer (edge tiles in both dimensions), a tiny
synthetic llama model against `CpuBackend` (prefill, decode, prefill-vs-token-by-token, truncate),
and the `Unsupported` refusal path.

With a real model (read-only; Qwen3-1.7B Q4_K_M is what the maintainers use):

```
LLMARIO_TEST_GGUF=/path/to/Qwen3-1.7B-Q4_K_M.gguf cargo test --release -p llmario-engine-metal
```

`real_model_matches_cpu` prefills 8 tokens then decodes 8 steps and requires max |Δlogit| ≤ 0.05
and identical argmax at every step against the CPU backend's exact f32 kernels
(`LLMARIO_CPU_KERNELS=scalar`, set by the test: the default int8-activation CPU kernels differ
from their own f32 oracle by ~0.7 in logits on this model). `real_model_prefill_consistency`
checks a 40-token prefill (GEMM + simdgroup-matrix attention) against the same tokens one at a time
(GEMV + decode attention).

## Benchmark and profile

```
# ours: 512-token prefill, 64 decode steps (raw token ids, greedy)
TOKS=$(python3 -c "print(','.join(str((i*7919+13)%150000) for i in range(512)))")
llmario-engine raw-run --model Qwen3-1.7B-Q4_K_M.gguf --tokens "$TOKS" --n 65 --ctx 1024 --device metal

# llama.cpp (build 11146)
llama-bench -m Qwen3-1.7B-Q4_K_M.gguf -ngl 99 -p 512 -n 64

# per-kernel GPU time of one decode step and one 512-token prefill
LLMARIO_TEST_GGUF=... cargo test --release -p llmario-engine-metal --test profile -- --nocapture
```

Measured 2026-10-09 on an Apple M4 Max (40-core GPU, 64 GB, macOS 27.0.1), Qwen3-1.7B Q4_K_M,
three back-to-back runs each:

| | decode tok/s | prefill tok/s |
|---|---|---|
| llmario-engine `--device metal` | 205–207 | 3289–3301 (first run after load 2870) |
| llama-bench build 11146 (`-ngl 99`) | 213 ± 10 | 4043 ± 21 |
| ratio | 0.97× | 0.81× |

Decode kernel profile (GPU time, one step at position ~513): GEMV 70 %, attention 15 %,
fused QK-norm/RoPE/KV 5 %, norms 4 %. Prefill (512 tokens): GEMM 95 %, attention 3 %.
