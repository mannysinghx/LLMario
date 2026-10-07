# Baseline: Apple M4 Max · 64 GB

Date 2026-10-07T20:45Z · load average 4.6 on 16 cores at start · concurrency 1 · 3 run(s) per prompt

| Edition | Model | Result |
|---|---|---|
| beta | `qwen3-8b-gguf-q4km` | not installed |
| beta | `qwen3-8b-mlx-4bit` | not installed |
| beta | `qwen3.5-9b-gguf-q4km` | measured |
| beta | `qwen3.5-9b-mlx-4bit` | not installed |
| beta | `gemma-4-12b-gguf-q4_0` | measured |
| beta | `gemma-4-12b-mlx-4bit` | not installed |
| beta | `ministral-3-14b-gguf-q4km` | not installed |
| beta | `ministral-3-14b-mlx-4bit` | not installed |
| beta | `gpt-oss-20b-gguf-mxfp4` | measured |
| beta | `gpt-oss-20b-mlx-mxfp4-q8` | not installed |
| production | `qwen3-8b-gguf-q4km` | not installed |
| production | `qwen3-8b-mlx-4bit` | not installed |
| production | `qwen3.5-9b-gguf-q4km` | measured |
| production | `qwen3.5-9b-mlx-4bit` | not installed |
| production | `gemma-4-12b-gguf-q4_0` | measured |
| production | `gemma-4-12b-mlx-4bit` | not installed |
| production | `ministral-3-14b-gguf-q4km` | not installed |
| production | `ministral-3-14b-mlx-4bit` | not installed |
| production | `gpt-oss-20b-gguf-mxfp4` | measured |
| production | `gpt-oss-20b-mlx-mxfp4-q8` | not installed |

## beta
### Concurrency 1

| Target | TTFT p50 | TTFT p95 | Decode tok/s / req | Aggregate tok/s | Peak footprint | Estimate | Quality | Report |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3.5-9b-gguf-q4km via llmario/llamacpp (latency profile) | 2711 ms | 7211 ms | 67.4 | 24.9 | 2.11 GiB | 8.32 GiB | 3/3 | [md](beta/20261007T204719Z-qwen3.5-9b-gguf-q4km.md) |
| gemma-4-12b-gguf-q4_0 via llmario/llamacpp (latency profile) | 4172 ms | 11013 ms | 48.9 | 17.0 | 3.03 GiB | 14.62 GiB | 3/3 | [md](beta/20261007T204930Z-gemma-4-12b-gguf-q4_0.md) |
| gpt-oss-20b-gguf-mxfp4 via llmario/llamacpp (latency profile) | 1267 ms | 3404 ms | 90.5 | 42.4 | 1.03 GiB | 14.18 GiB | 3/3 | [md](beta/20261007T205049Z-gpt-oss-20b-gguf-mxfp4.md) |


## production
### Concurrency 1

| Target | TTFT p50 | TTFT p95 | Decode tok/s / req | Aggregate tok/s | Peak footprint | Estimate | Quality | Report |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3.5-9b-gguf-q4km via llmario/llamacpp (latency profile) | 3191 ms | 8637 ms | 57.0 | 21.1 | 1.96 GiB | 8.32 GiB | 3/3 | [md](production/20261007T205232Z-qwen3.5-9b-gguf-q4km.md) |
| gemma-4-12b-gguf-q4_0 via llmario/llamacpp (latency profile) | 4352 ms | 11947 ms | 47.8 | 16.1 | 3.04 GiB | 14.62 GiB | 3/3 | [md](production/20261007T205431Z-gemma-4-12b-gguf-q4_0.md) |
| gpt-oss-20b-gguf-mxfp4 via llmario/llamacpp (latency profile) | 1238 ms | 3206 ms | 75.5 | 39.1 | 0.88 GiB | 14.18 GiB | 3/3 | [md](production/20261007T205553Z-gpt-oss-20b-gguf-mxfp4.md) |

## Read before comparing

- **Development machine, not 16 GB hardware.** This is the harness's first full run and the
  reference for this M4 Max (64 GB). It is not the Phase 1 16 GB baseline.
- **Model files were read from an external USB SSD (exFAT).** That slows loading, not decoding:
  the weights are in memory once loaded.
- **Beta and production run the same inference code.**
  - The gap between them comes from the harness. Production warms up with one short request; the
    beta warms up every prompt length (Phase 1).
  - Production's first measured prompts are slower. Short prompt: Qwen3.5 9B 51.8 vs 68.3 tok/s,
    gpt-oss-20b 57.1 vs 107.6.
  - Medium and long prompts are within about 2–9%.
  - Read the gap as a measurement difference, not a speed difference.
- **Peak footprint leaves out memory-mapped model weights.** gpt-oss-20b used 0.49 GiB right after
  loading 11.28 GiB of weights. Do not compare peak with the estimate for llama.cpp until this is
  resolved (open item in `docs/PHASES_16GB_AND_SPEED.md`).
- **Effective bandwidth:**
  - Dense models: 341–383 GB/s.
  - gpt-oss-20b: 1,096 GB/s, above the chip's 546 GB/s. An MoE model reads only its active experts
    per token, so this is expected (see the report footnote).
- **Load before Gemma 4 12B was 7.9**, just under the busy threshold of 8. It was raised by the
  previous model's run, because the 1-minute average lags.
- **The planner today, on a 16 GB Mac** (GPU limit 10.67 GiB):
  - Qwen3.5 9B: 8.32 GiB, fits.
  - Gemma 4 12B: 14.62 GiB, refused.
  - gpt-oss-20b: 14.18 GiB, refused.
  - The two refusals are the cases Phase 2 targets.
