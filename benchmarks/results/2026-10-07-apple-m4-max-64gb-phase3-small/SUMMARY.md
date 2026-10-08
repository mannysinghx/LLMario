# Baseline: Apple M4 Max · 64 GB

Date 2026-10-08T02:26Z · load average 4.8 on 16 cores at start · concurrency 1 · 3 run(s) per prompt

| Edition | Model | Result |
|---|---|---|
| beta | `qwen3.5-9b-gguf-q4km` | measured |
| beta | `gemma-4-12b-gguf-q4_0` | measured |
| beta | `gpt-oss-20b-gguf-mxfp4` | measured |
| beta | `qwen3-8b-mlx-4bit` | measured |

## beta
### Concurrency 1

| Target | TTFT p50 | TTFT p95 | Decode tok/s / req | Aggregate tok/s | Peak footprint | Estimate | Quality | Report |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3.5-9b-gguf-q4km via llmario/llamacpp (latency profile) | 2743 ms | 6976 ms | 62.5 | 24.2 | 6.11 GiB | 6.85 GiB | 3/3 | [md](beta/20261008T022752Z-qwen3.5-9b-gguf-q4km.md) |
| gemma-4-12b-gguf-q4_0 via llmario/llamacpp (latency profile) | 3671 ms | 9800 ms | 44.8 | 17.3 | 7.45 GiB | 8.69 GiB | 3/3 | [md](beta/20261008T023048Z-gemma-4-12b-gguf-q4_0.md) |
| gpt-oss-20b-gguf-mxfp4 via llmario/llamacpp (latency profile) | 865 ms | 2380 ms | 101.6 | 53.2 | 11.67 GiB | 13.18 GiB | 3/3 | [md](beta/20261008T023301Z-gpt-oss-20b-gguf-mxfp4.md) |
| qwen3-8b-mlx-4bit via llmario/mlx (latency profile) | 2404 ms | 6519 ms | 79.8 | 27.8 | 6.63 GiB | 8.48 GiB | 3/3 | [md](beta/20261008T023430Z-qwen3-8b-mlx-4bit.md) |


## Read before comparing

The Phase 3 settings, `memory_profile = "small"` and `kv_cache_type = "q8_0"`, on the M4 Max
development machine, not 16 GB hardware. Measured right after the standard run
([2026-10-07-apple-m4-max-64gb-phase3-std](../2026-10-07-apple-m4-max-64gb-phase3-std/SUMMARY.md)).

| Model | Estimate ÷ peak | Peak, standard → small | Decode tok/s | TTFT p50 | Quality |
|---|---:|---|---|---|---|
| Qwen3.5 9B (GGUF) | 1.12 | 7.22 → 6.11 GiB | 64.6 → 62.5 | 2.72 → 2.74 s | 3/3 both |
| Gemma 4 12B (GGUF) | 1.17 | 8.91 → 7.45 GiB | 41.4 → 44.8 | 4.37 → 3.67 s | 3/3 both |
| gpt-oss-20b (GGUF) | 1.13 | 11.98 → 11.67 GiB | 114.4 → 101.6 | 0.91 → 0.86 s | 3/3 both |
| Qwen3 8B (MLX) | 1.28 | 7.05 → 6.63 GiB | 79.5 → 79.8 | 2.20 → 2.40 s | 3/3 both |

- **Memory:** the estimate is at or above the measured peak for every model, in both runs.
- **Quality:** the checks are unchanged with the 8-bit KV cache.
- **Speed is not settled by these runs.**
  - Gemma 4 12B and gpt-oss-20b started at load 6.0–7.5, below the busy threshold (8) but not
    quiet, and the two runs differ in load.
  - The standard Gemma 4 12B run is internally inconsistent: its mean (41.4) is below every
    prompt's median.
  - Re-measure speed on a quiet machine before drawing conclusions about the 8-bit KV cache.
- **MLX** has no KV-quantization option, so its change comes only from the smaller reserves (1
  prompt-cache entry, 512 MiB buffer cache).
