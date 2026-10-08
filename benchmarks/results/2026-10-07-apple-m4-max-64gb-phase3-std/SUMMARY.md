# Baseline: Apple M4 Max · 64 GB

Date 2026-10-08T02:19Z · load average 4.6 on 16 cores at start · concurrency 1 · 3 run(s) per prompt

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
| qwen3.5-9b-gguf-q4km via llmario/llamacpp (latency profile) | 2719 ms | 7095 ms | 64.6 | 24.5 | 7.22 GiB | 7.71 GiB | 3/3 | [md](beta/20261008T022106Z-qwen3.5-9b-gguf-q4km.md) |
| gemma-4-12b-gguf-q4_0 via llmario/llamacpp (latency profile) | 4367 ms | 10603 ms | 41.4 | 15.1 | 8.91 GiB | 10.16 GiB | 3/3 | [md](beta/20261008T022325Z-gemma-4-12b-gguf-q4_0.md) |
| gpt-oss-20b-gguf-mxfp4 via llmario/llamacpp (latency profile) | 907 ms | 2602 ms | 114.4 | 54.3 | 11.98 GiB | 14.04 GiB | 3/3 | [md](beta/20261008T022437Z-gpt-oss-20b-gguf-mxfp4.md) |
| qwen3-8b-mlx-4bit via llmario/mlx (latency profile) | 2198 ms | 5926 ms | 79.5 | 29.4 | 7.05 GiB | 10.11 GiB | 3/3 | [md](beta/20261008T022604Z-qwen3-8b-mlx-4bit.md) |


## Read before comparing

The reference run for Phase 3: today's defaults (`memory_profile = "standard"`, `kv_cache_type =
"f16"`), on the M4 Max development machine, not 16 GB hardware. The Phase 3 settings were measured
right after this run on the same models:
[2026-10-07-apple-m4-max-64gb-phase3-small](../2026-10-07-apple-m4-max-64gb-phase3-small/SUMMARY.md).
Both runs used scratch home folders with a copy of the beta registry, plus Qwen3 8B (MLX) read from
an external drive.
