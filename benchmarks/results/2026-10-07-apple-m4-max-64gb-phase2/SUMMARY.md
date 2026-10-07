# Baseline: Apple M4 Max · 64 GB

Date 2026-10-07T23:31Z · load average 4.4 on 16 cores at start · concurrency 1 · 3 run(s) per prompt

| Edition | Model | Result |
|---|---|---|
| beta | `qwen3.5-9b-gguf-q4km` | measured |
| beta | `gemma-4-12b-gguf-q4_0` | measured |
| beta | `gpt-oss-20b-gguf-mxfp4` | measured |

## beta
### Concurrency 1

| Target | TTFT p50 | TTFT p95 | Decode tok/s / req | Aggregate tok/s | Peak footprint | Estimate | Quality | Report |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3.5-9b-gguf-q4km via llmario/llamacpp (latency profile) | 2714 ms | 7229 ms | 63.7 | 24.4 | 6.94 GiB | 7.62 GiB | 3/3 | [md](beta/20261007T233254Z-qwen3.5-9b-gguf-q4km.md) |
| gemma-4-12b-gguf-q4_0 via llmario/llamacpp (latency profile) | 4472 ms | 11809 ms | 49.5 | 16.2 | 9.36 GiB | 9.22 GiB | 3/3 | [md](beta/20261007T233509Z-gemma-4-12b-gguf-q4_0.md) |
| gpt-oss-20b-gguf-mxfp4 via llmario/llamacpp (latency profile) | 1162 ms | 2937 ms | 107.2 | 49.3 | 12.04 GiB | 14.01 GiB | 3/3 | [md](beta/20261007T233623Z-gpt-oss-20b-gguf-mxfp4.md) |


## Read before comparing

The validation run for per-layer KV accounting, before llama.cpp's context checkpoints were capped
and counted. Peak memory uses the corrected metric (the larger of footprint and resident size), so
memory-mapped weights are included.

- **Gemma 4 12B: estimate 9.22 GiB, peak 9.36 GiB.** Its estimate was below measured use. This run
  exposed the missing context checkpoints.
- **Fixed in the next run** ([2026-10-07-apple-m4-max-64gb-phase2-capped](../2026-10-07-apple-m4-max-64gb-phase2-capped/SUMMARY.md)).
  `--ctx-checkpoints 2` is now passed to llama.cpp, and checkpoint memory is counted in the estimate.
