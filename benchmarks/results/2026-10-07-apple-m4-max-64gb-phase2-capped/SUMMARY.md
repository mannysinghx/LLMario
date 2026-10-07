# Baseline: Apple M4 Max · 64 GB

Date 2026-10-07T23:41Z · load average 3.4 on 16 cores at start · concurrency 1 · 3 run(s) per prompt

| Edition | Model | Result |
|---|---|---|
| beta | `qwen3.5-9b-gguf-q4km` | measured |
| beta | `gemma-4-12b-gguf-q4_0` | measured |
| beta | `gpt-oss-20b-gguf-mxfp4` | measured |

## beta
### Concurrency 1

| Target | TTFT p50 | TTFT p95 | Decode tok/s / req | Aggregate tok/s | Peak footprint | Estimate | Quality | Report |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3.5-9b-gguf-q4km via llmario/llamacpp (latency profile) | 2777 ms | 7309 ms | 67.5 | 24.6 | 7.01 GiB | 7.71 GiB | 3/3 | [md](beta/20261007T234309Z-qwen3.5-9b-gguf-q4km.md) |
| gemma-4-12b-gguf-q4_0 via llmario/llamacpp (latency profile) | 4906 ms | 14062 ms | 44.3 | 14.1 | 9.37 GiB | 10.16 GiB | 3/3 | [md](beta/20261007T234519Z-gemma-4-12b-gguf-q4_0.md) |
| gpt-oss-20b-gguf-mxfp4 via llmario/llamacpp (latency profile) | 1419 ms | 3587 ms | 104.4 | 43.9 | 12.08 GiB | 14.04 GiB | 3/3 | [md](beta/20261007T234623Z-gpt-oss-20b-gguf-mxfp4.md) |


## Read before comparing

The Phase 2 exit check, run on the M4 Max development machine with per-layer KV accounting, the
corrected memory metric, and `--ctx-checkpoints 2`. The estimate is at or above measured peak for
every model:

| Model | Estimate | Peak | Estimate ÷ peak |
|---|---:|---:|---:|
| Qwen3.5 9B | 7.71 GiB | 7.01 GiB | 1.10 |
| Gemma 4 12B | 10.16 GiB | 9.37 GiB | 1.08 |
| gpt-oss-20b | 14.04 GiB | 12.08 GiB | 1.16 |

Gemma 4 12B in a chat (a long prompt and two follow-up turns) peaked at 9.81 GiB with llama.cpp's
default 32 checkpoints and 7.94 GiB with 2. The follow-up turns reused the prompt equally well.
Decode speeds in this run are not comparable with earlier runs: Gemma 4 12B started at load 6.7.
