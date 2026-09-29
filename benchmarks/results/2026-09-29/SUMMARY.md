# Cross-runtime comparison — 2026-09-29

**Machine:** Apple M4 Max (16 CPU / 40 GPU cores), 64 GB unified memory, macOS 27.0.1, AC power.
**Model:** Qwen3-1.7B, 4-bit. **Suite:** `default` v1 (short 55 / medium 1.8k / long 4.7k-token
prompts, 128 output tokens), 3 runs per prompt, 1 warm-up, temperature 0, no seed, **cold prefix
cache** (unique prefix per request). Quality checks use the exact suite prompts. Runs were
sequential; nothing else was generating. Each linked report embeds the full environment.

| Target | Engine | Model file | Context | Parallel slots |
|---|---|---|---:|---:|
| llmario → llama.cpp | llama.cpp build 11146 (7fe450e19), Homebrew | unsloth Q4_K_M, sha256 b139949c… | 8,192 (latency) / 40,960 / 4×8,192 (balanced) | 1 / 1 / 4 |
| raw llama-server | same binary and flags as llmario, no gateway | same file | 8,192 | 1 |
| Ollama 0.34.2 | its bundled llama-server 0.4.1-dev (391fac164), `-ub 2048`, `--chat-template chatml` | Ollama `qwen3:1.7b` Q4_K_M (different conversion) | 40,960 (default) | 1 (default) |
| llmario → MLX | mlx-lm 0.31.3 / mlx 0.32.2 | mlx-community 4-bit g64 | 8,192 / 4×8,192 | 1 / 4 |

## One request at a time

| Target | TTFT p50 | TTFT p95 | Decode tok/s / req | Aggregate tok/s | Peak footprint | Estimate | Quality | Report |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| llmario → llama.cpp b11146 (latency) | 495 ms | 1455 ms | 217.1 | 102.1 | 2.12 GiB | 3.58 GiB | 2/3 | [md](20260929T132028Z-qwen3-1.7b-gguf-q4km.md) |
| raw llama-server b11146, same flags as llmario (no gateway) | 494 ms | 1409 ms | 214.1 | 102.3 | 2.15 GiB | – | 2/3 | [md](20260929T132049Z-raw.md) |
| Ollama 0.34.2 qwen3:1.7b (defaults) | 482 ms | 1680 ms | 208.2 | 97.6 | 11.66 GiB | – | 2/3 | [md](20260929T132116Z-qwen3_1.7b.md) |
| llmario → MLX-LM 0.31.3 (latency) | 535 ms | 1381 ms | 224.5 | 102.6 | 3.18 GiB | 5.69 GiB | 3/3 | [md](20260929T132136Z-qwen3-1.7b-mlx-4bit.md) |
| llmario → llama.cpp b11146 (latency, --context 40960 = Ollama's context) | 498 ms | 1526 ms | 203.9 | 96.8 | 5.65 GiB | 7.08 GiB | 2/3 | [md](20260929T132221Z-qwen3-1.7b-gguf-q4km.md) |

## Four concurrent requests

| Target | TTFT p50 | TTFT p95 | Decode tok/s / req | Aggregate tok/s | Peak footprint | Estimate | Quality | Report |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| llmario → llama.cpp b11146 (balanced) | 1877 ms | 2596 ms | 70.0 | 113.8 | 4.78 GiB | 6.20 GiB | – | [md](20260929T131832Z-qwen3-1.7b-gguf-q4km.md) |
| llmario → MLX-LM 0.31.3 (balanced) | 1668 ms | 2344 ms | 71.1 | 117.7 | 8.51 GiB | 10.06 GiB | – | [md](20260929T131845Z-qwen3-1.7b-mlx-4bit.md) |
| Ollama 0.34.2 qwen3:1.7b (defaults) | 4880 ms | 6789 ms | 200.5 | 88.1 | 9.42 GiB | – | – | [md](20260929T131900Z-qwen3_1.7b.md) |

## What the data supports

1. **Gateway overhead is not measurable.** llmario → llama.cpp vs the same llama-server without
   llmario: TTFT p50 495 vs 494 ms, decode 217 vs 214 tok/s, peak 2.12 vs 2.15 GiB.
2. **Single-request speed is at parity with Ollama.** TTFT p50 is within ±10% across all targets,
   and run-to-run variation is of that size (Ollama: 568 ms and 482 ms on two runs). MLX decodes
   ~8% faster than Ollama (224.5 vs 208.2 tok/s), below the plan's 15% bar. **No single-request
   speed win is claimed.**
3. **Concurrency 4: llmario's balanced profile beats Ollama *defaults*.** Aggregate throughput
   +29% (llama.cpp, 113.8) and +34% (MLX, 117.7) vs 88.1 tok/s; median TTFT 1.9 / 1.7 s vs 4.9 s.
   Cause: Ollama 0.34.2 starts its engine with `-np 1`, so concurrent requests queue.
   Ollama's `OLLAMA_NUM_PARALLEL` was **not** tested (it needs an Ollama service restart).
4. **Memory.** At equal context (40,960), llmario → llama.cpp peaked at 5.65 GiB vs Ollama's
   6.42 GiB on its first run after load (−12%) and 11.66 GiB after further runs of the same process
   (−52%). Ollama does not pass `--cache-ram`, so llama.cpp's default 8 GiB host prompt cache applies;
   the measured growth (+5.3 GiB) is consistent with that, and llmario caps it at 1 GiB. At its
   default latency profile (8k context), llmario peaks at 2.12 GiB. That saving comes from sizing the
   context to the workload, a policy choice, not a faster engine.
5. **Memory estimates bounded every measured peak** on this machine (estimate / peak = 1.18–1.79x).
6. **Quality:** every llama.cpp-based target, including Ollama, fails the 220-row long-retrieval
   check the same way (the 1,024-token budget is spent reasoning); MLX passes 3/3. The
   quantizations differ (MLX affine 4-bit vs GGUF Q4_K_M), so one check does not rank them.

`superseded-quality-prompt-bug/` holds the first run of the single-request set. Its speed numbers
are valid, but its quality checks used cold-cache prompts (random prefix), so they were not
reproducible (llama.cpp answered "32" instead of "42"). The harness was fixed and the set rerun.

Tables generated with `scripts/bench-table.py benchmarks/results/2026-09-29`.
