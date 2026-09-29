# Benchmark plan

## Rules

1. Same model revision, quantization, prompt, output length, context, sampling, concurrency,
   device, power mode, and software versions on both sides of any comparison.
2. Prefill and decode are reported separately; cold start is reported separately from warm runs.
3. Every result file embeds the full environment manifest (hardware fingerprint, OS, backend
   name + version, model id + file SHA-256, profile values, llmario version).
4. Neutral and negative results are kept. The report never hides a run.
5. Absolute speed numbers are **not** CI gates (too noisy); CI only checks that the harness runs.

## Metrics (per request, aggregated per concurrency level)

| Metric | How it is measured |
|---|---|
| Cold start | Wall time from engine spawn to first successful warm-up token |
| TTFT | Client-side: request sent → first content (or reasoning) delta |
| Prefill rate | `prompt_tokens / TTFT` (usage from `stream_options.include_usage`) |
| Decode rate | `(completion_tokens − 1) / (t_last − t_first)` |
| E2E latency p50/p95 | Request sent → stream end |
| Aggregate throughput | Σ completion tokens / wall time of the level |
| Peak memory | Engine process physical footprint (macOS `proc_pid_rusage`, Linux `VmRSS`), sampled every 100 ms |
| Quality | Fixed checks on deterministic (temperature 0) answers, incl. a long-context retrieval check |

## Default suite (`benchmarks/suites/default.toml`)

- `short` — ~40-token prompt, 128 output tokens.
- `medium` — ~500-token prompt, 128 output tokens.
- `long` — ~3k-token generated record table + retrieval question (quality-checked).
- Arithmetic and instruction-following quality checks.

## Comparisons

- `llmario bench --model M` — through the llmario gateway.
- `llmario bench --url http://127.0.0.1:11434/v1 --remote-model qwen3:1.7b` — any
  OpenAI-compatible baseline (Ollama, raw llama-server, LM Studio…), same suite, same report.
- Claims must name device, model, quantization, workload, baseline, and method.
