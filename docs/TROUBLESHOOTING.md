# Troubleshooting

Start with `llmario doctor` — it shows what was detected and why a backend is unavailable.

| Symptom | Cause / fix |
|---|---|
| `✗ llamacpp … llama-server not found` | Install llama.cpp (`brew install llama.cpp`) or set `backends.llamacpp.server_path`. |
| `✗ mlx … mlx-lm not found` | Run `scripts/setup-mlx-venv.sh`, or `pip install mlx-lm`, or set `backends.mlx.python`. |
| `engine_start_failed … key not found in model: …` | The GGUF was written for a different runtime (e.g. Ollama's own layout) or needs a newer llama.cpp. Use a standard GGUF from the catalog or upgrade llama.cpp. The error includes the last engine log lines; full log in `~/.llmario/logs/<backend>-<model>.log`. |
| `insufficient_memory` (507) | The message states weights, KV and budget, and suggests a `--context` that fits. Lower `--context`, use `--profile latency`, pick a smaller quantization, or raise `runtime.memory_limit_gb`/lower `memory_headroom_gb` if you know the machine has room. |
| `server_busy` (503) | All slots of the loaded model are busy, or another model is serving and cannot be evicted. Retry after `Retry-After`, or use `--profile balanced/throughput` for more slots. |
| `context_length_exceeded` | Prompt ≈ bytes/4 tokens exceeds the per-request context. Start with a larger `--context` (costs KV memory: see `llmario model fit`). |
| Concurrency does not help on MLX | `mlx_lm.server` batches only requests without `seed`; llmario drops `seed` for greedy requests, but a `seed` with `temperature > 0` disables batching. Prefill-heavy workloads also gain little (see README results). |
| Reasoning models return empty answers | Thinking can consume the whole `max_tokens`. Raise `max_tokens`, or disable thinking per the model's convention (Qwen3: add `/no_think`). |
| `doctor` lists orphaned engines | An llmario process was killed hard (macOS has no parent-death signal). `kill <pid>` if you do not need it. On Linux engines die with llmario automatically. |
| Host rejected (401) through a proxy | In loopback mode llmario only answers loopback `Host` names. Use `127.0.0.1`/`localhost`, or run with `--allow-remote --api-key …`. |

Logs: gateway logs go to stderr (`-v` for more, `--log-json` for JSON). They never contain
prompt or completion text. `LLMARIO_LOG=debug` enables debug filters.
