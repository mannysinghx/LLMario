# Phase 6: speed-planner calibration on the M4 Max (64 GB, 546 GB/s published)

Suite `benchmarks/suites/decode.toml` (one prose prompt, 256 tokens), unique prompts, one request at
a time, 3 runs, `--no-quality`, default settings, quiet machine. "Reads" is the planner's bytes read
per generated token (active weights plus the cache at a 1,024-token working context).

| Engine | Model | Reads per token | Decode tok/s | Role | Predicted | Error |
|---|---|---:|---:|---|---:|---:|
| llama.cpp | Qwen3 1.7B Q4_K_M | 1.14 GiB | 140.2 | calibration | 140.2 | — |
| llama.cpp | Gemma 4 12B Q4_0 | 6.81 GiB | 51.2 | calibration | 51.2 | — |
| llama.cpp | Qwen3.5 0.8B Q4_0 | 0.54 GiB | 268.6 | held out | 171.1 | −36% |
| llama.cpp | Qwen3.5 9B Q4_K_M | 4.83 GiB | 61.1 | held out | 65.8 | +7.7% |
| llama.cpp | Qwen3.5 9B Q4_K_M MTP | 5.00 GiB | 63.4 | held out | 64.2 | +1.3% |
| llama.cpp | gpt-oss-20b MXFP4 (MoE) | 2.44 GiB | 105.6 | held out | 100.2 | −5.1% |
| MLX | Qwen3 1.7B 4-bit | 1.01 GiB | 253.2 | calibration | 253.2 | — |
| MLX | Qwen3.8 27B 4-bit | 13.63 GiB | 26.3 | calibration | 26.3 | — |
| MLX | Llama 3.2 3B 4-bit | 1.79 GiB | 162.2 | held out | 165.0 | +1.7% |
| MLX | Qwen3 8B 4-bit | 4.11 GiB | 87.3 | held out | 81.2 | −7.0% |

Model: time per token = bytes read ÷ (546 GB/s × efficiency) + overhead. Fitted on the calibration
pair per engine (chosen before the runs): llama.cpp efficiency 0.90, overhead 4.65 ms; MLX 0.73,
1.22 ms.

- **The reference-set models are within ±8%**, inside the plan's ±20%.
- **Below ~1B parameters the prediction is conservative:** Qwen3.5 0.8B runs faster than the fixed
  overhead fitted on larger models allows.
- Models were read from an external SSD; that affects loading, not decode speed.
