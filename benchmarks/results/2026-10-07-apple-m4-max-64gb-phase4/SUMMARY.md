# Phase 4: speculative decoding on the M4 Max (64 GB)

Suite `benchmarks/suites/speculative.toml` (prose, code edit, quoting; default quality checks),
unique prompts (cold cache), one request at a time, 3 runs per prompt, temperature 0. Each mode ran
in its own scratch home with a copy of the beta registry; `off/` is the baseline for every model.
Machine load at the start of each run: 3.3–5.0. Development machine, not 16 GB hardware.

| Mode | Model | Prose | Code edit | Quoting | Overall | Drafts accepted | Quality (off / on) | Estimate ÷ peak |
|---|---|---|---|---|---|---|---|---|
| n-gram | Qwen3.5 9B | +1% | −2% | −4% | −2% | 19% | 3/3 / 3/3 | 1.09 |
| n-gram | Gemma 4 12B | −8% | −10% | −2% | −8% | 17% | 3/3 / 3/3 | 1.07 |
| n-gram | gpt-oss-20b | +8% | +28% | +225% | **+87%** | 44% | 3/3 / 3/3 | 1.19 |
| MTP, 1 token | Qwen3.5 9B MTP | **+11%** | **+19%** | **+19%** | **+16%** | 93% | 3/3 / 3/3 | 1.09 |
| draft model | Qwen3.5 9B + Qwen3.5 0.8B | −47% | −33% | −23% | −34% | 76% | 3/3 / 3/3 | 1.10 |
| MLX draft | Qwen3 8B + Qwen3 1.7B | +17% | +46% | +40% | +31% | – | 3/3 / **2/3** | 1.58 |

Baseline decode (tok/s): Qwen3.5 9B 62.1, Gemma 4 12B 49.5, gpt-oss-20b 102.5, Qwen3.5 9B MTP 61.9,
Qwen3 8B (MLX) 83.1.

## Read before comparing

- **Defaults follow the plan's rule** (a mode that slows a workload, or changes answers, is off by
  default): `speculative = "auto"` is the default and uses only MTP, for models with MTP layers.
  n-gram, llama.cpp draft models and MLX draft models stay opt-in.
- **Thinking was on** for every model here (the suite does not turn it off). Reasoning text rarely
  repeats the prompt, which is why n-gram did little for Qwen3.5 9B and Gemma 4 12B. With thinking
  off, an earlier paired check gave n-gram +10% (Qwen3.5 9B), +61% (Gemma 4 12B) and +25%
  (gpt-oss-20b) on the code edit.
- **MTP needs 1 drafted token per step.** At llama.cpp's default of 3, MTP slowed prose by 41% on
  Qwen3.5 9B (a hybrid model keeps a copy of its recurrent state per drafted token, and rejected
  guesses cost a rollback). LLMario uses 1 by default (`draft_tokens`).
- **The MLX draft model changed an answer**: with it, Qwen3 8B spent its 1,024-token budget reasoning
  on the long-retrieval question instead of answering (5903 without the draft). MLX's speculative
  path is not output-identical here.
- **A llama.cpp draft model slows Qwen3.5 9B** even when most drafts are accepted: both models are
  hybrid (recurrent) and roll back on every rejection.
- **Memory:** every estimate stayed at or above the measured peak with speculation on (ratios
  1.07–1.58); the planner adds the draft model, MTP state copies and MTP cache.
