# Milestones

Status legend: ✅ done in repo · 🔬 needs real-hardware validation · ⏭ later phase

## Phase 1 vertical slices

| Slice | Content | Status |
|---|---|---|
| A | Workspace skeleton, CLI, typed config (defaults < file < env < flags), structured logging without prompt content | ✅ |
| B | `doctor` hardware/backends report; model manifest, catalog, checksum-verified `pull`, `add`, `remove`, `info`, `fit` | ✅ |
| C | llama.cpp adapter + smoke test | ⚠ launch + flag acceptance + failure path verified on Metal; generation with a standard GGUF pending · 🔬 CUDA / CPU-only Linux |
| D | OpenAI-compatible streaming API, cancellation, `/healthz`, `/v1/models`, `/metrics` | ✅ |
| E | Memory estimate + admission policy, LRU single-model lifecycle | ✅ |
| F | MLX-LM adapter, benchmark harness, JSON + Markdown report | ✅ validated on M4 Max (see benchmarks/results) |
| G | Docs, CI workflow, license/advisory config (`deny.toml`), MLX venv script | ✅ · installers/packaging ⏭ |

## Phase 2 (not started)

- Cached micro-autotuning keyed by hardware fingerprint + backend build + model hash + profile.
- vLLM / SGLang adapter for concurrent NVIDIA serving.
- Queue fairness, per-client limits, prefix-reuse measurement.
- Unix-socket IPC contract v2 for llama.cpp.
- Signed release artifacts, Homebrew tap, Linux packages.
