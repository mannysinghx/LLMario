# Milestones

Status legend: ✅ done in repo · 🔬 needs real-hardware validation · ⏭ later phase

## Phase 1 vertical slices

| Slice | Content | Status |
|---|---|---|
| A | Workspace skeleton, CLI, typed config (defaults < file < env < flags), structured logging without prompt content | ✅ |
| B | `doctor` hardware/backends report; model manifest, catalog, checksum-verified `pull`, `add`, `remove`, `info`, `fit` | ✅ |
| C | llama.cpp adapter + smoke test | ✅ validated on Apple Silicon Metal (Qwen3-1.7B Q4_K_M) · 🔬 CUDA / CPU-only Linux |
| D | OpenAI-compatible streaming API, cancellation, `/healthz`, `/v1/models`, `/metrics` | ✅ |
| E | Memory estimate + admission policy, LRU single-model lifecycle | ✅ |
| F | MLX-LM adapter, benchmark harness, JSON + Markdown report | ✅ validated on M4 Max (see benchmarks/results) |
| G | Docs, CI workflow, license/advisory config (`deny.toml`), MLX venv script | ✅ · installers/packaging ⏭ |

## Desktop app (added on request, 2026-09-29)

| Item | Status |
|---|---|
| Tauri 2 window: model picker with fit status, lazy load, unload | ✅ |
| Streaming chat with thinking, stats, Stop (cancels engine generation) | ✅ runtime client e2e-tested; used live with Qwen3-8B MLX |
| Download catalog models with progress (HF-cache reuse) | ✅ used live |
| Settings (profile, context, max tokens, temperature, system prompt, history) | ✅ |
| Safe Markdown rendering (escape-first) | ✅ `markdown.test.mjs` |
| Signed/notarized DMG, auto-update, Windows/Linux builds | ⏭ |

## Phase 2 (not started)

- Cached micro-autotuning keyed by hardware fingerprint + backend build + model hash + profile.
- vLLM / SGLang adapter for concurrent NVIDIA serving.
- Queue fairness, per-client limits, prefix-reuse measurement.
- Unix-socket IPC contract v2 for llama.cpp.
- Signed release artifacts, Homebrew tap, Linux packages.
