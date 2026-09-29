# Ops log

Actions with side effects outside the working tree.
Format: **timestamp · what · why · how to undo · verified?**

- 2026-09-28 22:25 · `brew install llama.cpp` (build 11146, commit 7fe450e19) · llama.cpp backend for the llama.cpp adapter · `brew uninstall llama.cpp` · verified: `llama-server --version` prints build 11146
- 2026-09-28 22:55 · created `~/.llmario/` (config home: registry, models, logs, run, bench) via `llmario doctor` · product state dir · `rm -rf ~/.llmario` (also removes pulled models) · verified: `llmario config` lists paths
- 2026-09-28 23:00 · `llmario model pull qwen3-1.7b-mlx-4bit` → `~/.llmario/models/qwen3-1.7b-mlx-4bit` (938 MiB, hard-linked/copied from `~/.cache/huggingface` blobs after SHA verification; 0 bytes downloaded; Hub API metadata call only) · model for validation · `llmario model remove qwen3-1.7b-mlx-4bit` (HF cache untouched) · verified: all 9 files matched Hub checksums
- 2026-09-28 23:40 · first e2e test run hung (fixture Drop deadlock) and was killed; it left 9 `target/debug/llmario mock-engine` processes, which were identified by command line and killed · test fixture fixed · n/a · verified: `pgrep -fl "llmario mock-engine"` empty
- 2026-09-29 05:50 · temporary `llmario model add` of Ollama blob `sha256-ed12a467…` (qwen3-vl 8b) as `ollama-qwen3-vl-8b`, read-only, then `model remove` (unregister only) · llama.cpp adapter smoke test · n/a · verified: blob still present, 6,140,392,576 bytes
- 2026-09-29 05:55 · MLX experiments with scratchpad LLMARIO_HOME (copied registry; config extra_args) · root-cause footprint gap · delete scratchpad dir · verified: `~/.llmario/config.toml` never written
