# Ops log

Actions with side effects outside the working tree.
Format: **timestamp · what · why · how to undo · verified?**

- 2026-09-28 22:25 · `brew install llama.cpp` (build 11146, commit 7fe450e19) · llama.cpp backend for the llama.cpp adapter · `brew uninstall llama.cpp` · verified: `llama-server --version` prints build 11146
- 2026-09-28 22:55 · created `~/.llmario/` (config home: registry, models, logs, run, bench) via `llmario doctor` · product state dir · `rm -rf ~/.llmario` (also removes pulled models) · verified: `llmario config` lists paths
- 2026-09-28 23:00 · `llmario model pull qwen3-1.7b-mlx-4bit` → `~/.llmario/models/qwen3-1.7b-mlx-4bit` (938 MiB, hard-linked/copied from `~/.cache/huggingface` blobs after SHA verification; 0 bytes downloaded; Hub API metadata call only) · model for validation · `llmario model remove qwen3-1.7b-mlx-4bit` (HF cache untouched) · verified: all 9 files matched Hub checksums
- 2026-09-28 23:40 · first e2e test run hung (fixture Drop deadlock) and was killed; it left 9 `target/debug/llmario mock-engine` processes, which were identified by command line and killed · test fixture fixed · n/a · verified: `pgrep -fl "llmario mock-engine"` empty
- 2026-09-29 05:50 · temporary `llmario model add` of Ollama blob `sha256-ed12a467…` (qwen3-vl 8b) as `ollama-qwen3-vl-8b`, read-only, then `model remove` (unregister only) · llama.cpp adapter smoke test · n/a · verified: blob still present, 6,140,392,576 bytes
- 2026-09-29 05:55 · MLX experiments with scratchpad LLMARIO_HOME (copied registry; config extra_args) · root-cause footprint gap · delete scratchpad dir · verified: `~/.llmario/config.toml` never written
- 2026-09-29 06:20 · `llmario model pull qwen3-1.7b-gguf-q4km` (user-approved): downloaded 1.03 GiB from huggingface.co/unsloth/Qwen3-1.7B-GGUF @ d7f544ee to `~/.llmario/models/qwen3-1.7b-gguf-q4km` · llama.cpp validation + cross-backend benchmark · `llmario model remove qwen3-1.7b-gguf-q4km` · verified: SHA-256 b139949c… matched Hub LFS hash before install
- 2026-09-29 06:20 · `ollama pull qwen3:1.7b` (user-approved) into Ollama 0.34.2's store (Ollama service was already running; not restarted) · baseline for comparison · `ollama rm qwen3:1.7b` · verified: `ollama show` → qwen3, Q4_K_M, 40960 ctx
- 2026-09-29 06:25 · benchmark matrix: temporary `llama-server` (Homebrew build) on 127.0.0.1:18999 for the no-gateway baseline, stopped by the script; Ollama service used as-is (read-only process inspection, `ps` sampling of pid 86842; no config or restart) · cross-runtime comparison · n/a · verified: port 18999 free afterwards, no leftover llama-server
- 2026-09-29 06:40 · reran the single-request set after fixing the quality-prompt bug; first set kept in `benchmarks/results/2026-09-29/superseded-quality-prompt-bug/` · reproducible quality checks · n/a · verified: SUMMARY.md tables contain only non-superseded runs
