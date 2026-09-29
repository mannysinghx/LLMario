# Contributing

## Ground rules

- **No performance claim without a reproducible report.** Attach the `llmario bench` JSON
  (it embeds hardware fingerprint, backend version, model hash, profile). Include neutral and
  negative results.
- **Adapters describe, the supervisor runs.** A new backend implements `EngineAdapter`
  (`crates/supervisor/src/adapter.rs`): probe, formats, `launch` → command line. It must speak
  IPC contract v1 (OpenAI-compatible HTTP on the given loopback port, `/health`,
  `stream_options.include_usage`). Only use documented upstream flags/APIs and record the
  tested version.
- **Never widen what reaches an engine.** The gateway allowlist (`crates/api/src/validate.rs`)
  is a security boundary; new fields need a test.
- **No prompt or completion text in logs.**

## Before a PR

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

The end-to-end tests (`crates/cli/tests/e2e.rs`) spawn the real `llmario mock-engine`
process; they need no GPU or weights. Tests that need real hardware are not part of CI; run
`llmario bench` on the target machine and attach the report.

## Adding a catalog model

Edit `crates/model_registry/catalog.toml`: pin exact files for GGUF, copy the license label
from the model card, and add both an MLX and a GGUF variant where they exist so the family can
be benchmarked across backends.
