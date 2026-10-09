# Native engine design (2026-10-08)

Design material for the LLMario Native Inference Engine: a from-scratch Rust engine that will run under the
existing supervisor and gateway as a third adapter. Nothing here is implemented yet; the existing llama.cpp and
MLX-LM adapters are unchanged.

| Read this | For |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | The technical architecture: goals and targets, memory governance, formats, CPU/GPU/NPU backends, KV cache, scheduling and placement, internet tools and security, API, testing, the milestone plan, risks, open decisions, licensing, appendices |
| [OPUS_5_5_BUILD_PROMPT.md](OPUS_5_5_BUILD_PROMPT.md) | The prompt to give Claude Opus 5.5 in a Claude Code session to build the engine milestone by milestone |
| [reports/Local LLM engine architecture research.md](reports/Local%20LLM%20engine%20architecture%20research.md) | The research synthesis the design is built on (372 sources; every number labelled measured / vendor-claimed / unverified) |
| [research_notes/Local LLM engine architecture research/](research_notes/Local%20LLM%20engine%20architecture%20research/) | The nine underlying research notes with the source links: engine landscape, quantization and formats, KV cache and attention, CPU execution, GPU/NPU backends, sharding and parallelism, OS memory governance, internet tools and agents, target models |

Status: draft v0.1 for review. When accepted, the architecture becomes ADR 0002 and milestone plans go under
`docs/engine/plans/` with a running `docs/engine/STATUS.md`.
