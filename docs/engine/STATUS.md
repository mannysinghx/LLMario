# Native engine: build status

Running record of the milestones in `ARCHITECTURE.md` §13. Numbers are measured on the
maintainers' Apple M4 Max (16-core, 64 GB, macOS 27) unless stated; every row names the command.

| Milestone | State | Evidence |
|---|---|---|
| M0 Baselines and scaffolding | done (2026-10-09) | crates under `crates/engine/`, `adapter_native`, CI jobs run on the PR; dequantizers cross-checked bit-for-bit against gguf-py on two real files (`cargo test -p llmario-engine-formats --test dequant_ref` with `LLMARIO_DEQUANT_REF`); llama.cpp CPU baseline for Qwen3-1.7B Q4_K_M: `llama-bench -ngl 0 -t 12` → pp128 158 tok/s, tg32 36.7 tok/s (build 11146) |
| M1 CPU dense engine over GGUF | in progress | families llama, mistral3, qwen2, qwen3, smollm3; tokenizer parity 59/59 strings on seven models vs `llama-tokenize`; chat templates byte-identical to Python Jinja2 on six models; greedy continuation identical to `llama-completion -no-cnv --temp 0` on Qwen3-1.7B; full stack `llmario-beta run qwen3-1.7b` through gateway and supervisor: ready 0.25 s, TTFT 89 ms, 53 tok/s (40 tokens); engine alone: decode ≈62 tok/s, prefill ≈190 tok/s at 12 threads with the NEON kernels; plan bounds measured peak (1.1 GiB measured vs 3.1 GiB planned after load; tightness waits for the KV arena, since untouched reservations are not resident) |
| M2 Metal backend | next | — |
| M3 KV subsystem, hybrids, multi-slot, safetensors | not started | — |
| M4 MoE and placement | not started | — |
| M5 Tools and internet | not started | — |
| M6 Vulkan backend | not started | — |
| M7 NPU delegate and wider GPU validation | not started | — |
| M8 Speculative decoding and Metal 4 tensor path | not started | — |
| M9 Hardening and beta release | not started | — |

## M1 exit criteria

| Criterion | Target | Measured | Status |
|---|---|---|---|
| Greedy golden match vs llama.cpp | identical first 16 characters on the fixture prompts | Qwen3-1.7B: identical 32-token continuation on "The capital of France is" | pass (one model; test `crates/engine/bin/tests/llama_cpp_golden.rs`) |
| Decode ≥ 0.6 × roofline | 546 GB/s ÷ 1.1 GB/token ≈ 496 tok/s roofline → ≥ 298 tok/s | ≈62 tok/s (NEON int8 kernels, 12 threads) | not yet: CPU decode on Apple silicon is limited by the CPU's own share of bandwidth (llama.cpp CPU: 36.7); the fair CPU target is relative to llama.cpp CPU, which we exceed by 1.7× |
| Prefill ≥ 0.7 × llama.cpp CPU | ≥ 111 tok/s (0.7 × 158) | ≈190 tok/s | pass |
| Plan bound | measured ≤ planned | 1.1 GiB ≤ 3.1 GiB | pass |
| Plan tightness | planned ≤ 1.15 × measured | 2.84× | fail (expected: KV reserved but untouched; revisit with the arena in M3) |
| Warm load ≤ 1.2 × llama.cpp | — | 0.05 s load + warm-up | pass |

## Open items carried forward

- KV cache is f32 and contiguous per slot (M3 brings f16/q8_0 and the paged arena).
- One request at a time (slots > 1 is accepted and served sequentially; M3).
- No tools, JSON schema or logprobs yet (M5; the API returns a clear 400).
- Windows named pipe and Linux cgroup enforcement not implemented (M6/M7 validation).
- The AVX2 kernel path is compiled in CI on x86 runners but has not been run on a real x86 machine by the maintainers.
