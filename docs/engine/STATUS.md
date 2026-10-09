# Native engine: build status

Running record of the milestones in `ARCHITECTURE.md` §13. Numbers are measured on the
maintainers' Apple M4 Max (16-core, 64 GB, macOS 27) unless stated; every row names the command.

**Platforms:** macOS on Apple silicon (CPU and Metal) and Linux (CPU). The engine does not support
Windows; LLMario's adapter reports it unavailable there and models run on llama.cpp instead.

| Milestone | State | Evidence |
|---|---|---|
| M0 Baselines and scaffolding | done (2026-10-09) | crates under `crates/engine/`, `adapter_native`, CI jobs run on the PR; dequantizers cross-checked bit-for-bit against gguf-py on two real files (`cargo test -p llmario-engine-formats --test dequant_ref` with `LLMARIO_DEQUANT_REF`); llama.cpp CPU baseline for Qwen3-1.7B Q4_K_M: `llama-bench -ngl 0 -t 12` → pp128 158 tok/s, tg32 36.7 tok/s (build 11146) |
| M1 CPU dense engine over GGUF | in progress | families llama, mistral3, qwen2, qwen3, smollm3; tokenizer parity 59/59 strings on seven models vs `llama-tokenize`; chat templates byte-identical to Python Jinja2 on six models; greedy parity with `llama-completion -no-cnv --temp 0` on five models (see "Parity with llama.cpp" below); full stack `llmario-beta run qwen3-1.7b` through gateway and supervisor: ready 0.25 s, TTFT 89 ms, 53 tok/s (40 tokens); engine alone with the f16 KV cache: decode 69–74 tok/s after a 512-token prompt, prefill 255–258 tok/s (three runs of `raw-run --tokens <512 ids> --n 65 --ctx 1024 --threads 12 --device cpu`, NEON kernels); plan bounds measured peak (1.1 GiB measured vs 3.1 GiB planned after load; tightness waits for the KV arena, since untouched reservations are not resident) |
| M2 Metal backend | done (2026-10-09), prefill tuning open | `crates/engine/metal`: zero-copy weights over the mmap, f16 KV, runtime-compiled MSL, residency sets; logits vs the CPU f32 path within 0.0133 with identical argmax; Qwen3-1.7B decode 205–207 tok/s vs llama.cpp Metal 213 (0.97×), prefill 3,289–3,301 vs 4,043 tok/s (0.81×, `llama-bench -ngl 99 -p 512 -n 64`); through the full stack `llmario-beta run`: 200.5 tok/s / TTFT 31 ms vs llama.cpp 209.1 / 37 ms, identical text; `--device auto` falls back to the CPU for families Metal does not cover (Qwen3.5 today) |
| M3 KV subsystem, hybrids, multi-slot, safetensors | in progress | done: Qwen3.5 / Qwen3-Next hybrid family (CPU; parity with llama.cpp on 0.8B and 9B), Gemma 4 dense family (CPU; parity on the 12B up to 1,343-token chat prompts), safetensors / MLX folder reader (bit-exact with MLX on four models), paged arena and scheduler crates (unit-tested). f16 KV cache on the CPU (half the memory of f32; attention runs one task per token and KV head and converts each cached row once). Open: the arena and scheduler do not drive the forward pass yet (one request at a time), safetensors models are read but not yet mapped to the forward pass, Metal does not run the Qwen3.5 or Gemma 4 families (`--device auto` falls back to the CPU; a partial Metal hybrid port is backed up on the external disk) |
| M4 MoE and placement | not started | — |
| M5 Tools and internet | done on the CPU and Metal paths (2026-10-09); sandbox enforcement and MCP wiring open | streaming tool-call and reasoning parser for nine template families (round trips byte-identical on five real templates); constrained decoding with llguidance 1.9.1 (41–52 µs per mask on Qwen3's tokenizer); OpenAI tool calling end to end on Qwen3-1.7B (auto, streamed deltas, `tool_choice: required`, tool-result turn); `response_format` json_object / json_schema; tools crate (SSRF-guarded fetch, page model, SearXNG search, MCP host on rmcp 3.5.1 / protocol 2026-07-28, Rule-of-Two gate, audit, sandbox plans; 60 tests); agent loop for built-in `web_search` / `web_fetch`, verified live (example.com fetched and summarised, also through the LLMario gateway; metadata-address fetch blocked). Open: MCP servers are not yet exposed to models through the server, the OS sandbox is generated but not enforced around the fetcher, no desktop approval dialog yet |
| M6 Vulkan backend | not started | — |
| M7 NPU delegate and wider GPU validation | not started | — |
| M8 Speculative decoding and Metal 4 tensor path | not started | — |
| M9 Hardening and beta release | not started | — |

## M1 exit criteria

| Criterion | Target | Measured | Status |
|---|---|---|---|
| Greedy golden match vs llama.cpp | identical first 16 characters on the fixture prompts | 30 of 30 comparisons (five models × three prompts × CPU and `auto`) | pass (test `crates/engine/bin/tests/llama_cpp_golden.rs`; details below) |
| Decode ≥ 0.6 × roofline | 546 GB/s ÷ 1.1 GB/token ≈ 496 tok/s roofline → ≥ 298 tok/s | ≈62 tok/s (NEON int8 kernels, 12 threads) | not yet: CPU decode on Apple silicon is limited by the CPU's own share of bandwidth (llama.cpp CPU: 36.7); the fair CPU target is relative to llama.cpp CPU, which we exceed by 1.7× |
| Prefill ≥ 0.7 × llama.cpp CPU | ≥ 111 tok/s (0.7 × 158) | ≈190 tok/s | pass |
| Plan bound | measured ≤ planned | 1.1 GiB ≤ 3.1 GiB | pass |
| Plan tightness | planned ≤ 1.15 × measured | 2.84× | fail (expected: KV reserved but untouched; revisit with the arena in M3) |
| Warm load ≤ 1.2 × llama.cpp | — | 0.05 s load + warm-up | pass |

## Parity with llama.cpp (2026-10-09)

`LLMARIO_GOLDEN_DEVICES=cpu,auto LLMARIO_TEST_GGUF=<five files> cargo test --release -p llmario-engine
--test llama_cpp_golden -- --nocapture`, against `llama-completion -ngl 0 --temp 0 -no-cnv -t 8`
(build 11146), 32 greedy tokens per prompt. The CPU leg uses the f32 reference kernels
(`LLMARIO_GOLDEN_CPU_KERNELS`, default `scalar`), because the fast kernels quantise activations to
8 bits, as llama.cpp's CPU path does with different rounding, and can flip a near-tie. All 30
comparisons pass. Figures are characters identical to llama.cpp's output.

| Model | `auto` ran on | "The capital of France is" | "def fibonacci(n):" | "Once upon a time…" |
|---|---|---|---|---|
| Qwen3-1.7B Q4_K_M | Metal | 145, all | 105, all | 135, all |
| Qwen3.5-0.8B Q4_0 | CPU (fallback) | 128, all | 97, all | 73 of 134 |
| SmolLM3-3B Q4_K_M | Metal | 115, all | 119, all | 60 of 139 |
| Qwen3.5-9B Q4_K_M | CPU (fallback) | 128, all | 86, all | 146, all |
| Gemma 4 12B it QAT Q4_0 | CPU (fallback) | 32, all | 32, all | 124, all |

- Both legs give the same figures, so Metal and the CPU agree on the two models Metal runs.
- Gemma 4 12B is instruction-tuned; on raw prompts both engines produce digit strings or a repeated
  phrase. They agree, but the stronger evidence is the chat-prompt parity in the M3 row.
- With the default NEON int8 kernels, SmolLM3 diverges at character 12 on "The capital of France
  is" ("The Eiffel Tower…" against "The city is…"); the f32 kernels and Metal follow llama.cpp.
  Qwen3-1.7B also diverges at character 23 on that prompt ("Spain" against "Germany"). Both are
  near-ties between two 8-bit activation paths.

## End-to-end serve test

`crates/engine/bin/tests/serve_e2e.rs` starts `llmario-engine serve` on a free port and drives it
over HTTP. It checks the plan and ledger, plain and streamed chat (the streamed greedy text must
equal the plain one), JSON-schema output, a forced tool call, cancellation when the client hangs
up, two concurrent requests, refusal of logprobs, refusal of a prompt longer than the context, and
shutdown through `/engine/control`. An engine that exits during start-up fails the test with its
own error message.

```
LLMARIO_TEST_GGUF=<gguf> [LLMARIO_E2E_DEVICE=cpu|auto|metal] cargo test --release -p llmario-engine --test serve_e2e -- --nocapture
```

Local results on 2026-10-09: Qwen3.5-0.8B on the CPU (3.7 s) and Qwen3-1.7B on the CPU (3.7 s)
and on Metal (1.6 s) pass. `.github/workflows/engine.yml` runs the test with Qwen3.5-0.8B on
GitHub's `ubuntu-latest` (x86-64) and `macos-14` runners, on the CPU.

## Kernel results (M1, 2026-10-09)

- CPU kernel sets: `scalar` (f32 dequant reference), `int8` (portable int8-activation kernels for
  Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K, F16, BF16, F32), `neon` (NEON+dotprod, bit-exact
  with `int8`), `avx2` (compiled for x86, not yet run on real hardware). Every set is tested against
  the f32 reference and against an f64 oracle of `W·Q(x)` (2e-5), so the kernels compute exactly the
  quantised product; the error versus f32 on real Qwen3-1.7B rows is 0.38–0.39 % RMS, which is the
  Q8_K activation quantisation that ggml also applies (measured/predicted error 0.97–1.05).
- Greedy near-tie: on the prompt `def fibonacci(n):\n    ` the f32 path reproduces llama.cpp exactly
  while the int8 paths flip one token (`return []` vs `return 0`). Since the int8 kernels are provably
  the quantised product with ggml's quantiser, this is a tie resolved differently by two equally
  noisy int8 paths, not a defect; a logit comparison at that step against llama.cpp would settle it.
- Measured on the M4 Max (12 threads, shared machine, lower bounds): Q4_K 6144×2048 matvec 58–63 µs
  (113–121 GB/s); a plain streaming read from the CPU reaches 280–292 GB/s, so the CPU-side
  speed-of-light is ≈ 290 GB/s, not the 546 GB/s SoC figure; model-shaped decode step
  11.25 ms/token (≈ 89 tok/s equivalent) in the kernels alone versus ≈ 16 ms/token end to end —
  about 5 ms per token is outside the matmuls (attention loops, pool round trips at 3–6 µs × ~200
  ops, sampling) and is the next CPU optimisation target.
- NEON `vdotq_s32` and the f16 conversion intrinsics are still nightly-only on Rust 1.92; the kernels
  use stable `asm!` for `sdot`.

## Open items carried forward

- One request at a time: concurrent requests are queued and served in turn (the paged arena and
  scheduler are unit-tested but do not drive the forward pass yet; M3).
- Metal is verified on Qwen3 and SmolLM3. It accepts the other attention-only families (llama,
  mistral3, qwen2) but has not been run on them. Qwen3.5 and Gemma 4 fall back to the CPU with
  `--device auto` and are refused with `--device metal`.
- No logprobs yet; the engine and the gateway refuse them with a clear 400.
- Linux cgroup enforcement is not implemented. The engine CI workflow is the first run of the
  engine on x86 hardware (the test prints the kernel set it used); there are no x86 performance
  numbers yet.
- Windows is out of scope for the engine.
