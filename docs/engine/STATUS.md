# Native engine: build status

Running record of the milestones in `ARCHITECTURE.md` §13. Numbers are measured on the
maintainers' Apple M4 Max (16-core, 64 GB, macOS 27) unless stated; every row names the command.

**Platforms:** macOS on Apple silicon (CPU and Metal) and Linux (CPU). The engine does not support
Windows; LLMario's adapter reports it unavailable there and models run on llama.cpp instead.

| Milestone | State | Evidence |
|---|---|---|
| M0 Baselines and scaffolding | done (2026-10-09) | crates under `crates/engine/`, `adapter_native`, CI jobs run on the PR; dequantizers cross-checked bit-for-bit against gguf-py on two real files (`cargo test -p llmario-engine-formats --test dequant_ref` with `LLMARIO_DEQUANT_REF`); llama.cpp CPU baseline for Qwen3-1.7B Q4_K_M: `llama-bench -ngl 0 -t 12` → pp128 158 tok/s, tg32 36.7 tok/s (build 11146; this understated llama.cpp's CPU path: a strictly CPU-only re-run on 2026-10-10 gives far more, see the M1 exit criteria) |
| M1 CPU dense engine over GGUF | in progress | families llama, mistral3, qwen2, qwen3, smollm3; tokenizer parity 59/59 strings on seven models vs `llama-tokenize`; chat templates byte-identical to Python Jinja2 on six models; greedy parity with `llama-completion -no-cnv --temp 0` on five models (see "Parity with llama.cpp" below); full stack `llmario-beta run qwen3-1.7b` through gateway and supervisor: ready 0.25 s, TTFT 89 ms, 53 tok/s (40 tokens); engine alone with the f16 KV cache: decode 69–74 tok/s after a 512-token prompt, prefill 255–258 tok/s (three runs of `raw-run --tokens <512 ids> --n 65 --ctx 1024 --threads 12 --device cpu`, NEON kernels); plan bounds measured peak (1.1 GiB measured vs 3.1 GiB planned after load; tightness waits for the KV arena, since untouched reservations are not resident) |
| M2 Metal backend | done (2026-10-09), prefill tuning open | `crates/engine/metal`: zero-copy weights over the mmap, f16 KV, runtime-compiled MSL, residency sets; logits vs the CPU f32 path within 0.0133 with identical argmax; Qwen3-1.7B decode 205–207 tok/s vs llama.cpp Metal 213 (0.97×), prefill 3,289–3,301 vs 4,043 tok/s (0.81×, `llama-bench -ngl 99 -p 512 -n 64`); through the full stack `llmario-beta run`: 200.5 tok/s / TTFT 31 ms vs llama.cpp 209.1 / 37 ms, identical text; `--device auto` falls back to the CPU for families Metal does not cover (Qwen3.5 today) |
| M3 KV subsystem, hybrids, multi-slot, safetensors | in progress | done: Qwen3.5 / Qwen3-Next hybrid family (CPU; parity with llama.cpp on 0.8B and 9B), Gemma 4 dense family (CPU; parity on the 12B up to 1,343-token chat prompts), safetensors / MLX folder reader (bit-exact with MLX on four models); paged KV cache on the CPU and Metal whose memory follows the cached tokens, q8_0 KV on both, continuous batching over a shared block pool, and the SSD tier for evicted conversations (see "Memory" below). Gemma 4 on Metal (2026-10-10): the sliding-window layers keep their rows in a second pool of 32-position blocks released as the window slides; 12B decode 48.2–49.8 tok/s vs llama.cpp Metal 51.0–51.5 (CPU path: 15), prefill 403–457 vs 556–588 tok/s; greedy parity on Metal 32, 32, 124 characters (f16 and q8_0, same as the CPU); 32K context, physical footprint after load and after a 3,510-token prompt 477 / 853 MiB vs llama.cpp 1,263 / 1,937 MiB (f16) and 472 / 744 vs 800 / 1,173 MiB (q8_0). Qwen3.5 on Metal (2026-10-10): Gated DeltaNet kernels (causal conv with each sequence's history, a fused delta-rule scan with the state in registers, gated norm) and gated attention; each sequence's recurrent state is one buffer created on its first token and freed when cleared. 0.8B decode 378–385 tok/s vs llama.cpp Metal 231–260 (CPU path: 189), prefill 4,160–5,197 vs 8,806–8,843; 9B decode 66.6–68.3 vs 65.4–66.9 (CPU path: 26), prefill 636–689 vs 808–822; greedy parity on Metal 128, 97, 73 (0.8B) and 128, 86, 146 (9B) characters with f16 and q8_0, the CPU's figures; 9B at 32K, footprint after load / after a 3,776-token prompt 509 / 628 MiB vs llama.cpp 1,366 / 1,502 (f16) and 508 / 572 vs 886 / 1,023 (q8_0). Open: safetensors models are read but not yet mapped to the forward pass, recurrent and window caches have no turn-end checkpoints yet, prompts of the Gemma 4 and Qwen3.5 families on Metal use the decode attention kernel (no flash kernel with a window mask or 256/512-wide heads yet) and the DeltaNet scan runs token by token |
| M4 MoE and placement | in progress | done: Qwen3 MoE (`qwen3moe`) on the CPU and Metal, greedy parity with llama.cpp on Qwen3-Coder-30B-A3B (161, 96 and 105 matching characters on the three fixture prompts), decode 47–51 tok/s vs llama.cpp's CPU path 27–41, prefill 58–76 vs 74–102 tok/s (four interleaved rounds on 2026-10-10 under heavy background load, so llama.cpp's figures varied ±2–14 tok/s; commands under the M1 exit criteria); the planner streams routed experts from disk when the weights exceed memory (see "Memory"). On Metal (2026-10-10, same model): decode 96–99 tok/s vs llama.cpp Metal 82–100, prefill 1,196–1,200 vs 1,624–1,627 tok/s (`llama-bench -ngl 99 -fa auto -p 512` / `-n 128 -d 512` against `raw-run --device metal`, two interleaved rounds); greedy parity on Metal 161, 96, 105 characters with the f16 cache and 161, 112, 151 with q8_0; the serve e2e tests pass on Metal. A plan that streams experts from disk runs on the CPU (Metal keeps every weight resident), and `--device metal` refuses it. Open: a GPU expert cache so streamed plans can use Metal, prefill speed on Metal, other MoE families, placement across devices |
| M5 Tools and internet | done on the CPU and Metal paths (2026-10-09); sandbox enforcement and MCP wiring open | streaming tool-call and reasoning parser for nine template families (round trips byte-identical on five real templates); constrained decoding with llguidance 1.9.1 (41–52 µs per mask on Qwen3's tokenizer); OpenAI tool calling end to end on Qwen3-1.7B (auto, streamed deltas, `tool_choice: required`, tool-result turn); `response_format` json_object / json_schema; tools crate (SSRF-guarded fetch, page model, SearXNG search, MCP host on rmcp 3.5.1 / protocol 2026-07-28, Rule-of-Two gate, audit, sandbox plans; 60 tests); agent loop for built-in `web_search` / `web_fetch`, verified live (example.com fetched and summarised, also through the LLMario gateway; metadata-address fetch blocked). Open: MCP servers are not yet exposed to models through the server, the OS sandbox is generated but not enforced around the fetcher, no desktop approval dialog yet |
| M6 Vulkan backend | not started | — |
| M7 NPU delegate and wider GPU validation | not started | — |
| M8 Speculative decoding and Metal 4 tensor path | not started | — |
| M9 Hardening and beta release | not started | — |

## M1 exit criteria

| Criterion | Target | Measured | Status |
|---|---|---|---|
| Greedy golden match vs llama.cpp | identical first 16 characters on the fixture prompts | 30 of 30 comparisons (five models × three prompts × CPU and `auto`) | pass (test `crates/engine/bin/tests/llama_cpp_golden.rs`; details below) |
| Decode ≥ 0.6 × roofline | 546 GB/s ÷ 1.1 GB/token ≈ 496 tok/s roofline → ≥ 298 tok/s | 68–97 tok/s (NEON int8 kernels, 12 threads, 512 tokens in the cache) | not yet. Correction (2026-10-10): this row used to say the engine beats llama.cpp's CPU path by 1.7×, against a 36.7 tok/s baseline that understated llama.cpp. Re-measured strictly on the CPU, llama.cpp decodes 82–117 tok/s, so the engine is 17–27 % slower on this dense model (and faster on Qwen3 MoE, see M4) |
| Prefill ≥ 0.7 × llama.cpp CPU | ≥ 0.7 × 263–300 tok/s (pp512) | 221–248 tok/s | pass (0.75–0.86×; llama.cpp's CPU prefill uses Apple's Accelerate BLAS, which this engine does not use) |
| Plan bound | measured ≤ planned | 1.1 GiB ≤ 3.1 GiB | pass |
| Plan tightness | planned ≤ 1.15 × measured | 2.84× | fail (expected: KV reserved but untouched; revisit with the arena in M3) |
| Warm load ≤ 1.2 × llama.cpp | — | 0.05 s load + warm-up | pass |

Comparison commands (2026-10-10, four interleaved rounds, load average 14–44 from macOS background services):
`llama-bench -m <gguf> -ngl 0 -dev none -nopo 1 -nkvo 1 -t 12 -p 512 -n 0` and `… -p 0 -n 64 -d 512`
(no GPU device, no op offload to the GPU, KV on the CPU) against `llmario-engine raw-run --device cpu
--threads 12 --ctx 1024 --n 65` with a 512-token prompt.

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

Since the paged cache and batching (2026-10-09) the test also checks that the KV memory in use is
above zero and below the reservation, that the backend runs the plan's KV type
(`LLMARIO_E2E_KV=f16|q8_0`), and that three concurrent requests share batched steps
(`LLMARIO_E2E_SLOTS`, default 2). A second test, `kv_disk_tier_end_to_end`, runs one slot with
`--kv-cache-dir`: a long conversation pushed out by another request is saved, asked again it is
read back (`cached_tokens` = prompt − 1) with the same greedy answer, and after a restart it is read
back again; the directory is 0700 and the files 0600. Both tests pass on Qwen3-1.7B (CPU; Metal
with f16 and q8_0), Qwen3.5-0.8B and Gemma 4 12B on the CPU (for the last two the cache cannot be
rewound, so only the save is checked; their restores are covered by the snapshot unit tests).

## Memory (2026-10-09)

The goal is an engine that runs well on 16 GB machines and smaller. Memory now follows use instead
of being reserved for the whole context up front. Qwen3-1.7B Q4_K_M with a 32K context, physical
footprint (`/usr/bin/footprint <pid>`: anonymous and wired memory; the 1.03 GiB of mapped weights
are file pages the OS shares and can drop, and are not included), one slot, after load and after
a 3,910-token chat prompt. "Before" is the `main` build of 2026-10-09 (06d642e):

| Device, KV | Before: after load | Before: after prompt | Now: after load | Now: after prompt |
|---|---|---|---|---|
| CPU, f16 | 3,659 MiB | 3,712 MiB | 75 MiB | 555 MiB |
| CPU, q8_0 | — | — | 75 MiB | 355 MiB |
| Metal, f16 | 3,879 MiB | 3,882 MiB | 295 MiB | 724 MiB |
| Metal, q8_0 | — | — | 292 MiB | 520 MiB |

Command: `llmario-engine serve --ctx 32768 --parallel 1 --threads 8 --device cpu|metal
[--kv-type f16|q8_0]`, then one chat request. Gemma 4 12B and Qwen3.5-9B on the CPU (recorded
with commit 832094c, same metric): 167–183 MiB after load, 403–641 MiB after 1,000 tokens.

- **Paged KV cache.** 32-token blocks of every full-attention layer's K and V rows, backed when a
  sequence first reaches them and released when no sequence uses them (lazily-backed mappings on
  the CPU; one shared `MTLBuffer` per block on Metal, reached through per-sequence tables of GPU
  addresses, because Metal charges a buffer's whole size at creation). Window rings and recurrent
  state are created on a sequence's first token and freed when it is cleared. Shared partial blocks
  are copied on write.
- **q8_0 KV cache** (ggml `block_q8_0`, 17/32 of f16) on the CPU and Metal, chosen automatically
  when the memory ceiling is 16 GiB or less, and by the planner before it shortens the context.
  Greedy parity with llama.cpp run with `-ctk q8_0 -ctv q8_0 -fa on`: 30 of 30 comparisons, as with
  f16 (`LLMARIO_GOLDEN_KV=q8_0`).
- **Continuous batching over a shared pool.** Slots share one block pool; each step is one batched
  forward carrying every decoding request's next token and then prompt chunks. Batched results equal
  separate runs (unit tests on the CPU and Metal; three concurrent requests in the e2e test).
- **MoE experts streamed from disk.** For a mixture-of-experts model larger than memory the plan
  keeps the dense weights and a resident share of the experts (a quarter, or four tokens' worth of
  active experts if more) and leaves the rest to the page cache. Qwen3-Coder-30B-A3B (17.3 GiB) at
  a 16 GiB ceiling plans 5.75 GiB at an 8K context with 12.26 GiB of experts streamed
  (`llmario-engine plan <gguf> --memory-limit 17179869184 --ctx 8192`).
- **SSD tier.** A conversation pushed out of memory is written to a file and read back when it
  continues (`--kv-cache-dir`; on by default from the LLMario app under `$LLMARIO_HOME/cache/kv`,
  8 GiB, `[backends.native] kv_cache_gb`, 0 = off). Files are written straight from the cache
  memory, made durable by a background thread before they are renamed into place, keyed by model
  and cache layout, owner-only, and evicted least recently used first. Qwen3-1.7B, 4,374-token
  prompt asked again after another request took the only slot:

| Device, KV | Prompt computed | Read back from disk | File | Save / restore |
|---|---|---|---|---|
| CPU, f16 | 52.2 s | 0.38 s | 480 MiB | 59 / 94 ms |
| CPU, q8_0 | 49.6 s | 0.26 s | 255 MiB | 26 / 27 ms |
| Metal, f16 | 1.64 s | 0.11 s | 480 MiB | 217 / 53 ms |
| Metal, q8_0 | 1.74 s | 0.08 s | 255 MiB | 163 / 29 ms |

  Times are whole requests (8 generated tokens) measured by the client; the same greedy answer in
  every run. For comparison, `llama-bench -ngl 0 -t 8 -p 4096` gives 122 tok/s on the CPU (with
  Apple's Accelerate BLAS, which this engine does not use) against about 84 tok/s here.

CPU decode also got faster for every model (Qwen3-1.7B: 91.6–97.9 vs 70.6–72.7 tok/s, 12 threads,
interleaved A/B against `main`): decode attention splits a task's keys across threads when there
are fewer (row, KV head) tasks than threads, and the thread pool hands work over with less
waiting. With the paged cache Metal prefill is unchanged and decode is about 2 % slower.

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

- MoE models run on Metal only when every weight fits in memory; a plan that streams experts from
  disk runs on the CPU until the GPU has an expert cache. MoE prefill on Metal is 0.74 × llama.cpp.
- Expert streaming is planned and the plan is tested; decode speed under real memory pressure on a
  16 GB machine has not been measured yet.
- The paged cache costs about 2 % of Metal decode speed.
- Recurrent (Qwen3.5) and window (Gemma 4) caches cannot be rewound, so the SSD tier and in-memory
  prefix reuse help them only when the whole saved state is a prefix of the next prompt; turn-end
  checkpoints (Architecture §8.6) are not built yet.
- CPU prefill at 4K tokens is 0.69 × llama.cpp's Accelerate-backed CPU path (84 vs 122 tok/s).
- CPU decode on dense models is 17–27 % behind llama.cpp's CPU path (Qwen3-1.7B, 2026-10-10).
- Metal is verified on Qwen3, SmolLM3, Qwen3 MoE, Gemma 4 and Qwen3.5. It accepts the other
  attention-only families (llama, mistral3, qwen2) but has not been run on them. Its DeltaNet
  kernels cover 128-wide heads and a 4-tap conv (Qwen3.5's geometry); other hybrids are refused.
- On Metal, the batched (prompt) matrix kernels round activations to f16 inside their tiles, as
  llama.cpp's do: a prompt computed at once differs from the same tokens one at a time by up to
  ~0.004 in the logits of the tiny Gemma 4 test model (0.0003 token by token against the CPU).
- No logprobs yet; the engine and the gateway refuse them with a clear 400.
- Linux cgroup enforcement is not implemented. The engine CI workflow is the first run of the
  engine on x86 hardware (the test prints the kernel set it used); there are no x86 performance
  numbers yet.
- Windows is out of scope for the engine.
