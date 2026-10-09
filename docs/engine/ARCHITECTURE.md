# LLMario Native Inference Engine — Technical Architecture

**Status:** Draft v0.2 for review (v0.2, 2026-10-09: open-source-only constraint applied) · **Date:** 2026-10-08 · **Applies to:** LLMario ≥ 0.3 (beta channel first)
**Relation to ADR 0001:** ADR 0001 ("control plane over existing engines") stays in force. This document proposes
the native engine as a *third adapter* behind the same supervisor and gateway, and becomes ADR 0002 when accepted.

**Companion material (all in this repository)**

| File | What it is |
|---|---|
| `docs/engine/reports/Local LLM engine architecture research.md` | The research synthesis this design is built on: 14 sections, 372 sources, every number labelled *measured* / *vendor-claimed* / *community* / *unverified* / *background* |
| `docs/engine/research_notes/Local LLM engine architecture research/*.md` | The nine underlying research notes (engine landscape, quantization and formats, KV cache and attention, CPU execution, GPU/NPU backends, sharding and parallelism, OS memory governance, internet tools and agents, target models) |
| `docs/engine/OPUS_5_5_BUILD_PROMPT.md` | The implementation prompt for the coding model that will build this engine |
| `docs/adr/0001-architecture.md` | The current control-plane design the engine plugs into |

**How to read this document.** Sections 1–3 say what is being built and why. Sections 4–12 are the design; each
ends with **Decisions** (what is fixed) and **Evidence** (which measurements forced it, with pointers into the
research report, cited as "Report §<heading>"). Section 13 is the delivery plan with exit criteria that are
measurable on the hardware LLMario already owns. Section 14 lists risks, Section 15 the open decisions, Section 16
the licensing policy. The appendices hold the formulas, tables and interface sketches that the build prompt refers
to by name.

**Conventions.** GiB = 1024³ bytes, GB = 10⁹ bytes; the report keeps whichever unit the source used. "tok/s" for
decode is single-stream unless stated. "Roofline" or "speed-of-light" means `η · BW_peak / bytes_per_token`
(Appendix A). Every performance number quoted here comes from the report and carries its label.

---

## 0. One-page summary

LLMario will ship its own inference engine, written in Rust under Apache-2.0, that runs as one process per loaded
model under the existing supervisor and speaks the existing OpenAI-compatible contract to the existing gateway.
It is designed around four commitments that no current local engine makes together:

1. **A deterministic, printable memory plan before any allocation, and a hard promise never to swap.** The engine
   computes weights, KV cache, recurrent state, scratch, caches and drafter buffers per device with a no-allocation
   dry run, compares them with per-device budgets that come from the operating system (Metal working-set size,
   DXGI budget, cgroup limits), degrades in a fixed, visible order (caches → context → KV precision → host offload →
   refuse) and verifies the plan against the measured peak after load. The plan and a live memory ledger are API
   surfaces, shown in the desktop app and the CLI.
2. **Decode at the memory-bandwidth speed-of-light, prefill on the matrix units.** Decode is already
   bandwidth-bound in every engine (55–80 % of peak on Apple and RTX hardware; community-measured), so the engine
   logs its own speed-of-light per device and treats any shortfall as a bug. The engineering budget goes where the
   measured headroom is largest: the CPU mixture-of-experts path (today 3–4× under the bandwidth bound), few-row
   GEMM kernels that make speculative decoding pay, and tensor-level placement with a hot-expert GPU cache
   (1.6–2.2× measured).
3. **Placement instead of parallelism inside one machine; parallel work only where it cannot cause contention.**
   Each tensor is assigned to one device by the planner (dense path on the fastest device, routed experts in host
   RAM, KV where the attention runs). Within a device, work is parallel across SIMD lanes, threads and streams
   under strict rules: one thread issues GPU work, the CPU pool is sized to performance cores and parks instead of
   spinning, and no component may allocate outside its planned budget. Multi-device execution is an interface
   (`Device`, `Transport`), not a v1 feature, because the measurements show it is a capacity win, not a speed win.
4. **Internet access that is safe by construction.** Tool calling uses the model's own chat template as the
   source of truth (eight wire formats, incremental parsers, grammar-enforced arguments), the Model Context
   Protocol revision 2026-07-28 through the official Rust SDK, built-in search and fetch tools with a windowed page
   model, and a policy gate that applies Meta's "Rule of Two": once untrusted web content has entered a session,
   any state-changing or exfiltration-capable tool call needs human approval. Fetchers and local MCP servers run in
   an OS sandbox with an egress allowlist.

Everything else — GGUF and safetensors readers, quantized kernels for CPU/Metal/Vulkan, paged KV cache, prefix
caching, continuous batching, speculative decoding — is implemented from the same MIT/Apache sources every other
engine learned from, with the specific traps the research catalogued (uncapped buffer caches, worst-case KV
preallocation, Windows system-memory fallback, macOS compression of unwired weights, 4-bit keys collapsing some
models, spin-waiting threads starving the GPU) designed out rather than patched. **Only open-source technology is used** (Section 16, rule 0): every library,
runtime, kernel toolchain and service the engine depends on is OSI-licensed; the only exceptions are the operating
systems' own APIs (Metal, DXGI/Win32), reached through open-source crates. That rules out CUDA, Apple's Accelerate
framework, closed NPU runtimes and paid search APIs, and it is why the single GPU path for every non-Apple GPU is
Vulkan.

The delivery plan (Section 13) has ten milestones. The first is baselines on the maintainers' own machines, because
the one comparison the research could not find anywhere is a Rust CPU engine against llama.cpp on the same box.

---

## 1. Goals, non-goals and targets

### 1.1 Problem statement

LLMario today runs llama.cpp and MLX-LM as child processes (ADR 0001). That bought a working product quickly, and
the measurements in the README show the gateway adds no measurable overhead and that LLMario's resource policy
beats Ollama's defaults by 29–34 % at four concurrent requests. But the premise of ADR 0001 — "the wedge is
configuration and resource policy, not kernels" — has a ceiling: the control plane cannot see inside the engine. It
cannot know where a tensor lives, cannot bound the engine's caches except through whatever flags the engine
exposes, cannot plan memory exactly (its estimates bound the measured peak at 1.18–1.79×, which is safe but wastes
up to 44 % of the budget), cannot share one KV arena across models, cannot run tools inside the generation loop,
and cannot fix the engine's own resource bugs (an MLX allocator retaining 2.5 GiB after a batch, llama-server's
spinning threads, Metal residency loss after two seconds idle).

The user's requirements for the new engine are, verbatim: *use very little memory or use it very efficiently; use
the CPU and any local GPU cores efficiently; run smaller and mid-sized open-source models on local machines; connect
the models to the internet; shard the models and memory into parallel operations without introducing resource
issues.* The research report translates each into measurable terms, and this section fixes them as targets.

### 1.2 Goals

| # | Goal | What it means concretely |
|---|---|---|
| G1 | Predictable, minimal memory | Peak measured memory ≤ planned peak for every supported model and profile; planned ≤ 1.15 × measured; zero growth over an 8-hour soak; weights shared through the page cache across processes; nothing is ever paged to swap by design |
| G2 | Bandwidth-bound decode, compute-bound prefill | Dense decode ≥ 0.6 × roofline on CPU, Metal and Vulkan for 4–32B 4-bit models; CPU MoE decode ≥ 0.5 × active-bytes roofline; prefill within the measured envelope of llama.cpp on the same device (≥ 0.8× at M2 exit, ≥ 1.0× by M8) |
| G3 | Small and mid-size open models | All 37 families in `docs/MODELS.md` plus the 2026 additions in the report (dense GQA, sliding-window, MLA, Gated DeltaNet, Mamba-2, short-conv hybrids, MoE with shared experts, MXFP4/NVFP4 experts, MTP drafters) on 8/16/32/64/128 GB machines, from GGUF and safetensors files without re-quantization |
| G4 | Internet-connected models | Tool calling for every template family in the catalog, built-in `web_search` / `web_fetch` / `retrieve` tools, an MCP client, and a Responses-shaped agent loop with budgets; safe by construction (policy gate, sandbox, SSRF guard, provenance tagging) |
| G5 | Parallelism without resource issues | Continuous batching over N slots sharing one KV arena; tensor placement across CPU, GPU and NPU planned before load; thread pool sized to physical performance cores with bounded spinning; a single GPU-issuing thread per device; per-device byte budgets enforced by the supervisor; thermal and power adaptation; multi-device as an interface |
| G6 | Observability | The plan, the ledger, the speed-of-light, kernel-matrix choices, speculative acceptance, preemptions and effective bandwidth are logged and exported, so that "slow and silent" becomes "slow with a reason" |
| G7 | Platform coverage | macOS/Apple silicon through Metal (the OS GPU API, reached only through the open-source `objc2-metal` crate) first; every other GPU (NVIDIA, AMD, Intel, integrated) through one open-standard Vulkan backend second; Intel NPUs through the open-source OpenVINO runtime; CPU-only everywhere |
| G8 | Open-source only | Every dependency, runtime, kernel toolchain and service is OSI-licensed; no CUDA, no Accelerate/BNNS, no closed NPU runtimes, no proprietary search APIs; the build never needs Xcode's shader compiler or a vendor SDK (Section 16) |

### 1.3 Non-goals (v1)

- Training, fine-tuning, LoRA merging (LoRA *loading* for inference is a later milestone).
- Datacenter throughput (hundreds of concurrent streams, tensor parallelism over NVLink).
- Multi-machine inference. The `Device`/`Transport` interfaces are designed; no transport ships in v1.
- A new model file format. GGUF and safetensors are read natively; only a derived, disposable repack cache is written.
- NPU kernels. NPUs are reached only through vendor runtimes behind a delegate boundary.
- Vision and audio encoders in the first milestones (text path only; side-car `mmproj` files are recognised and ignored until M9).
- Sub-4-bit KV codebooks, dense-weight SSD streaming, score-based KV eviction (SnapKV/H2O class), in-machine CPU–GPU tensor parallelism — all shown by the evidence to lose to simpler alternatives (Report §Adopt first, defer, avoid).

### 1.4 Target hardware tiers

| Tier | Representative machines | Usable budget (planner default) | Model class at 4-bit weights, 8-bit KV |
|---|---|---|---|
| 8 GB | MacBook Air M1/M2 8 GB, thin Windows laptops | ≈5.3 GB (2/3 of RAM minus 1 GB) | ≤ 4B dense/hybrid; 128K context only for hybrids and Gemma 4 E2B |
| 16 GB | MacBook Pro/Air 16 GB, desktops with 16 GB and an 8–12 GB GPU | ≈10.7 GB | 8–12B dense (Qwen3.5-9B, Gemma 4 12B, Granite 4.2 8B, Llama 3.1 8B), LFM2.5-8B-A1B, gpt-oss-20b only with offload |
| 32 GB | MacBook Pro M4/M5 Pro 32–48 GB, RTX 4090/5090 desktops | ≈21.3 GB (+ VRAM on discrete systems) | the whole 24–31B class; 35B-A3B MoEs need a raised working set or a Q3 variant |
| 64 GB | MacBook Pro Max 64 GB (the maintainers' validated machine), Strix Halo 64 GB | ≈48 GB | 70B at Q4_K_M, 35B-A3B at 128K, Qwen3-Next-80B (tight); gpt-oss-120b needs expert offload |
| 128 GB | Mac Studio / MacBook Pro 128 GB, Strix Halo 128 GB, DGX Spark | ≈96 GB (Metal working-set cap) | gpt-oss-120b, Qwen3.5-122B-A10B, Mistral Small 4, Llama 4 Scout |

Numbers computed in Report §Target models from the models' `config.json` files; the budget rule reproduces Gemma 4's
published int8-KV figures within 0.02 GB.

### 1.5 Quantitative acceptance targets

These are the numbers the milestones in Section 13 are checked against. They are deliberately expressed relative to
the machine's own roofline and to engines already installed on the maintainers' machines, because absolute
benchmark numbers rot within months (Report §Decode already runs at 55–80 %).

| Metric | Target | How measured |
|---|---|---|
| Plan bound | measured peak ≤ planned peak, every model in the fit matrix | job-object peak (Windows), cgroup `memory.peak` (Linux), `phys_footprint` + Metal `currentAllocatedSize` (macOS) after a 3-prompt suite |
| Plan tightness | planned ≤ 1.15 × measured | same |
| Soak | footprint drift < 1 % over 8 h of mixed traffic | `llmario bench --soak` |
| Dense decode | ≥ 0.6 × `η=1` roofline on each backend (i.e. ≥ 60 % of peak bandwidth); for the CPU backend on Apple silicon the peak is the CPU-measured streaming bandwidth (≈ 290 GB/s on an M4 Max, M1 finding), not the SoC figure | `llmario bench`, bytes/token from the plan, bandwidth from the hardware table or a streaming-read probe |
| MoE decode (CPU, host experts) | ≥ 0.5 × active-bytes roofline | same |
| Prefill | ≥ 0.8 × llama.cpp on the same device at M2 exit; ≥ 1.0× by M8 | `llmario bench --cache cold` on the 1.8k and 4.7k-token prompts |
| Time to first token | ≤ llama.cpp + 10 % at M2 exit | same |
| Load time (warm page cache) | ≤ 1.2 × llama.cpp mmap load | engine log |
| Idle | after unload, engine process exits and the supervisor's ledger returns to baseline within 2 s | supervisor test |
| Greedy fidelity | greedy 64-token outputs identical across CPU/Metal/Vulkan for f16 KV, and KLD ≤ 0.01 versus the CPU f32 reference per backend | golden tests |
| Tool-call parsing | 100 % of a pinned corpus of template renders per family parse to the expected calls, streaming and non-streaming | parser corpus tests |
| Safety | SSRF corpus fully blocked; injection corpus cannot trigger a state-changing tool without an approval event | security tests |

---

## 2. Design principles

1. **Measure, do not estimate.** Plans come from a no-allocation dry run of the real graph, not from formulas; the
   measured peak is fed back into the planner after every load. llama.cpp's `--fit` and Ollama's exact-measurement
   scheduler both abandoned heuristics for this reason (Report §Memory governance).
2. **Plan before you allocate; allocate once; never grow by reallocation.** Weights are mapped, the KV arena and
   scratch are reserved at admission, caches have byte limits counted in the plan. The MLX buffer pool holding
   108 GB against 34 GB expected is the anti-pattern (Report §No engine budgets memory jointly).
3. **Bytes per token is the yardstick.** Every device reports a speed-of-light; decode kernels are judged by
   achieved bandwidth, prefill kernels by achieved matrix throughput.
4. **The operating system is a co-tenant, not a resource pool.** Budgets come from the OS (`recommendedMaxWorkingSetSize`,
   DXGI budget, cgroup limits); headroom is explicit; pressure signals trigger a fixed degradation ladder; the
   engine volunteers itself to the OOM killer before the desktop session does.
5. **Place tensors, do not parallelise them across devices.** Inside one machine, the planner assigns every tensor
   to one device and computes where the bytes live during decode; cross-device tensor parallelism is avoided.
6. **One process per loaded model; one thread per device issuing work.** Crash isolation and byte caps belong to
   the supervisor; inside the engine a single inference thread owns the step loop.
7. **Degrade visibly, refuse loudly, fall back never silently.** Every kernel path has an explicit support matrix;
   an unsupported combination is a startup error with the matrix printed, not a 25× slower CPU fallback.
8. **The template is the truth for tool calls.** Parsers are per family and incremental; grammar enforcement makes
   malformed arguments impossible; `tool_choice` is enforced, not advisory.
9. **Untrusted content cannot complete an attack.** Rule of Two, provenance tags, sandbox, egress allowlist.
10. **Permissive licences only.** MIT/Apache-2.0/BSD code may be ported with attribution; GPL, AGPL, CC-BY-NC,
    BUSL and FAIR material are design references at most (Section 16).
11. **Stable Rust on the main path.** Nightly-only intrinsics are not allowed in the build; SME2 and AMX go through
    `asm!` or FFI to vendor `.S` kernels.
12. **Additive to LLMario.** The engine is a new adapter and a new binary; existing adapters, the gateway contract
    and the user-facing commands keep working unchanged.

---

## 3. System context and integration with LLMario

### 3.1 Where the engine sits

```text
 Desktop app (Tauri)         CLI (llmario)            Your apps (OpenAI SDK, MCP-aware agents)
          │                        │                              │
          └──────────────┬─────────┴──────────────────────────────┘
                         ▼
   API gateway (crates/api) ── loopback, API key, Host/Origin check, field allowlist, SSE relay, metrics
                         ▼
   Planner (crates/supervisor::planner) ── picks model variant + engine; asks the native engine for an exact plan
                         ▼
   Supervisor (crates/supervisor) ── admission by bytes per device → launch / warm-up / relaunch / byte-based LRU
                         ▼
   Adapters ── llama.cpp (llama-server) · MLX-LM (mlx_lm.server) · NATIVE (llmario-engine) · mock (CI)
                         ▼
   Engine child processes ── contract v1: HTTP on 127.0.0.1:<random>
                            contract v2 (native): HTTP/1.1 over a Unix domain socket (macOS/Linux) or a
                                                 named pipe (Windows), plus /engine/* extension endpoints
```

The native engine is `crates/adapter_native` (an `EngineAdapter` implementation) plus a new binary
`llmario-engine` built from the crates under `crates/engine/`. `BackendKind` gains `Native`; `ModelFormat` gains
`Safetensors` (a Hugging Face-style folder: `config.json`, `tokenizer.json`, `*.safetensors`, optional
`quantization_config`; MLX folders are a subset). Selection rules in `planner::backend_for` prefer the native
engine when it reports support for the model's architecture and quantization types, and fall back to the existing
adapters otherwise; `--backend native|llamacpp|mlx` overrides.

### 3.2 Contract v2

Contract v1 (ADR 0001) is OpenAI-compatible HTTP on an ephemeral loopback port. The native engine keeps that wire
format so the gateway's byte-level SSE relay is reused unchanged, and adds:

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions`, `POST /v1/responses`, `POST /v1/embeddings`, `GET /v1/models` | OpenAI-compatible surface (Section 11) |
| `GET /engine/plan` | The memory plan the engine was admitted with (JSON + rendered table) |
| `GET /engine/ledger` | Live memory ledger per device (Section 5.1) |
| `GET /engine/stats` | Speed-of-light per device, achieved bandwidth, acceptance rates, preemptions, slot states |
| `POST /engine/control` | `{"action":"sleep"}` (free KV and scratch, keep weights mapped), `{"action":"wake"}`, `{"action":"drop_caches"}`, `{"action":"exit"}` |
| `GET /healthz` | Readiness including backend self-test result |

Transport: a Unix domain socket under `$LLMARIO_HOME/run/engine-<id>.sock` (mode 0600) on macOS and Linux; a named
pipe `\\.\pipe\llmario-engine-<id>` with a DACL limited to the current user on Windows. The supervisor passes the
path on the command line; the gateway's `reqwest` client connects over the socket. Loopback TCP remains available
(`--listen 127.0.0.1:0`) for debugging and for the `llmario bench --url` path.

Process lifecycle commands mirror llama-server's router protocol (Report §Sharding and scheduling): the supervisor
sends `/engine/control exit`, waits a bounded time, then kills. The engine writes a one-line JSON state record to
stdout on every state change (`starting`, `planning`, `loading`, `ready`, `sleeping`, `exiting`) so the supervisor
and `llmario doctor` can read it without the socket.

### 3.3 Planning across the process boundary

`llmario-engine plan --model <path> --profile <json> --hardware <json>` runs the planner without loading weights and
prints the plan as JSON within one second (it parses headers only). The supervisor calls it from
`EngineAdapter::extra_memory_bytes` (or, better, a new `plan` method on the trait, defaulting to the heuristic
estimator for external engines) and uses the engine's number as the admission source of truth. The existing
`supervisor::memory::estimate` remains for llama.cpp and MLX-LM.

Because the supervisor can also run several engines, it holds the *machine-level* ledger: the sum of admitted plans
per device, the OS budgets, and the headroom. The engine holds the *process-level* ledger. The two are compared
after each load and the difference is logged (G1, G6).

### 3.4 What changes in the desktop app and CLI

- The "fits in memory" pill becomes exact for native-engine models and shows the plan on hover (per-device bars for
  weights, KV, state, scratch, caches).
- Settings gain *Web access* (off / read-only browsing / tools with approval) and an MCP servers panel; the tool
  approval dialog is the same UI for built-in tools and MCP tools.
- `llmario doctor` prints the native engine's capability report (backends, ISA paths, kernel matrix, self-test
  results, OS budgets, wired-memory limit and whether it was raised).
- `llmario model fit` uses the engine planner; `llmario bench` is unchanged (it speaks OpenAI HTTP) and gains
  `--soak` and `--speed-of-light` columns.

**Decisions.** Native engine = third adapter + one binary; contract v2 = v1 wire format over a local socket plus
`/engine/*`; the engine's planner becomes the admission source of truth for native loads; existing adapters
unchanged.
**Evidence.** Process-per-model is where every mature system converged (llama-server router, Ollama runners,
LM Studio llmster, vLLM V1 EngineCore; Report §Sharding and scheduling); LLMario's current estimates bound peaks at
1.18–1.79× (README) — the planner closes that gap.

---

## 4. Engine process architecture

### 4.1 Crate map

All engine crates live under `crates/engine/` and are prefixed `llmario-engine-`. Dependencies point downward only.

| Crate | Responsibility | Key external dependencies (licence) |
|---|---|---|
| `core` | Tensor and dtype definitions, block-quant layouts (all GGUF types, MXFP4/NVFP4, MLX affine), graph IR, static allocator (simulate and real modes), `Device`/`Buffer`/`Kernel` traits, memory ledger types | `half` (MIT/Apache), `bytemuck` |
| `formats` | GGUF reader (header, metadata, tensor table, split files, side-cars), safetensors reader + `config.json` schema, `quantization_config` (AWQ/GPTQ/compressed-tensors) repack-at-load, MLX folder support, content hashing | `memmap2` (MIT/Apache), `serde_json`, `sha2` |
| `tokenizer` | GGUF-embedded vocabularies (byte-level BPE, SentencePiece, Tekken, o200k/cl100k variants) and `tokenizer.json` via the `tokenizers` crate; incremental detokenisation with UTF-8 boundary handling | `tokenizers` (Apache-2.0) |
| `model` | Architecture descriptions (`ArchSpec`) and graph builders for each family; RoPE families; cache-class declarations per layer; drafter (MTP/DFlash/EAGLE) graph builders | — |
| `plan` | Planner: symbolic shapes, no-alloc simulation, per-device budgets, degradation ladder, plan serialisation, post-load verification | — |
| `kv` | Three cache classes, paged arena, block tables, prefix cache (hash chain), checkpoints, RAM/SSD tiers, KV quantization with rotation | `sha2` |
| `cpu` | CPU backend: ISA dispatch, block kernels (GEMV/GEMM/dequant), attention, SSM/DeltaNet scans, MoE path, thread pool, repack cache | `rayon` is *not* used (own pool); `libc`, `windows-sys` |
| `metal` | Metal backend (Apple silicon) | `objc2-metal` (Zlib/Apache/MIT), `objc2-foundation` |
| `vulkan` | Vulkan backend | `ash` (Apache/MIT), `shaderc`/`naga` at build time |
| `npu` | NPU delegate trait, mock delegate, optional OpenVINO GenAI delegate behind a feature (the only NPU runtime with an open-source licence at the time of writing) | OpenVINO GenAI C API via FFI (Apache-2.0) |
| `sched` | Slots, continuous batching, token budget steps, admission per request, preemption by recompute, idle policy | — |
| `decode` | Sampling (CPU and on-device), logit processors, grammar masks, speculative decoding (n-gram, MTP, DFlash/EAGLE), greedy-equivalence checks | `llguidance` (MIT) |
| `chat` | Jinja chat templates, per-family tool-call formats and incremental parsers, auto-parser, thinking-history policies, Responses item model and Chat Completions projection | `minijinja` (Apache-2.0) |
| `tools` | Built-in tools (`web_search`, `web_fetch`, `retrieve`), MCP client (`rmcp`), tool registry and classification, policy gate, provenance tagging, sandbox launchers, egress proxy | `rmcp` (Apache-2.0), `reqwest`+`rustls`, `dom_smoothie` (MIT), `htmd` (Apache-2.0), `texting_robots` (MIT/Apache), `rusqlite`+`sqlite-vec` (MIT/Apache), `text-splitter` (MIT) |
| `server` | HTTP over UDS/named pipe/TCP, OpenAI endpoints, `/engine/*`, SSE streaming, metrics | `axum`/`hyper`, `tokio` |
| `bin` (`llmario-engine`) | CLI: `serve`, `plan`, `probe`, `bench kernels`, `verify model` | `clap` |
| `testkit` | Golden-token fixtures, reference implementations in scalar Rust, memory-peak probes per OS, corpus loaders | — |

`crates/adapter_native` (outside the engine tree) implements `EngineAdapter` for the supervisor.

### 4.2 Threads and processes

```text
 llmario (supervisor)                  llmario-engine (one per loaded model)
 ┌─────────────────────┐              ┌───────────────────────────────────────────────────────────┐
 │ tokio runtime       │  spawn/ctl   │ main: parse args → plan (or load plan id) → load → serve   │
 │ gateway, planner,   │─────────────▶│                                                           │
 │ machine ledger,     │  UDS/pipe    │  [inference thread]  step loop: build batch → run graph → │
 │ byte-based LRU      │◀────────────▶│     sample → emit events; owns ALL device command issue    │
 └─────────────────────┘              │  [cpu pool: P-core count] parked when idle or when a GPU  │
                                      │     owns the step; woken per fused op                     │
                                      │  [tokio runtime: 2–4 workers] HTTP, SSE, tokenisation,    │
                                      │     template rendering, tool execution, MCP I/O           │
                                      │  [residency keep-alive thread] (Metal only)               │
                                      │  [tool subprocesses] fetcher, stdio MCP servers — OS      │
                                      │     sandboxed, egress through the local allowlist proxy   │
                                      └───────────────────────────────────────────────────────────┘
```

Rules that keep this free of resource issues (G5):

- **Exactly one thread issues device commands** (Metal command buffers, Vulkan queue submits) per device: the inference thread. Tool and HTTP work never touches a device.
- **The CPU pool is sized to the number of physical performance-class cores** (all performance clusters on Apple
  silicon — llama.cpp's `hw.perflevel0.physicalcpu` default undercounts the M5 Max by 3×; P-cores only on Intel
  hybrids; physical cores, never SMT siblings, on AMD/Intel desktops). The count is a plan parameter and a user
  setting.
- **Workers spin for a bounded time (tens of microseconds) then park on a futex/condvar.** When a GPU backend owns
  the step, the pool is parked, not polled. llama.cpp's default 100 ms spin (`--poll 50`) is the documented
  pathology (Report §CPU execution).
- **Priorities follow the user's attention**: `user-initiated` QoS (macOS) / normal class (Windows) / nice 0
  (Linux) while a request is streaming to a foreground client; `utility` QoS / EcoQoS / `SCHED_BATCH` when the
  engine is backgrounded (Apple's guidance: utility or lower at least 90 % of the time when the user is not
  active).
- **The tokio runtime is small (2–4 workers)** and does no heavy compute; tokenisation of long prompts is chunked so
  it cannot block the HTTP tasks.
- **Tool subprocesses are sandboxed and budgeted** (Section 10.7); their memory is counted under the engine's
  `runtime_fixed` category with a cap.

### 4.3 Request data flow

1. Gateway forwards a request over the socket. `server` validates it against the allowlist (the gateway already
   did; the engine re-validates because it also accepts direct socket clients in debug mode).
2. `chat` renders the transcript with the model's template (tools, thinking policy), tokenises it, and checks the
   prefix cache for the longest cached block chain.
3. `sched` admits the request: KV blocks needed = ⌈(prompt + max_tokens) / block⌉ × conservativeness; if not
   available, queue or retract the lowest-priority slot (recompute later). The request gets a slot.
4. The step loop builds a batch at token granularity: decode tokens for all active slots first, then prefill
   chunks up to the token budget (`n_ubatch`). One graph execution per step; sampling on device where available;
   grammar masks applied per slot.
5. Events (tokens, tool-call fragments, reasoning fragments, usage) stream back as SSE; the `chat` parser runs
   incrementally; when a tool call completes, the agent loop (Section 10.6) executes it and re-enters at step 2.
6. On completion the slot's KV blocks either go to the prefix cache (refcounted) or are freed; checkpoints for
   window/recurrent models are stored if the session policy says so.

**Decisions.** Single inference thread; P-core-sized parked pool; small tokio runtime; sandboxed tool processes;
crate map as above.
**Evidence.** Barrier and spin overhead measurements in llama.cpp PRs (30 % of decode time in barriers; 2-vCPU VM
147 s → 4.4 s with yielding; SMT threads reducing throughput; M5 Max thread undercount) and the "never more than one
thread issuing GPU work" rule derived from ggml's allocator-reuse race (Report §CPU execution, §Sharding and
scheduling).

---

## 5. Memory architecture: ledger, budgets, planner, allocators, OS integration

This is the centre of the design. Every other subsystem receives its memory from the plan and reports back to the
ledger.

### 5.1 The ledger

The ledger is a per-device table the engine maintains itself, because OS process counters mislead: Linux RSS and
the Windows working set count shared file-backed pages, macOS reports mapped weights under "Cached Files", and
Metal wired memory is invisible to `ps` (Report §Memory governance).

```text
Ledger
  devices: [host, gpu0, npu0?]
  per device:
    weights_mapped          bytes of model files mapped (file-backed, evictable by the OS)
    weights_resident_est    sampled with mincore / QueryWorkingSetEx / vm_region (estimate, labelled as such)
    weights_wired           bytes pinned: Metal residency sets, mlock, device-local Vulkan allocations
    weights_private         anonymous copies: repacked tiles, in-situ-quantised tensors, dequantised scratch
    kv_arena_reserved       bytes reserved once at admission
    kv_arena_in_use         blocks in use × block bytes (+ recurrent state × sequences)
    scratch_reserved        activation plan peak per graph shape (prefill, decode, drafter)
    caches                  prompt_cache_ram, repack_cache_mapped, grammar_cache, page_cache (tools)
    drafter                 MTP/DFlash/EAGLE weights + verification buffers
    runtime_fixed           code, allocator metadata, tokio, tokenizer, templates, tool subprocess cap
  totals: planned_peak, measured_peak (post-load, running), budget, headroom, pressure_state
```

`GET /engine/ledger` returns it; the supervisor sums admitted plans into a machine ledger; `llmario doctor` prints
both. Prometheus gauges mirror every row.

### 5.2 Budgets and headroom

| OS | Device ceiling | Host ceiling | Pressure signal | Enforcement |
|---|---|---|---|---|
| macOS | `MTLDevice.recommendedMaxWorkingSetSize` (≈2/3 of RAM below 32 GB, ≈3/4 above; community) — the GPU can wire no more; raising it needs root and the undocumented `iogpu.wired_limit_mb` sysctl, which the engine never runs itself (it prints the command and the risk) | physical RAM − OS reserve − sampled resident memory of other processes | `DispatchSource.makeMemoryPressureSource` (normal/warning/critical); `os_proc_available_memory` is not available on macOS | none from the OS; the supervisor enforces by sleep-then-kill |
| Windows | `IDXGIAdapter3::QueryVideoMemoryInfo` local-segment `Budget` (the OS's own per-process GPU budget) or `VK_EXT_memory_budget` | `GlobalMemoryStatusEx` available minus reserve | `CreateMemoryResourceNotification` low/high; DXGI budget-change event | job object with `JOB_OBJECT_LIMIT_PROCESS_MEMORY` = planned peak + margin; `PeakProcessMemoryUsed` read back for free |
| Linux | `VK_EXT_memory_budget`; integrated GPUs share the host ceiling | cgroup v2 `memory.max`/`memory.high` if present, else `MemAvailable` minus reserve | PSI `some`/`full` triggers; cgroup events | the engine places itself in a child cgroup with `memory.high` = planned peak + margin where it has permission; `oom_score_adj` raised so the engine dies before the desktop session |

Headroom per device = max(1 GiB, 10 % of the ceiling) by default (llama.cpp's `--fit-target` of 1,024 MiB and
Ollama's 80 % rule bracket this), plus a user-visible "keep N GB free for other apps" setting that LLMario already
exposes as `memory_headroom_gb`. Unified-memory machines (Apple, Strix Halo, DGX Spark) have one pool with two
ceilings: wired (GPU) and total; the planner respects both.

### 5.3 The planner

Input: model metadata (headers only), the profile (slots, context, KV types, speculative settings), the hardware
report, the budgets, the user's placement overrides.
Output: a `Plan` — per tensor group: device, residency mode (mapped / mapped+wired / private), block type; per
device: KV arena size, recurrent state, scratch peak per graph shape, cache limits, drafter bytes; totals; the
degradation steps taken and the reasons; a content hash.

Algorithm:

1. **Describe.** Parse GGUF/safetensors headers into an `ArchSpec` (layers, head counts, head dims, window pattern,
   expert counts and top-k, SSM dims, vocabulary, RoPE family and scaling, drafter heads) without touching tensor
   data.
2. **Shape.** Build the symbolic graph for the worst-case prefill step (`n_tokens = n_ubatch`, `n_kv = n_ctx`,
   `n_seq = slots`), the decode step (`n_tokens = slots`), and the drafter/verification step if enabled.
3. **Place.** Apply the placement policy (Section 9.4): dense path tensors on the fastest device, routed experts in
   host RAM when they do not fit, KV where attention runs, with user overrides.
4. **Simulate.** Run the static allocator in *simulate* mode over each graph per device to get the scratch peak
   (ggml's `no_alloc` idea — the one feature that made `--fit` possible). Compute KV bytes per cache class
   (Appendix A), recurrent state × slots, cache limits from the profile, drafter bytes, `runtime_fixed` from a
   measured constant per platform.
5. **Check.** Compare per-device totals with ceilings minus headroom.
6. **Degrade** (fixed order, each step logged with before/after):
   1. prompt cache and repack cache limits → 0 (repack cache stays mapped, never private);
   2. slots → next lower profile (throughput 16 → balanced 4 → 1);
   3. context → halve toward the floor (4,096 tokens, or the user's floor);
   4. KV precision → f16 → q8_0 keys and values → q8_0 keys with q4_0 values (only on models that pass the 4-bit
      value guard; 4-bit keys are never chosen automatically — Report §KV cache);
   5. placement → move routed experts to host; then whole layers from the input side so the final layers and the
      output head stay on the fastest device;
   6. refuse with a structured error listing the shortfall per device and the nearest configuration that fits.
7. **Emit** the plan (JSON and a rendered table), store it under `$LLMARIO_HOME/run/plans/<hash>.json`, and print
   the per-device speed-of-light next to it.
8. **Verify after load**: record the measured peak (Section 5.2 enforcement column), compare with the plan, log the
   ratio, and persist `(model hash, profile, backend version) → measured peak` so the next plan for the same
   configuration can tighten `runtime_fixed`.

The plan is immutable for the life of the process; a different profile means a new process. `GET /engine/plan`
returns it.

### 5.4 Allocators

| Memory class | Mechanism | Rules |
|---|---|---|
| Weights (default) | read-only shared `mmap` of the model file (`PROT_READ, MAP_SHARED`; `FILE_MAP_READ` on Windows); bounded prefault of the tensors the plan marks hot (`madvise(MADV_WILLNEED)` / `MADV_POPULATE_READ` on Linux ≥ 5.14, `PrefetchVirtualMemory` on Windows); on macOS Metal buffers are created over the mapping with `newBufferWithBytesNoCopy` in shared storage mode (page-aligned views, each ≤ `maxBufferLength`) and kept resident with `MTLResidencySet`s | one mapping per file, shared through the page cache with any other process using the same file; no private copies unless the plan says `private`; `mlock`/`VirtualLock` only for explicitly marked latency-critical regions and only via `MLOCK_ONFAULT` where available |
| Weights (cold GPU load, discrete GPU) | direct I/O (`O_DIRECT`/`FILE_FLAG_NO_BUFFERING`, io_uring on Linux) straight into device buffers | used only when the plan places a tensor group on a discrete device and the file is cold; measured 10× faster cold loads on NVMe, but slower warm loads on Macs, so never the default on unified memory (Report §Memory governance) |
| Weights (repacked / in-situ quantised) | a derived cache file under `$LLMARIO_HOME/cache/packed/<model-digest>/<layout-id>.bin`, itself mmapped | generated in the background after first load, only for tensors the prefill GEMM path uses; deletable at any time; counted as `repack_cache_mapped`; never produced as anonymous memory at load (llama.cpp's load-time repack costs 17 → 77 s on Phi-4 and defeats page-cache sharing) |
| KV arena | one reservation per device at admission: `MTLHeap` (placement heap) on Metal, one device-local `VkDeviceMemory` block per plan on Vulkan, sub-allocated by the engine (sparse binding to grow in fixed chunks where the device supports it; ggml-cuda's reserve-once-grow-in-chunks pattern), one anonymous mapping (`MADV_HUGEPAGE` where available) on CPU | carved into fixed blocks (Section 8.2); freed by index; never grown by reallocation; recurrent state in a separate pool with its own block size |
| Scratch / activations | static per-graph plan with ggml-alloc-style lifetimes (best-fit over free blocks, in-place reuse only for a whitelisted op set, inputs never overwritten, outputs never freed), reserved once per graph shape from the same heap/arena family | the plan's scratch number is exact because it is the allocator's own simulation |
| Caches | size-bucketed buffer cache with an explicit byte limit from the plan, purged on pressure; prompt cache (RAM) with a byte limit; grammar cache with an entry limit | nothing is "bounded only by the memory limit" |
| Small objects | `mimalloc` (MIT) as the global allocator for metadata, strings, JSON, tokens | never for weights, KV or scratch, which bypass `malloc` entirely so they can be aligned, locked and accounted individually |

### 5.5 OS integration details

**macOS.** `MTLResidencySet` per buffer group (weights, KV, scratch), a keep-alive thread that re-requests residency
every 5 ms while a 3-minute keep-alive counter is positive (llama.cpp PR #11427 pattern; without it decode on an
M2 Ultra went from 26 ms to 472 ms per token after ~2 s idle — Report §Memory governance), residency released after
the idle window so other apps can use the memory. Memory-pressure dispatch source: *warning* → drop caches, stop
admitting new requests, release idle slots; *critical* → cancel queued requests, sleep (free KV and scratch), tell
the supervisor. Thermal state (`ProcessInfo.thermalState`): *serious* → halve prefill chunk and batch, disable
speculation and extra slots; *critical* or Low Power Mode → concurrency 1, no polling. QoS classes as in 4.2. The
App Sandbox (for a future App Store build) is compatible with everything here except raising the wired limit.

**Windows.** DXGI budget polled and subscribed; if `CurrentUsage` approaches `Budget` the engine treats it as
*warning*. **System-memory spill detection:** NVIDIA's Windows driver (since 536.40) silently lets GPU allocations spill into system RAM at a 5–10× slowdown (measured for CUDA; the Vulkan-specific behaviour is unverified), so the engine never allocates beyond the `VK_EXT_memory_budget`/DXGI budget, samples shared-GPU-memory growth into the ledger, runs a per-step time regression check (a step that becomes ≥ 3× slower while the local heap reads as full), then logs the driver setting that disables the spill ("Prefer No Sysmem Fallback") and refuses to admit more. The supervisor wraps the engine in a job object
with a memory limit equal to the plan plus margin, reads `PeakProcessMemoryUsed` after load, and sets EcoQoS when
the engine is backgrounded.

**Linux.** The engine moves itself into a child cgroup (when `cgroup.subtree_control` permits) with `memory.high`
= plan + margin — throttling, not killing — and `memory.max` unset; subscribes to PSI triggers (`some` 150 ms per
1 s window for *warning*, `full` 100 ms per 1 s for *critical*, the systemd-oomd neighbourhood); sets
`oom_score_adj` 500. THP is requested with `MADV_HUGEPAGE` on the KV arena and scratch only (file mappings are not
eligible). Flatpak builds get no cgroup limit; the engine says so in `doctor`.

### 5.6 Admission, idle and multi-model policy (supervisor side)

- Admission is by bytes per device, not by model count: a new load is admitted when `Σ admitted plans + new plan ≤
  ceiling − headroom` on every device the plan touches. Count caps remain as a secondary limit.
- LRU across models is by bytes with pin and per-model TTL (Ollama's "not true LRU" and llama-server's count-only
  eviction are the anti-patterns; oMLX's RAM-minus-8-GB contract is the closest prior art).
- Idle policy: slot state parked to the RAM prompt cache; after the idle window the engine *sleeps* (frees KV and
  scratch, releases residency, keeps weights mapped — the page cache keeps them hot for a fast wake); after the
  unload TTL the process exits.
- Wiring policy: when two engines map the same file, at most one wires it.
- Back-pressure propagates: on *critical* the supervisor refuses new loads machine-wide and may kill the
  lowest-priority engine.

**Decisions.** Own ledger; OS-derived ceilings with explicit headroom; no-alloc planner with the six-step ladder;
mmap-first weights with Metal shared buffers and residency sets; one-time arenas for KV and scratch; byte-limited
caches; byte-based LRU across models; sleep-then-kill.
**Evidence.** Report §Memory governance (field defaults of llama.cpp `--fit`, Ollama 80 % rule and exact-measurement
scheduler, MLX 1.5× working-set limit and pool retention, macOS idle unwiring and compression, WDDM fallback,
cgroup/PSI semantics, PyTorch expandable-segments measurement, flash-moe and FlashMoE streaming results) and
§No engine budgets memory jointly.

---

## 6. Model formats and loading

### 6.1 Inputs the engine reads

| Input | Status | Notes |
|---|---|---|
| GGUF, single file or `-00001-of-NNNNN` split set | M1 | every ggml block type listed in 6.3; side-car files recognised by name and base-model hash: `mmproj-*` (vision/audio projector; ignored until M9), `mtp-*`, `dflash-*`, `eagle3-*` (drafters; M8) |
| Safetensors folder (`config.json`, `tokenizer.json`, `tokenizer_config.json`, `*.safetensors`, optional `generation_config.json`) | M3 (BF16/F16 and MLX affine), M4 (AWQ/GPTQ/compressed-tensors) | MLX folders are safetensors folders whose `config.json` carries `quantization: {bits, group_size}`; BF16 weights are quantised in situ at load (ISQ) to the plan's target type into the repack cache |
| LoRA adapters (GGUF-LoRA, PEFT safetensors) | M9 | applied at load (merged) or as separate matmuls for small ranks |
| EXL3 | not planned | on-disk layout undocumented, CUDA-only kernels (Report §Read GGUF and safetensors natively) |

The engine never writes a distribution format. It writes one derived artefact: the repack cache (Section 5.4).

### 6.2 GGUF reader

- Header version 3; little-endian only (big-endian files are refused with a clear error); `general.alignment`
  (default 32) honoured; tensor data offsets validated against file size before any mapping is used.
- Block sizes come from the C struct definitions in `ggml-common.h`, transcribed into `core::blocks` with a
  `static_assert`-style test that recomputes each file's tensor byte counts and compares them with the header's
  offsets (the IQ4_XS entry is 136 bytes in `gguf-py` and 140 in the C header; the C header is what writers use).
- Metadata keys: the spec's `general.*`, `<arch>.*` keys plus the keys llama.cpp's converter writes but the spec
  page does not list — sliding-window size and layer pattern, per-layer head-count arrays, YaRN betas and
  attention factors, SSM dimensions and group counts, expert counts/top-k/shared experts, MLA latent dims,
  `tokenizer.ggml.*`, `tokenizer.chat_template` (and named variants). The reader is written against llama.cpp's
  converter source as of the pinned commit and carries a test fixture per family.
- Split files: all parts are mapped; a tensor never spans parts.
- Everything is zero-copy: a tensor is a `(file index, offset, length, type, shape, strides)` view over the mapping.

### 6.3 Block types and kernel priority

| Priority | Types | Why |
|---|---|---|
| P0 (M1) | F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, Q8_K (activations) | the installed base: every mainstream GGUF and every `output.weight` (Q6_K) |
| P1 (M3) | Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, IQ4_NL, IQ4_XS | UD-Q3/Q4 recipes, bartowski IQ4_XS files |
| P2 (M4) | MXFP4 (gpt-oss, `MXFP4_MOE` recipes), NVFP4 | gpt-oss and Blackwell-native files; native FP4 tensor-core prefill is CUDA-only today, so under Vulkan these types run through the dequant-to-int8/f16 paths (decode is unaffected: bandwidth-bound) |
| P3 (M4) | IQ3_S/XXS, IQ2_XXS/XS/S/M, IQ1_S/M (codebook grids) | every Unsloth low-bit file needs them; slower decode (5–10 %) is acceptable |
| P4 (M9) | TQ1_0, TQ2_0 | ternary models only if the catalog adopts them |
| Parallel track (M3) | MLX affine 2/3/4/5/6/8-bit, group 32/64/128 | Apple ecosystem files already on users' disks |
| Load-time repack (M4) | AWQ, GPTQ, compressed-tensors 4-bit packings → engine 4-bit tile layout | repacking, not re-quantising, so quality is preserved |

Per-tensor type dispatch is mandatory from M1: a single file mixes Q4_K, Q6_K, Q8_0 and F32 (norms are never
quantised), and Unsloth/EXL3-style recipes vary types per layer. Validation of any quantised path uses KLD against a
BF16 reference on chat-templated held-out data, not perplexity (Report §Read GGUF and safetensors natively).

### 6.4 Tokenizers

The engine implements the GGUF vocabulary models itself (no Python, no external files): SentencePiece
(`llama`-style unigram/BPE with byte fallback), byte-level BPE with the pre-tokenizer regex selected by
`tokenizer.ggml.pre` (llama-bpe, qwen2, gpt-4o/o200k, tekken, gemma, deepseek, chatglm, and the 2026 additions as
they appear in llama.cpp's converter), special-token handling, `add_bos`/`add_eos` flags, and an incremental
detokeniser that holds back incomplete UTF-8 sequences. `tokenizer.json` folders use the `tokenizers` crate.
Vocabularies in the catalog span 100,278 to 262,144 entries; mask-cache memory for grammars is sized from the
actual vocabulary (Section 10.3). Round-trip tests compare against llama.cpp's tokenizer on a pinned corpus per
family.

### 6.5 Chat templates

Templates are rendered with `minijinja` plus the filters and globals the catalog's templates use (`tojson`,
`strftime_now`, `raise_exception`, `items`, string methods, `namespace`). The renderer exposes the variables the
2026 families expect: `tools`, `add_generation_prompt`, `enable_thinking`, `reasoning_content`,
`preserve_thinking`, `truncate_history_thinking`, `clear_thinking`, `bos_token`, `eos_token`. Each template's
content hash maps to a tool-format family (Section 10.2); substring sniffing is forbidden (a MiMo distill was
mis-detected as Qwen3-Coder by substring and its tool calls never completed — Report §Internet and tools).
Templates that *raise* when `arguments` is a string rather than a mapping (Gemma 4, LFM2, Qwen3.5) are the reason
arguments stay JSON objects internally.

### 6.6 Load pipeline

```text
open files → parse headers → ArchSpec → plan (or validate the supervisor's plan hash)
  → map files (mmap; direct I/O for cold discrete-GPU groups)
  → create device views (Metal: shared-mode buffers over the mapping; Vulkan on discrete GPUs: copies into the device-local arena;
    CPU: the mapping itself) → bounded prefault of hot tensors
  → reserve KV arena and scratch from the plan → build and compile graphs for each shape bucket
  → warm-up (one tiny prefill + one decode step; compiles pipeline states, probes Metal 4 tensor path)
  → numerical self-test (reduced graph vs CPU reference; fails loudly on mismatch)
  → measure peak, compare with plan, log → ready
  → background: repack-cache generation for prefill tensors (low priority, cancellable)
```

### 6.7 The model description layer

Model-zoo churn is the main maintenance risk (new families monthly). The engine therefore separates *architecture
description* from *kernels*: `model::ArchSpec` is data (parsed from metadata), and each family is a small graph
builder composed from shared blocks — attention variants (GQA, sliding window, K=V global, MLA absorbed, sinks,
QK-norm, gated output), FFN variants (SwiGLU, clamped SwiGLU, GeGLU, relu²), MoE block (router types, shared
experts, grouped expert GEMM), Gated DeltaNet block, Mamba-2 block, gated short-conv block, per-layer embeddings,
cross-layer KV sharing, μP multipliers, logit soft-capping, hyper-connections, looped layers. Adding a family that
reuses existing blocks is a ~100-line module plus a golden-token fixture; a family needing a new block is a
milestone item. The coverage order follows the report's finding that one kernel set (the Qwen3.5 stack) unlocks 14
of the 37 catalog families.

**Decisions.** GGUF first, safetensors second, no new format; C-struct block sizes; converter-driven metadata;
in-engine tokenizers; `minijinja` templates with hash-based family detection; per-tensor type dispatch from M1;
description layer separated from kernels.
**Evidence.** Report §Read GGUF and safetensors natively (format details, block-size discrepancy, side-car
convention, KLD validation, licence line) and §Target models (operator union, tokenizer families, 2026 conversion
pitfalls).

---

## 7. Compute architecture

### 7.1 Graph IR and execution

- **Static graphs per shape bucket.** The graph for a decode step and for each prefill chunk size is built once,
  planned once (buffer lifetimes), and replayed with new KV positions and block tables. On Vulkan this is a pre-recorded command buffer per shape bucket, re-recorded only when node properties change (the warm-up / compare / update idea ggml uses for CUDA graphs); on Metal one
  command buffer per step with pre-built pipeline states; on CPU a pre-scheduled op list.
- **Fusion passes** (applied before planning): RMSNorm + residual + scale; RoPE + write-to-KV; gate/up GLU for
  dense and expert matmuls; top-k routing (softmax/sigmoid + top-k + renormalise); MoE weighted reduction; SSM
  conv + SiLU; QKV projections into one matmul where the weights are contiguous; sampling on device.
- **Backend scheduler.** Each op runs on the device that owns its weights during decode. During prefill,
  host-resident weights are offloaded to the GPU only when the batch is ≥ 32 tokens (the Metal backend's
  `GGML_OP_OFFLOAD_MIN_BATCH` rule). Activations crossing devices are batched per layer into one transfer.
- **Independent branches** (Q/K/V projections, attention vs. parallel FFN in Falcon-style layers) are graph nodes
  with explicit buffer lifetimes so a backend may run them on separate streams. Concurrent streams are enabled per
  device by the autotune (they gave +43 % on an RTX 5090 and regressed a GB10 — Report §Sharding and scheduling).

### 7.2 Operator set and numeric rules

Appendix C lists the full union with the families that need each op. Rules that are easy to get wrong:

- Attention accumulates in fp32 on CPU and Metal (f16 accumulation overflowed to NaN on Nemotron 3 Nano with
  head_dim 64 and sinks); logits are fp32; soft-capping applied before sampling.
- Every recurrent state (Gated DeltaNet, KDA, Mamba-2, short-conv) is fp32, even with 8-bit KV.
- Norm weights with `(1 + w)` gains (Gemma, MiniMax, Qwen4Exp) are applied exactly once; a test checks that a
  converted model's norm does not fold the `+1` twice.
- RoPE families (default/partial, interleaved mRoPE, YaRN with attention factor, Llama-3 piecewise, LongRoPE,
  linear with per-layer base, NoPE with Llama 4 temperature) are one `RopeSpec` enum consumed by one kernel family
  per backend.

### 7.3 CPU backend

**ISA matrix (priority order).** AVX2+FMA (`vpmaddubsw`) on every x86 machine; AVX-VNNI (Intel Core 12th gen
through Core Ultra 300, which disable AVX-512); AVX-512 VNNI/VBMI (Zen 4/5, Xeon); NEON dotprod + i8mm (every
Apple M-series, Snapdragon X, Cortex-A7xx/X); SME2 for prefill on M4/M5, X2 Elite and Dimensity 9500 (via `asm!`
or FFI to KleidiAI `.S` kernels — SME intrinsics do not exist in Rust); AMX only for Xeon workstations (nightly
intrinsics; `asm!` path, optional). SVE is skipped (no consumer part with 256-bit vectors; llama.cpp's SVE kernels
never run on 128-bit cores). Dispatch is runtime (`is_x86_feature_detected!`, `is_aarch64_feature_detected!`,
`sysctl hw.optional.arm.FEAT_SME2`) with `#[target_feature]` multiversioned kernels; Rust ≥ 1.89 for stable
AVX-512. Note (found in M1): the NEON dot-product intrinsic `vdotq_s32` and the f16 conversion
intrinsics are still nightly-only on Rust 1.92, so the NEON kernels emit `sdot` through stable `asm!`.

**Kernel families per block type.** (1) scalar reference; (2) decode GEMV on the *native* block layout with int8
activations (Q8_K/Q8_0 per row, quantised once per op); (3) prefill GEMM on an *interleaved* tile layout (4×4/4×8
NEON, 8×8 AVX2, 8×4/16×4 VNNI, 16×64 SME2) used when `n_tokens ≥ 8–32`, tiles derived from the register file,
cache blocking only here; (4) dequantise-to-f16/f32 for the few ops that need it. Decode is bandwidth-bound, so
repacking buys ≤ 10 % there; prefill gains are 1.5–4× (Report §CPU execution). Repacked tiles come from the
background cache file (Section 5.4), never from a load-time copy.

**Attention and recurrent kernels.** Flash attention with fp32 accumulation, block-table aware, sinks, sliding and
chunked masks, quantised K/V (tile-wise dequant), head dims 64/128/256 (+192/128 and 576/512 for MLA, 512 for
Gemma 4 global). Chunked Gated DeltaNet scan (conv kernel 4, separate QK/V head counts), Mamba-2 SSD scan with
grouped states, gated short-conv (`conv_L_cache` 3), KDA — all fp32.

**MoE expert path (the biggest unclaimed CPU win).** `mul_mat_id` streams only the top-k selected experts' rows per
token; activation quantisation happens once per layer and is shared across experts; work items are (expert,
row-chunk) pairs over a global queue with work stealing (not one parallel region per expert); the token→expert map
is not rebuilt for single-token decode; the next layer's selected experts are prefetched (`madvise(MADV_WILLNEED)`
on their pages when the model exceeds RAM, prefetch instructions otherwise) using the router output of the current
layer; dense-first tensors (attention, router, shared experts, norms, head) stay on the fastest device when a GPU
exists. Target: ≥ 0.5 × active-bytes roofline where llama.cpp measures 0.25–0.33 (80B-A3B at 7.74 tok/s on a
~65 GB/s laptop against a 20–30 tok/s bound).

**Threading.** Pool = physical performance-class cores (Section 4.2); one parallel region per fused op; static row
partitioning for decode so each thread's weight stream stays contiguous; a global chunk queue for prefill GEMM and
for mixed SME2/NEON execution (static splits lost up to 2×); bounded spin then park; no barrier-per-op executor.
NUMA: first-touch allocation per node, per-node thread groups, no cross-node repack (interleaved repack regressed
7.5–15 % on a dual Xeon).

**Optional accelerators.** KleidiAI (Apache-2.0) SME2 int4 kernels through FFI on M4+ and Snapdragon X, and the engine's own `asm!` SME2 GEMM where KleidiAI has no kernel; T-MAC/bitnet-style LUT kernels (MIT) only if ternary models enter the catalog. Apple's Accelerate/BNNS framework is closed source and is not used, so the undocumented AMX unit on M1–M3 is not reached; SME on M4+ is reachable from open code, which is where the measured prefill gains are anyway.

### 7.4 Metal backend (phase 1 GPU)

Metal is the operating system's GPU API, not a bundled library: the engine reaches it only through the open-source
`objc2-metal` crate and shader source it owns, and nothing proprietary is linked or required at build time. The
all-open-stack alternative on macOS is the Vulkan backend over Mesa's KosmicKrisp (MIT, macOS 26+) or MoltenVK
(Apache-2.0), measured 10–20 % slower on the one cross-API data point; it is kept as a test path and as the fallback
if the maintainers decide not to target Metal at all (Section 15).

- **Buffers.** Weights: `newBufferWithBytesNoCopy` over the mmap in `MTLResourceStorageModeShared`, page-aligned
  views each ≤ `maxBufferLength`, overlapping views so every tensor fits in one view, at most 64 buffers (ggml's
  structure). KV and scratch: one `MTLHeap` placement heap per plan. Residency: `MTLResidencySet` per group,
  keep-alive thread (5 ms while active, 3-minute counter), released on sleep.
- **Budget.** `recommendedMaxWorkingSetSize`; opt-in flag to plan against a raised `iogpu.wired_limit_mb` with the
  command and the swap risk printed; `currentAllocatedSize` sampled into the ledger.
- **Kernels** (`kernels/metal/*.metal`, compiled at runtime by the OS Metal compiler through `newLibraryWithSource` from the embedded source and cached per OS build; an optional `.metallib` precompile exists for release builds but the build never requires Xcode; `MTLCompileOptions.languageVersion` set explicitly —
  the Metal 4 tensor path was silently inert in llama.cpp until this was fixed):
  bandwidth-tuned quantised GEMV per block type (decode); simdgroup-matrix GEMM (prefill); few-row GEMM for 2–16
  rows (speculative verification, small batches, MoE expert batches — the kernels that turned MTP from a 12–45 %
  loss into a 3.4× win); SDPA vector kernel for ≤ 8 queries with a two-pass split-K variant above ~1–8K keys; SDPA
  matrix kernel for prefill; fused RMSNorm/residual, RoPE+KV write, GLU, MoE routing (softmax+top-k), SSM conv+SiLU,
  Gated DeltaNet chunk scan, dequant, argmax/top-k/top-p sampling on device (no logits readback at batch 1).
- **Metal 4 tensor path.** On devices whose capability probe *and* a test-compile of a small f16/bf16 tensor kernel
  succeed (M5/A19 and later), prefill GEMM, MoE expert GEMM and prefill attention use `MTLTensor`/`matmul2d`
  cooperative-tensor kernels; decode stays on GEMV (measured: 2–4× prefill, ~0 % decode). Pipeline-state creation
  cost (1–2 s per tensor kernel) is paid once at warm-up and cached.
- **Command model.** One command queue; one command buffer per step; the inference thread encodes; indirect command
  buffers are deferred. Capability flags from `supportsFamily:`; startup self-test against the CPU reference.
- **Binding.** `objc2-metal` 0.3.x (the `metal` crate is deprecated). Resource and synchronisation calls are
  `unsafe`; they are wrapped in a small safe layer with debug assertions.

### 7.5 NVIDIA, AMD and Intel GPUs: Vulkan only (no CUDA)

CUDA is a proprietary SDK and is excluded by the open-source-only rule. Every non-Apple GPU is served by the
Vulkan backend below, which runs on the vendor driver the OS ships or on open-source drivers (Mesa RADV, ANV, NVK)
and needs no vendor toolkit at build time. What this costs, with the measured basis: on an RTX 5090 llama.cpp's
Vulkan backend reaches ~79 % of CUDA prefill and ~91 % of CUDA decode (community-measured); native FP4
tensor-core prefill (+43–68 % on NVFP4 files) and CUDA-graph launch savings (≤ 1.2×) are not available, and the
hot-expert GPU cache was measured under CUDA but the technique is backend-neutral. Decode, the part users feel at
batch 1, is bandwidth-bound and therefore nearly unaffected. ROCm/HIP (open source, MIT/Apache) is a possible
later prefill path for AMD if a measurement shows ≥ 1.2× over Vulkan on the same device; it is not a v1 backend.

### 7.6 Vulkan backend (phase 2 GPU, the single path for every non-Apple GPU)

- `ash` (not `wgpu`, which hides cooperative matrices and f16/int8 dot features); GLSL compiled to SPIR-V at build
  time; runtime probing of `VK_KHR_cooperative_matrix`, `VK_NV_cooperative_matrix2`,
  `VK_KHR_shader_integer_dot_product`, 16-bit storage, `VK_EXT_memory_budget`.
- Three matmul paths (scalar, KHR coopmat, NV coopmat2) with per-format dequant callbacks; indirected MoE loads;
  flash attention with a coopmat path and a scalar path.
- Robustness: per-device feature allow/deny list (shipped file plus user overrides), CLI/env switches to disable
  coopmat and flash attention, numerical self-test at startup, first-run autotune cached per (device, driver).
- Coverage: NVIDIA (within ~79 % prefill / ~91 % decode of CUDA on a 5090), AMD RDNA 3/4 and Strix Halo (Vulkan
  leads decode, ROCm leads prefill), Intel Arc A/B and integrated GPUs; MoltenVK/KosmicKrisp on macOS only as a
  portability test path.
- ROCm/HIP and SYCL are not separate backends in v1; the Vulkan backend covers their devices with one kernel set.

### 7.7 NPU delegate boundary

```rust
pub trait NpuDelegate: Send {
    fn capabilities(&self) -> NpuCaps;            // dtypes, max prompt, max model size, address-space limit
    fn plan(&self, spec: &ArchSpec, budget: Bytes) -> Result<NpuPlan>;
    fn load(&mut self, model: &ModelFiles, plan: &NpuPlan) -> Result<()>;
    fn prefill(&mut self, tokens: &[Token]) -> Result<Logits>;
    fn decode(&mut self, token: Token) -> Result<Logits>;
    fn embed(&mut self, tokens: &[Token]) -> Result<Vec<f32>>;
}
```

Implementations wrap open-source runtimes only: OpenVINO GenAI (Apache-2.0; Intel NPUs, with Intel's MIT Linux NPU
driver) ships in v1 behind a cargo feature. Qualcomm Hexagon (QAIRT/GenieX: closed binaries, redistribution terms
unverified), AMD XDNA2 (Ryzen AI software and FastFlowLM kernels are closed; the open mlir-aie/IRON toolchain still
needs a licensed `xchesscc`) and the Apple Neural Engine (Core ML only) are not targeted until an open-source runtime
exists; the delegate trait is where they would plug in. The delegate is a budgeted, one-model-at-a-time device used, in priority order, for
embeddings, ≤ 2B draft models, prefill offload of 4–8B models on laptops, and full decode only for ≤ 2B assistants
in battery mode (measured 8B decode on NPUs is 8–12 tok/s against 2–4× that on the same machine's iGPU). The
Apple Neural Engine is not targeted (Core ML only, 32 MB SRAM working set).

### 7.8 Backend selection and autotune

`llmario-engine probe` enumerates devices, ISA features, driver versions, OS budgets and the kernel matrix; runs a
30-second microbenchmark (GEMV per block type, GEMM, attention at three KV lengths, host↔device bandwidth) on first
run; caches results under `$LLMARIO_HOME/cache/autotune/<device-id>-<driver>-<engine-version>.json`. The planner
reads the cache to choose per-op backends and to compute the speed-of-light. Users override with `--device`,
`--no-coopmat`, `--no-tensor-path`, `--cpu-threads`.

### 7.9 Kernel authoring and testing rules

- Kernels live under `crates/engine/kernels/{cpu,metal,vulkan}/`; each has a scalar Rust reference in
  `testkit`; every (kernel × dtype × shape class) is tested against the reference with per-dtype tolerances, plus
  property tests on random block data and fuzzing of the block decoders.
- Kernel source hashes feed every cache key (PTX cache, repack cache, autotune).
- No Rust-native GPU kernel language in v1 (CubeCL, rust-gpu are tracked, not depended on).
- Unsafe code is confined to `cpu::simd`, backend FFI layers and the mmap layer, each with a `SAFETY` comment and a
  Miri/ASan job in CI for the CPU paths.

**Decisions.** Static per-shape graphs with explicit lifetimes; ops run where their bytes live; int8-dot CPU kernels
with native-layout GEMV and cached interleaved GEMM; rewritten MoE path; Metal → Vulkan order with no CUDA (proprietary); NPUs behind a
delegate; native kernel languages; autotune cached per device.
**Evidence.** Report §CPU execution (ISA matrix, repack gains, barrier losses, MoE shortfall, Rust reachability),
§GPU and NPU backends (Metal buffer/residency/tensor-path structure, BaseRT decode/prefill split, graph-replay and
arena patterns, Vulkan parity and bugs, NPU measurements, kernel-authoring options).

---

## 8. KV cache and attention

### 8.1 Three cache classes

Every layer declares its cache class in `ArchSpec`; the planner and the arena treat them separately.

| Class | Used by | Per-sequence cost | Storage |
|---|---|---|---|
| FullAttention | global attention layers (all dense GQA models; Gemma 4 global layers with K=V and head_dim 512; gpt-oss full layers; MLA layers store the 576-wide latent, never the expanded heads) | `2 · n_kv_heads · head_dim · bytes` per token per layer (`1 ·` for K=V layers; `(d_latent + d_rope) · bytes` for MLA) | paged blocks |
| Window | sliding/chunked layers (Gemma 3/4 local 512–1,024; gpt-oss alternating 128; Mistral/Llama 4 chunked 8,192) | ring of `window + n_batch` tokens | fixed ring per sequence, allocated from the arena in whole blocks |
| Recurrent | Gated DeltaNet (Qwen3.5 stack), KDA, Mamba-2 (Nemotron 3/3.5, Granite 4.0-H, Falcon-H1), gated short-conv (LFM2/2.5) | fixed fp32 state per sequence (≈ 24–160 MB for the 2026 hybrid class), paid per *concurrent sequence*, not per token | separate pool with its own block size (never inflate the attention block size to fit SSM state, the vLLM single-page-size trap) |

Cross-layer KV sharing (Gemma 4 E-series: 18–20 layers read another layer's cache) is a pointer in the block
table, not a copy. Appendix A has the per-token numbers for 13 catalog models.

### 8.2 Paged arena

- Reserved once per device at admission (Section 5.4); carved into fixed blocks of 32 tokens (CPU, Metal) or 64
  (Vulkan on discrete GPUs), configurable, always ≥ the attention kernel's tile; each block holds K and V (or latent) for one layer
  group; per-sequence block tables per layer group; reference counts for sharing; an LRU free queue; freed by
  index; no defragmentation needed because blocks are fixed-size.
- Layer groups: layers with identical cache class and shape share one block pool (vLLM's hybrid manager idea);
  window layers keep their rings in whole blocks so prefix-cache eviction can free them.
- Capacity accounting is exact: `blocks_free × block_bytes` is what admission checks; growth by reallocation is
  forbidden (the MLX pool retention and llama.cpp's contiguous `n_ctx × n_seq_max` buffers are the two failure
  modes this avoids; a paged draft in llama.cpp went from 26 to 247 concurrent sequences on an A10G at ≈ 3 % cost).

### 8.3 KV precision policy

| Setting | Default | Guard |
|---|---|---|
| f16 keys and values | when the plan fits | — |
| q8_0 keys and values | default on 8/16 GB machines and whenever the ladder reaches step 4 | near-lossless (Qwen2.5-7B PPL 7.9605 → 7.9940, KLD 0.0018; measured) |
| q8_0 keys, q4_0 values | allowed by the ladder | values are nearly free at 4 bits (1 of 500 ARC answers changed) |
| q4_0 keys | never automatic | keys collapse some models (Qwen2.5-7B PPL → 1,561); opt-in only, and the engine runs a 30-second KLD/answer-churn check against q8_0 on load and refuses if it fails |
| FP8 keys/values | deferred: needs 8-bit float storage in the Vulkan driver (unverified) | vLLM's finding: "FP8 is the best default" |
| int8 recurrent state | deferred | SGLang/Quamba2 do it; fp32 until measured |

Block Hadamard rotation (64-wide) is applied to Q, K and V before caching on all models except MLA (llama.cpp's
`attn-rot`, merged 2026-04-01: gpt-oss-20b AIME25 under Q4_0 KV 2.0 % → 21.7 %, decode cost 0.88–0.93×). Keys and
values may have different types on every backend; the kernel matrix (head dims × K type × V type × mask/sink flags
× block table) is explicit per backend and an unsupported entry is a startup error, never a silent CPU fallback
(llama.cpp's CUDA path fell back to CPU attention at 25× slower prefill without `GGML_CUDA_FA_ALL_QUANTS`). Sub-4-bit
codebook schemes (TurboQuant class) are not implemented: vLLM's measured study found 4-bit ≈ 96 % and 3-bit −20
points with 10–68 % latency cost, and llama.cpp closed its TurboQuant PR as no better than same-bpw formats.

### 8.4 Attention kernels

Block-table aware from the first version on every backend; GQA with head-group broadcasting; learned sinks
(gpt-oss) in the softmax denominator; sliding and chunked masks; QK-norm before RoPE where the family requires;
MLA absorbed attention (576/512 dims); decode kernel for ≤ 8 queries with split-K above ~1–8K keys; prefill
matrix kernel with chunked prefill (chunks of 512–2,048 tokens, decodes scheduled before prefill chunks); fp32
accumulation (Section 7.2). Sparse-attention kernels (DeepSeek NSA/DSA, MInference) are out of scope; DSA models,
if admitted, run with the mask-based path and their indexer cache counted.

### 8.5 Prefix cache

Blocks are hashed in a chain: `SHA-256(parent_hash ‖ block_tokens ‖ extra)` where `extra` = model digest, KV
types, RoPE/YaRN parameters, chat-template hash, and a per-client salt derived from the API key (so two API keys
never share blocks — CVE-2025-46570 gave an attacker AUC 0.99 at an 8-token prefix on shared caches). Full blocks
only; LRU over free blocks with reference counts; enabled by default (vLLM measures < 1 % throughput loss at 0 %
hit rate); hit/miss/evict counters in `/engine/stats`. A radix tree for token-granular reuse is deferred.

### 8.6 Checkpoints and tiers

Window and recurrent caches cannot be partially trimmed (`seq_rm` fails on hybrids; llama-server forces full
re-processing on Qwen3.5 turns, with ≈ 8-minute stalls reported). The engine therefore treats a **checkpoint** —
the window rings, the recurrent state and the block-chain head at a chat boundary — as a first-class budgeted
object: taken at assistant-turn ends, limited by count and bytes per slot, restored on the next turn of the same
conversation. Whole prompt states tier to RAM (byte limit in the plan) and to SSD (`$LLMARIO_HOME/cache/kv/`,
files 0600 with a validating header carrying model digest, KV types, engine version and salt scope). SSD restore
pays off whenever `bytes / disk_bandwidth < prefill_time`, true on most consumer GPUs and all CPU-only runs at
≥ 16K tokens (a 27B at 100K context went from > 1 minute to ≈ 0.2 s per message with slot restore; 1–4.4 GB per
session). Persistence is a setting (default: RAM tier on, SSD tier on for the desktop app, off for `serve` unless
enabled) and plaintext KV never leaves the app data directory (KV files are invertible to prompts).

### 8.7 Context handling

- Context shifting (sink + window truncation) is opt-in, cuts at chat-message boundaries, and is unavailable for
  window-cache models (recompute from the nearest checkpoint instead).
- YaRN/RoPE scaling factor is a per-request parameter with a server-side cap (Qwen's card: factor 2.0 for ≈ 65K,
  none below 32K; static factors degrade short prompts); because cached keys are post-RoPE, the factor is part of
  the prefix-cache key.
- Context length per request is enforced by the gateway (existing) and by the engine (admission).

**Decisions.** Three cache classes; fixed-block paged arena per device; q8_0 default with Hadamard rotation and a
hard guard on 4-bit keys; explicit kernel matrix; SHA-256 chained, salted prefix cache; checkpoints as budgeted
objects with RAM/SSD tiers; opt-in truncation; per-request YaRN in the key.
**Evidence.** Report §KV cache (per-token accounting, quantisation measurements, paging comparison, eviction
research vs. shipped tiering, CVE and inversion results, RoPE interface).

---

## 9. Scheduling, placement and parallelism

### 9.1 Step loop and continuous batching

- **Slots** come from the LLMario profile (`latency` 1, `balanced` 4, `throughput` 16) and share one KV arena per
  device. A slot owns a block table per layer group, a recurrent-state slot, a sampler state, a grammar state and a
  speculative state.
- **Token-budget steps.** Each step schedules up to `n_ubatch` tokens (512 default; 2,048 for `throughput`):
  all slots with a pending decode token first, then prefill chunks of the oldest admitted prompts until the budget
  is spent (vLLM V1's "does not distinguish prefill from decode" budget with decode-first ordering; llama-server's
  `-b 2048 -ub 512` shape). One graph execution per step for the matching shape bucket.
- **Sampling** runs on the device for batch-1 greedy/top-k/top-p (no logits readback); otherwise logits for the
  batch come back once. Grammar masks (Section 10.3) are applied per slot before sampling; penalties, min-p, seed
  and temperature follow the existing API semantics.
- **Events** are emitted per slot per step; the SSE writer runs on the tokio runtime.

### 9.2 Admission, preemption and priorities

- Per-request admission: blocks needed = ⌈(prompt − cached_prefix + max_tokens) / block⌉ × conservativeness
  (default 1.0; SGLang's knob exists because clients over-declare `max_tokens`). If the arena cannot satisfy it the
  request waits (bounded by `queue_timeout_secs`) or, for a higher-priority request, the lowest-priority running
  slot is **retracted and later recomputed** from its prefix-cache blocks. KV is never swapped to host memory
  (vLLM removed swapping; SGLang retracts).
- Priorities: interactive (desktop app) > API default > batch; round-robin within a class; starvation bounded by
  age.
- Per-request context is enforced twice (gateway and engine); a request exceeding the admitted context is refused
  with the plan's limit in the error.

### 9.3 Parallelism inside one device

| Where | Mechanism | Guard against resource issues |
|---|---|---|
| CPU matmul/GEMV | row partitions across the P-core pool; (expert, chunk) work stealing for MoE; global chunk queue for prefill | pool size = physical performance cores; bounded spin; parked when a GPU owns the step |
| CPU ↔ GPU during MoE decode | host experts computed on CPU while the GPU runs attention of the *same* step only where the graph exposes independence (KTransformers' expert deferral, up to 1.45×); otherwise sequential | explicit dependency edges; the inference thread remains the only issuer |
| GPU streams | independent graph branches (Q/K/V, parallel FFN) on 2–3 streams when the autotune shows a gain | per-device allow list from the autotune (regressed on GB10); explicit buffer lifetimes prevent the allocator-reuse race |
| Launch overhead | pre-recorded Vulkan command buffers; Metal single command buffer per step; pre-built pipeline states | graphs disabled around host-sync ops |
| Transfers | double-buffered, per-layer batched activation transfers for host-resident experts | one transfer per layer, counted in the plan's scratch |

### 9.4 Placement across devices

The planner assigns each tensor group to a device before load, deterministically from metadata and the autotune:

1. Embeddings, attention projections, KV, routers, shared experts, norms and the output head go to the fastest
   device that fits them.
2. Routed experts go to the fastest device while they fit; otherwise to host RAM, with a **hot-expert cache** on the
   GPU sized at ~10 % of expert bytes (LRU by expert, filled for batches ≤ 32 tokens — llama.cpp PR #29887
   measured 25.0 → 40.7 tok/s on an RTX 4090 and 30.8 → 67.8 on a 5090 for a 93.7 GiB MoE with 77–89 % hit rates).
3. Dense layers spill to host from the input side only after all routed experts have spilled.
4. During decode every op runs where its bytes live; during prefill, host-resident weights are offloaded to the
   GPU at batches ≥ 32.
5. On unified-memory systems (Apple, Strix Halo, DGX Spark) "host" and "device" are the same bytes; placement then
   chooses the *compute unit*: tiny models (≤ 1B) decode faster on the CPU than on Metal (M4 Pro gemma3-1B: CPU
   161.5 vs GPU 147.8 tok/s; community), and the NPU delegate may take prefill for 4–8B models on laptops.
6. Users can pin with `--place '<tensor regex>=<device>'` (llama.cpp's `-ot` idiom); the plan prints the result.

What is *not* done: CPU–GPU tensor parallelism inside one machine (no mainstream system does it; PCIe round trips
per layer at batch 1 are the same "sequential and latency-bound" regime that costs 43–47 % of decode over 10 GbE).

### 9.5 Multi-device interfaces (designed, not built)

```rust
pub trait Device: Send + Sync {
    fn id(&self) -> DeviceId;
    fn kind(&self) -> DeviceKind;               // Cpu, Metal, Vulkan, Npu, Remote
    fn memory(&self) -> MemoryDescriptor;       // ceiling, headroom, wired cap, page size
    fn bandwidth(&self) -> BandwidthDescriptor; // measured GB/s (autotune), host<->device GB/s
    fn latency(&self) -> LatencyDescriptor;     // per-dispatch and per-transfer microseconds (autotune)
    fn alloc_arena(&self, bytes: u64) -> Result<Arena>;
    fn kernels(&self) -> &dyn KernelSet;
}
pub trait Transport: Send + Sync {
    fn send(&self, dst: DeviceId, t: &TensorView) -> Result<()>;
    fn recv(&self, src: DeviceId, t: &mut TensorView) -> Result<()>;
    fn all_reduce(&self, group: &[DeviceId], t: &mut TensorView) -> Result<()> { Err(Unsupported) }
}
```

A remote device is just another `Device` whose descriptors make the placement solver avoid it unless capacity
demands it (prima.cpp's memory-aware assignment is the model: 70B from 10,120 ms/token swapping to 674 ms/token).
Candidate transports later: llama.cpp RPC interoperability, RDMA over Thunderbolt 5 on macOS 26 (tensor
parallelism measured ~1.5–1.7× on four Mac Studios; ~100 memory-region limit), 10 GbE pipeline for capacity only.

### 9.6 Speculative decoding

In strict order of evidence, each stage auto-disables when it does not pay:

| Stage | Default | Measured basis |
|---|---|---|
| N-gram / suffix lookup over the prompt and prior outputs | on; off below ~30 % acceptance | no weights; −17 % tg at 32 % acceptance in one long-context test; strongest on agentic loops that repeat tool output |
| Model-shipped MTP heads (Qwen3.5/3.6/3.8, Qwen3-Next, GLM, Nemotron 3.5, Gemma 4's drafter), side-car `mtp-*.gguf` or in-file | on for dense targets, `n_draft` 2–3 | RTX 3090 Qwen3.8-27B: 41.6 → 66.4 tok/s (+59.8 %); loss on Metal until few-row kernels existed |
| DFlash2 / EAGLE-3 drafters (SpecForge, MIT) | opt-in, dense targets | +51.9 % on the 3090; 30.2 → 110.0 tok/s on code on an M3 Ultra with few-row kernels (3.4×) |
| Classic draft models | opt-in only | −29.8 % with a 0.8B drafter on the 3090 |
| Any speculation on ≤ 3B-active MoE targets | off | every configuration lost on Qwen3.6-35B-A3B |

Requirements: few-row (2–16) GEMM kernels on every backend (Section 7); drafter weights and verification buffers
("a few hundred MB" plus ≈ 500 MB) in the plan; a greedy-equivalence test — with temperature 0 the speculative
stream must reproduce the serial stream token-for-token on the fixture set (one 2026 study saw divergence in
76–80 % of requests); acceptance rate and tokens-per-step in `/engine/stats`.

**Decisions.** Token-budget steps with decode-first chunked prefill; retract-and-recompute; placement rules 1–6;
no in-machine tensor parallelism; `Device`/`Transport` traits only; the speculative ladder above.
**Evidence.** Report §Sharding and scheduling (offload and expert-cache measurements, DGX Spark TP/PP numbers,
process-model survey, scheduler comparison, distributed results, speculative table, incident catalogue).

---

## 10. Internet connectivity: tools, MCP, agent loop and security

### 10.1 Canonical transcript

The engine's transcript is the OpenAI Responses item model: `message`, `reasoning`, `function_call`,
`function_call_output`, `web_search_call`, `mcp_call`, `mcp_approval_request`, `mcp_approval_response`, with
annotations (`url_citation`). Chat Completions is a lossy projection (tool calls become `tool_calls`, reasoning
becomes `reasoning_content`). Reasoning items are passed back during a tool loop according to each family's
policy: Qwen3.5 re-renders thinking only after the last user query; gpt-oss's `analysis` channel is passed back
within the tool loop and dropped once the turn ends in `final` (and never shown to end users); Gemma 4 uses the
`<|channel>thought` channel. The desktop app's collapsible "Thinking" section keeps working through the projection.

### 10.2 Tool-call formats and parsers

Appendix D tabulates the eight wire families verified from the models' own chat templates. The engine ships one
incremental parser per family with a `NEED_MORE_INPUT` state and an arguments buffer that delays emission until
the tool name is known (llama.cpp's PEG-parser design), an auto-parser that infers the family by differential
rendering of the template (`JSON_NATIVE | TAG_WITH_JSON | TAG_WITH_TAGGED | PYTHONIC`) as a fallback, arguments
kept as `serde_json::Value` objects internally, a `tool_call_id → name` map carried in the transcript, and
template detection by content hash. Each family has a pinned corpus of rendered prompts and expected parses
(streaming and non-streaming) in `testkit`. Reliability below ~4B parameters is poor on every source (self-reported
BFCL-V4: Qwen3.5-4B 0.503, 2B 0.436, 0.8B 0.253) and the benchmark itself was rated "Flawed" (24 of 50 sampled
tasks defective), so the engine assumes malformed calls: validate against the tool's JSON Schema, retry once with
a repair prompt, default small models to single-call mode with grammar enforcement, and allow parallel calls only
when the template advertises them.

### 10.3 Constrained decoding

`llguidance` (MIT, Rust-native, the backend inside llama.cpp, vLLM, SGLang, mistral.rs and OpenAI) is embedded:
no precomputation, ~50 µs of CPU per token on a 128k tokenizer (authors' measurement), startup ~2 ms. Grammars are
**lazy**: unconstrained until the reasoning end token or a tool-call opener triggers, constrained thereafter, so
"think then JSON" works for every family. Compiled grammars are cached by (tool-set hash, tokenizer hash,
format family) — llama.cpp's own GBNF build fails around 60 tools, which MCP-heavy sessions reach. `tool_choice`
(`none`, `auto`, `required`, a specific function) and `response_format: {type: "json_schema", strict: true}` are
enforced by grammar, never advisory. Mask-cache memory is sized from the actual vocabulary (up to 262,144 entries)
and counted in `runtime_fixed`. XGrammar-2 structural tags are a later option behind the same interface.

### 10.4 MCP host

- `rmcp` (Apache-2.0) implementing revision **2026-07-28** with 2025-11-25 compatibility: no protocol session or
  `initialize` handshake, version and capabilities in `_meta` on every request, `server/discover`, Multi
  Round-Trip Requests (`input_required` → surfaced to the client as an elicitation/approval event), Tasks extension
  optional; HTTP+SSE and Dynamic Client Registration (deprecated) not implemented.
- Transports: stdio child processes (launched under the sandbox of Section 10.7 with the exact command shown
  untruncated) and Streamable HTTP with OAuth; OAuth metadata fetches obey the SSRF rules (private, loopback,
  link-local, CGNAT and metadata ranges blocked; DNS pinned; egress proxy).
- Configuration: `$LLMARIO_HOME/mcp.toml` — per server: command or URL, environment allowlist, and a permission
  profile per tool (`allow` / `ask` / `deny`, default `ask`); tool classification is engine-owned (Section 10.7);
  `readOnlyHint`/`destructiveHint` are read but never trusted; `structuredContent` validated against
  `outputSchema`; timeouts per call; `isError` results fed back to the model; every call audited.

### 10.5 Built-in tools

| Tool | Design |
|---|---|
| `web_search` | provider adapters behind one result schema (title, url, snippet, published, source): self-hosted SearXNG (AGPL-3.0, called over HTTP as a service, so its licence does not reach the engine; the JSON format must be enabled on the instance) and a generic JSON/OpenSearch endpoint adapter for any other self-hosted engine (for example YaCy, or a Meilisearch/OpenSearch index over the user's own documents). Proprietary hosted APIs (Brave, Exa, Ollama's hosted search) are not integrated; a user who wants one can expose it through an MCP server of their own. Default: none configured → tool absent; the desktop app offers a SearXNG setup panel (instance URL). Max 10 results, each truncated to 8,000 characters before windowing. |
| `web_fetch` | SSRF guard → fetch → extract → page model. Guard: `http`/`https` on 80/443 (plus user-allowed ports); block RFC1918, loopback, link-local, CGNAT (100.64/10), cloud-metadata ranges, `::1`, `fc00::/7`, `fe80::/10` and IPv4-mapped IPv6 forms; resolve, validate every address, connect only to validated addresses (DNS pinning), re-validate every redirect; text content types only; 600 KB cap and 20 s deadline by default; `texting_robots` for `robots.txt` (on by default, user-overridable); per-provider rate limits. Extract: `dom_smoothie` (Readability) → `htmd` (Markdown); strip hidden/off-screen/zero-font/transparent text and invisible Unicode. Page model (gpt-oss `simple_browser`): 80-column wrap, numbered `L{i}:` lines, 1,024-token view windows, `find` within page, link ids `【id†title†domain】`, per-session page cache, citations `【cursor†L{a}-L{b}】` rewritten to `url_citation` annotations, "do not quote more than 10 words" instruction in the tool description. Headless Chromium (`chromiumoxide`) for SPAs is deferred (enlarges the attack surface). |
| `retrieve` | a local index over fetched pages and user-added documents: Markdown-aware chunks of 300–800 tokens with URL/title/heading path prepended (contextual retrieval), embeddings from the engine's own GGUF path (Qwen3-Embedding-0.6B, EmbeddingGemma 2 270M; Apache-2.0) or `fastembed-rs`, hybrid BM25 + vector search in `sqlite-vec` under the app directory, optional reranker. |
| `code_exec` | not in v1 (M10 candidate; requires the full sandbox story and its own policy class) |

Built-in tools are exposed to models as ordinary function tools and to API clients as Responses built-in tool
types (`{"type":"web_search"}`, `{"type":"web_fetch"}`, `{"type":"retrieve"}`, `{"type":"mcp","server_label":…}`).

### 10.6 Agent loop

```text
render(transcript, tools, thinking policy) → generate(lazy grammar) → parse incrementally
  → validate arguments against JSON Schema (one repair retry) → POLICY GATE
  → execute (built-in or MCP; parallel for independent calls when the template advertises parallel calls;
    timeouts; concurrency cap) → window/truncate results (8,000 chars or 1,024-token windows; provenance-tagged)
  → append function_call_output items → repeat
until: no tool calls, or max_steps, or max_tool_tokens, or reasoning budget, or wall-clock budget, or client cancel
```

Budgets are API parameters (`max_steps`, `max_tool_tokens`, `reasoning: {budget_tokens}`, `max_output_tokens`)
because small models loop. Streaming emits Responses events (`response.output_item.added`,
`response.function_call_arguments.delta`, `response.mcp_approval_request`, …) and the Chat Completions projection
streams `tool_calls` deltas.

### 10.7 Security model

- **Classification** (engine-owned, per tool): `read_only_local` (retrieve over the user's own index),
  `open_world_read` (search, fetch), `state_changing` (writes anywhere), `exfiltration_capable` (any tool that can
  send data out, including fetch with query strings derived from context — treated as exfiltration-capable once
  the session is tainted).
- **Session taint**: set when any untrusted content (web page, search result, MCP tool output from a server not
  marked trusted) enters the context.
- **Rule of Two gate**: a session may satisfy at most two of {processes untrusted input, has access to sensitive
  data or systems, can change state or communicate externally} without a human; once tainted, every
  `state_changing` or `exfiltration_capable` call blocks on an approval event (desktop dialog showing the full
  arguments; API clients receive an `mcp_approval_request` item and answer with `mcp_approval_response`). The
  default browsing profile is search + fetch + read-only retrieval only.
- **Provenance**: every tool result is wrapped in a tagged block with a random nonce suffix so page content cannot
  close the block, and carries `source`, `url`, `fetched_at`; the template renders it in the family's tool-result
  position.
- **Injection heuristics** (hidden-text stripping, pattern logging) are advisory and logged, never the defence
  (a 95 % detector is "a failing grade").
- **Sandbox** for the fetcher and stdio MCP servers: macOS Seatbelt profile (`sandbox_init`; filesystem read-only
  except a scratch dir; network only to the local egress proxy socket); Linux bubblewrap + seccomp with Landlock as
  fallback; Windows AppContainer or restricted token + job object + firewall rule limiting egress to the proxy.
  **Egress proxy**: a local proxy (Unix socket / loopback with a per-session token) enforcing a domain allowlist;
  new domains prompt the user (Anthropic's sandbox design reported 84 % fewer permission prompts with this
  combination). Both layers are required: without network isolation a compromised fetcher can exfiltrate keys.
- **Secrets**: provider keys in the OS keychain (Keychain / Credential Manager / Secret Service), never in prompts,
  logs or config files; environment passed to MCP servers is an explicit allowlist.
- **API**: loopback token and `Origin`/`Host` validation (existing gateway behaviour; MCP's HTTP transport returns
  403 on invalid `Origin`); JSON bodies only.
- **Audit**: tool name, argument hash, classification, decision, duration, bytes in/out, provider — never prompt or
  page content; OpenTelemetry GenAI spans with message content Opt-In only.

**Decisions.** Responses item model as the transcript; eight parser families + auto-parser; `llguidance` lazy
grammars with a compiled cache; `rmcp` on 2026-07-28; provider adapters for search; a hardened fetcher with the
gpt-oss page model; the Rule-of-Two gate; two-layer sandbox with egress allowlist; keychain secrets; content-free
audit.
**Evidence.** Report §Internet and tools (template survey, BFCL audit, llguidance measurements, MCP changelog and
normative duties, provider terms, `simple_browser` design, safewebfetch controls, embedder and store options) and
§Security (lethal trifecta, Rule of Two, CaMeL, OWASP LLM01, Anthropic sandbox, CVE and OTel guidance).

---

## 11. API surface, observability and benchmarking

### 11.1 Endpoints and fields

| Endpoint | Supported | Notes |
|---|---|---|
| `POST /v1/responses` | input items, instructions, `tools` (function + built-in + mcp), `tool_choice`, `parallel_tool_calls`, `text.format` (text / json_schema strict), `reasoning` (`effort`, `budget_tokens`), `max_output_tokens`, `max_steps`, `max_tool_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `seed`, penalties, `stream`, `store: false` only, `previous_response_id` (in-memory, session-scoped) | native path |
| `POST /v1/chat/completions` | existing fields plus `tools`, `tool_choice`, `parallel_tool_calls`, `response_format` (json_object, json_schema), `reasoning_effort`, `stream_options.include_usage`, `logprobs`/`top_logprobs` ≤ 20 | projection of the Responses path |
| `POST /v1/embeddings` | text input, batching, `dimensions` for Matryoshka models | embedding models from the catalog |
| `GET /v1/models`, `/v1/models/{id}` | existing fields plus `engine.plan_hash`, capabilities (tools, json_schema, vision=false) | |
| `GET /engine/plan`, `/engine/ledger`, `/engine/stats`, `POST /engine/control`, `GET /healthz`, `GET /metrics` | Section 3.2 | |
| Not supported in v1 (clear 400) | images/audio input, `n > 1`, `store: true`, file search over hosted stores, computer use | |

The gateway's allowlist (`crates/api/validate.rs`) is extended for the new fields; `model` is still always rewritten.

### 11.2 Observability

- **Load log** (also `GET /engine/plan`): the plan table per device, the speed-of-light per device
  (`η=1` bandwidth bound and the autotuned expected η), the kernel matrix in use (block types × backends, KV
  pairs), the backend/driver versions, self-test results, residency and wired limits, autotune source and age.
- **Per request**: TTFT, prompt tokens, cached prefix tokens, prefill tok/s, decode tok/s, achieved GB/s versus
  speed-of-light, speculative acceptance, preemptions/retractions, degradation events, tool calls and policy
  decisions (hashed).
- **Per step**: tokens scheduled, device time, host↔device bytes, shared-GPU-memory growth on Windows.
- **Exports**: Prometheus `/metrics` (ledger gauges, histograms for TTFT/duration/tok-s, counters for cache
  hits/evictions, acceptance, retractions, approvals); structured JSON logs that never contain prompt or page
  content (`-v` / `--log-json` conventions unchanged); optional OpenTelemetry with content Opt-In.
- `llmario doctor` prints the engine capability report and both ledgers.

### 11.3 Benchmarking

- `llmario bench` keeps working unchanged (it speaks OpenAI HTTP) and gains columns: planned vs measured peak,
  speed-of-light ratio, achieved GB/s, acceptance rate; `--soak <hours>` runs mixed traffic and reports footprint
  drift; the existing rules in `docs/BENCHMARK_PLAN.md` (cold prefix cache by default, exact suite prompts for
  quality) apply.
- `llmario-engine bench kernels` runs the autotune suite and prints achieved bandwidth and matrix throughput per
  kernel against the device's theoretical peak.
- `llmario-engine verify model --reference <bf16 model>` computes KLD on a chat-templated held-out set for a
  quantised file (the catalog's quality label source).
- CI gates: mock and small CPU models on GitHub runners (Linux, Windows); the maintainers' M4 Max runs the Metal
  suite before every release (manual or self-hosted runner); results land under `benchmarks/results/<date>-…` as
  today.

---

## 12. Testing and verification strategy

| Layer | Tests |
|---|---|
| Block decoders and kernels | every (type × backend × ISA) against the scalar reference with per-type tolerances; property tests on random blocks; fuzzing of GGUF/safetensors parsers and block decoders; Miri/ASan on CPU `unsafe` paths |
| Tokenizers and templates | round-trip corpora per family against llama.cpp's tokenizer output; rendered prompts against `apply_chat_template` fixtures pinned to template hashes |
| Parsers and grammars | the eight-family corpus (streaming at every byte boundary and whole); fuzz: tokens sampled under a grammar always decode to schema-valid JSON; `tool_choice` enforcement |
| Planner and allocator | determinism (same inputs → same plan hash); simulated sizes equal real allocation sizes; ladder steps reproduce documented fixtures; refusal messages name the limiting device |
| Golden fidelity | per family: greedy 64-token outputs identical across CPU/Metal/Vulkan at f16 KV; KLD ≤ 0.01 vs the CPU f32 reference per backend; the `(1+w)` norm, sinks, MLA, GDN/Mamba state, sliding-window and K=V paths each have a dedicated fixture |
| Memory | plan bound and tightness for every model in the fit matrix (CPU small models in CI; the Mac suite before release); 8-hour soak with < 1 % drift; sleep/wake returns to baseline; residency keep-alive measured (no idle cliff) |
| Isolation | multi-slot cross-contamination test for recurrent-state models (the bug class that yanked mlx-lm 0.31.0 and llama.cpp #29002); salted prefix cache never shares across API keys; checkpoint restore equals recompute |
| Speculation | greedy equivalence on the fixture set; acceptance auto-disable thresholds |
| Process | kill −9 mid-stream → gateway error → relaunch within the existing 3-per-10-minutes policy; exit protocol; socket permissions; Windows named-pipe DACL |
| Security | SSRF corpus (every blocked range and encoding, redirect chains, DNS rebinding); sandbox probes (fetcher attempting to read `~/.ssh`, to connect past the proxy, to spawn processes); injection corpus asserting that no `state_changing` tool executes without an approval event; audit log contains no content |
| Performance gates | per milestone (Section 13): speed-of-light ratios, prefill vs llama.cpp, TTFT, load time |

---

## 13. Delivery plan

Ten milestones, each with a scope, the models it must run, and exit criteria measurable on hardware LLMario owns
(an M4 Max 64 GB validated today; GitHub-hosted Linux and Windows runners for CPU paths). Sizes are relative
(S < M < L < XL); dates depend on staffing and are deliberately absent. Every milestone ends with a benchmark
report under `benchmarks/results/` and an entry in `docs/OPS_LOG.md` for anything with side effects.

| # | Milestone | Scope | Models (first run) | Exit criteria | Size |
|---|---|---|---|---|---|
| M0 | Baselines and scaffolding | `crates/engine/*` skeleton with the crate map of 4.1; `crates/adapter_native`; `BackendKind::Native`, `ModelFormat::Safetensors`; CI jobs (Linux, Windows, Miri on CPU paths); `testkit` with golden-token and memory-peak probes; `llmario-engine plan` producing an `ArchSpec` and a plan from GGUF headers; speed-of-light table; KLD harness; baseline measurements of llama.cpp (pinned build) and mlx-lm (pinned) on the M4 Max for the reference set | Qwen3-1.7B, Qwen3.5-9B, Gemma 4 12B, Qwen3.6-35B-A3B, gpt-oss-20b (as baselines only) | baseline report committed; plan JSON validated against llama.cpp's own buffer sizes for the reference set (within 5 %) | S |
| M1 | CPU dense engine over GGUF | P0 block types; dense GQA families; in-engine tokenizers; template rendering; sampling; planner v1 (host only); mmap loading with bounded prefault; P-core thread pool; decode GEMV + prefill GEMM (interleaved tiles via the background repack cache); CPU flash attention (fp32 accumulate); contract v2 over UDS/named pipe; SSE streaming; `llmario run`/`serve`/`bench` through the native adapter | Llama 3.1 8B, Qwen3 1.7B/8B, Granite 4.2 3B/8B, Phi-4 mini, SmolLM3, Ministral 3 14B, Mistral Small 3.2 24B, Olmo 3 7B | greedy golden match vs llama.cpp on the fixture set; decode ≥ 0.6 × roofline on the M4 Max CPU and on an x86 runner; prefill ≥ 0.7 × llama.cpp CPU on the 1.8k-token prompt; plan bound and tightness hold; load ≤ 1.2 × llama.cpp warm | L |
| M2 | Metal backend | shared-mode buffers over mmap; `MTLHeap` arena; residency sets with keep-alive; GEMV/simdgroup GEMM/SDPA/fused kernels; on-device sampling; self-test; budget from `recommendedMaxWorkingSetSize`; memory-pressure and thermal handling | the M1 set | decode ≥ 0.9 × llama.cpp Metal; prefill ≥ 0.8 ×; TTFT ≤ llama.cpp + 10 %; plan bound; no idle cliff (decode latency after 5 s idle within 10 % of steady state); 8-hour soak < 1 % drift | L |
| M3 | KV subsystem, hybrids, multi-slot, safetensors | paged arena with three cache classes; prefix cache; checkpoints and RAM tier; KV q8_0/q4_0 with rotation and the 4-bit-key guard; 4/16-slot continuous batching with retraction; Gated DeltaNet, Mamba-2, gated short-conv blocks; sliding-window, K=V, per-layer embeddings, soft-capping; safetensors BF16 (ISQ) and MLX affine reader; P1 block types | Qwen3.5 0.8B–9B, Qwen3.6-27B, Qwen3.8-27B, Ornith 1.5 9B, Gemma 4 E2B/E4B/12B/31B, LFM2.5 1.2B/2.6B, Granite 4.0-H Small, Nemotron 3 Nano (text), an MLX 4-bit Qwen3.5-9B | recurrent-state isolation tests pass under 16-slot batching; 4-slot aggregate ≥ llama.cpp `balanced`; Gemma 4 12B at 128K within plan on a 32 GB budget; greedy match vs mlx-lm on the MLX fixture; prefix-cache hit path measured | XL |
| M4 | MoE, placement and the degradation ladder | router variants, shared experts, grouped expert GEMM; MXFP4/NVFP4 and I-quant types; the CPU expert path; host/device placement planner with per-tensor overrides; ladder steps 5–6 end to end; MLA absorbed cache; AWQ/GPTQ repack-at-load | gpt-oss-20b, Qwen3.6-35B-A3B, Qwen3 Coder 30B-A3B, Gemma 4 26B-A4B, GLM-4.7-Flash, LFM2.5-8B-A1B, Nemotron 3.5 Lightning, gpt-oss-120b on 64 GB via the ladder | CPU MoE decode ≥ 0.5 × active-bytes roofline (x86 runner and M4 Max CPU); gpt-oss-20b on Metal ≥ 0.9 × llama.cpp; ladder fixtures reproduce documented refusals; plan bound on every model | XL |
| M5 | Tools and internet | Responses API and Chat Completions tools; eight parser families + auto-parser; `llguidance`; MCP host (`rmcp`); `web_search`/`web_fetch`/`retrieve`; policy gate; sandbox + egress proxy on all three OSes; desktop *Web access* setting, approval dialog, MCP panel; audit; `THREAT_MODEL.md` update | Qwen3.5-9B, Gemma 4 12B, gpt-oss-20b, GLM-4.7-Flash, Ministral 3 14B, Granite 4.2 8B, LFM2.5-2.6B as tool-calling fixtures | parser corpus 100 %; grammar fuzz clean; SSRF and injection suites pass; an end-to-end search+fetch+answer run on Qwen3.5-9B with citations; approval flow exercised from the desktop app and from the API | L |
| M6 | Vulkan backend (NVIDIA, AMD, Intel, integrated) | `ash` path with GLSL→SPIR-V at build time; KHR/NV cooperative-matrix and integer-dot-product probing with scalar fallbacks; per-device deny list; self-test; autotune; device-local arena; pre-recorded command buffers; block-table attention; hot-expert GPU cache; Windows DXGI budget, job objects and EcoQoS; system-memory spill detection. **Needs an NVIDIA or AMD discrete-GPU machine** (Linux and Windows), which LLMario does not have validated today | the M1–M4 sets | decode ≥ 0.9 × llama.cpp Vulkan and ≥ 0.8 × llama.cpp CUDA on the same NVIDIA device; prefill ≥ 0.8 × llama.cpp Vulkan; plan bound including WDDM shared-memory growth; 8-hour soak on Windows; hot-expert cache ≥ 1.5 × on a host-offloaded MoE | L |
| M7 | NPU delegate and wider GPU validation | OpenVINO GenAI delegate (embeddings, prefill offload, ≤ 2B decode) on Intel NPUs; Vulkan validation on Strix Halo, Intel Arc and integrated GPUs; an optional open-source ROCm prefill path only if measured ≥ 1.2 × over Vulkan on the same AMD device. **Needs a Strix Halo or Intel Arc machine and an Intel NPU laptop** | the M1 and M4 sets on Vulkan; Qwen3.5-0.8B/2B and an embedding model on the NPU | ≥ 0.85 × llama.cpp Vulkan on each device; self-test catches a deliberately corrupted kernel; delegate passes embedding and ≤ 2B decode tests within its budget | M |
| M8 | Speculative decoding and Metal 4 tensor path | n-gram lookup; MTP heads (side-car and in-file); DFlash2/EAGLE-3 drafter loading; few-row GEMM on all backends; greedy-equivalence; Metal 4 `matmul2d` path with test-compile probe (**needs an M5-class Mac**) | Qwen3.8-27B (+ `mtp-` side-car), Gemma 4 31B (+ drafter), Qwen3.5-9B | ≥ +40 % decode on Qwen3.8-27B with MTP on Metal and Vulkan; zero regression with speculation off; greedy equivalence on fixtures; prefill ≥ 1.0 × llama.cpp on the M5 machine with the tensor path | M |
| M9 | Hardening and beta release | LoRA loading; `mmproj` recognition (vision itself deferred); SSD KV tier; P4 types; catalog licence flags; capability report in `doctor`; ADR 0002; docs; soak on all platforms; security review; beta channel release per `llmario-beta-branch` conventions | the full catalog | every target in 1.5 met on the validated machines; beta release notes published | M |

**Later (M10+).** Vision and audio encoders; `code_exec` sandbox; multi-device transports; XGrammar-2 structural
tags; overlap scheduler; int8 recurrent state; FP8 KV; radix-tree prefix cache; headless-browser fetch.

**Hardware the plan assumes.** The M4 Max 64 GB (have); an NVIDIA or AMD discrete-GPU machine with Linux and Windows (M6); a Strix Halo
or Intel Arc box and an Intel NPU laptop (M7); an M5-class Mac (M8); memory-tier tests on 8 GB and 16 GB Macs or,
until then, the existing `--memory-limit-gb` emulation.

**Decision point at M2 exit.** If, after the effort allotted to M2, the Metal kernels are below 0.7 × llama.cpp on
decode or prefill, the maintainers decide between (a) more kernel work, (b) a temporary FFI "kernel provider"
that calls ggml-metal kernels (MIT) behind the backend trait while native kernels catch up, or (c) narrowing the
first release to CPU + Metal GEMV paths. The memory architecture, planner and scheduler are unaffected by that
choice, which is the point of the layering.

---

## 14. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation / trigger |
|---|---|---|---|
| Kernel parity with llama.cpp/MLX takes longer than planned | high | schedule | roofline-relative targets; the engine ships as an opt-in backend behind `--backend native` with llama.cpp/MLX adapters intact; the M2 decision point |
| Model-zoo churn (new families monthly) | certain | maintenance | the description layer (6.7); golden fixtures per family; tracking llama.cpp's architecture list; a family needing a new block is scheduled, not improvised |
| Hybrid recurrent-state corruption under batching | medium | correctness | dedicated isolation tests (the mlx-lm 0.31.0 and llama.cpp #29002 bug class); per-sequence state pools |
| Metal API drift (macOS 26 Metal 4, residency semantics) | medium | availability | capability probes with test-compiles; self-test at startup; fallbacks that are logged, never silent |
| Windows WDDM system-memory fallback hides OOM | high on Windows | silent 5–10 × slowdown | detection (5.5), job objects, printed guidance; refuse admission once detected |
| macOS wired cap below "fits in RAM" | medium | thrash | plan against `recommendedMaxWorkingSetSize`; opt-in raise with explicit risk text; never silent |
| Licence contamination from GPL/NC references | medium | legal | Section 16 policy; `cargo deny` (`deny.toml` exists) with an allowlist; clean-room rule for flagged methods; NOTICE propagation |
| Tool-use security incidents (injection → exfiltration) | medium | user harm | Rule-of-Two gate default; two-layer sandbox; red-team corpus in CI; conservative default profile (search + fetch only) |
| Search provider availability (public SearXNG instances disable JSON output; rate limits) | low | feature gaps | user-supplied self-hosted instance; the generic endpoint adapter; no persistence of results beyond the session |
| Scope creep | high | schedule | milestone exit criteria; the "later" list; the avoid column of Report §Adopt first, defer, avoid |
| Unvalidated platforms (Linux and Windows GPUs, NPUs) | high | coverage claims | the support matrix (`docs/support-matrix.toml`) only claims what was measured; hardware acquisition is called out per milestone |
| Benchmark rot | certain | comparisons | every number carries a version; baselines re-measured per milestone on the same machine |

---

## 15. Open decisions (for the maintainers)

| Decision | Options | Recommendation |
|---|---|---|
| Keep the MLX-LM adapter long-term? | keep as fallback; drop once M3 proves MLX-folder support | keep through M9; revisit with the beta's measurements |
| Windows contract transport | named pipe vs loopback TCP | named pipe with a DACL (no port, no Host-header surface); TCP for debugging |
| Default SSD KV tier for `serve` | on / off | off for `serve`, on for the desktop app; both settable |
| Repack-cache disk budget | fixed GB vs % of free | 10 % of free disk capped at 20 GB, deletable from the UI |
| Approval UX for API clients without a UI | reject by default vs. queue with timeout | queue the `mcp_approval_request` with a 60-second timeout then reject; a per-key "auto-approve read-only" setting |
| Telemetry | none (current) vs opt-in local-only stats | keep none; `/metrics` stays local |
| Raising `iogpu.wired_limit_mb` | never vs print command vs run with sudo | print the command and the risk; never run it |
| Metal vs an all-open Vulkan stack on macOS | Metal through `objc2-metal` (the OS API, open-source bindings) vs Vulkan over Mesa KosmicKrisp/MoltenVK only | Metal: it is the OS API reached through open-source code only, and 10–20 % faster on the one data point; keep Vulkan-on-macOS as a test target and fallback |
| Safetensors priority vs GGUF-only until M4 | as planned (M3) vs later | as planned: Apple users have MLX folders on disk already |
| Gemma 4 E2B/E4B licence label in the catalog | "Gemma Terms of Use" (current) vs Apache-2.0 (Hub tag and tech report) | recheck and correct the catalog (flagged by the research) |
| Catalog flags for non-OSI licences | add a "restricted" badge | add it: Flash-Next (qwen-community-1.0), Nemotron 3.5 (OpenMDW-1.1), LFM2.5 (LFM Open License), Llama (community), Falcon-H1, MiniMax, Kimi |

---

## 16. Licensing policy

0. **Open-source only.** Every library, runtime, kernel toolchain and service the engine builds against or calls at
   runtime must be open source under an OSI-approved licence. The only exceptions are the operating systems' own
   APIs and the GPU drivers the OS ships (Metal, DXGI/Win32, Vulkan loaders), reached through open-source crates.
   Consequences: no CUDA, no Apple Accelerate/BNNS, no closed NPU runtimes (QAIRT/GenieX, Ryzen AI software,
   FastFlowLM kernels, Core ML), no proprietary search APIs; the build never requires Xcode's shader compiler or a
   vendor SDK; `cargo deny` enforces the dependency side and code review enforces the runtime side.
1. The engine is Apache-2.0. Code ported from MIT, Apache-2.0 or BSD-3 projects is allowed with copyright notices
   kept and `NOTICE` updated (`docs/LICENSES.md` is the inventory).
2. Reference material that is GPL-3.0 (QTIP, QuIP#), AGPL (koboldcpp, forge-ml, GPTQModel's Swordfish kernel),
   CC-BY-NC (SpinQuant, any4/tinygemm, LayerSkip, Nexa OmniNeural), BUSL (gmlx/mlx-kquant) or FAIR (Cake) may be
   read for understanding only; nothing is ported from it, and any clean-room re-implementation from a paper is a
   decision for counsel, recorded in `docs/LICENSES.md`.
3. Only open-source NPU runtimes are integrated (OpenVINO GenAI, Apache-2.0, plus Intel's MIT Linux NPU driver)
   behind an optional cargo feature. Closed runtimes (QAIRT/GenieX, Ryzen AI software, FastFlowLM kernels, Core
   ML/ANE) stay out until an open-source path exists.
4. Model weights keep their own licences; the catalog shows them and flags non-OSI terms (Section 15).
5. `cargo deny` runs in CI with an allowlist of licences; a new dependency outside the allowlist fails the build.
6. Reusable, verified-permissive sources this design draws on: llama.cpp/ggml and the GGUF spec (MIT); MLX
   (MIT); mistral.rs and candle (MIT/Apache); vLLM, SGLang, FlashInfer, kvcached, LMCache (Apache-2.0);
   FlashAttention (BSD-3); KleidiAI, oneDNN (Apache-2.0); safetensors (Apache-2.0); llguidance (MIT); rmcp
   (Apache-2.0); SpecForge, prima.cpp, distributed-llama (MIT); the Rust crates named in 4.1.

---

## Appendix A. Formulas and worked numbers

**Speed-of-light decode.** `tok/s_max = η · BW_peak / (active_weight_bytes + kv_bytes_read(ctx) + state_bytes)`,
with `η` ≈ 0.6–0.8 for a well-threaded dense model (measured envelopes: M4 Max ≈ 0.58, M5 Max ≈ 0.74, RTX
3090/4090/5090 ≈ 0.66/0.72/0.64 on 7B Q4_0; a Ryzen AI 9 HX 370 at 3.54 tok/s against 3.4 predicted for a 19 GB
32B Q4). The engine logs `η=1` and the autotuned expected `η` next to every measured rate.

**Per-token KV bytes (one sequence).**
`Σ_attention_layers 2 · n_kv_heads · head_dim · bytes_per_elem` — with `1 ·` for K=V layers (Gemma 4 global),
`(d_latent + d_rope) · bytes` for MLA, `0` for recurrent/linear layers, and window layers capped at
`window + n_batch` tokens. Recurrent state per sequence: Gated DeltaNet `n_v_heads · head_v_dim · head_k_dim · 4 B`
plus conv state; Mamba-2 `n_heads · head_dim · d_state · 4 B` plus conv state (fp32, from the models' declared
`mamba_ssm_dtype`).

| Model | KV per token (BF16) | Fixed per sequence | KV at 128K (BF16 / q8_0 / q4_0) |
|---|---|---|---|
| Llama 3.1 8B | 128 KiB | — | 16 / 8.5 / 4.5 GiB |
| Qwen3-8B | 144 KiB | — | 18 GiB |
| Mistral Small 3.2 24B | 160 KiB | — | 20 GiB |
| Gemma 3 27B (5:1 sliding 1,024) | 80 KiB (global layers) | 416 MiB of windows | 10.4 GiB (62 GiB if local layers were cached in full) |
| Qwen3.5/3.6/3.8-27B (16 attention layers × 4 KV heads × 256) | ≈ 64 KiB (computed here) | ≈ 40–160 MB GDN state | ≈ 8 / 4.3 / 2.2 GiB |
| gpt-oss-20b | 24 KiB | — | 3.0 GiB |
| Qwen3-Next-80B | 24 KiB | ≈ 37.7 MiB | 3.0 GiB |
| Granite 4.0-H Small | 16 KiB | ≈ 73.7 MiB | 2.0 GiB |
| Nemotron 3 Nano | 6 KiB | ≈ 24 MiB | 0.75 GiB |
| DeepSeek-V3.2 (MLA latent 576) | 8.6 GiB at 128K | indexer keys 7.8 KB/token | ≈ 610 GiB if expanded per head — MLA must be served absorbed |

Source: Report §KV cache (computed from each model's `config.json`); the Qwen3.5 row is this document's arithmetic.

**Planner usable-budget rule (from the report, validated against Gemma 4's published int8-KV figures within
0.02 GB):** usable = 2/3 of RAM at 8/16/32 GB and 3/4 at 64 GB, minus 1.0 GB overhead; on macOS the GPU ceiling
is `recommendedMaxWorkingSetSize` regardless.

## Appendix B. Memory-fit matrix (4-bit weights, 8-bit KV)

| Tier (usable) | Fits with the stated context | Needs the ladder (offload/streaming) |
|---|---|---|
| 8 GB (5.3 GB) | Qwen3.5-0.8/2/4B, Gemma 4 E2B, LFM2.5-1.2/2.6B, Granite 4.2 3B, SmolLM3, Llama 3.2 — 128K only for hybrids and Gemma 4 E2B; GQA-128 dense 4B default ≤ 32K | everything ≥ 7B |
| 16 GB (10.7 GB) | Qwen3.5-9B / Ornith 9B (8.9 GB at 128K), Gemma 4 12B (8.7) and E4B, LFM2.5-8B-A1B, Mellum2.1, Granite 4.2 8B and Llama 3.1 8B to 32K, Ministral 3 14B at 8K | gpt-oss-20b (12.1–12.8 GB resident under MLX; tight), 24–27B |
| 32 GB (21.3 GB) | Qwen3.8/3.6-27B (17.56 GB UD-Q4_K_XL; 22.2 GB at 128K, tight), Gemma 4 31B (21.8 at 128K) and 26B-A4B (16.4), gpt-oss-20b (14.7), GLM-4.7-Flash (22.8 with the absorbed MLA cache), Nemotron 3.5 Lightning (20.4), Muse Glimmer (18.6), Mistral 24B to 32K | 35B-A3B MoEs (20.4 GB weights + overhead) unless the working set is raised or a Q3 variant is used |
| 64 GB (48 GB) | the above plus 35B-A3B at 128K (22.8 GB), Qwen3-Next-80B (≈ 47.5, tight), 70B dense at Q4_K_M (38.5–45.4 GiB) | gpt-oss-120b (64.4 GB), Qwen3.5-122B-A10B (~71), Mistral Small 4 (~68), Llama 4 Scout (~63), GLM-4.5-Air (~64), Qwen3.8-Flash-Next (94.9 GB at IQ4_XS) |

Source: Report §Target models.

## Appendix C. Operator union (what the kernels must cover)

Beyond RMSNorm, RoPE, GQA flash attention, SwiGLU and a tied-or-untied output head:

| Operator / feature | Families that need it |
|---|---|
| Gated DeltaNet chunked scan (conv kernel 4, separate QK/V head counts) | Qwen3.5 / 3.6 / 3.8 stack, Qwen3-Next, Ornith 1.5, MiMo distill, Clef |
| KDA | GLM-5.3-Flash (upper bound) |
| Mamba-2 SSD scan with grouped states | Nemotron 3 / 3.5, Granite 4.0-H, Falcon-H1 |
| Gated short convolution (`conv_L_cache` 3) | LFM2 / LFM2.5 |
| Sliding-window masks 128 / 512 / 1,024 / 2,048 / 4,096 and 8,192-token chunks | gpt-oss, Gemma 3/4, Mistral, Llama 4 |
| K=V global attention with `global_head_dim` 512; cross-layer KV sharing; per-layer embeddings | Gemma 4 (12B / 26B-A4B / 31B; E-series) |
| Learned attention sinks; clamped SwiGLU (`swiglu_limit` 7, α 1.702); head_dim 64 | gpt-oss |
| MLA with absorbed compressed cache (kv_lora 512 + rope 64) | GLM-4.7-Flash, Mistral Small 4, Kimi K2, DeepSeek |
| RoPE families: default/partial (0.25 on Qwen3.5), interleaved mRoPE [11,11,10], YaRN with attention factor (×4 Qwen3, ×32 gpt-oss, ×40 DeepSeek, ×128 Mistral Small 4), Llama-3 piecewise (×8, ×16 Llama 4), LongRoPE (Phi-4-mini), linear with per-layer base (Gemma 3), NoPE + Llama 4 temperature | as listed |
| Grouped-expert GEMM with softmax-renormalised, sigmoid-with-bias / `noaux_tc` (`routed_scaling_factor` 1.8–2.5), `sqrtsoftplus` and top-1-plus-shared routers; shared experts; MXFP4 expert weights | Qwen MoE, Gemma 4 26B-A4B, gpt-oss, GLM, Nemotron 3.5, LFM2.5-8B-A1B, DeepSeek, MiniMax |
| GeGLU, relu², `(1 + w)` norm gains, μP multipliers, logit soft-capping, gated attention outputs, QK-norm | Gemma, Nemotron (relu²), MiniMax, Qwen4Exp, Granite 4.2 (μP), Muse Glimmer (gated attention, NoPE global) |
| Hyper-connections, looped layers (`num_loops` 2), n-gram/engram embeddings, sparse-attention indexer | Qwen3.8-Flash-Next, Nanbeige4.2, DeepSeek V4.1 (upper bounds) |
| MTP / DFlash / DSpark / EAGLE-3 drafter execution | every 2026 flagship ships a drafter |
| Numeric rules | fp32 attention accumulation on CPU/Metal; fp32 recurrent state; fp32 logits |

Tokenizers: byte-level BPE at 151,936 / 248,320 (Qwen), SentencePiece 262,144 (Gemma), o200k variants 201,088 /
200,064 (gpt-oss, Phi-4-mini, MiniMax), 128,256 (Llama 3, SmolLM3), 202,048 (Llama 4, Muse), Tekken 131,072
(Mistral), cl100k-derived 100,352 / 100,278 (Granite, Phi-4, Olmo). Source: Report §Target models.

## Appendix D. Tool-call wire formats (verified from chat templates)

| Family | Call syntax | Result rendering / notes |
|---|---|---|
| Hermes JSON | `<tool_call>{"name":…,"arguments":{…}}</tool_call>` | Qwen3, SmolLM3, Granite 4.0 |
| XML function/parameter | `<tool_call><function=name><parameter=k>v</parameter></function></tool_call>` | Qwen3.5 (results inside a *user* turn as `<tool_response>`; thinking re-rendered only after the last user query), Qwen3-Coder, Nemotron 3 (nested-XML tool declarations) |
| GLM | `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>` | `<\|observation\|>` results; GLM-4.7 |
| Gemma 4 native | `<\|tool_call>call:name{key:<\|"\|>value<\|"\|>}<tool_call\|>` | `<\|channel>thought` thinking channel; template raises if `arguments` is a string |
| Llama 3.x / Llama 4 | `<\|python_tag\|>` + JSON (single call) / pythonic lists | Llama 3.1, 3.2, Llama 4, Muse Glimmer uses an `<atem:invoke>` protocol |
| Mistral control tokens | `[TOOL_CALLS]name[ARGS]{json}` with 9-character call ids | Mistral Small 3.x, Ministral 3, Devstral 2 (Ministral places tool calls inside the reasoning block — known pitfall) |
| gpt-oss Harmony | channels (`analysis`, `commentary`, `final`) with built-in `browser`/`python` tool conventions | `analysis` is not safety-tuned and never shown to users; prior analysis passed back during a tool loop and dropped once the turn ends in `final` |
| LFM2 pythonic | `<\|tool_call_start\|>[f(a='b')]<\|tool_call_end\|>` | template raises on string arguments |

Engines' parser tables: vLLM ≈ 30 `--tool-call-parser` values; Ollama `Add(chunk, done)`; mlx-lm 13 parser
modules; llama.cpp PEG parser with `NEED_MORE_INPUT`, lazy trigger grammars and a differential-rendering
auto-parser. Source: Report §Internet and tools.

## Appendix E. OS API cheat sheet

| Need | macOS | Windows | Linux |
|---|---|---|---|
| GPU/host ceiling | `MTLDevice.recommendedMaxWorkingSetSize`; `iogpu.wired_limit_mb` (root, non-persistent) | `IDXGIAdapter3::QueryVideoMemoryInfo` budget; `VK_EXT_memory_budget` | `VK_EXT_memory_budget`; cgroup v2 `memory.max`/`memory.high`; `MemAvailable` |
| Pressure signal | `DispatchSource.makeMemoryPressureSource` (warning/critical) | `CreateMemoryResourceNotification`; DXGI budget-change event | PSI triggers (`/proc/pressure/memory`, cgroup `memory.pressure`); cgroup events |
| Enforce a cap | none (supervisor sleep-then-kill) | job object `JOB_OBJECT_LIMIT_PROCESS_MEMORY`; `PeakProcessMemoryUsed` | child cgroup `memory.high`; `oom_score_adj` |
| Keep GPU memory resident | `MTLResidencySet` + keep-alive | n/a (WDDM manages; detect sysmem fallback) | n/a |
| Prefault / prefetch | `madvise(MADV_WILLNEED)` | `PrefetchVirtualMemory` | `madvise(MADV_WILLNEED)`, `MADV_POPULATE_READ` (≥ 5.14) |
| Lock | `mlock` (limits) | `VirtualLock` after `SetProcessWorkingSetSizeEx` | `mlock`/`MLOCK_ONFAULT` (`RLIMIT_MEMLOCK`, `CAP_IPC_LOCK`) |
| Huge pages (engine-owned anonymous buffers only) | n/a | large pages need `SeLockMemoryPrivilege` (avoid) | `MADV_HUGEPAGE` (THP `madvise` mode); hugetlbfs needs boot reservation (avoid) |
| Resident-memory estimate | `vm_region`/`phys_footprint` | `QueryWorkingSetEx` | `mincore`, `/proc/self/smaps_rollup` |
| Priority when backgrounded | `pthread_set_qos_class_self_np` utility/background | `SetProcessInformation(ProcessPowerThrottling)` EcoQoS | `nice`, `SCHED_BATCH` |
| Thermal / power | `ProcessInfo.thermalState`, `isLowPowerModeEnabled` | `GUID_POWER_SAVING_STATUS`, `GUID_ACDC_POWER_SOURCE` | `/sys/class/thermal` |
| Sandbox | Seatbelt (`sandbox_init`) | AppContainer / restricted token + job object | bubblewrap + seccomp; Landlock |

Source: Report §Memory governance and §Security.

## Appendix F. Interface sketches (illustrative, not final)

```rust
// core: tensors and devices
pub struct TensorView { pub dtype: BlockType, pub shape: [u64; 4], pub strides: [u64; 4], pub buf: BufferId, pub offset: u64 }
pub enum BlockType { F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K,
                     IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, MXFP4, NVFP4, TQ1_0, TQ2_0,
                     MlxAffine { bits: u8, group: u16 } }

// model: architecture description
pub struct ArchSpec {
    pub family: Family, pub n_layer: u32, pub d_model: u32, pub vocab: u32,
    pub layers: Vec<LayerSpec>,           // per layer: Attention{GQA, window, k_eq_v, mla, sinks, qk_norm} | DeltaNet | Mamba2 | ShortConv
    pub ffn: FfnSpec,                     // SwiGLU | ClampedSwiGLU | GeGLU | ReluSq | Moe{n_expert, top_k, shared, router}
    pub rope: RopeSpec, pub norm: NormSpec, pub softcap: Option<f32>, pub tokenizer: TokenizerSpec,
    pub drafter: Option<DrafterSpec>, pub cache_classes: Vec<CacheClass>,
}

// plan
pub struct Plan {
    pub hash: Hash, pub profile: ResolvedProfile,
    pub placement: Vec<(TensorGroup, DeviceId, Residency)>,
    pub kv: Vec<KvArenaSpec>,             // per device: blocks, block_bytes, recurrent_pool_bytes
    pub scratch: Vec<(DeviceId, GraphShape, u64)>,
    pub caches: CacheLimits, pub drafter_bytes: u64, pub runtime_fixed: u64,
    pub totals: Vec<(DeviceId, u64 /*planned*/, u64 /*budget*/, u64 /*headroom*/)>,
    pub degradations: Vec<DegradationStep>, pub speed_of_light: Vec<(DeviceId, f64)>,
}

// kv
pub trait KvArena { fn alloc_blocks(&mut self, n: u32) -> Option<Vec<BlockId>>; fn free(&mut self, b: &[BlockId]);
                    fn free_blocks(&self) -> u32; fn block_bytes(&self) -> u64; }
pub struct SequenceCache { pub tables: Vec<BlockTable>, pub windows: Vec<RingRef>, pub recurrent: Vec<StateRef>, pub prefix_hashes: Vec<Hash> }

// sched
pub struct Step { pub decodes: Vec<(SlotId, Token)>, pub prefills: Vec<(SlotId, Range<u32>)>, pub shape: GraphShape }

// tools / policy
pub enum ToolClass { ReadOnlyLocal, OpenWorldRead, StateChanging, ExfiltrationCapable }
pub struct PolicyDecision { pub allow: bool, pub needs_approval: bool, pub reason: &'static str }
pub trait PolicyGate { fn decide(&self, session: &SessionState, tool: &ToolRef, args: &serde_json::Value) -> PolicyDecision; }

// supervisor extension (crates/supervisor::adapter)
pub trait EngineAdapter: Send + Sync {
    // existing methods unchanged …
    fn plan(&self, model: &ModelEntry, profile: &ResolvedProfile, hw: &HardwareReport, cfg: &Config) -> Option<Plan> { None }
}
```

## Appendix G. Glossary

**Arena** — a memory region reserved once and carved internally. **Block (KV)** — a fixed group of token positions
(32–64) for one layer group. **Cache class** — FullAttention / Window / Recurrent. **Checkpoint** — window rings +
recurrent state + block-chain head at a chat boundary. **Degradation ladder** — the fixed order of reductions the
planner applies before refusing. **Ledger** — the engine's own per-device memory accounting. **Placement** — the
assignment of a tensor group to a device. **Plan** — the immutable per-process memory and placement decision.
**Repack cache** — derived interleaved-tile copies of prefill tensors, mmapped, deletable. **Rule of Two** — a
session may hold at most two of {untrusted input, sensitive access, external side effects} without a human.
**Slot** — one concurrent sequence. **Speed-of-light** — bandwidth-bound decode rate for the active bytes per
token. **Taint** — the session flag set once untrusted content has entered the context.

## Appendix H. Evidence index

| Decision | Where the evidence is |
|---|---|
| Memory governance, ledger, budgets, ladder, allocators, OS integration | Report §Memory governance; notes `os_memory_and_resource_governance.md` KQ1–KQ6 |
| Niche definition (no joint budgeting anywhere) and copy/avoid lists | Report §No engine budgets memory jointly; notes `engine_landscape.md` KQ2–KQ4 |
| Formats, block types, quant quality ranking, repack cache | Report §Read GGUF and safetensors natively; notes `quantization_and_formats.md` |
| KV classes, paging, precision policy, prefix cache, checkpoints, RoPE | Report §KV cache; notes `kv_cache_and_attention.md` |
| CPU ISA matrix, kernels, threading, MoE path | Report §CPU execution; notes `cpu_execution.md` KQ2–KQ8 |
| Metal/Vulkan/NPU designs and ordering (CUDA excluded as proprietary) | Report §GPU and NPU backends; notes `gpu_npu_backends.md` KQ1–KQ9 |
| Placement, process model, scheduling, speculative decoding, incidents | Report §Sharding and scheduling; notes `sharding_parallelism_distributed.md` |
| Tool formats, grammars, MCP, web tools, security | Report §Internet and tools and §Security; notes `internet_tools_and_agents.md` |
| Model coverage, operator union, fit matrix, catalog flags | Report §Target models; notes `target_models_2026.md` |
| Everything deferred or avoided | Report §Adopt first, defer, avoid (decision table) and §Consolidated open gaps |
