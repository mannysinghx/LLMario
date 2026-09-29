# ADR 0001 — Control plane over existing engines

**Status:** accepted · **Date:** 2026-09-28

## Context

We want local inference that is easy to launch, resource-aware, and measurably fast on
named hardware. Mature open-source engines (llama.cpp, MLX-LM, vLLM, SGLang) already own
the kernels. Rewriting them would cost years and would not beat them in general.

## Decision

1. **Rust control plane, engines out of process.** The `llmario` binary contains the CLI,
   API gateway, planner, supervisor, hardware profiler, model registry, memory estimator,
   and benchmark harness. Every inference engine runs as a **child process**, so an engine
   crash or a Python dependency conflict cannot take down model management.
2. **Adapters describe engines; the supervisor runs them.** An adapter is a small, pure
   description: how to probe the backend, which models it accepts, and the exact command line
   for a given model + profile. Process spawning, readiness, crash detection, and shutdown
   are shared code in `llmario-supervisor`.
3. **IPC contract v1 = OpenAI-compatible HTTP on loopback, ephemeral port.** Both
   launch engines (`llama-server`, `mlx_lm.server`) speak it natively, which keeps the
   per-token path to a single byte-level proxy hop. Unix domain sockets were
   preferred in the plan; `llama-server` supports them but `mlx_lm.server` does not, so v1
   uses `127.0.0.1:<random>` for both and a later contract version can move llama.cpp to a
   socket. The contract version is declared by each adapter (`CONTRACT_VERSION`).
4. **The gateway allowlists request fields.** Client JSON is never forwarded verbatim. The
   `model` field is always rewritten to the engine-local name, because `mlx_lm.server`
   will load (and download) any model named in a request.
5. **Memory is admission-controlled.** Weights + KV cache + runtime overhead are estimated
   from model metadata (GGUF header / MLX `config.json`) before a load. Loads that do not fit
   the budget are refused with the limiting factor and alternatives. We never silently
   switch models.
6. **One loaded model by default** (`max_loaded_models = 1`), LRU eviction, and a model
   that is serving requests is never evicted.
7. **Profiles are explicit value sets** (`latency`, `balanced`, `throughput`), mapped by
   each adapter to flags the backend actually supports. The resolved values are printed.

## Launch backends (MVP)

| Backend | Pinned/tested version | Format | Where |
|---|---|---|---|
| llama.cpp `llama-server` | build 11146 (commit 7fe450e19) | GGUF | macOS Metal, Linux CUDA/CPU |
| MLX-LM `mlx_lm.server` | mlx-lm 0.31.3, mlx 0.32.2 | MLX safetensors | Apple Silicon only |
| mock (in-tree) | — | `mock` | CI and tests |

vLLM / SGLang are Phase 2 adapters; the adapter trait and the gateway do not need to change
to add them.

## Consequences

- We depend on upstream HTTP behavior; adapters pin tested versions and the doctor reports
  the detected version.
- The proxy hop adds a small amount of latency. `llmario bench` measures it against the raw
  engine and against other OpenAI-compatible servers so the overhead is visible.
- Engines started by `llmario` are recorded in `~/.llmario/run/` so an orphan left by a
  killed control plane can be found (`llmario doctor`) without guessing.

## Findings from the first measurements (2026-09-28, M4 Max)

These changed the implementation; each is covered by a test or a recorded benchmark.

1. **`seed` disables MLX batching.** `mlx_lm.server` 0.31.3 batches only requests without a
   `seed`. The gateway drops `seed` for MLX when decoding is greedy (temperature 0/unset),
   which cannot change the output. A seed with temperature > 0 is kept.
2. **Prefix caches inflate naive benchmarks.** Repeating identical prompts measured 62k tok/s
   "prefill" (cache hits). `llmario bench` defaults to `--cache cold` (unique prefix per
   request); `--cache warm` measures reuse deliberately.
3. **MLX footprint ≠ MLX active memory.** At concurrency 4 with ~4.7k-token prompts the engine
   footprint reached 11.5 GiB against a 5.1 GiB estimate. Controlled runs attributed it to the
   allocator buffer cache (2.5 GiB retained after a batch) and the server prompt cache
   (~3.4 GiB for 8 entries; `--prompt-cache-bytes` alone did not bound it). Fix: the MLX
   adapter launches `mlx_lm.server`'s public `main()` through a one-line shim that calls the
   public `mx.set_cache_limit(1 GiB)`, uses fewer prompt-cache entries per profile, and the
   estimate counts both at their upper bound. Measured peak is now below the estimate.
4. **`--prompt-concurrency 1` beat 2** for the balanced profile on this suite (124.7 vs 103.6
   aggregate tok/s, lower TTFT); the balanced default was changed. Throughput-profile values are
   still unvalidated.
5. **GGUF files from other runtimes may not load in upstream llama.cpp** (Ollama's `qwen3vl`
   blob lacks `rope.dimension_sections`). Startup failures surface the engine's last log lines;
   only port-bind races are retried.
6. **Quality checks must not use cold-cache prompts.** The cold-mode nonce changed the prompt,
   and greedy answers changed with it (llama.cpp: "32" in one run, "42" in the next). Quality
   cases now always use the exact suite prompt; a unit test pins this.
7. **Comparison with Ollama 0.34.2** (benchmarks/results/2026-09-29): Ollama runs its own bundled
   llama-server (0.4.1-dev) with `-np 1`, 40,960 context, `--chat-template chatml`, and no
   `--cache-ram` (default 8 GiB). llmario's gateway overhead was not measurable; single-request speed
   was at parity; concurrency-4 throughput and TTFT were better with llmario's balanced profile; peak
   memory was lower at equal context. The wedge is configuration and resource policy, not kernels,
   which matches this ADR's premise.
