# llmario — adaptive local LLM runtime

llmario is a small Rust control plane that makes local inference easy to launch, resource-aware,
and measurable. It **does not reimplement inference kernels**: it detects your hardware, picks
and supervises a proven open-source engine (llama.cpp or MLX-LM), plans memory before loading,
and serves one stable OpenAI-compatible API on loopback.

```text
CLI (llmario) ─┐
               ├─ API gateway (OpenAI-compatible, loopback, auth, field allowlist, SSE relay)
               ├─ planner (model variant + backend selection, per-request limits)
               ├─ supervisor (engine processes, readiness, crash relaunch, LRU, admission)
               ├─ hardware profiler · model registry · memory planner · benchmark harness
               └─ adapters ─► llama-server (GGUF) | mlx_lm.server (MLX) | mock (CI)
```

**Status: Phase 1 MVP.** Validated end-to-end on an Apple M4 Max (64 GB) with MLX-LM.
See [what is and is not validated](#support-status) before relying on it.

## Quick start

```bash
cargo build --release
./target/release/llmario doctor                    # hardware, backends, memory budget
./target/release/llmario model catalog             # pullable models
./target/release/llmario model pull qwen3-1.7b     # best variant for this machine, checksum-verified
./target/release/llmario run qwen3-1.7b "Explain KV caches in two sentences."
./target/release/llmario serve                     # http://127.0.0.1:11500/v1
```

Backends are separate open-source installs:

| Backend | Install | Tested version |
|---|---|---|
| llama.cpp (`llama-server`) | `brew install llama.cpp` or build [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | build 11146 (7fe450e19) |
| MLX-LM (Apple Silicon) | `scripts/setup-mlx-venv.sh` (pinned venv) or `pip install mlx-lm` | mlx-lm 0.31.3, mlx 0.32.2 |

Any OpenAI client works against `serve`:

```bash
curl -s http://127.0.0.1:11500/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"qwen3-1.7b","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

## Commands

| Command | What it does |
|---|---|
| `doctor [--json]` | CPU features, RAM, GPU/unified memory budget, backend versions, per-model fit, orphaned engines |
| `model catalog` / `list` / `info` | Catalog of pullable models; installed models with source commit, license, checksums |
| `model pull <id\|family>` | HTTPS download, SHA-256 / git-SHA-1 verified, atomic install; reuses matching blobs from the local Hugging Face cache |
| `model add <path>` | Register an existing `.gguf` file (or GGUF blob) or MLX directory without copying |
| `model remove` / `verify` / `fit` | Remove (deletes only files llmario downloaded); re-hash; memory estimate without loading |
| `run <model> [prompt]` | Terminal chat through the same gateway path as `serve`; prints TTFT and tok/s |
| `serve` | OpenAI-compatible API: `/v1/models`, `/v1/chat/completions`, `/healthz`, `/metrics` |
| `bench -m <model>` | Reproducible benchmark (TTFT, prefill, decode, p50/p95, peak memory, quality) → JSON + Markdown |
| `bench --url <base> -m <name>` | Same suite against any OpenAI-compatible server (Ollama, raw llama-server, …) |
| `config` | Effective configuration and file locations (API key redacted) |

Profiles (`--profile`): `latency` (1 slot × 8k ctx), `balanced` (4 × 8k), `throughput`
(16 × 4k). Values are printed on start; `--context` overrides the per-request context.

## What llmario adds on top of the engines

- **Execution-path selection.** Ask for a family (`qwen3-1.7b`); llmario picks the installed
  variant for this hardware (MLX on Apple Silicon, llama.cpp elsewhere; `--backend` overrides).
  An exact id is never silently substituted.
- **Memory planning before loading.** Weights + KV cache (all slots) + backend extras are
  estimated from GGUF headers or `config.json`; loads that do not fit are refused with the
  limiting factor and a context that would fit. Calibrated against measured peaks (below).
- **Backend-aware request handling.** Example found by our benchmark: `mlx_lm.server` silently
  disables continuous batching for any request with a `seed`; llmario drops `seed` when
  decoding is greedy (output-identical) so batching stays on.
- **Safety defaults.** Loopback bind, explicit `--allow-remote` + API key, Host-header check
  (DNS rebinding), JSON-only bodies, request-field allowlist (MLX would otherwise load any model
  a request names), offline engines, no prompt logging. See [THREAT_MODEL](docs/THREAT_MODEL.md).
- **Honest benchmarks.** Cold-prefix mode by default so prefix caches cannot inflate prefill
  numbers; failures and neutral results stay in the report.

## Measured results (Apple M4 Max, 64 GB, AC power)

Qwen3-1.7B MLX 4-bit through llmario, default suite, temperature 0, cold prefix cache.
Full reports: [`benchmarks/results/`](benchmarks/results/).

| Profile | Conc | TTFT p50 | Decode tok/s per request | Aggregate tok/s | Peak footprint | llmario estimate | Quality |
|---|---:|---:|---:|---:|---:|---:|---:|
| latency | 1 | 566 ms | 223.7 | 98.1 | 3.18 GiB | 5.69 GiB | 3/3 |
| balanced | 1 | 534 ms | 225.7 | 102.9 | 3.75 GiB | 10.06 GiB | 2/3 |
| balanced | 4 | 1,953 ms | 93.3 | 123.0 | 8.06 GiB | 10.06 GiB | – |

- Cold start (spawn → first token): 1.3–1.4 s. Short prompts decode at ~257 tok/s; prefill is
  ~3.2–3.4k tok/s for 1.8k–4.7k-token prompts.
- **Neutral result:** on this prefill-heavy suite, 4 concurrent requests raise aggregate
  throughput only ~20% while TTFT rises ~3.6x. Use `latency` for interactive single-user work.
- **Quality caveat:** the long-context retrieval check passes under `latency` but fails under
  `balanced` at temperature 0 (the model exhausts its 1,024-token budget reasoning). Balanced
  routes even single requests through MLX's batched path, which is numerically different, so
  greedy output can diverge on borderline cases.
- **Memory estimates are upper bounds here** (1.2–1.8x the measured peak). An earlier estimate
  was *exceeded* 2.3x at concurrency 4. Root cause: MLX's allocator buffer cache and prompt cache,
  now capped and counted (see ADR). This is why estimates ship with measured calibration.
- No comparison against another runtime has been run yet, so **no speed-up claim is made.**
  `llmario bench --url` exists for exactly that comparison.

## Support status

| Path | Status |
|---|---|
| Apple Silicon + MLX-LM | ✅ validated (M4 Max, macOS 27.0.1) |
| Apple Silicon + llama.cpp (Metal) | ⚠ adapter launches build 11146 and flags are accepted; full generation with a standard GGUF not yet run |
| Linux x86_64 + CUDA (llama.cpp) | 🔬 implemented (nvidia-smi detection, hybrid offload planning), unvalidated |
| CPU-only (llama.cpp) | 🔬 implemented, unvalidated |
| AMD ROCm / Intel | ⏭ detection only; no support claimed |
| vLLM / SGLang | ⏭ Phase 2 |

Machine-readable: [`docs/support-matrix.toml`](docs/support-matrix.toml).

## Development

```bash
cargo test --workspace          # unit tests + end-to-end tests with a real mock engine process
cargo clippy --workspace --all-targets
cargo fmt --all
```

Layout: `crates/{core,hardware,model_registry,supervisor,adapter_llamacpp,adapter_mlx,adapter_mock,api,benchmark,cli}`.
Design: [ADR 0001](docs/adr/0001-architecture.md) · [milestones](docs/MILESTONES.md) ·
[benchmark plan](docs/BENCHMARK_PLAN.md) · [API](docs/API.md) ·
[troubleshooting](docs/TROUBLESHOOTING.md) · [contributing](CONTRIBUTING.md).

## License

Proposed: Apache-2.0 (declared in `Cargo.toml`). All 240 transitive crates are under
permissive or weak-copyleft licenses; see [docs/LICENSES.md](docs/LICENSES.md). Backends and
model weights carry their own licenses. llmario never redistributes weights.
