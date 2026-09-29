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

## Desktop app

`LLMario.app` is a native window for chatting with local models, built with
[Tauri 2](https://tauri.app) (MIT/Apache-2.0) and the system webview. It runs the same runtime as
the CLI inside the app: pick or download a model, and replies stream in with a collapsible
"thinking" section, first-token latency and tokens/s. **Stop** really cancels generation. Each
model shows whether it fits in memory before you load it. Chat history stays on this computer,
and Settings can turn it off or clear it.

```bash
cargo install tauri-cli --version "^2" --locked   # once
cd apps/desktop/src-tauri && cargo tauri build --bundles app
open ../../../target/release/bundle/macos/LLMario.app
```

For development, `cargo run -p llmario-desktop` opens the window without bundling. The UI is plain
HTML/CSS/JS in `apps/desktop/ui` (no bundler, no network assets, strict CSP). Model output is
rendered by a small escaping Markdown renderer with tests in `markdown.test.mjs`. The window's
only IPC permission is `core:default`: no filesystem, shell, or HTTP plugins. Launched from Finder,
the app adopts your login shell's `PATH` so it finds `llama-server` and MLX.

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

Qwen3-1.7B 4-bit, default suite, temperature 0, cold prefix cache, 3 runs per prompt.
Full comparison with method and caveats: [`benchmarks/results/2026-09-29/SUMMARY.md`](benchmarks/results/2026-09-29/SUMMARY.md).

| Target | Conc | TTFT p50 | Decode tok/s / req | Aggregate tok/s | Peak footprint | Quality |
|---|---:|---:|---:|---:|---:|---:|
| llmario → llama.cpp (latency, 8k ctx) | 1 | 495 ms | 217.1 | 102.1 | 2.12 GiB | 2/3 |
| raw llama-server, same flags | 1 | 494 ms | 214.1 | 102.3 | 2.15 GiB | 2/3 |
| llmario → MLX (latency, 8k ctx) | 1 | 535 ms | 224.5 | 102.6 | 3.18 GiB | 3/3 |
| Ollama 0.34.2 defaults (40k ctx) | 1 | 482 ms | 208.2 | 97.6 | 6.42–11.66 GiB | 2/3 |
| llmario → llama.cpp (40k ctx, = Ollama) | 1 | 498 ms | 203.9 | 96.8 | 5.65 GiB | 2/3 |
| llmario → llama.cpp (balanced) | 4 | 1,877 ms | 70.0 | 113.8 | 4.78 GiB | – |
| llmario → MLX (balanced) | 4 | 1,668 ms | 71.1 | 117.7 | 8.51 GiB | – |
| Ollama 0.34.2 defaults (1 slot) | 4 | 4,880 ms | 200.5 | 88.1 | 9.42 GiB | – |
| Ollama, `NUM_PARALLEL=4`, 8k ctx (= balanced) | 4 | 1,326 ms | 56.6 | 104.5 | 7.01 GiB | – |

What this supports, and what it does not:
- **Gateway overhead is not measurable:** llmario vs the same llama-server without llmario.
- **Single-request speed is at parity with Ollama** (within run-to-run noise; MLX decodes ~8%
  faster). No single-request speed win is claimed.
- **At concurrency 4 the win is mostly Ollama's defaults.** vs defaults (`-np 1`): +29–34%
  aggregate throughput, ~2.7x lower median TTFT. vs Ollama tuned to the same 4 × 8k slots:
  only +9–13% throughput (below the 15% bar), and **Ollama's median TTFT is better** (1.33 s vs
  1.67–1.88 s, not yet explained; micro-batch size ruled out).
- **Lower memory at equal context (llama.cpp path):** 4.78 vs 7.01 GiB at 4 × 8k (−32%); 5.65 vs
  6.42 GiB right after load and 11.66 GiB after sustained use at 1 × 40k. Ollama leaves llama.cpp's 8 GiB host prompt cache uncapped;
  llmario caps it at 1 GiB. The default-profile saving (2.12 GiB) comes from sizing context to the
  workload, not from a faster engine.
- **llmario's estimates bounded every measured peak** (estimate / peak = 1.18–1.79x). An earlier
  MLX estimate was exceeded 2.3x; the root cause is fixed and documented in the ADR.

## Support status

| Path | Status |
|---|---|
| Apple Silicon + MLX-LM | ✅ validated (M4 Max, macOS 27.0.1) |
| Apple Silicon + llama.cpp (Metal) | ✅ validated (M4 Max, build 11146, Qwen3-1.7B Q4_K_M) |
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

Layout: `crates/{core,hardware,model_registry,supervisor,adapter_llamacpp,adapter_mlx,adapter_mock,api,benchmark,runtime,cli}` and `apps/desktop` (Tauri app).
Design: [ADR 0001](docs/adr/0001-architecture.md) · [milestones](docs/MILESTONES.md) ·
[benchmark plan](docs/BENCHMARK_PLAN.md) · [API](docs/API.md) ·
[troubleshooting](docs/TROUBLESHOOTING.md) · [contributing](CONTRIBUTING.md).

## License

Proposed: Apache-2.0 (declared in `Cargo.toml`). All 240 transitive crates are under
permissive or weak-copyleft licenses; see [docs/LICENSES.md](docs/LICENSES.md). Backends and
model weights carry their own licenses. llmario never redistributes weights.
