<p align="center">
  <img src="assets/icon-1024.png" alt="LLMario icon" width="112" height="112">
</p>

<h1 align="center">LLMario</h1>

<p align="center">
  <b>Run open LLMs on your own computer: a desktop chat app, a CLI, and an OpenAI-compatible local API.</b><br>
  Hardware-aware engine selection · memory planning before loading · reproducible benchmarks · private by default
</p>

---

> **This is the `beta` branch: LLMario Beta.** A separate app that installs and runs alongside
> LLMario without touching it: `LLMario Beta.app`, the `llmario-beta` command, data in
> `~/.llmario-beta`, settings from `LLMARIO_BETA_*` variables, and the local API on port **11501**.
> Beta builds are published as GitHub pre-releases. What it is for and the order of work:
> [docs/PHASES_16GB_AND_SPEED.md](docs/PHASES_16GB_AND_SPEED.md). The rest of this README
> describes production LLMario; in the beta, read `llmario` as `llmario-beta`.
>
> Reuse models you already downloaded for LLMario (registered in place, never copied):
> `llmario-beta model add ~/.llmario/models/<model-id>/<file>.gguf` (or an MLX model folder).

---

LLMario is an open-source local LLM runtime written in Rust. It does **not** reimplement
inference kernels. Instead it detects your hardware, picks and supervises a proven open-source
engine ([llama.cpp](https://github.com/ggml-org/llama.cpp) or Apple's
[MLX-LM](https://github.com/ml-explore/mlx-lm)), checks that a model fits in memory *before*
loading it, and gives you three ways to use it:

| | What it is | Start it |
|---|---|---|
| 🖥 **Desktop app** | A native window for macOS and Windows (preview) to download models and chat with them | `LLMario.app` / Start menu |
| ⌨️ **CLI** | `llmario doctor`, `model pull`, `run`, `serve`, `bench`, … | `llmario --help` |
| 🔌 **Local API** | OpenAI-compatible `/v1/chat/completions` on `127.0.0.1` | `llmario serve` |

🌐 **Website:** [llmario.com](https://llmario.com): how it works and a 3-step setup guide.

⬇️ **Download:** [macOS (.dmg)](https://github.com/mannysinghx/LLMario/releases/download/v0.2.0/LLMario-0.2.0-macos-universal.dmg) · [Windows, preview (.exe)](https://github.com/mannysinghx/LLMario/releases/download/v0.2.0/LLMario-0.2.0-windows-x64-setup.exe) · [Windows command line (.zip)](https://github.com/mannysinghx/LLMario/releases/download/v0.2.0/llmario-0.2.0-windows-x64.zip), with checksums on the [releases page](https://github.com/mannysinghx/LLMario/releases/latest). The Windows files are not code-signed yet: if SmartScreen warns, click **More info → Run anyway**.

Everything runs on your machine. Prompts and replies are never sent anywhere, and they never appear in logs.

**Status:** Phase 1 MVP, validated end to end on an Apple M4 Max (64 GB) with both MLX-LM and
llama.cpp. Linux/CUDA paths are implemented but not yet validated on real hardware; see
[Support status](#support-status).

## Contents

- [Desktop app](#desktop-app)
- [Install](#install)
- [Command line](#command-line)
- [Local API](#local-api)
- [Models](#models)
- [How it works](#how-it-works)
- [Performance](#performance)
- [Security and privacy](#security-and-privacy)
- [Configuration](#configuration)
- [Support status](#support-status)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License and credits](#license-and-credits)

## Desktop app

`LLMario.app` is a native window for chatting with local models. It is built with
[Tauri 2](https://tauri.app) and the system webview, and runs the full LLMario runtime inside the
app, so it needs no separate server and no Ollama.

**Features**

- **Chats** — sidebar with your conversations; `⌘N` starts a new chat. History is stored only on
  this computer and can be turned off or cleared in Settings.
- **Model picker** — every installed model shows its engine (MLX or llama.cpp) and whether it
  **fits in memory**, before you load it. Models that do not fit, or whose engine is missing, are
  disabled with the reason.
- **Streaming replies** — Markdown with code blocks and a copy button; a collapsible
  **Thinking** section for reasoning models (e.g. Qwen3); first-token latency, tokens/second,
  and token count under each reply.
- **Stop** — really cancels generation in the engine; it does not just hide the output.
- **Model library** — 37 researched open-model families (Qwen3.8/3.6/3.5, Gemma 4, gpt-oss,
  GLM-4.7, Nemotron 3.5, Mistral, Devstral, Phi-4, Llama, Granite 4.2, LFM2.5, Ornith, Olmo 3,
  SmolLM3 …) with 69 verified downloads. Search and filter by task (chat, reasoning, coding,
  agents, multilingual, long context, small), by *fits this computer* and by *runs on my engines*.
  Each model shows its publisher, release, size, license, what it is good at, the **exact Hugging
  Face repo and file** (copyable, pinned to a commit), the memory it needs **on your machine**, and
  whether your installed engine version can load its architecture. One recommended download per
  family; oversized downloads need a second click. Files are checksum-verified and matching blobs
  in your Hugging Face cache are reused.
- **Drag and drop your own models** — drop a `.gguf` file or an MLX model folder anywhere on
  the window. LLMario reads the model, checks it fits, registers it **in place** (never copied;
  removing it never deletes your files), then selects and loads it.
- **Settings** — performance profile (Latency / Balanced / Throughput), context length, max
  reply length, temperature, system prompt, show thinking, save history.
- **Light and dark mode** follow macOS.

**Using it**

1. Open LLMario. If no model is installed, the welcome screen offers the models recommended for
   your hardware. Click **Download**.
2. Pick the model in the top bar. It loads on your first message; the status pill shows
   *Loading…* and then *Ready · MLX · loaded in 1.4 s*.
3. Type and press **Enter** (**Shift+Enter** adds a new line). Press the ■ button to stop.
4. **Unload** frees the model's memory. Quitting the app stops every engine process.
5. To use a model you already have, drag its `.gguf` file (or its MLX folder: `config.json` +
   `.safetensors`) onto the window. Blobs named by a hash, like those in other runtimes' caches,
   get their real name from the file's header. The same file dropped twice is recognised.

## Install

### Requirements

- macOS 13+ on Apple Silicon (validated), Windows 10/11 x64 (preview; tested in CI), or Linux (CLI; unvalidated)
- [Rust](https://rustup.rs) 1.80+
- At least one inference engine:

| Engine | Use for | Install | Tested version |
|---|---|---|---|
| llama.cpp | GGUF models; Mac, Windows, Linux, CPU, NVIDIA | `brew install llama.cpp` (macOS), `winget install ggml.llamacpp` (Windows), or build [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | build 11146 (7fe450e19) |
| MLX-LM | MLX models; Apple Silicon only (usually fastest there) | `./scripts/setup-mlx-venv.sh` (pinned venv) or `pip install mlx-lm` | mlx-lm 0.31.3, mlx 0.32.2 |

### One-command install (macOS)

```bash
git clone https://github.com/mannysinghx/LLMario.git && cd LLMario && ./scripts/install-macos.sh
```

This checks prerequisites, then installs the `llmario` command and `LLMario.app` (`--cli-only` and
`--check` are available). Or build the pieces yourself:

### Build the CLI

```bash
git clone https://github.com/mannysinghx/LLMario.git
cd LLMario
cargo build --release
./target/release/llmario doctor
```

### Build the desktop app (macOS)

```bash
cargo install tauri-cli --version "^2" --locked      # once
cd apps/desktop/src-tauri
cargo tauri build --bundles app
ditto ../../../target/release/bundle/macos/LLMario.app /Applications/LLMario.app
```

This local build is ad-hoc signed, which macOS accepts on the machine that built it. For development,
`cargo run -p llmario-desktop` opens the window without bundling.

### Build a shareable app (signed + notarized)

```bash
scripts/release-macos.sh      # → dist/LLMario-<version>-macos-universal.dmg (+ .sha256)
```

The script makes a universal (Apple Silicon + Intel) app, signs it with your **Developer ID**
certificate (hardened runtime, secure timestamp), has Apple **notarize** it, **staples** the ticket
to the app and the DMG, and checks the result with Gatekeeper. Recipients can open it with no
warnings, even offline. They still need an engine (`brew install llama.cpp`, or `pip install mlx-lm`
on Apple Silicon). One-time setup (certificate and notary credentials) and troubleshooting:
[docs/RELEASING.md](docs/RELEASING.md).

## Command line

```bash
llmario doctor                          # hardware, engines, memory budget, which models fit
llmario model catalog                   # models available to download
llmario model pull qwen3-1.7b           # family name → best variant for this machine
llmario model pull qwen3-8b-gguf-q4km   # or an exact variant
llmario model list
llmario model fit qwen3-8b --context 32768   # memory estimate without loading
llmario run qwen3-1.7b "Explain KV caches in two sentences."
llmario run qwen3-1.7b                  # interactive chat (/reset, /exit)
llmario serve                           # OpenAI-compatible API on 127.0.0.1:11500
llmario bench -m qwen3-1.7b --concurrency 1,4
llmario config                          # effective settings and file locations
```

| Command | What it does |
|---|---|
| `doctor [--json]` | CPU features, RAM, GPU/unified-memory budget, engine versions, per-model fit, orphaned engine processes |
| `model catalog [filter] [--json]` | The model library for this machine: what each model is for, size, memory needed, engine support, ★ recommended variant |
| `model pull <id\|family> [--force] [--no-hf-cache]` | HTTPS download, SHA-256 / git-SHA-1 verified, atomic install; reuses matching Hugging Face cache blobs |
| `model add <path> [--name] [--family]` | Register a model already on disk (`.gguf` file, GGUF blob, or MLX folder) without copying |
| `model list [--json]` / `info <id>` | Installed models with source commit, license, quantization, checksums |
| `model remove <id>` | Deletes files LLMario downloaded; only unregisters files you added |
| `model verify <id>` | Re-hash files against the recorded SHA-256 |
| `model fit <name> [--profile] [--context]` | Weights + KV cache + overhead vs. your memory budget |
| `run <model> [prompt] [--system] [--max-tokens] [--temperature]` | Terminal chat through the same path as the API; prints TTFT and tok/s |
| `serve [--host] [--port] [--api-key] [--allow-remote] [--preload <model>]` | Start the API |
| `bench -m <model> [--url <base>] [--concurrency] [--runs] [--cache cold\|warm]` | Reproducible benchmark → JSON + Markdown report |

Common flags on `run`, `serve` and `bench`: `--profile latency|balanced|throughput`,
`--context <tokens>`, `--memory-limit-gb <GiB>`, `--backend llamacpp|mlx`. Add `-v` for logs
(never containing prompt text) or `--log-json` for JSON logs.

## Local API

`llmario serve` listens on `http://127.0.0.1:11500/v1`. Any OpenAI client works:

```bash
curl -s http://127.0.0.1:11500/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3-1.7b","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:11500/v1", api_key="unused")
reply = client.chat.completions.create(model="qwen3-1.7b",
                                       messages=[{"role": "user", "content": "Hello!"}])
print(reply.choices[0].message.content)
```

| Endpoint | Notes |
|---|---|
| `GET /v1/models`, `GET /v1/models/{id}` | Installed models, with format, backend, quantization, license, size, context, loaded state |
| `POST /v1/chat/completions` | Streaming and non-streaming. `model` may be an exact id or a family name; the response says which model and engine served it (`x-llmario-model`, `x-llmario-backend`) |
| `GET /healthz` | Liveness, no auth |
| `GET /metrics` | Prometheus: requests by outcome, tokens, TTFT and duration histograms, engine memory |

Supported: `messages` (text), `stream`, `stream_options.include_usage`, `max_tokens`,
`temperature`, `top_p`, `top_k`, `min_p`, `stop`, `seed`, penalties. **Not supported yet (clear
400 error):** tools/function calling, JSON mode / constrained output, images/audio, logprobs,
`n > 1`. Closing the connection cancels generation. Full details: [docs/API.md](docs/API.md).

## Models

### Model library

**[docs/MODELS.md](docs/MODELS.md)** lists every model with exact Hugging Face names. As of
2026-09-29 the library has 37 families and 69 downloads:

| Good at | Examples |
|---|---|
| General chat and reasoning | Qwen3.8 27B, Qwen3.6 27B / 35B-A3B, Gemma 4 (E2B–31B), gpt-oss 20B/120B, GLM-4.7 Flash, Mistral Small 3.2, Ministral 3 14B, Phi-4, Llama 3.1/3.2, Olmo 3, Ornith 1.5 |
| Coding and agents | Qwen3 Coder 30B-A3B, Devstral Small 2, gpt-oss, Granite 4.2 |
| Small and fast | Qwen3.5 0.8B–4B, Gemma 4 E2B/E4B, LFM2.5 (1.2B–8B-A1B), SmolLM3 3B, Phi-4 mini, Llama 3.2 1B/3B |
| Long context | Gemma 4 (256K), Qwen3.x (256K), Nemotron 3.5 Lightning (up to 1M), Devstral Small 2 (384K) |

```bash
llmario model catalog                 # grouped by family, with fit and engine support for this machine
llmario model catalog coding          # filter
llmario model pull qwen3.5-9b         # family → the recommended variant for this machine
```

The library is generated by [`scripts/catalog/build.py`](scripts/catalog/build.py) from a curated
list ([`scripts/catalog/sources.toml`](scripts/catalog/sources.toml)). For every entry it queries
the Hub, pins the commit, picks the 4-bit file (and every part of split files), and reads the
architecture, context length and attention shape from the GGUF header or `config.json`. So the
app can compute memory needs and engine compatibility **before** you download. At startup LLMario
reads which architectures your installed llama.cpp build and MLX-LM version support, and marks
models they cannot load yet (for example, Gemma 4 12B's MLX build needs a newer MLX-LM than 0.31.3).
Ask for a **family** and LLMario picks the best runnable variant; an **exact id** is never silently
swapped. License labels come from the model cards; "open weights" does not mean unrestricted use.

### Your own model files

In the desktop app, **drag the file or folder onto the window**. From the command line:

```bash
llmario model add ~/Downloads/Mistral-7B-Instruct-Q4_K_M.gguf
llmario model add ~/models/my-mlx-model/ --name my-model
```

GGUF files are recognised by extension or by their `GGUF` header (including blobs from other
runtimes' caches). MLX models are folders with `config.json` and `.safetensors` weights. Files are
hashed and registered in place, never copied, and `remove` never deletes them. Custom Python code
in a model folder is never executed. Some runtimes write GGUF variants that upstream llama.cpp
cannot load; LLMario then shows the engine's own error message.

### Where things live

`~/.llmario/` (override with `LLMARIO_HOME`): `config.toml`, `registry.toml` (installed models),
`models/` (downloads), `logs/` (engine logs), `bench/` (benchmark reports), `run/` (engine process
records for orphan detection).

## How it works

```text
 Desktop app (Tauri)      CLI (llmario)       Your apps (OpenAI SDK)
          │                    │                      │
          └────────────┬───────┴──────────────────────┘
                       ▼
        API gateway ─ loopback only, API key, Host check, field allowlist, SSE relay, metrics
                       ▼
        Planner ───── picks model variant + engine, enforces per-request context
                       ▼
        Supervisor ── memory plan → admission → launch / warm-up / crash relaunch / LRU unload
                       ▼
        Adapters ──── llama.cpp (llama-server)   ·   MLX-LM (mlx_lm.server)   ·   mock (CI)
                       ▼
        Engine child processes on 127.0.0.1:<random port>, model files from ~/.llmario/models
```

- **Engines run as separate processes.** A crash in an engine cannot take down LLMario. The next
  request relaunches it, up to 3 times in 10 minutes.
- **Memory is planned, not hoped for.** Weights + KV cache for all slots + engine extras (prompt
  caches, MLX buffer cache, Python runtime) are estimated from the GGUF header or `config.json`
  and compared with your budget (unified-memory GPU working set or VRAM, minus headroom). Loads
  that do not fit are refused with a context size that would.
- **Profiles are explicit values**, printed at startup and translated into flags each engine
  really supports:

| Profile | Parallel requests | Context per request | Best for |
|---|---:|---:|---|
| `latency` (default) | 1 | 8,192 | One person chatting |
| `balanced` | 4 | 8,192 | A chat plus apps/agents using the API |
| `throughput` | 16 | 4,096 | Many concurrent requests |

- **Engine-specific fixes.** Measured examples: `mlx_lm.server` disables batching for requests
  with a `seed`, so LLMario drops it when decoding is greedy (the output is identical). MLX's
  allocator cache is capped so memory stays within the estimate.

Design record: [ADR 0001](docs/adr/0001-architecture.md) · crate layout under [Development](#development).

## Performance

Measured on an Apple M4 Max, 64 GB, AC power. Qwen3-1.7B 4-bit, built-in benchmark suite
(55 / 1.8k / 4.7k-token prompts, 128 output tokens), temperature 0, 3 runs per prompt, cold prefix
cache. Full reports with method and caveats: [benchmarks/results/2026-09-29/SUMMARY.md](benchmarks/results/2026-09-29/SUMMARY.md).

| Target | Concurrent requests | Time to first token (median) | Decode tok/s per request | Aggregate tok/s | Peak memory |
|---|---:|---:|---:|---:|---:|
| LLMario → llama.cpp | 1 | 495 ms | 217 | 102 | 2.12 GiB |
| raw llama-server, same flags | 1 | 494 ms | 214 | 102 | 2.15 GiB |
| LLMario → MLX | 1 | 535 ms | 225 | 103 | 3.18 GiB |
| Ollama 0.34.2 defaults | 1 | 482 ms | 208 | 98 | 6.4–11.7 GiB |
| LLMario → llama.cpp (balanced) | 4 | 1,877 ms | 70 | 114 | 4.78 GiB |
| LLMario → MLX (balanced) | 4 | 1,668 ms | 71 | 118 | 8.51 GiB |
| Ollama defaults (1 slot) | 4 | 4,880 ms | 201 | 88 | 9.42 GiB |
| Ollama, `NUM_PARALLEL=4`, 8k context | 4 | 1,326 ms | 57 | 105 | 7.01 GiB |

What this does and does not show:

- LLMario's gateway adds **no measurable overhead** over calling the engine directly.
- **Single-request speed is at parity** with Ollama, which also uses llama.cpp. No speed win is claimed.
- With 4 concurrent requests LLMario beats **Ollama's defaults** by 29–34% in throughput. Against
  Ollama **tuned** to the same 4 × 8k slots, the gain is only 9–13%, and Ollama has the better
  median time to first token (not yet explained).
- **Memory is lower at equal context:** −32% at 4 × 8k on the llama.cpp path. Ollama leaves
  llama.cpp's 8 GiB host prompt cache uncapped; LLMario caps it at 1 GiB.
- LLMario's memory **estimates bounded every measured peak** (estimate ÷ peak = 1.18–1.79×).

Reproduce: `llmario bench -m <model>` or `llmario bench --url <any OpenAI-compatible base> -m <name>`.
Rules for fair comparisons: [docs/BENCHMARK_PLAN.md](docs/BENCHMARK_PLAN.md).

## Security and privacy

- **Local only by default.** The API binds to `127.0.0.1`. Binding any other address requires
  `--allow-remote` **and** an API key. `/metrics` is also protected by the key.
- **DNS-rebinding protection.** In local mode, requests whose `Host` header is not a loopback
  name are rejected. Only JSON bodies are accepted, so web pages cannot post forms to the API.
- **Nothing reaches an engine unchecked.** The gateway rebuilds each request from an allowlist
  and always replaces `model` (MLX would otherwise load, and even download, any model a request
  names). Engines run with Hugging Face offline mode. Remote model code is never executed.
- **Verified downloads.** HTTPS only, checksums checked before files become visible, path
  traversal rejected, disk space checked first.
- **No prompt logging, no telemetry.** The only network traffic is model downloads you start.
- **Desktop app:** no remote assets, a strict Content-Security-Policy, only core IPC permissions
  (no filesystem, shell or HTTP plugins), and model output rendered by an escape-first Markdown
  renderer with tests against HTML injection.

Threat model: [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).

## Configuration

Precedence: built-in defaults < `~/.llmario/config.toml` < `LLMARIO_*` environment variables <
command-line flags. `llmario config` prints the effective values.

```toml
[server]
host = "127.0.0.1"
port = 11500
allow_remote = false
# api_key = "…"            # prefer the LLMARIO_API_KEY environment variable

[runtime]
profile = "latency"         # latency | balanced | throughput
# context = 8192            # per-request context; default comes from the profile
max_loaded_models = 1
# memory_headroom_gb = 6.4  # default: max(2 GiB, 10% of RAM)
# memory_limit_gb = 24
queue_timeout_secs = 120
request_timeout_secs = 900
engine_start_timeout_secs = 300
idle_unload_secs = 0        # 0 = keep loaded
max_restarts = 3
kv_accounting = "per-layer" # per-layer | conservative (every layer counted as full attention)
memory_profile = "standard" # standard | small | auto (small on 16 GB or less): smaller prompt caches

[backends]
# prefer = "mlx"            # llamacpp | mlx
[backends.llamacpp]
# server_path = "/opt/homebrew/bin/llama-server"
# gpu_layers = 999
kv_cache_type = "f16"       # f16 | q8_0 (8-bit KV cache: 17/32 of the memory)
speculative = "off"         # off | ngram | mtp | draft: guess tokens ahead (same output; faster
                            # when the answer repeats its input, can be slower on free writing)
# draft_model = "qwen3.5-0.8b-gguf-q4_0"   # for speculative = "draft"
# draft_tokens = 1          # tokens guessed per step (default: 1 for mtp, 3 for draft)
extra_args = []
[backends.mlx]
# python = "~/.llmario/venvs/mlx/bin/python"
# draft_model = "qwen3-1.7b-mlx-4bit"      # speculative decoding, one request at a time
extra_args = []
```

| Environment variable | Purpose |
|---|---|
| `LLMARIO_HOME` | Data directory (default `~/.llmario`) |
| `LLMARIO_HOST`, `LLMARIO_PORT`, `LLMARIO_API_KEY` | Server settings |
| `LLMARIO_PROFILE`, `LLMARIO_CONTEXT` | Runtime profile and context |
| `LLMARIO_LLAMA_SERVER`, `LLMARIO_MLX_PYTHON` | Engine locations |
| `LLMARIO_LOG` | Log filter, e.g. `debug` |
| `HF_TOKEN`, `HF_ENDPOINT` | Gated models / Hub mirror for `model pull` |

## Support status

| Platform | Engine | Status |
|---|---|---|
| macOS, Apple Silicon | MLX-LM | ✅ validated (M4 Max, macOS 27) |
| macOS, Apple Silicon | llama.cpp (Metal) | ✅ validated (build 11146) |
| Linux x86_64 + NVIDIA | llama.cpp (CUDA) | 🔬 implemented: `nvidia-smi` detection, hybrid CPU/GPU offload planning. Unvalidated |
| Linux / macOS, CPU only | llama.cpp | 🔬 implemented, unvalidated |
| Windows x64, CPU | llama.cpp | 🟡 preview: download, chat and engine cleanup pass in CI (Windows Server 2025, build 11146); not benchmarked or used day to day |
| Windows x64 + NVIDIA | llama.cpp | 🔬 implemented (`nvidia-smi` detection, offload planning), unvalidated |
| AMD ROCm, Intel | — | detection only, no support claimed |
| vLLM / SGLang | — | planned (Phase 2) |
| Desktop app | macOS | ✅ used for real chats |
| Desktop app | Windows | 🟡 preview: installer installs, app launches and uninstalls in CI; not yet used on a real PC |
| Desktop app | Linux | not set up |

Machine-readable: [docs/support-matrix.toml](docs/support-matrix.toml).

## Development

```text
crates/
  core            config, paths, error codes, profiles
  hardware        CPU/GPU/memory detection, process footprint
  model_registry  catalog, GGUF/MLX inspection, verified downloads
  supervisor      memory planner, engine selection, process lifecycle, admission
  adapter_*       llama.cpp, MLX-LM and mock engine adapters
  api             OpenAI-compatible gateway, streaming relay, metrics
  runtime         shared runtime assembly + streaming chat client (CLI and desktop)
  benchmark       benchmark harness and reports
  cli             the `llmario` binary
apps/desktop/
  src-tauri       Tauri 2 app (Rust commands, app state)
  ui              HTML/CSS/JS interface (no bundler)
  preview         browser preview of the UI with a simulated backend (not shipped)
benchmarks/       suites and published results (no model weights)
docs/             ADR, threat model, API, benchmark plan, milestones, support matrix
scripts/          installer, catalog builder, release, MLX venv setup, benchmark tables, icon
website/          public single-page site (static, Caddy on Railway)
```

```bash
cargo test                                     # unit + end-to-end tests (real mock engine process, no GPU)
cargo clippy --all-targets -- -D warnings
cargo fmt --all
node --test apps/desktop/ui/markdown.test.mjs  # UI renderer security tests
cargo build -p llmario-desktop                 # desktop app (macOS)
```

Adding an engine means implementing `EngineAdapter` (probe, formats, command line) against IPC
contract v1: OpenAI-compatible HTTP on a loopback port. See [CONTRIBUTING.md](CONTRIBUTING.md).
Performance claims need a reproducible `llmario bench` report, including neutral and negative results.

## Troubleshooting

| Problem | Fix |
|---|---|
| `✗ mlx … not found` / `✗ llamacpp … not found` | Install the engine (see [Install](#install)), then run `llmario doctor` |
| *Model does not fit* (507 / red badge) | Lower the context, use the `latency` profile, or choose a smaller or lower-bit model |
| *Server busy* (503) | All slots busy; wait, or use the `balanced` profile |
| Reasoning model gives an empty answer | Its thinking used up the reply budget. Raise max reply length, or turn thinking off (Qwen3: add `/no_think`) |
| An engine fails to start | The error includes the engine's last log lines; the full log is in `~/.llmario/logs/` |
| `doctor` lists orphaned engines | An LLMario process was killed hard. Run `kill <pid>` if you do not need the engine |

More: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## Roadmap

- Bundle llama.cpp inside the app so recipients need no separate engine install
- Linux desktop build; Windows code signing and AMD/Intel GPU offload on Windows
- vLLM / SGLang adapter for multi-user NVIDIA serving
- Cached auto-tuning per hardware, engine build, model hash and profile
- Tool calling and JSON mode where engines support them

## License and credits

LLMario is licensed under the [Apache License 2.0](LICENSE); see also [NOTICE](NOTICE). All 240
Rust dependencies are permissively licensed; see [docs/LICENSES.md](docs/LICENSES.md) and
[deny.toml](deny.toml).

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT),
[MLX](https://github.com/ml-explore/mlx) and [MLX-LM](https://github.com/ml-explore/mlx-lm) (MIT),
[Tauri](https://tauri.app) (MIT/Apache-2.0), [axum](https://github.com/tokio-rs/axum) and
[tokio](https://tokio.rs) (MIT). Models are published by their authors under their own licenses;
LLMario never redistributes model weights.
