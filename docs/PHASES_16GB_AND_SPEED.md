# Phases: 16 GB machines and faster tokens (LLMario Beta)

**Status:** proposed · **Date:** 2026-10-07 · **Scope:** planner, engine adapters, catalog, benchmark.
No code has changed yet; this is the plan and the evidence behind its order.

**Delivery:** everything here ships first in **LLMario Beta**, a separate app that installs and runs
alongside production. The production app, its releases and llmario.com are not changed. A phase
reaches production only when it [graduates](#graduating-a-phase-to-production).

Two goals:

1. Run good models **comfortably on 16 GB machines**.
2. Generate **more tokens per second**.

Two findings shape the order:

- **The memory planner over-counts KV cache** for most recent models (measured 4× on Qwen3.8 27B),
  so models that would fit on 16 GB are refused today.
- **Decode speed is limited by memory bandwidth, and the engines already reach about 82% of it**
  (measured). Speed has to come from reading fewer bytes per token and from producing several
  tokens per read, not from new kernels or an in-process engine.

## Recommended order

| # | Phase | Main outcome | Depends on | Relative effort | Risk |
|---|---|---|---|---|---|
| 0 | [LLMario Beta app](#phase-0--llmario-beta-app) ✅ built | A beta that installs, runs and stores data separately from production | — | S–M | Low |
| 1 | [Measurement baseline](#phase-1--measurement-baseline) 🟡 harness built | Numbers we can trust, on real 16 GB hardware | 0 | S | Low |
| 2 | [Accurate KV accounting](#phase-2--accurate-kv-accounting) 🟡 built | Gemma 4 12B and Qwen3.5 9B (MLX) fit 16 GB Macs | 1 | M | Medium |
| 3 | [Small-machine memory profile](#phase-3--small-machine-memory-profile) 🟡 built, opt-in | Ministral 3 14B and Phi-4 fit; cheaper long-context decode | 1, 2 | S–M | Low–medium |
| 4 | [Speculative decoding](#phase-4--speculative-decoding) 🟡 built (MTP on by default) | More than one token per weight read (+12–30% measured with a draft model) | 1 | M | Low |
| 5 | [16 GB catalog tier](#phase-5--16-gb-catalog-tier) 🟡 built | Fast MoE and 3-bit options recommended by default | 2, 3, 4 | S | Low |
| 6 | [Speed planner](#phase-6--speed-planner) 🟡 built | "~N tok/s on this computer" and automatic choice of the fastest setup | 1–5 | L | Medium |
| 7 | [GPU/CPU split and MoE offload](#phase-7--gpucpu-split-and-moe-expert-offload) 🟡 built, opt-in | gpt-oss-20b and 27B at 3-bit run on 16 GB Macs (tight, slower) | 1, 2 | M | Medium–high |
| 8 | [Multi-machine sharding](#phase-8--multi-machine-sharding-optional) (optional) 📝 design only | Models larger than one machine | 7, threat-model update | L | High |

Why this order:

- The beta shell comes first, so no feature work can reach production users by accident.
- The cheapest, highest-value, lowest-risk work follows, and every phase is accepted or rejected on
  Phase 1 measurements.
- Work that is slower by nature (7) or changes the security model (8) comes last.

## Rules for every phase

- **Beta first.** Work happens on a long-lived `beta` branch. `main`, production releases and
  llmario.com stay unchanged until a phase graduates.
- **Additive.** New behavior sits behind a config value; defaults change only after a measurement
  supports it.
- **Measured.** A phase is done only with an `llmario bench` report on 16 GB hardware showing: the
  estimate is at or above the measured peak, tok/s, and unchanged quality checks.
- **One batched push per phase** to `beta`, with [OPS_LOG.md](OPS_LOG.md) entries for anything outside
  the working tree.

## Phase 0 — LLMario Beta app

**Goal:** a second app that installs, runs and keeps its data separately from production, before any
feature work.

**Status (2026-10-07): built on `beta`; real-window checks remain.** See [Phase 0 results](#phase-0-results).

**What must differ.** These are the points where a beta build would collide with production today:

| What | Production | Beta | Where it is set |
|---|---|---|---|
| Desktop app identity. Chat history and settings live in the webview's storage, which is keyed by this identity, so a new identity keeps them apart. | `dev.llmario.desktop`, "LLMario", `LLMario.app` | `dev.llmario.desktop.beta`, "LLMario Beta", `LLMario Beta.app` | [tauri.conf.json](../apps/desktop/src-tauri/tauri.conf.json), edited on the beta branch so development builds are separate too |
| Data home: registry, config, downloads, logs, engine records | `~/.llmario` | `~/.llmario-beta` | [paths.rs](../crates/core/src/paths.rs) (`LLMARIO_HOME` already overrides it) |
| Environment variables | `LLMARIO_*` | `LLMARIO_BETA_*`; production's variables are ignored, so an exported `LLMARIO_HOME` cannot point the beta at production data | `ENV_PREFIX` in [core/src/lib.rs](../crates/core/src/lib.rs) |
| API port | 11500 | 11501 | `DEFAULT_PORT` in [core/src/lib.rs](../crates/core/src/lib.rs) |
| Command-line package and binary | `llmario` | `llmario-beta`, so `cargo install` tracks it separately | [crates/cli/Cargo.toml](../crates/cli/Cargo.toml) |
| Desktop binary | `llmario-desktop` | `llmario-desktop-beta` | [apps/desktop/src-tauri/Cargo.toml](../apps/desktop/src-tauri/Cargo.toml) |
| Installer | deletes and replaces `/Applications/LLMario.app` | must never touch `LLMario.app` | [install-macos.sh:60](../scripts/install-macos.sh) |
| Release files | `LLMario-<version>-…`, GitHub release marked Latest | `LLMario-Beta-<version>-…`, published as a GitHub **pre-release** so `/releases/latest`, which the README and llmario.com link to, stays on production | [release-macos.sh:81](../scripts/release-macos.sh) |
| Website | llmario.com | not linked; no `website/` changes on the beta branch | Railway deploys `website/**` from `main` only |

**Also**

- **Branch and CI:** work on a long-lived `beta` branch. CI runs only on pushes to `main` and on pull
  requests ([ci.yml](../.github/workflows/ci.yml), [windows.yml](../.github/workflows/windows.yml)),
  so add `beta` to the push triggers on the beta branch.
- **Keep the beta identity isolated.** Put it in a few small files and commits, so graduating
  phases can merge to `main` without it.
- **Share engines and model files without writing to production:**
  - The beta uses the same llama.cpp and MLX installs.
  - Models already downloaded for production can be registered in the beta with
    `llmario-beta model add <path>`. Files are registered in place and never copied or deleted, so
    nothing is downloaded twice.
  - The beta never writes to `~/.llmario`.
  - The managed MLX venv defaults to the data home, so point `backends.mlx.python` at the existing one.
- **Running both apps at once:** each app plans memory only for its own engines. On a 16 GB machine,
  models loaded in both apps at the same time can exceed memory. The beta should warn when production
  engines are running; it can read production's engine records in `~/.llmario/run/` without writing.
- **Version:** `0.3.0-beta.N`. Check that the bundler accepts a pre-release version on both macOS and
  Windows.

**Exit**

- Production and beta are installed side by side on one Mac.
- Both launch and chat, with separate model lists, history and settings.
- Removing the beta leaves production working.
- Production's `llmario doctor` does not report beta engines as orphans.

**Undo:** delete `LLMario Beta.app` and `~/.llmario-beta`. Production is untouched.

### Phase 0 results

Three commits on `beta`:

1. **Centralize app identity** (no behavior change; can merge to `main` as is). The env prefix,
   default port and command name in hints come from constants in `llmario-core`.
2. **LLMario Beta identity** (beta only). Everything in the table above, plus the installer, the
   release script, the MLX venv script, CI on pushes to `beta`, the Windows workflow, and a README
   notice.
3. **Warn when the other edition has models loaded.**
   - `doctor`, engine launches and the desktop app (as a toast) say, for example, "LLMario is also
     running 1 model (q17, 966 MiB)".
   - Production's engine records are only read; stale ones are never removed.

Verified:

- fmt, clippy with warnings as errors (workspace and desktop), `cargo test` (83 passed), UI
  renderer tests.
- The release bundle is `LLMario Beta.app`: identifier `dev.llmario.desktop.beta`, executable
  `llmario-desktop-beta`, version `0.3.0-beta.1`, valid ad-hoc signature.
- **Side by side under one scratch `HOME`**, using production `llmario` built from `main` together
  with `llmario-beta`:
  - Both served the same GGUF at the same time, on ports 11500 and 11501.
  - Each kept its own registry, logs and engine records.
  - Neither `doctor` reported the other's engine as an orphan.
  - The beta ignored `LLMARIO_PORT` and honored `LLMARIO_BETA_PORT`.
  - No engine was left after shutdown.
  - The real `~/.llmario` was unchanged.
- Pre-release versions: tauri-bundler 2.10 writes `0.3.0.0` for the Windows installer's numeric
  version and passes the string unchanged into the macOS Info.plist.

Still to check before Phase 0 is closed:

- Install `LLMario Beta.app` next to `LLMario.app`. Chat in both windows and confirm chats and
  settings stay separate. Remove the beta and confirm production still works.
- The Windows workflow on the first push to `beta`: installer, launch and uninstall under the beta
  names.
- That Apple notarization accepts `0.3.0-beta.1` as `CFBundleShortVersionString`, on the first beta
  release.

## Phase 1 — Measurement baseline

**Goal:** trustworthy numbers before anything changes.

**Status (2026-10-07): harness built and validated on `beta`; the 16 GB run is still to do.** See
[Phase 1 results](#phase-1-results).

**Why first after the shell:** during this investigation, a llama.cpp run on a busy machine (load
average 26) gave 92–178 tok/s for the same model and flags that measured 217 tok/s in the
[published benchmark](../benchmarks/results/2026-09-29/SUMMARY.md). llama.cpp's `ngram-mod`
speculation keeps a cache across requests, so a repeated prompt "decoded" at over 1,000 tok/s by
replaying the previous answer. Without a protocol, later phases would be judged on noise.

**Work**

- Get 16 GB test hardware: a base M-series Mac (M2, M3 or M4, 16 GB). For Phase 7, optionally a
  16 GB PC with an 8 GB NVIDIA GPU. `--memory-limit-gb` on a larger Mac does not reproduce the macOS
  GPU limit or real memory pressure.
- Benchmark protocol: quiet machine (check load average first), AC power, warm-up as long as the test,
  at least 3 runs, cold prefix cache (already the default), and unique prompts whenever a mode keeps a
  cross-request cache.
- Reference set: Qwen3 8B, Qwen3.5 9B, Gemma 4 12B, Ministral 3 14B, gpt-oss-20b, as GGUF and MLX
  where both exist.
- Extend the bench report with effective bandwidth (bytes read per token × tok/s) and, when the engine
  reports it, speculative acceptance (llama-server returns `draft_n` and `draft_n_accepted` in
  `timings`).
- Benchmark production and beta on the same hardware, so every later change is compared with
  production.

**Exit:** a baseline for the reference set on 16 GB hardware in `benchmarks/results/`, for production
and for the beta shell.

### Phase 1 results

Built on `beta`:

- **Quiet-machine guard:** `bench` refuses to run on a busy machine unless `--allow-busy`, and the
  report records the load.
- **Warm-up over every prompt length.**
- **Effective bandwidth and draft acceptance** per concurrency level, in report schema 2.
- **Reference set:** `benchmarks/reference-16gb.toml`, with a test that every id is in the catalog.
- **Runner:** `scripts/bench-baseline.sh`.
  - Checks AC power and load.
  - Benchmarks every installed reference model in the beta, and in production with
    `--production`.
  - Records refusals as results.
  - Writes `SUMMARY.md` and `environment.json` with no home paths.
  - Runs on the stock macOS bash 3.2 and python3 3.9.

Validated by a dry run on the M4 Max development machine with Qwen3 1.7B GGUF, in a scratch home:

- Both editions were measured and summarized.
- The beta report shows load, bandwidth (205 GB/s at 185 tok/s) and the busy flag.
- The outputs contain no personal paths.
- The production comparison is not meaningful there. The runs were at load 5–8, and production
  still warms up with only the short prompt.

**First full run** (2026-10-07, M4 Max development machine, not 16 GB hardware), in
[benchmarks/results/2026-10-07-apple-m4-max-64gb](../benchmarks/results/2026-10-07-apple-m4-max-64gb/SUMMARY.md).
Three reference models (GGUF), beta and production, concurrency 1, quiet machine (load 4.6 at
start):

| Model | Beta decode tok/s | Production decode tok/s | Beta eff. GB/s | Planner estimate | Quality |
|---|---:|---:|---:|---:|---|
| Qwen3.5 9B Q4_K_M | 67.4 | 57.0 | 383 | 8.32 GiB | 3/3 both |
| Gemma 4 12B Q4_0 | 48.9 | 47.8 | 341 | 14.62 GiB | 3/3 both |
| gpt-oss-20b MXFP4 | 90.5 | 75.5 | 1,096 (MoE: overstated) | 14.18 GiB | 3/3 both |

The beta–production gap is the harness, not inference. Production warms up with one short request,
so its first measured prompts are slower. On medium and long prompts the two are within about 2–9%.

Open items:

- **The 16 GB run** still needs a 16 GB Mac. Three reference GGUFs are on the external drive, linked
  as the beta's models folder (`~/.llmario-beta/models`).
- **Peak memory leaves out memory-mapped weights for llama.cpp.** gpt-oss-20b used 0.49 GiB right
  after loading 11.28 GiB of weights.
  - Likely cause (not yet verified): llama.cpp maps the weight file into memory, and macOS does not
    count those pages in the process footprint.
  - This blocks Phase 2's "estimate at or above measured peak" check. Measure system-level memory
    while a model loads, or run llama.cpp without memory mapping, before calibrating.
- **Consecutive runs inherit load.** Gemma 4 12B started at load 7.9, raised by the previous model's
  run. Add a cool-down between models in `scripts/bench-baseline.sh`.
- **Comparing with production:** use the medium and long prompts, or measure production through the
  beta's harness (`llmario-beta bench --url`), so both get the same warm-up.
- **Calibrate the busy threshold.** At load 8.0 on 16 cores the guard let the dry run through,
  and those numbers were 9–15% below the quiet published baseline. Measure one model at several
  loads before tightening the threshold.
- `yoke-derive` on `main`: fixed by PR #2 (merged as `9932492`).

## Phase 2 — Accurate KV accounting

**Goal:** the planner counts only caches that actually grow.

**Problem:** [memory.rs:83](../crates/supervisor/src/memory.rs) and `ModelShape::kv_bytes_per_token`
([manifest.rs:20](../crates/model_registry/src/manifest.rs)) assume every layer stores a full KV
cache at the largest head count and head size. The engines do not:

- llama.cpp keeps a window-sized cache for sliding-window layers.
- MLX-LM uses `RotatingKVCache` for those layers.
- Linear-attention layers keep a fixed-size state.

The catalog builder also takes the maximum of per-layer values (`as_int` in
[build.py](../scripts/catalog/build.py)).

| Model | Planned KV | Real KV | Source |
|---|---|---|---|
| Qwen3.8 27B | 256 KiB/token | 64 KiB/token + 147 MiB fixed | **measured** (MLX) |
| Gemma 4 12B | 6.0 GiB at 8k | ~0.6 GiB at 8k | computed: 40 of 48 layers sliding, window 1024 |
| Qwen3.5 9B | 128 KiB/token | 32 KiB/token + fixed state | computed: 8 of 32 layers full attention |
| gpt-oss-20b | 48 KiB/token | ~24 KiB/token | computed: 12 of 24 layers sliding, window 128 |
| OLMo 3 7B | 4.0 GiB at 8k | ~2.7 GiB at 8k | computed: 24 of 32 layers sliding, window 4096 |

**Work**

- Catalog builder (`gguf_facts`, `mlx_facts` in [build.py](../scripts/catalog/build.py)): record each
  layer's attention type, the sliding window, and the linear-state size. Sources are `layer_types`,
  `sliding_window` and `full_attention_interval` in `config.json`, and the per-layer arrays in GGUF
  headers. [gguf.rs](../crates/model_registry/src/gguf.rs) already keeps those arrays.
- `ModelShape`: add optional fields with serde defaults, so existing registry files and the catalog
  stay valid.
- [inspect.rs](../crates/model_registry/src/inspect.rs): the same for user-added models.
- `memory.rs`: KV = full layers × context + sliding layers × min(context, window + micro-batch) +
  fixed linear state. The MLX prompt-cache bound
  ([adapter_mlx/src/lib.rs:37](../crates/adapter_mlx/src/lib.rs)) must use the same function.
- Keep the conservative fallback when layer types are unknown, and keep the worst case *per layer*
  (the comment at `gguf.rs:34` explains why the planner wants worst cases).

**Exit**

- On the reference set, the estimate is at or above the measured peak everywhere.
- For hybrid and sliding-window models, estimate ÷ peak is about 1.4 or less.
- The beta catalog marks Gemma 4 12B and Qwen3.5 9B (MLX) as fitting a 16 GB Mac.

**Risk and undo:** an under-estimate means an engine that runs out of memory or swaps. Mitigations:
calibrate against bench peaks, keep a config switch back to the old formula, and the change reverts
as one commit.

**Status (2026-10-07):** implemented on `beta` and checked on the M4 Max development machine; not
yet on 16 GB hardware.

- **Memory measurement fixed first.** On macOS the engine's memory is now the larger of its
  footprint and its resident size. The footprint alone left out llama.cpp's memory-mapped weights
  (gpt-oss-20b: 0.48 GiB reported, 11.49 GiB resident).
- **Per-layer layouts** are read from GGUF headers and MLX `config.json`
  ([layout.rs](../crates/model_registry/src/layout.rs)). They match what the engines allocate:

  | Model | Planned now | Engine allocated |
  |---|---|---|
  | Qwen3.5 9B (GGUF, 8k) | 256 MiB + 50.25 MiB state | 256 MiB + 50.25 MiB (llama.cpp) |
  | Gemma 4 12B (GGUF, 8k) | 128 + 480 MiB | 128 + 480 MiB (llama.cpp) |
  | gpt-oss-20b (GGUF, 8k) | 192 + 18 MiB | 192 + 18 MiB (llama.cpp) |
  | Qwen3.8 27B (MLX) | 64 KiB/token + 146.8 MiB | 64 KiB/token + 146.8 MiB (mlx-lm) |

- **llama.cpp context checkpoints** are capped at 2 per slot (`--ctx-checkpoints 2`) and counted.
  The default 32 added 2.54 GiB to Gemma 4 12B in a chat; with 2 it added 0.66 GiB, and follow-up
  turns reused the prompt just as well.
- **Catalog:** 35 of 69 entries gained a layout (`scripts/catalog/build.py --layouts`, read at each
  entry's pinned revision). The rest are plain full attention, or a layout the planner cannot size
  exactly; those keep the old formula.
- **Switch back:** `[runtime] kv_accounting = "conservative"`. `model refresh` re-reads layouts for
  models registered before this change.

Exit check:

| Criterion | Result |
|---|---|
| Estimate at or above measured peak | Met for the three GGUF reference models: estimate ÷ peak 1.10 (Qwen3.5 9B), 1.08 (Gemma 4 12B), 1.16 (gpt-oss-20b) ([results](../benchmarks/results/2026-10-07-apple-m4-max-64gb-phase2-capped/SUMMARY.md)). MLX reference models not yet measured. |
| Estimate ÷ peak about 1.4 or less | Met: 1.08–1.16. |
| Gemma 4 12B and Qwen3.5 9B (MLX) fit a 16 GB Mac, from the catalog | Qwen3.5 9B (MLX): met, 9.01 GiB (was 11.12, refused). Gemma 4 12B (MLX): **not met**, 12.11 GiB (was 17.92) against 10.67 GiB; mlx-lm 0.31.3 also cannot load its `gemma4_unified` model type. The GGUF Gemma 4 12B fits: 10.16 GiB planned, 9.37 GiB measured. |

Open items:

- **MLX prompt-cache bound is probably high for sliding-window models.** Each saved entry is counted
  at window + one 2048-token prefill step. mlx-lm's code trims a rotating cache back to the window
  after the first generated token, so entries likely hold only the window (read from the code, not
  measured). Lowering it would bring Gemma 4 12B (MLX) to about 10.9 GiB, still over 10.67. Measure
  before changing.
- **Layouts that over-count (safe direction):**
  - Gemma 4 E2B/E4B share KV across layers, and every layer is counted.
  - Nemotron 3.5 Lightning (GGUF) counts its feed-forward-only layers as recurrent state. llama.cpp
    counts only the Mamba layers (from llama.cpp's rule; not measured).
- **Several slots:** sliding-window caches are assumed to hold one window per slot. Not measured.
- **Older beta builds:** the installed `LLMario Beta.app` (0.3.0-beta.1) keeps the layout fields
  when it reads the registry but drops them if it rewrites it. `model refresh` restores them.

## Phase 3 — Small-machine memory profile

**Goal:** fixed overheads scale with RAM, and the KV cache can be 8-bit.

**Work** (each is its own config value; defaults are set from Phase 1 measurements):

- **llama.cpp 8-bit KV cache:** `-ctk q8_0 -ctv q8_0` (supported by the tested build 11146). The
  fixed 2-byte element size in the planner (`KV_ELEM_BYTES`,
  [memory.rs:17](../crates/supervisor/src/memory.rs)) becomes per-type. Verify on Metal and CPU that
  a quantized V cache works with `--flash-attn auto`, which the adapter already passes.
- **Host prompt cache:** scale `--cache-ram` (fixed at 1 GiB in
  [adapter_llamacpp/src/lib.rs:20](../crates/adapter_llamacpp/src/lib.rs)), for example to 256 MiB on
  machines with 16 GB or less.
- **MLX:** on machines with 16 GB or less, use 1 prompt-cache entry and a 512 MiB buffer cache
  ([adapter_mlx/src/lib.rs:31](../crates/adapter_mlx/src/lib.rs)). `mlx_lm.server` 0.31.3 has no
  KV-quantization option.
- **"Comfortable" target:** on machines with 16 GB or less, a planned total of about 10 GiB or less.
  That stays inside the Mac GPU limit (10.67 GiB) and leaves about 6 GB for macOS and apps. Models
  between that and the hard budget show as "fits, tight".

**Expected (computed):** Ministral 3 14B goes from 11.2 to 9.8 GiB; Phi-4 from 11.6 to 10.1 GiB.

**Exit:** measured peak at or below the estimate; quality checks unchanged with 8-bit KV; tok/s
change recorded. An 8-bit KV cache also halves KV reads at long context, which helps speed.

**Risk:** 8-bit KV may lower quality on some models, and smaller caches reduce prefix reuse (time to
first token). Opt-in until measured.

**Status (2026-10-07):** implemented on `beta`, opt-in, and measured on the M4 Max development
machine. Not yet measured on 16 GB hardware, so the defaults are unchanged.

- **Settings:**
  - `[backends.llamacpp] kv_cache_type = "q8_0"` passes `-ctk q8_0 -ctv q8_0`. The planner sizes
    the KV cache at 17/32 of f16; recurrent state stays f32.
  - `[runtime] memory_profile = "small"`, or `"auto"` (small on 16 GB or less). Small means
    llama.cpp `--cache-ram 256`, and for MLX 1 prompt-cache entry and a 512 MiB buffer cache.
- **8-bit KV on build 11146** (Metal and CPU): llama.cpp turns flash attention on itself
  ("required for quantized V cache"), and the models answered correctly. Every cache came out at
  exactly 17/32 of f16 (Qwen3.5 9B 256 → 136 MiB, Gemma 4 12B 608 → 323 MiB, gpt-oss-20b 210 →
  111.6 MiB).
- **"Fits, tight"** (display only, on by default): on machines with 16 GB or less, a plan above 5/8
  of RAM (10 GiB on 16 GB) shows as tight in the CLI, `doctor` and the desktop app. `doctor` also
  shows the memory profile and KV cache type in effect.
- **Planner on a 16 GB Mac** (computed; these two models are not downloaded):

  | Model | Standard | Small | Small + q8_0 |
  |---|---|---|---|
  | Ministral 3 14B | 11.15 GiB, refused | 10.40, fits (tight) | 9.81, fits |
  | Phi-4 | 11.61, refused | 10.86, refused | 10.12, fits (tight) |

Exit check ([results](../benchmarks/results/2026-10-07-apple-m4-max-64gb-phase3-small/SUMMARY.md),
four models, standard then small + q8_0):

| Criterion | Result |
|---|---|
| Measured peak at or below the estimate | Met in both runs: estimate ÷ peak 1.12–1.28 with the Phase 3 settings. Peaks fell by 0.31–1.47 GiB. |
| Quality checks unchanged with 8-bit KV | Met: 3/3 on every model in both runs. |
| tok/s change recorded | Recorded: Qwen3.5 9B −3%, Gemma 4 12B +8%, gpt-oss-20b −11%, Qwen3 8B (MLX) 0%. Not conclusive: two runs started at load 6–7.5. |

Open items:

- **Measure on 16 GB hardware,** including Ministral 3 14B and Phi-4 (about 8–9 GiB of downloads
  each), before making `memory_profile = "auto"` or `q8_0` the default.
- **Re-measure speed on a quiet machine,** to see whether q8_0 costs decode speed (gpt-oss-20b's
  −11% came with a higher load).
- **Prefix reuse with the smaller host cache** is not measured (time to first token for repeated
  long prompts).
- **q8_0 on Windows** (Vulkan or CUDA builds) is untested; it needs flash attention on that device.

## Phase 4 — Speculative decoding

**Goal:** more than one token per pass over the weights.

**Evidence (M4 Max):**

- A separate draft model, Qwen3 8B with Qwen3 1.7B in MLX: +12% on prose, +30% on a code edit.
- llama.cpp `ngram-simple` on a code edit: 46 of 48 guessed tokens accepted. The speed gain is
  unconfirmed because that run was noisy.

**Order within the phase**

1. **N-gram lookup** (llama.cpp `--spec-type ngram-simple`). It costs no memory, which suits 16 GB
   machines, and it helps when output repeats the input: code edits, RAG, agents.
2. **MTP heads** for Qwen3.5, 3.6 and 3.8, which ship a built-in draft layer (`--spec-type draft-mtp`).
   unsloth publishes a 1.37 GB MTP GGUF for Qwen3.8 27B. MLX-LM 0.31.3 discards MTP weights on load,
   so this is llama.cpp only for now.
3. **A separate draft model** with the same tokenizer, when memory allows. In MLX-LM a draft model
   turns off request batching, so use it with the `latency` profile only.

**Work**

- Adapter flags come from config (for example `backends.llamacpp.spec_type`), never from requests. The
  gateway already strips a client-sent `draft_model`
  ([validate.rs:48](../crates/api/src/validate.rs)).
- The memory planner counts draft or MTP weights and their KV.
- The bench report records acceptance.
- The catalog records MTP companion files.

**Exit:** gain per mode measured on the reference set with unique prompts. A mode that slows a
workload down is off by default for it.

**Risk:** low; it is opt-in and engine-side. Undo by turning the config off.

**Status (2026-10-07):** implemented on `beta` and measured on the M4 Max development machine
([results](../benchmarks/results/2026-10-07-apple-m4-max-64gb-phase4/SUMMARY.md)).

- **Settings:** `[backends.llamacpp] speculative = "auto" | "off" | "ngram" | "mtp" | "draft"`, with
  `draft_model` and `draft_tokens`; `[backends.mlx] draft_model` and `draft_tokens` (one request at a
  time). Requests can never set them.
- **Planner:** n-gram adds no memory; MTP adds one copy of the recurrent state per drafted token, a
  KV cache for the MTP layers and ~160 MiB of compute; a draft model adds its weights, its own
  cache and compute. All matched llama.cpp's own allocations; estimates stayed at or above the
  measured peak in every speculative run (1.07–1.58).
- **Catalog:** the six unsloth MTP builds (Qwen3.5 0.8B, 2B, 4B, 9B; Qwen3.6 27B, 35B-A3B), ids
  ending in `-mtp`, with an "MTP" badge in the desktop library. MTP layers are read from the GGUF
  header (`nextn_predict_layers`).
- **Bench** records draft acceptance; `benchmarks/suites/speculative.toml` covers prose, a code edit
  and quoting.

Exit check (unique prompts, quiet machine, against each model's own baseline):

| Mode | Prose | Code edit | Quoting | Quality | Default |
|---|---|---|---|---|---|
| MTP, 1 drafted token (Qwen3.5 9B MTP) | +11% | +19% | +19% | unchanged | **on** (`auto`, MTP builds only) |
| n-gram (Qwen3.5 9B / Gemma 4 12B / gpt-oss-20b) | +1% / −8% / +8% | −2% / −10% / +28% | −4% / −2% / +225% | unchanged | off |
| llama.cpp draft model (Qwen3.5 9B + 0.8B) | −47% | −33% | −23% | unchanged | off |
| MLX draft model (Qwen3 8B + 1.7B) | +17% | +46% | +40% | **one answer changed** | off |

MTP needs 1 drafted token per step: llama.cpp's default of 3 slowed prose by 41% on Qwen3.5 9B.

Open items:

- **Measure on 16 GB hardware.** Gains on a base chip (68–120 GB/s) are not measured.
- **n-gram for code and quoting:** a big win when output repeats the input (gpt-oss-20b quoting
  +225%), a small loss otherwise. Per-request mode selection would need the workload; not planned.
- **MTP builds exist only for Qwen3.5 and Qwen3.6** (unsloth). MLX-LM discards MTP weights.

## Phase 5 — 16 GB catalog tier

**Goal:** whatever a 16 GB user downloads fits comfortably and is fast.

**Work**

- MoE models read only their active experts per token, so they are the fastest models that fit on
  16 GB. gpt-oss-20b (about 3.6B of 21B parameters active) and LFM2.5 8B-A1B (about 1B active) are
  already in the catalog; add others where license and engine support allow.
- Add 3-bit variants for 14–27B models, labelled with the quality cost. Example: unsloth's Qwen3.8
  27B `UD-IQ3_XXS` (10.9 GB) and `UD-Q2_K_XL` (9.8 GB). Add MLX 3-bit builds where they are published.
- Add a "comfortable on this computer" filter using the Phase 2–3 estimates, and attach MTP
  companions from Phase 4.

**Exit:** on 16 GB hardware, every recommended variant meets the comfortable target and a minimum
speed set from Phase 1.

**Why here:** it needs the Phase 2–3 estimates and the Phase 4 MTP support to know what qualifies.

**Status (2026-10-07):** implemented on `beta`; the exit check run on a simulated 16 GB Mac.

- **Catalog:** 3-bit builds of the 14–27B families, labelled with the quality cost (Q3_K_M for
  Ministral 3 14B, Phi-4, Mistral Small 3.2 24B, Devstral Small 2 24B, Qwen3.6 27B, Gemma 4 26B-A4B;
  Q3_K_XL for Qwen3.8 27B; Phi-4 3-bit for MLX), plus the MTP builds from Phase 4. The catalog has 83
  downloads.
- **Recommendations** (one per family): only variants that fit comfortably (not "tight") and run at
  least 10 tok/s by the Phase 6 speed planner; then 4-bit before 3-bit, the preferred engine, and
  the plain build before the MTP one.
- **Filters:** `model catalog --comfortable`; "Fits comfortably" in the desktop library.

Exit check on a simulated 16 GB Mac (16 GB RAM, 10.67 GiB GPU limit, base M4 at 120 GB/s, default
settings): 20 families get a recommendation, **every one comfortable and predicted at 15 tok/s or
more** (Qwen3.5 9B MLX 9.01 GiB ~19 tok/s; Ministral 3 14B Q3_K_M 9.58 GiB ~15 tok/s; LFM2.5 8B-A1B
7.16 GiB ~67 tok/s). Families with no comfortable variant get none: Gemma 4 12B (10.16 GiB, tight;
9.41 GiB with `memory_profile = "auto"`), Phi-4, the 24–35B models and gpt-oss-20b (Phase 7).

Open items:

- **Confirm on real 16 GB hardware** (fit and speed).
- **Small MoE models are the fastest that fit** (LFM2.5 8B-A1B); add more as publishers release them.

## Phase 6 — Speed planner

**Goal:** LLMario predicts tok/s and picks the fastest setup for each machine.

**Why:** decode tok/s ≈ bandwidth × efficiency ÷ bytes read per token. On the M4 Max (published
546 GB/s), Qwen3.8 27B ran at 29.5 tok/s (15.13 GB per token ≈ 446 GB/s) and Qwen3 8B at 97.8 tok/s
(≈ 451 GB/s): 82% efficiency both times. 16 GB machines are mostly base chips at 68–120 GB/s
(Apple's published figures), so the projected speed for an 8B 4-bit model on a base M4 is about
21 tok/s. That projection is not measured.

**Work**

- `HardwareReport` ([hardware/src/lib.rs:35](../crates/hardware/src/lib.rs)) gains memory bandwidth:
  a table of Apple chips (published specs), `nvidia-smi` for NVIDIA, and a measured value from
  `llmario bench` when available.
- Prediction: bytes per token = active weights (MoE-aware) + KV read at the chosen context. Show
  "~N tok/s" beside "fits" in the model picker and the catalog
  ([runtime/src/library.rs](../crates/runtime/src/library.rs)).
- Mode selection: MTP, then n-gram, then a draft model. Choose between MLX and llama.cpp by measured
  speed instead of the fixed preference in
  [planner.rs](../crates/supervisor/src/planner.rs).
- Cache autotune results keyed by hardware fingerprint, engine build, model hash and profile. This is
  the "cached micro-autotuning" item already listed in Phase 2 of [MILESTONES.md](MILESTONES.md).

**Exit:** predicted tok/s is within ±20% of measured on the reference set, and recommendations change
only when a measurement supports it.

**Status (2026-10-07):** implemented on `beta`; calibrated and checked on the M4 Max.

- **Bytes read per token** come from the model files: the GGUF tensor index or the safetensors
  headers. MoE experts count at the share used per token (gpt-oss-20b reads 21% of its file per
  token, Qwen3.6 35B-A3B 12%); the embedding table is skipped next to a separate output head.
  Stored in the registry and the catalog (82 of 83 entries).
- **Prediction:** time per token = bytes read ÷ (published bandwidth × efficiency) + a fixed
  per-token overhead. Apple's published bandwidth per chip is built in.
- **Autotune:** `bench` records each measured decode speed in `autotune.json`, keyed by hardware,
  engine version, model hash, profile, speculative mode, GPU/CPU split and KV cache type. A
  measured model shows its measured speed; others use this computer's measured bandwidth (from
  runs with no speculation and no split) once it has any.
- **Shown** as "~N tok/s" in `model catalog`, `model fit` (with bytes read per token) and the desktop
  app. When two installed variants of a family are both measured, the faster one is chosen.
- **Mode selection:** `speculative = "auto"` (Phase 4).

Exit check ([calibration](../benchmarks/results/2026-10-07-apple-m4-max-64gb-phase6/SUMMARY.md)):
calibrated on Qwen3 1.7B and Gemma 4 12B (llama.cpp: 90% efficiency, 4.65 ms per token) and Qwen3
1.7B and Qwen3.8 27B (MLX: 73%, 1.22 ms), chosen before seeing the results. Held-out models:

| Model | Measured | Predicted | Error |
|---|---:|---:|---:|
| Qwen3.5 9B (GGUF) | 61.1 | 65.8 | +7.7% |
| Qwen3.5 9B MTP (GGUF) | 63.4 | 64.2 | +1.3% |
| gpt-oss-20b (GGUF, MoE) | 105.6 | 100.2 | −5.1% |
| Qwen3 8B (MLX) | 87.3 | 81.2 | −7.0% |
| Llama 3.2 3B (MLX) | 162.2 | 165.0 | +1.7% |
| Qwen3.5 0.8B (GGUF) | 268.6 | 171.1 | −36% |

**Met for the reference set** (all within ±8%). Below ~1B parameters the prediction is conservative.

Open items:

- **Calibrate on a base chip.** The per-token overhead was fitted on an M4 Max; on a base M4 it may
  differ. The first `bench` on any machine replaces the published figure with a measured one.
- **NVIDIA bandwidth** has no table; predictions there need a measurement first.

## Phase 7 — GPU/CPU split and MoE expert offload

**Goal:** run models somewhat larger than the GPU limit, knowing it costs speed.

**Work**

- **Apple Silicon:** plan against two limits (GPU working set and RAM) and pass a partial `-ngl`.
  Today Metal always gets every layer on the GPU
  ([memory.rs:121](../crates/supervisor/src/memory.rs)). The alternative is to let llama.cpp compute
  the split with `llama-fit-params`, or `--fit` with `--fit-target`. Both are in build 11146 but
  have no effect today, because LLMario always sets `-ngl` and `-c`.
- **NVIDIA:** validate the existing hybrid offload
  ([memory.rs:134](../crates/supervisor/src/memory.rs)) and add `--n-cpu-moe`. Attention then stays
  on the GPU and only the active experts are read from system RAM.
- The speed planner accounts for the slower CPU layers.

**Exit:** on a 16 GB Mac, gpt-oss-20b loads and runs without swapping, with its tok/s recorded. A
load that would swap is refused cleanly.

**Why late:** it needs 16 GB hardware to validate, and the results are tight (12–13 GiB of 16) and
slower. It is useful, but not "comfortable".

**Status (2026-10-07):** implemented on `beta`, opt-in (`[backends.llamacpp] offload = "auto"`);
the exit check run on a simulated 16 GB Mac. Not run on 16 GB hardware or on NVIDIA.

- **Planner:** on Apple Silicon the plan checks two limits, the GPU working set (or
  `gpu_memory_limit_gb`) and RAM. A model over the GPU limit that fits in RAM is split: for MoE
  models the expert weights of the first N layers stay in RAM for the CPU (`--n-cpu-moe N`, the
  smallest N that brings the GPU part under the limit with 512 MiB to spare); dense models get a
  partial `-ngl`. With `offload` off (the default) such a model is refused, as before. A model
  larger than RAM is refused either way. MLX cannot split.
- **Loading:** a split model is loaded with `--load-mode none` instead of memory-mapped. Measured
  on gpt-oss-20b with the experts of 8 of 24 layers on the CPU (direct `llama-server`, build 11146):

  | Loading | GPU buffer | Peak | Decode |
  |---|---:|---:|---:|
  | memory-mapped (llama.cpp default) | 11.5 GiB | 14.74 GiB | 65.8 tok/s |
  | memory-mapped, `--no-repack` | 11.5 GiB | 11.71 GiB | 59.6 tok/s |
  | `--load-mode none` (used) | 7.53 GiB | 12.50 GiB | 74.1 tok/s |

  Memory-mapped, Metal maps the whole file as one GPU buffer (over a 16 GB Mac's 10.67 GiB limit)
  and the CPU's repacked copy of its experts comes on top.
- **Speed:** the CPU's part of each token's read counts at 15% of the published bandwidth
  (predicted 70.0 tok/s, measured 61.2 and 70.7 in two runs). `model fit` notes the split and that
  it is slower.
- **Autotune:** a split run is its own setup and never stands for the machine's bandwidth. Before
  this was keyed, one split run lowered every other prediction (Qwen3.5 9B: 66 → 36 tok/s; it
  runs at 61).
- **NVIDIA:** `--n-cpu-moe` for MoE models over the VRAM; the existing partial offload for dense
  models is unchanged. Both unvalidated.

Exit check ([results](../benchmarks/results/2026-10-07-apple-m4-max-64gb-phase7/SUMMARY.md)):
gpt-oss-20b (11.28 GiB of weights) on the M4 Max with the GPU limited to a 16 GB Mac's 10.67 GiB and
`memory_profile = "small"`:

| Setup | Estimate | Peak | Decode |
|---|---:|---:|---:|
| `offload` off | refused (13.29 GiB over the 10.67 GiB GPU limit) | – | – |
| split, memory-mapped (first run) | 13.29 GiB | **14.75 GiB** | 69.5 tok/s |
| split, `--load-mode none` (two runs) | 13.29 GiB | 12.56 / 12.51 GiB | 70.7 / 61.2 tok/s |
| all on the GPU, no limit, standard profile (Phase 6, for comparison) | 14.04 GiB | 11.51 GiB | 105.6 tok/s |

**Met on the simulation:** it loads, the estimate stays above the measured peak, and decode is
61–71 tok/s (a third to two fifths slower than all on the GPU; the split varies more between runs). The first run's peak exceeded the estimate; that led to
the loading change. On a real 16 GB Mac the plan's 14 GiB budget (RAM minus 2 GiB) is the swap
guard: a larger load is refused (unit-tested). Open: the run on 16 GB hardware, where macOS's
own memory use decides whether 12.6 GiB stays out of swap.

## Phase 8 — Multi-machine sharding (optional)

This is the only approach that adds memory.

- `mlx_lm.server` 0.31.3 already supports distributed serving. Tensor parallelism covers llama,
  qwen2, qwen3, qwen3_5 and gpt_oss, among others. Pipelining covers only a few MoE architectures.
- The Homebrew build of llama.cpp ships without RPC support.
- Both protocols are unauthenticated network traffic, which conflicts with the loopback-only
  [threat model](THREAT_MODEL.md). Both are slow without Thunderbolt-class links.

Prerequisites: a threat-model update and supervision of processes across several hosts. Not
recommended until phases 2–7 are done.

**Status (2026-10-07): design only**, by decision. The threat model gains T15, and
[MULTI_MACHINE.md](MULTI_MACHINE.md) records what the engines offer (checked: mlx-lm 0.31.3 serves
tensor-parallel and pipelined models across machines; the Homebrew llama.cpp has no RPC), the threats
(unauthenticated, unencrypted tensor traffic), the requirements for any implementation (dedicated
link or authenticated tunnel, mutual authentication, lease-based shutdown, content-hash check) and
exit criteria. Nothing opens a network port.

## Graduating a phase to production

A phase moves from the beta to production only when all of these hold:

- Its exit criteria pass in the beta on 16 GB hardware, and it has been used in the beta without
  regressions.
- The bench report shows it is at least as good as production on the same hardware, both in memory
  (estimate at or above peak) and in speed.
- Its commits, **without** the beta identity from Phase 0, merge from `beta` to `main` through a pull
  request with CI green.
- It ships in a normal production release. Production defaults change only with the bench report
  that justified them.

A phase can be reverted in the beta at any time without affecting production.

## Not planned

- **Rewriting the gateway or embedding the engine in-process.** Gateway overhead measured as zero
  (217 vs 214 tok/s against raw llama-server), and the engines already run at about 82% of memory
  bandwidth, so the most this could gain is about 1.2×.
- **Streaming weights from SSD** to run models larger than RAM. A dense model rereads all its weights
  for every token, which is far too slow (not measured).
- **Qwen3.8 27B at 4-bit on 16 GB.** Its measured peak is 15.8 GiB.

## Appendix: evidence

**Measured** on an M4 Max (64 GB), AC power, 2026-10-07. The machine was under heavy background load
during the llama.cpp runs, so those speeds are not used as evidence.

| Measurement | Result |
|---|---|
| Qwen3.8 27B MLX 4-bit, KV cache | 64 KiB per token + 147 MiB fixed; 16 of 64 layers keep a KV cache |
| Qwen3.8 27B MLX 4-bit, memory | 14.09 GiB resident after load; 15.8 GiB peak at 6k tokens |
| Qwen3.8 27B MLX 4-bit, decode | 29.5 tok/s |
| Qwen3 8B MLX 4-bit, decode | 97.8 tok/s; with a Qwen3 1.7B draft (2 tokens): 109.9 prose, 126.3 code edit |
| llama.cpp `ngram-simple`, Qwen3 1.7B, code edit | 46 of 48 drafted tokens accepted; tok/s inconclusive |

**Computed** with the planner's own formula for 16 GB machines (8k context, `latency` profile).
Budgets: 10.67 GiB on a Mac, 14.0 GiB on a PC. ✅ fits a Mac and a PC · 🟡 PC only · ❌ neither.

| Model | Today | + real KV | + 8-bit KV | + scaled overheads |
|---|---|---|---|---|
| Qwen3.5 9B MLX 4-bit | 11.1 🟡 | 9.1 ✅ | 9.1 ✅ | 8.3 ✅ |
| Gemma 4 12B Q4_0 | 14.6 ❌ | 9.2 ✅ | 8.9 ✅ | 8.2 ✅ |
| Ministral 3 14B Q4_K_M | 11.2 🟡 | 11.2 🟡 | 10.6 ✅ | 9.8 ✅ |
| Phi-4 Q4_0 | 11.6 🟡 | 11.6 🟡 | 10.9 🟡 | 10.1 ✅ |
| gpt-oss-20b MXFP4 | 14.2 ❌ | 14.0 ❌ | 13.9 🟡 | 13.2 🟡 |
| Qwen3.8 27B UD-IQ3_XXS | 14.6 ❌ | 13.3 🟡 | 13.0 🟡 | 12.2 🟡 |
| Qwen3.8 27B MLX 4-bit | 24.3 ❌ | 20.2 ❌ | — | 19.1 ❌ |

**Engine capabilities checked** on the tested versions:

- **llama.cpp build 11146:**
  - Has `--fit` and `llama-fit-params`, `-ctk`/`-ctv`, `--n-cpu-moe`, `-ot`, `--load-mode`.
  - Has `--spec-type`: draft, EAGLE3, MTP, dflash and n-gram modes.
  - The Homebrew build has no RPC.
- **MLX-LM 0.31.3:**
  - `--draft-model` turns off batching; `--pipeline` and distributed serving are available.
  - The server has no KV quantization, and MTP weights are dropped on load.
