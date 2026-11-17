# Progress

Where the work stands, so any session or person can pick it up. Engine milestones and their
measurements are in [engine/STATUS.md](engine/STATUS.md); side effects outside the working tree
are in [OPS_LOG.md](OPS_LOG.md).

## Where it stopped (2026-11-17, 07:00 PST)

- `main` is at the 2026-11-17 release of the engine memory work (PR #20, merged) plus an ops-log
  commit. Branch `ui-engine-memory` (worktree `.claude/worktrees/llmario-performance-optimization-3de13c`,
  recreated for this work; the earlier branch of that name was deleted) holds the desktop-app
  follow-up, **committed locally, not pushed**: awaiting the owner's push approval. The pass
  record is `<git common dir>/ci-local/<sha>.pass`.
- The follow-up: the desktop app shows the built-in engine's memory state (new Tauri command
  `engine_memory`, `crates/supervisor/src/engine_stats.rs`, `apps/desktop/ui/memory-state.js`) and
  the load-time memory warnings (`MemoryPlan.warnings`, `LoadInfo.memory_warnings`). Nothing under
  `website/` changed, so nothing redeploys.
- Not seen working in the running app: the UI code is unit-tested (node) and the Rust side is
  unit-tested; the stats field names were checked against a live engine. The pill appears only
  after the engine has polled the system level with a conversation cached.
- Worktree and branch are the owner's to remove after merge.

## Done and verified

| Piece | Proof | Result |
|---|---|---|
| Benchmark driver `scripts/engine/bench_cpu.py`, `bench_pressure.py` | ran on five models | reports under `benchmarks/results/2026-11-16-apple-m4-max-64gb-cpu-shared/` |
| Memory-pressure relief in the engine (`crates/engine/server/src/pressure.rs`, `engine.rs` `relieve` / `shed_idle`) | `LLMARIO_TEST_GGUF=<gguf> cargo test --release -p llmario-engine --test serve_e2e -- --test-threads 1` | 4 of 4 pass on Qwen3-1.7B and Qwen3.5-0.8B; 476 → 127 MiB measured |
| Read-ahead at load, dense models with room in memory (`crates/engine/model/src/stream.rs` `readahead`) | `bench_cpu.py --modes serve`, old and new interleaved | cold load −28 % to −35 % on three models, MoE unchanged |
| Paging detection and availability note (`engine.rs` `account_paging`, `crates/engine/plan` `availability_note`) | `cargo test -p llmario-engine-plan` | 5 pass |
| Headroom taken once; supervisor estimate for the native engine (`crates/supervisor/src/memory.rs`, `crates/adapter_native`) | `cargo test -p llmario-supervisor`; `llmario-engine plan <9B gguf> --memory-limit 6442450944 --memory-headroom 0` | 34 pass; fits (was refused) |
| Engine default threads = performance cores (`default_threads`) | three interleaved `raw-run` rounds | decode 93–100 vs 77–86 tok/s; prompts 246–249 vs 269–273 |
| Security fixes (docs/THREAT_MODEL.md T17–T22) | unit tests in `crates/engine/server` (`guard_tests`), `crates/engine/formats`, `crates/api`, `crates/model_registry`; e2e `browser_requests_are_refused` | pass |
| Local CI, six jobs (Linux and macOS tests, both engine e2e legs, desktop, licenses) | `scripts/ci-local.sh` | PASS on 259ba98 and on the head |
| Whole workspace | `cargo fmt --all --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test` | clean; 479 passed, 0 failed |
| Greedy parity with llama.cpp unchanged | `LLMARIO_GOLDEN_DEVICES=cpu LLMARIO_TEST_GGUF=<Qwen3-1.7B> cargo test --release -p llmario-engine --test llama_cpp_golden` | pass (one model; the kernels were not touched) |

## In flight / interrupted

Nothing half-done in the tree.

## Open findings and decisions

| Item | Where | Severity | Who decides |
|---|---|---|---|
| RUSTSEC-2026-0543: chrono 0.4.45 reported unmaintained (a notice, not a vulnerability; no upgrade available). Excepted in `deny.toml` on 2026-11-17 at the owner's word. Still to decide: whether to move the six workspace crates that use chrono to jiff or time. That alone does not let the exception go, because rmcp 3.10.0 and schemars 1.2.2 depend on chrono too | `deny.toml` `[advisories]`; `Cargo.lock:55` | Low | owner |
| Web tools: the Rule-of-Two gate is not wired into the agent loop, so injected web text can make a model fetch a URL carrying conversation data | `crates/engine/server/src/agent.rs` `run_inner`; gate in `crates/engine/tools/src/policy.rs` | High, only with `web_access = true` (off by default) | owner: fix before promoting web access |
| No per-launch secret between gateway and engine; port chosen before the engine binds it | `crates/supervisor/src/engine.rs` `free_port` | Low (another local user) | owner |
| KV disk tier keeps conversations unencrypted and "Clear history" does not clear it | `crates/engine/server/src/kvtier.rs`; `apps/desktop/ui/app.js` | Medium/Low | owner (product: wire Clear history to the tier, add a maximum age) |
| Chat templates from model files render without a time or size limit | `crates/engine/chat/src/env.rs` | Low | engineering |
| Error responses include local paths and engine log tails for remote authenticated clients | `crates/api/src/chat.rs` | Low | engineering |
| Existing `~/.llmario` keeps mode 0755 (only new homes are 0700) | `crates/core/src/paths.rs` `ensure` | Low | owner (tighten existing homes?) |
| `cargo deny check bans` fails (duplicate crate versions; not part of CI, which checks advisories, licenses and sources) | `deny.toml` | Info | engineering |

## Next steps, in order

1. Done: `scripts/ci-local.sh` (first run failed on two things older than this branch, a lint
   new in Rust 1.100 and three advisories against rmcp 3.5.1; both fixed in 259ba98).
2. Owner review, then one push and a PR into `main` (no Railway redeploy: nothing under `website/`).
3. Done in the desktop app on `ui-engine-memory`: `paging` and `memory_pressure` from `/engine/stats`.
   Still open: show them in `llmario ps` and `/metrics`; call `/engine/control` `sleep` for idle engines.
4. Separate thread counts for prompts (all cores) and decode (performance cores): measured gap
   7–9 % on prompts.
5. Wire the Rule-of-Two gate into the agent loop (first row above).
6. Measure on a real 8 or 16 GB machine and on Linux x86-64 (nothing here was).

## How to resume

```bash
mount | grep LLMarioBuild || diskutil image attach --nobrowse /Volumes/ARGUS5/AIPROJECTS/LLMarioBuild.sparsebundle
cd /Volumes/ARGUS5/PROJECTS/LLMario/.claude/worktrees/llmario-performance-optimization-3de13c
cargo build --release && cargo test
# before/after: build main into another folder and pass both binaries
scripts/engine/bench_cpu.py --engine base=<old llmario-engine> --engine new=target/release/llmario-engine \
  --model /Volumes/ARGUS5/AIPROJECTS/LLMario-models/qwen3-1.7b-gguf-q4km/Qwen3-1.7B-Q4_K_M.gguf \
  --modes raw,serve --threads 8,12,16 --rounds 3 --quiet-idle 75 --out <dir>
scripts/engine/bench_pressure.py <gguf> 2800 1 <old llmario-engine> target/release/llmario-engine
```

Inputs: the model files under `/Volumes/ARGUS5/AIPROJECTS/` (`LLMario-models`,
`LLMario-beta-models`, `llmario-engine-moe-home/models`); `llama-bench` from Homebrew llama.cpp
(build 11146) for the comparison column. The "before" binary of this session was built from
7eeafe6 and lives in `target/baseline/` (build image; rebuildable, not in the repository).
