# Multi-machine sharding (Phase 8): design

**Status: design only.** Nothing in LLMario opens a network port for this, and nothing is planned to
until the open questions below are answered and a second machine is available to test on. This
document records what the engines offer, the threats, and the shape an implementation would take.

## Why

Sharding is the only way to run a model larger than one machine's memory: two 16 GB Macs could
together hold a model that neither can alone. Phases 2–7 make the most of one machine; this phase
would add machines.

## What the engines offer (checked on the development Mac, 2026-10-07)

| Engine | Multi-machine support | Notes |
|---|---|---|
| mlx-lm 0.31.3 (`mlx_lm.server`) | Yes, through MLX's distributed launcher (`mx.distributed`) | **Tensor parallelism** (each layer split across machines) for models whose code has `shard()`: llama, qwen2, qwen3, qwen3_5, gpt_oss, ministral3, glm4_moe, glm4_moe_lite, deepseek_v2/v3/v32, minimax, kimi_k25, exaone_moe, step3p5 and others. **Pipelining** (whole layers per machine, `--pipeline`) for deepseek_v2/v3/v32, glm4_moe, glm4_moe_lite, kimi_k25, ministral3. |
| llama.cpp (Homebrew build 11146) | Not in this build | Upstream llama.cpp has an RPC backend (`rpc-server` plus `--rpc host:port`), but the Homebrew build ships without it, so it would mean a custom build. |

## Threats

Both protocols carry raw tensors between machines with **no authentication and no encryption**. That
conflicts with LLMario's loopback-only design ([threat model](THREAT_MODEL.md), T1 and T3):

| # | Threat | Why it matters |
|---|---|---|
| S1 | Anyone on the network connects to a worker's port | A worker accepts tensors from whoever connects: they can feed it garbage, crash it, or (in the worst case, through a parser bug in the engine) run code on it. |
| S2 | Traffic is read in transit | Activations passed between machines are derived from the user's prompt; with enough of them a prompt can be partly reconstructed. Prompts are private user data. |
| S3 | A rogue machine joins | Without mutual authentication, a machine pretending to be a worker can return wrong results (silently wrong answers) or record everything sent to it. |
| S4 | Ports stay open after use | A worker left running keeps listening after LLMario stops. |
| S5 | Model files differ between machines | Mismatched files give wrong answers with no error. |

## Requirements for any implementation

1. **Off by default, explicit opt-in per machine,** with a list of allowed peers. The LLMario API
   itself stays loopback-only; only the engines' cluster traffic crosses machines.
2. **No plain traffic on a shared network.** Cluster traffic either stays on a dedicated
   point-to-point link (a Thunderbolt cable between two Macs) or goes through an authenticated,
   encrypted tunnel (WireGuard, or SSH port forwarding) with engine ports bound to loopback on each
   end. LLMario checks this at start and refuses otherwise.
3. **Mutual authentication of the control channel** (the part LLMario writes): a shared key or
   per-machine keys set up once, never sent in the clear.
4. **Supervision across machines.** A small `worker` mode of LLMario on each extra machine starts and
   stops its engine process only when the coordinator asks over the authenticated channel, reports
   health, and stops its engine when the coordinator goes away (lease with a timeout), so S4 cannot
   happen.
5. **Same files everywhere.** Before a run, every machine reports the model's content hash (the
   registry already records one); a mismatch refuses the run (S5).
6. **Memory and speed planning per machine.** Tensor parallelism over N machines needs about
   weights ÷ N + KV ÷ N + overhead on each; pipelining needs each machine's share of layers. Speed is
   bounded by the slowest machine plus the link: tensor parallelism synchronizes twice per layer per
   token, so link latency matters more than bandwidth, and it is only worthwhile over
   Thunderbolt-class links (microseconds), not Wi-Fi or ordinary Ethernet (milliseconds).

## Shape of an implementation (not built)

- `[cluster]` config: `peers = ["mac-b.local"]`, `transport = "thunderbolt" | "tunnel"`, a key file.
- `llmario worker` on each extra machine: authenticated control endpoint, engine launch on request,
  lease-based shutdown, content-hash report.
- Coordinator: the existing supervisor gains remote engine handles; the MLX adapter launches
  `mlx_lm.server` under MLX's distributed launcher with the peers; the memory planner checks each
  machine's budget; the speed planner adds a link term.

## Exit criteria for a future implementation

- Two machines run a model larger than either one alone, with tokens per second recorded.
- A port scan from a third machine finds no listening engine port (tunnel mode) or none outside the
  dedicated link (Thunderbolt mode).
- Stopping the coordinator, or unplugging a worker, ends every engine process within the lease
  timeout, and LLMario reports the failure clearly.
- A model-file mismatch between machines is refused before loading.

## Open questions

- Which link to support first: Thunderbolt bridge only (simplest to make safe), or tunnels too?
- Does MLX's launcher accept a fixed loopback address per peer, so that it works through SSH or
  WireGuard tunnels without exposing its ports? (Not verified.)
- Is a llama.cpp build with RPC worth maintaining for GGUF users, given Homebrew does not ship it?
