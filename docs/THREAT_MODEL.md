# Threat model

Scope: the `llmario` control plane, its HTTP API, model download, and the engine child
processes it launches. Out of scope: vulnerabilities inside upstream engines themselves
(we reduce exposure to them, we do not fix them).

## Assets

- Prompts and completions (private user data).
- The host: files, credentials, compute.
- Model files and their integrity.

## Threats and mitigations

| # | Threat | Mitigation in this repo |
|---|---|---|
| T1 | API reachable from the LAN / internet | Binds `127.0.0.1` by default. Any non-loopback bind requires `--allow-remote` **and** an API key; startup prints a warning. `/metrics` is loopback-only unless an API key is set. |
| T2 | Request makes an engine load an arbitrary model (mlx_lm loads any `model` it is sent, from HF, possibly with remote code) | Gateway allowlists request fields and **always overwrites `model`** with the engine-local name. `draft_model`, `adapters`, etc. are dropped. `--trust-remote-code` is never passed. |
| T3 | Engine ports hijacked by another local process | Engines bind `127.0.0.1` on a random port chosen per launch. Remaining risk: any local user can talk to a loopback port (documented; UDS is a planned contract v2). |
| T4 | Malicious or corrupted downloaded model | HTTPS only; every file checked against the Hub's SHA-256 (LFS) or git blob SHA-1 (small files) before an atomic rename. Remote file names are validated (no absolute paths, `..`, or backslashes). Disk space is checked before download. |
| T5 | Model repos with custom Python code | Never executed. `trust_remote_code` is not exposed in MVP. |
| T6 | Prompt/completion leakage via logs | The gateway never logs message content; logs record sizes, timings, and status only. Engine logs go to `~/.llmario/logs/` at default verbosity (no prompt dumping flags). |
| T7 | Engine crash takes down the server | Engines are child processes; crashes surface as `engine_crashed` (HTTP 502) and the next request relaunches. |
| T8 | Memory exhaustion / swap storm | Admission control estimates weights + KV + overhead vs. budget with headroom; refuses with an explanation. |
| T9 | Orphaned engine processes after a hard kill | Engine PIDs recorded under `~/.llmario/run/`; `doctor` reports live orphans. We never auto-kill a process we cannot prove we started. |
| T10 | Dependency compromise | Small dependency set, `Cargo.lock` committed, `cargo deny` config for licenses/advisories (see `deny.toml`). Python engine pinned via `scripts/setup-mlx-venv.sh`. |
| T11 | Telemetry leakage | No telemetry is implemented. Nothing leaves the machine except explicit `model pull` downloads. |
| T12 | Brute-force of API key | Constant-time comparison; loopback default. Rate limiting is future work. |
| T13 | DNS rebinding / drive-by requests from a web page | In loopback mode, requests whose `Host` is not a loopback name are rejected; handlers accept only `application/json` bodies, so a cross-site form post cannot reach them and a JSON post triggers a CORS preflight llmario does not answer. Tested end-to-end. |
| T14 | Engine fetching from the network | MLX engines run with `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`; llama.cpp is given a local file path only. |
| T15 | Multi-machine sharding (engines exchanging tensors over the network) | **Not implemented.** The engines' cluster protocols (MLX distributed, llama.cpp RPC) are unauthenticated and unencrypted, so nothing in LLMario uses them. Requirements for any future implementation (dedicated link or authenticated tunnel, mutual authentication, lease-based shutdown, content-hash check) are in [MULTI_MACHINE.md](MULTI_MACHINE.md). |
| T16 | Speculative-decoding settings taken from requests | Speculation (n-gram, MTP, draft model) is set only in `config.toml`; the gateway drops a client-sent `draft_model` (T2), so a request cannot make an engine load another model as a draft. |
| T17 | A web page the user has open reaches an engine's loopback port (a cross-site `fetch`, or DNS rebinding) | The gateway had this protection (T13); LLMario's own engine did not until 2026-11-16. Its HTTP port now refuses a non-loopback `Host`, any `Origin` header and any body that is not `application/json`, before a handler runs (`browser_refusal` in `crates/engine/server/src/api.rs`; unit and end-to-end tests). llama.cpp and MLX engines rely on their own servers' behaviour. |
| T18 | A crafted model file crashes or misleads the engine | GGUF tensor sizes use checked arithmetic (a size that wrapped around 64 bits used to pass the end-of-file check), nested metadata arrays are refused (they recursed without bound) and string summaries cut on a character boundary. Tested with crafted headers. Residual: the chat template of a model file is rendered without a time or size limit, and the `unsafe` kernels trust the dimensions the loader validated. |
| T19 | Conversation state on disk | LLMario's engine saves evicted conversations (token ids and KV state) under `$LLMARIO_HOME/cache/kv`, owner-only (0700 directory, 0600 files), unencrypted, evicted by size only. Since 2026-11-16 it also writes them when the system is short of memory. **Not cleared by the desktop app's "Clear history"**; delete the directory, or set `[backends.native] kv_cache_gb = 0` to turn the tier off. |
| T20 | Web content read by a model steers it (prompt injection), e.g. into fetching a URL that carries the conversation | Only with `[backends.native] web_access = true` (off by default). Fetched text is wrapped and marked untrusted, and a failed fetch's error text is reduced to one plain line. **Unresolved:** the Rule-of-Two gate in `crates/engine/tools` is not wired into the agent loop, so a model that obeys injected text can still call `web_fetch` with data in the URL. Keep web access off for conversations that hold private data. |
| T21 | Secrets reaching engine processes or other local users | Engine children no longer inherit `LLMARIO_API_KEY`, `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`. Data directories LLMario creates are owner-only, and `logs`, `run` and `tmp` are tightened when they already exist. An existing `~/.llmario` keeps its mode: run `chmod 700 ~/.llmario` if `config.toml` holds an API key. |
| T22 | Client-controlled text in metrics and logs | A request's `model` string is cut to 96 safe characters before it becomes a metric label or log field, and at most 256 distinct labels are kept (the rest count as `other`). |

## Residual risks

- Loopback ports are visible to all local users on multi-user machines. An engine port is picked
  before the engine binds it, so another local user could bind it first and receive prompts; a
  per-launch secret between gateway and engine is not implemented yet.
- The model endpoint override (`HF_ENDPOINT`) must be `https://`, or plain `http://` on an exact
  loopback host (it used to be a prefix test).
- Error responses include local paths and the engine's last log lines, also for authenticated
  remote clients in `--allow-remote` mode.
- Upstream engine parsers (GGUF, safetensors, Jinja chat templates) process untrusted files.
  Run downloaded models from sources you trust.
- A GPU/CPU split (`offload = "auto"`) keeps part of a model in RAM; on a 16 GB Mac this is
  tight by design (see Phase 7 of [PHASES_16GB_AND_SPEED.md](PHASES_16GB_AND_SPEED.md)) and has
  not been verified on 16 GB hardware.
- Model licenses: "open weights" ≠ unrestricted. `llmario model info` shows the license label
  recorded in the catalog; users are responsible for the terms.
