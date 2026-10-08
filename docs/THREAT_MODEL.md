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

## Residual risks

- Loopback ports are visible to all local users on multi-user machines.
- Upstream engine parsers (GGUF, safetensors, Jinja chat templates) process untrusted files.
  Run downloaded models from sources you trust.
- A GPU/CPU split (`offload = "auto"`) keeps part of a model in RAM; on a 16 GB Mac this is
  tight by design (see Phase 7 of [PHASES_16GB_AND_SPEED.md](PHASES_16GB_AND_SPEED.md)) and has
  not been verified on 16 GB hardware.
- Model licenses: "open weights" ≠ unrestricted. `llmario model info` shows the license label
  recorded in the catalog; users are responsible for the terms.
