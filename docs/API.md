# API behaviour

Base URL: `http://127.0.0.1:11500/v1` (change with `--host/--port` or `[server]` config).
Only the behaviour documented here is implemented.

## Authentication and exposure

- Default bind is loopback. Non-loopback requires `--allow-remote` **and** an API key.
- With an API key set, every endpoint except `/healthz` requires `Authorization: Bearer <key>`.
- In loopback mode, requests whose `Host` header is not a loopback name are rejected (401),
  which blocks DNS-rebinding attacks from web pages.
- Bodies must be `application/json`; cross-site form posts cannot reach handlers.

## `GET /v1/models`, `GET /v1/models/{id}`

OpenAI list/model objects for installed models, plus an `llmario` object:
`family`, `format`, `backend`, `quantization`, `license`, `size_bytes`, `context_max`, `loaded`.

## `POST /v1/chat/completions`

`model` may be an exact id or a family name. The response `model` field and the
`x-llmario-model` / `x-llmario-backend` headers name the concrete model and backend used (`llamacpp`, `mlx`, or `native` for LLMario's own engine).

| Field | Behaviour |
|---|---|
| `messages` | `system`, `developer` (sent as `system`), `user`, `assistant`; `tool` (with `tool_call_id`) and assistant `tool_calls` on the native engine. Content may be a string or an array of `text` parts. |
| `stream`, `stream_options.include_usage` | SSE streaming. Usage chunk is sent only when requested. |
| `max_tokens` / `max_completion_tokens` | Default: profile default, capped by remaining context. |
| `temperature` (0–2), `top_p`, `top_k`, `min_p`, `stop`, `seed`, `presence_penalty`, `frequency_penalty`, `repetition_penalty` | Forwarded. On MLX, `seed` is dropped when decoding is greedy so batching stays enabled (output-identical). |
| `n` | Only `1`. |
| `response_format` | `{"type":"text"}` everywhere. `json_object` and `json_schema` on the native engine only: output is grammar-constrained to valid JSON (matching the schema); with reasoning models the constraint starts after the reasoning block. Other backends: **400 `unsupported_feature`**. |
| `tools`, `tool_choice`, `parallel_tool_calls` | Native engine only (other backends: **400 `unsupported_feature`**). Function tools in OpenAI format; calls come back as `message.tool_calls` / `delta.tool_calls` with `finish_reason: "tool_calls"`; `tool_choice` `required` or a named function is enforced by grammar for the Hermes, Qwen XML, Gemma 4, Mistral and Harmony template families (parsed but not enforced for the others). Built-in tools `{"type":"web_search"}` and `{"type":"web_fetch"}` are run by the engine itself — see below. |
| `functions`, `logprobs`, `logit_bias`, `audio`, `modalities`, `prediction`, image/audio content parts | **400 `unsupported_feature`** — not implemented in this release. |
| anything else (`user`, `metadata`, `store`, …) | Dropped, never forwarded to the engine. |

Reasoning models: engines may stream thinking as `delta.reasoning_content` (llama.cpp and the native
engine) or `delta.reasoning` (MLX-LM); llmario relays them unchanged.

### Built-in web tools (native engine)

With `[backends.native] web_access = true`, a request may add `{"type":"web_fetch"}` (open a page,
move within it, follow numbered links, find text) and, when `[backends.native] searxng_url` points at
a self-hosted [SearXNG](https://github.com/searxng/searxng) instance with JSON output enabled,
`{"type":"web_search"}`. The engine runs these calls itself and continues generating, up to 4 tool
rounds, then answers. Fetching is SSRF-guarded (http/https on ports 80/443 only, private, loopback,
link-local and metadata addresses refused, DNS pinned, redirects re-checked, 600 KB / 20 s limits,
`robots.txt` honoured); page text is wrapped as untrusted data before the model sees it. Streams show
each call as a `delta.llmario_tool_event` (`name`, `arguments`, `status`, `url`, `ms`); non-streaming
responses list them in `llmario.tool_events`. Requests for a built-in tool that is not enabled get a
400 that names the setting. No hosted or paid search API is used.

### Errors

OpenAI shape: `{"error": {"message", "type", "code", "param"}}`.

| HTTP | `code` | When |
|---|---|---|
| 400 | `invalid_request` | Malformed body or fields |
| 400 | `unsupported_feature` | See table above |
| 400 | `context_length_exceeded` | Prompt (≈4 bytes/token estimate) or `max_tokens` exceeds the per-request context; or the engine reports overflow |
| 401 | `invalid_api_key` | Missing/wrong key, or non-loopback Host header in loopback mode |
| 404 | `model_not_found` | Not installed |
| 502 | `engine_crashed` / `engine_start_failed` | Engine exited or failed to load; the next request relaunches (max 3 crashes / 10 min) |
| 503 | `server_busy` (+ `Retry-After`) | All slots busy, or the loaded model is serving and a swap cannot evict it, for `queue_timeout_secs` |
| 503 | `backend_unavailable` | Backend for the model's format not installed |
| 504 | `timeout` | `request_timeout_secs` exceeded |
| 507 | `insufficient_memory` | Memory plan exceeds budget; message includes a context that would fit |

A mid-stream failure is sent as an SSE `data: {"error": …}` event followed by `data: [DONE]`.

### Cancellation

Closing the connection cancels generation: the gateway drops the upstream connection
(engines stop generating) and releases the slot immediately. Tested end-to-end.

## `GET /healthz`

`{"status":"ok","version":…,"loaded_models":N}` — no auth, no private data.

## `GET /metrics`

Prometheus text: `llmario_requests_total{model,outcome}`, prompt/completion token counters,
`llmario_ttft_seconds` and `llmario_request_duration_seconds` histograms,
`llmario_loaded_models`, per-engine active requests, measured footprint and estimate.
Protected by the API key when one is set.

## `GET /llmario/monitor`

One JSON snapshot of the runtime, drawn by `llmario top` and the desktop app's Monitor panel
(type: `llmario_api::monitor::Snapshot`). Protected by the API key when one is set. It never holds
prompt or reply text, local file paths or engine ports.

| Field | What it is |
|---|---|
| `state` | `idle`, `reading` (a streamed request's prompt is being processed), `generating`, `running` (only non-streaming requests in flight: the gateway sees no tokens until they finish, so it cannot tell their phase), `queued` (waiting for a slot or a model to load), or `error` (the last request failed in the past 15 s) |
| `speed_tok_s`, `live[]` | Requests in flight: phase (`reading` / `generating` when streamed, `running` when not), `streaming`, elapsed time, time to first token, estimated prompt tokens (characters / 4), streamed pieces so far, live tokens per second (tokens after the first over the time after the first) |
| `engines[]` | Per loaded model: backend, pid, family, format, architecture, quantization, file size, layers, context per slot and slots, batch, profile, active requests, context in use, load time, idle time, planned and measured memory, CPU %, storage read/write rates, and for LLMario's own engine its `/engine/stats` body (`engine_stats`: device, threads, kernels, prefill/decode speed, cache reuse, KV disk tier, memory pressure, paging) |
| `recent[]` | The last 64 finished requests, newest first: time, model (up to 256 characters), outcome, prompt / cached / completion tokens, TTFT, duration, decode tok/s (from the engine's `timings` when it sends them, else tokens after the first over the time after the first, as `llmario run` and `bench` measure it) |
| `totals` | Since the gateway started: requests by outcome and token counts |
| `machine` | CPU, OS, core split, unified memory, published memory bandwidth, GPU, runtime profile |
| `system` | CPU %, cores, load average, memory and swap, and per GPU: utilisation, memory in use (and total for discrete GPUs), temperature, power and limit, PCIe link |
| `gateway` | The gateway's own process (in the desktop app, the app) |

Readings a machine does not give are `null`, never zero: rates (CPU %, disk bytes per second) are
measured between two snapshots, so the first one after start, or after a pause of more than 5 s,
has none; process CPU comes from CPU-time deltas (200 means two cores busy); Apple GPUs report
utilisation and memory through `ioreg`, while temperature and power need `powermetrics` with root
and stay `null`; NVIDIA readings come from `nvidia-smi`. A GPU tool that runs past 2 s is killed and GPU readings
pause for a minute; a snapshot taken while another one is still reading the machine reuses the last
reading. The machine and the engines are read only
when a snapshot is requested, so the monitor costs nothing while nobody polls it; what the gateway
always keeps is a fixed ring of the last 64 requests (numbers only) and one atomic counter per
streamed token of each request in flight.
