# llmario-engine-chat

Renders OpenAI-style chat messages into a prompt string with the model's own Jinja chat
template (`tokenizer.chat_template` from the GGUF), byte-identical to what Python Jinja2
produces through `transformers.apply_chat_template`. Rendering only; parsing model output
back into tool calls lives on the decode side (ARCHITECTURE.md 10.2).

## Public API

```rust
let tpl = ChatTemplate::new(template, bos_token, eos_token)?;          // or ::with_now(.., Some(fixed_time))
let text = tpl.render(&RenderRequest {
    messages: vec![Message::new("system", "Be brief."), Message::new("user", "Hi")],
    add_generation_prompt: true,
    tools: Some(json!([{"type": "function", "function": {...}}])),   // OpenAI tool schemas
    enable_thinking: Some(false),                                      // passed only when Some
    extra: Map::new(),                                                 // reasoning_effort, xml_tools, ...
    arguments_mode: ArgumentsMode::Auto,
})?;
tpl.content_hash();    // SHA-256 hex of the template text
tpl.detect_family();   // TemplateFamily::{Hermes, QwenXml, Glm, Gemma4, Llama3, Mistral, Harmony, Lfm2, ChatMl, Unknown}
```

* `Message { role, content: Content (string | list of {type: "text", text} | null), name, tool_calls, tool_call_id, reasoning_content }`.
  Messages reach the template as plain maps with exactly the keys that are set, in OpenAI
  order; tool calls as `{"id", "type": "function", "function": {"name", "arguments"}}`.
* Tool-call `arguments` are exposed as a JSON **object** (Qwen3.5, Gemma 4 and LFM2 raise on
  strings). `ArgumentsMode::Auto` retries with the JSON **text** when the template raises an
  error that mentions arguments; `render_detailed` reports which form was used.
* `tools: None` renders `tools` as Python `None` (transformers always passes the kwarg), so
  `tools is none` / `tools is defined` behave as on the reference side. `documents` is `None` too.
* Errors: `ChatError::Raised(msg)` when the template called `raise_exception` (the input is
  not acceptable to this template), `Compile`, `Render`, `InvalidRequest` otherwise.
* `ChatTemplate::context(req, arguments_as_string)` returns the exact variable map handed to
  the template; the parity test feeds the same map to Python.

## Environment (what transformers does, replicated)

transformers compiles templates with
`ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[AssistantTracker, loopcontrols])`,
a `tojson` filter that is `json.dumps(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False)`
and the globals `raise_exception` and `strftime_now`. The minijinja environment here sets the same
knobs (`trim_blocks`, `lstrip_blocks`, `keep_trailing_newline=false`, lenient undefined, no
auto-escaping, `loop_controls`) and adds what minijinja does not do the Python way:

| Behaviour | Python / transformers | This crate |
|---|---|---|
| `{{ value }}` output | `str()`: `True`, `None`, `1e-05`, `{'k': 'v'}` | custom formatter (`pyfmt::py_str`) |
| `string` filter, `join` filter | `str()` on items | overridden |
| `tojson` / `to_json` | `json.dumps(ensure_ascii=False)`: `, ` and `: ` separators, no HTML escaping, Python float repr, insertion order | custom filter (`pyfmt::json_dumps`) with `ensure_ascii`, `indent`, `separators`, `sort_keys` |
| `is sequence` | true for lists, dicts and strings | overridden |
| `is number` | `bool` is a `Number` | overridden |
| `.strip()`, `.split()`, `.startswith()`, `.items()`, `.get()`, ... | Python methods | `minijinja-contrib` pycompat |
| `{% generation %}` ... `{% endgeneration %}` | transformers extension, body passes through | rewritten to `{% if true %}` ... `{% endif %}` before compiling (identical whitespace control) |
| `strftime_now(fmt)` | `datetime.now().strftime(fmt)` | chrono `format` on the local clock, or the fixed `now` |
| `raise_exception(msg)` | raises `TemplateError` | returns an error carrying the message (`ChatError::Raised`) |
| dict ordering | insertion order | `preserve_order` on both `minijinja` and `serde_json` |
| `messages[0].role` on a dict without the key | `Undefined` (falsy, prints empty) | lenient undefined |

Verified Python-faithful as well: `~` concatenation (operands go through the formatter, so
`true ~ ''` is `True`), `+` on strings, `|string`, `|join`, `strftime_now` with `%-d`, `%j`, `%A`.

Known approximations (not hit by the six verified templates):

* `repr()` of exotic non-printable Unicode inside `str(dict)` uses an approximate printable table.
* `strftime` follows chrono: month/day names are always English (the reference runs with the
  C locale too); Python-only specifiers that chrono lacks make `strftime_now` return an error.

## Family detection

`detect_family` first matches the template's SHA-256 against a table of hand-verified templates
(Qwen3, Qwen3.5, Gemma 4, gpt-oss, SmolLM3, OLMo 3 from the local GGUFs), then applies structural
rules that each need at least two independent markers (`src/family.rs` documents the table). A
single substring never decides.

## Running the parity test

```sh
# once: a python with jinja2
python3 -m venv /path/to/pyref && /path/to/pyref/bin/pip install jinja2

LLMARIO_PYTHON=/path/to/pyref/bin/python \
LLMARIO_TEST_GGUF="/models/Qwen3-1.7B-Q4_K_M.gguf,/models/gemma-4-12b-it-qat-q4_0.gguf" \
cargo test -p llmario-engine-chat --test jinja2_parity -- --nocapture
```

For each GGUF the test extracts `tokenizer.chat_template` and the BOS/EOS token strings,
renders 17 conversations (system+user, user only, multi-turn, tool definitions with nested
schemas, a tool call plus tool result, `reasoning_content`, `enable_thinking` true/false,
`add_generation_prompt` false, content as text parts, Unicode, empty system, a multi-step tool
loop, parallel tool calls, typed arguments, a late system message, a tool result without a
call) with a fixed `strftime_now` clock, and compares each with `scripts/engine/render_ref.py`
run by `LLMARIO_PYTHON`. Both sides rendering the same bytes, or both raising (the message is
compared too), or both failing with a non-raise error, is parity; anything else fails. The test
also asserts the detected family per `general.architecture` and that the structural rules agree
with the hash table. Without the two variables the test prints a skip message and passes.
`LLMARIO_TEST_TMP` overrides where the template and input files are written (default: the
system temp dir).

Last verified result (six local GGUFs, 17 conversations each):

| Model | matched | both raised | both errored | mismatched |
|---|---|---|---|---|
| Qwen3-1.7B | 16 | 0 | 1 (`content_parts`: template does `str + list`) | 0 |
| Qwen3.5-0.8B | 16 | 1 (`system_not_first`: "System message must be at the beginning.") | 0 | 0 |
| Gemma 4 12B | 17 | 0 | 0 | 0 |
| gpt-oss-20b | 15 | 1 (`tool_result_without_call`) | 1 (`content_parts`) | 0 |
| SmolLM3-3B | 16 | 0 | 1 (`content_parts`: `list.replace`) | 0 |
| OLMo-3-7B | 16 | 0 | 1 (`content_parts`) | 0 |
