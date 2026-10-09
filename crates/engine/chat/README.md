# llmario-engine-chat

Renders OpenAI-style chat messages into a prompt string with the model's own Jinja chat
template (`tokenizer.chat_template` from the GGUF), byte-identical to what Python Jinja2
produces through `transformers.apply_chat_template`, and parses the model's streamed output
back into reasoning, content and tool-call events per template family (ARCHITECTURE.md 10.2,
Appendix D).

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
tpl.detect_family();   // TemplateFamily::{Hermes, QwenXml, Glm, Gemma4, Llama3, Mistral, Harmony, Lfm2, Olmo3, ChatMl, Unknown}
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
single substring never decides. OLMo 3 is its own family (`Olmo3`: `<function_calls>` pythonic
calls inside ChatML turns); Llama 4's `<|header_start|>`/`<|eot|>` tokens map to `Llama3`.

## Output parser

```rust
use llmario_engine_chat::{OutputParser, OutputEvent, ParserOptions, ToolCallRepair, RepairVerdict};

let family = tpl.detect_family();
let options = ParserOptions { reasoning_open: OutputParser::prompt_opens_reasoning(family, &prompt) };
let mut parser = OutputParser::with_options(family, Some(&tools), options);   // tools: the request's OpenAI tool list
for piece in detokenised_text_pieces {
    for event in parser.push(piece) { /* stream it */ }
}
for event in parser.finish() { /* flush at EOS / stop / length / cancel */ }
```

Events, in output order:

| `OutputEvent` | Meaning |
|---|---|
| `Reasoning(String)` | text inside the family's thinking markers (or Harmony's `analysis` channel); never shown to end users by default |
| `Content(String)` | visible assistant text |
| `ToolCallStart { index, id, name }` | a call begins; `index` counts calls in this output from 0; `id` is `call_<hex>_<index>` (9 alphanumerics for Mistral) or the id the model wrote (Mistral list form) |
| `ToolCallArgumentsDelta { index, delta }` | JSON text of the arguments. JSON-native formats (Hermes, Harmony, Mistral, Llama JSON) stream the model's own bytes as soon as the name is known; tagged/pythonic formats (Qwen XML, GLM, Gemma 4, LFM2, OLMo, Llama 4) stream one converted parameter at a time as object fragments (`{`, `"k":v`, `,"k2":v2`, `}`). Concatenated deltas always parse to the `ToolCallEnd` arguments |
| `ToolCallEnd { index, arguments }` | the call is complete; `arguments` is always a JSON object (pythonic and Gemma syntaxes are converted; XML/GLM text values are typed from the tool's JSON Schema, falling back to JSON-when-valid-else-string) |
| `Invalid { reason }` | advisory: something could not be parsed. The raw text follows as `Content`; an `Invalid` between a `ToolCallStart` and its `ToolCallEnd` cancels that call. Also emitted (before the `ToolCallStart`) when a call names a tool the request did not declare |

Text that could be the start of a marker is held back until it either becomes one or cannot
(llama.cpp's `NEED_MORE_INPUT`), so partial tags never leak and the stream never stalls once
`finish` runs. A call whose closing tag is missing at end of output is accepted when its
arguments are complete (models often stop right after them); a call cut off mid-arguments is
reported as `Invalid` with its raw text preserved. Turn-end markers the server did not strip
(`<|im_end|>`, `<turn|>`, `<|eot_id|>`, `</s>`, ...) are dropped silently.

### Per-family markers (sources in `src/parser/bodies.rs` and `tests/fixtures/SOURCES.md`)

| `TemplateFamily` | Models | Reasoning | Tool call | Verified from |
|---|---|---|---|---|
| `Hermes` | Qwen3, SmolLM3, Granite 4 | `<think>` … `</think>` (Qwen3 opens it itself; thinking off renders `<think>\n\n</think>` into the prompt) | `<tool_call>\n{"name": "f", "arguments": {...}}\n</tool_call>`; content before/between calls | Qwen3-1.7B GGUF template; SmolLM3 GGUF (system instruction only, the template does not render `tool_calls`) |
| `QwenXml` | Qwen3.5, Qwen3-Coder, Nemotron 3 | `<think>` … `</think>`; the prompt ends with `<think>\n` so `reasoning_open` is set | `<tool_call>\n<function=f>\n<parameter=k>\nv\n</parameter>\n</function>\n</tool_call>`; mappings/sequences as JSON, other values Python `str()` (`True`, `None`, `1e-05`) | Qwen3.5-0.8B GGUF template |
| `Glm` | GLM-4.7 | `<think>` … `</think>`; the prompt ends with `<think>` (or `</think>` when thinking is off) | `<tool_call>f<arg_key>k</arg_key><arg_value>v</arg_value>…</tool_call>`; strings raw, other values JSON | zai-org/GLM-4.7-Flash `chat_template.jinja` (no local model) |
| `Gemma4` | Gemma 4 | `<|channel>thought\n` … `<channel|>` (prompt ends with `<|channel>thought\n<channel|>` when thinking is off, with `<|channel>thought\n` after a tool response) | `<|tool_call>call:f{k:<|"|>v<|"|>,n:1,o:{x:true},l:[null]}<tool_call|>`; keys bare and dictsorted, strings between `<|"|>` with no escaping; the turn with calls ends in `<|tool_response>` | gemma-4-12b-it GGUF template |
| `Llama3` | Llama 3.1/3.2/3.3, Llama 4 | none | `<|python_tag|>{"type": "function", "name": "f", "parameters": {...}}<|eom_id|>`, the same object at the start of the output, `<function=f>{json}</function>`, Llama 4 `[f(a="x"), g(n=1)]<|eot|>`, built-ins `<|python_tag|>ns.call(k="v")` | meta-llama/llama-models `prompt_format.md` for 3.1 and 4 (HF repos gated; no local model) |
| `Mistral` | Mistral Small 3.x, Ministral 3, Devstral 2, Magistral | `[THINK]` … `[/THINK]` | `[TOOL_CALLS]f[ARGS]{json}` per call after any content; older `[TOOL_CALLS][{"name": "f", "arguments": {...}, "id": "D681PevKs"}]` list form | mistralai/Devstral-Small-2-24B-Instruct-2512 `chat_template.jinja` (no local model; list form from the research notes, unverified against a template) |
| `Harmony` | gpt-oss | `<|channel|>analysis<|message|>` … `<|end|>` | `<|channel|>commentary to=functions.f <|constrain|>json<|message|>{json}<|call|>` (also the history order `<|start|>assistant to=functions.f<|channel|>commentary json<|message|>`); `final` channel is content, `commentary` without recipient is content; one call per turn in the template | gpt-oss-20b GGUF template; OpenAI Harmony guide |
| `Lfm2` | LFM2 / LFM2.5 | `<think>` … `</think>` | `<|tool_call_start|>[f(a='b', n=1, m={"k": 1})]<|tool_call_end|>`; strings single-quoted with `\\ \' \n \r` escapes | LiquidAI/LFM2-1.2B `chat_template.jinja` (no local model) |
| `Olmo3` | OLMo 3 | `<think>` … `</think>` | `<function_calls>f(a="x", n=1)\ng(b=true)</function_calls>`; values JSON, calls newline-separated | olmo-3-7b-instruct GGUF template |
| `ChatMl`, `Unknown` | anything else | `<think>` … `</think>` | Hermes JSON (`<tool_call>{json}</tool_call>`), the de-facto generic format | — |

How reasoning is detected: a family's opener switches the parser into reasoning mode until its
closer (or end of output: a never-closed block is still reasoning). When the rendered prompt
already ends with the opener (`OutputParser::prompt_opens_reasoning`), `ParserOptions::reasoning_open`
starts the parser inside the block; a repeated opener at the very start is swallowed and a
stray closer in content mode is dropped. Harmony has no markers: the `analysis` channel is
reasoning structurally.

### Validation and repair

`ToolCallRepair::new(Some(&tools)).check(name, &arguments)` coerces what can be coerced losslessly
(`"10"` where an integer is declared, `"true"` for a boolean, JSON text for an object) and then
validates `type` (string or list), `required`, `properties`, `enum`, `items` and `nullable`
(no external crate). It returns `Valid(coerced)`, `Repair { violations, instruction }` with a
one-paragraph instruction for the model ("The call to `f` was invalid: `arguments.location`
is required but missing; … Reply with a corrected call to `f` in the same format …"), or
`UnknownTool { instruction }` listing the declared tools. `parser::schema::{validate, coerce,
type_text}` are public for callers that want the pieces.

### Tests

`tests/tool_call_round_trip.rs` renders an assistant turn with one and two tool calls (plus
content and reasoning where the family renders them, and a plain answer) with each family's
template from `tests/fixtures/` (vendored verbatim for the Apache-2.0/MIT templates, minimal
equivalents written here for Gemma 4, Devstral, LFM2, Llama 3.1 and Llama 4; see
`SOURCES.md`), takes the model-generated part, feeds it one character at a time, in
pseudo-random chunks and in one piece, and asserts the same calls come back with identical
arguments and that the concatenated deltas parse to them. Negative cases cover truncated calls
(`Invalid` on `finish`, raw text kept), malformed bodies (`Invalid`, raw text kept), reasoning
with and without a closer, `reasoning_open` streams, content interleaved with calls, Harmony
header variants, Llama/Mistral variants, schema typing, undeclared tool names and
marker hold-back. With `LLMARIO_TEST_GGUF` set, the same turns are rendered with the real
GGUF templates (Qwen3, Qwen3.5, Gemma 4, gpt-oss, OLMo 3), compared byte for byte with the
fixtures' rendering, and parsed as well.

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
