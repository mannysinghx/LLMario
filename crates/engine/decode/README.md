# llmario-engine-decode

What the native engine runs after each forward pass: turn one logits vector into a token, record it
for the penalty window, report log-probabilities, decide how much streamed text is safe to emit
given the request's stop strings, and — when a request carries a `response_format`, a grammar or
tools — mask the logits so the output is guaranteed to follow the schema (constrained decoding,
Architecture §10.3).

Pure Rust (stable, 2021 edition), `#![forbid(unsafe_code)]`. Dependencies: `serde`/`serde_json`
(parameters and specs), `tracing` (fallback warnings), `llguidance` 1.9.1 (MIT; the grammar
engine, see *Constrained decoding* below) and `llmario-engine-tokenizer` (to adapt the vocabulary).
The PRNG is an in-crate xoshiro256\*\* seeded through splitmix64.

## Responsibility

| In scope | Out of scope (other crates) |
|---|---|
| `SamplingParams` (the API's sampling knobs with llama.cpp defaults) | Running the model, producing logits |
| `Sampler`: penalties → top-k → top-p → min-p → temperature → multinomial draw, or argmax | Detokenising tokens into text (`tokenizer`) |
| `LogitProcessor` hook applied before the penalties (grammar masks, logit bias) | Rendering tool definitions into the prompt and parsing the generated call (`chat`) |
| `grammar`: `GrammarProcessor` (JSON Schema / Lark / regex / tool-call grammars on llguidance), lazy triggers, `GrammarTokenizer`, `GrammarCache` | Mapping API fields (`response_format`, `tools`, `tool_choice`) to a `GrammarSpec` (`server`) |
| `top_logprobs` / `token_logprob` for the API's `logprobs` | On-device sampling kernels (`cpu`, Metal, Vulkan); they must reproduce these semantics |
| `StopMatcher`: streaming stop-string matching with hold-back | The SSE writer that emits the text |

## Public API

```rust
pub type Token = u32;

pub struct SamplingParams {
    pub temperature: f32,        // 1.0; <= 0 → greedy
    pub top_k: i32,              // 40;  <= 0 disables
    pub top_p: f32,              // 0.95; >= 1.0 disables
    pub min_p: f32,              // 0.05; <= 0 disables
    pub repeat_penalty: f32,     // 1.0 disables
    pub repeat_last_n: usize,    // 64; 0 disables all penalties
    pub presence_penalty: f32,   // 0.0
    pub frequency_penalty: f32,  // 0.0
    pub seed: Option<u64>,       // None → entropy seed, reported by Sampler::seed()
    pub n_probs: usize,          // 0; stored for the server
}   // Default + serde (missing fields take the defaults)

pub trait LogitProcessor: Send {
    fn process(&mut self, logits: &mut [f32]);   // rewrite in place; -inf forbids a token
    fn accept(&mut self, token: Token);          // advance state after a token is committed
}

impl Sampler {
    pub fn new(params: SamplingParams, n_vocab: usize) -> Self;
    pub fn with_processor(self, p: Box<dyn LogitProcessor>) -> Self;
    pub fn set_processor(&mut self, p: Option<Box<dyn LogitProcessor>>);
    pub fn take_processor(&mut self) -> Option<Box<dyn LogitProcessor>>;
    pub fn sample(&mut self, logits: &[f32]) -> Token;      // does NOT record the token
    pub fn accept(&mut self, token: Token);                 // penalty window + processor
    pub fn accept_prompt(&mut self, token: Token);          // penalty window only (prompt tokens)
    pub fn reset(&mut self);                                // clears window, reseeds PRNG
    pub fn apply_penalties(&self, logits: &mut [f32]);      // for device-side samplers
    pub fn top_logprobs(&self, logits: &[f32], n: usize) -> Vec<(Token, f32)>;
    pub fn token_logprob(&self, logits: &[f32], token: Token) -> f32;
    pub fn params(&self) -> &SamplingParams;
    pub fn n_vocab(&self) -> usize;
    pub fn seed(&self) -> u64;
    pub fn window(&self) -> impl Iterator<Item = Token> + '_;
}

// grammar (constrained decoding)
pub enum GrammarSpec { JsonSchema(serde_json::Value), Lark(String), Regex(String), ToolCalls(ToolCallSpec) }
pub struct ToolCallSpec { pub family: ToolFamily, pub tools: Vec<ToolDef>, pub choice: ToolChoice }
pub enum ToolFamily { Hermes, QwenXml, Gemma4, Llama3, Mistral, Harmony }
pub struct ToolDef { pub name: String, pub parameters: serde_json::Value }   // OpenAI function.parameters
pub enum ToolChoice { Auto, Required, Named(String) }                       // `none` = attach no processor
pub struct LazyTrigger(pub Vec<u8>);                                        // LazyTrigger::new("</think>")
pub enum GrammarState { Watching, Active, Complete, Disabled }
pub enum GrammarError { InvalidSpec(String), Compile(String), Tokenizer(String) }

impl GrammarTokenizer {                                   // llguidance TokenizerEnv; build once per model
    pub fn from_tokenizer(tok: &Tokenizer) -> Result<Self, GrammarError>;
    pub fn from_pieces(pieces: Vec<Vec<u8>>, special: &[bool], eos: Token, eog: &[Token]) -> Result<Self, GrammarError>;
    pub fn hash(&self) -> u64;                            // content hash: cache key
    pub fn n_vocab(&self) -> usize;  pub fn eos(&self) -> Token;  pub fn eog(&self) -> &[Token];
    pub fn is_special(&self, id: Token) -> bool;  pub fn special_token_id(&self, text: &str) -> Option<Token>;
    pub fn piece(&self, id: Token) -> &[u8];
}
impl GrammarCache {                                       // LRU of compiled grammars + one llguidance factory per tokenizer
    pub fn new(limit: usize) -> Self;
    pub fn get_or_compile(&mut self, spec: &GrammarSpec, env: &Arc<GrammarTokenizer>) -> Result<CompiledGrammar, GrammarError>;
    pub fn counters(&self) -> (u64, u64);                 // (hits, misses)
    pub fn len(&self) -> usize;  pub fn clear(&mut self);
}
impl GrammarProcessor {                                   // implements LogitProcessor
    pub fn new(grammar: CompiledGrammar, lazy: Option<LazyTrigger>) -> Self;
    pub fn compile(spec: &GrammarSpec, env: &Arc<GrammarTokenizer>, lazy: Option<LazyTrigger>) -> Result<Self, GrammarError>;
    pub fn state(&self) -> GrammarState;  pub fn error(&self) -> Option<&str>;
    pub fn stats(&self) -> &GrammarStats;                 // masks, mean/max mask time, steps watching
}

pub enum StopResult { Flush, Hold(usize), Matched { emit_up_to: usize } }

impl StopMatcher {
    pub fn new(stop_strings: Vec<String>) -> Self;          // empty strings are ignored
    pub fn push(&mut self, text_fragment: &str) -> StopResult;
    pub fn push_bytes(&mut self, fragment: &[u8]) -> StopResult;
    pub fn finish(&mut self) -> StopResult;                 // EOS / max_tokens: release held tail
    pub fn released_len(&self) -> usize;                    // prefix of the stream safe to emit
    pub fn total_len(&self) -> usize;
    pub fn held(&self) -> &[u8];
    pub fn is_matched(&self) -> bool;
    pub fn is_empty(&self) -> bool;
    pub fn reset(&mut self);
}
```

### Engine loop sketch

```rust
let mut sampler = Sampler::new(params, n_vocab);
let mut stops = StopMatcher::new(request.stop);
for t in prompt_tokens { sampler.accept_prompt(t); }   // llama-server also penalises the prompt
if let Some(spec) = grammar_spec_for(&request) {       // response_format / tools / grammar
    let g = cache.lock().get_or_compile(&spec, &model.grammar_tokenizer)?;
    sampler.set_processor(Some(Box::new(GrammarProcessor::new(g, lazy_trigger_for(&template)))));
}
let mut sent = 0;
loop {
    let logits = model.forward(...);
    let token = sampler.sample(&logits);
    sampler.accept(token);
    if token == eos { stops.finish(); emit(&text[sent..stops.released_len()]); break; }
    let frag = detok.push(token);                        // &str fragment
    text.push_str(&frag);
    match stops.push(&frag) {
        StopResult::Matched { .. } => { emit(&text[sent..stops.released_len()]); break; }
        _ => { emit(&text[sent..stops.released_len()]); sent = stops.released_len(); }
    }
}
```

## Sampling semantics versus llama.cpp

The chain is llama.cpp's `common/sampling.cpp` default (`penalties; dry; top_n_sigma; top_k;
typ_p; top_p; min_p; xtc; temperature` followed by `dist`), with the samplers this crate does not
implement (DRY, top-n-sigma, typical-p, XTC, mirostat, infill) treated as absent — which is what
their default parameters make them in llama.cpp too.

| Stage | llama.cpp | This crate | Notes |
|---|---|---|---|
| Logit processor | grammar sampler in the chain (position depends on `grammar_first`) | `LogitProcessor::process` on a private copy of the logits, before penalties | The caller's `logits` slice is never modified. |
| Penalties | `llama_sampler_penalties(last_n, repeat, freq, present)`: for every token in the last-`n` ring buffer, `logit <= 0 ? logit *= repeat : logit /= repeat`, then `logit -= count*freq + present` | identical | No-op when `repeat_last_n == 0` or all three penalties are neutral. The window only contains tokens passed to `accept`; pass the prompt tokens too if you want llama-server's behaviour. |
| Greedy | `temp <= 0` → `llama_sampler_temp` keeps only the argmax, `dist` draws it | `temperature <= 0` → argmax of the processed, penalised logits (first index on ties); the PRNG is not advanced | Identical output. |
| top-k | partial sort, keep `k` (skipped when `k <= 0` or `k >= size`) | bounded min-heap scan for `k <= 1024`, `select_nth_unstable` for larger `k`; ties broken by lower token id | O(n_vocab) + O(k log k). |
| top-p | softmax over the *current candidates*, keep the smallest prefix with cumulative `p >= top_p` (inclusive), `min_keep = 1` | identical | Skipped when `top_p >= 1`. Evaluated at temperature 1 because temperature comes later in the chain. |
| min-p | keep `logit >= max_logit + ln(min_p)` (sorted and unsorted paths), `min_keep = 1` | identical | Equivalent to `p_i >= min_p * p_max`; skipped when `min_p <= 0`. |
| Temperature | `logit /= temp` | identical | |
| Draw | softmax over the survivors, uniform draw, first candidate whose cumulative probability exceeds it (`std::mt19937`) | softmax (f32), uniform `f64` draw from xoshiro256\*\* | Same distribution; **not** the same random stream as llama.cpp for a given seed, so sampled outputs are comparable statistically, greedy outputs exactly. |
| Seed | `LLAMA_DEFAULT_SEED` means "random" | `seed: None` draws from `std::collections::hash_map::RandomState` entropy; `Sampler::seed()` reports it | `reset()` reseeds with the same seed, like `llama_sampler_reset`. |
| Degenerate input | undefined | everything `-inf` → returns the best candidate without panicking | |

Ordering guarantees: filtering is deterministic across platforms (ties resolved by token id), so a
seed reproduces the same token sequence on every machine, given identical logits.

### Log-probabilities

`top_logprobs(logits, n)` and `token_logprob(logits, t)` are `log_softmax(logits / temperature)`
over the **full vocabulary**, without the processor, penalties or truncation (the model's own
distribution after temperature, which is what the OpenAI `logprobs` field reports). With greedy
decoding the raw logits are used. llama-server's default (`post_sampling_probs = false`) likewise
reports pre-truncation probabilities, but without temperature scaling.

### Cost per call

One copy of the logits, one O(n_vocab) pass each for the processor, penalties (over the ≤
`repeat_last_n` distinct window tokens only), top-k, softmax; no allocation after the first call.
Only when top-k is disabled and top-p or min-p is active is the full candidate list sorted
(O(n_vocab log n_vocab)), the same thing llama.cpp does in that configuration. A debug-build test
over a 262,144-entry vocabulary with `top_k = 40` runs in a few milliseconds per call.

## Constrained decoding (grammar)

`llguidance` 1.9.1 (MIT) is embedded as a crate: no precomputation per grammar, a byte trie of the
vocabulary built once per model, an Earley parser with a lazily built regex-derivative lexer, and
"slicer" masks for the common token classes. API used: `ParserFactory::new_simple` (one per
tokenizer; holds the slicer), `ParserFactory::create_parser(TopLevelGrammar)`, `Matcher`
(`compute_mask_or_eos`, `consume_token`, `is_stopped`/`stop_reason`, cheap `clone` for a fresh
parser state), `TopLevelGrammar::{from_json_schema, from_lark, from_regex}`, and `toktrie`'s
`TokenizerEnv`/`TokTrie`/`SimpleVob`.

### Processor life cycle

```
Watching ──trigger bytes seen──▶ Active ──grammar accepting & nothing more allowed──▶ Complete
   │                               │
   └── no trigger: Active from step 0      └── parser error ──▶ Disabled (unconstrained, warning logged)
```

- `process`: `Watching`/`Disabled` leave the logits untouched; `Active` computes llguidance's token
  mask and sets every token outside it to `-inf` (end-of-generation tokens are in the mask exactly
  when the grammar is in an accepting state); `Complete` keeps only the end-of-generation tokens.
- `accept`: `Watching` scans the token's rendered bytes for the next trigger (a trigger may span
  tokens or end mid-token; the bytes after it are fed to the parser); `Active` advances the parser
  and moves to `Complete` when llguidance reports `NoExtension`/`EndOfSentence`, or to `Disabled`
  on any error (a token outside the mask, an out-of-range id). Nothing panics mid-generation.
- Prompt tokens must go through `Sampler::accept_prompt`, not `accept`: the grammar covers only the
  generated text.

### Lazy triggers

`GrammarProcessor::new(grammar, Some(LazyTrigger::new("</think>")))` keeps decoding unconstrained
until `</think>` has been generated, then constrains ("think, then JSON"). Triggers compose in
order: a user trigger is waited for first, then the spec's own trigger — `ToolChoice::Auto` adds
the family's opener, so `</think>` + `Auto` means "free text and thinking; the first opener after
the thinking block starts a well-formed call". An opener emitted *inside* the thinking block does
not count. Triggers are matched on bytes (control tokens render as their text), independent of
tokenisation.

### Tool-call grammars

| Family | Grammar after the opener | Opener (`auto` trigger) | Format tokens |
|---|---|---|---|
| Hermes | `%json` of `{"name": <const>, "arguments": <schema>}` (`anyOf` over the tools), then `</tool_call>`; further `<tool_call>…</tool_call>` blocks allowed | `<tool_call>` | text |
| QwenXml | `<function=NAME>` then `<parameter=k>` *value* `</parameter>` per parameter, `</function>`, `</tool_call>`; strings are free text up to `</parameter>` (lazy suffix), other types `%json` of the property schema | `<tool_call>` | text |
| Gemma4 | `call:NAME{k:v,…}` then `<tool_call\|>`; strings between `<\|"\|>` tokens, numbers/booleans/null bare, arrays and objects recursive, enum/const spelled out, anything else `%json` | `<\|tool_call>` | `<\|tool_call>`, `<tool_call\|>`, `<\|"\|>` |
| Llama3 | `%json` of `{"name": <const>, "parameters": <schema>}`; single call | `<\|python_tag\|>` | `<\|python_tag\|>` |
| Mistral | `NAME` `[ARGS]` `%json <schema>`; further `[TOOL_CALLS]…` calls allowed | `[TOOL_CALLS]` | `[TOOL_CALLS]`, `[ARGS]` |
| Harmony | `NAME` (` <\|constrain\|>json`)? `<\|message\|>` `%json <schema>` `<\|call\|>` | `<\|channel\|>commentary to=functions.` | `<\|channel\|>`, `<\|constrain\|>`, `<\|message\|>`, `<\|call\|>` |

`tool_choice`: `Required` starts the grammar at the opener (the first bytes *must* be the opener;
no EOS before a call); `Named(name)` is `Required` with the tool set reduced to that tool; `Auto`
is the lazy form above. Format tokens are referenced by id (`<[id]>`) when the tokenizer has them
as control tokens and as text otherwise, so the same builder works on real and synthetic vocabularies.

Argument schemas: JSON families pass the tool's `parameters` to llguidance as-is (`{}`/missing →
any object; set `additionalProperties: false` for OpenAI-strict behaviour). The tag families
(QwenXml, Gemma4) spell parameters out in schema order — required ones mandatory, optional ones
skippable, each at most once — which matches what the templates teach the models but rejects a
model that reorders arguments.

Special tokens are never matchable by grammar text: control tokens live behind llguidance's `0xFF`
marker in the trie, so a literal `"<|python_tag|>"` only matches the plain-text spelling byte by
byte. Every end-of-generation id of the tokenizer (`eog_ids`) ends a completed grammar. Byte `0xFF`
(never valid UTF-8) is therefore also unmatchable.

JSON whitespace: every `%json` (sub-)grammar carries `x-guidance.whitespace_pattern =
[ \t\n\r]{1,40}` unless the caller's schema root already has an `x-guidance` object, and the
whitespace between tool calls is bounded the same way — like llama.cpp's schema converter, so a
greedy decode cannot emit blanks forever.

### Cache

`GrammarCache` is keyed by `(GrammarSpec::hash(), GrammarTokenizer::hash())` (FNV-1a over the
canonical JSON of the spec, and over the vocabulary's pieces, special flags and end-of-generation
ids). It holds one llguidance `ParserFactory` per tokenizer (the slicer precomputation, 0.2 s in
release / 1.1–1.4 s in debug for 150k–260k tokens) and an LRU of compiled grammars up to `limit`;
a hit clones the fresh parser (shared compiled grammar, independent state). Wrap it in a `Mutex`
to share across request handlers.

### Measured cost (Apple Silicon, this machine)

| | release | debug |
|---|---|---|
| 262,144-entry synthetic vocab, JSON-schema grammar, mean mask / max | 52 µs / 1.6 ms | 1.3 ms / 35 ms |
| Qwen3-1.7B tokenizer (151,936), JSON-schema grammar, mean / max | 41 µs / 1.6 ms | 0.9 ms / 23 ms |
| Qwen3-1.7B tokenizer, Hermes `required` with two tools, mean / max | 51 µs / 1.6 ms | 1.0 ms / 26 ms |
| Grammar compile (JSON schema, 5 properties) | 0.7–1 ms | 2.7 ms |
| `GrammarTokenizer::from_tokenizer` (Qwen3) | 70 ms | 0.46 s |

The max is the first mask of a dense lexeme (a free-form string); later masks hit llguidance's
slicer cache. The architecture's budget is ~1 ms per token in release.

## Stop-string semantics

`StopMatcher` works on bytes over the concatenation of every fragment pushed (offsets are into that
stream). After each push it finds

- the earliest-starting complete match of any stop string in the unreleased tail, and
- the earliest-starting suffix of the tail that is a proper prefix of some stop string.

It returns `Matched { emit_up_to }` (text before the stop, stop excluded) when a complete match
exists and no partial match starts before it; `Hold(n)` when a partial match is pending (hold the
last `n` bytes; `released_len()` is what may be emitted); `Flush` otherwise. The decision is
therefore independent of how the text was chunked, which llama-server's per-token
`find_stopping_strings` is not. Overlapping stops (`ab`, `abc`) resolve to the same emitted text
either way. Once matched, every later call returns the same `Matched`.

UTF-8: when fragments are pushed as `&str`, every released offset is a character boundary of the
stream — a held tail starts where a stop string's first byte matched, and stop strings are valid
UTF-8 — so the server can slice its accumulated `String` at `released_len()` safely. `push_bytes`
exists for byte-streaming detokenisers; the boundary guarantee then rests on the caller's fragments.

`finish()` releases a held tail that never became a stop string (EOS, `max_tokens`). Empty stop
strings are dropped; with no stop strings every push is `Flush` in O(1).

## Tests

`cargo test -p llmario-engine-decode` — 46 tests. Grammar (15): tokenizer adapter (special
marking, hash stability, errors); JSON-schema greedy fuzz over 200 seeded random-logit walks, every
output parsed and checked against the schema; the same through the sampler's multinomial path;
lazy trigger (`</think>` as one token, split across bytes, and ending mid-token with leftover
bytes fed to the parser); `tool_choice = required` allows only opener prefixes as the first token
and no EOS; a completed regex grammar leaves only end-of-generation tokens; control tokens
unmatchable by text but reachable by `<[id]>`; parser errors fall back to unconstrained without
panicking; all six families under `required`, `named` and `auto` (free text, opener, then a
well-formed call; both tools reachable; the named tool pinned); `</think>` then `auto`; cache
hits/LRU eviction/per-tokenizer keys/independent states; spec hash and serde; mask timing on a
262,144-entry vocabulary (printed; loose guard: mean < 5 ms release, < 50 ms debug); and, when
`LLMARIO_TEST_GGUF` names a GGUF, the real Qwen3 tokenizer under a JSON schema and a Hermes
`required` grammar from fixed logits (output parsed). Sampler/stop (31): PRNG reference vector and determinism;
parameter defaults and serde; same seed ⇒ same sequence (and after `reset`); greedy argmax;
top-k, top-p (inclusive cut, renormalisation after top-k), min-p (sorted and unsorted paths) and
temperature on known distributions; multinomial frequencies; penalty arithmetic against
hand-computed values, window sliding and reset; processor ordering and accept; fully-masked logits;
heap top-k against partial selection; the stop matcher on `</s>` byte by byte, false starts,
overlapping stops, earlier-partial precedence, multi-byte stops split across fragments, repeated
prefix characters, `finish`/`reset`, and a long multi-stop stream; and a loose timing guard on a
262,144-entry vocabulary.
