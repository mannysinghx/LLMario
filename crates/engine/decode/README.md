# llmario-engine-decode

What the native engine runs after each forward pass: turn one logits vector into a token, record it
for the penalty window, report log-probabilities, and decide how much streamed text is safe to emit
given the request's stop strings.

Pure Rust (stable, 2021 edition), no `unsafe`, no dependencies beyond `serde` (for
`SamplingParams`). The PRNG is an in-crate xoshiro256\*\* seeded through splitmix64.

## Responsibility

| In scope | Out of scope (other crates) |
|---|---|
| `SamplingParams` (the API's sampling knobs with llama.cpp defaults) | Running the model, producing logits |
| `Sampler`: penalties → top-k → top-p → min-p → temperature → multinomial draw, or argmax | Detokenising tokens into text (`tokenizer`) |
| `LogitProcessor` hook applied before the penalties (grammar masks, logit bias land here later) | The grammar engine itself (Architecture §10.3) |
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
    pub fn reset(&mut self);                                // clears window, reseeds PRNG
    pub fn apply_penalties(&self, logits: &mut [f32]);      // for device-side samplers
    pub fn top_logprobs(&self, logits: &[f32], n: usize) -> Vec<(Token, f32)>;
    pub fn token_logprob(&self, logits: &[f32], token: Token) -> f32;
    pub fn params(&self) -> &SamplingParams;
    pub fn n_vocab(&self) -> usize;
    pub fn seed(&self) -> u64;
    pub fn window(&self) -> impl Iterator<Item = Token> + '_;
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
for t in prompt_tokens { sampler.accept(t); }          // llama-server also penalises the prompt
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

`cargo test -p llmario-engine-decode` — 31 tests: PRNG reference vector and determinism;
parameter defaults and serde; same seed ⇒ same sequence (and after `reset`); greedy argmax;
top-k, top-p (inclusive cut, renormalisation after top-k), min-p (sorted and unsorted paths) and
temperature on known distributions; multinomial frequencies; penalty arithmetic against
hand-computed values, window sliding and reset; processor ordering and accept; fully-masked logits;
heap top-k against partial selection; the stop matcher on `</s>` byte by byte, false starts,
overlapping stops, earlier-partial precedence, multi-byte stops split across fragments, repeated
prefix characters, `finish`/`reset`, and a long multi-stop stream; and a loose timing guard on a
262,144-entry vocabulary.
