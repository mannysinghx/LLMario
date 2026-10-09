//! Constrained (grammar-guided) decoding on top of [llguidance](https://docs.rs/llguidance).
//!
//! - [`GrammarSpec`] describes what to enforce: a JSON Schema, a Lark grammar, a regex, or a
//!   tool-call format for one of the chat families ([`ToolCallSpec`]).
//! - [`GrammarTokenizer`] adapts the engine tokenizer into llguidance's byte trie (built once per
//!   model); [`GrammarCache`] keeps compiled grammars keyed by (spec hash, tokenizer hash).
//! - [`GrammarProcessor`] implements [`LogitProcessor`]: `process` sets every token the grammar
//!   forbids to `-inf`, `accept` advances the parser. A [`LazyTrigger`] (or a tool family's own
//!   opener in `tool_choice = auto`) keeps generation unconstrained until the trigger bytes have
//!   been produced, so "think, then JSON" works for every family.
//!
//! The processor never panics mid-generation: a parser error logs a warning and the processor
//! falls back to unconstrained decoding for the rest of the sequence (see
//! [`GrammarProcessor::state`]).

mod cache;
mod env;
mod tools;

#[cfg(test)]
mod tests;

use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use llguidance::api::{StopReason, TopLevelGrammar};
use llguidance::toktrie::SimpleVob;
use llguidance::{Matcher, ParserFactory};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use cache::GrammarCache;
pub use env::{Fnv1a, GrammarTokenizer};
pub use tools::{ToolCallSpec, ToolChoice, ToolDef, ToolFamily};

use crate::sampler::LogitProcessor;
use crate::Token;

/// What the processor enforces.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GrammarSpec {
    /// `response_format: {type: "json_schema"}` — the output is one JSON value matching the
    /// schema (llguidance's JSON Schema subset: `$ref`/`$defs`, `anyOf`/`oneOf`, `enum`/`const`,
    /// `required`, `additionalProperties`, string `pattern`/`format`/lengths, numeric bounds,
    /// array item bounds).
    JsonSchema(Value),
    /// A grammar in llguidance's Lark-like syntax (`start:` rule, `%json {...}` and `%regex`
    /// sub-grammars, `<[id]>` token references).
    Lark(String),
    /// The output matches this regular expression (Rust `regex` syntax, anchored).
    Regex(String),
    /// A tool call in one chat family's wire format, constrained to the given tool set.
    ToolCalls(ToolCallSpec),
}

impl GrammarSpec {
    /// Stable content hash (FNV-1a over the canonical JSON encoding of the spec).
    pub fn hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        h.write(b"llmario-grammar-spec-v1");
        match serde_json::to_vec(self) {
            Ok(bytes) => h.write(&bytes),
            Err(_) => h.write(format!("{self:?}").as_bytes()),
        }
        h.finish()
    }

    fn describe(&self) -> &'static str {
        match self {
            GrammarSpec::JsonSchema(_) => "json_schema",
            GrammarSpec::Lark(_) => "lark",
            GrammarSpec::Regex(_) => "regex",
            GrammarSpec::ToolCalls(_) => "tool_calls",
        }
    }
}

/// Bytes that must be generated before the grammar starts constraining (for example
/// `</think>` for reasoning models, or a tool-call opener). Matched on the rendered bytes of the
/// accepted tokens, so a trigger may span several tokens or end in the middle of one; the bytes
/// of the token after the trigger are fed to the parser.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LazyTrigger(pub Vec<u8>);

impl LazyTrigger {
    pub fn new(text: impl AsRef<[u8]>) -> Self {
        LazyTrigger(text.as_ref().to_vec())
    }
}

/// Why a grammar could not be built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GrammarError {
    /// The spec is malformed (empty tool set, unknown tool in `tool_choice`, ...).
    InvalidSpec(String),
    /// llguidance rejected the grammar.
    Compile(String),
    /// The tokenizer could not be adapted (no EOS, bad ids, ...).
    Tokenizer(String),
}

impl fmt::Display for GrammarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GrammarError::InvalidSpec(s) => write!(f, "invalid grammar spec: {s}"),
            GrammarError::Compile(s) => write!(f, "grammar compile error: {s}"),
            GrammarError::Tokenizer(s) => write!(f, "grammar tokenizer error: {s}"),
        }
    }
}

impl std::error::Error for GrammarError {}

/// Whitespace allowed between JSON lexemes in every JSON (sub-)grammar: at most 40 blanks, like
/// llama.cpp's schema converter. Unbounded whitespace would let a greedy decode loop forever on
/// a blank token; `x-guidance` set by the caller on the schema root takes precedence.
pub const JSON_WHITESPACE_PATTERN: &str = "[ \\t\\n\\r]{1,40}";

/// `schema` with the engine's JSON compile options (`x-guidance`) unless it already carries some.
pub fn with_json_options(schema: &Value) -> Value {
    match schema {
        Value::Object(m) if !m.contains_key("x-guidance") => {
            let mut m = m.clone();
            m.insert(
                "x-guidance".into(),
                serde_json::json!({ "whitespace_pattern": JSON_WHITESPACE_PATTERN }),
            );
            Value::Object(m)
        }
        other => other.clone(),
    }
}

/// A quiet llguidance parser factory for a tokenizer. Holds the token-slice precomputation, so
/// build one per model and reuse it ([`GrammarCache`] does this).
pub fn new_parser_factory(env: &Arc<GrammarTokenizer>) -> Result<ParserFactory, GrammarError> {
    let mut factory = ParserFactory::new_simple(&env.tok_env())
        .map_err(|e| GrammarError::Compile(format!("parser factory: {e}")))?;
    factory.quiet();
    Ok(factory)
}

/// A grammar compiled against one tokenizer: a fresh llguidance parser plus the trigger bytes
/// that gate it. Cloning is cheap (the compiled grammar is shared); each clone is an independent
/// parser state.
#[derive(Clone)]
pub struct CompiledGrammar {
    matcher: Matcher,
    triggers: Vec<Vec<u8>>,
    env: Arc<GrammarTokenizer>,
    spec_hash: u64,
    grammar_text: String,
}

impl fmt::Debug for CompiledGrammar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompiledGrammar")
            .field("spec_hash", &format_args!("{:016x}", self.spec_hash))
            .field("tokenizer_hash", &format_args!("{:016x}", self.env.hash()))
            .field("triggers", &self.triggers)
            .finish()
    }
}

impl CompiledGrammar {
    /// Compiles `spec` for the tokenizer behind `factory`.
    pub fn compile(
        spec: &GrammarSpec,
        env: &Arc<GrammarTokenizer>,
        factory: &ParserFactory,
    ) -> Result<Self, GrammarError> {
        let (grammar, triggers, text) = lower(spec, env)?;
        let mut matcher = Matcher::new(factory.create_parser(grammar));
        if let Some(e) = matcher.get_error() {
            return Err(GrammarError::Compile(format!("{}: {e}", spec.describe())));
        }
        for w in matcher.grammar_warnings() {
            tracing::warn!(grammar = spec.describe(), "llguidance: {w}");
        }
        Ok(CompiledGrammar {
            matcher,
            triggers,
            env: env.clone(),
            spec_hash: spec.hash(),
            grammar_text: text,
        })
    }

    /// The Lark text handed to llguidance (JSON-schema specs are wrapped in a `%json` rule).
    pub fn grammar_text(&self) -> &str {
        &self.grammar_text
    }

    /// Trigger byte strings implied by the spec (a tool family's opener in `auto` mode).
    pub fn triggers(&self) -> &[Vec<u8>] {
        &self.triggers
    }

    pub fn spec_hash(&self) -> u64 {
        self.spec_hash
    }

    pub fn tokenizer(&self) -> &Arc<GrammarTokenizer> {
        &self.env
    }
}

/// Lowers a spec to llguidance's grammar plus the implied triggers and the grammar text.
fn lower(
    spec: &GrammarSpec,
    env: &GrammarTokenizer,
) -> Result<(TopLevelGrammar, Vec<Vec<u8>>, String), GrammarError> {
    match spec {
        GrammarSpec::JsonSchema(schema) => {
            if !schema.is_object() && !schema.is_boolean() {
                return Err(GrammarError::InvalidSpec(
                    "json_schema must be an object".into(),
                ));
            }
            let schema = with_json_options(schema);
            let text = format!("start: %json {}\n", schema);
            Ok((TopLevelGrammar::from_json_schema(schema), Vec::new(), text))
        }
        GrammarSpec::Lark(text) => {
            if text.trim().is_empty() {
                return Err(GrammarError::InvalidSpec("empty lark grammar".into()));
            }
            Ok((
                TopLevelGrammar::from_lark(text.clone()),
                Vec::new(),
                text.clone(),
            ))
        }
        GrammarSpec::Regex(rx) => {
            if rx.is_empty() {
                return Err(GrammarError::InvalidSpec("empty regex".into()));
            }
            let g = TopLevelGrammar::from_regex(rx);
            let text = g.grammars[0].lark_grammar.clone().unwrap_or_default();
            Ok((g, Vec::new(), text))
        }
        GrammarSpec::ToolCalls(tc) => {
            let built = tools::build(tc, env)?;
            Ok((
                TopLevelGrammar::from_lark(built.lark.clone()),
                built.triggers,
                built.lark,
            ))
        }
    }
}

/// Where the processor is in its life cycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrammarState {
    /// Unconstrained: waiting for the trigger bytes.
    Watching,
    /// The parser is active and masks every step.
    Active,
    /// The grammar is complete and allows nothing more: only end-of-generation tokens pass.
    Complete,
    /// The parser failed; decoding continues unconstrained (see [`GrammarProcessor::error`]).
    Disabled,
}

/// Per-sequence mask timing, for metrics and tests.
#[derive(Clone, Debug, Default)]
pub struct GrammarStats {
    /// Masks computed (steps while `Active`).
    pub masks: u64,
    pub mask_time_total: Duration,
    pub mask_time_max: Duration,
    /// Steps spent unconstrained before the trigger fired.
    pub steps_watching: u64,
}

impl GrammarStats {
    pub fn mask_time_mean(&self) -> Duration {
        if self.masks == 0 {
            Duration::ZERO
        } else {
            self.mask_time_total / self.masks as u32
        }
    }
}

/// Watches the rendered byte stream for an ordered list of trigger strings.
struct TriggerWatch {
    pending: VecDeque<Vec<u8>>,
    tail: Vec<u8>,
}

impl TriggerWatch {
    fn new(triggers: Vec<Vec<u8>>) -> Self {
        TriggerWatch {
            pending: triggers.into_iter().filter(|t| !t.is_empty()).collect(),
            tail: Vec::new(),
        }
    }

    fn done(&self) -> bool {
        self.pending.is_empty()
    }

    /// Feeds `bytes`; returns the bytes after the last trigger once every trigger has fired.
    fn push(&mut self, bytes: &[u8]) -> Option<Vec<u8>> {
        self.tail.extend_from_slice(bytes);
        loop {
            let Some(trigger) = self.pending.front() else {
                return Some(std::mem::take(&mut self.tail));
            };
            match find(&self.tail, trigger) {
                Some(pos) => {
                    let rest = self.tail.split_off(pos + trigger.len());
                    self.tail = rest;
                    self.pending.pop_front();
                }
                None => {
                    // Keep only what could still be the start of the trigger.
                    let keep = trigger.len() - 1;
                    if self.tail.len() > keep {
                        self.tail.drain(..self.tail.len() - keep);
                    }
                    return None;
                }
            }
        }
    }
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

enum Phase {
    Watching(TriggerWatch),
    Active,
    Complete,
    Disabled(String),
}

/// The constrained-decoding [`LogitProcessor`].
pub struct GrammarProcessor {
    grammar: CompiledGrammar,
    matcher: Matcher,
    phase: Phase,
    mask: Option<SimpleVob>,
    stats: GrammarStats,
}

impl fmt::Debug for GrammarProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GrammarProcessor")
            .field("state", &self.state())
            .field("grammar", &self.grammar)
            .field("stats", &self.stats)
            .finish()
    }
}

impl GrammarProcessor {
    /// Creates a processor from a compiled grammar. `lazy` (for example `</think>`) is waited
    /// for *before* any trigger the spec itself implies (a tool opener in `auto` mode).
    pub fn new(grammar: CompiledGrammar, lazy: Option<LazyTrigger>) -> Self {
        let mut triggers: Vec<Vec<u8>> = Vec::new();
        if let Some(l) = lazy {
            triggers.push(l.0);
        }
        triggers.extend(grammar.triggers.iter().cloned());
        let watch = TriggerWatch::new(triggers);
        let matcher = grammar.matcher.clone();
        let mut p = GrammarProcessor {
            grammar,
            matcher,
            phase: Phase::Watching(watch),
            mask: None,
            stats: GrammarStats::default(),
        };
        if matches!(&p.phase, Phase::Watching(w) if w.done()) {
            p.start_parser(&[]);
        }
        p
    }

    /// Compiles `spec` with a one-off parser factory (use [`GrammarCache`] in the server).
    pub fn compile(
        spec: &GrammarSpec,
        env: &Arc<GrammarTokenizer>,
        lazy: Option<LazyTrigger>,
    ) -> Result<Self, GrammarError> {
        let factory = new_parser_factory(env)?;
        let grammar = CompiledGrammar::compile(spec, env, &factory)?;
        Ok(Self::new(grammar, lazy))
    }

    pub fn state(&self) -> GrammarState {
        match &self.phase {
            Phase::Watching(_) => GrammarState::Watching,
            Phase::Active => GrammarState::Active,
            Phase::Complete => GrammarState::Complete,
            Phase::Disabled(_) => GrammarState::Disabled,
        }
    }

    /// The parser error that disabled the grammar, if any.
    pub fn error(&self) -> Option<&str> {
        match &self.phase {
            Phase::Disabled(e) => Some(e),
            _ => None,
        }
    }

    pub fn stats(&self) -> &GrammarStats {
        &self.stats
    }

    pub fn grammar(&self) -> &CompiledGrammar {
        &self.grammar
    }

    /// The last mask computed (`Active` only): a set bit per allowed token.
    pub fn last_mask(&self) -> Option<&SimpleVob> {
        self.mask.as_ref()
    }

    fn disable(&mut self, what: &str, err: impl fmt::Display) {
        let msg = format!("{what}: {err}");
        tracing::warn!(
            spec_hash = format_args!("{:016x}", self.grammar.spec_hash),
            "grammar disabled, continuing unconstrained: {msg}"
        );
        self.mask = None;
        self.phase = Phase::Disabled(msg);
    }

    /// Starts the parser after the triggers fired, feeding `leftover` (bytes of the triggering
    /// token that followed the trigger) through a greedy trie tokenisation.
    fn start_parser(&mut self, leftover: &[u8]) {
        self.phase = Phase::Active;
        if !leftover.is_empty() {
            let toks = self.grammar.env.trie().greedy_tokenize(leftover);
            let total: usize = toks.iter().map(|&t| self.grammar.env.piece(t).len()).sum();
            if total != leftover.len() {
                self.disable(
                    "trigger leftover",
                    format!(
                        "{} bytes after the trigger are not tokenisable",
                        leftover.len()
                    ),
                );
                return;
            }
            if let Err(e) = self.matcher.consume_tokens(&toks) {
                self.disable("trigger leftover", e);
                return;
            }
        }
        self.after_advance();
    }

    /// Updates the phase from the matcher's stop state after tokens were consumed.
    fn after_advance(&mut self) {
        if self.matcher.is_stopped() {
            let reason = self.matcher.stop_reason();
            if reason.is_ok() {
                self.phase = Phase::Complete;
            } else {
                let err = self
                    .matcher
                    .get_error()
                    .unwrap_or_else(|| reason.to_string());
                self.disable("parser stopped", err);
            }
        }
    }

    fn force_eog(&self, logits: &mut [f32]) {
        let n = logits.len();
        let keep: Vec<Token> = self
            .grammar
            .env
            .eog()
            .iter()
            .copied()
            .filter(|&t| (t as usize) < n)
            .collect();
        if keep.is_empty() {
            return;
        }
        let saved: Vec<f32> = keep.iter().map(|&t| logits[t as usize]).collect();
        logits.fill(f32::NEG_INFINITY);
        for (&t, &v) in keep.iter().zip(saved.iter()) {
            logits[t as usize] = v;
        }
    }
}

/// `-inf` for every token whose bit is clear in `mask`. Entries beyond the mask are forbidden.
fn apply_mask(mask: &SimpleVob, logits: &mut [f32]) {
    let words = mask.as_slice();
    for (wi, chunk) in logits.chunks_mut(32).enumerate() {
        let w = words.get(wi).copied().unwrap_or(0);
        if w == u32::MAX {
            continue;
        }
        if w == 0 {
            chunk.fill(f32::NEG_INFINITY);
            continue;
        }
        for (bit, l) in chunk.iter_mut().enumerate() {
            if w & (1u32 << bit) == 0 {
                *l = f32::NEG_INFINITY;
            }
        }
    }
}

impl LogitProcessor for GrammarProcessor {
    fn process(&mut self, logits: &mut [f32]) {
        match &self.phase {
            Phase::Watching(_) => {
                self.stats.steps_watching += 1;
            }
            Phase::Disabled(_) => {}
            Phase::Complete => self.force_eog(logits),
            Phase::Active => {
                let start = Instant::now();
                match self.matcher.compute_mask_or_eos() {
                    Ok(mask) => {
                        let n = logits.len().min(mask.len());
                        let mut any = false;
                        for i in 0..n {
                            if mask.is_allowed(i as u32) {
                                any = true;
                                break;
                            }
                        }
                        if !any {
                            self.disable("mask", "no token allowed");
                            return;
                        }
                        apply_mask(&mask, logits);
                        self.mask = Some(mask);
                        let dt = start.elapsed();
                        self.stats.masks += 1;
                        self.stats.mask_time_total += dt;
                        self.stats.mask_time_max = self.stats.mask_time_max.max(dt);
                    }
                    Err(e) => {
                        if self.matcher.stop_reason() == StopReason::NotStopped
                            || !self.matcher.stop_reason().is_ok()
                        {
                            self.disable("compute_mask", e);
                        } else {
                            self.phase = Phase::Complete;
                            self.force_eog(logits);
                        }
                    }
                }
            }
        }
    }

    fn accept(&mut self, token: Token) {
        match &mut self.phase {
            Phase::Watching(watch) => {
                let piece = self.grammar.env.piece(token).to_vec();
                if let Some(leftover) = watch.push(&piece) {
                    self.start_parser(&leftover);
                }
            }
            Phase::Active => {
                if self.grammar.env.eog().contains(&token) {
                    // End of generation: the sampler could only have drawn it when the mask
                    // allowed it (grammar accepting); nothing further is constrained.
                    self.phase = Phase::Complete;
                    self.mask = None;
                    return;
                }
                if let Err(e) = self.matcher.consume_token(token) {
                    self.disable("consume_token", e);
                    return;
                }
                self.after_advance();
            }
            Phase::Complete | Phase::Disabled(_) => {}
        }
    }
}
