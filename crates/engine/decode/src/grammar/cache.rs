//! Compiled-grammar cache keyed by (spec hash, tokenizer hash).
//!
//! Two levels: one llguidance `ParserFactory` per tokenizer (it holds the token-slice
//! precomputation, the expensive part for a 150k–260k vocabulary) and an LRU of compiled
//! grammars. A hit hands out a clone of the fresh parser, so every request gets independent
//! state without recompiling the grammar.

use std::collections::HashMap;
use std::sync::Arc;

use llguidance::ParserFactory;

use super::{CompiledGrammar, GrammarError, GrammarSpec, GrammarTokenizer};

struct Entry {
    grammar: CompiledGrammar,
    last_used: u64,
}

/// LRU cache of compiled grammars. Not internally synchronised: wrap it in a `Mutex` to share
/// across request handlers.
pub struct GrammarCache {
    factories: HashMap<u64, Arc<ParserFactory>>,
    entries: HashMap<(u64, u64), Entry>,
    limit: usize,
    tick: u64,
    hits: u64,
    misses: u64,
}

impl std::fmt::Debug for GrammarCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrammarCache")
            .field("entries", &self.entries.len())
            .field("limit", &self.limit)
            .field("tokenizers", &self.factories.len())
            .field("hits", &self.hits)
            .field("misses", &self.misses)
            .finish()
    }
}

impl GrammarCache {
    /// A cache holding at most `limit` compiled grammars (`limit = 0` caches nothing but still
    /// reuses the per-tokenizer parser factories).
    pub fn new(limit: usize) -> Self {
        GrammarCache {
            factories: HashMap::new(),
            entries: HashMap::new(),
            limit,
            tick: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// The parser factory for a tokenizer, built on first use.
    pub fn factory(
        &mut self,
        env: &Arc<GrammarTokenizer>,
    ) -> Result<Arc<ParserFactory>, GrammarError> {
        if let Some(f) = self.factories.get(&env.hash()) {
            return Ok(f.clone());
        }
        let f = Arc::new(super::new_parser_factory(env)?);
        self.factories.insert(env.hash(), f.clone());
        Ok(f)
    }

    /// The compiled grammar for `spec` on `env`, compiling and caching it on a miss.
    pub fn get_or_compile(
        &mut self,
        spec: &GrammarSpec,
        env: &Arc<GrammarTokenizer>,
    ) -> Result<CompiledGrammar, GrammarError> {
        self.tick += 1;
        let key = (spec.hash(), env.hash());
        if let Some(e) = self.entries.get_mut(&key) {
            e.last_used = self.tick;
            self.hits += 1;
            return Ok(e.grammar.clone());
        }
        self.misses += 1;
        let factory = self.factory(env)?;
        let grammar = CompiledGrammar::compile(spec, env, &factory)?;
        if self.limit > 0 {
            if self.entries.len() >= self.limit {
                if let Some(&victim) = self
                    .entries
                    .iter()
                    .min_by_key(|(_, e)| e.last_used)
                    .map(|(k, _)| k)
                {
                    self.entries.remove(&victim);
                }
            }
            self.entries.insert(
                key,
                Entry {
                    grammar: grammar.clone(),
                    last_used: self.tick,
                },
            );
        }
        Ok(grammar)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn limit(&self) -> usize {
        self.limit
    }

    /// (hits, misses) since creation.
    pub fn counters(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }

    /// Drops every cached grammar (parser factories are kept).
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}
