//! Adapter from the engine tokenizer to llguidance's `TokenizerEnv`.
//!
//! llguidance works on a byte trie of the vocabulary: every token id maps to the bytes it
//! renders. Control tokens (`<|im_end|>`, `<|python_tag|>`, `[TOOL_CALLS]`, ...) are stored
//! behind llguidance's `0xFF` marker byte so that grammar *text* can never match them — a
//! grammar literal `"<|python_tag|>"` only matches the plain-text spelling, byte by byte, while
//! the real control token is reachable only through an explicit `<[id]>` token reference.

use std::sync::Arc;

use llguidance::toktrie::{TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};
use llmario_engine_tokenizer::Tokenizer;

use super::GrammarError;
use crate::Token;

/// FNV-1a 64-bit, used for the tokenizer and grammar-spec hashes (stable across runs and
/// platforms, unlike `DefaultHasher`).
#[derive(Clone, Copy, Debug)]
pub struct Fnv1a(u64);

impl Default for Fnv1a {
    fn default() -> Self {
        Self::new()
    }
}

impl Fnv1a {
    pub fn new() -> Self {
        Fnv1a(0xcbf2_9ce4_8422_2325)
    }
    pub fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    pub fn write_u64(&mut self, v: u64) {
        self.write(&v.to_le_bytes());
    }
    pub fn finish(&self) -> u64 {
        self.0
    }
}

/// The vocabulary as llguidance sees it, plus a content hash used as the cache key.
///
/// Build it once per loaded model (`from_tokenizer`) and share it as an `Arc`; building the trie
/// for a 150k–260k vocabulary takes tens of milliseconds.
pub struct GrammarTokenizer {
    trie: TokTrie,
    hash: u64,
    eos: Token,
    eog: Vec<Token>,
}

impl std::fmt::Debug for GrammarTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrammarTokenizer")
            .field("n_vocab", &self.n_vocab())
            .field("hash", &format_args!("{:016x}", self.hash))
            .field("eos", &self.eos)
            .field("eog", &self.eog)
            .finish()
    }
}

impl GrammarTokenizer {
    /// Builds the environment from the engine tokenizer: one byte piece per token
    /// (`Tokenizer::token_to_piece`), control tokens marked special, EOS = `eos()` (or `eot()`,
    /// or the first end-of-generation id), and every `eog_ids()` entry accepted as end of
    /// grammar.
    pub fn from_tokenizer(tok: &Tokenizer) -> Result<Self, GrammarError> {
        let n = tok.n_vocab();
        let mut pieces = Vec::with_capacity(n);
        let mut special = Vec::with_capacity(n);
        for id in 0..n as u32 {
            pieces.push(tok.token_to_piece(id));
            special.push(tok.is_control(id));
        }
        let eog: Vec<Token> = tok.eog_ids().collect();
        let eos = tok
            .eos()
            .or_else(|| tok.eot())
            .or_else(|| eog.first().copied())
            .ok_or_else(|| {
                GrammarError::Tokenizer("tokenizer has no end-of-generation token".into())
            })?;
        Self::from_pieces(pieces, &special, eos, &eog)
    }

    /// Builds the environment from raw pieces. `special[i]` marks tokens that grammar text must
    /// never match (control tokens); `eos` and every id in `eog` are always treated as special
    /// and all of them end a completed grammar.
    pub fn from_pieces(
        pieces: Vec<Vec<u8>>,
        special: &[bool],
        eos: Token,
        eog: &[Token],
    ) -> Result<Self, GrammarError> {
        let n = pieces.len();
        if n == 0 {
            return Err(GrammarError::Tokenizer("empty vocabulary".into()));
        }
        if special.len() != n {
            return Err(GrammarError::Tokenizer(format!(
                "special flags length {} != vocabulary size {n}",
                special.len()
            )));
        }
        let in_range = |t: Token| (t as usize) < n;
        if !in_range(eos) {
            return Err(GrammarError::Tokenizer(format!(
                "eos token {eos} out of range for vocabulary of {n}"
            )));
        }
        if let Some(bad) = eog.iter().find(|&&t| !in_range(t)) {
            return Err(GrammarError::Tokenizer(format!(
                "eog token {bad} out of range for vocabulary of {n}"
            )));
        }
        let mut eog_all: Vec<Token> = Vec::with_capacity(eog.len() + 1);
        eog_all.push(eos);
        for &t in eog {
            if !eog_all.contains(&t) {
                eog_all.push(t);
            }
        }
        // The tokenizer reports end-of-generation ids from a set: order them so the hash is
        // stable across processes.
        eog_all[1..].sort_unstable();

        let mut hasher = Fnv1a::new();
        hasher.write_u64(n as u64);
        let mut words: Vec<Vec<u8>> = Vec::with_capacity(n);
        for (id, piece) in pieces.into_iter().enumerate() {
            let is_special = special[id] || eog_all.contains(&(id as Token));
            hasher.write_u64(piece.len() as u64);
            hasher.write(&piece);
            hasher.write(&[is_special as u8]);
            if is_special {
                if piece.is_empty() {
                    // An unrenderable control token: keep it unmatchable but unique.
                    let mut w = vec![TokTrie::SPECIAL_TOKEN_MARKER];
                    w.extend_from_slice(format!("<[{id}]>").as_bytes());
                    words.push(w);
                } else {
                    let mut w = Vec::with_capacity(piece.len() + 1);
                    w.push(TokTrie::SPECIAL_TOKEN_MARKER);
                    w.extend_from_slice(&piece);
                    words.push(w);
                }
            } else {
                words.push(piece);
            }
        }
        for &t in &eog_all {
            hasher.write_u64(t as u64);
        }

        let info = TokRxInfo::new(n as u32, eos);
        let trie = TokTrie::from(&info, &words).with_eos_tokens(&eog_all);
        Ok(GrammarTokenizer {
            trie,
            hash: hasher.finish(),
            eos,
            eog: eog_all,
        })
    }

    /// Content hash of the vocabulary (pieces, special flags, end-of-generation ids).
    pub fn hash(&self) -> u64 {
        self.hash
    }

    pub fn n_vocab(&self) -> usize {
        self.trie.vocab_size()
    }

    /// The primary end-of-sequence token.
    pub fn eos(&self) -> Token {
        self.eos
    }

    /// Every token that ends a completed grammar (`eos` first).
    pub fn eog(&self) -> &[Token] {
        &self.eog
    }

    /// True for tokens stored behind the special marker (grammar text cannot match them).
    pub fn is_special(&self, id: Token) -> bool {
        (id as usize) < self.n_vocab() && self.trie.is_special_token(id)
    }

    /// Id of the special token whose rendered text is exactly `text`, if any.
    pub fn special_token_id(&self, text: &str) -> Option<Token> {
        if text.is_empty() {
            return None;
        }
        let mut key = Vec::with_capacity(text.len() + 1);
        key.push(TokTrie::SPECIAL_TOKEN_MARKER);
        key.extend_from_slice(text.as_bytes());
        self.trie.token_id_at_bytes(&key)
    }

    /// The bytes a token renders as (special tokens without the marker byte). Empty for ids out
    /// of range.
    pub fn piece(&self, id: Token) -> &[u8] {
        if (id as usize) >= self.n_vocab() {
            return &[];
        }
        let bytes = self.trie.token(id);
        match bytes.first() {
            Some(&b) if b == TokTrie::SPECIAL_TOKEN_MARKER => &bytes[1..],
            _ => bytes,
        }
    }

    pub fn trie(&self) -> &TokTrie {
        &self.trie
    }

    /// The shared environment handle llguidance's `ParserFactory` takes.
    pub fn tok_env(self: &Arc<Self>) -> TokEnv {
        self.clone()
    }
}

impl TokenizerEnv for GrammarTokenizer {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        self.trie.greedy_tokenize(s)
    }

    /// Greedy trie tokenisation is not the model's canonical tokenisation, so llguidance must
    /// not force whole tokens from it; it then constrains by bytes, which is always correct.
    fn tokenize_is_canonical(&self) -> bool {
        false
    }
}
