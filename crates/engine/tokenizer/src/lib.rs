//! Tokenizer of the LLMario native engine, built from GGUF metadata only and matching llama.cpp
//! token for token.
//!
//! # What it reads
//!
//! [`Tokenizer::from_gguf`] consumes the `tokenizer.ggml.*` keys of a [`GgufFile`]:
//! `model` (`"gpt2"`/`"gemma4"` = byte-pair encoding with merges, `"llama"` = SentencePiece with
//! scores and byte fallback), `pre` (pre-tokenizer family, see [`pretok`]), `tokens`, `scores`,
//! `token_type`, `merges`, the special ids (`bos`, `eos`, `eot`, `eom`, `pad`, `unknown`, `sep`,
//! fill-in-the-middle ids), `add_bos_token`, `add_eos_token`, `add_space_prefix`,
//! `remove_extra_whitespaces`, and `tokenizer.chat_template`. Defaults and the end-of-generation
//! heuristics follow llama.cpp's `llama-vocab.cpp`.
//!
//! # Public API
//!
//! - [`Tokenizer::encode`] — text to ids, with optional BOS/EOS (`add_special`) and literal
//!   special tokens in the text (`parse_special`).
//! - [`Tokenizer::decode`] / [`Tokenizer::decode_with_special`] — ids to text; the latter renders
//!   control tokens as their text.
//! - [`Tokenizer::token_to_piece`], [`Tokenizer::token_type`], [`Tokenizer::n_vocab`],
//!   [`Tokenizer::bos`], [`Tokenizer::eos`], [`Tokenizer::is_eog`], [`Tokenizer::chat_template`]
//!   and friends.
//! - [`Detokenizer`] — streaming decode that only yields complete UTF-8.
//!
//! # Deviations from llama.cpp (deliberate)
//!
//! - `decode` does not apply llama.cpp's `clean_spaces` post-pass (it would break round trips);
//!   it strips the leading space of the first rendered piece when `add_space_prefix` is set.
//! - Unknown `tokenizer.ggml.pre` ids fall back to `"default"` with a warning instead of failing;
//!   families llama.cpp handles only with bespoke splitters this crate does not port (kimi-k2,
//!   afmoe, tiny_aya, superbpe) are refused.
//! - When several end-of-turn candidates exist and no `eot_token_id` is set, [`Tokenizer::eot`]
//!   picks them in a fixed priority order (llama.cpp's choice depends on hash iteration order).
//!   `is_eog` is unaffected: every candidate is in the end-of-generation set either way.
//! - SentencePiece byte fallback uses the unknown token when a byte has no token (llama.cpp
//!   throws); byte-level decoding copies unmapped codepoints through instead of emitting
//!   `[UNK_BYTE_..]` markers.

#![forbid(unsafe_code)]

pub mod bpe;
pub mod detok;
pub mod pretok;
pub mod spm;
pub mod unicode;
pub mod unicode_data;

use std::collections::{HashMap, HashSet};

use llmario_engine_core::EngineError;
use llmario_engine_formats::GgufFile;

use bpe::{byte_token_text, BpeOpts, BpeSession};
pub use detok::Detokenizer;
pub use pretok::{PreConfig, PreType};
use spm::SpmSession;

/// Errors from building a tokenizer out of GGUF metadata.
#[derive(thiserror::Error, Debug)]
pub enum TokenizerError {
    #[error("missing metadata key {0}")]
    MissingKey(String),
    #[error("bad metadata value for {0}: {1}")]
    BadValue(String, String),
    #[error("unsupported tokenizer model {0:?}")]
    UnsupportedModel(String),
    #[error("unsupported pre-tokenizer: {0}")]
    UnsupportedPre(String),
    #[error("regex: {0}")]
    Regex(String),
}

impl From<TokenizerError> for EngineError {
    fn from(e: TokenizerError) -> Self {
        EngineError::Format(e.to_string())
    }
}

/// `tokenizer.ggml.token_type` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenType {
    Undefined = 0,
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
}

impl TokenType {
    fn from_i64(v: i64) -> TokenType {
        match v {
            1 => TokenType::Normal,
            2 => TokenType::Unknown,
            3 => TokenType::Control,
            4 => TokenType::UserDefined,
            5 => TokenType::Unused,
            6 => TokenType::Byte,
            _ => TokenType::Undefined,
        }
    }
}

/// The vocabulary model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VocabKind {
    /// SentencePiece (`tokenizer.ggml.model = "llama"`).
    Spm,
    /// Byte-pair encoding with merges (`"gpt2"`, `"gemma4"`).
    Bpe,
}

// Token attribute bits (llama.cpp's `llama_token_attr`).
const ATTR_UNKNOWN: u16 = 1 << 0;
const ATTR_UNUSED: u16 = 1 << 1;
const ATTR_NORMAL: u16 = 1 << 2;
const ATTR_CONTROL: u16 = 1 << 3;
const ATTR_USER_DEFINED: u16 = 1 << 4;
const ATTR_BYTE: u16 = 1 << 5;
const ATTR_LSTRIP: u16 = 1 << 7;
const ATTR_RSTRIP: u16 = 1 << 8;
const ATTR_SPECIAL: u16 = ATTR_UNKNOWN | ATTR_CONTROL;

/// Token texts llama.cpp recognises as end-of-turn when `eot_token_id` is absent, in the
/// priority order this crate uses.
const EOT_TEXTS: &[&str] = &[
    "<|im_end|>",
    "<|eot_id|>",
    "<|end|>",
    "<end_of_turn>",
    "<|endoftext|>",
    "<|end_of_text|>",
    "<EOT>",
    "_<EOT>",
    "[EOT]",
    "<｜end▁of▁sentence｜>",
    "<end_of_utterance>",
];

/// Token texts llama.cpp always treats as end-of-generation.
const EOG_TEXTS: &[&str] = &[
    "<|eot_id|>",
    "<|im_end|>",
    "<|end|>",
    "<|return|>",
    "<|call|>",
    "<|flush|>",
    "<|calls|>",
    "<end_of_turn>",
    "<|endoftext|>",
    "</s>",
    "<|eom_id|>",
    "<EOT>",
    "_<EOT>",
    "[EOT]",
    "[EOS]",
    "<|end_of_text|>",
    "<end_of_utterance>",
    "<eos>",
    "<turn|>",
    "<|tool_response>",
    "<｜end▁of▁sentence｜>",
    "[e~[",
];

const FIM_PAD_TEXTS: &[&str] = &["<|fim_pad|>", "<fim-pad>", "<fim_pad>", "<PAD>", "[PAD]"];
const FIM_REP_TEXTS: &[&str] = &[
    "<|fim_repo|>",
    "<|repo_name|>",
    "<fim-repo>",
    "<REPO>",
    "<reponame>",
];
const FIM_SEP_TEXTS: &[&str] = &["<|file_sep|>"];

/// A tokenizer built from GGUF metadata. Cheap to share across threads (`Send + Sync`).
pub struct Tokenizer {
    kind: VocabKind,
    pre_name: String,
    pre: PreConfig,
    pretok: Option<pretok::PreTokenizer>,
    tokens: Vec<Box<[u8]>>,
    scores: Vec<f32>,
    types: Vec<TokenType>,
    attrs: Vec<u16>,
    token_to_id: HashMap<Box<[u8]>, u32>,
    merges: HashMap<Box<[u8]>, u32>,
    /// Control, user-defined and unknown tokens, longest text first.
    special_tokens: Vec<u32>,
    eog: HashSet<u32>,
    bos: Option<u32>,
    eos: Option<u32>,
    eot: Option<u32>,
    eom: Option<u32>,
    unk: Option<u32>,
    sep: Option<u32>,
    pad: Option<u32>,
    add_bos: bool,
    add_eos: bool,
    add_space_prefix: bool,
    remove_extra_whitespaces: bool,
    chat_template: Option<String>,
}

/// A piece of input after special-token splitting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Frag {
    Text(usize, usize),
    Token(u32),
}

const ESCAPED_SPACE: &[u8] = "\u{2581}".as_bytes();

fn escape_whitespace(text: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len() + 8);
    for &b in text {
        if b == b' ' {
            out.extend_from_slice(ESCAPED_SPACE);
        } else {
            out.push(b);
        }
    }
    out
}

fn unescape_whitespace(text: &[u8], out: &mut Vec<u8>) {
    let mut i = 0;
    while i < text.len() {
        if text[i..].starts_with(ESCAPED_SPACE) {
            out.push(b' ');
            i += ESCAPED_SPACE.len();
        } else {
            out.push(text[i]);
            i += 1;
        }
    }
}

fn is_c_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0B | 0x0C | b'\r')
}

/// Byte value of a `<0xXX>` token text, if it has that shape.
fn byte_token_value(text: &[u8]) -> Option<u8> {
    if text.len() == 6 && text.starts_with(b"<0x") && text[5] == b'>' {
        let hex = std::str::from_utf8(&text[3..5]).ok()?;
        u8::from_str_radix(hex, 16).ok()
    } else {
        None
    }
}

impl Tokenizer {
    /// Build a tokenizer from the metadata of an opened GGUF file.
    pub fn from_gguf(g: &GgufFile) -> Result<Tokenizer, TokenizerError> {
        let key = |k: &str| format!("tokenizer.ggml.{k}");
        let model = g
            .get_str(&key("model"))
            .ok_or_else(|| TokenizerError::MissingKey(key("model")))?;
        let kind = match model {
            "llama" => VocabKind::Spm,
            "gpt2" | "gemma4" => VocabKind::Bpe,
            other => return Err(TokenizerError::UnsupportedModel(other.to_string())),
        };

        // Tokens, scores, types.
        let toks = g
            .get_array(&key("tokens"))
            .ok_or_else(|| TokenizerError::MissingKey(key("tokens")))?;
        let n = toks.len();
        let mut tokens: Vec<Box<[u8]>> = Vec::with_capacity(n);
        for (i, v) in toks.iter().enumerate() {
            let s = v.as_str().ok_or_else(|| {
                TokenizerError::BadValue(key("tokens"), format!("entry {i} is not a string"))
            })?;
            if s.is_empty() {
                tokens.push(format!("[EMPTY_{i}]").into_bytes().into_boxed_slice());
            } else {
                tokens.push(s.as_bytes().into());
            }
        }
        let scores: Vec<f32> = match g.get_array(&key("scores")) {
            Some(a) if a.len() >= n => a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect(),
            Some(a) => {
                return Err(TokenizerError::BadValue(
                    key("scores"),
                    format!("{} scores for {n} tokens", a.len()),
                ))
            }
            None => vec![0.0; n],
        };
        let types: Vec<TokenType> = match g.get_array(&key("token_type")) {
            Some(a) if a.len() >= n => a
                .iter()
                .map(|v| TokenType::from_i64(v.as_i64().unwrap_or(0)))
                .collect(),
            Some(a) => {
                return Err(TokenizerError::BadValue(
                    key("token_type"),
                    format!("{} types for {n} tokens", a.len()),
                ))
            }
            None => vec![TokenType::Normal; n],
        };
        let mut attrs: Vec<u16> = types
            .iter()
            .map(|t| match t {
                TokenType::Normal => ATTR_NORMAL,
                TokenType::Unknown => ATTR_UNKNOWN,
                TokenType::Control => ATTR_CONTROL,
                TokenType::UserDefined => ATTR_USER_DEFINED,
                TokenType::Unused => ATTR_UNUSED,
                TokenType::Byte => ATTR_BYTE,
                TokenType::Undefined => 0,
            })
            .collect();
        let mut token_to_id: HashMap<Box<[u8]>, u32> = HashMap::with_capacity(n);
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }

        // Merges.
        let mut merges = HashMap::new();
        if kind == VocabKind::Bpe {
            let arr = g
                .get_array(&key("merges"))
                .ok_or_else(|| TokenizerError::MissingKey(key("merges")))?;
            merges.reserve(arr.len());
            for (i, v) in arr.iter().enumerate() {
                let s = v.as_str().ok_or_else(|| {
                    TokenizerError::BadValue(key("merges"), format!("entry {i} is not a string"))
                })?;
                // llama.cpp splits at the first space after index 0 and keys on (first, second).
                if s.len() > 1 && s.as_bytes()[1..].contains(&b' ') {
                    merges.insert(s.as_bytes().into(), i as u32);
                }
            }
        }

        // Pre-tokenizer and the flags it implies.
        let (pre_name, pre) = match kind {
            VocabKind::Spm => (
                String::new(),
                PreConfig {
                    pre_type: PreType::Default,
                    ignore_merges: false,
                    add_bos: true,
                    clean_spaces: false,
                    escape_whitespaces: false,
                },
            ),
            VocabKind::Bpe => {
                let name = if model == "gemma4" {
                    "gemma4".to_string()
                } else {
                    g.get_str(&key("pre")).unwrap_or("").to_string()
                };
                let cfg = if name.is_empty() {
                    tracing::warn!(
                        "tokenizer.ggml.pre missing; using 'default' (quality may degrade)"
                    );
                    pretok::pre_config("default")
                } else {
                    match pretok::pre_config(&name) {
                        Some(c) => Some(c),
                        None => {
                            tracing::warn!("unknown tokenizer.ggml.pre {name:?}; using 'default'");
                            pretok::pre_config("default")
                        }
                    }
                };
                let cfg = cfg.ok_or_else(|| TokenizerError::UnsupportedPre(name.clone()))?;
                if cfg.pre_type == PreType::Unsupported {
                    return Err(TokenizerError::UnsupportedPre(name));
                }
                (name, cfg)
            }
        };
        let pretok = match kind {
            VocabKind::Bpe => Some(pretok::PreTokenizer::new(pre.pre_type)?),
            VocabKind::Spm => None,
        };

        let mut add_space_prefix = kind == VocabKind::Spm;
        if let Some(v) = g.get_bool(&key("add_space_prefix")) {
            add_space_prefix = v;
        }
        let remove_extra_whitespaces = g
            .get_bool(&key("remove_extra_whitespaces"))
            .unwrap_or(false);

        // Special ids: defaults, then keys (ignored when out of range, like llama.cpp).
        let (mut bos, mut eos, mut unk) = match kind {
            VocabKind::Spm => (Some(1u32), Some(2u32), Some(0u32)),
            VocabKind::Bpe if model == "gemma4" => (None, None, None),
            VocabKind::Bpe => (Some(11), Some(11), None),
        };
        if pre_name == "glm4" || pre_name == "chatglm-bpe" {
            bos = None;
        }
        let (mut eot, mut eom, mut sep, mut pad) = (None, None, None, None);
        let (mut fim_pad, mut fim_rep, mut fim_sep) = (None, None, None);
        let in_range = |id: Option<u32>| id.filter(|&i| (i as usize) < n);
        let read_id = |name: &str, cur: Option<u32>| -> Option<u32> {
            let cur = in_range(cur);
            match g.get_u32(&key(name)) {
                Some(v) if (v as usize) < n => Some(v),
                Some(v) => {
                    tracing::warn!(
                        "bad special token {}={v}, keeping default {cur:?}",
                        key(name)
                    );
                    cur
                }
                None => cur,
            }
        };
        bos = read_id("bos_token_id", bos);
        eos = read_id("eos_token_id", eos);
        eot = read_id("eot_token_id", eot);
        eom = read_id("eom_token_id", eom);
        unk = read_id("unknown_token_id", unk);
        sep = read_id("seperator_token_id", sep);
        pad = read_id("padding_token_id", pad);
        fim_pad = read_id("fim_pad_token_id", fim_pad);
        fim_rep = read_id("fim_rep_token_id", fim_rep);
        fim_sep = read_id("fim_sep_token_id", fim_sep);

        let mut add_bos = pre.add_bos;
        let mut add_eos = false;
        if let Some(v) = g.get_bool(&key("add_bos_token")) {
            add_bos = v;
        }
        if let Some(v) = g.get_bool(&key("add_eos_token")) {
            add_eos = v;
        }
        if pre.pre_type == PreType::Gemma4 && !add_bos {
            tracing::warn!(
                "overriding add_bos_token to true for Gemma 4 (llama.cpp does the same)"
            );
            add_bos = true;
        }

        // Auto-detect special tokens by text (llama.cpp's workaround for missing metadata).
        let find = |texts: &[&str]| -> Option<u32> {
            texts
                .iter()
                .find_map(|t| token_to_id.get(t.as_bytes()).copied())
        };
        let mark_control = |id: u32, attrs: &mut Vec<u16>| {
            if attrs[id as usize] & ATTR_CONTROL == 0 {
                tracing::warn!(
                    "control-looking token {id} {:?} was not control-type; overriding",
                    String::from_utf8_lossy(&tokens[id as usize])
                );
                attrs[id as usize] |= ATTR_CONTROL;
            }
        };
        if eot.is_none() {
            eot = find(EOT_TEXTS);
            if let Some(id) = eot {
                mark_control(id, &mut attrs);
            }
        }
        if eom.is_none() {
            eom = find(&["<|eom_id|>"]);
            if let Some(id) = eom {
                mark_control(id, &mut attrs);
            }
        }
        for (slot, texts) in [
            (&mut fim_pad, FIM_PAD_TEXTS),
            (&mut fim_rep, FIM_REP_TEXTS),
            (&mut fim_sep, FIM_SEP_TEXTS),
        ] {
            if slot.is_none() {
                *slot = find(texts);
                if let Some(id) = *slot {
                    mark_control(id, &mut attrs);
                }
            }
        }

        // Control tokens whose text mentions "unused" are unused.
        for (i, t) in tokens.iter().enumerate() {
            if attrs[i] & ATTR_CONTROL != 0 && memfind(t, b"unused").is_some() {
                attrs[i] |= ATTR_UNUSED;
            }
        }

        // End-of-generation set.
        let mut eog: HashSet<u32> = HashSet::new();
        eog.extend([fim_pad, fim_rep, fim_sep].into_iter().flatten());
        for text in EOG_TEXTS {
            if let Some(&id) = token_to_id.get(text.as_bytes()) {
                eog.insert(id);
                mark_control(id, &mut attrs);
            }
        }
        // gpt-oss: always render these.
        for text in ["<|channel|>", "<|message|>", "<|start|>", "<|constrain|>"] {
            if let Some(&id) = token_to_id.get(text.as_bytes()) {
                attrs[id as usize] = ATTR_USER_DEFINED;
            }
        }
        eog.extend([eos, eot, eom].into_iter().flatten());
        // o200k_harmony / solar-open: "<|end|>" is not end-of-generation next to return/call.
        {
            let has = |t: &str| {
                token_to_id
                    .get(t.as_bytes())
                    .is_some_and(|id| eog.contains(id))
            };
            let has_return = has("<|return|>");
            let has_call = has("<|call|>") || has("<|calls|>");
            let has_flush = has("<|flush|>");
            let has_tool_response = has("<|tool_response>");
            let end_id = token_to_id.get(b"<|end|>".as_slice()).copied();
            let s_id = token_to_id.get(b"</s>".as_slice()).copied();
            if let Some(end_id) = end_id {
                if eog.contains(&end_id) && has_call && (has_return || has_flush) {
                    eog.remove(&end_id);
                    attrs[end_id as usize] = ATTR_USER_DEFINED;
                }
            }
            // gemma4 / paddleocr: "</s>" is not end-of-generation next to "<|tool_response>".
            if let Some(s_id) = s_id {
                if has_tool_response && eog.contains(&s_id) {
                    eog.remove(&s_id);
                    attrs[s_id as usize] = ATTR_NORMAL;
                }
            }
        }

        // Special token cache: longest text first (stable on id for equal lengths).
        let mut special_tokens: Vec<u32> = (0..n as u32)
            .filter(|&id| {
                attrs[id as usize] & (ATTR_CONTROL | ATTR_USER_DEFINED | ATTR_UNKNOWN) != 0
            })
            .collect();
        special_tokens.sort_by(|&a, &b| {
            tokens[b as usize]
                .len()
                .cmp(&tokens[a as usize].len())
                .then(a.cmp(&b))
        });

        // Per-model attribute quirks llama.cpp hard-codes.
        let model_name = g.get_str("general.name").unwrap_or("").to_lowercase();
        if model_name.contains("phi-3") || model_name.contains("phi3") {
            for &id in &special_tokens {
                attrs[id as usize] |= ATTR_RSTRIP;
            }
            if let Some(&id) = token_to_id.get(b"</s>".as_slice()) {
                attrs[id as usize] |= ATTR_RSTRIP;
            }
            for t in ["<unk>", "<s>", "<|endoftext|>"] {
                if let Some(&id) = token_to_id.get(t.as_bytes()) {
                    attrs[id as usize] &= !ATTR_RSTRIP;
                }
            }
        }
        let _ = ATTR_LSTRIP;

        let chat_template = g.get_str("tokenizer.chat_template").map(str::to_string);

        Ok(Tokenizer {
            kind,
            pre_name,
            pre,
            pretok,
            tokens,
            scores,
            types,
            attrs,
            token_to_id,
            merges,
            special_tokens,
            eog,
            bos,
            eos,
            eot,
            eom,
            unk,
            sep,
            pad,
            add_bos,
            add_eos,
            add_space_prefix,
            remove_extra_whitespaces,
            chat_template,
        })
    }

    // ---- metadata accessors -------------------------------------------------------------

    pub fn kind(&self) -> VocabKind {
        self.kind
    }
    /// The `tokenizer.ggml.pre` id (empty for SentencePiece).
    pub fn pre_name(&self) -> &str {
        &self.pre_name
    }
    pub fn pre_type(&self) -> PreType {
        self.pre.pre_type
    }
    pub fn n_vocab(&self) -> usize {
        self.tokens.len()
    }
    pub fn bos(&self) -> Option<u32> {
        self.bos
    }
    pub fn eos(&self) -> Option<u32> {
        self.eos
    }
    pub fn eot(&self) -> Option<u32> {
        self.eot
    }
    pub fn eom(&self) -> Option<u32> {
        self.eom
    }
    pub fn unk(&self) -> Option<u32> {
        self.unk
    }
    pub fn sep(&self) -> Option<u32> {
        self.sep
    }
    pub fn pad(&self) -> Option<u32> {
        self.pad
    }
    pub fn add_bos(&self) -> bool {
        self.add_bos
    }
    pub fn add_eos(&self) -> bool {
        self.add_eos
    }
    pub fn add_space_prefix(&self) -> bool {
        self.add_space_prefix
    }
    pub fn remove_extra_whitespaces(&self) -> bool {
        self.remove_extra_whitespaces
    }
    /// The Jinja chat template embedded in the file, if any.
    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }
    /// True for every token that ends generation: `eos`, `eot`, `eom` and the control tokens
    /// llama.cpp recognises by text (`<|im_end|>`, `<|eot_id|>`, `<end_of_turn>`, ...).
    pub fn is_eog(&self, id: u32) -> bool {
        self.eog.contains(&id)
    }
    /// All end-of-generation ids.
    pub fn eog_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.eog.iter().copied()
    }
    /// The `tokenizer.ggml.token_type` of a token (`Undefined` for ids out of range).
    pub fn token_type(&self, id: u32) -> TokenType {
        self.types
            .get(id as usize)
            .copied()
            .unwrap_or(TokenType::Undefined)
    }
    /// Control or unknown token (rendered only by `decode_with_special`).
    pub fn is_control(&self, id: u32) -> bool {
        self.attrs
            .get(id as usize)
            .is_some_and(|a| a & ATTR_SPECIAL != 0)
    }
    /// Raw vocabulary text of a token (byte-level BPE tokens are in their encoded form).
    pub fn token_text(&self, id: u32) -> Option<&[u8]> {
        self.tokens.get(id as usize).map(|t| &t[..])
    }
    /// Score of a token (SentencePiece); 0 when the file has none.
    pub fn token_score(&self, id: u32) -> f32 {
        self.scores.get(id as usize).copied().unwrap_or(0.0)
    }
    /// Look a token up by its raw vocabulary text.
    pub fn token_to_id(&self, text: &[u8]) -> Option<u32> {
        self.token_to_id.get(text).copied()
    }

    // ---- encoding ------------------------------------------------------------------------

    /// Tokenize `text`. `add_special` adds BOS/EOS according to the model's `add_bos_token` /
    /// `add_eos_token`; `parse_special` lets control tokens written literally in the text (such
    /// as `<|im_start|>`) become single tokens (user-defined tokens always do).
    pub fn encode(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<u32> {
        let frags = self.partition(text, parse_special);
        let mut out = Vec::with_capacity(text.len() / 3 + 2);
        match self.kind {
            VocabKind::Spm => {
                let mut session = SpmSession::new(&self.token_to_id, &self.scores, self.unk);
                let mut prev_special = true;
                if add_special && self.add_bos {
                    if let Some(bos) = self.bos {
                        out.push(bos);
                    }
                }
                for frag in frags {
                    match frag {
                        Frag::Text(a, b) => {
                            let mut piece = Vec::with_capacity(b - a + 1);
                            if self.add_space_prefix && prev_special {
                                piece.push(b' ');
                            }
                            piece.extend_from_slice(&text.as_bytes()[a..b]);
                            let escaped = escape_whitespace(&piece);
                            session.tokenize(&escaped, &mut out);
                            prev_special = false;
                        }
                        Frag::Token(id) => {
                            out.push(id);
                            prev_special = true;
                        }
                    }
                }
                if add_special && self.add_eos {
                    if let Some(eos) = self.eos {
                        out.push(eos);
                    }
                }
            }
            VocabKind::Bpe => {
                let pretok = match &self.pretok {
                    Some(p) => p,
                    None => return out,
                };
                let mut session = BpeSession::new(&self.token_to_id, &self.merges);
                let opts = BpeOpts {
                    ignore_merges: self.pre.ignore_merges,
                    gemma4_newlines: self.pre.pre_type == PreType::Gemma4,
                    byte_encode: pretok.byte_encode,
                };
                if add_special && self.add_bos {
                    if let Some(bos) = self.bos {
                        out.push(bos);
                    }
                }
                for frag in frags {
                    match frag {
                        Frag::Text(a, b) => {
                            let chunk = &text[a..b];
                            let words = if self.pre.escape_whitespaces {
                                let escaped = chunk.replace(' ', "\u{2581}");
                                pretok.split(&escaped)
                            } else {
                                pretok.split(chunk)
                            };
                            for w in &words {
                                session.tokenize_word(w, opts, &mut out);
                            }
                        }
                        Frag::Token(id) => out.push(id),
                    }
                }
                if add_special && self.add_eos {
                    if let Some(eos) = self.eos {
                        out.push(eos);
                    }
                }
            }
        }
        out
    }

    /// llama.cpp's `tokenizer_st_partition`: carve literal special tokens out of the text, longest
    /// token first, each token scanned over every text fragment before the next.
    fn partition(&self, text: &str, parse_special: bool) -> Vec<Frag> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut frags = vec![Frag::Text(0, text.len())];
        let mut present = [false; 256];
        for &b in text.as_bytes() {
            present[b as usize] = true;
        }
        let mut pieces: Vec<Frag> = Vec::new();
        for &id in &self.special_tokens {
            let attr = self.attrs[id as usize];
            if !parse_special && attr & ATTR_SPECIAL != 0 {
                continue;
            }
            let tok = &self.tokens[id as usize];
            if tok.is_empty() || !present[tok[0] as usize] {
                continue;
            }
            let Ok(tok_str) = std::str::from_utf8(tok) else {
                continue;
            };
            let mut i = 0;
            while i < frags.len() {
                let Frag::Text(a, b) = frags[i] else {
                    i += 1;
                    continue;
                };
                pieces.clear();
                let mut off = a;
                while let Some(p) = text[off..b].find(tok_str) {
                    let m = off + p;
                    let mut left_end = m;
                    if attr & ATTR_LSTRIP != 0 {
                        while left_end > off && is_c_space(text.as_bytes()[left_end - 1]) {
                            left_end -= 1;
                        }
                    }
                    if left_end > off {
                        pieces.push(Frag::Text(off, left_end));
                    }
                    pieces.push(Frag::Token(id));
                    let mut right = m + tok.len();
                    if attr & ATTR_RSTRIP != 0 {
                        while right < b && is_c_space(text.as_bytes()[right]) {
                            right += 1;
                        }
                    }
                    off = right;
                    if off >= b {
                        break;
                    }
                }
                if pieces.is_empty() {
                    i += 1;
                    continue;
                }
                if off < b {
                    pieces.push(Frag::Text(off, b));
                }
                let added = pieces.len();
                frags.splice(i..=i, pieces.drain(..));
                i += added;
            }
        }
        frags
    }

    // ---- decoding ------------------------------------------------------------------------

    /// Render one token's bytes as llama.cpp's `token_to_piece(special)` does; control and
    /// unknown tokens render as nothing unless `special`.
    pub(crate) fn piece_into(&self, id: u32, special: bool, out: &mut Vec<u8>) {
        let Some(text) = self.tokens.get(id as usize) else {
            return;
        };
        let attr = self.attrs[id as usize];
        if !special && attr & ATTR_SPECIAL != 0 {
            return;
        }
        if attr & (ATTR_SPECIAL | ATTR_USER_DEFINED) != 0 {
            out.extend_from_slice(text);
        } else if attr & ATTR_NORMAL != 0 {
            match self.kind {
                VocabKind::Spm => unescape_whitespace(text, out),
                VocabKind::Bpe if self.pre.escape_whitespaces => unescape_whitespace(text, out),
                VocabKind::Bpe => out.extend_from_slice(&unicode::byte_decode(text)),
            }
        } else if attr & ATTR_BYTE != 0 {
            match byte_token_value(text) {
                Some(b) => out.push(b),
                None => out.extend_from_slice(text),
            }
        }
    }

    /// The rendered bytes of a token (control tokens as their text). Byte tokens yield one byte,
    /// so the result need not be valid UTF-8 on its own.
    pub fn token_to_piece(&self, id: u32) -> Vec<u8> {
        let mut out = Vec::new();
        self.piece_into(id, true, &mut out);
        out
    }

    fn decode_impl(&self, ids: &[u32], special: bool) -> String {
        let mut bytes = Vec::with_capacity(ids.len() * 4);
        let mut remove_space = self.add_space_prefix;
        for &id in ids {
            let start = bytes.len();
            self.piece_into(id, special, &mut bytes);
            if remove_space && bytes.len() > start {
                if bytes[start] == b' ' {
                    bytes.remove(start);
                }
                remove_space = false;
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Ids to text. Control/unknown tokens render as nothing; invalid UTF-8 becomes U+FFFD.
    pub fn decode(&self, ids: &[u32]) -> String {
        self.decode_impl(ids, false)
    }

    /// Like [`Tokenizer::decode`] but control tokens render as their text.
    pub fn decode_with_special(&self, ids: &[u32]) -> String {
        self.decode_impl(ids, true)
    }

    /// Id of the `<0xXX>` / single-byte token for a byte, if the vocabulary has one.
    pub fn byte_to_token(&self, b: u8) -> Option<u32> {
        match self.kind {
            VocabKind::Spm => self
                .token_to_id
                .get(byte_token_text(b).as_slice())
                .or_else(|| self.token_to_id.get(&[b][..]))
                .copied(),
            VocabKind::Bpe => {
                let mut buf = [0u8; 4];
                let s = unicode::byte_to_char(b).encode_utf8(&mut buf);
                self.token_to_id.get(s.as_bytes()).copied()
            }
        }
    }
}

fn memfind(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
mod tests;
