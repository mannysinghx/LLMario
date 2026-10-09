//! Pre-tokenization: splitting text into the chunks that BPE merges run on, selected by
//! `tokenizer.ggml.pre` exactly as llama.cpp's `llm_tokenizer_bpe` constructor does.
//!
//! llama.cpp applies each pattern of a pre-type in order, refining the previous split. Five
//! patterns have hand-written state machines in its `unicode.cpp` (`gpt2`, `llama3`, `qwen2`,
//! `qwen35`, newline runs); those are ported literally here. Every other pattern runs through
//! `std::regex` over a *collapsed* text where each non-ASCII codepoint is replaced by one marker
//! byte for its category (`\xD1` number, `\xD2` letter, `\xD3` punctuation, `\xD4` mark, `\xD5`
//! symbol, `\xD0` other, `\x0B` whitespace) and `\p{..}` classes are rewritten to those markers
//! plus their ASCII members. [`Collapsed`] reproduces that text and [`translate`] that rewrite, so
//! `fancy-regex` sees the same haystack and pattern semantics as llama.cpp's `std::regex`.

use crate::unicode::{self, byte_encode, flags, tolower, Flags};
use crate::TokenizerError;

/// The pre-tokenizer families llama.cpp distinguishes (its `llama_vocab_pre_type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub enum PreType {
    Default,
    Llama3,
    Jais2,
    DeepseekLlm,
    Deepseek3Llm,
    Spark25,
    Youtu,
    DeepseekCoder,
    Falcon,
    Starcoder,
    Gpt2,
    Qwen2,
    Qwen35,
    Poro,
    Chatglm4,
    Viking,
    Tekken,
    Chameleon,
    Gpt4o,
    GraniteEmbMulti,
    Bailingmoe,
    SeedCoder,
    Ufakzeka,
    Grok2,
    Laguna,
    ExaoneMoe,
    Gemma4,
    SarvamMoe,
    Minicpm5,
    /// Known to llama.cpp but only through a bespoke splitter this crate does not port
    /// (kimi-k2, afmoe, tiny_aya, superbpe).
    Unsupported,
}

/// What a `tokenizer.ggml.pre` id implies besides the split patterns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreConfig {
    pub pre_type: PreType,
    /// Emit a whole chunk as one token when it is in the vocabulary (skip merges).
    pub ignore_merges: bool,
    /// llama.cpp forces `add_bos` on for these families (the GGUF key still overrides).
    pub add_bos: bool,
    /// llama.cpp applies its `clean_spaces` detokenize pass; recorded, not applied.
    pub clean_spaces: bool,
    /// Spaces are turned into `▁` before splitting and tokens carry `▁` (Gemma 4 style).
    pub escape_whitespaces: bool,
}

/// Map a `tokenizer.ggml.pre` id to its configuration; `None` for ids llama.cpp rejects.
pub fn pre_config(name: &str) -> Option<PreConfig> {
    let base = |pre_type| PreConfig {
        pre_type,
        ignore_merges: false,
        add_bos: false,
        clean_spaces: true,
        escape_whitespaces: false,
    };
    let no_clean = |pre_type| PreConfig {
        clean_spaces: false,
        ..base(pre_type)
    };
    Some(match name {
        "default" => base(PreType::Default),
        "minicpm5" => PreConfig {
            ignore_merges: true,
            ..base(PreType::Minicpm5)
        },
        "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "falcon-h1" | "pixtral" | "midm-2.0"
        | "lfm2" | "jina-v5-nano" => PreConfig {
            ignore_merges: true,
            add_bos: true,
            ..base(PreType::Llama3)
        },
        "deepseek-llm" => no_clean(PreType::DeepseekLlm),
        "deepseek-coder" => no_clean(PreType::DeepseekCoder),
        "deepseek-v3" => no_clean(PreType::Deepseek3Llm),
        "spark2_5" => no_clean(PreType::Spark25),
        "youtu" => PreConfig {
            ignore_merges: true,
            ..no_clean(PreType::Youtu)
        },
        "falcon" => base(PreType::Falcon),
        "mpt" | "olmo" | "jais" | "trillion" | "granite-docling" | "gpt-2" | "phi-2"
        | "jina-es" | "jina-de" | "gigachat" | "jina-v2-es" | "jina-v2-de" | "a.x-4.0"
        | "mellum" | "modern-bert" | "exaone4" | "jina-v1-en" | "jina-v2-code" | "roberta-bpe" => {
            if matches!(name, "trillion" | "granite-docling") {
                no_clean(PreType::Gpt2)
            } else {
                base(PreType::Gpt2)
            }
        }
        "jais-2" => base(PreType::Jais2),
        "gemma4" | "granite-embed-multi-311m" => PreConfig {
            escape_whitespaces: true,
            ..base(PreType::Gemma4)
        },
        "sarvam-moe" => PreConfig {
            escape_whitespaces: true,
            ..no_clean(PreType::SarvamMoe)
        },
        "refact" | "starcoder" | "codeshell" | "exaone" | "minerva-7b" | "mellum2" => {
            base(PreType::Starcoder)
        }
        "command-r" | "smollm" => no_clean(PreType::Starcoder),
        "qwen2" | "deepseek-r1-qwen" | "kormo" | "f2llmv2" => no_clean(PreType::Qwen2),
        "megrez" | "stablelm2" | "hunyuan" | "solar-open" => {
            if name == "megrez" || name == "stablelm2" {
                base(PreType::Qwen2)
            } else {
                no_clean(PreType::Qwen2)
            }
        }
        "qwen35" => no_clean(PreType::Qwen35),
        "dbrx" | "smaug-bpe" => base(PreType::Llama3),
        "poro-chat" | "bloom" | "gpt3-finnish" => {
            if name == "poro-chat" {
                no_clean(PreType::Poro)
            } else {
                base(PreType::Poro)
            }
        }
        "glm4" | "chatglm-bpe" => base(PreType::Chatglm4),
        "viking" => no_clean(PreType::Viking),
        "tekken" => PreConfig {
            ignore_merges: true,
            add_bos: true,
            ..no_clean(PreType::Tekken)
        },
        "exaone-moe" => base(PreType::ExaoneMoe),
        "chameleon" => PreConfig {
            add_bos: true,
            ..no_clean(PreType::Chameleon)
        },
        "gpt-4o" | "llama4" | "kanana2" | "talkie" | "minimax-m2" => no_clean(PreType::Gpt4o),
        "granite-embed-multi-97m" => PreConfig {
            ignore_merges: true,
            ..no_clean(PreType::GraniteEmbMulti)
        },
        "bailingmoe" | "bailingmoe2" | "llada-moe" => no_clean(PreType::Bailingmoe),
        "seed-coder" => no_clean(PreType::SeedCoder),
        "hunyuan-dense" | "hy_v4" | "joyai-llm" => no_clean(PreType::Deepseek3Llm),
        "ufakzeka" => no_clean(PreType::Ufakzeka),
        "grok-2" => no_clean(PreType::Grok2),
        "laguna" => no_clean(PreType::Laguna),
        "kimi-k2" | "afmoe" | "tiny_aya" | "cohere2moe" | "superbpe" => {
            no_clean(PreType::Unsupported)
        }
        _ => return None,
    })
}

/// llama.cpp's split patterns per family, and whether chunks are GPT-2 byte-encoded.
fn patterns(pre: PreType) -> Result<(Vec<&'static str>, bool), TokenizerError> {
    const LLAMA3: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const GPT2: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)";
    const QWEN2: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const QWEN35: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const NEWLINES: &str = r"[^\n]+|[\n]+";
    const GPT4O: &str = r"[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))*((?=[\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const TEKKEN: &str = r"[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))*((?=[\p{L}])([^A-Z]))+|[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const DEEPSEEK3: &str = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";
    Ok(match pre {
        PreType::Llama3 | PreType::Chatglm4 => (vec![LLAMA3], true),
        PreType::Jais2 => (vec![r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s{512}(?!\S)|\s{256}(?!\S)|\s{128}(?!\S)|\s{64}(?!\S)|\s{32}(?!\S)|\s{16}(?!\S)|\s{8}(?!\S)|\s{4}(?!\S)|\s{1,2}(?!\S)|\s{1}"], true),
        PreType::DeepseekLlm => (
            vec![
                "[\r\n]",
                // llama.cpp's source has three range ends NFC-normalised (U+1F7D/1FD3/1FDB became
                // U+03CE/0390/038A, reversing the ranges); the intended endpoints are restored here.
                "\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-\u{1F7D}ᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-\u{1FD3}ῖ-\u{1FDB}ῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
                r"\s?[!-/:-~！-／：-～‘-‟　-。]+",
                r"\s+$",
                r"[一-龥ࠀ-一가-퟿]+",
                r"\p{N}+",
            ],
            true,
        ),
        PreType::Deepseek3Llm => (vec![r"\p{N}{1,3}", r"[一-龥぀-ゟ゠-ヿ]+", DEEPSEEK3], true),
        PreType::Spark25 => (
            vec![
                r"\p{N}{1,3}",
                r"[一-龥぀-ゟ゠-ヿ]+",
                "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+|[\r\n]|\\s+(?!\\S)|\\s+",
                r"\p{N}",
            ],
            true,
        ),
        PreType::Youtu => (
            vec![
                r"[가-힣ㄱ-ㆎ]+|[！…“”‘’—：；，、-〿︰-﹏]+|[ㄅ-ㄯ]+|[一-龥぀-ゟ゠-ヿ]+",
                r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            ],
            true,
        ),
        PreType::DeepseekCoder => (
            vec!["[\r\n]", r"\s?\p{L}+", r"\s?\p{P}+", r"[一-龥ࠀ-一가-퟿]+", r"\p{N}"],
            true,
        ),
        PreType::Falcon => (
            vec![r"[\p{P}\$\+<=>\^~\|`]+", GPT2, "[0-9][0-9][0-9]"],
            true,
        ),
        PreType::Starcoder => (vec![r"\p{N}", GPT2], true),
        PreType::Gpt2 => (vec![GPT2], true),
        PreType::Qwen2 => (vec![QWEN2], true),
        PreType::Qwen35 => (vec![QWEN35], true),
        PreType::Poro => (vec![r" ?[^(\s|.,!?…。，、।۔،)]+"], true),
        PreType::Viking => (vec![r" ?[^(\s|.,!?…。，、।۔،)]+", r"\p{N}"], true),
        PreType::Tekken => (vec![TEKKEN], true),
        PreType::Chameleon => (
            vec![
                "<sentinel:[0-9]+>",
                "(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z",
                "([\\t\\n]|    |  )",
                r"\p{N}",
                r"[\p{P}!-/:-@\[-`{-~]",
                GPT2,
            ],
            true,
        ),
        PreType::Gpt4o => (vec![GPT4O], true),
        PreType::GraniteEmbMulti => (vec![r"[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"], true),
        PreType::Bailingmoe => (vec![r"'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"], true),
        PreType::SeedCoder => (vec![r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1}| ?[^\s\p{L}\p{N}\r\n]+|\s*[\r\n]+|\s+(?!\S)|\s+"], true),
        PreType::Ufakzeka => (vec![r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"], true),
        PreType::Grok2 => (vec![QWEN2], true),
        PreType::Laguna => (vec![NEWLINES, QWEN2], true),
        PreType::ExaoneMoe => (vec![r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?(?:\p{L}\p{M}*(?: \p{L}\p{M}*)*)+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]?|\s*[\r\n]|\s+(?!\S)|\s+"], true),
        PreType::Gemma4 | PreType::SarvamMoe => (vec![NEWLINES], false),
        PreType::Minicpm5 => (vec![r"\p{N}{1,3}", r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"], true),
        PreType::Default => (
            vec![r"[\p{P}\$\+<=>\^~\|]+", GPT2, r"\p{N}+", "[0-9][0-9][0-9]"],
            true,
        ),
        PreType::Unsupported => {
            return Err(TokenizerError::UnsupportedPre(
                "pre-tokenizer needs a bespoke splitter this crate does not implement".into(),
            ))
        }
    })
}

/// Which representation of the text a regex step runs on (mirrors llama.cpp's choice).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Haystack {
    /// Category-collapsed text (pattern used `\p{..}`).
    Collapsed,
    /// Codepoints as-is except non-ASCII whitespace -> U+000B.
    Codepoint,
}

enum Step {
    Gpt2,
    Llama3,
    Qwen2,
    Qwen35,
    Newlines,
    Regex {
        re: fancy_regex::Regex,
        haystack: Haystack,
    },
}

/// A compiled pre-tokenizer: the ordered split steps of one pre-type.
pub struct PreTokenizer {
    steps: Vec<Step>,
    /// Chunks are GPT-2 byte-encoded (false for Gemma 4-style raw UTF-8 vocabularies).
    pub byte_encode: bool,
}

impl PreTokenizer {
    pub fn new(pre: PreType) -> Result<Self, TokenizerError> {
        let (exprs, byte_encode) = patterns(pre)?;
        let mut steps = Vec::with_capacity(exprs.len());
        for expr in exprs {
            steps.push(compile_step(expr)?);
        }
        Ok(PreTokenizer { steps, byte_encode })
    }

    /// Split `text` into chunks (as bytes, byte-encoded when [`Self::byte_encode`]).
    pub fn split(&self, text: &str) -> Vec<Vec<u8>> {
        let cpts: Vec<u32> = text.chars().map(|c| c as u32).collect();
        let need_collapsed = self.steps.iter().any(|s| {
            matches!(
                s,
                Step::Regex {
                    haystack: Haystack::Collapsed,
                    ..
                }
            )
        });
        let need_cpt = self.steps.iter().any(|s| {
            matches!(
                s,
                Step::Regex {
                    haystack: Haystack::Codepoint,
                    ..
                }
            )
        });
        let collapsed = need_collapsed.then(|| Collapsed::build(&cpts, Haystack::Collapsed));
        let codepoint = need_cpt.then(|| Collapsed::build(&cpts, Haystack::Codepoint));

        let mut offsets = vec![cpts.len()];
        for step in &self.steps {
            offsets = match step {
                Step::Gpt2 => split_gpt2(&cpts, &offsets),
                Step::Llama3 => split_llama3(&cpts, &offsets),
                Step::Qwen2 => split_qwen2(&cpts, &offsets, false),
                Step::Qwen35 => split_qwen2(&cpts, &offsets, true),
                Step::Newlines => split_newlines(&cpts, &offsets),
                Step::Regex { re, haystack } => {
                    let hay = match haystack {
                        Haystack::Collapsed => collapsed.as_ref(),
                        Haystack::Codepoint => codepoint.as_ref(),
                    };
                    // Built above whenever such a step exists.
                    match hay {
                        Some(h) => split_regex(re, h, &offsets),
                        None => offsets,
                    }
                }
            };
        }

        let mut words = Vec::with_capacity(offsets.len());
        let mut start = 0;
        let mut buf = String::new();
        for len in offsets {
            buf.clear();
            for &c in &cpts[start..start + len] {
                buf.push(char::from_u32(c).unwrap_or('\u{FFFD}'));
            }
            start += len;
            if self.byte_encode {
                words.push(byte_encode(buf.as_bytes()).into_bytes());
            } else {
                words.push(buf.as_bytes().to_vec());
            }
        }
        words
    }
}

fn compile_step(expr: &str) -> Result<Step, TokenizerError> {
    const GPT2: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)";
    const LLAMA3: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const QWEN2: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const QWEN35: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const NEWLINES: &str = r"[^\n]+|[\n]+";
    Ok(match expr {
        GPT2 => Step::Gpt2,
        LLAMA3 => Step::Llama3,
        QWEN2 => Step::Qwen2,
        QWEN35 => Step::Qwen35,
        NEWLINES => Step::Newlines,
        _ => {
            let haystack = if uses_category(expr) {
                Haystack::Collapsed
            } else {
                Haystack::Codepoint
            };
            let translated = translate(expr, haystack == Haystack::Collapsed)?;
            let re = fancy_regex::RegexBuilder::new(&translated)
                .backtrack_limit(50_000_000)
                .build()
                .map_err(|e| TokenizerError::Regex(format!("{expr}: {e}")))?;
            Step::Regex { re, haystack }
        }
    })
}

const CATEGORIES: &[(&str, char, &str)] = &[
    ("\\p{N}", '\u{D1}', "0-9"),
    ("\\p{L}", '\u{D2}', "A-Za-z"),
    ("\\p{P}", '\u{D3}', r"!-#%-*,-/:-;?-@\[-\]_\{\}"),
    ("\\p{M}", '\u{D4}', ""),
    ("\\p{S}", '\u{D5}', r"\$\+<=>\^`\|~"),
    ("\\p{Lu}", '\u{D2}', "A-Za-z"),
    ("\\p{Ll}", '\u{D2}', "A-Za-z"),
    ("\\p{Lt}", '\u{D2}', "A-Za-z"),
    ("\\p{Lm}", '\u{D2}', "A-Za-z"),
    ("\\p{Lo}", '\u{D2}', "A-Za-z"),
];

fn uses_category(expr: &str) -> bool {
    CATEGORIES.iter().any(|(name, _, _)| expr.contains(name))
}

/// Rewrite an ECMAScript pattern from llama.cpp into `fancy-regex` syntax with identical meaning
/// on the haystack it will run on: `\p{..}` becomes the collapsed marker plus ASCII members
/// (when `collapsed`), `\s`/`\S`/`\d` become the ASCII classes `std::regex` uses.
fn translate(expr: &str, collapsed: bool) -> Result<String, TokenizerError> {
    const SPACE: &str = r" \t\n\x0B\x0C\r";
    let b = expr.as_bytes();
    let mut out = String::with_capacity(expr.len() * 2);
    let mut inside = false;
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c == b'[' && (i == 0 || b[i - 1] != b'\\') {
            out.push('[');
            inside = true;
            i += 1;
            continue;
        }
        if inside && c == b']' && b[i - 1] != b'\\' {
            out.push(']');
            inside = false;
            i += 1;
            continue;
        }
        if c == b'\\' && i + 1 < b.len() {
            let n = b[i + 1];
            if n == b'p' && i + 2 < b.len() && b[i + 2] == b'{' {
                let close = expr[i..]
                    .find('}')
                    .map(|p| p + i)
                    .ok_or_else(|| TokenizerError::Regex(format!("unterminated \\p in {expr}")))?;
                let pat = &expr[i..=close];
                if collapsed {
                    if let Some((_, marker, ascii)) =
                        CATEGORIES.iter().find(|(name, _, _)| *name == pat)
                    {
                        if !inside {
                            out.push('[');
                        }
                        out.push_str(&format!("\\x{{{:X}}}", *marker as u32));
                        out.push_str(ascii);
                        if !inside {
                            out.push(']');
                        }
                        i = close + 1;
                        continue;
                    }
                }
                return Err(TokenizerError::Regex(format!(
                    "unsupported unicode class {pat} in {expr}"
                )));
            }
            match n {
                b's' => {
                    if inside {
                        out.push_str(SPACE);
                    } else {
                        out.push('[');
                        out.push_str(SPACE);
                        out.push(']');
                    }
                }
                b'S' => {
                    if inside {
                        return Err(TokenizerError::Regex(format!("\\S inside class in {expr}")));
                    }
                    out.push_str("[^");
                    out.push_str(SPACE);
                    out.push(']');
                }
                b'd' => out.push_str(if inside { "0-9" } else { "[0-9]" }),
                b'D' => {
                    if inside {
                        return Err(TokenizerError::Regex(format!("\\D inside class in {expr}")));
                    }
                    out.push_str("[^0-9]");
                }
                _ => {
                    out.push('\\');
                    out.push(n as char);
                }
            }
            i += 2;
            continue;
        }
        // Copy one (possibly multi-byte) character.
        let ch = expr[i..].chars().next().unwrap_or('\u{FFFD}');
        out.push(ch);
        i += ch.len_utf8();
    }
    Ok(out)
}

/// A haystack string for the regex steps plus the byte offset of every codepoint in it.
struct Collapsed {
    text: String,
    /// `starts[i]` is the byte offset of codepoint `i`; `starts[n]` is the length.
    starts: Vec<usize>,
}

impl Collapsed {
    fn build(cpts: &[u32], mode: Haystack) -> Collapsed {
        let mut text = String::with_capacity(cpts.len() * 2);
        let mut starts = Vec::with_capacity(cpts.len() + 1);
        for &cpt in cpts {
            starts.push(text.len());
            let ch = if cpt < 128 {
                cpt as u8 as char
            } else {
                let f = flags(cpt);
                match mode {
                    Haystack::Collapsed => {
                        if f.is_whitespace() {
                            '\u{0B}'
                        } else {
                            match f.category() {
                                unicode::NUMBER => '\u{D1}',
                                unicode::LETTER => '\u{D2}',
                                unicode::PUNCTUATION => '\u{D3}',
                                unicode::ACCENT_MARK => '\u{D4}',
                                unicode::SYMBOL => '\u{D5}',
                                _ => '\u{D0}',
                            }
                        }
                    }
                    Haystack::Codepoint => {
                        if f.is_whitespace() {
                            '\u{0B}'
                        } else {
                            char::from_u32(cpt).unwrap_or('\u{FFFD}')
                        }
                    }
                }
            };
            text.push(ch);
        }
        starts.push(text.len());
        Collapsed { text, starts }
    }

    fn index_of_byte(&self, byte: usize) -> usize {
        self.starts.partition_point(|&s| s < byte)
    }
}

/// llama.cpp's `unicode_regex_split_stl`: within each existing chunk, every match becomes a chunk
/// and the text between matches becomes chunks too.
fn split_regex(re: &fancy_regex::Regex, hay: &Collapsed, offsets: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(offsets.len());
    let mut start = 0;
    for &len in offsets {
        let end = start + len;
        let sub = &hay.text[hay.starts[start]..hay.starts[end]];
        let base = hay.starts[start];
        let mut cur = start;
        for m in re.find_iter(sub) {
            let m = match m {
                Ok(m) => m,
                Err(e) => {
                    tracing::warn!("pre-tokenizer regex gave up ({e}); rest of chunk kept whole");
                    break;
                }
            };
            if m.start() == m.end() {
                continue;
            }
            let ma = hay.index_of_byte(base + m.start());
            let mb = hay.index_of_byte(base + m.end());
            if ma > cur {
                out.push(ma - cur);
            }
            out.push(mb - ma);
            cur = mb;
        }
        if cur < end {
            out.push(end - cur);
        }
        start = end;
    }
    out
}

const OUT_OF_RANGE: u32 = u32::MAX;

/// One chunk of codepoints with llama.cpp's out-of-range semantics.
struct Win<'a> {
    cpts: &'a [u32],
    ini: usize,
    end: usize,
}

impl Win<'_> {
    fn cpt(&self, pos: usize) -> u32 {
        if self.ini <= pos && pos < self.end {
            self.cpts[pos]
        } else {
            OUT_OF_RANGE
        }
    }
    fn flags(&self, pos: usize) -> Flags {
        if self.ini <= pos && pos < self.end {
            flags(self.cpts[pos])
        } else {
            Flags(0)
        }
    }
}

/// Visit each chunk of `offsets` as a [`Win`] and collect the sub-chunk lengths `f` emits.
fn for_each_chunk(
    cpts: &[u32],
    offsets: &[usize],
    mut f: impl FnMut(&Win<'_>, &mut Vec<usize>),
) -> Vec<usize> {
    let mut out = Vec::with_capacity(offsets.len());
    let mut start = 0;
    for &len in offsets {
        let w = Win {
            cpts,
            ini: start,
            end: start + len,
        };
        start += len;
        f(&w, &mut out);
    }
    out
}

/// Tracks `_prev_end` / `_add_token` of the llama.cpp splitters.
struct Emit<'a> {
    prev_end: usize,
    out: &'a mut Vec<usize>,
}

impl Emit<'_> {
    fn add(&mut self, end: usize) {
        if end > self.prev_end {
            self.out.push(end - self.prev_end);
        }
        self.prev_end = end;
    }
}

/// The `'s|'t|'re|'ve|'m|'ll|'d` rule; returns the new position when it matched.
fn contraction(w: &Win<'_>, pos: usize, case_insensitive: bool) -> Option<usize> {
    if w.cpt(pos) != '\'' as u32 || pos + 1 >= w.end {
        return None;
    }
    let low = |c: u32| if case_insensitive { tolower(c) } else { c };
    let n1 = low(w.cpt(pos + 1));
    if n1 == 's' as u32 || n1 == 't' as u32 || n1 == 'm' as u32 || n1 == 'd' as u32 {
        return Some(pos + 2);
    }
    if pos + 2 < w.end {
        let n2 = low(w.cpt(pos + 2));
        if (n2 == 'e' as u32 && (n1 == 'r' as u32 || n1 == 'v' as u32))
            || (n1 == 'l' as u32 && n2 == 'l' as u32)
        {
            return Some(pos + 3);
        }
    }
    None
}

/// GPT-2: `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)`
fn split_gpt2(cpts: &[u32], offsets: &[usize]) -> Vec<usize> {
    for_each_chunk(cpts, offsets, |w, out| {
        let mut e = Emit {
            prev_end: w.ini,
            out,
        };
        let mut pos = w.ini;
        while pos < w.end {
            let cpt = w.cpt(pos);
            let fl = w.flags(pos);
            if let Some(p) = contraction(w, pos, false) {
                pos = p;
                e.add(pos);
                continue;
            }
            let space = cpt == ' ' as u32;
            let mut f2 = if space { w.flags(pos + 1) } else { fl };
            if f2.is_letter() {
                pos += usize::from(space);
                while f2.is_letter() {
                    pos += 1;
                    f2 = w.flags(pos);
                }
                e.add(pos);
                continue;
            }
            if f2.is_number() {
                pos += usize::from(space);
                while f2.is_number() {
                    pos += 1;
                    f2 = w.flags(pos);
                }
                e.add(pos);
                continue;
            }
            let other = |f: Flags| !(f.is_whitespace() | f.is_letter() | f.is_number()) && f.any();
            if other(f2) {
                pos += usize::from(space);
                while other(f2) {
                    pos += 1;
                    f2 = w.flags(pos);
                }
                e.add(pos);
                continue;
            }
            let mut nws = 0;
            while w.flags(pos + nws).is_whitespace() {
                nws += 1;
            }
            if nws > 1 && w.cpt(pos + nws) != OUT_OF_RANGE {
                pos += nws - 1;
                e.add(pos);
                continue;
            }
            if nws > 0 {
                pos += nws;
                e.add(pos);
                continue;
            }
            pos += 1;
            e.add(pos);
        }
    })
}

/// LLAMA3: `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
fn split_llama3(cpts: &[u32], offsets: &[usize]) -> Vec<usize> {
    for_each_chunk(cpts, offsets, |w, out| {
        let mut e = Emit {
            prev_end: w.ini,
            out,
        };
        let mut pos = w.ini;
        while pos < w.end {
            let cpt = w.cpt(pos);
            let fl = w.flags(pos);
            if let Some(p) = contraction(w, pos, true) {
                pos = p;
                e.add(pos);
                continue;
            }
            if !(cpt == '\r' as u32 || cpt == '\n' as u32 || fl.is_number())
                && (fl.is_letter() || w.flags(pos + 1).is_letter())
            {
                pos += 1;
                while w.flags(pos).is_letter() {
                    pos += 1;
                }
                e.add(pos);
                continue;
            }
            if fl.is_number() {
                let mut ini = pos;
                while w.flags(pos).is_number() {
                    pos += 1;
                    if pos - ini >= 3 {
                        e.add(pos);
                        ini = pos;
                    }
                }
                e.add(pos);
                continue;
            }
            if punct_run(w, &mut e, &mut pos, cpt, fl, false) {
                continue;
            }
            whitespace_tail(w, &mut e, &mut pos);
        }
    })
}

/// QWEN2 (and QWEN35 when `marks`): like LLAMA3 but single digits, and letter runs that also
/// take combining marks for Qwen 3.5.
fn split_qwen2(cpts: &[u32], offsets: &[usize], marks: bool) -> Vec<usize> {
    for_each_chunk(cpts, offsets, |w, out| {
        let mut e = Emit {
            prev_end: w.ini,
            out,
        };
        let mut pos = w.ini;
        while pos < w.end {
            let cpt = w.cpt(pos);
            let fl = w.flags(pos);
            if let Some(p) = contraction(w, pos, true) {
                pos = p;
                e.add(pos);
                continue;
            }
            let letterish = |f: Flags| f.is_letter() || (marks && f.is_accent_mark());
            if !(cpt == '\r' as u32 || cpt == '\n' as u32 || fl.is_number())
                && (letterish(fl) || letterish(w.flags(pos + 1)))
            {
                pos += 1;
                while letterish(w.flags(pos)) {
                    pos += 1;
                }
                e.add(pos);
                continue;
            }
            if fl.is_number() {
                pos += 1;
                e.add(pos);
                continue;
            }
            if punct_run(w, &mut e, &mut pos, cpt, fl, marks) {
                continue;
            }
            whitespace_tail(w, &mut e, &mut pos);
        }
    })
}

/// ` ?[^\s\p{L}\p{N}]+[\r\n]*` (with `\p{M}` excluded too when `marks`).
fn punct_run(
    w: &Win<'_>,
    e: &mut Emit<'_>,
    pos: &mut usize,
    cpt: u32,
    fl: Flags,
    marks: bool,
) -> bool {
    let other = |f: Flags| {
        !(f.is_whitespace() | f.is_letter() | f.is_number() | (marks && f.is_accent_mark()))
            && f.any()
    };
    let space = cpt == ' ' as u32;
    let mut f2 = if space { w.flags(*pos + 1) } else { fl };
    if other(f2) && fl.any() {
        *pos += usize::from(space);
        while other(f2) {
            *pos += 1;
            f2 = w.flags(*pos);
        }
        let mut c2 = w.cpt(*pos);
        while c2 == '\r' as u32 || c2 == '\n' as u32 {
            *pos += 1;
            c2 = w.cpt(*pos);
        }
        e.add(*pos);
        return true;
    }
    false
}

/// `\s*[\r\n]+|\s+(?!\S)|\s+` and the single-codepoint fallback.
fn whitespace_tail(w: &Win<'_>, e: &mut Emit<'_>, pos: &mut usize) {
    let mut nws = 0;
    let mut last_rn = 0;
    while w.flags(*pos + nws).is_whitespace() {
        let c2 = w.cpt(*pos + nws);
        if c2 == '\r' as u32 || c2 == '\n' as u32 {
            last_rn = *pos + nws + 1;
        }
        nws += 1;
    }
    if last_rn > 0 {
        *pos = last_rn;
        e.add(*pos);
        return;
    }
    if nws > 1 && w.cpt(*pos + nws) != OUT_OF_RANGE {
        *pos += nws - 1;
        e.add(*pos);
        return;
    }
    if nws > 0 {
        *pos += nws;
        e.add(*pos);
        return;
    }
    *pos += 1;
    e.add(*pos);
}

/// `[^\n]+|[\n]+`
fn split_newlines(cpts: &[u32], offsets: &[usize]) -> Vec<usize> {
    for_each_chunk(cpts, offsets, |w, out| {
        let mut pos = w.ini;
        while pos < w.end {
            let is_nl = w.cpt(pos) == '\n' as u32;
            let run = pos;
            while pos < w.end && (w.cpt(pos) == '\n' as u32) == is_nl {
                pos += 1;
            }
            out.push(pos - run);
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunks(pre: PreType, text: &str) -> Vec<String> {
        let p = PreTokenizer::new(pre).unwrap();
        let p = PreTokenizer {
            steps: p.steps,
            byte_encode: false,
        };
        p.split(text)
            .into_iter()
            .map(|w| String::from_utf8(w).unwrap())
            .collect()
    }

    #[test]
    fn gpt2_splitter() {
        assert_eq!(
            chunks(PreType::Gpt2, "Hello world's  test 123!!\n\n x"),
            ["Hello", " world", "'s", " ", " test", " 123", "!!", "\n\n", " x"]
        );
        assert_eq!(chunks(PreType::Gpt2, "a   "), ["a", "   "]);
        assert_eq!(chunks(PreType::Gpt2, "   a"), ["  ", " a"]);
    }

    #[test]
    fn llama3_splitter_groups_digits_by_three() {
        assert_eq!(
            chunks(PreType::Llama3, "x 1234567 It'S\n\n  y"),
            ["x", " ", "123", "456", "7", " It", "'S", "\n\n", " ", " y"]
        );
        assert_eq!(
            chunks(PreType::Llama3, "hi!!\n\nthere"),
            ["hi", "!!\n\n", "there"]
        );
    }

    #[test]
    fn qwen2_splits_every_digit() {
        assert_eq!(
            chunks(PreType::Qwen2, "a 1234 b"),
            ["a", " ", "1", "2", "3", "4", " b"]
        );
    }

    #[test]
    fn qwen35_keeps_marks_with_letters() {
        // e + COMBINING ACUTE ACCENT stays one chunk for Qwen 3.5 but splits for Qwen 2.
        assert_eq!(chunks(PreType::Qwen35, "e\u{301}x"), ["e\u{301}x"]);
        // Qwen 2 lets the mark act as the optional non-letter prefix of "x".
        assert_eq!(chunks(PreType::Qwen2, "e\u{301}x"), ["e", "\u{301}x"]);
        assert_eq!(chunks(PreType::Qwen2, "e\u{301} x"), ["e", "\u{301}", " x"]);
    }

    #[test]
    fn newline_splitter() {
        assert_eq!(
            chunks(PreType::Gemma4, "a b\n\nc\n"),
            ["a b", "\n\n", "c", "\n"]
        );
    }

    #[test]
    fn gpt4o_regex_path() {
        assert_eq!(
            chunks(PreType::Gpt4o, "Hello WORLD it's 12345 don't ünïcode  \n x"),
            [
                "Hello",
                " WORLD",
                " it's",
                " ",
                "123",
                "45",
                " don't",
                " ünïcode",
                "  \n",
                " x"
            ]
        );
    }

    #[test]
    fn smollm_number_prepass_then_gpt2() {
        assert_eq!(chunks(PreType::Starcoder, "ab 12"), ["ab", " ", "1", "2"]);
    }

    #[test]
    fn default_prepass_punctuation() {
        assert_eq!(
            chunks(PreType::Default, "a+b 1234"),
            ["a", "+", "b", " ", "123", "4"]
        );
    }

    #[test]
    fn translate_rewrites_classes() {
        assert_eq!(
            translate(r"\s+(?!\S)", true).unwrap(),
            r"[ \t\n\x0B\x0C\r]+(?![^ \t\n\x0B\x0C\r])"
        );
        assert_eq!(
            translate(r"[^\s\p{L}]", true).unwrap(),
            r"[^ \t\n\x0B\x0C\r\x{D2}A-Za-z]"
        );
        assert_eq!(translate(r"\p{N}+", true).unwrap(), r"[\x{D1}0-9]+");
        assert_eq!(translate(r"\d{1,3}", false).unwrap(), r"[0-9]{1,3}");
    }

    #[test]
    fn unknown_and_unsupported_pre() {
        assert!(pre_config("no-such-pre").is_none());
        assert_eq!(
            pre_config("kimi-k2").unwrap().pre_type,
            PreType::Unsupported
        );
        assert!(PreTokenizer::new(PreType::Unsupported).is_err());
        assert!(pre_config("llama-bpe").unwrap().add_bos);
        assert!(pre_config("llama-bpe").unwrap().ignore_merges);
        assert!(!pre_config("dbrx").unwrap().ignore_merges);
    }

    #[test]
    fn every_supported_pre_type_compiles() {
        for name in [
            "default",
            "llama3",
            "deepseek-llm",
            "deepseek-coder",
            "deepseek-v3",
            "spark2_5",
            "youtu",
            "falcon",
            "mpt",
            "starcoder",
            "gpt-2",
            "jais-2",
            "gemma4",
            "sarvam-moe",
            "refact",
            "command-r",
            "qwen2",
            "qwen35",
            "stablelm2",
            "olmo",
            "dbrx",
            "smaug-bpe",
            "poro-chat",
            "glm4",
            "viking",
            "jais",
            "tekken",
            "smollm",
            "codeshell",
            "bloom",
            "gpt3-finnish",
            "exaone",
            "exaone4",
            "exaone-moe",
            "chameleon",
            "minerva-7b",
            "megrez",
            "gpt-4o",
            "granite-embed-multi-97m",
            "bailingmoe",
            "seed-coder",
            "hunyuan",
            "hunyuan-dense",
            "hy_v4",
            "joyai-llm",
            "ufakzeka",
            "grok-2",
            "laguna",
            "minimax-m2",
            "solar-open",
            "mellum2",
            "minicpm5",
            "trillion",
            "granite-docling",
        ] {
            let cfg = pre_config(name).unwrap_or_else(|| panic!("{name}"));
            PreTokenizer::new(cfg.pre_type).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
    }
}
