//! Unit tests on synthetic vocabularies written as real GGUF files.

use std::collections::HashMap;

use llmario_engine_formats::gguf::writer::GgufWriter;
use llmario_engine_formats::{GgufFile, MetaValue};

use crate::unicode::byte_encode;
use crate::{Detokenizer, TokenType, Tokenizer, VocabKind};

fn write(w: &GgufWriter) -> (tempfile::TempDir, GgufFile) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vocab.gguf");
    std::fs::write(&path, w.to_bytes()).unwrap();
    let g = GgufFile::open(&path).unwrap();
    (dir, g)
}

fn strs(v: &[&str]) -> MetaValue {
    MetaValue::Array(v.iter().map(|s| MetaValue::Str(s.to_string())).collect())
}

/// SentencePiece vocabulary: unk/bos/eos, 256 byte tokens, a handful of scored pieces and one
/// user-defined token.
fn spm_vocab() -> (Vec<String>, Vec<f32>, Vec<i32>) {
    let mut tokens = vec!["<unk>".to_string(), "<s>".to_string(), "</s>".to_string()];
    let mut scores = vec![0.0f32, 0.0, 0.0];
    let mut types = vec![2i32, 3, 3];
    for b in 0..=255u8 {
        tokens.push(format!("<0x{b:02X}>"));
        scores.push(0.0);
        types.push(6);
    }
    let pieces: &[(&str, f32)] = &[
        ("\u{2581}", -1.0),
        ("\u{2581}a", -2.0),
        ("b", -3.0),
        ("\u{2581}ab", -1.5),
        ("ab", -4.0),
        ("a", -5.0),
        ("\u{2581}h", -9.5),
        ("\u{2581}he", -9.0),
        ("\u{2581}hel", -8.0),
        ("\u{2581}hello", -5.0),
        ("lo", -8.0),
        ("llo", -10.0),
        ("h", -20.0),
        ("e", -20.0),
        ("l", -20.0),
        ("o", -20.0),
    ];
    for (p, s) in pieces {
        tokens.push(p.to_string());
        scores.push(*s);
        types.push(1);
    }
    tokens.push("<|im_start|>".to_string());
    scores.push(0.0);
    types.push(4);
    (tokens, scores, types)
}

fn spm_tokenizer() -> (tempfile::TempDir, Tokenizer) {
    let (tokens, scores, types) = spm_vocab();
    let mut w = GgufWriter::new();
    w.meta("general.architecture", MetaValue::Str("llama".into()))
        .meta("tokenizer.ggml.model", MetaValue::Str("llama".into()))
        .meta(
            "tokenizer.ggml.tokens",
            MetaValue::Array(tokens.into_iter().map(MetaValue::Str).collect()),
        )
        .meta(
            "tokenizer.ggml.scores",
            MetaValue::Array(scores.into_iter().map(MetaValue::F32).collect()),
        )
        .meta(
            "tokenizer.ggml.token_type",
            MetaValue::Array(types.into_iter().map(MetaValue::I32).collect()),
        )
        .meta("tokenizer.ggml.bos_token_id", MetaValue::U32(1))
        .meta("tokenizer.ggml.eos_token_id", MetaValue::U32(2))
        .meta("tokenizer.ggml.unknown_token_id", MetaValue::U32(0))
        .meta("tokenizer.ggml.add_bos_token", MetaValue::Bool(true))
        .meta(
            "tokenizer.chat_template",
            MetaValue::Str("{{ messages }}".into()),
        );
    let (dir, g) = write(&w);
    let t = Tokenizer::from_gguf(&g).unwrap();
    (dir, t)
}

fn id(t: &Tokenizer, s: &str) -> u32 {
    t.token_to_id(s.as_bytes())
        .unwrap_or_else(|| panic!("no token {s:?}"))
}

#[test]
fn spm_merges_by_score_and_falls_back_to_bytes() {
    let (_d, t) = spm_tokenizer();
    assert_eq!(t.kind(), VocabKind::Spm);
    assert_eq!(t.bos(), Some(1));
    assert_eq!(t.eos(), Some(2));
    assert!(t.add_bos());
    assert!(t.add_space_prefix());
    assert_eq!(t.chat_template(), Some("{{ messages }}"));
    assert_eq!(t.encode("", true, true), [1]);
    assert_eq!(t.encode("", false, true), []);
    assert_eq!(t.encode("ab", true, true), [1, id(&t, "\u{2581}ab")]);
    assert_eq!(
        t.encode("ab ab", false, true),
        [id(&t, "\u{2581}ab"), id(&t, "\u{2581}ab")]
    );
    assert_eq!(t.encode("hello", false, true), [id(&t, "\u{2581}hello")]);
    // 'x' has no piece: the byte token.
    assert_eq!(
        t.encode("x", false, true),
        [id(&t, "\u{2581}"), id(&t, "<0x78>")]
    );
    // A 4-byte character becomes four byte tokens.
    let emoji = t.encode("\u{1F600}", false, true);
    assert_eq!(emoji.len(), 5);
    assert_eq!(t.token_type(emoji[1]), TokenType::Byte);
    assert_eq!(t.decode(&emoji), "\u{1F600}");
    // Like llama.cpp, the leading-space strip applies to the first rendered piece, so a rendered
    // BOS keeps the space that follows it.
    assert_eq!(
        t.decode_with_special(&t.encode("\u{1F600}", true, true)),
        "<s> \u{1F600}"
    );
    assert_eq!(t.decode(&t.encode("\u{1F600}", true, true)), "\u{1F600}");
}

#[test]
fn spm_user_defined_tokens_split_even_without_parse_special() {
    let (_d, t) = spm_tokenizer();
    let im = id(&t, "<|im_start|>");
    let a = id(&t, "\u{2581}a");
    let sp = id(&t, "\u{2581}");
    let b = id(&t, "b");
    // After a special token the next fragment gets a space prefix again.
    assert_eq!(t.encode("a<|im_start|>b", true, true), [1, a, im, sp, b]);
    assert_eq!(t.encode("a<|im_start|>b", true, false), [1, a, im, sp, b]);
    // Control tokens only split when parse_special is on.
    let with = t.encode("a</s>b", false, true);
    assert_eq!(with, [a, 2, sp, b]);
    let without = t.encode("a</s>b", false, false);
    assert!(!without.contains(&2));
    assert!(t.is_eog(2));
    assert!(!t.is_eog(im));
    assert_eq!(t.decode_with_special(&with), "a</s> b");
    assert_eq!(t.decode(&with), "a b");
}

#[test]
fn spm_detokenizer_holds_back_partial_utf8() {
    let (_d, t) = spm_tokenizer();
    let ids = t.encode("a \u{1F600} b", true, true);
    let mut d = Detokenizer::new(&t);
    let mut parts = Vec::new();
    for &i in &ids {
        parts.push(d.push(i));
    }
    assert!(d.pending().is_empty());
    assert_eq!(parts.concat(), t.decode(&ids));
    assert_eq!(parts.concat(), "a \u{1F600} b");
    // The emoji only appears once its last byte arrives.
    let emoji_at = parts.iter().position(|p| p.contains('\u{1F600}')).unwrap();
    assert!(parts[emoji_at - 1].is_empty() && parts[emoji_at - 2].is_empty());
    // A truncated sequence is flushed lossily.
    let mut d = Detokenizer::new(&t);
    assert_eq!(d.push(id(&t, "<0xF0>")), "");
    assert_eq!(d.pending(), [0xF0]);
    assert_eq!(d.flush(), "\u{FFFD}");
}

/// Byte-level BPE vocabulary: 256 byte tokens, merge-built pieces, control tokens.
fn bpe_tokenizer(pre: &str, add_bos: Option<bool>) -> (tempfile::TempDir, Tokenizer) {
    let mut tokens: Vec<String> = (0..=255u8).map(|b| byte_encode(&[b])).collect();
    let mut types = vec![1i32; 256];
    let pieces = [
        "he", "ll", "hell", "hello", "Ġw", "or", "ld", "Ġwor", "Ġworld", "ĊĊ", "12", "123",
        "Ġhello",
    ];
    for p in pieces {
        tokens.push(p.to_string());
        types.push(1);
    }
    for c in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"] {
        tokens.push(c.to_string());
        types.push(3);
    }
    tokens.push("<tool_call>".to_string());
    types.push(4);
    let merges = [
        "h e", "l l", "he ll", "hell o", "Ġ w", "o r", "l d", "Ġw or", "Ġwor ld", "Ċ Ċ", "1 2",
        "12 3", "Ġ hello",
    ];
    let mut w = GgufWriter::new();
    w.meta("general.architecture", MetaValue::Str("qwen2".into()))
        .meta("tokenizer.ggml.model", MetaValue::Str("gpt2".into()))
        .meta("tokenizer.ggml.pre", MetaValue::Str(pre.into()))
        .meta(
            "tokenizer.ggml.tokens",
            MetaValue::Array(tokens.into_iter().map(MetaValue::Str).collect()),
        )
        .meta(
            "tokenizer.ggml.token_type",
            MetaValue::Array(types.into_iter().map(MetaValue::I32).collect()),
        )
        .meta("tokenizer.ggml.merges", strs(&merges))
        .meta("tokenizer.ggml.eos_token_id", MetaValue::U32(256 + 13 + 1))
        .meta(
            "tokenizer.ggml.padding_token_id",
            MetaValue::U32(256 + 13 + 2),
        );
    if let Some(b) = add_bos {
        w.meta("tokenizer.ggml.add_bos_token", MetaValue::Bool(b));
    }
    let (dir, g) = write(&w);
    let t = Tokenizer::from_gguf(&g).unwrap();
    (dir, t)
}

#[test]
fn bpe_merges_in_rank_order_per_chunk() {
    let (_d, t) = bpe_tokenizer("qwen2", Some(false));
    assert_eq!(t.kind(), VocabKind::Bpe);
    assert_eq!(t.pre_name(), "qwen2");
    assert!(!t.add_bos());
    assert_eq!(t.eos(), Some(id(&t, "<|im_end|>")));
    assert_eq!(t.pad(), Some(id(&t, "<|endoftext|>")));
    assert_eq!(
        t.encode("hello world", true, true),
        [id(&t, "hello"), id(&t, "Ġworld")]
    );
    // qwen2 splits digits one by one, so "123" never merges.
    assert_eq!(
        t.encode("123", false, true),
        [id(&t, "1"), id(&t, "2"), id(&t, "3")]
    );
    // Chunks never merge across the pre-tokenizer boundary ("hello" + " world").
    assert_eq!(
        t.encode("hello\n\n world", false, true),
        [id(&t, "hello"), id(&t, "ĊĊ"), id(&t, "Ġworld")]
    );
    assert_eq!(
        t.decode(&t.encode("hello\n\nworld", false, true)),
        "hello\n\nworld"
    );
    let (_d, t) = bpe_tokenizer("gpt-2", None);
    assert_eq!(t.encode("123", false, true), [id(&t, "123")]);
    assert_eq!(t.encode(" 123", false, true), [id(&t, "Ġ"), id(&t, "123")]);
}

#[test]
fn bpe_special_tokens_and_eog() {
    let (_d, t) = bpe_tokenizer("qwen2", Some(false));
    let start = id(&t, "<|im_start|>");
    let end = id(&t, "<|im_end|>");
    let eot = id(&t, "<|endoftext|>");
    let tool = id(&t, "<tool_call>");
    let ids = t.encode("<|im_start|>hello<|im_end|>", false, true);
    assert_eq!(ids, [start, id(&t, "hello"), end]);
    assert_eq!(t.decode(&ids), "hello");
    assert_eq!(t.decode_with_special(&ids), "<|im_start|>hello<|im_end|>");
    // Without parse_special the control token text is tokenized as ordinary bytes.
    let plain = t.encode("<|im_start|>hello", false, false);
    assert!(!plain.contains(&start));
    assert_eq!(t.decode(&plain), "<|im_start|>hello");
    // User-defined tokens split regardless.
    assert!(t.encode("a<tool_call>b", false, false).contains(&tool));
    assert!(t.is_eog(end) && t.is_eog(eot) && !t.is_eog(start) && !t.is_eog(tool));
    assert_eq!(t.eot(), Some(end));
    assert!(t.is_control(start) && !t.is_control(tool));
    assert_eq!(t.token_to_piece(start), b"<|im_start|>");
    assert_eq!(t.token_to_piece(id(&t, "Ġworld")), b" world");
    assert_eq!(t.token_type(start), TokenType::Control);
    assert_eq!(t.token_type(tool), TokenType::UserDefined);
    assert_eq!(t.byte_to_token(b'h'), Some(b'h' as u32));
}

#[test]
fn bpe_round_trips_arbitrary_text_and_streams_utf8() {
    let (_d, t) = bpe_tokenizer("qwen2", Some(false));
    let text = "héllo wörld \u{1F600}\u{1F468}\u{200D}\u{1F469} 中文 مرحبا\t\n  end";
    let ids = t.encode(text, true, true);
    assert_eq!(t.decode(&ids), text);
    let mut d = Detokenizer::new(&t);
    let streamed: String = ids.iter().map(|&i| d.push(i)).collect();
    assert_eq!(streamed, text);
    assert!(d.pending().is_empty());
}

#[test]
fn bpe_add_bos_defaults_follow_pre_type() {
    let (_d, t) = bpe_tokenizer("llama-bpe", None);
    assert!(t.add_bos(), "llama-bpe forces add_bos");
    assert_eq!(t.encode("", true, true), [t.bos().unwrap()]);
    let (_d, t) = bpe_tokenizer("llama-bpe", Some(false));
    assert!(!t.add_bos(), "the GGUF key wins");
    let (_d, t) = bpe_tokenizer("gpt-2", None);
    assert!(!t.add_bos());
}

#[test]
fn unknown_pre_falls_back_to_default_and_bad_model_errors() {
    let (_d, t) = bpe_tokenizer("made-up-pre-2027", Some(false));
    assert_eq!(t.pre_type(), crate::PreType::Default);
    assert_eq!(
        t.encode("hello world", false, true),
        [id(&t, "hello"), id(&t, "Ġworld")]
    );

    let mut w = GgufWriter::new();
    w.meta("tokenizer.ggml.model", MetaValue::Str("rwkv".into()))
        .meta("tokenizer.ggml.tokens", strs(&["a"]));
    let (_d, g) = write(&w);
    assert!(matches!(
        Tokenizer::from_gguf(&g),
        Err(crate::TokenizerError::UnsupportedModel(_))
    ));
    let mut w = GgufWriter::new();
    w.meta("tokenizer.ggml.tokens", strs(&["a"]));
    let (_d, g) = write(&w);
    assert!(matches!(
        Tokenizer::from_gguf(&g),
        Err(crate::TokenizerError::MissingKey(_))
    ));
}

#[test]
fn special_token_partition_prefers_longest_and_scans_left_to_right() {
    let (_d, t) = bpe_tokenizer("qwen2", Some(false));
    let start = id(&t, "<|im_start|>");
    let end = id(&t, "<|im_end|>");
    let ids = t.encode("<|im_end|><|im_start|><|im_start|>", false, true);
    assert_eq!(ids, [end, start, start]);
    assert_eq!(
        t.encode("<|im_start|", false, true)
            .iter()
            .filter(|&&i| i == start)
            .count(),
        0
    );
}

#[test]
fn encode_is_fast_enough_on_large_input() {
    let (_d, t) = bpe_tokenizer("qwen2", Some(false));
    let unit = "hello world, 12345 héllo\n\n  <|im_start|>user\nwörld 中文 \u{1F600} ok. ";
    let text = unit.repeat(100 * 1024 / unit.len() + 1);
    assert!(text.len() >= 100 * 1024);
    let started = std::time::Instant::now();
    let ids = t.encode(&text, true, true);
    let elapsed = started.elapsed();
    assert!(!ids.is_empty());
    // Debug builds are several times slower than release; the release bound is well under 1 s.
    assert!(
        elapsed.as_secs_f64() < 5.0,
        "encode of {} bytes took {elapsed:?}",
        text.len()
    );
    assert_eq!(t.decode_with_special(&ids), text);
    let _: HashMap<u32, u32> = HashMap::new();
}
