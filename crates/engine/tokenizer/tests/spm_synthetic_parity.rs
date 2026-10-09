//! SentencePiece parity with llama.cpp on a synthetic vocabulary.
//!
//! No SentencePiece (`tokenizer.ggml.model = "llama"`) GGUF is in the local catalog, but
//! `llama-tokenize` loads a vocab-only GGUF, so this test writes one: three control tokens, the
//! 256 `<0xXX>` byte tokens, every prefix and every short substring of a word list as scored
//! pieces (scores quantized so that ties are common and the queue's tie-break is exercised),
//! and two user-defined tokens. It then compares `encode(text, true, true)` with
//!
//! ```text
//! llama-tokenize -m <synthetic.gguf> -f <tmpfile> --no-escape --ids --log-disable
//! ```
//!
//! over the same corpus as `llama_cpp_parity.rs`. Runs only when `LLMARIO_TEST_GGUF` is set
//! (any value); the binary path can be overridden with `LLMARIO_LLAMA_TOKENIZE`:
//!
//! ```text
//! LLMARIO_TEST_GGUF=1 cargo test -p llmario-engine-tokenizer --test spm_synthetic_parity -- --nocapture
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use llmario_engine_formats::gguf::writer::GgufWriter;
use llmario_engine_formats::{GgufFile, MetaValue};
use llmario_engine_tokenizer::{Detokenizer, Tokenizer, VocabKind};

const WORDS: &str =
    "the quick brown fox jumps over lazy dog hello world how are you today sure they \
said fine isn't going all caps mixed case main println let return range else none indented line \
another call cost approx many spaces here leading trailing double inside tabs null byte odd line \
separators user assistant text more start of turn model name family flags quotes dash ellipsis \
mixed english 中文 русский 日本語 test identifier snake camel path resource query string fragment \
example users file local bin json key nested true ok null hello hello hello aaaa aaaaaaaa";

const CORPUS: &[(&str, bool)] = &[
    ("", false),
    ("Hello world", false),
    ("hello world", false),
    ("The quick brown fox jumps over the lazy dog.", false),
    ("I'm sure they'd've said it's fine, isn't it? You'll see; we're not going.", false),
    ("fn main() {\n    println!(\"hi\");\n\tlet x = 1;\n}\n", false),
    ("    indented line\n\n\n    another\n", false),
    ("1234567890 and 12345678901234567890123456789", false),
    ("!!! ??? ... ,,, ;;; ::: --- *** ### @@@ &&& ||| ^^^ ~~~ ``` ''' \"\"\"", false),
    ("Hello 😀 world 🌍🚀 and 👨‍👩‍👧‍👦 family", false),
    ("你好，世界！这是一个测试。", false),
    ("مرحبا بالعالم، كيف حالك؟", false),
    ("Mixed: English 中文 العربية русский 日本語 😀 123 abc", false),
    ("naïve café résumé Ünïcödé Straße", false),
    (" leading space", false),
    ("trailing space ", false),
    ("  two leading", false),
    ("double  space  inside", false),
    ("many     spaces     here", false),
    ("\n", false),
    ("\n\n\n", false),
    ("line one\nline two\n\nline four\n", false),
    (" \n \n  \n", false),
    ("\t\t\ttabs\t\t\t", false),
    ("\u{00A0}nbsp\u{00A0}and\u{2003}em\u{3000}ideographic", false),
    ("<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n", true),
    ("hello</s>world", true),
    ("<s>hello", true),
    ("<|im_start|", false),
    ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", false),
    ("ThisIsAVeryLongCamelCaseIdentifierThatGoesOnAndOn_and_snake_case_too", false),
    ("http://example.com/path/to/resource?query=string&other=1#fragment", false),
    ("JSON: {\"key\": [1, 2, {\"nested\": null}], \"ok\": true}", false),
    ("  \u{1F600}  ", false),
    ("\u{0000}null byte inside", false),
    ("hellohellohello hello hello", false),
];

fn score_of(piece: &str) -> f32 {
    // FNV-1a, quantized to half-steps so many pieces share a score.
    let mut h: u64 = 0xcbf29ce484222325;
    for b in piece.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    -1.0 - (h % 20) as f32 * 0.5
}

fn build_vocab() -> (Vec<String>, Vec<f32>, Vec<i32>) {
    let mut tokens = vec!["<unk>".to_string(), "<s>".to_string(), "</s>".to_string()];
    let mut scores = vec![0.0f32; 3];
    let mut types = vec![2i32, 3, 3];
    for b in 0..=255u8 {
        tokens.push(format!("<0x{b:02X}>"));
        scores.push(0.0);
        types.push(6);
    }
    let mut pieces: BTreeMap<String, f32> = BTreeMap::new();
    let mut add = |p: &str| {
        if !p.is_empty() && !p.contains(' ') && !pieces.contains_key(p) {
            pieces.insert(p.to_string(), score_of(p));
        }
    };
    add("\u{2581}");
    for word in WORDS.split_whitespace() {
        let esc = format!("\u{2581}{word}");
        let chars: Vec<char> = esc.chars().collect();
        for i in 0..chars.len() {
            for len in 1..=4.min(chars.len() - i) {
                add(&chars[i..i + len].iter().collect::<String>());
            }
            add(&chars[..=i].iter().collect::<String>());
        }
        let lower: String = word.to_lowercase();
        add(&format!("\u{2581}{lower}"));
    }
    for c in "0123456789.,;:!?-_'\"()[]{}<>/\\|=+*&^%$#@~`\u{2581}".chars() {
        add(&c.to_string());
    }
    for (p, s) in pieces {
        tokens.push(p);
        scores.push(s);
        types.push(1);
    }
    for t in ["<|im_start|>", "<|im_end|>"] {
        tokens.push(t.to_string());
        scores.push(0.0);
        types.push(4);
    }
    (tokens, scores, types)
}

fn write_gguf(dir: &Path) -> PathBuf {
    let (tokens, scores, types) = build_vocab();
    let mut w = GgufWriter::new();
    w.meta("general.architecture", MetaValue::Str("llama".into()))
        .meta("general.name", MetaValue::Str("synthetic spm".into()))
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
        .meta("tokenizer.ggml.add_bos_token", MetaValue::Bool(true));
    let path = dir.join("synthetic-spm.gguf");
    std::fs::write(&path, w.to_bytes()).unwrap();
    path
}

fn reference_ids(bin: &Path, model: &Path, text: &str, dir: &Path) -> Vec<u32> {
    let prompt = dir.join("prompt.txt");
    std::fs::write(&prompt, text).unwrap();
    let out = Command::new(bin)
        .arg("-m")
        .arg(model)
        .arg("-f")
        .arg(&prompt)
        .arg("--no-escape")
        .arg("--ids")
        .arg("--log-disable")
        .output()
        .expect("run llama-tokenize");
    assert!(
        out.status.success(),
        "llama-tokenize failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .rev()
        .find(|l| l.starts_with('['))
        .unwrap_or_else(|| panic!("no id list in output: {stdout}"));
    line.trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect()
}

#[test]
fn synthetic_spm_matches_llama_tokenize() {
    if std::env::var_os("LLMARIO_TEST_GGUF").is_none() {
        eprintln!("LLMARIO_TEST_GGUF not set; skipping SPM parity test");
        return;
    }
    let bin = std::env::var_os("LLMARIO_LLAMA_TOKENIZE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/opt/homebrew/bin/llama-tokenize"));
    assert!(bin.exists(), "{} not found", bin.display());
    let dir = tempfile::tempdir().unwrap();
    let path = write_gguf(dir.path());
    let g = GgufFile::open(&path).unwrap();
    let t = Tokenizer::from_gguf(&g).unwrap();
    assert_eq!(t.kind(), VocabKind::Spm);
    assert!(t.add_bos() && t.add_space_prefix());
    let mut ok = 0;
    let mut failures = Vec::new();
    for &(text, has_special) in CORPUS {
        let want = reference_ids(&bin, &path, text, dir.path());
        let got = t.encode(text, true, true);
        if want == got {
            ok += 1;
        } else {
            failures.push(format!(
                "{text:?}\n   llama.cpp: {want:?}\n   ours:      {got:?}"
            ));
        }
        if !has_special {
            let plain = t.encode(text, false, true);
            let dec = t.decode(&plain);
            if dec != text {
                failures.push(format!("round trip of {text:?} gave {dec:?}"));
            }
            let mut d = Detokenizer::new(&t);
            let streamed: String =
                plain.iter().map(|&i| d.push(i)).collect::<String>() + &d.flush();
            if streamed != dec {
                failures.push(format!("detokenizer gave {streamed:?} for {text:?}"));
            }
        }
    }
    eprintln!(
        "synthetic SPM: n_vocab={} corpus ok={ok} bad={}",
        t.n_vocab(),
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
