//! Token-for-token parity with llama.cpp's `llama-tokenize`, on real GGUF files.
//!
//! Runs only when `LLMARIO_TEST_GGUF` lists GGUF paths (comma-separated). For every path and every
//! corpus string it compares `Tokenizer::encode(text, add_special = true, parse_special = true)`
//! with the ids printed by
//!
//! ```text
//! llama-tokenize -m <gguf> -f <tmpfile> --no-escape --ids --log-disable
//! ```
//!
//! `-f` + `--no-escape` feeds the exact bytes (`-p` would process `\n` escapes and the shell would
//! mangle quotes); `--ids` prints a Python-style list; `--log-disable` keeps stderr quiet.
//! `llama-tokenize` passes `add_special = <model's add_bos_token>` and `parse_special = true`
//! (its defaults), which matches `encode(.., true, true)` for every model whose `add_eos_token`
//! is false (all models in the catalog); the test asserts that precondition.
//!
//! It also checks `decode(encode(text)) == text` for strings without special tokens and that the
//! incremental `Detokenizer` yields the same text as `decode`, and prints the time to encode a
//! 100 KB text. The binary path can be overridden with `LLMARIO_LLAMA_TOKENIZE`.
//!
//! Command used for the catalog models (see the crate README):
//!
//! ```text
//! LLMARIO_TEST_GGUF="/Volumes/Extreme Pro/AIProjects/LLMario-models/qwen3-1.7b-gguf-q4km/Qwen3-1.7B-Q4_K_M.gguf,\
//! /Volumes/Extreme Pro/AIProjects/LLMario-beta-models/qwen3.5-0.8b-gguf-q4_0/Qwen3.5-0.8B-Q4_0.gguf,\
//! /Volumes/Extreme Pro/AIProjects/LLMario-beta-models/gemma-4-12b-gguf-q4_0/gemma-4-12b-it-qat-q4_0.gguf,\
//! /Volumes/Extreme Pro/AIProjects/LLMario-beta-models/gpt-oss-20b-gguf-mxfp4/gpt-oss-20b-MXFP4.gguf,\
//! /Volumes/Extreme Pro/AIProjects/LoopForgeProject/platform/.models/smollm3-3b-q4_k_m.gguf,\
//! /Volumes/Extreme Pro/AIProjects/LoopForgeProject/platform/.models/olmo-3-7b-instruct-q4_k_m.gguf,\
//! /Volumes/Extreme Pro/AIProjects/LoopForgeProject/platform/.models/smollm2-1.7b-instruct-q4_k_m.gguf" \
//!   cargo test -p llmario-engine-tokenizer --test llama_cpp_parity -- --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use llmario_engine_formats::GgufFile;
use llmario_engine_tokenizer::{Detokenizer, Tokenizer, VocabKind};

/// Strings with special tokens written literally are marked so the round-trip check skips them.
const CORPUS: &[(&str, bool)] = &[
    ("", false),
    ("Hello world", false),
    ("Hello, world! How are you today?", false),
    ("The quick brown fox jumps over the lazy dog.", false),
    ("I'm sure they'd've said it's fine, isn't it? You'll see; we're not going.", false),
    ("IT'S ALL CAPS AND it'S MiXeD CaSe", false),
    ("fn main() {\n    println!(\"hi\");\n\tlet x = 1;\n}\n", false),
    ("def f(x):\n    if x:\n        return [i for i in range(10)]\n    else:\n\t\treturn None\n", false),
    ("    indented line\n\n\n    another\n", false),
    ("a\tb\t\tc\r\nd\re", false),
    ("1234567890", false),
    ("12345678901234567890123456789", false),
    ("Call 555-123-4567 at 10:30 on 2026-10-09, cost $1,234.56 (approx. 1e10).", false),
    ("x1 y22 z333 w4444 v55555", false),
    ("!!! ??? ... ,,, ;;; ::: --- *** ### @@@ &&& ||| ^^^ ~~~ ``` ''' \"\"\"", false),
    ("()[]{}<>/\\|-_=+*&^%$#@!~`", false),
    ("Hello 😀 world 🌍🚀 and 👨‍👩‍👧‍👦 family 🇺🇸 flags", false),
    ("𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝕞𝕒𝕥𝕙 𐍈 𠜎𠜱𠝹", false),
    ("你好，世界！这是一个测试。", false),
    ("日本語のテキストです。漢字とひらがなとカタカナ。", false),
    ("한국어 문장입니다. 안녕하세요!", false),
    ("مرحبا بالعالم، كيف حالك؟", false),
    ("שלום עולם, מה שלומך?", false),
    ("Привет, мир! Как дела?", false),
    ("Ελληνικά κείμενο με τόνους: άέήίόύώ", false),
    ("हिन्दी में कुछ पाठ। धन्यवाद!", false),
    ("ไทย ภาษาไทย ทดสอบ", false),
    ("Mixed: English 中文 العربية русский 日本語 😀 123 abc", false),
    ("naïve café résumé Ünïcödé Straße", false),
    ("e\u{301}le\u{300}ve (combining marks) ñ ã õ", false),
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
    ("<|endoftext|>", true),
    ("text<|endoftext|>more text<|im_end|>", true),
    ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>", true),
    ("<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n", true),
    ("<|start|>user<|message|>hi<|end|><|start|>assistant", true),
    ("<|im_start|", false),
    ("<tool_call>\n{\"name\": \"f\"}\n</tool_call>", true),
    ("Pneumonoultramicroscopicsilicovolcanoconiosis", false),
    ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", false),
    ("ThisIsAVeryLongCamelCaseIdentifierThatGoesOnAndOnForeverAndEverWithoutAnyBreaks_and_snake_case_too", false),
    ("http://example.com/path/to/resource?query=string&other=1#fragment", false),
    ("user@example.com, C:\\Users\\name\\file.txt, /usr/local/bin", false),
    ("JSON: {\"key\": [1, 2, {\"nested\": null}], \"ok\": true}", false),
    ("Ab1 cD2 Ef3 — em dash – en dash … ellipsis « quotes » „low“", false),
    ("  \u{1F600}  ", false),
    ("\u{0000}null byte inside", false),
    ("\u{0085}\u{2028}\u{2029} odd line separators", false),
];

/// Special ids llama.cpp reports at load (`print_info: BOS token = 1 '<s>'` lines of `-v`).
#[derive(Default, Debug)]
struct RefSpecial {
    bos: Option<u32>,
    eos: Option<u32>,
    eot: Option<u32>,
    eom: Option<u32>,
    pad: Option<u32>,
    eog: Vec<u32>,
}

fn reference_special(bin: &Path, model: &Path) -> RefSpecial {
    let out = Command::new(bin)
        .arg("-m")
        .arg(model)
        .arg("-p")
        .arg("x")
        .arg("-v")
        .output()
        .expect("run llama-tokenize -v");
    let text =
        String::from_utf8_lossy(&out.stderr).into_owned() + &String::from_utf8_lossy(&out.stdout);
    let mut r = RefSpecial::default();
    for line in text.lines() {
        let Some(rest) = line.split("print_info: ").nth(1) else {
            continue;
        };
        let Some((name, value)) = rest.split_once(" token") else {
            continue;
        };
        let id: Option<u32> = value
            .split('=')
            .nth(1)
            .and_then(|v| v.trim().split(' ').next())
            .and_then(|v| v.parse().ok());
        let Some(id) = id else {
            continue;
        };
        match name.trim() {
            "BOS" => r.bos = Some(id),
            "EOS" => r.eos = Some(id),
            "EOT" => r.eot = Some(id),
            "EOM" => r.eom = Some(id),
            "PAD" => r.pad = Some(id),
            "EOG" => r.eog.push(id),
            _ => {}
        }
    }
    r.eog.sort_unstable();
    r
}

fn llama_tokenize() -> PathBuf {
    std::env::var_os("LLMARIO_LLAMA_TOKENIZE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/opt/homebrew/bin/llama-tokenize"))
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
fn matches_llama_tokenize_on_catalog_models() {
    let Some(list) = std::env::var_os("LLMARIO_TEST_GGUF") else {
        eprintln!("LLMARIO_TEST_GGUF not set; skipping parity test");
        return;
    };
    let bin = llama_tokenize();
    assert!(bin.exists(), "{} not found", bin.display());
    let dir = tempfile::tempdir().unwrap();
    let mut failures = Vec::new();
    for model in list
        .to_string_lossy()
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let path = Path::new(model);
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let started = Instant::now();
        let g = GgufFile::open(path).unwrap();
        let t = Tokenizer::from_gguf(&g).unwrap();
        let load = started.elapsed();
        assert!(
            !t.add_eos(),
            "{model}: add_eos_token models need --no-bos handling"
        );
        let special = reference_special(&bin, path);
        let mut eog: Vec<u32> = t.eog_ids().collect();
        eog.sort_unstable();
        for (what, want, got) in [
            ("BOS", special.bos, t.bos()),
            ("EOS", special.eos, t.eos()),
            ("PAD", special.pad, t.pad()),
        ] {
            if want != got {
                failures.push(format!("{name}: {what} llama.cpp={want:?} ours={got:?}"));
            }
        }
        if special.eog != eog {
            failures.push(format!(
                "{name}: EOG set llama.cpp={:?} ours={eog:?}",
                special.eog
            ));
        }
        if special.eot != t.eot() || special.eom != t.eom() {
            eprintln!(
                "{name}: note: EOT/EOM differ (llama.cpp picks by hash order): llama.cpp eot={:?} eom={:?} ours eot={:?} eom={:?}",
                special.eot, special.eom, t.eot(), t.eom()
            );
        }
        let mut ok = 0;
        let mut bad = 0;
        for &(text, has_special) in CORPUS {
            let want = reference_ids(&bin, path, text, dir.path());
            assert!(
                !want.is_empty() || text.is_empty(),
                "{name}: empty reference for {text:?}"
            );
            let got = t.encode(text, true, true);
            if want == got {
                ok += 1;
            } else {
                bad += 1;
                failures.push(format!(
                    "{name}: {text:?}\n   llama.cpp: {want:?}\n   ours:      {got:?}"
                ));
            }
            if !has_special {
                let plain = t.encode(text, false, true);
                let dec = t.decode(&plain);
                if dec != text {
                    bad += 1;
                    failures.push(format!("{name}: round trip of {text:?} gave {dec:?}"));
                }
                let mut d = Detokenizer::new(&t);
                let streamed: String =
                    plain.iter().map(|&i| d.push(i)).collect::<String>() + &d.flush();
                if streamed != dec {
                    bad += 1;
                    failures.push(format!(
                        "{name}: detokenizer gave {streamed:?} for {text:?}"
                    ));
                }
            }
        }
        // Timing on a 100 KB text built from the corpus.
        let mut big = String::new();
        for (s, _) in CORPUS.iter().cycle() {
            if big.len() >= 100 * 1024 {
                break;
            }
            big.push_str(s);
            big.push(' ');
        }
        let started = Instant::now();
        let ids = t.encode(&big, true, true);
        let enc = started.elapsed();
        eprintln!(
            "{name}: model={} pre={} n_vocab={} load={load:?} corpus ok={ok} bad={bad} | encode {} bytes -> {} tokens in {enc:?}",
            match t.kind() {
                VocabKind::Spm => "llama",
                VocabKind::Bpe => "gpt2/gemma4",
            },
            t.pre_name(),
            t.n_vocab(),
            big.len(),
            ids.len()
        );
        assert!(
            enc.as_secs_f64() < 2.0,
            "{model}: encoding 100 KB took {enc:?}"
        );
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
