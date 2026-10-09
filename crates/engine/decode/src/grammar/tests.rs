//! Constrained-decoding tests on a synthetic byte-level vocabulary (no model weights), plus one
//! test on the real Qwen3 tokenizer when `LLMARIO_TEST_GGUF` is set.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use serde_json::{json, Value};

use super::*;
use crate::rng::Xoshiro256StarStar;
use crate::sampler::{LogitProcessor, Sampler};
use crate::SamplingParams;

/// Synthetic vocabulary: every byte, a few multi-byte text tokens, the control tokens of the
/// tool-call formats, and three end-of-generation tokens.
struct Vocab {
    env: Arc<GrammarTokenizer>,
    ids: HashMap<Vec<u8>, Token>,
    /// Logit offset per token: multi-byte, non-blank tokens sit in a higher range so closing
    /// tokens are picked quickly once allowed, which keeps random walks short.
    bias: Vec<f32>,
}

const NORMAL: &[&str] = &[
    "{\"",
    "\":",
    "\", \"",
    "\"}",
    "name",
    "arguments",
    "true",
    "false",
    "null",
    "<tool_call>",
    "</tool_call>",
    "<tool_call>\n",
    "</think>",
    "</think>{",
    "<think>",
    "<function=",
    "<parameter=",
    "</parameter>",
    "</function>",
    "call:",
    "commentary to=functions.",
    "get_weather",
    "search",
    "Paris",
    "  ",
    "location",
    "\n</parameter>",
];

const CONTROL: &[&str] = &[
    "<|python_tag|>",
    "[TOOL_CALLS]",
    "[ARGS]",
    "<|channel|>",
    "<|constrain|>",
    "<|message|>",
    "<|call|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|\"|>",
    "<|eom_id|>",
    "<|im_end|>",
];

fn vocab() -> Vocab {
    let mut pieces: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
    let mut special = vec![false; 256];
    for s in NORMAL {
        pieces.push(s.as_bytes().to_vec());
        special.push(false);
    }
    for s in CONTROL {
        pieces.push(s.as_bytes().to_vec());
        special.push(true);
    }
    let ids: HashMap<Vec<u8>, Token> = pieces
        .iter()
        .enumerate()
        .map(|(i, p)| (p.clone(), i as Token))
        .collect();
    let eos = ids[b"<|im_end|>".as_slice()];
    let eog = [
        eos,
        ids[b"<|eom_id|>".as_slice()],
        ids[b"<|call|>".as_slice()],
    ];
    let bias = pieces
        .iter()
        .map(|p| {
            if p.len() > 1 && !p.iter().all(u8::is_ascii_whitespace) {
                4.0
            } else {
                0.0
            }
        })
        .collect();
    let env = GrammarTokenizer::from_pieces(pieces, &special, eos, &eog).unwrap();
    Vocab {
        env: Arc::new(env),
        ids,
        bias,
    }
}

impl Vocab {
    fn id(&self, s: &str) -> Token {
        self.ids[s.as_bytes()]
    }
    fn n(&self) -> usize {
        self.env.n_vocab()
    }
    /// Rendered text of the tokens (control tokens as their text), end-of-generation excluded.
    fn render(&self, toks: &[Token]) -> String {
        let mut out = Vec::new();
        for &t in toks {
            if self.env.eog().contains(&t) {
                continue;
            }
            out.extend_from_slice(self.env.piece(t));
        }
        String::from_utf8_lossy(&out).into_owned()
    }
    /// Tokens whose logits survive `process` on finite logits.
    fn allowed(&self, p: &mut GrammarProcessor) -> Vec<Token> {
        let mut logits = vec![0.0f32; self.n()];
        p.process(&mut logits);
        logits
            .iter()
            .enumerate()
            .filter(|(_, l)| l.is_finite())
            .map(|(i, _)| i as Token)
            .collect()
    }
}

/// Pseudo-random logits plus the vocabulary's bias (see [`Vocab::bias`]).
fn random_logits(rng: &mut Xoshiro256StarStar, v: &Vocab) -> Vec<f32> {
    v.bias
        .iter()
        .map(|b| b + rng.next_f64() as f32 * 4.0)
        .collect()
}

/// `base` rotated by a step-dependent offset: fresh-looking logits without regenerating them.
fn rotated(base: &[f32], step: usize) -> Vec<f32> {
    let n = base.len();
    let off = (step * 7919 + 13) % n;
    let mut l = Vec::with_capacity(n);
    l.extend_from_slice(&base[off..]);
    l.extend_from_slice(&base[..off]);
    l
}

fn argmax(l: &[f32]) -> Token {
    let mut best = 0;
    for (i, &v) in l.iter().enumerate() {
        if v > l[best] {
            best = i;
        }
    }
    best as Token
}

/// Greedy decode under the processor from seeded random logits until an end-of-generation
/// token or `max_steps`. Returns the tokens (EOG included when produced).
fn decode(v: &Vocab, p: &mut GrammarProcessor, seed: u64, max_steps: usize) -> Vec<Token> {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
    let mut toks = Vec::new();
    for _ in 0..max_steps {
        let mut logits = random_logits(&mut rng, v);
        p.process(&mut logits);
        let t = argmax(&logits);
        assert!(logits[t as usize].is_finite(), "everything masked");
        p.accept(t);
        toks.push(t);
        if v.env.eog().contains(&t) {
            break;
        }
    }
    toks
}

fn weather_tools() -> Vec<ToolDef> {
    vec![
        ToolDef::new(
            "get_weather",
            json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string", "maxLength": 6},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    "days": {"type": "integer", "minimum": 1, "maximum": 7}
                },
                "required": ["location"],
                "additionalProperties": false
            }),
        ),
        ToolDef::new(
            "search",
            json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "maxLength": 5},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                    "safe": {"type": "boolean"}
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        ),
    ]
}

fn tool_spec(family: ToolFamily, choice: ToolChoice) -> GrammarSpec {
    GrammarSpec::ToolCalls(ToolCallSpec {
        family,
        tools: weather_tools(),
        choice,
    })
}

fn person_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "name": {"type": "string", "maxLength": 8},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "tags": {"type": "array", "items": {"type": "string", "maxLength": 4}, "maxItems": 3},
            "active": {"type": "boolean"},
            "kind": {"type": "string", "enum": ["a", "b", "c"]}
        },
        "required": ["name", "age", "active"],
        "additionalProperties": false
    })
}

fn check_person(v: &Value) {
    let o = v.as_object().expect("object");
    assert!(o["name"].as_str().is_some_and(|s| s.chars().count() <= 8));
    let age = o["age"].as_i64().expect("age integer");
    assert!((0..=150).contains(&age), "age {age}");
    assert!(o["active"].is_boolean());
    if let Some(tags) = o.get("tags") {
        let a = tags.as_array().expect("tags array");
        assert!(a.len() <= 3);
        assert!(a
            .iter()
            .all(|t| t.as_str().is_some_and(|s| s.chars().count() <= 4)));
    }
    if let Some(k) = o.get("kind") {
        assert!(["a", "b", "c"].contains(&k.as_str().unwrap()));
    }
    for k in o.keys() {
        assert!(["name", "age", "tags", "active", "kind"].contains(&k.as_str()));
    }
}

fn check_weather_args(name: &str, args: &Value) {
    let o = args.as_object().expect("arguments object");
    match name {
        "get_weather" => {
            assert!(o["location"].is_string());
            if let Some(u) = o.get("unit") {
                assert!(["celsius", "fahrenheit"].contains(&u.as_str().unwrap()));
            }
            if let Some(d) = o.get("days") {
                assert!((1..=7).contains(&d.as_i64().unwrap()));
            }
        }
        "search" => {
            assert!(o["query"].is_string());
            if let Some(l) = o.get("limit") {
                assert!((1..=20).contains(&l.as_i64().unwrap()));
            }
        }
        other => panic!("unknown tool {other}"),
    }
}

#[test]
fn env_marks_control_tokens_special_and_hashes_content() {
    let v = vocab();
    assert_eq!(v.n(), 256 + NORMAL.len() + CONTROL.len());
    assert!(v.env.is_special(v.id("<|python_tag|>")));
    assert!(!v.env.is_special(v.id("<tool_call>")));
    assert_eq!(
        v.env.special_token_id("<|python_tag|>"),
        Some(v.id("<|python_tag|>"))
    );
    assert_eq!(v.env.special_token_id("<tool_call>"), None);
    assert_eq!(v.env.piece(v.id("<|python_tag|>")), b"<|python_tag|>");
    assert_eq!(v.env.piece(v.id("Paris")), b"Paris");
    assert_eq!(v.env.eos(), v.id("<|im_end|>"));
    assert_eq!(v.env.eog().len(), 3);
    // Same content, same hash; a different special flag changes it.
    let again = vocab();
    assert_eq!(again.env.hash(), v.env.hash());
    let pieces: Vec<Vec<u8>> = (0..v.n() as u32).map(|i| v.env.piece(i).to_vec()).collect();
    let mut special: Vec<bool> = (0..v.n() as u32).map(|i| v.env.is_special(i)).collect();
    special[v.id("Paris") as usize] = true;
    let changed =
        GrammarTokenizer::from_pieces(pieces, &special, v.env.eos(), v.env.eog()).unwrap();
    assert_ne!(changed.hash(), v.env.hash());
    // Errors instead of panics on bad input.
    assert!(GrammarTokenizer::from_pieces(vec![], &[], 0, &[]).is_err());
    assert!(GrammarTokenizer::from_pieces(vec![vec![b'a']], &[false], 7, &[]).is_err());
}

#[test]
fn json_schema_greedy_fuzz_is_always_valid() {
    let v = vocab();
    let factory = new_parser_factory(&v.env).unwrap();
    let spec = GrammarSpec::JsonSchema(person_schema());
    let compiled = CompiledGrammar::compile(&spec, &v.env, &factory).unwrap();
    let mut longest = 0;
    for seed in 0..200u64 {
        let mut p = GrammarProcessor::new(compiled.clone(), None);
        assert_eq!(p.state(), GrammarState::Active);
        let toks = decode(&v, &mut p, seed, 400);
        assert!(
            v.env.eog().contains(toks.last().unwrap()),
            "seed {seed}: no EOS within budget; state {:?} text {:?}",
            p.state(),
            v.render(&toks)
        );
        assert_eq!(p.state(), GrammarState::Complete, "seed {seed}");
        let text = v.render(&toks);
        let value: Value = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("seed {seed}: invalid JSON {text:?}: {e}"));
        check_person(&value);
        longest = longest.max(toks.len());
        assert!(p.stats().masks >= 2);
    }
    eprintln!("json fuzz: longest output {longest} tokens");
}

#[test]
fn json_schema_through_sampler_stochastic_path() {
    // The sampler's multinomial draw must never land on a masked token either.
    let v = vocab();
    let spec = GrammarSpec::JsonSchema(person_schema());
    for seed in 0..20u64 {
        let p = GrammarProcessor::compile(&spec, &v.env, None).unwrap();
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: Some(seed),
            ..Default::default()
        };
        let mut s = Sampler::new(params, v.n()).with_processor(Box::new(p));
        // Prompt tokens go through accept_prompt and must not disturb the grammar.
        s.accept_prompt(v.id("Paris"));
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        let mut toks = Vec::new();
        for _ in 0..600 {
            let logits = random_logits(&mut rng, &v);
            let t = s.sample(&logits);
            s.accept(t);
            toks.push(t);
            if v.env.eog().contains(&t) {
                break;
            }
        }
        assert!(v.env.eog().contains(toks.last().unwrap()), "seed {seed}");
        let text = v.render(&toks);
        let value: Value = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("seed {seed}: invalid JSON {text:?}: {e}"));
        check_person(&value);
    }
}

#[test]
fn lazy_trigger_unconstrained_until_think_end() {
    let v = vocab();
    let spec = GrammarSpec::JsonSchema(person_schema());
    let mut p =
        GrammarProcessor::compile(&spec, &v.env, Some(LazyTrigger::new("</think>"))).unwrap();
    assert_eq!(p.state(), GrammarState::Watching);
    let all: Vec<Token> = (0..v.n() as Token).collect();
    assert_eq!(v.allowed(&mut p), all, "unconstrained before the trigger");
    for t in [v.id("<think>"), v.id("Paris"), b' ' as Token, v.id("name")] {
        p.accept(t);
        assert_eq!(p.state(), GrammarState::Watching);
    }
    assert_eq!(v.allowed(&mut p), all);
    assert_eq!(p.stats().steps_watching, 2);
    // Trigger as one token.
    p.accept(v.id("</think>"));
    assert_eq!(p.state(), GrammarState::Active);
    let allowed = v.allowed(&mut p);
    assert!(allowed.contains(&(b'{' as Token)));
    assert!(!allowed.contains(&(b'a' as Token)));
    assert!(!allowed.contains(&v.id("Paris")));
    assert!(allowed.len() < all.len());
    let toks = decode(&v, &mut p, 1, 400);
    let value: Value = serde_json::from_str(&v.render(&toks)).unwrap();
    check_person(&value);

    // Trigger split across single-byte tokens.
    let mut p =
        GrammarProcessor::compile(&spec, &v.env, Some(LazyTrigger::new("</think>"))).unwrap();
    for b in b"xx</thin" {
        p.accept(*b as Token);
        assert_eq!(p.state(), GrammarState::Watching);
    }
    p.accept(b'k' as Token);
    assert_eq!(p.state(), GrammarState::Watching);
    p.accept(b'>' as Token);
    assert_eq!(p.state(), GrammarState::Active);

    // Trigger ending mid-token: the leftover `{` is fed to the parser.
    let mut p =
        GrammarProcessor::compile(&spec, &v.env, Some(LazyTrigger::new("</think>"))).unwrap();
    p.accept(v.id("</think>{"));
    assert_eq!(p.state(), GrammarState::Active, "{:?}", p.error());
    let allowed = v.allowed(&mut p);
    assert!(
        allowed.contains(&(b'"' as Token)),
        "after `{{` a key must start"
    );
    assert!(!allowed.contains(&(b'{' as Token)));
    let toks = decode(&v, &mut p, 2, 400);
    let text = format!("{{{}", v.render(&toks));
    let value: Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text:?}: {e}"));
    check_person(&value);
}

#[test]
fn tool_choice_required_forces_the_opener_first() {
    let v = vocab();
    let mut p = GrammarProcessor::compile(
        &tool_spec(ToolFamily::Hermes, ToolChoice::Required),
        &v.env,
        None,
    )
    .unwrap();
    assert_eq!(p.state(), GrammarState::Active);
    let allowed = v.allowed(&mut p);
    assert!(!allowed.is_empty());
    for &t in &allowed {
        let piece = v.env.piece(t);
        let ok = b"<tool_call>".starts_with(piece) || piece.starts_with(b"<tool_call>");
        assert!(
            ok,
            "token {t} {:?} allowed as first bytes",
            String::from_utf8_lossy(piece)
        );
    }
    assert!(allowed.contains(&v.id("<tool_call>")));
    assert!(allowed.contains(&(b'<' as Token)));
    assert!(!allowed.contains(&(b'a' as Token)));
    assert!(
        !allowed.contains(&v.env.eos()),
        "a call is required: no EOS first"
    );
    for seed in 0..30u64 {
        let mut p = GrammarProcessor::compile(
            &tool_spec(ToolFamily::Hermes, ToolChoice::Required),
            &v.env,
            None,
        )
        .unwrap();
        let toks = decode(&v, &mut p, seed, 400);
        let text = v.render(&toks);
        assert!(text.starts_with("<tool_call>"), "seed {seed}: {text:?}");
        assert_eq!(p.state(), GrammarState::Complete, "seed {seed}: {text:?}");
        let calls = hermes_calls(&text);
        assert!(!calls.is_empty(), "seed {seed}: {text:?}");
        for c in calls {
            check_weather_args(c["name"].as_str().unwrap(), &c["arguments"]);
        }
    }
}

/// Parses every `<tool_call>…</tool_call>` segment of a Hermes output.
fn hermes_calls(text: &str) -> Vec<Value> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<tool_call>") {
        let after = &rest[start + "<tool_call>".len()..];
        let end = after.find("</tool_call>").expect("closing tag");
        let body = after[..end].trim();
        out.push(serde_json::from_str(body).unwrap_or_else(|e| panic!("{body:?}: {e}")));
        rest = &after[end + "</tool_call>".len()..];
    }
    assert!(rest.trim().is_empty(), "trailing text {rest:?}");
    out
}

#[test]
fn complete_grammar_forces_eos() {
    let v = vocab();
    let mut p = GrammarProcessor::compile(&GrammarSpec::Regex("abc".into()), &v.env, None).unwrap();
    assert_eq!(v.allowed(&mut p), vec![b'a' as Token]);
    p.accept(b'a' as Token);
    p.accept(b'b' as Token);
    assert_eq!(v.allowed(&mut p), vec![b'c' as Token]);
    assert_eq!(p.state(), GrammarState::Active);
    p.accept(b'c' as Token);
    assert_eq!(p.state(), GrammarState::Complete);
    let mut eog: Vec<Token> = v.env.eog().to_vec();
    eog.sort_unstable();
    let mut allowed = v.allowed(&mut p);
    allowed.sort_unstable();
    assert_eq!(
        allowed, eog,
        "only end-of-generation tokens after the grammar completes"
    );
    // Logit values of the EOG tokens are preserved, everything else is -inf.
    let mut logits: Vec<f32> = (0..v.n()).map(|i| i as f32).collect();
    p.process(&mut logits);
    for (i, l) in logits.iter().enumerate() {
        if eog.contains(&(i as Token)) {
            assert_eq!(*l, i as f32);
        } else {
            assert_eq!(*l, f32::NEG_INFINITY);
        }
    }
    p.accept(v.env.eos());
    assert_eq!(p.state(), GrammarState::Complete);
}

#[test]
fn special_tokens_are_not_matchable_by_grammar_text() {
    let v = vocab();
    let tag = v.id("<|python_tag|>");
    // Text literal: only the byte spelling is allowed, never the control token.
    let mut p = GrammarProcessor::compile(
        &GrammarSpec::Lark("start: \"<|python_tag|>\"".into()),
        &v.env,
        None,
    )
    .unwrap();
    let allowed = v.allowed(&mut p);
    assert_eq!(allowed, vec![b'<' as Token]);
    assert!(!allowed.contains(&tag));
    // Token reference: exactly the control token.
    let mut p = GrammarProcessor::compile(
        &GrammarSpec::Lark(format!("start: <[{tag}]>")),
        &v.env,
        None,
    )
    .unwrap();
    assert_eq!(v.allowed(&mut p), vec![tag]);
    p.accept(tag);
    assert_eq!(p.state(), GrammarState::Complete);
}

#[test]
fn parser_errors_fall_back_to_unconstrained() {
    let v = vocab();
    let mut p = GrammarProcessor::compile(&GrammarSpec::Regex("abc".into()), &v.env, None).unwrap();
    // A token outside the mask (the engine would never sample it) must not panic.
    p.accept(b'z' as Token);
    assert_eq!(p.state(), GrammarState::Disabled);
    assert!(p.error().is_some());
    let all: Vec<Token> = (0..v.n() as Token).collect();
    assert_eq!(v.allowed(&mut p), all);
    p.accept(b'q' as Token);
    assert_eq!(p.state(), GrammarState::Disabled);
    // Out-of-range token ids are tolerated too.
    let mut p = GrammarProcessor::compile(&GrammarSpec::Regex("abc".into()), &v.env, None).unwrap();
    p.accept(999_999);
    assert_eq!(p.state(), GrammarState::Disabled);
    // Compile errors are reported, not panicked.
    assert!(matches!(
        GrammarProcessor::compile(
            &GrammarSpec::Lark("start: undefined_rule".into()),
            &v.env,
            None
        ),
        Err(GrammarError::Compile(_))
    ));
    assert!(matches!(
        GrammarProcessor::compile(&GrammarSpec::Regex(String::new()), &v.env, None),
        Err(GrammarError::InvalidSpec(_))
    ));
    assert!(matches!(
        GrammarProcessor::compile(
            &GrammarSpec::JsonSchema(json!({"type": "object", "if": {}})),
            &v.env,
            None
        ),
        Err(GrammarError::Compile(_))
    ));
}

fn family_check(family: ToolFamily, text: &str, pinned: Option<&str>) {
    let names: Vec<&str> = if let Some(n) = pinned {
        vec![n]
    } else {
        vec!["get_weather", "search"]
    };
    let name_in = |hay: &str| names.iter().any(|n| hay.contains(n));
    match family {
        ToolFamily::Hermes => {
            let calls = hermes_calls(text);
            assert!(!calls.is_empty());
            for c in calls {
                let n = c["name"].as_str().unwrap();
                assert!(names.contains(&n), "{n}");
                check_weather_args(n, &c["arguments"]);
            }
        }
        ToolFamily::Llama3 => {
            let body = text
                .strip_prefix("<|python_tag|>")
                .expect("python_tag first");
            let c: Value =
                serde_json::from_str(body.trim()).unwrap_or_else(|e| panic!("{body:?}: {e}"));
            let n = c["name"].as_str().unwrap();
            assert!(names.contains(&n));
            check_weather_args(n, &c["parameters"]);
        }
        ToolFamily::Mistral => {
            let mut rest = text;
            let mut count = 0;
            while let Some(s) = rest.strip_prefix("[TOOL_CALLS]") {
                let (n, after) = s.split_once("[ARGS]").expect("[ARGS]");
                assert!(names.contains(&n), "{n}");
                let end = after.find("[TOOL_CALLS]").unwrap_or(after.len());
                let args: Value = serde_json::from_str(after[..end].trim())
                    .unwrap_or_else(|e| panic!("{after:?}: {e}"));
                check_weather_args(n, &args);
                rest = after[end..].trim_start();
                count += 1;
            }
            assert!(count >= 1 && rest.is_empty(), "{text:?}");
        }
        ToolFamily::Harmony => {
            let s = text
                .strip_prefix("<|channel|>commentary to=functions.")
                .expect("harmony header");
            let (head, body) = s.split_once("<|message|>").expect("<|message|>");
            let n = head.strip_suffix(" <|constrain|>json").unwrap_or(head);
            assert!(names.contains(&n), "{n}");
            assert!(
                !body.contains("<|call|>"),
                "<|call|> is end-of-generation: {text:?}"
            );
            let args: Value =
                serde_json::from_str(body).unwrap_or_else(|e| panic!("{body:?}: {e}"));
            check_weather_args(n, &args);
        }
        ToolFamily::QwenXml => {
            assert!(text.starts_with("<tool_call>"), "{text:?}");
            assert!(text.trim_end().ends_with("</tool_call>"), "{text:?}");
            assert!(name_in(text));
            let func = text.split("<function=").nth(1).unwrap();
            let n = func.split('>').next().unwrap();
            assert!(names.contains(&n));
            let required = if n == "get_weather" {
                "location"
            } else {
                "query"
            };
            assert!(
                text.contains(&format!("<parameter={required}>")),
                "{text:?}"
            );
            assert!(text.contains("</parameter>"), "{text:?}");
            assert!(text.contains("</function>"), "{text:?}");
            if text.contains("<parameter=days>") {
                let d = text.split("<parameter=days>").nth(1).unwrap();
                let d = d.split("</parameter>").next().unwrap().trim();
                let d: i64 = d.parse().unwrap_or_else(|_| panic!("days {d:?}"));
                assert!((1..=7).contains(&d));
            }
        }
        ToolFamily::Gemma4 => {
            assert!(text.starts_with("<|tool_call>call:"), "{text:?}");
            assert!(text.trim_end().ends_with("<tool_call|>"), "{text:?}");
            let n = text["<|tool_call>call:".len()..].split('{').next().unwrap();
            assert!(names.contains(&n), "{n}");
            let required = if n == "get_weather" {
                "location"
            } else {
                "query"
            };
            // Gemma 4 keys are bare: `call:name{k:<|"|>v<|"|>}`.
            assert!(text.contains(&format!("{required}:<|\"|>")), "{text:?}");
            if text.contains(",unit:") {
                assert!(
                    text.contains("unit:<|\"|>celsius<|\"|>")
                        || text.contains("unit:<|\"|>fahrenheit<|\"|>")
                );
            }
            if text.contains(",safe:") {
                assert!(text.contains("safe:true") || text.contains("safe:false"));
            }
            if text.contains(",days:") {
                let d = text.split(",days:").nth(1).unwrap();
                let d: String = d
                    .chars()
                    .take_while(|c| c.is_ascii_digit() || *c == '-')
                    .collect();
                let d: i64 = d
                    .parse()
                    .unwrap_or_else(|_| panic!("days {d:?} in {text:?}"));
                assert!((1..=7).contains(&d));
            }
        }
    }
}

const FAMILIES: [ToolFamily; 6] = [
    ToolFamily::Hermes,
    ToolFamily::QwenXml,
    ToolFamily::Gemma4,
    ToolFamily::Llama3,
    ToolFamily::Mistral,
    ToolFamily::Harmony,
];

fn opener(family: ToolFamily) -> &'static str {
    match family {
        ToolFamily::Hermes | ToolFamily::QwenXml => "<tool_call>",
        ToolFamily::Gemma4 => "<|tool_call>",
        ToolFamily::Llama3 => "<|python_tag|>",
        ToolFamily::Mistral => "[TOOL_CALLS]",
        ToolFamily::Harmony => "<|channel|>",
    }
}

#[test]
fn every_family_required_and_named() {
    let v = vocab();
    let factory = new_parser_factory(&v.env).unwrap();
    for family in FAMILIES {
        let required =
            CompiledGrammar::compile(&tool_spec(family, ToolChoice::Required), &v.env, &factory)
                .unwrap_or_else(|e| panic!("{family:?}: {e}\n"));
        assert!(required.triggers().is_empty());
        let mut seen = std::collections::HashSet::new();
        for seed in 0..25u64 {
            let mut p = GrammarProcessor::new(required.clone(), None);
            let first = v.allowed(&mut p);
            let op = opener(family).as_bytes();
            for &t in &first {
                let piece = v.env.piece(t);
                assert!(
                    op.starts_with(piece) || piece.starts_with(op),
                    "{family:?}: first token {t} {:?}",
                    String::from_utf8_lossy(piece)
                );
            }
            let toks = decode(&v, &mut p, seed, 500);
            let text = v.render(&toks);
            assert_eq!(
                p.state(),
                GrammarState::Complete,
                "{family:?} seed {seed}: {text:?}"
            );
            family_check(family, &text, None);
            if text.contains("search") {
                seen.insert("search");
            }
            if text.contains("get_weather") {
                seen.insert("get_weather");
            }
            if seed == 0 {
                eprintln!("{family:?} required: {text:?}");
            }
        }
        assert_eq!(seen.len(), 2, "{family:?}: both tools reachable");

        let named = CompiledGrammar::compile(
            &tool_spec(family, ToolChoice::Named("search".into())),
            &v.env,
            &factory,
        )
        .unwrap();
        for seed in 0..5u64 {
            let mut p = GrammarProcessor::new(named.clone(), None);
            let text = v.render(&decode(&v, &mut p, seed, 500));
            assert_eq!(p.state(), GrammarState::Complete, "{family:?}: {text:?}");
            // (`get_weather` may still appear inside a free-text string value; the pinned check
            // looks at the name slot.)
            family_check(family, &text, Some("search"));
        }
        assert!(matches!(
            CompiledGrammar::compile(
                &tool_spec(family, ToolChoice::Named("nope".into())),
                &v.env,
                &factory
            ),
            Err(GrammarError::InvalidSpec(_))
        ));
    }
}

#[test]
fn every_family_auto_is_lazy_on_the_opener() {
    let v = vocab();
    let factory = new_parser_factory(&v.env).unwrap();
    let all: Vec<Token> = (0..v.n() as Token).collect();
    for family in FAMILIES {
        let auto = CompiledGrammar::compile(&tool_spec(family, ToolChoice::Auto), &v.env, &factory)
            .unwrap_or_else(|e| panic!("{family:?}: {e}"));
        assert_eq!(auto.triggers().len(), 1);
        assert!(auto.triggers()[0].starts_with(opener(family).as_bytes()));
        // Free text, then EOS: never constrained.
        let mut p = GrammarProcessor::new(auto.clone(), None);
        assert_eq!(p.state(), GrammarState::Watching);
        for t in [v.id("Paris"), b' ' as Token, v.id("name"), v.id("<think>")] {
            assert_eq!(v.allowed(&mut p), all);
            p.accept(t);
        }
        p.accept(v.env.eos());
        assert_eq!(p.state(), GrammarState::Watching);
        // Free text, then the opener: constrained from there on, and the call is well formed.
        for seed in 0..10u64 {
            let mut p = GrammarProcessor::new(auto.clone(), None);
            let prefix = [v.id("Paris"), b' ' as Token];
            for &t in &prefix {
                p.accept(t);
            }
            let trigger = String::from_utf8(auto.triggers()[0].clone()).unwrap();
            let mut pre = Vec::new();
            match family {
                ToolFamily::Harmony => {
                    pre.push(v.id("<|channel|>"));
                    pre.push(v.id("commentary to=functions."));
                }
                _ => pre.push(v.id(&trigger)),
            }
            for &t in &pre {
                p.accept(t);
            }
            assert_eq!(p.state(), GrammarState::Active, "{family:?}");
            let toks = decode(&v, &mut p, seed, 500);
            let text = format!("{}{}", v.render(&pre), v.render(&toks));
            assert_eq!(p.state(), GrammarState::Complete, "{family:?}: {text:?}");
            family_check(family, &text, None);
        }
    }
}

#[test]
fn hermes_auto_after_think_end_then_opener() {
    // A user trigger (`</think>`) is waited for before the family's own opener.
    let v = vocab();
    let mut p = GrammarProcessor::compile(
        &tool_spec(ToolFamily::Hermes, ToolChoice::Auto),
        &v.env,
        Some(LazyTrigger::new("</think>")),
    )
    .unwrap();
    p.accept(v.id("<tool_call>"));
    assert_eq!(
        p.state(),
        GrammarState::Watching,
        "an opener inside thinking does not count"
    );
    p.accept(v.id("</think>"));
    assert_eq!(p.state(), GrammarState::Watching);
    p.accept(v.id("<tool_call>\n"));
    assert_eq!(p.state(), GrammarState::Active);
    let toks = decode(&v, &mut p, 3, 400);
    let text = format!("<tool_call>\n{}", v.render(&toks));
    family_check(ToolFamily::Hermes, &text, None);
}

#[test]
fn grammar_cache_hits_and_evicts() {
    let v = vocab();
    let mut cache = GrammarCache::new(2);
    let a = GrammarSpec::Regex("a+".into());
    let b = GrammarSpec::Regex("b+".into());
    let c = GrammarSpec::Regex("c+".into());
    let ga = cache.get_or_compile(&a, &v.env).unwrap();
    assert_eq!(cache.counters(), (0, 1));
    let ga2 = cache.get_or_compile(&a, &v.env).unwrap();
    assert_eq!(cache.counters(), (1, 1));
    assert_eq!(ga.spec_hash(), ga2.spec_hash());
    cache.get_or_compile(&b, &v.env).unwrap();
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.counters(), (1, 2));
    cache.get_or_compile(&a, &v.env).unwrap(); // `a` is now the most recently used
    assert_eq!(cache.counters(), (2, 2));
    cache.get_or_compile(&c, &v.env).unwrap(); // evicts `b`, the least recently used
    assert_eq!(cache.len(), 2, "limit enforced");
    assert_eq!(cache.counters(), (2, 3));
    cache.get_or_compile(&a, &v.env).unwrap();
    assert_eq!(cache.counters(), (3, 3));
    cache.get_or_compile(&b, &v.env).unwrap();
    assert_eq!(cache.counters(), (3, 4));
    // A different tokenizer is a different key, and the factory is per tokenizer.
    let other = vocab_with_extra_token();
    cache.get_or_compile(&a, &other).unwrap();
    assert_eq!(cache.counters(), (3, 5));
    // Cached entries hand out independent parser states.
    let mut p1 = GrammarProcessor::new(cache.get_or_compile(&a, &v.env).unwrap(), None);
    let mut p2 = GrammarProcessor::new(cache.get_or_compile(&a, &v.env).unwrap(), None);
    p1.accept(b'a' as Token);
    p1.accept(v.env.eos());
    assert_eq!(p1.state(), GrammarState::Complete);
    assert_eq!(p2.state(), GrammarState::Active);
    assert_eq!(v.allowed(&mut p2), vec![b'a' as Token]);
    cache.clear();
    assert!(cache.is_empty());
    let zero = GrammarCache::new(0);
    assert_eq!(zero.limit(), 0);
}

fn vocab_with_extra_token() -> Arc<GrammarTokenizer> {
    let v = vocab();
    let mut pieces: Vec<Vec<u8>> = (0..v.n() as u32).map(|i| v.env.piece(i).to_vec()).collect();
    let mut special: Vec<bool> = (0..v.n() as u32).map(|i| v.env.is_special(i)).collect();
    pieces.push(b"extra".to_vec());
    special.push(false);
    Arc::new(GrammarTokenizer::from_pieces(pieces, &special, v.env.eos(), v.env.eog()).unwrap())
}

#[test]
fn spec_hash_is_stable_and_serde_round_trips() {
    let spec = tool_spec(ToolFamily::QwenXml, ToolChoice::Named("search".into()));
    let s = serde_json::to_string(&spec).unwrap();
    let back: GrammarSpec = serde_json::from_str(&s).unwrap();
    assert_eq!(back, spec);
    assert_eq!(back.hash(), spec.hash());
    assert_ne!(
        spec.hash(),
        tool_spec(ToolFamily::Hermes, ToolChoice::Named("search".into())).hash()
    );
    assert_ne!(
        spec.hash(),
        tool_spec(ToolFamily::QwenXml, ToolChoice::Auto).hash()
    );
    let js = GrammarSpec::JsonSchema(json!({"type": "string"}));
    assert_eq!(
        js.hash(),
        GrammarSpec::JsonSchema(json!({"type": "string"})).hash()
    );
    assert_ne!(js.hash(), GrammarSpec::Lark("start: \"x\"".into()).hash());
}

/// A 262,144-entry vocabulary: every byte plus distinct ASCII strings of 2–4 letters.
fn big_vocab() -> Arc<GrammarTokenizer> {
    const N: usize = 262_144;
    let mut pieces: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
    let letters: Vec<u8> = (b'a'..=b'z').chain(b'A'..=b'Z').collect();
    let mut i = 0usize;
    while pieces.len() < N - 1 {
        let mut s = Vec::new();
        let mut x = i;
        loop {
            s.push(letters[x % letters.len()]);
            x /= letters.len();
            if x == 0 {
                break;
            }
        }
        // Mix in a few JSON-ish shapes so the trie is not all letters.
        match i % 7 {
            0 => s.insert(0, b' '),
            1 => s.push(b'"'),
            2 => s.insert(0, b'"'),
            _ => {}
        }
        pieces.push(s);
        i += 1;
    }
    pieces.push(b"<|im_end|>".to_vec());
    let mut special = vec![false; N];
    special[N - 1] = true;
    let eos = (N - 1) as Token;
    Arc::new(GrammarTokenizer::from_pieces(pieces, &special, eos, &[eos]).unwrap())
}

#[test]
fn mask_timing_on_262k_vocab() {
    let t0 = Instant::now();
    let env = big_vocab();
    let t_env = t0.elapsed();
    assert_eq!(env.n_vocab(), 262_144);
    let t1 = Instant::now();
    let factory = new_parser_factory(&env).unwrap();
    let t_factory = t1.elapsed();
    let t2 = Instant::now();
    let compiled =
        CompiledGrammar::compile(&GrammarSpec::JsonSchema(person_schema()), &env, &factory)
            .unwrap();
    let t_compile = t2.elapsed();
    let n = env.n_vocab();
    let mut total_steps = 0u64;
    let mut stats = GrammarStats::default();
    for seed in 0..5u64 {
        let mut p = GrammarProcessor::new(compiled.clone(), None);
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        let base: Vec<f32> = (0..n).map(|_| rng.next_f64() as f32).collect();
        let mut out = Vec::new();
        for step in 0..200 {
            let mut l = rotated(&base, step);
            p.process(&mut l);
            let t = argmax(&l);
            assert!(l[t as usize].is_finite());
            p.accept(t);
            out.push(t);
            total_steps += 1;
            if env.eog().contains(&t) {
                break;
            }
        }
        assert_eq!(p.state(), GrammarState::Complete, "seed {seed}");
        let text: Vec<u8> = out
            .iter()
            .filter(|t| !env.eog().contains(t))
            .flat_map(|&t| env.piece(t).to_vec())
            .collect();
        let value: Value = serde_json::from_slice(&text).unwrap();
        check_person(&value);
        let s = p.stats();
        stats.masks += s.masks;
        stats.mask_time_total += s.mask_time_total;
        stats.mask_time_max = stats.mask_time_max.max(s.mask_time_max);
    }
    let mean = stats.mask_time_mean();
    eprintln!(
        "mask_timing_on_262k_vocab: env build {t_env:?}, factory {t_factory:?}, compile {t_compile:?}; \
         {} masks over {total_steps} steps: mean {mean:?}, max {:?} ({})",
        stats.masks,
        stats.mask_time_max,
        if cfg!(debug_assertions) { "debug build" } else { "release build" }
    );
    // Loose guard: the architecture targets ~1 ms per token (llguidance's own figure is ~50 µs on
    // a 128k tokenizer in release builds); debug builds are an order of magnitude slower.
    let limit = if cfg!(debug_assertions) { 50 } else { 5 };
    assert!(
        mean < std::time::Duration::from_millis(limit),
        "mean mask time {mean:?} exceeds {limit} ms"
    );
}

/// Real tokenizer (set `LLMARIO_TEST_GGUF` to one or more comma-separated GGUF paths; the first
/// one whose tokenizer loads is used — Qwen3-1.7B in the engine's test setup).
#[test]
fn real_tokenizer_json_schema_and_hermes() {
    let Ok(list) = std::env::var("LLMARIO_TEST_GGUF") else {
        eprintln!("real_tokenizer_json_schema_and_hermes: skipped (LLMARIO_TEST_GGUF not set)");
        return;
    };
    let Some(path) = list.split(',').map(str::trim).find(|p| !p.is_empty()) else {
        return;
    };
    let gguf =
        llmario_engine_formats::GgufFile::open(std::path::Path::new(path)).expect("open gguf");
    let tok = llmario_engine_tokenizer::Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let t0 = Instant::now();
    let env = Arc::new(GrammarTokenizer::from_tokenizer(&tok).unwrap());
    let t_env = t0.elapsed();
    let t1 = Instant::now();
    let factory = new_parser_factory(&env).unwrap();
    let t_factory = t1.elapsed();
    let n = env.n_vocab();
    assert_eq!(n, tok.n_vocab());
    eprintln!(
        "real tokenizer {path}: n_vocab {n}, eos {} eog {:?}; env {t_env:?}, factory {t_factory:?}",
        env.eos(),
        env.eog()
    );

    // Fixed logits: a deterministic pseudo-random vector, perturbed per step.
    let mut rng = Xoshiro256StarStar::seed_from_u64(42);
    let base: Vec<f32> = (0..n).map(|_| rng.next_f64() as f32).collect();
    let run = |p: &mut GrammarProcessor, max_steps: usize| -> Vec<Token> {
        let mut out = Vec::new();
        for step in 0..max_steps {
            let mut l = rotated(&base, step);
            p.process(&mut l);
            let t = argmax(&l);
            assert!(l[t as usize].is_finite());
            p.accept(t);
            out.push(t);
            if env.eog().contains(&t) {
                break;
            }
        }
        out
    };

    let t2 = Instant::now();
    let compiled =
        CompiledGrammar::compile(&GrammarSpec::JsonSchema(person_schema()), &env, &factory)
            .unwrap();
    let t_compile = t2.elapsed();
    let mut p = GrammarProcessor::new(compiled, None);
    let toks = run(&mut p, 300);
    assert_eq!(p.state(), GrammarState::Complete, "{:?}", tok.decode(&toks));
    let text = tok.decode(&toks);
    let value: Value =
        serde_json::from_str(text.trim()).unwrap_or_else(|e| panic!("{text:?}: {e}"));
    check_person(&value);
    eprintln!(
        "real tokenizer json_schema: compile {t_compile:?}; {} tokens, {} masks, mean {:?}, max {:?}: {text}",
        toks.len(),
        p.stats().masks,
        p.stats().mask_time_mean(),
        p.stats().mask_time_max
    );

    let compiled = CompiledGrammar::compile(
        &tool_spec(ToolFamily::Hermes, ToolChoice::Required),
        &env,
        &factory,
    )
    .unwrap();
    let mut p = GrammarProcessor::new(compiled, None);
    let toks = run(&mut p, 600);
    let text = tok.decode_with_special(&toks);
    assert_eq!(p.state(), GrammarState::Complete, "{text:?}");
    let text = text
        .trim_end_matches("<|im_end|>")
        .trim_end_matches("<|endoftext|>");
    assert!(text.starts_with("<tool_call>"), "{text:?}");
    for c in hermes_calls(text) {
        check_weather_args(c["name"].as_str().unwrap(), &c["arguments"]);
    }
    eprintln!(
        "real tokenizer hermes required: {} tokens, mean mask {:?}, max {:?}: {text}",
        toks.len(),
        p.stats().mask_time_mean(),
        p.stats().mask_time_max
    );
}
