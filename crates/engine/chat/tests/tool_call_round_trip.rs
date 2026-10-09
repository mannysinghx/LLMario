//! Round trips per tool-call family: render an assistant turn with the (vendored or minimal)
//! template in `tests/fixtures`, take the model-generated part, feed it to the parser one
//! character at a time, in pseudo-random chunks and in one piece, and check that the same
//! calls come back with identical JSON arguments. Negative cases, reasoning extraction and
//! content interleaving are covered per family below.
//!
//! When `LLMARIO_TEST_GGUF` names the local model files, the assistant turn is also rendered
//! with each file's real `tokenizer.chat_template`, compared byte for byte with the fixture's
//! rendering, and parsed as well.

use llmario_engine_chat::parser::schema::type_text;
use llmario_engine_chat::{
    ChatTemplate, Content, Message, OutputEvent, OutputParser, ParserOptions, RenderRequest,
    TemplateFamily, ToolCall,
};
use llmario_engine_formats::GgufFile;
use serde_json::{json, Map, Value};
use std::path::Path;

fn fixture(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn tools() -> Value {
    json!([
        {"type": "function", "function": {"name": "get_weather", "description": "Weather.", "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                "days": {"type": "integer"}
            },
            "required": ["location"]
        }}},
        {"type": "function", "function": {"name": "search_flights", "description": "Flights.", "parameters": {
            "type": "object",
            "properties": {
                "route": {"type": "object", "properties": {"from": {"type": "string"}, "to": {"type": "string"}, "via": {"type": "array", "items": {"type": "string"}}}, "required": ["from", "to"]},
                "passengers": {"type": "array", "items": {"type": "object", "properties": {"age": {"type": "integer"}, "frequent_flyer": {"type": "boolean"}}}},
                "max_price": {"type": "number", "nullable": true},
                "direct": {"type": "boolean"},
                "note": {"type": "string"}
            },
            "required": ["route", "passengers"]
        }}}
    ])
}

fn flights_call() -> ToolCall {
    ToolCall {
        id: Some("call_a".into()),
        name: "search_flights".into(),
        arguments: json!({
            "route": {"from": "LHR", "to": "JFK", "via": []},
            "passengers": [{"age": 34, "frequent_flyer": true}, {"age": 29, "frequent_flyer": false}],
            "max_price": 899.5,
            "direct": false,
            "note": "line1\nline2 \"q\" é 'single'"
        }),
    }
}

fn weather_call() -> ToolCall {
    ToolCall {
        id: Some("call_b".into()),
        name: "get_weather".into(),
        arguments: json!({"location": "New York, US", "unit": "celsius", "days": 3}),
    }
}

/// A family's fixture and what its template can render in one assistant turn.
struct Case {
    name: &'static str,
    family: TemplateFamily,
    fixture: &'static str,
    bos: Option<&'static str>,
    eos: Option<&'static str>,
    enable_thinking: Option<bool>,
    /// The template renders at most this many calls per turn (gpt-oss: one).
    max_calls: usize,
    /// The template renders visible content alongside calls.
    content_with_calls: bool,
    /// The template renders `reasoning_content` on the last assistant turn.
    reasoning_with_calls: bool,
    /// Architecture name in the GGUF (`general.architecture`) for the real-template check.
    arch: Option<&'static str>,
}

const CASES: &[Case] = &[
    Case {
        name: "hermes/qwen3",
        family: TemplateFamily::Hermes,
        fixture: "qwen3.jinja",
        bos: None,
        eos: Some("<|im_end|>"),
        enable_thinking: Some(true),
        max_calls: 2,
        content_with_calls: true,
        reasoning_with_calls: true,
        arch: Some("qwen3"),
    },
    Case {
        name: "qwen_xml/qwen3.5",
        family: TemplateFamily::QwenXml,
        fixture: "qwen35.jinja",
        bos: None,
        eos: Some("<|im_end|>"),
        enable_thinking: Some(true),
        max_calls: 2,
        content_with_calls: true,
        reasoning_with_calls: true,
        arch: Some("qwen35"),
    },
    Case {
        name: "glm/glm-4.7",
        family: TemplateFamily::Glm,
        fixture: "glm47.jinja",
        bos: None,
        eos: Some("<|user|>"),
        enable_thinking: Some(true),
        max_calls: 2,
        content_with_calls: true,
        reasoning_with_calls: true,
        arch: None,
    },
    Case {
        name: "gemma4",
        family: TemplateFamily::Gemma4,
        fixture: "gemma4.jinja",
        bos: Some("<bos>"),
        eos: Some("<eos>"),
        enable_thinking: Some(true),
        max_calls: 2,
        content_with_calls: true,
        reasoning_with_calls: true,
        arch: Some("gemma4"),
    },
    // gpt-oss reads a `thinking` field (not `reasoning_content`) and renders the content of a
    // tool-call turn as the `analysis` channel, so content comes back as reasoning there.
    Case {
        name: "harmony/gpt-oss",
        family: TemplateFamily::Harmony,
        fixture: "gptoss.jinja",
        bos: Some("<|startoftext|>"),
        eos: Some("<|return|>"),
        enable_thinking: None,
        max_calls: 1,
        content_with_calls: true,
        reasoning_with_calls: false,
        arch: Some("gpt-oss"),
    },
    Case {
        name: "llama3.1 json",
        family: TemplateFamily::Llama3,
        fixture: "llama31.jinja",
        bos: Some("<|begin_of_text|>"),
        eos: Some("<|eot_id|>"),
        enable_thinking: None,
        max_calls: 1,
        content_with_calls: false,
        reasoning_with_calls: false,
        arch: None,
    },
    Case {
        name: "llama4 pythonic",
        family: TemplateFamily::Llama3,
        fixture: "llama4.jinja",
        bos: Some("<|begin_of_text|>"),
        eos: Some("<|eot|>"),
        enable_thinking: None,
        max_calls: 2,
        content_with_calls: false,
        reasoning_with_calls: false,
        arch: None,
    },
    Case {
        name: "mistral/devstral2",
        family: TemplateFamily::Mistral,
        fixture: "mistral.jinja",
        bos: Some("<s>"),
        eos: Some("</s>"),
        enable_thinking: None,
        max_calls: 2,
        content_with_calls: true,
        reasoning_with_calls: false,
        arch: None,
    },
    Case {
        name: "lfm2",
        family: TemplateFamily::Lfm2,
        fixture: "lfm2.jinja",
        bos: Some("<|startoftext|>"),
        eos: Some("<|im_end|>"),
        enable_thinking: None,
        max_calls: 2,
        content_with_calls: true,
        reasoning_with_calls: false,
        arch: None,
    },
    Case {
        name: "olmo3",
        family: TemplateFamily::Olmo3,
        fixture: "olmo3.jinja",
        bos: Some("<|endoftext|>"),
        eos: Some("<|endoftext|>"),
        enable_thinking: None,
        max_calls: 2,
        content_with_calls: true,
        reasoning_with_calls: false,
        arch: Some("olmo2"),
    },
];

fn template(case: &Case, source: &str) -> ChatTemplate {
    let now = chrono::DateTime::parse_from_rfc3339("2026-03-04T05:06:07+00:00").unwrap();
    ChatTemplate::with_now(source, case.bos, case.eos, Some(now))
        .unwrap_or_else(|e| panic!("{}: {e}", case.name))
}

/// Render the conversation with and without the final assistant message and return the
/// model-generated part plus whether the prompt ended inside the reasoning block.
fn assistant_part(case: &Case, tpl: &ChatTemplate, assistant: &Message) -> (String, ParserOptions) {
    let history = vec![
        Message::new("system", "You can call tools."),
        Message::new(
            "user",
            "Flights LHR to JFK for two adults under 900, and the weather there?",
        ),
    ];
    let mut prompt_req = RenderRequest {
        messages: history.clone(),
        add_generation_prompt: true,
        tools: Some(tools()),
        enable_thinking: case.enable_thinking,
        ..Default::default()
    };
    if case.family == TemplateFamily::Harmony {
        prompt_req
            .extra
            .insert("reasoning_effort".into(), json!("low"));
    }
    let prompt = tpl
        .render(&prompt_req)
        .unwrap_or_else(|e| panic!("{} prompt: {e}", case.name));
    let mut full_req = prompt_req.clone();
    full_req.messages.push(assistant.clone());
    full_req.add_generation_prompt = false;
    let full = tpl
        .render(&full_req)
        .unwrap_or_else(|e| panic!("{} full: {e}", case.name));
    // The generation prompt is a prefix of the full render, except that a template which
    // emits its reasoning opener in the prompt (Qwen3.5 `<think>\n`, GLM `<think>`) may render
    // history differently (GLM writes `<|assistant|></think>` for a turn without reasoning).
    // So the opener is stripped from the prompt first; the parser swallows a repeated opener
    // at the start of a reasoning-open stream.
    let reasoning_open = OutputParser::prompt_opens_reasoning(case.family, &prompt);
    let mut base = prompt.as_str();
    if reasoning_open {
        let (open, _) = OutputParser::reasoning_markers(case.family).unwrap();
        let tail = base.trim_end_matches(['\n', ' ']);
        base = &tail[..tail.len() - open.trim_end_matches('\n').len()];
    }
    let generated = full.strip_prefix(base).unwrap_or_else(|| {
        panic!(
            "{}: the generation prompt is not a prefix of the full render\nprompt: {prompt:?}\nfull: {full:?}",
            case.name
        )
    });
    let options = ParserOptions { reasoning_open };
    (generated.to_string(), options)
}

#[derive(Debug, Default)]
struct Summary {
    reasoning: String,
    content: String,
    calls: Vec<(String, Value, String)>,
    invalid: Vec<String>,
}

fn summarize(events: &[OutputEvent]) -> Summary {
    let mut s = Summary::default();
    let mut open: Option<usize> = None;
    for e in events {
        match e {
            OutputEvent::Reasoning(t) => s.reasoning.push_str(t),
            OutputEvent::Content(t) => s.content.push_str(t),
            OutputEvent::ToolCallStart { index, name, id } => {
                assert_eq!(*index, s.calls.len(), "call indexes are sequential");
                assert!(open.is_none(), "a call started while another is open");
                assert!(!id.is_empty());
                open = Some(*index);
                s.calls.push((name.clone(), Value::Null, String::new()));
            }
            OutputEvent::ToolCallArgumentsDelta { index, delta } => {
                assert_eq!(open, Some(*index), "delta for a call that is not open");
                s.calls[*index].2.push_str(delta);
            }
            OutputEvent::ToolCallEnd { index, arguments } => {
                assert_eq!(open, Some(*index), "end for a call that is not open");
                assert!(arguments.is_object());
                s.calls[*index].1 = arguments.clone();
                open = None;
            }
            OutputEvent::Invalid { reason } => {
                open = None;
                s.invalid.push(reason.clone());
            }
        }
    }
    assert!(open.is_none(), "a call was left open: {events:?}");
    s
}

#[derive(Clone, Copy, Debug)]
enum Feed {
    Whole,
    Chars,
    Random(u64),
}

fn run(
    case_family: TemplateFamily,
    options: ParserOptions,
    text: &str,
    feed: Feed,
) -> Vec<OutputEvent> {
    let mut p = OutputParser::with_options(case_family, Some(&tools()), options);
    let mut events = Vec::new();
    match feed {
        Feed::Whole => events.extend(p.push(text)),
        Feed::Chars => {
            for (i, c) in text.char_indices() {
                events.extend(p.push(&text[i..i + c.len_utf8()]));
            }
        }
        Feed::Random(seed) => {
            let mut x = seed;
            let mut rest = text;
            while !rest.is_empty() {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let n = 1 + ((x >> 33) % 9) as usize;
                let mut end = rest.len();
                for (count, (i, _)) in rest.char_indices().enumerate() {
                    if count == n {
                        end = i;
                        break;
                    }
                }
                events.extend(p.push(&rest[..end]));
                rest = &rest[end..];
            }
        }
    }
    events.extend(p.finish());
    events
}

/// Expected arguments after the family's wire format: Qwen XML strings are untyped text
/// unless the schema types them, which it does for every field used here.
fn check_round_trip(case: &Case, text: &str, options: ParserOptions, assistant: &Message) {
    let expected_calls = assistant.tool_calls.clone().unwrap_or_default();
    let feeds = [
        Feed::Whole,
        Feed::Chars,
        Feed::Random(1),
        Feed::Random(7),
        Feed::Random(99),
    ];
    for feed in feeds {
        let events = run(case.family, options, text, feed);
        for e in &events {
            if let OutputEvent::Content(t) | OutputEvent::Reasoning(t) = e {
                assert!(
                    !t.contains("<tool")
                        && !t.contains("</think")
                        && !t.contains("<|")
                        && !t.contains("[TOOL")
                        && !t.contains("<function"),
                    "{} {feed:?}: a marker leaked into the text stream: {t:?}",
                    case.name
                );
            }
        }
        let s = summarize(&events);
        assert!(
            s.invalid.is_empty(),
            "{} {feed:?}: invalid {:?}\ntext: {text:?}\nevents: {events:#?}",
            case.name,
            s.invalid
        );
        assert_eq!(
            s.calls.len(),
            expected_calls.len(),
            "{} {feed:?}: call count\ntext: {text:?}\nevents: {events:#?}",
            case.name
        );
        for (got, want) in s.calls.iter().zip(&expected_calls) {
            assert_eq!(got.0, want.name, "{} {feed:?}: name", case.name);
            assert_eq!(
                got.1, want.arguments,
                "{} {feed:?}: arguments for {}",
                case.name, want.name
            );
            let from_deltas: Value = serde_json::from_str(&got.2).unwrap_or_else(|e| {
                panic!(
                    "{} {feed:?}: deltas are not JSON ({e}): {:?}",
                    case.name, got.2
                )
            });
            assert_eq!(
                from_deltas, want.arguments,
                "{} {feed:?}: deltas",
                case.name
            );
        }
        let want_content = match &assistant.content {
            Content::Text(t) if case.content_with_calls || expected_calls.is_empty() => t.clone(),
            _ => String::new(),
        };
        if case.family == TemplateFamily::Harmony && !expected_calls.is_empty() {
            assert_eq!(
                s.reasoning.trim(),
                want_content.trim(),
                "{} {feed:?}: analysis",
                case.name
            );
            assert_eq!(s.content.trim(), "", "{} {feed:?}: content", case.name);
            continue;
        }
        if case.reasoning_with_calls {
            if let Some(r) = &assistant.reasoning_content {
                assert_eq!(s.reasoning.trim(), r, "{} {feed:?}: reasoning", case.name);
            }
        } else {
            assert_eq!(
                s.reasoning, "",
                "{} {feed:?}: unexpected reasoning",
                case.name
            );
        }
        assert_eq!(
            s.content.trim(),
            want_content.trim(),
            "{} {feed:?}: content",
            case.name
        );
    }
}

fn assistant_messages(case: &Case) -> Vec<Message> {
    let mut out = Vec::new();
    for n in 1..=case.max_calls {
        let calls: Vec<ToolCall> = [flights_call(), weather_call()]
            .into_iter()
            .take(n)
            .collect();
        let mut m = Message::new(
            "assistant",
            if case.content_with_calls {
                "Let me check."
            } else {
                ""
            },
        );
        m.tool_calls = Some(calls);
        if case.reasoning_with_calls {
            m.reasoning_content = Some("Need flights and weather.".into());
        }
        out.push(m);
        if case.content_with_calls {
            let mut bare = Message::new("assistant", "");
            bare.tool_calls = Some(vec![weather_call()]);
            out.push(bare);
        }
    }
    // A plain answer (no calls) with reasoning where the template renders it.
    let mut plain = Message::new("assistant", "BA117 at 812, 18 °C.");
    if case.reasoning_with_calls {
        plain.reasoning_content = Some("Done.".into());
    }
    out.push(plain);
    out
}

#[test]
fn round_trips_per_family_with_fixture_templates() {
    for case in CASES {
        let tpl = template(case, &fixture(case.fixture));
        assert_eq!(
            tpl.detect_family(),
            case.family,
            "{}: fixture family",
            case.name
        );
        for assistant in assistant_messages(case) {
            let (text, options) = assistant_part(case, &tpl, &assistant);
            check_round_trip(case, &text, options, &assistant);
        }
    }
}

#[test]
fn real_templates_match_fixtures_and_round_trip() {
    let Ok(list) = std::env::var("LLMARIO_TEST_GGUF") else {
        eprintln!("skipped: LLMARIO_TEST_GGUF not set");
        return;
    };
    let mut checked = 0;
    for path in list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let gguf = GgufFile::open(Path::new(path)).unwrap_or_else(|e| panic!("{path}: {e}"));
        let arch = gguf.architecture().unwrap_or("?").to_string();
        let source = gguf
            .get_str("tokenizer.chat_template")
            .unwrap_or_else(|| panic!("{path}: no chat template"))
            .to_string();
        drop(gguf);
        let Some(case) = CASES.iter().find(|c| c.arch == Some(arch.as_str())) else {
            eprintln!("{path}: arch {arch} has no round-trip case, skipped");
            continue;
        };
        let real = template(case, &source);
        let fix = template(case, &fixture(case.fixture));
        assert_eq!(real.detect_family(), case.family, "{path}: family");
        for assistant in assistant_messages(case) {
            let (real_text, real_opts) = assistant_part(case, &real, &assistant);
            let (fix_text, fix_opts) = assistant_part(case, &fix, &assistant);
            assert_eq!(
                real_text, fix_text,
                "{path}: fixture {} renders the assistant turn differently",
                case.fixture
            );
            assert_eq!(real_opts, fix_opts, "{path}: reasoning_open differs");
            check_round_trip(case, &real_text, real_opts, &assistant);
            checked += 1;
        }
        println!("{path}: {arch} verified against {}", case.fixture);
    }
    assert!(checked > 0, "no GGUF matched a round-trip case");
}

fn events(family: TemplateFamily, options: ParserOptions, text: &str) -> Vec<OutputEvent> {
    run(family, options, text, Feed::Chars)
}

fn whole(family: TemplateFamily, text: &str) -> Summary {
    summarize(&run(family, ParserOptions::default(), text, Feed::Whole))
}

#[test]
fn truncated_calls_are_invalid_on_finish_and_text_is_preserved() {
    let cases: Vec<(TemplateFamily, &str)> = vec![
        (TemplateFamily::Hermes, "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Par"),
        (TemplateFamily::QwenXml, "<tool_call>\n<function=get_weather>\n<parameter=location>\nParis"),
        (TemplateFamily::Glm, "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Par"),
        (TemplateFamily::Gemma4, "<|tool_call>call:get_weather{location:<|\"|>Paris"),
        (TemplateFamily::Llama3, "<|python_tag|>{\"name\": \"get_weather\", \"parameters\": {\"location\": \"Par"),
        (TemplateFamily::Llama3, "[get_weather(location=\"Paris\""),
        (TemplateFamily::Mistral, "[TOOL_CALLS]get_weather[ARGS]{\"location\": \"Par"),
        (TemplateFamily::Lfm2, "<|tool_call_start|>[get_weather(location='Par"),
        (TemplateFamily::Olmo3, "<function_calls>get_weather(location=\"Par"),
        (TemplateFamily::Harmony, "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\": \"Par"),
    ];
    for (family, text) in cases {
        for feed in [Feed::Whole, Feed::Chars, Feed::Random(3)] {
            let ev = run(family, ParserOptions::default(), text, feed);
            let s = summarize(&ev);
            assert_eq!(s.invalid.len(), 1, "{family:?} {feed:?}: {ev:#?}");
            assert!(
                s.invalid[0].contains("truncated"),
                "{family:?}: {}",
                s.invalid[0]
            );
            // The raw text survives as content (Harmony keeps the JSON body; the header is
            // structural).
            let expect = if family == TemplateFamily::Harmony {
                "{\"location\": \"Par"
            } else {
                text
            };
            assert_eq!(s.content, expect, "{family:?} {feed:?}: raw text preserved");
            assert!(
                s.calls.iter().all(|c| c.1.is_null()),
                "{family:?}: no completed call"
            );
        }
    }
}

#[test]
fn malformed_calls_are_invalid_with_raw_text_in_content() {
    let cases: Vec<(TemplateFamily, &str)> = vec![
        (TemplateFamily::Hermes, "<tool_call>not json at all</tool_call>"),
        (TemplateFamily::Hermes, "<tool_call>{\"name\": \"get_weather\", \"arguments\": {bad}}</tool_call>"),
        (TemplateFamily::QwenXml, "<tool_call>\n<function=get_weather>\n<param>oops</param>\n</function>\n</tool_call>"),
        (TemplateFamily::Glm, "<tool_call>get_weather<arg_key>location</arg_key><oops/></tool_call>"),
        (TemplateFamily::Gemma4, "<|tool_call>call:get_weather{location <|\"|>Paris<|\"|>}<tool_call|>"),
        (TemplateFamily::Mistral, "[TOOL_CALLS]get_weather[ARGS]{\"location\": tru,}"),
        (TemplateFamily::Lfm2, "<|tool_call_start|>[get_weather('Paris')]<|tool_call_end|>"),
        (TemplateFamily::Olmo3, "<function_calls>get_weather(location=\"Paris\" 1)</function_calls>"),
        (TemplateFamily::Harmony, "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\": }<|call|>"),
    ];
    for (family, text) in cases {
        let ev = run(family, ParserOptions::default(), text, Feed::Whole);
        let s = summarize(&ev);
        assert_eq!(s.invalid.len(), 1, "{family:?}: {ev:#?}");
        let expect = if family == TemplateFamily::Harmony {
            "{\"location\": }"
        } else {
            text
        };
        assert_eq!(s.content, expect, "{family:?}: raw text preserved\n{ev:#?}");
        assert!(
            s.calls.iter().all(|c| c.1.is_null()),
            "{family:?}: no completed call"
        );
        // Character feeding yields the same text, possibly split across more events.
        let s2 = summarize(&events(family, ParserOptions::default(), text));
        assert_eq!(s2.content, expect, "{family:?} chars");
        assert_eq!(s2.invalid.len(), 1, "{family:?} chars");
    }
}

#[test]
fn reasoning_with_and_without_closing_tag() {
    // Opened and closed by the model (Qwen3 with thinking on).
    let s = whole(
        TemplateFamily::Hermes,
        "<think>\nlet me see\n</think>\n\nHello.",
    );
    assert_eq!(s.reasoning.trim(), "let me see");
    assert_eq!(s.content.trim(), "Hello.");
    // Never closed: the rest of the output is reasoning.
    let s = summarize(&events(
        TemplateFamily::Hermes,
        ParserOptions::default(),
        "<think>still thinking",
    ));
    assert_eq!(s.reasoning, "still thinking");
    assert_eq!(s.content, "");
    // The prompt ended with `<think>\n` (Qwen3.5), so the output starts inside the block.
    let open = ParserOptions {
        reasoning_open: true,
    };
    let s = summarize(&events(
        TemplateFamily::QwenXml,
        open,
        "I need a tool.\n</think>\n\nSure.",
    ));
    assert_eq!(s.reasoning.trim(), "I need a tool.");
    assert_eq!(s.content.trim(), "Sure.");
    // A repeated opener right at the start is swallowed.
    let s = summarize(&events(
        TemplateFamily::QwenXml,
        open,
        "<think>\nagain\n</think>\n\nok",
    ));
    assert_eq!(s.reasoning.trim(), "again");
    assert_eq!(s.content.trim(), "ok");
    // Reasoning open and never closed before end of output.
    let s = summarize(&events(TemplateFamily::Glm, open, "pondering forever"));
    assert_eq!(s.reasoning, "pondering forever");
    assert_eq!(s.content, "");
    // Thinking disabled: GLM's prompt ended with `</think>`, output has no markers.
    let s = whole(TemplateFamily::Glm, "Plain answer.");
    assert_eq!(
        (s.reasoning.as_str(), s.content.as_str()),
        ("", "Plain answer.")
    );
    // A stray closer in content mode is dropped, not shown.
    let s = whole(TemplateFamily::Glm, "</think>Answer.");
    assert_eq!((s.reasoning.as_str(), s.content.as_str()), ("", "Answer."));
    // Gemma 4: opened by the model, closed, then content; and the open-from-prompt form.
    let s = summarize(&events(
        TemplateFamily::Gemma4,
        ParserOptions::default(),
        "<|channel>thought\nhmm\n<channel|>Answer<turn|>",
    ));
    assert_eq!((s.reasoning.trim(), s.content.trim()), ("hmm", "Answer"));
    let s = summarize(&events(TemplateFamily::Gemma4, open, "hmm<channel|>Answer"));
    assert_eq!((s.reasoning.trim(), s.content.trim()), ("hmm", "Answer"));
    // Harmony: analysis is reasoning, final is content, stop tokens may be stripped.
    let s = summarize(&events(TemplateFamily::Harmony, ParserOptions::default(), "<|channel|>analysis<|message|>deep<|end|><|start|>assistant<|channel|>final<|message|>Hi there"));
    assert_eq!(
        (s.reasoning.as_str(), s.content.as_str()),
        ("deep", "Hi there")
    );
    // Mistral Magistral-style [THINK] blocks.
    let s = whole(TemplateFamily::Mistral, "[THINK]why[/THINK]because");
    assert_eq!(
        (s.reasoning.as_str(), s.content.as_str()),
        ("why", "because")
    );
}

#[test]
fn content_interleaved_with_calls() {
    let text = "Before.<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}\n</tool_call>\nMiddle.<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Rome\", \"days\": 2}}</tool_call>After.<|im_end|>";
    for feed in [Feed::Whole, Feed::Chars, Feed::Random(5)] {
        let s = summarize(&run(
            TemplateFamily::Hermes,
            ParserOptions::default(),
            text,
            feed,
        ));
        assert!(s.invalid.is_empty(), "{feed:?}: {:?}", s.invalid);
        assert_eq!(
            s.content.split_whitespace().collect::<Vec<_>>(),
            ["Before.", "Middle.After."]
        );
        assert_eq!(s.calls.len(), 2);
        assert_eq!(s.calls[0].1, json!({"location": "Paris"}));
        assert_eq!(s.calls[1].1, json!({"location": "Rome", "days": 2}));
        // JSON-native deltas are the model's own bytes.
        assert_eq!(s.calls[0].2, "{\"location\": \"Paris\"}");
    }
    // Gemma 4 renders calls before content; GLM and Qwen XML content before calls.
    let s = whole(
        TemplateFamily::Gemma4,
        "<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}<tool_call|>Checking.<turn|>",
    );
    assert_eq!(s.content, "Checking.");
    assert_eq!(s.calls[0].1, json!({"location": "Paris"}));
    let s = whole(TemplateFamily::Glm, "Looking.<tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>2</arg_value></tool_call><tool_call>get_weather<arg_key>location</arg_key><arg_value>Rome</arg_value></tool_call><|user|>");
    assert_eq!(s.content, "Looking.");
    assert_eq!(s.calls.len(), 2);
    assert_eq!(s.calls[0].1, json!({"location": "Paris", "days": 2}));
    assert_eq!(s.calls[0].2, "{\"location\":\"Paris\",\"days\":2}");
}

#[test]
fn harmony_generation_order_and_variants() {
    // The order the model generates (differs from the template's history rendering).
    let text = "<|channel|>analysis<|message|>Need weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\": \"Paris\"}<|call|>";
    for feed in [Feed::Whole, Feed::Chars, Feed::Random(11)] {
        let s = summarize(&run(
            TemplateFamily::Harmony,
            ParserOptions::default(),
            text,
            feed,
        ));
        assert!(s.invalid.is_empty(), "{feed:?}: {:?}", s.invalid);
        assert_eq!(s.reasoning, "Need weather.");
        assert_eq!(s.content, "");
        assert_eq!(s.calls.len(), 1);
        assert_eq!(s.calls[0].0, "get_weather");
        assert_eq!(s.calls[0].1, json!({"location": "Paris"}));
        assert_eq!(s.calls[0].2, "{\"location\": \"Paris\"}");
    }
    // Two calls in one output, a commentary preamble, then a final answer with <|return|>.
    let text = "<|channel|>commentary<|message|>I will check both.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\": \"Paris\"}<|call|><|start|>assistant<|channel|>commentary to=functions.search_flights <|constrain|>json<|message|>{\"route\": {\"from\": \"A\", \"to\": \"B\"}, \"passengers\": []}<|call|><|start|>assistant<|channel|>final<|message|>Done.<|return|>";
    let s = summarize(&events(
        TemplateFamily::Harmony,
        ParserOptions::default(),
        text,
    ));
    assert!(s.invalid.is_empty());
    assert_eq!(s.content, "I will check both.Done.");
    assert_eq!(s.calls.len(), 2);
    assert_eq!(
        s.calls[1].1,
        json!({"route": {"from": "A", "to": "B"}, "passengers": []})
    );
    // History order (`to=` before `<|channel|>`), as the template renders it.
    let s = whole(TemplateFamily::Harmony, "<|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{\"location\": \"Oslo\"}<|call|>");
    assert_eq!(s.calls[0].1, json!({"location": "Oslo"}));
    // Built-in tools keep their namespace; the call id is still generated.
    let s = whole(
        TemplateFamily::Harmony,
        "<|channel|>analysis to=browser.search code<|message|>{\"query\": \"x\"}<|call|>",
    );
    assert_eq!(s.calls[0].0, "browser.search");
    // Control tokens stripped by the server: plain text is content.
    let s = summarize(&events(
        TemplateFamily::Harmony,
        ParserOptions::default(),
        "Just an answer without any header.",
    ));
    assert_eq!(s.content, "Just an answer without any header.");
    assert!(s.invalid.is_empty());
}

#[test]
fn llama_and_mistral_variants() {
    // Llama 3.1 JSON without <|python_tag|> at the start of the output (HF template form).
    let s = summarize(&events(
        TemplateFamily::Llama3,
        ParserOptions::default(),
        "{\"name\": \"get_weather\", \"parameters\": {\"location\": \"Paris\"}}<|eot_id|>",
    ));
    assert!(s.invalid.is_empty(), "{:?}", s.invalid);
    assert_eq!(s.calls[0].1, json!({"location": "Paris"}));
    assert_eq!(s.content, "");
    // ... but ordinary text that merely starts with a brace is content.
    let s = whole(TemplateFamily::Llama3, "{not a call} really");
    assert_eq!(s.content, "{not a call} really");
    assert_eq!(s.invalid.len(), 1);
    // Zero-shot <function=…> form.
    let s = summarize(&events(
        TemplateFamily::Llama3,
        ParserOptions::default(),
        "<function=get_weather>{\"location\": \"Paris\", \"days\": 1}</function><|eot_id|>",
    ));
    assert_eq!(s.calls[0].1, json!({"location": "Paris", "days": 1}));
    // Llama 4 pythonic with double-quoted strings and parallel calls, no brackets for a built-in.
    let s = summarize(&events(
        TemplateFamily::Llama3,
        ParserOptions::default(),
        "[get_weather(location=\"San Francisco\", days=2), get_weather(location='Seattle')]<|eot|>",
    ));
    assert_eq!(s.calls.len(), 2);
    assert_eq!(
        s.calls[0].1,
        json!({"location": "San Francisco", "days": 2})
    );
    assert_eq!(s.calls[1].1, json!({"location": "Seattle"}));
    let s = whole(
        TemplateFamily::Llama3,
        "<|python_tag|>brave_search.call(query=\"gold price\")<|eom_id|>",
    );
    assert_eq!(s.calls[0].0, "brave_search.call");
    assert_eq!(s.calls[0].1, json!({"query": "gold price"}));
    assert_eq!(
        s.invalid.len(),
        1,
        "undeclared tool is flagged but still delivered"
    );
    // A JSON list of call objects after <|python_tag|>.
    let s = whole(TemplateFamily::Llama3, "<|python_tag|>[{\"name\": \"get_weather\", \"parameters\": {\"location\": \"A\"}}, {\"name\": \"get_weather\", \"arguments\": {\"location\": \"B\"}}]");
    assert_eq!(s.calls.len(), 2);
    assert_eq!(s.calls[1].1, json!({"location": "B"}));
    // Mistral: new form, two calls, content before; and the older list form with ids.
    let s = summarize(&events(TemplateFamily::Mistral, ParserOptions::default(), "On it.[TOOL_CALLS]get_weather[ARGS]{\"location\": \"Paris\"}[TOOL_CALLS]get_weather[ARGS]{\"location\": \"Rome\"}</s>"));
    assert_eq!(s.content, "On it.");
    assert_eq!(s.calls.len(), 2);
    assert_eq!(s.calls[1].1, json!({"location": "Rome"}));
    let ev = run(TemplateFamily::Mistral, ParserOptions::default(), "[TOOL_CALLS][{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}, \"id\": \"D681PevKs\"}]", Feed::Chars);
    let s = summarize(&ev);
    assert_eq!(s.calls[0].1, json!({"location": "Paris"}));
    let ids: Vec<&str> = ev
        .iter()
        .filter_map(|e| match e {
            OutputEvent::ToolCallStart { id, .. } => Some(id.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(ids[0].len(), 9, "Mistral ids are 9 alphanumerics: {ids:?}");
}

#[test]
fn arguments_before_name_and_double_encoded_arguments() {
    // Name after arguments: no deltas until the object completes, then one delta.
    let ev = run(TemplateFamily::Hermes, ParserOptions::default(), "<tool_call>{\"arguments\": {\"location\": \"Paris\"}, \"name\": \"get_weather\"}</tool_call>", Feed::Chars);
    let s = summarize(&ev);
    assert_eq!(s.calls[0].1, json!({"location": "Paris"}));
    let deltas = ev
        .iter()
        .filter(|e| matches!(e, OutputEvent::ToolCallArgumentsDelta { .. }))
        .count();
    assert_eq!(deltas, 1);
    // Arguments as a JSON-encoded string are unwrapped.
    let s = whole(TemplateFamily::Hermes, "<tool_call>{\"name\": \"get_weather\", \"arguments\": \"{\\\"location\\\": \\\"Paris\\\"}\"}</tool_call>");
    assert_eq!(s.calls[0].1, json!({"location": "Paris"}));
    // Missing closer at end of output is accepted once the JSON is complete.
    let s = whole(
        TemplateFamily::Hermes,
        "<tool_call>{\"name\": \"get_weather\", \"arguments\": {}}",
    );
    assert!(s.invalid.is_empty());
    assert_eq!(s.calls[0].1, json!({}));
}

#[test]
fn xml_values_are_typed_by_schema_or_kept_as_text() {
    let text = "<tool_call>\n<function=search_flights>\n<parameter=route>\n{\"from\": \"LHR\", \"to\": \"JFK\"}\n</parameter>\n<parameter=passengers>\n[{\"age\": 1}]\n</parameter>\n<parameter=max_price>\nNone\n</parameter>\n<parameter=direct>\nTrue\n</parameter>\n<parameter=note>\n12\n</parameter>\n<parameter=mystery>\n42\n</parameter>\n</function>\n</tool_call>";
    let s = summarize(&events(
        TemplateFamily::QwenXml,
        ParserOptions::default(),
        text,
    ));
    assert!(s.invalid.is_empty());
    assert_eq!(
        s.calls[0].1,
        json!({"route": {"from": "LHR", "to": "JFK"}, "passengers": [{"age": 1}], "max_price": null, "direct": true, "note": "12", "mystery": 42})
    );
    // Without tool definitions: JSON when valid, text otherwise.
    let mut p = OutputParser::new(TemplateFamily::QwenXml, None);
    let mut ev = p.push("<tool_call><function=f><parameter=a>\nhello world\n</parameter><parameter=b>\n7\n</parameter></function></tool_call>");
    ev.extend(p.finish());
    let s = summarize(&ev);
    assert_eq!(s.calls[0].1, json!({"a": "hello world", "b": 7}));
    assert_eq!(type_text("x", None), json!("x"));
}

#[test]
fn gemma_nested_values_and_turn_end() {
    let text = "<|channel>thought\nplan\n<channel|><|tool_call>call:search_flights{direct:false,max_price:null,note:<|\"|>a,b}{<|\"|>,passengers:[{age:34,frequent_flyer:true}],route:{from:<|\"|>LHR<|\"|>,to:<|\"|>JFK<|\"|>,via:[]}}<tool_call|><turn|>";
    for feed in [Feed::Whole, Feed::Chars, Feed::Random(2)] {
        let s = summarize(&run(
            TemplateFamily::Gemma4,
            ParserOptions::default(),
            text,
            feed,
        ));
        assert!(s.invalid.is_empty(), "{feed:?}: {:?}", s.invalid);
        assert_eq!(s.reasoning.trim(), "plan");
        assert_eq!(s.content, "");
        assert_eq!(
            s.calls[0].1,
            json!({"direct": false, "max_price": null, "note": "a,b}{", "passengers": [{"age": 34, "frequent_flyer": true}], "route": {"from": "LHR", "to": "JFK", "via": []}})
        );
    }
}

#[test]
fn undeclared_tool_names_are_flagged_but_delivered() {
    let s = whole(
        TemplateFamily::Hermes,
        "<tool_call>{\"name\": \"nope\", \"arguments\": {\"a\": 1}}</tool_call>",
    );
    assert_eq!(s.invalid.len(), 1);
    assert!(s.invalid[0].contains("undeclared tool \"nope\""));
    assert_eq!(s.calls.len(), 1);
    assert_eq!(s.calls[0].1, json!({"a": 1}));
}

#[test]
fn partial_markers_never_leak_and_finish_releases_them() {
    let mut p = OutputParser::new(TemplateFamily::Hermes, None);
    assert_eq!(
        p.push("Hello <tool_"),
        vec![OutputEvent::Content("Hello ".into())]
    );
    assert_eq!(p.push("ca"), vec![]);
    // It was not a marker after all.
    assert_eq!(
        p.push("t is here"),
        vec![OutputEvent::Content("<tool_cat is here".into())]
    );
    assert_eq!(p.push(" <"), vec![OutputEvent::Content(" ".into())]);
    assert_eq!(p.finish(), vec![OutputEvent::Content("<".into())]);
    let mut p = OutputParser::new(TemplateFamily::Gemma4, None);
    assert_eq!(p.push("x<|chan"), vec![OutputEvent::Content("x".into())]);
    assert_eq!(
        p.push("nel>thought\nt"),
        vec![OutputEvent::Reasoning("t".into())]
    );
    assert_eq!(p.push("<chan"), vec![]);
    assert_eq!(p.push("nel|>y"), vec![OutputEvent::Content("y".into())]);
    assert_eq!(p.finish(), vec![]);
}

#[test]
fn events_shape_for_streaming_servers() {
    let mut p = OutputParser::new(TemplateFamily::Hermes, Some(&tools()));
    let mut ev = p.push("<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"loc");
    ev.extend(p.push("ation\": \"Paris\"}}</tool_call>"));
    ev.extend(p.finish());
    assert!(
        matches!(&ev[0], OutputEvent::ToolCallStart { index: 0, name, .. } if name == "get_weather")
    );
    assert_eq!(
        ev[1],
        OutputEvent::ToolCallArgumentsDelta {
            index: 0,
            delta: "{\"loc".into()
        }
    );
    assert_eq!(
        ev[2],
        OutputEvent::ToolCallArgumentsDelta {
            index: 0,
            delta: "ation\": \"Paris\"}".into()
        }
    );
    assert_eq!(
        ev[3],
        OutputEvent::ToolCallEnd {
            index: 0,
            arguments: json!({"location": "Paris"})
        }
    );
    assert_eq!(ev.len(), 4);
    let _unused: Map<String, Value> = Map::new();
}
