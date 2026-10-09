//! Reference-parity test: renders a fixed set of conversations with every GGUF's own chat
//! template through this crate and through Python Jinja2 (`scripts/engine/render_ref.py`, the
//! transformers environment) and compares byte for byte.
//!
//! Runs only when both are set:
//! * `LLMARIO_TEST_GGUF`   comma-separated GGUF paths (read-only; only the metadata is read)
//! * `LLMARIO_PYTHON`      a python with `jinja2` installed
//!
//! A conversation counts as parity when both sides render identical bytes, or when the template
//! itself raises on both sides. Any other outcome is a mismatch and fails the test.

use chrono::DateTime;
use llmario_engine_chat::{
    ChatError, ChatTemplate, Content, ContentPart, Message, RenderRequest, TemplateFamily, ToolCall,
};
use llmario_engine_formats::GgufFile;
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use std::process::Command;

const FIXED_NOW: &str = "2026-03-04T05:06:07+00:00";

fn msg(role: &str, content: &str) -> Message {
    Message::new(role, content)
}

fn tools() -> Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City and country, e.g. \"Paris, France\""},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"},
                        "days": {"type": "integer", "description": "Forecast horizon in days", "default": 1}
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_flights",
                "description": "Search flights between two airports.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "route": {
                            "type": "object",
                            "description": "Origin and destination",
                            "properties": {
                                "from": {"type": "string"},
                                "to": {"type": "string"},
                                "via": {"type": "array", "items": {"type": "string"}, "description": "Optional stopovers"}
                            },
                            "required": ["from", "to"]
                        },
                        "passengers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"age": {"type": "integer"}, "frequent_flyer": {"type": "boolean"}},
                                "required": ["age"]
                            }
                        },
                        "max_price": {"type": "number", "nullable": true, "description": "Upper price bound"}
                    },
                    "required": ["route", "passengers"]
                }
            }
        }
    ])
}

fn call(id: &str, name: &str, args: Value) -> ToolCall {
    ToolCall {
        id: Some(id.to_string()),
        name: name.to_string(),
        arguments: args,
    }
}

fn tool_result(id: &str, name: &str, content: &str) -> Message {
    let mut m = msg("tool", content);
    m.tool_call_id = Some(id.to_string());
    m.name = Some(name.to_string());
    m
}

fn conversations() -> Vec<(&'static str, RenderRequest)> {
    let mut out: Vec<(&'static str, RenderRequest)> = Vec::new();
    let base = |messages: Vec<Message>| RenderRequest {
        messages,
        add_generation_prompt: true,
        ..Default::default()
    };

    out.push((
        "system_user",
        base(vec![
            msg("system", "You are a terse assistant."),
            msg("user", "What is the capital of France?"),
        ]),
    ));
    out.push(("user_only", base(vec![msg("user", "Hello there.")])));
    out.push((
        "multi_turn",
        base(vec![
            msg("system", "You are a terse assistant."),
            msg("user", "Pick a number."),
            msg("assistant", "Seven."),
            msg("user", "Double it."),
        ]),
    ));
    let mut with_tools = base(vec![
        msg("system", "You can call tools."),
        msg("user", "Weather in Paris?"),
    ]);
    with_tools.tools = Some(tools());
    out.push(("tools_definitions", with_tools));

    let mut tool_loop = base(vec![
        msg("system", "You can call tools."),
        msg("user", "Weather in Paris?"),
        {
            let mut m = msg("assistant", "");
            m.tool_calls = Some(vec![call(
                "call_1",
                "get_weather",
                json!({"location": "Paris, France", "unit": "celsius"}),
            )]);
            m
        },
        tool_result(
            "call_1",
            "get_weather",
            "{\"temp_c\": 18, \"sky\": \"cloudy\"}",
        ),
    ]);
    tool_loop.tools = Some(tools());
    out.push(("tool_call_and_result", tool_loop));

    let mut reasoning = base(vec![
        msg("user", "Is 17 prime?"),
        {
            let mut m = msg("assistant", "Yes, 17 is prime.");
            m.reasoning_content = Some("17 has no divisors other than 1 and itself.".into());
            m
        },
        msg("user", "And 18?"),
        {
            let mut m = msg("assistant", "No, 18 = 2 × 9.");
            m.reasoning_content = Some("18 is even.".into());
            m
        },
    ]);
    reasoning.add_generation_prompt = false;
    out.push(("reasoning_content", reasoning));

    let mut think_on = base(vec![
        msg("system", "Think step by step."),
        msg("user", "2+2?"),
    ]);
    think_on.enable_thinking = Some(true);
    out.push(("enable_thinking_true", think_on));
    let mut think_off = base(vec![
        msg("system", "Think step by step."),
        msg("user", "2+2?"),
    ]);
    think_off.enable_thinking = Some(false);
    out.push(("enable_thinking_false", think_off));

    let mut no_prompt = base(vec![
        msg("system", "Be brief."),
        msg("user", "Say hi."),
        msg("assistant", "Hi."),
    ]);
    no_prompt.add_generation_prompt = false;
    out.push(("no_generation_prompt", no_prompt));

    out.push((
        "content_parts",
        base(vec![
            Message::new(
                "system",
                Content::Parts(vec![ContentPart::Text {
                    text: "Answer in haiku.".into(),
                }]),
            ),
            Message::new(
                "user",
                Content::Parts(vec![
                    ContentPart::Text {
                        text: "Describe ".into(),
                    },
                    ContentPart::Text {
                        text: "autumn.".into(),
                    },
                ]),
            ),
        ]),
    ));

    out.push((
        "unicode",
        base(vec![
            msg("system", "Répondez en français, s'il vous plaît — « toujours »."),
            msg(
                "user",
                "日本語でお願いします 🙏🏽 — tabs\tand \"quotes\" and back\\slashes and <tags> & ampersands; עברית",
            ),
        ]),
    ));
    out.push((
        "empty_system",
        base(vec![msg("system", ""), msg("user", "Anything?")]),
    ));

    let mut multi_step = base(vec![
        msg("system", "You can call tools."),
        msg("user", "Flights from LHR to JFK for two adults under 900?"),
        {
            let mut m = msg("assistant", "Let me check.");
            m.reasoning_content = Some("Need flight search.".into());
            m.tool_calls = Some(vec![call(
                "call_a",
                "search_flights",
                json!({
                    "route": {"from": "LHR", "to": "JFK", "via": []},
                    "passengers": [{"age": 34, "frequent_flyer": true}, {"age": 29, "frequent_flyer": false}],
                    "max_price": 899.5
                }),
            )]);
            m
        },
        tool_result(
            "call_a",
            "search_flights",
            "[{\"id\": \"BA117\", \"price\": 812.0}]",
        ),
        msg("assistant", "BA117 at 812."),
        msg("user", "What about the weather there?"),
    ]);
    multi_step.tools = Some(tools());
    multi_step.enable_thinking = Some(true);
    out.push(("multi_step_tool_loop", multi_step));

    let mut parallel = base(vec![
        msg("user", "Weather in Paris and Rome?"),
        {
            let mut m = msg("assistant", "");
            m.tool_calls = Some(vec![
                call(
                    "call_p",
                    "get_weather",
                    json!({"location": "Paris, France"}),
                ),
                call(
                    "call_r",
                    "get_weather",
                    json!({"location": "Rome, Italy", "days": 3}),
                ),
            ]);
            m
        },
        tool_result("call_p", "get_weather", "{\"temp_c\": 18}"),
        tool_result("call_r", "get_weather", "{\"temp_c\": 24}"),
    ]);
    parallel.tools = Some(tools());
    out.push(("parallel_tool_calls_no_system", parallel));

    let mut typed_args = base(vec![
        msg("user", "Book it."),
        {
            let mut m = msg("assistant", "");
            m.tool_calls = Some(vec![call(
                "call_t",
                "search_flights",
                json!({
                    "route": {"from": "A", "to": "B"},
                    "passengers": [{"age": 1, "frequent_flyer": false}],
                    "max_price": null,
                    "ratio": 0.00001,
                    "big": 1e21,
                    "flag": true,
                    "note": "line1\nline2 \"q\" é"
                }),
            )]);
            m
        },
        tool_result("call_t", "search_flights", "[]"),
    ]);
    typed_args.tools = Some(tools());
    out.push(("typed_arguments", typed_args));

    // Inputs some templates refuse, to check that raise_exception parity (message included).
    out.push((
        "system_not_first",
        base(vec![
            msg("user", "Hi"),
            msg("system", "Late system prompt."),
            msg("user", "Still there?"),
        ]),
    ));
    out.push((
        "tool_result_without_call",
        base(vec![
            msg("user", "Hi"),
            tool_result("call_x", "get_weather", "{\"temp_c\": 1}"),
        ]),
    ));

    out
}

#[derive(Debug, PartialEq)]
enum Side {
    Ok(Vec<u8>),
    Raised(String),
    Error(String),
}

fn python_render(python: &str, script: &Path, tpl_path: &Path, inputs_path: &Path) -> Side {
    let out = Command::new(python)
        .arg("-I")
        .arg(script)
        .arg(tpl_path)
        .arg(inputs_path)
        .output()
        .expect("spawn python");
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    match out.status.code() {
        Some(0) => Side::Ok(out.stdout),
        Some(3) => Side::Raised(stderr),
        _ => Side::Error(stderr),
    }
}

fn token_text(gguf: &GgufFile, id_key: &str) -> Option<String> {
    let id = gguf.get_u32(id_key)? as usize;
    let tokens = gguf.get_array("tokenizer.ggml.tokens")?;
    tokens.get(id)?.as_str().map(str::to_string)
}

fn expected_family(arch: &str) -> Option<TemplateFamily> {
    Some(match arch {
        "qwen3" => TemplateFamily::Hermes,
        "qwen35" => TemplateFamily::QwenXml,
        "gemma4" => TemplateFamily::Gemma4,
        "gpt-oss" => TemplateFamily::Harmony,
        "smollm3" => TemplateFamily::Hermes,
        "olmo2" => TemplateFamily::Olmo3,
        _ => return None,
    })
}

fn first_diff(a: &[u8], b: &[u8]) -> String {
    let idx = a
        .iter()
        .zip(b.iter())
        .position(|(x, y)| x != y)
        .unwrap_or(a.len().min(b.len()));
    let window = |s: &[u8]| {
        let start = idx.saturating_sub(60);
        let end = (idx + 60).min(s.len());
        String::from_utf8_lossy(&s[start..end]).to_string()
    };
    format!(
        "first difference at byte {idx} (rust len {}, python len {})\n  rust:   {:?}\n  python: {:?}",
        a.len(),
        b.len(),
        window(a),
        window(b)
    )
}

#[test]
fn jinja2_parity() {
    let Ok(gguf_list) = std::env::var("LLMARIO_TEST_GGUF") else {
        eprintln!("jinja2_parity: skipped (LLMARIO_TEST_GGUF not set)");
        return;
    };
    let Ok(python) = std::env::var("LLMARIO_PYTHON") else {
        eprintln!(
            "jinja2_parity: skipped (LLMARIO_PYTHON not set; point it at a python with jinja2)"
        );
        return;
    };
    let script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../scripts/engine/render_ref.py")
        .canonicalize()
        .expect("render_ref.py exists");
    let tmp: PathBuf = std::env::var_os("LLMARIO_TEST_TMP")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("llmario-chat-parity"));
    std::fs::create_dir_all(&tmp).expect("tmp dir");
    let now = DateTime::parse_from_rfc3339(FIXED_NOW).unwrap();

    let mut failures = Vec::new();
    for (model_idx, path) in gguf_list
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .enumerate()
    {
        let gguf = GgufFile::open(Path::new(path)).unwrap_or_else(|e| panic!("open {path}: {e}"));
        let arch = gguf.architecture().unwrap_or("?").to_string();
        let template = gguf
            .get_str("tokenizer.chat_template")
            .unwrap_or_else(|| panic!("{path}: no tokenizer.chat_template"))
            .to_string();
        let bos = token_text(&gguf, "tokenizer.ggml.bos_token_id");
        let eos = token_text(&gguf, "tokenizer.ggml.eos_token_id");
        drop(gguf);

        let tpl = ChatTemplate::with_now(&template, bos.as_deref(), eos.as_deref(), Some(now))
            .unwrap_or_else(|e| panic!("{path}: {e}"));
        let family = tpl.detect_family();
        println!(
            "== {} (arch {arch}, hash {}, family {}, bos {:?}, eos {:?})",
            Path::new(path).file_name().unwrap().to_string_lossy(),
            tpl.content_hash(),
            family.as_str(),
            bos,
            eos
        );
        if let Some(want) = expected_family(&arch) {
            if want != family {
                failures.push(format!(
                    "{arch}: family {} != expected {}",
                    family.as_str(),
                    want.as_str()
                ));
            }
            // The structural rules must agree with the hash table: a trailing comment changes
            // the hash without changing the structure.
            let mutated = format!("{template}{{# parity #}}");
            let by_markers = ChatTemplate::new(&mutated, None, None)
                .unwrap()
                .detect_family();
            if by_markers != want {
                failures.push(format!(
                    "{arch}: structural detection {} != expected {}",
                    by_markers.as_str(),
                    want.as_str()
                ));
            }
        }

        let tpl_path = tmp.join(format!("{model_idx}.jinja"));
        std::fs::write(&tpl_path, &template).unwrap();

        let (mut matched, mut both_raised, mut both_errored, mut mismatched) = (0, 0, 0, 0);
        for (name, req) in conversations() {
            let rust = match tpl.render_detailed(&req) {
                Ok(r) => (Side::Ok(r.text.into_bytes()), r.arguments_as_string),
                Err(ChatError::Raised(m)) => (Side::Raised(m), false),
                // Keep the first line only: minijinja appends a multi-line debug dump.
                Err(e) => (
                    Side::Error(e.to_string().lines().next().unwrap_or("").to_string()),
                    false,
                ),
            };
            let ctx: Map<String, Value> = tpl.context(&req, rust.1).unwrap();
            let inputs = json!({"context": ctx, "now": FIXED_NOW});
            let inputs_path = tmp.join(format!("{model_idx}-{name}.json"));
            std::fs::write(&inputs_path, serde_json::to_vec(&inputs).unwrap()).unwrap();
            let py = python_render(&python, &script, &tpl_path, &inputs_path);

            let verdict = match (&rust.0, &py) {
                (Side::Ok(a), Side::Ok(b)) if a == b => {
                    matched += 1;
                    format!(
                        "ok ({} bytes{})",
                        a.len(),
                        if rust.1 { ", string args" } else { "" }
                    )
                }
                (Side::Raised(a), Side::Raised(b)) => {
                    both_raised += 1;
                    if a == b {
                        format!("both raised: {a}")
                    } else {
                        format!("both raised (messages differ): rust={a:?} python={b:?}")
                    }
                }
                (Side::Error(a), Side::Error(b)) => {
                    both_errored += 1;
                    format!("both errored: rust={a} python={b}")
                }
                (Side::Ok(a), Side::Ok(b)) => {
                    mismatched += 1;
                    let d = first_diff(a, b);
                    failures.push(format!("{arch}/{name}: {d}"));
                    format!("MISMATCH {d}")
                }
                (r, p) => {
                    mismatched += 1;
                    let d = format!("rust={r:?} python={p:?}");
                    failures.push(format!("{arch}/{name}: {d}"));
                    format!("MISMATCH {d}")
                }
            };
            println!("  {name:<32} {verdict}");
        }
        println!(
            "  summary: {matched} matched, {both_raised} raised on both sides, {both_errored} errored on both sides, {mismatched} mismatched"
        );
    }
    assert!(
        failures.is_empty(),
        "parity failures:\n{}",
        failures.join("\n")
    );
}
