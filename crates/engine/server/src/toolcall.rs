//! Tool calling and reasoning on the OpenAI Chat Completions surface (Architecture §10.1–§10.3).
//!
//! - [`to_chat_messages`] converts OpenAI request messages (assistant `tool_calls` with JSON-text
//!   `arguments`, `tool` results with `tool_call_id`) into the chat crate's message model, where
//!   arguments are JSON objects because several templates raise on strings.
//! - [`Assembler`] runs the family's streaming [`OutputParser`] over the generated text and turns
//!   its events into OpenAI deltas (`content`, `reasoning_content`, `tool_calls`) for streams, or a
//!   final assistant message for non-streaming responses.
//! - [`tool_grammar`] maps `tool_choice` onto a constrained-decoding grammar for the families the
//!   decode crate has grammars for; other families still parse calls, just without enforcement.

use llmario_engine_chat::{
    Content, Message, OutputEvent, OutputParser, ParserOptions, TemplateFamily, ToolCall,
};
use llmario_engine_decode::{GrammarSpec, ToolCallSpec, ToolChoice, ToolDef, ToolFamily};
use serde_json::{json, Map, Value};

/// Convert OpenAI-format messages into the chat crate's model.
pub fn to_chat_messages(raw: &[Value]) -> Result<Vec<Message>, String> {
    let mut out = Vec::with_capacity(raw.len());
    for (i, m) in raw.iter().enumerate() {
        let role = m
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("messages[{i}].role is required"))?;
        let content: Content = match m.get("content") {
            // OpenAI clients send `content: null` on assistant turns that only carry tool calls.
            // Templates such as Qwen3's test `'</think>' in message.content`, which fails on
            // None, so null becomes "" (llama.cpp normalises it the same way).
            None | Some(Value::Null) => Content::Text(String::new()),
            Some(v) => serde_json::from_value(v.clone())
                .map_err(|e| format!("messages[{i}].content: {e}"))?,
        };
        let mut msg = Message::new(role, content);
        msg.name = m.get("name").and_then(Value::as_str).map(str::to_string);
        msg.tool_call_id = m
            .get("tool_call_id")
            .and_then(Value::as_str)
            .map(str::to_string);
        msg.reasoning_content = m
            .get("reasoning_content")
            .and_then(Value::as_str)
            .map(str::to_string);
        if let Some(calls) = m.get("tool_calls").and_then(Value::as_array) {
            let mut v = Vec::with_capacity(calls.len());
            for (j, c) in calls.iter().enumerate() {
                let f = c.get("function").unwrap_or(c);
                let name = f.get("name").and_then(Value::as_str).ok_or_else(|| {
                    format!("messages[{i}].tool_calls[{j}].function.name is required")
                })?;
                let arguments = match f.get("arguments") {
                    // OpenAI sends the arguments as JSON text.
                    Some(Value::String(s)) if s.trim().is_empty() => json!({}),
                    Some(Value::String(s)) => serde_json::from_str(s).map_err(|e| {
                        format!("messages[{i}].tool_calls[{j}].function.arguments is not JSON: {e}")
                    })?,
                    Some(other) => other.clone(),
                    None => json!({}),
                };
                v.push(ToolCall {
                    id: c.get("id").and_then(Value::as_str).map(str::to_string),
                    name: name.to_string(),
                    arguments,
                });
            }
            if !v.is_empty() {
                msg.tool_calls = Some(v);
            }
        }
        out.push(msg);
    }
    Ok(out)
}

/// OpenAI `tool_choice`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Choice {
    None,
    Auto,
    Required,
    Named(String),
}

pub fn parse_tool_choice(v: Option<&Value>) -> Result<Choice, String> {
    match v {
        None | Some(Value::Null) => Ok(Choice::Auto),
        Some(Value::String(s)) => match s.as_str() {
            "none" => Ok(Choice::None),
            "auto" => Ok(Choice::Auto),
            "required" => Ok(Choice::Required),
            other => Err(format!("unknown tool_choice '{other}'")),
        },
        Some(o) => o
            .pointer("/function/name")
            .and_then(Value::as_str)
            .map(|n| Choice::Named(n.to_string()))
            .ok_or_else(|| {
                "tool_choice object must be {\"type\":\"function\",\"function\":{\"name\":...}}"
                    .into()
            }),
    }
}

fn decode_family(f: TemplateFamily) -> Option<ToolFamily> {
    Some(match f {
        TemplateFamily::Hermes | TemplateFamily::ChatMl => ToolFamily::Hermes,
        TemplateFamily::QwenXml => ToolFamily::QwenXml,
        TemplateFamily::Gemma4 => ToolFamily::Gemma4,
        TemplateFamily::Mistral => ToolFamily::Mistral,
        TemplateFamily::Harmony => ToolFamily::Harmony,
        // Llama 3 JSON and Llama 4 pythonic share a template family here, and GLM / LFM2 /
        // OLMo 3 have no grammar yet: their calls are parsed but not enforced.
        _ => return None,
    })
}

/// Grammar and lazy trigger for a tool-call request, when the family has a grammar.
///
/// `auto` constrains only after the family's call opener appears (the grammar's own trigger);
/// `required`/named force a call — after the reasoning block when the model will reason first
/// (`reasoning_close` is the family's closing marker in that case).
pub fn tool_grammar(
    family: TemplateFamily,
    tools: &Value,
    choice: &Choice,
    reasoning_close: Option<&str>,
) -> Option<(GrammarSpec, Option<String>)> {
    let fam = decode_family(family)?;
    let defs: Vec<ToolDef> = tools
        .as_array()?
        .iter()
        .filter_map(|t| {
            let f = t.get("function").unwrap_or(t);
            let name = f.get("name")?.as_str()?;
            let params = f
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| json!({"type": "object"}));
            Some(ToolDef::new(name, params))
        })
        .collect();
    if defs.is_empty() {
        return None;
    }
    let choice = match choice {
        Choice::None => return None,
        Choice::Auto => ToolChoice::Auto,
        Choice::Required => ToolChoice::Required,
        Choice::Named(n) => ToolChoice::Named(n.clone()),
    };
    let lazy = match choice {
        ToolChoice::Auto => None,
        _ => reasoning_close.map(str::to_string),
    };
    Some((
        GrammarSpec::ToolCalls(ToolCallSpec {
            family: fam,
            tools: defs,
            choice,
        }),
        lazy,
    ))
}

/// Whether the model will open a reasoning block of its own after `prompt`: the family has
/// markers, thinking was not switched off, and the prompt does not already end with a closed
/// (empty) reasoning block. Returns the closing marker in that case.
pub fn expected_reasoning_close(
    family: TemplateFamily,
    prompt: &str,
    enable_thinking: Option<bool>,
) -> Option<&'static str> {
    if enable_thinking == Some(false) {
        return None;
    }
    let (_, close) = OutputParser::reasoning_markers(family)?;
    if prompt.trim_end().ends_with(close.trim_end()) {
        return None;
    }
    Some(close)
}

/// Turns parser events into OpenAI output.
pub struct Assembler {
    parser: OutputParser,
    content: String,
    reasoning: String,
    calls: Vec<Value>,
    any_call: bool,
}

impl Assembler {
    pub fn new(family: TemplateFamily, tools: Option<&Value>, prompt: &str) -> Assembler {
        let opts = ParserOptions {
            reasoning_open: OutputParser::prompt_opens_reasoning(family, prompt),
        };
        Assembler {
            parser: OutputParser::with_options(family, tools, opts),
            content: String::new(),
            reasoning: String::new(),
            calls: Vec::new(),
            any_call: false,
        }
    }

    /// Feed generated text; returns stream deltas (each a `choices[0].delta` object).
    pub fn push(&mut self, text: &str) -> Vec<Value> {
        let ev = self.parser.push(text);
        self.apply(ev)
    }

    /// End of generation; returns the last deltas.
    pub fn finish(&mut self) -> Vec<Value> {
        let ev = self.parser.finish();
        self.apply(ev)
    }

    pub fn has_tool_calls(&self) -> bool {
        self.any_call
    }

    /// The finish reason to report: `tool_calls` when a call completed.
    pub fn finish_reason<'a>(&self, engine_reason: &'a str) -> &'a str {
        if self.any_call {
            "tool_calls"
        } else {
            engine_reason
        }
    }

    /// The final assistant message for non-streaming responses.
    pub fn message(&self) -> Value {
        let mut m = Map::new();
        m.insert("role".into(), json!("assistant"));
        m.insert(
            "content".into(),
            if self.content.is_empty() && self.any_call {
                Value::Null
            } else {
                json!(self.content)
            },
        );
        if !self.reasoning.is_empty() {
            m.insert("reasoning_content".into(), json!(self.reasoning));
        }
        if self.any_call {
            m.insert("tool_calls".into(), Value::Array(self.calls.clone()));
        }
        Value::Object(m)
    }

    fn apply(&mut self, events: Vec<OutputEvent>) -> Vec<Value> {
        let mut out = Vec::new();
        for e in events {
            match e {
                OutputEvent::Content(t) => {
                    // The blank lines a template puts between the reasoning block and the
                    // answer are not part of the answer (llama.cpp trims them too).
                    let t = if self.content.is_empty() && !self.reasoning.is_empty() {
                        t.trim_start().to_string()
                    } else {
                        t
                    };
                    if !t.is_empty() {
                        self.content.push_str(&t);
                        out.push(json!({"content": t}));
                    }
                }
                OutputEvent::Reasoning(t) => {
                    if !t.is_empty() {
                        self.reasoning.push_str(&t);
                        out.push(json!({"reasoning_content": t}));
                    }
                }
                OutputEvent::ToolCallStart { index, id, name } => {
                    out.push(json!({"tool_calls": [{
                        "index": index, "id": id, "type": "function",
                        "function": {"name": name, "arguments": ""}
                    }]}));
                    while self.calls.len() <= index {
                        self.calls.push(Value::Null);
                    }
                    self.calls[index] = json!({
                        "id": id, "type": "function",
                        "function": {"name": name, "arguments": ""}
                    });
                }
                OutputEvent::ToolCallArgumentsDelta { index, delta } => {
                    out.push(json!({"tool_calls": [{
                        "index": index, "function": {"arguments": delta}
                    }]}));
                }
                OutputEvent::ToolCallEnd { index, arguments } => {
                    if let Some(c) = self.calls.get_mut(index) {
                        c["function"]["arguments"] =
                            json!(serde_json::to_string(&arguments).unwrap_or_default());
                        self.any_call = true;
                    }
                }
                OutputEvent::Invalid { reason } => {
                    tracing::debug!(%reason, "model output did not parse as a tool call");
                }
            }
        }
        // A started call that never ended was cancelled by the parser (Invalid); drop it.
        if !out.is_empty() || self.any_call {
            self.calls.retain(|c| !c.is_null());
        }
        out
    }

    /// Calls that started but never completed (for the non-streaming message).
    pub fn prune_incomplete(&mut self) {
        let complete: Vec<Value> = self
            .calls
            .iter()
            .filter(|c| {
                c.pointer("/function/arguments")
                    .and_then(Value::as_str)
                    .is_some_and(|s| !s.is_empty())
            })
            .cloned()
            .collect();
        self.calls = complete;
        self.any_call = !self.calls.is_empty();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_openai_tool_messages() {
        let raw = vec![
            json!({"role": "user", "content": "weather?"}),
            json!({"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
            }]}),
            json!({"role": "tool", "tool_call_id": "call_1", "content": "18C"}),
        ];
        let m = to_chat_messages(&raw).unwrap();
        assert_eq!(m.len(), 3);
        let calls = m[1].tool_calls.as_ref().unwrap();
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, json!({"city": "Paris"}));
        assert_eq!(m[2].tool_call_id.as_deref(), Some("call_1"));
        assert!(to_chat_messages(&[json!({"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "{bad"}}]})]).is_err());
    }

    #[test]
    fn tool_choice_parsing() {
        assert_eq!(parse_tool_choice(None).unwrap(), Choice::Auto);
        assert_eq!(
            parse_tool_choice(Some(&json!("required"))).unwrap(),
            Choice::Required
        );
        assert_eq!(
            parse_tool_choice(Some(
                &json!({"type": "function", "function": {"name": "f"}})
            ))
            .unwrap(),
            Choice::Named("f".into())
        );
        assert!(parse_tool_choice(Some(&json!("sometimes"))).is_err());
    }

    #[test]
    fn hermes_call_becomes_openai_tool_call() {
        let tools = json!([{"type": "function", "function": {"name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]);
        let mut a = Assembler::new(
            TemplateFamily::Hermes,
            Some(&tools),
            "<|im_start|>assistant\n",
        );
        let mut deltas = Vec::new();
        for chunk in ["<think>\nneed weather\n</think>\n\n<tool_", "call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"] {
            deltas.extend(a.push(chunk));
        }
        deltas.extend(a.finish());
        assert!(a.has_tool_calls());
        assert_eq!(a.finish_reason("stop"), "tool_calls");
        let msg = a.message();
        assert_eq!(
            msg["reasoning_content"].as_str().unwrap().trim(),
            "need weather"
        );
        let args: Value = serde_json::from_str(
            msg["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(args, json!({"city": "Paris"}));
        assert_eq!(msg["tool_calls"][0]["function"]["name"], "get_weather");
        // Streamed argument deltas reassemble to the same JSON.
        let streamed: String = deltas
            .iter()
            .filter_map(|d| {
                d.pointer("/tool_calls/0/function/arguments")
                    .and_then(Value::as_str)
            })
            .collect();
        assert_eq!(
            serde_json::from_str::<Value>(&streamed).unwrap(),
            json!({"city": "Paris"})
        );
    }

    #[test]
    fn plain_answer_has_no_calls() {
        let mut a = Assembler::new(TemplateFamily::Hermes, None, "");
        a.push("<think>\nok\n</think>\n\nParis.");
        a.finish();
        assert!(!a.has_tool_calls());
        assert_eq!(a.message()["content"], "Paris.");
        assert_eq!(a.finish_reason("stop"), "stop");
    }

    #[test]
    fn grammar_only_for_supported_families() {
        let tools = json!([{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]);
        assert!(tool_grammar(
            TemplateFamily::Hermes,
            &tools,
            &Choice::Required,
            Some("</think>")
        )
        .is_some());
        assert!(tool_grammar(TemplateFamily::Glm, &tools, &Choice::Required, None).is_none());
        assert!(tool_grammar(TemplateFamily::Hermes, &tools, &Choice::None, None).is_none());
        let (_, lazy) = tool_grammar(
            TemplateFamily::Hermes,
            &tools,
            &Choice::Auto,
            Some("</think>"),
        )
        .unwrap();
        assert_eq!(lazy, None, "auto relies on the opener trigger");
    }
}
