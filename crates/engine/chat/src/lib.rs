//! Native engine `chat` crate: renders OpenAI-style chat messages into a prompt string with the
//! model's own Jinja chat template, byte-identical to what Python Jinja2 produces through
//! `transformers.apply_chat_template`.
//!
//! * [`ChatTemplate::new`] compiles a template with the HuggingFace environment (see `env`).
//! * [`ChatTemplate::render`] renders a [`RenderRequest`]. Messages are exposed to the template
//!   as plain JSON-like maps exactly as transformers passes them; tool-call `arguments` are a
//!   JSON object (mapping) by default, with an automatic fallback to the string form for
//!   templates that raise on mappings.
//! * [`ChatTemplate::content_hash`] / [`ChatTemplate::detect_family`] identify the template.
//!
//! Only rendering lives here; parsing model output back into tool calls is the decode side's job.

mod env;
mod family;
mod pyfmt;

pub use family::TemplateFamily;

use chrono::{DateTime, FixedOffset};
use minijinja::value::Value as JValue;
use minijinja::Environment;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

const TEMPLATE_NAME: &str = "chat_template";

/// Errors from compiling or rendering a chat template.
#[derive(Debug, thiserror::Error)]
pub enum ChatError {
    /// The template does not parse/compile.
    #[error("chat template failed to compile: {0}")]
    Compile(String),
    /// The template called `raise_exception(...)`: it refuses this input (bad role order,
    /// string arguments, images in a system message, ...). The payload is the template's message.
    #[error("chat template raised: {0}")]
    Raised(String),
    /// Any other runtime failure (unknown filter, type error, ...).
    #[error("chat template failed to render: {0}")]
    Render(String),
    /// The request itself is malformed.
    #[error("invalid render request: {0}")]
    InvalidRequest(String),
}

impl From<ChatError> for llmario_engine_core::EngineError {
    fn from(e: ChatError) -> Self {
        llmario_engine_core::EngineError::Format(e.to_string())
    }
}

/// One content part of a multi-part message (`{"type": "text", "text": ...}`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
}

/// Message content: a string, a list of parts, or JSON `null` (an assistant turn that only
/// carries tool calls). Exposed to the template unchanged.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
    Null,
}

impl Default for Content {
    fn default() -> Self {
        Content::Text(String::new())
    }
}

impl From<&str> for Content {
    fn from(s: &str) -> Self {
        Content::Text(s.to_string())
    }
}

impl From<String> for Content {
    fn from(s: String) -> Self {
        Content::Text(s)
    }
}

impl Content {
    fn to_json(&self) -> Value {
        match self {
            Content::Text(s) => Value::String(s.clone()),
            Content::Parts(parts) => Value::Array(
                parts
                    .iter()
                    .map(|p| serde_json::to_value(p).unwrap_or(Value::Null))
                    .collect(),
            ),
            Content::Null => Value::Null,
        }
    }
}

/// A tool call on an assistant message. `arguments` is kept as a JSON value (normally an
/// object); the renderer decides whether the template sees a mapping or its JSON text.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

/// An OpenAI Chat Completions style message.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(default)]
    pub content: Content,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

impl Message {
    pub fn new(role: &str, content: impl Into<Content>) -> Message {
        Message {
            role: role.to_string(),
            content: content.into(),
            ..Default::default()
        }
    }

    /// The map the template sees: `role`, `content`, then `name`, `tool_calls`,
    /// `tool_call_id`, `reasoning_content` when present. Tool calls are rendered in the OpenAI
    /// shape `{"id", "type": "function", "function": {"name", "arguments"}}`.
    fn to_json(&self, arguments_as_string: bool) -> Result<Value, ChatError> {
        let mut m = Map::new();
        m.insert("role".into(), Value::String(self.role.clone()));
        m.insert("content".into(), self.content.to_json());
        if let Some(name) = &self.name {
            m.insert("name".into(), Value::String(name.clone()));
        }
        if let Some(calls) = &self.tool_calls {
            let mut list = Vec::with_capacity(calls.len());
            for call in calls {
                let mut function = Map::new();
                function.insert("name".into(), Value::String(call.name.clone()));
                function.insert(
                    "arguments".into(),
                    arguments_json(&call.arguments, arguments_as_string)?,
                );
                let mut tc = Map::new();
                if let Some(id) = &call.id {
                    tc.insert("id".into(), Value::String(id.clone()));
                }
                tc.insert("type".into(), Value::String("function".into()));
                tc.insert("function".into(), Value::Object(function));
                list.push(Value::Object(tc));
            }
            m.insert("tool_calls".into(), Value::Array(list));
        }
        if let Some(id) = &self.tool_call_id {
            m.insert("tool_call_id".into(), Value::String(id.clone()));
        }
        if let Some(r) = &self.reasoning_content {
            m.insert("reasoning_content".into(), Value::String(r.clone()));
        }
        Ok(Value::Object(m))
    }
}

/// Object form: a JSON string that parses as JSON is unwrapped; anything else passes through.
/// String form: a string passes through; anything else is serialised compactly.
fn arguments_json(args: &Value, as_string: bool) -> Result<Value, ChatError> {
    if as_string {
        return Ok(match args {
            Value::String(_) => args.clone(),
            other => Value::String(serde_json::to_string(other).map_err(|e| {
                ChatError::InvalidRequest(format!("tool call arguments not serialisable: {e}"))
            })?),
        });
    }
    Ok(match args {
        Value::String(s) => serde_json::from_str::<Value>(s).unwrap_or_else(|_| args.clone()),
        other => other.clone(),
    })
}

/// How tool-call `arguments` are exposed to the template.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ArgumentsMode {
    /// Try the mapping form; if the template raises an error mentioning arguments, retry with
    /// the string form.
    #[default]
    Auto,
    /// Always a JSON object (Qwen3.5, Gemma 4, LFM2 raise on strings).
    Object,
    /// Always the JSON text (templates that print `arguments` verbatim).
    String,
}

/// Everything a render needs besides the template.
#[derive(Clone, Debug, Default)]
pub struct RenderRequest {
    pub messages: Vec<Message>,
    pub add_generation_prompt: bool,
    /// OpenAI tool definitions (`[{"type": "function", "function": {...}}, ...]`), passed to
    /// the template as `tools`; `None` renders `tools` as Python `None`, as transformers does.
    pub tools: Option<Value>,
    /// Passed as `enable_thinking` only when `Some`, so `enable_thinking is defined` matches
    /// whether the caller set it.
    pub enable_thinking: Option<bool>,
    /// Extra template variables (`reasoning_effort`, `preserve_thinking`, `xml_tools`, ...).
    /// They cannot replace `messages`, `tools` or `add_generation_prompt`.
    pub extra: Map<String, Value>,
    pub arguments_mode: ArgumentsMode,
}

/// The result of [`ChatTemplate::render_detailed`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rendered {
    pub text: String,
    /// Whether the fallback (string) form of tool-call arguments was used.
    pub arguments_as_string: bool,
}

/// A compiled chat template.
pub struct ChatTemplate {
    source: String,
    env: Environment<'static>,
    bos_token: Option<String>,
    eos_token: Option<String>,
    hash: String,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplate")
            .field("hash", &self.hash)
            .field("bos_token", &self.bos_token)
            .field("eos_token", &self.eos_token)
            .field("len", &self.source.len())
            .finish()
    }
}

impl ChatTemplate {
    /// Compile `template`. `bos_token`/`eos_token` are exposed as template variables when given
    /// (transformers passes the tokenizer's special tokens the same way).
    pub fn new(
        template: &str,
        bos_token: Option<&str>,
        eos_token: Option<&str>,
    ) -> Result<ChatTemplate, ChatError> {
        ChatTemplate::with_now(template, bos_token, eos_token, None)
    }

    /// Like [`ChatTemplate::new`] with a fixed clock for `strftime_now` (tests, reproducible
    /// prompts). `None` uses the local wall clock like Python's `datetime.now()`.
    pub fn with_now(
        template: &str,
        bos_token: Option<&str>,
        eos_token: Option<&str>,
        now: Option<DateTime<FixedOffset>>,
    ) -> Result<ChatTemplate, ChatError> {
        let mut env = env::build_environment(now);
        let compiled_source = env::rewrite_generation_tags(template);
        env.add_template_owned(TEMPLATE_NAME.to_string(), compiled_source)
            .map_err(|e| ChatError::Compile(e.to_string()))?;
        let hash = format!("{:x}", Sha256::digest(template.as_bytes()));
        Ok(ChatTemplate {
            source: template.to_string(),
            env,
            bos_token: bos_token.map(str::to_string),
            eos_token: eos_token.map(str::to_string),
            hash,
        })
    }

    /// The template text exactly as given.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// SHA-256 (lowercase hex) of the template text.
    pub fn content_hash(&self) -> String {
        self.hash.clone()
    }

    /// Tool-call family, by known hash first and structural markers second (see `family`).
    pub fn detect_family(&self) -> TemplateFamily {
        family::detect(&self.source, &self.hash)
    }

    /// Render to the prompt string.
    pub fn render(&self, req: &RenderRequest) -> Result<String, ChatError> {
        self.render_detailed(req).map(|r| r.text)
    }

    /// Render and report which arguments form was used.
    pub fn render_detailed(&self, req: &RenderRequest) -> Result<Rendered, ChatError> {
        match req.arguments_mode {
            ArgumentsMode::Object => self.render_with(req, false),
            ArgumentsMode::String => self.render_with(req, true),
            ArgumentsMode::Auto => match self.render_with(req, false) {
                Err(ChatError::Raised(msg))
                    if msg.to_ascii_lowercase().contains("argument")
                        && req.messages.iter().any(|m| m.tool_calls.is_some()) =>
                {
                    self.render_with(req, true)
                }
                other => other,
            },
        }
    }

    fn render_with(
        &self,
        req: &RenderRequest,
        args_as_string: bool,
    ) -> Result<Rendered, ChatError> {
        let ctx = self.context(req, args_as_string)?;
        let text = self.render_context(&ctx)?;
        Ok(Rendered {
            text,
            arguments_as_string: args_as_string,
        })
    }

    /// The exact variable set handed to the template (what `transformers` passes as
    /// `render(messages=..., tools=..., documents=None, add_generation_prompt=..., **kwargs)`).
    /// Public so a reference renderer can be fed identical inputs.
    pub fn context(
        &self,
        req: &RenderRequest,
        arguments_as_string: bool,
    ) -> Result<Map<String, Value>, ChatError> {
        let mut ctx = Map::new();
        if let Some(bos) = &self.bos_token {
            ctx.insert("bos_token".into(), Value::String(bos.clone()));
        }
        if let Some(eos) = &self.eos_token {
            ctx.insert("eos_token".into(), Value::String(eos.clone()));
        }
        if let Some(b) = req.enable_thinking {
            ctx.insert("enable_thinking".into(), Value::Bool(b));
        }
        ctx.insert("documents".into(), Value::Null);
        for (k, v) in &req.extra {
            ctx.insert(k.clone(), v.clone());
        }
        let messages = req
            .messages
            .iter()
            .map(|m| m.to_json(arguments_as_string))
            .collect::<Result<Vec<_>, _>>()?;
        ctx.insert("messages".into(), Value::Array(messages));
        ctx.insert("tools".into(), req.tools.clone().unwrap_or(Value::Null));
        ctx.insert(
            "add_generation_prompt".into(),
            Value::Bool(req.add_generation_prompt),
        );
        Ok(ctx)
    }

    /// Render a prepared context (see [`ChatTemplate::context`]).
    pub fn render_context(&self, ctx: &Map<String, Value>) -> Result<String, ChatError> {
        let template = self
            .env
            .get_template(TEMPLATE_NAME)
            .map_err(|e| ChatError::Compile(e.to_string()))?;
        template
            .render(JValue::from_serialize(ctx))
            .map_err(|e| match env::raised_message(&e) {
                Some(msg) => ChatError::Raised(msg),
                None => ChatError::Render(format!("{e:#}")),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tpl(src: &str) -> ChatTemplate {
        ChatTemplate::new(src, Some("<s>"), Some("</s>")).expect("compiles")
    }

    fn req(messages: Vec<Message>) -> RenderRequest {
        RenderRequest {
            messages,
            add_generation_prompt: true,
            ..Default::default()
        }
    }

    #[test]
    fn chatml_round_trip() {
        let t = tpl(
            "{{ bos_token }}{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
        );
        let out = t
            .render(&req(vec![
                Message::new("system", "Be brief."),
                Message::new("user", "Hi"),
            ]))
            .unwrap();
        assert_eq!(
            out,
            "<s><|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn trim_and_lstrip_blocks_match_jinja2() {
        // With trim_blocks the newline after `%}` is dropped; with lstrip_blocks the indentation
        // before `{%` is dropped; `{{ }}` lines keep both.
        let t = tpl("{% for m in messages %}\n    {% if true %}\n  {{ m.role }}\n    {% endif %}\n{% endfor %}\n");
        let out = t.render(&req(vec![Message::new("user", "x")])).unwrap();
        assert_eq!(out, "  user\n");
    }

    #[test]
    fn python_literals_and_methods() {
        let t = tpl(
            "{{ true }}|{{ none }}|{{ 1e-5 }}|{{ 'a,b'.split(',') }}|{{ ' x '.strip() }}|{{ {'k': [1, 2.0, True, None]} | string }}|{{ [1, true, 'x'] | join(', ') }}",
        );
        assert_eq!(
            t.render(&req(vec![])).unwrap(),
            "True|None|1e-05|['a', 'b']|x|{'k': [1, 2.0, True, None]}|1, True, x"
        );
    }

    #[test]
    fn tojson_uses_python_spacing() {
        let t = tpl("{{ tools | tojson }}|{{ messages[0].content | tojson }}|{{ {'a': 1} | tojson(indent=2) }}");
        let mut r = req(vec![Message::new("user", "héllo \"q\"")]);
        r.tools = Some(
            serde_json::json!([{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]),
        );
        assert_eq!(
            t.render(&r).unwrap(),
            "[{\"type\": \"function\", \"function\": {\"name\": \"f\", \"parameters\": {\"type\": \"object\", \"properties\": {}}}}]|\"héllo \\\"q\\\"\"|{\n  \"a\": 1\n}"
        );
    }

    #[test]
    fn raise_exception_is_reported_as_raised() {
        let t = tpl("{{ raise_exception('nope: ' ~ messages[0].role) }}");
        match t.render(&req(vec![Message::new("tool", "")])) {
            Err(ChatError::Raised(m)) => assert_eq!(m, "nope: tool"),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn strftime_now_is_fixed_when_requested() {
        let now = DateTime::parse_from_rfc3339("2026-03-04T05:06:07+00:00").unwrap();
        let t = ChatTemplate::with_now(
            "{{ strftime_now('%d %B %Y') }}|{{ strftime_now('%Y-%m-%d') }}",
            None,
            None,
            Some(now),
        )
        .unwrap();
        assert_eq!(t.render(&req(vec![])).unwrap(), "04 March 2026|2026-03-04");
    }

    #[test]
    fn generation_tags_render_their_body() {
        let t = tpl(
            "{% for m in messages %}{% generation %}{{ m.content }}{% endgeneration %}{% endfor %}",
        );
        assert_eq!(
            t.render(&req(vec![Message::new("assistant", "ok")]))
                .unwrap(),
            "ok"
        );
    }

    #[test]
    fn undefined_is_lenient_like_jinja2() {
        let t = tpl("{% if messages[0].role == 'system' %}S{% endif %}{% if enable_thinking is defined %}T{% endif %}{% if 'x' in missing %}M{% endif %}{{ missing }}|{{ tools is none }}");
        assert_eq!(
            t.render(&req(vec![Message::new("user", "")])).unwrap(),
            "|True"
        );
        let mut r = req(vec![Message::new("user", "")]);
        r.enable_thinking = Some(false);
        assert_eq!(t.render(&r).unwrap(), "T|True");
    }

    #[test]
    fn arguments_fall_back_to_string_when_template_raises() {
        let t = tpl(
            "{% for m in messages %}{% if m.tool_calls %}{% for c in m.tool_calls %}{% if c.function.arguments is not string %}{{ raise_exception('arguments must be a string') }}{% endif %}{{ c.function.name }}:{{ c.function.arguments }}{% endfor %}{% endif %}{% endfor %}",
        );
        let mut m = Message::new("assistant", "");
        m.tool_calls = Some(vec![ToolCall {
            id: Some("call_1".into()),
            name: "f".into(),
            arguments: serde_json::json!({"a": 1, "b": "x"}),
        }]);
        let r = req(vec![m]);
        let rendered = t.render_detailed(&r).unwrap();
        assert!(rendered.arguments_as_string);
        assert_eq!(rendered.text, "f:{\"a\":1,\"b\":\"x\"}");
        let mut strict = r.clone();
        strict.arguments_mode = ArgumentsMode::Object;
        assert!(matches!(t.render(&strict), Err(ChatError::Raised(_))));
    }

    #[test]
    fn arguments_object_form_keeps_insertion_order() {
        let t = tpl(
            "{% for c in messages[0].tool_calls %}{{ c.function.arguments | tojson }}{% endfor %}",
        );
        let mut m = Message::new("assistant", Content::Null);
        m.tool_calls = Some(vec![ToolCall {
            id: None,
            name: "f".into(),
            arguments: Value::String("{\"z\": 1, \"a\": [true, null]}".into()),
        }]);
        assert_eq!(
            t.render(&req(vec![m])).unwrap(),
            "{\"z\": 1, \"a\": [true, null]}"
        );
    }

    #[test]
    fn content_parts_and_null_are_exposed_as_given() {
        let t = tpl("{% for m in messages %}{% if m.content is string %}S{% elif m.content is none %}N{% else %}{% for p in m.content %}{{ p.type }}={{ p.text }}{% endfor %}{% endif %}{% endfor %}");
        let msgs = vec![
            Message::new("user", "s"),
            Message::new("assistant", Content::Null),
            Message::new(
                "user",
                Content::Parts(vec![ContentPart::Text { text: "hi".into() }]),
            ),
        ];
        assert_eq!(t.render(&req(msgs)).unwrap(), "SNtext=hi");
    }

    #[test]
    fn hash_and_family() {
        let t = tpl("{{ x }}");
        assert_eq!(
            t.content_hash(),
            format!("{:x}", Sha256::digest(b"{{ x }}"))
        );
        assert_eq!(t.content_hash().len(), 64);
        assert_eq!(t.detect_family(), TemplateFamily::Unknown);
        let hermes = tpl("<tools>{{ tool | tojson }}</tools><tool_call>{}</tool_call>");
        assert_eq!(hermes.detect_family(), TemplateFamily::Hermes);
    }

    #[test]
    fn tilde_concat_and_strftime_modifiers_match_python() {
        // `~` operands go through the configured formatter, so booleans/none/floats/lists
        // concatenate with Python `str()` semantics; chrono accepts the `%-d` padding modifier.
        let now = DateTime::parse_from_rfc3339("2026-03-04T05:06:07+00:00").unwrap();
        let t = ChatTemplate::with_now(
            "{{ true ~ '|' ~ none ~ '|' ~ 1.0 ~ '|' ~ [1, true] }}|{{ strftime_now('%-d %b %Y|%j|%A') }}",
            None,
            None,
            Some(now),
        )
        .unwrap();
        let out = t.render(&req(vec![])).unwrap();
        // Verified against Python Jinja2 with the same inputs (scripts/engine/render_ref.py).
        assert_eq!(out, "True|None|1.0|[1, True]|4 Mar 2026|063|Wednesday");
    }

    #[test]
    fn compile_errors_are_reported() {
        assert!(matches!(
            ChatTemplate::new("{% if %}", None, None),
            Err(ChatError::Compile(_))
        ));
    }

    #[test]
    fn message_json_shape_matches_openai() {
        let mut m = Message::new("assistant", "c");
        m.tool_calls = Some(vec![ToolCall {
            id: Some("id1".into()),
            name: "f".into(),
            arguments: serde_json::json!({"q": "x"}),
        }]);
        m.reasoning_content = Some("why".into());
        let v = m.to_json(false).unwrap();
        assert_eq!(
            serde_json::to_string(&v).unwrap(),
            "{\"role\":\"assistant\",\"content\":\"c\",\"tool_calls\":[{\"id\":\"id1\",\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":{\"q\":\"x\"}}}],\"reasoning_content\":\"why\"}"
        );
        let mut t = Message::new("tool", "42");
        t.tool_call_id = Some("id1".into());
        t.name = Some("f".into());
        assert_eq!(
            serde_json::to_string(&t.to_json(false).unwrap()).unwrap(),
            "{\"role\":\"tool\",\"content\":\"42\",\"name\":\"f\",\"tool_call_id\":\"id1\"}"
        );
    }
}
