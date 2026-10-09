//! Per-family call-body parsers. Each receives the text after the call opener as it grows and
//! reports when the call is complete (how many bytes it consumed), that it needs more text, or
//! that the text is malformed.
//!
//! Wire formats and where each was read from:
//!
//! * **Hermes JSON** — `<tool_call>\n{"name": "f", "arguments": {...}}\n</tool_call>`;
//!   content before/between calls allowed. Source: `tokenizer.chat_template` of
//!   Qwen3-1.7B-Q4_K_M.gguf (also SmolLM3-3B's `xml_tools` instruction and Granite 4.0).
//! * **Qwen XML** — `<tool_call>\n<function=f>\n<parameter=k>\nv\n</parameter>\n</function>\n</tool_call>`;
//!   mappings/sequences are JSON text, other values Python `str()`. Source: Qwen3.5-0.8B-Q4_0.gguf
//!   (same syntax in Qwen3-Coder and Nemotron 3 per the research notes).
//! * **GLM** — `<tool_call>f<arg_key>k</arg_key><arg_value>v</arg_value>…</tool_call>`; strings
//!   raw, other values JSON. Source: zai-org/GLM-4.7-Flash `chat_template.jinja` (MIT).
//! * **Gemma 4** — `<|tool_call>call:f{k:<|"|>v<|"|>,n:1,o:{x:true},l:[null]}<tool_call|>`; keys
//!   sorted by the template, bare (nested too), strings between `<|"|>` delimiters with no
//!   escaping. Source: gemma-4-12b-it-qat-q4_0.gguf.
//! * **Llama 3.x / 4** — `<|python_tag|>{"type": "function", "name": "f", "parameters": {...}}<|eom_id|>`
//!   (single call), the same object without `<|python_tag|>` (the HF template renders
//!   `{"name": …, "parameters": …}`), `<function=f>{json}</function>` (zero-shot variant),
//!   `[f(a="x"), g(n=1)]<|eot|>` (Llama 4 pythonic, parallel calls) and `<|python_tag|>ns.call(k="v")`
//!   built-ins. Sources: meta-llama/llama-models `models/llama3_1/prompt_format.md` and
//!   `models/llama4/prompt_format.md` (the HF repos are gated).
//! * **Mistral** — `[TOOL_CALLS]f[ARGS]{json}` per call, content before allowed; the older
//!   `[TOOL_CALLS][{"name": "f", "arguments": {...}, "id": "abc123def"}]` list form is accepted
//!   too. Source: mistralai/Devstral-Small-2-24B-Instruct-2512 `chat_template.jinja`
//!   (Apache-2.0); the list form from the research notes (Mistral Small 3.x, unverified
//!   against a template in this session).
//! * **LFM2** — `<|tool_call_start|>[f(a='b', n=1, m={"k": 1})]<|tool_call_end|>`; strings
//!   single-quoted with `\\ \' \n \r` escapes, mappings/lists JSON, other scalars Python
//!   `str()`. Source: LiquidAI/LFM2-1.2B `chat_template.jinja`.
//! * **OLMo 3** — `<function_calls>f(a="x", n=1)\ng(b=true)</function_calls>`; values are
//!   JSON (`tojson`), calls newline-separated. Source: olmo-3-7b-instruct-q4_k_m.gguf.

use super::scan::{
    expect_one_of, expect_tag, find_tag, parse_gemma_key, parse_gemma_value, parse_py_value,
    ws_len, Cursor, JsonObjectScanner, JsonValueScanner, PErr,
};
use super::schema::type_text;
use super::{ArgAssembler, Emitter};
use serde_json::{Map, Value};

/// Which body parser a family's call opener starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BodyKind {
    Hermes,
    QwenXml,
    Glm,
    Gemma4,
    Pythonic { closer: Option<&'static str> },
    Mistral,
    Llama,
}

pub(crate) trait CallBody {
    /// `body` is all text after the opener so far. `Ok(Some(n))`: the call is complete and
    /// occupied `body[..n]`. `Ok(None)`: wait for more text. `Err`: malformed.
    fn advance(&mut self, body: &str, em: &mut Emitter) -> Result<Option<usize>, String>;
    /// The output ended inside this call. `Ok` when the call could still be completed (only
    /// a closer was missing); `Err(reason)` when it is truncated.
    fn finish(&mut self, body: &str, em: &mut Emitter) -> Result<(), String>;
}

pub(crate) fn new_body(kind: BodyKind, opener: &str) -> Box<dyn CallBody> {
    match kind {
        BodyKind::Hermes => Box::new(HermesBody::default()),
        BodyKind::QwenXml => Box::new(XmlBody::default()),
        BodyKind::Glm => Box::new(GlmBody::default()),
        BodyKind::Gemma4 => Box::new(GemmaBody::default()),
        BodyKind::Pythonic { closer } => Box::new(PythonicBody::new(closer)),
        BodyKind::Mistral => Box::new(MistralBody::default()),
        BodyKind::Llama => Box::new(LlamaBody::new(opener)),
    }
}

/// After a complete call: consume `closer` when present, wait when the tail could still
/// become it, accept the call without it otherwise (models often stop right after the
/// arguments). `None` = wait.
fn lenient_closer(rest: &str, closer: &str) -> Option<usize> {
    let ws = ws_len(rest);
    match expect_tag(&rest[ws..], closer) {
        Ok(n) => Some(ws + n),
        Err(PErr::Incomplete) => None,
        Err(PErr::Syntax(_)) => Some(0),
    }
}

fn syntax(what: &str, e: PErr) -> String {
    match e {
        PErr::Syntax(s) => format!("{what}: {s}"),
        PErr::Incomplete => format!("{what}: unexpected end of input"),
    }
}

// ---------------------------------------------------------------------------------------------
// JSON call objects: {"name": "f", "arguments"|"parameters": {...}, "id"?: "..."}

/// One JSON call object, streamed. Shared by Hermes, Mistral's list form and Llama JSON.
struct JsonCallObject {
    scanner: JsonObjectScanner,
    fed: usize,
    args_keys: &'static [&'static str],
    name: Option<String>,
    index: Option<usize>,
    args_member: Option<usize>,
    streamed_to: Option<usize>,
    finalized: bool,
}

impl JsonCallObject {
    fn new(args_keys: &'static [&'static str]) -> JsonCallObject {
        JsonCallObject {
            scanner: JsonObjectScanner::new(),
            fed: 0,
            args_keys,
            name: None,
            index: None,
            args_member: None,
            streamed_to: None,
            finalized: false,
        }
    }

    fn explicit_id(&self) -> Option<String> {
        let i = self.scanner.members.iter().position(|m| m.key == "id")?;
        serde_json::from_str::<String>(self.scanner.value_text(i)?).ok()
    }

    /// `text` starts at the object's first byte (leading whitespace allowed).
    fn advance(&mut self, text: &str, em: &mut Emitter) -> Result<Option<usize>, String> {
        if self.finalized {
            return Ok(self.scanner.done);
        }
        self.scanner.feed(&text[self.fed..]);
        self.fed = text.len();
        if let Some(e) = &self.scanner.error {
            return Err(format!("tool call JSON is malformed: {e}"));
        }
        if self.name.is_none() {
            if let Some(i) = self.scanner.members.iter().position(|m| m.key == "name") {
                if let Some(t) = self.scanner.value_text(i) {
                    match serde_json::from_str::<Value>(t) {
                        Ok(Value::String(s)) => self.name = Some(s),
                        _ => return Err("tool call \"name\" is not a string".to_string()),
                    }
                }
            }
        }
        if self.args_member.is_none() {
            self.args_member = self
                .scanner
                .members
                .iter()
                .position(|m| self.args_keys.contains(&m.key.as_str()));
        }
        if let (Some(name), Some(ai)) = (self.name.clone(), self.args_member) {
            let (start, end) = {
                let m = &self.scanner.members[ai];
                (
                    m.value_start,
                    m.value_end.unwrap_or(self.scanner.text.len()),
                )
            };
            if self.scanner.text.as_bytes()[start] == b'{' {
                if self.index.is_none() {
                    let id = self.explicit_id();
                    self.index = Some(em.start(&name, id));
                    self.streamed_to = Some(start);
                }
                let from = self.streamed_to.unwrap_or(start);
                if end > from {
                    let index = self.index.unwrap_or(0);
                    em.delta(index, &self.scanner.text[from..end]);
                    self.streamed_to = Some(end);
                }
            }
        }
        if let Some(done) = self.scanner.done {
            self.finalize(em)?;
            return Ok(Some(done));
        }
        Ok(None)
    }

    fn finalize(&mut self, em: &mut Emitter) -> Result<(), String> {
        let name = self
            .name
            .clone()
            .ok_or_else(|| "tool call object has no \"name\"".to_string())?;
        let args = match self.args_member {
            Some(i) => {
                let t = self.scanner.value_text(i).unwrap_or("{}");
                let v: Value = serde_json::from_str(t)
                    .map_err(|e| format!("tool call arguments are not valid JSON: {e}"))?;
                match v {
                    // Double-encoded arguments ("{\"a\": 1}") are unwrapped.
                    Value::String(s) => {
                        serde_json::from_str::<Value>(&s).unwrap_or(Value::String(s))
                    }
                    other => other,
                }
            }
            None => Value::Object(Map::new()),
        };
        let args = match args {
            Value::Object(m) => Value::Object(m),
            Value::Null => Value::Object(Map::new()),
            other => {
                return Err(format!(
                    "tool call arguments must be a JSON object, got {other}"
                ))
            }
        };
        let index = match self.index {
            Some(i) => i,
            None => {
                let i = em.start(&name, self.explicit_id());
                let text = serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string());
                em.delta(i, &text);
                i
            }
        };
        em.end(index, args);
        self.finalized = true;
        Ok(())
    }
}

/// `[obj, obj, ...]` or a single `obj`, each a [`JsonCallObject`].
struct JsonCallList {
    single: bool,
    args_keys: &'static [&'static str],
    pos: usize,
    state: ListState,
}

enum ListState {
    Open,
    Elem,
    InElem {
        obj: Box<JsonCallObject>,
        start: usize,
    },
    AfterElem,
    Done,
}

impl JsonCallList {
    fn new(single: bool, args_keys: &'static [&'static str]) -> JsonCallList {
        JsonCallList {
            single,
            args_keys,
            pos: 0,
            state: if single {
                ListState::Elem
            } else {
                ListState::Open
            },
        }
    }

    fn advance(&mut self, body: &str, em: &mut Emitter) -> Result<Option<usize>, String> {
        loop {
            let rest = &body[self.pos..];
            let ws = ws_len(rest);
            let r = &rest[ws..];
            match &mut self.state {
                ListState::Open => {
                    if r.is_empty() {
                        return Ok(None);
                    }
                    if !r.starts_with('[') {
                        return Err("expected '[' to open the tool call list".to_string());
                    }
                    self.pos += ws + 1;
                    self.state = ListState::Elem;
                }
                ListState::Elem => {
                    if r.is_empty() {
                        return Ok(None);
                    }
                    if r.starts_with(']') && !self.single {
                        self.pos += ws + 1;
                        self.state = ListState::Done;
                        return Ok(Some(self.pos));
                    }
                    if !r.starts_with('{') {
                        return Err("expected a tool call object".to_string());
                    }
                    self.pos += ws;
                    self.state = ListState::InElem {
                        obj: Box::new(JsonCallObject::new(self.args_keys)),
                        start: self.pos,
                    };
                }
                ListState::InElem { obj, start } => match obj.advance(&body[*start..], em)? {
                    Some(n) => {
                        self.pos = *start + n;
                        if self.single {
                            self.state = ListState::Done;
                            return Ok(Some(self.pos));
                        }
                        self.state = ListState::AfterElem;
                    }
                    None => return Ok(None),
                },
                ListState::AfterElem => {
                    if r.is_empty() {
                        return Ok(None);
                    }
                    if r.starts_with(',') {
                        self.pos += ws + 1;
                        self.state = ListState::Elem;
                    } else if r.starts_with(']') {
                        self.pos += ws + 1;
                        self.state = ListState::Done;
                        return Ok(Some(self.pos));
                    } else {
                        return Err("expected ',' or ']' after a tool call object".to_string());
                    }
                }
                ListState::Done => return Ok(Some(self.pos)),
            }
        }
    }

    fn is_done(&self) -> bool {
        matches!(self.state, ListState::Done)
    }
}

// ---------------------------------------------------------------------------------------------
// Hermes

#[derive(Default)]
struct HermesBody {
    obj: Option<JsonCallObject>,
    end: Option<usize>,
}

impl CallBody for HermesBody {
    fn advance(&mut self, body: &str, em: &mut Emitter) -> Result<Option<usize>, String> {
        if self.end.is_none() {
            let obj = self
                .obj
                .get_or_insert_with(|| JsonCallObject::new(&["arguments", "parameters"]));
            match obj.advance(body, em)? {
                Some(e) => self.end = Some(e),
                None => return Ok(None),
            }
        }
        let e = self.end.unwrap_or(0);
        Ok(lenient_closer(&body[e..], "</tool_call>").map(|n| e + n))
    }

    fn finish(&mut self, _body: &str, _em: &mut Emitter) -> Result<(), String> {
        if self.end.is_some() {
            Ok(())
        } else {
            Err("tool call truncated before its JSON completed".to_string())
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Qwen XML

#[derive(Default)]
struct XmlBody {
    pos: usize,
    state: XmlState,
    name: String,
    assembler: Option<ArgAssembler>,
}

#[derive(Default, PartialEq, Eq)]
enum XmlState {
    #[default]
    Function,
    Params,
    AfterFunction,
    Done,
}

/// `<parameter>` values are wrapped in newlines by the template: strip exactly one on each side.
fn strip_xml_value(raw: &str) -> &str {
    let s = raw.strip_prefix('\n').unwrap_or(raw);
    s.strip_suffix('\n').unwrap_or(s)
}

impl XmlBody {
    fn finalize(&mut self, em: &mut Emitter) {
        if let Some(a) = self.assembler.take() {
            a.finish(em);
        }
        self.state = XmlState::Done;
    }
}

impl CallBody for XmlBody {
    fn advance(&mut self, body: &str, em: &mut Emitter) -> Result<Option<usize>, String> {
        loop {
            let rest = &body[self.pos..];
            let ws = ws_len(rest);
            let r = &rest[ws..];
            match self.state {
                XmlState::Function => {
                    match expect_tag(r, "<function=") {
                        Ok(_) => {}
                        Err(PErr::Incomplete) => return Ok(None),
                        Err(e) => return Err(syntax("expected <function=name>", e)),
                    }
                    let Some(gt) = r.find('>') else {
                        return Ok(None);
                    };
                    self.name = r["<function=".len()..gt].trim().to_string();
                    if self.name.is_empty() {
                        return Err("empty function name".to_string());
                    }
                    let index = em.start(&self.name, None);
                    self.assembler = Some(ArgAssembler::new(em, index));
                    self.pos += ws + gt + 1;
                    self.state = XmlState::Params;
                }
                XmlState::Params => {
                    match expect_one_of(r, &["<parameter=", "</function>", "</tool_call>"]) {
                        Ok(0) => {
                            let Some(gt) = r.find('>') else {
                                return Ok(None);
                            };
                            let key = r["<parameter=".len()..gt].trim().to_string();
                            let after = &r[gt + 1..];
                            let Ok(end) = find_tag(after, "</parameter>") else {
                                return Ok(None);
                            };
                            let raw = strip_xml_value(&after[..end]);
                            let schema = em
                                .tools()
                                .get(&self.name)
                                .and_then(|s| s.property(&key))
                                .cloned();
                            let value = type_text(raw, schema.as_ref());
                            if let Some(a) = self.assembler.as_mut() {
                                a.arg(em, &key, value);
                            }
                            self.pos += ws + gt + 1 + end + "</parameter>".len();
                        }
                        Ok(1) => {
                            self.pos += ws + "</function>".len();
                            self.state = XmlState::AfterFunction;
                        }
                        Ok(_) => {
                            self.pos += ws + "</tool_call>".len();
                            self.finalize(em);
                            return Ok(Some(self.pos));
                        }
                        Err(PErr::Incomplete) => return Ok(None),
                        Err(e) => return Err(syntax("expected <parameter=…> or </function>", e)),
                    }
                }
                XmlState::AfterFunction => {
                    return match lenient_closer(rest, "</tool_call>") {
                        Some(n) => {
                            self.pos += n;
                            self.finalize(em);
                            Ok(Some(self.pos))
                        }
                        None => Ok(None),
                    };
                }
                XmlState::Done => return Ok(Some(self.pos)),
            }
        }
    }

    fn finish(&mut self, _body: &str, em: &mut Emitter) -> Result<(), String> {
        match self.state {
            XmlState::AfterFunction => {
                self.finalize(em);
                Ok(())
            }
            XmlState::Done => Ok(()),
            _ => Err("XML tool call truncated before </function>".to_string()),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// GLM

#[derive(Default)]
struct GlmBody {
    pos: usize,
    state: GlmState,
    name: String,
    assembler: Option<ArgAssembler>,
}

#[derive(Default, PartialEq, Eq)]
enum GlmState {
    #[default]
    Name,
    Args,
    Done,
}

impl GlmBody {
    fn finalize(&mut self, em: &mut Emitter) {
        if let Some(a) = self.assembler.take() {
            a.finish(em);
        }
        self.state = GlmState::Done;
    }
}

impl CallBody for GlmBody {
    fn advance(&mut self, body: &str, em: &mut Emitter) -> Result<Option<usize>, String> {
        loop {
            let rest = &body[self.pos..];
            match self.state {
                GlmState::Name => {
                    let Some(lt) = rest.find('<') else {
                        return Ok(None);
                    };
                    let r = &rest[lt..];
                    match expect_one_of(r, &["<arg_key>", "</tool_call>"]) {
                        Ok(which) => {
                            self.name = rest[..lt].trim().to_string();
                            if self.name.is_empty() {
                                return Err("empty function name".to_string());
                            }
                            let index = em.start(&self.name, None);
                            self.assembler = Some(ArgAssembler::new(em, index));
                            self.pos += lt;
                            self.state = GlmState::Args;
                            if which == 1 {
                                continue;
                            }
                        }
                        Err(PErr::Incomplete) => return Ok(None),
                        Err(e) => return Err(syntax("expected <arg_key> after the name", e)),
                    }
                }
                GlmState::Args => {
                    let ws = ws_len(rest);
                    let r = &rest[ws..];
                    match expect_one_of(r, &["<arg_key>", "</tool_call>"]) {
                        Ok(0) => {
                            let after = &r["<arg_key>".len()..];
                            let Ok(k_end) = find_tag(after, "</arg_key>") else {
                                return Ok(None);
                            };
                            let key = after[..k_end].trim().to_string();
                            let after_key = &after[k_end + "</arg_key>".len()..];
                            let ws2 = ws_len(after_key);
                            let v = &after_key[ws2..];
                            match expect_tag(v, "<arg_value>") {
                                Ok(_) => {}
                                Err(PErr::Incomplete) => return Ok(None),
                                Err(e) => return Err(syntax("expected <arg_value>", e)),
                            }
                            let v_body = &v["<arg_value>".len()..];
                            let Ok(v_end) = find_tag(v_body, "</arg_value>") else {
                                return Ok(None);
                            };
                            let raw = &v_body[..v_end];
                            let schema = em
                                .tools()
                                .get(&self.name)
                                .and_then(|s| s.property(&key))
                                .cloned();
                            let value = type_text(raw, schema.as_ref());
                            if let Some(a) = self.assembler.as_mut() {
                                a.arg(em, &key, value);
                            }
                            self.pos += ws
                                + "<arg_key>".len()
                                + k_end
                                + "</arg_key>".len()
                                + ws2
                                + "<arg_value>".len()
                                + v_end
                                + "</arg_value>".len();
                        }
                        Ok(_) => {
                            self.pos += ws + "</tool_call>".len();
                            self.finalize(em);
                            return Ok(Some(self.pos));
                        }
                        Err(PErr::Incomplete) => return Ok(None),
                        Err(e) => return Err(syntax("expected <arg_key> or </tool_call>", e)),
                    }
                }
                GlmState::Done => return Ok(Some(self.pos)),
            }
        }
    }

    fn finish(&mut self, body: &str, em: &mut Emitter) -> Result<(), String> {
        match self.state {
            GlmState::Args if body[self.pos..].trim().is_empty() => {
                // Only the closer is missing.
                self.finalize(em);
                Ok(())
            }
            GlmState::Done => Ok(()),
            _ => Err("GLM tool call truncated before its arguments completed".to_string()),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Gemma 4

#[derive(Default)]
struct GemmaBody {
    pos: usize,
    state: GemmaState,
    assembler: Option<ArgAssembler>,
}

#[derive(Default, PartialEq, Eq)]
enum GemmaState {
    #[default]
    Head,
    Pairs,
    Close,
    Done,
}

impl CallBody for GemmaBody {
    fn advance(&mut self, body: &str, em: &mut Emitter) -> Result<Option<usize>, String> {
        loop {
            let rest = &body[self.pos..];
            match self.state {
                GemmaState::Head => {
                    let mut c = Cursor::new(rest);
                    c.skip_ws();
                    match expect_tag(c.rest(), "call:") {
                        Ok(n) => c.pos += n,
                        Err(PErr::Incomplete) => return Ok(None),
                        Err(e) => return Err(syntax("expected call:name{…}", e)),
                    }
                    c.skip_ws();
                    let name = match c.identifier() {
                        Ok(n) => n,
                        Err(PErr::Incomplete) => return Ok(None),
                        Err(e) => return Err(syntax("expected a function name", e)),
                    };
                    c.skip_ws();
                    match c.peek() {
                        None => return Ok(None),
                        Some('{') => {
                            c.bump();
                        }
                        Some(x) => return Err(format!("expected '{{' after the name, got {x:?}")),
                    }
                    let index = em.start(&name, None);
                    self.assembler = Some(ArgAssembler::new(em, index));
                    self.pos += c.pos;
                    self.state = GemmaState::Pairs;
                }
                GemmaState::Pairs => {
                    let mut c = Cursor::new(rest);
                    c.skip_ws();
                    match c.peek() {
                        None => return Ok(None),
                        Some('}') => {
                            c.bump();
                            self.pos += c.pos;
                            self.state = GemmaState::Close;
                            continue;
                        }
                        Some(',') => {
                            c.bump();
                            self.pos += c.pos;
                            continue;
                        }
                        Some(_) => {}
                    }
                    let key = match parse_gemma_key(&mut c) {
                        Ok(k) => k,
                        Err(PErr::Incomplete) => return Ok(None),
                        Err(e) => return Err(syntax("expected an argument name", e)),
                    };
                    c.skip_ws();
                    match c.peek() {
                        None => return Ok(None),
                        Some(':') => {
                            c.bump();
                        }
                        Some(x) => return Err(format!("expected ':' after {key:?}, got {x:?}")),
                    }
                    let value = match parse_gemma_value(&mut c) {
                        Ok(v) => v,
                        Err(PErr::Incomplete) => return Ok(None),
                        Err(e) => return Err(syntax(&format!("bad value for {key:?}"), e)),
                    };
                    if let Some(a) = self.assembler.as_mut() {
                        a.arg(em, &key, value);
                    }
                    self.pos += c.pos;
                }
                GemmaState::Close => {
                    return match lenient_closer(rest, "<tool_call|>") {
                        Some(n) => {
                            self.pos += n;
                            if let Some(a) = self.assembler.take() {
                                a.finish(em);
                            }
                            self.state = GemmaState::Done;
                            Ok(Some(self.pos))
                        }
                        None => Ok(None),
                    };
                }
                GemmaState::Done => return Ok(Some(self.pos)),
            }
        }
    }

    fn finish(&mut self, _body: &str, em: &mut Emitter) -> Result<(), String> {
        match self.state {
            GemmaState::Close => {
                if let Some(a) = self.assembler.take() {
                    a.finish(em);
                }
                self.state = GemmaState::Done;
                Ok(())
            }
            GemmaState::Done => Ok(()),
            _ => Err("Gemma tool call truncated before its closing '}'".to_string()),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Pythonic: [f(a='b', n=1), g()] (LFM2, Llama 4), f(a="x")\ng(b=1) (OLMo 3), ns.call(k="v")

struct PythonicBody {
    closer: Option<&'static str>,
    pos: usize,
    state: PyState,
    bracketed: bool,
    calls: usize,
    assembler: Option<ArgAssembler>,
}

#[derive(PartialEq, Eq)]
enum PyState {
    Open,
    Calls,
    Args,
    Closer,
    Done,
}

impl PythonicBody {
    fn new(closer: Option<&'static str>) -> PythonicBody {
        PythonicBody {
            closer,
            pos: 0,
            state: PyState::Open,
            bracketed: false,
            calls: 0,
            assembler: None,
        }
    }
}

impl CallBody for PythonicBody {
    fn advance(&mut self, body: &str, em: &mut Emitter) -> Result<Option<usize>, String> {
        loop {
            let rest = &body[self.pos..];
            let ws = ws_len(rest);
            let r = &rest[ws..];
            match self.state {
                PyState::Open => {
                    if r.is_empty() {
                        return Ok(None);
                    }
                    if r.starts_with('[') {
                        self.bracketed = true;
                        self.pos += ws + 1;
                    } else {
                        self.pos += ws;
                    }
                    self.state = PyState::Calls;
                }
                PyState::Calls => {
                    if r.is_empty() {
                        return Ok(None);
                    }
                    if self.bracketed && r.starts_with(']') {
                        self.pos += ws + 1;
                        self.state = PyState::Closer;
                        continue;
                    }
                    if r.starts_with(',') {
                        self.pos += ws + 1;
                        continue;
                    }
                    if let (false, Some(closer)) = (self.bracketed, self.closer) {
                        match expect_tag(r, closer) {
                            Ok(_) => {
                                self.pos += ws;
                                self.state = PyState::Closer;
                                continue;
                            }
                            Err(PErr::Incomplete) => return Ok(None),
                            Err(PErr::Syntax(_)) => {}
                        }
                    }
                    let mut c = Cursor::new(r);
                    let name = match c.identifier() {
                        Ok(n) => n,
                        Err(PErr::Incomplete) => return Ok(None),
                        Err(e) => return Err(syntax("expected a function name", e)),
                    };
                    c.skip_ws();
                    match c.peek() {
                        None => return Ok(None),
                        Some('(') => {
                            c.bump();
                        }
                        Some(x) => return Err(format!("expected '(' after {name:?}, got {x:?}")),
                    }
                    let index = em.start(&name, None);
                    self.assembler = Some(ArgAssembler::new(em, index));
                    self.calls += 1;
                    self.pos += ws + c.pos;
                    self.state = PyState::Args;
                }
                PyState::Args => {
                    if r.is_empty() {
                        return Ok(None);
                    }
                    if r.starts_with(')') {
                        self.pos += ws + 1;
                        if let Some(a) = self.assembler.take() {
                            a.finish(em);
                        }
                        self.state = if !self.bracketed && self.closer.is_none() {
                            PyState::Closer
                        } else {
                            PyState::Calls
                        };
                        continue;
                    }
                    if r.starts_with(',') {
                        self.pos += ws + 1;
                        continue;
                    }
                    let mut c = Cursor::new(r);
                    let key = match c.identifier() {
                        Ok(k) => k,
                        Err(PErr::Incomplete) => return Ok(None),
                        Err(e) => return Err(syntax("expected keyword argument", e)),
                    };
                    c.skip_ws();
                    match c.peek() {
                        None => return Ok(None),
                        Some('=') => {
                            c.bump();
                        }
                        Some(x) => {
                            return Err(format!(
                                "expected '=' after argument {key:?}, got {x:?} (positional arguments are not supported)"
                            ))
                        }
                    }
                    let value = match parse_py_value(&mut c) {
                        Ok(v) => v,
                        Err(PErr::Incomplete) => return Ok(None),
                        Err(e) => return Err(syntax(&format!("bad value for {key:?}"), e)),
                    };
                    if let Some(a) = self.assembler.as_mut() {
                        a.arg(em, &key, value);
                    }
                    self.pos += ws + c.pos;
                }
                PyState::Closer => {
                    return match self.closer {
                        Some(closer) => match lenient_closer(rest, closer) {
                            Some(n) => {
                                self.pos += n;
                                self.state = PyState::Done;
                                Ok(Some(self.pos))
                            }
                            None => Ok(None),
                        },
                        None => {
                            self.state = PyState::Done;
                            Ok(Some(self.pos))
                        }
                    };
                }
                PyState::Done => return Ok(Some(self.pos)),
            }
        }
    }

    fn finish(&mut self, body: &str, _em: &mut Emitter) -> Result<(), String> {
        match self.state {
            PyState::Closer | PyState::Done => Ok(()),
            PyState::Calls
                if !self.bracketed && self.calls > 0 && body[self.pos..].trim().is_empty() =>
            {
                Ok(())
            }
            _ => Err("pythonic tool call truncated".to_string()),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Mistral

#[derive(Default)]
struct MistralBody {
    pos: usize,
    state: MistralState,
}

#[derive(Default)]
enum MistralState {
    #[default]
    Decide,
    Name,
    Args {
        index: usize,
        scanner: JsonValueScanner,
        fed: usize,
    },
    List(JsonCallList),
    Done,
}

impl CallBody for MistralBody {
    fn advance(&mut self, body: &str, em: &mut Emitter) -> Result<Option<usize>, String> {
        loop {
            let rest = &body[self.pos..];
            let ws = ws_len(rest);
            let r = &rest[ws..];
            match &mut self.state {
                MistralState::Decide => {
                    if r.is_empty() {
                        return Ok(None);
                    }
                    self.state = if r.starts_with('[') || r.starts_with('{') {
                        self.pos += ws;
                        MistralState::List(JsonCallList::new(
                            r.starts_with('{'),
                            &["arguments", "parameters"],
                        ))
                    } else {
                        MistralState::Name
                    };
                }
                MistralState::Name => {
                    let Ok(p) = find_tag(r, "[ARGS]") else {
                        return Ok(None);
                    };
                    let name = r[..p].trim();
                    if name.is_empty() {
                        return Err("empty function name before [ARGS]".to_string());
                    }
                    let index = em.start(name, None);
                    self.pos += ws + p + "[ARGS]".len();
                    self.state = MistralState::Args {
                        index,
                        scanner: JsonValueScanner::new(),
                        fed: 0,
                    };
                }
                MistralState::Args {
                    index,
                    scanner,
                    fed,
                } => {
                    let before = scanner.text.len();
                    scanner.feed(&rest[*fed..]);
                    *fed = rest.len();
                    if let Some(e) = &scanner.error {
                        return Err(format!("[ARGS] is not a JSON object: {e}"));
                    }
                    let end = scanner.done.unwrap_or(scanner.text.len());
                    if end > before {
                        em.delta(*index, &scanner.text[before..end]);
                    }
                    if let Some(done) = scanner.done {
                        let v: Value = serde_json::from_str(&scanner.text[..done])
                            .map_err(|e| format!("[ARGS] is not valid JSON: {e}"))?;
                        if !v.is_object() {
                            return Err("[ARGS] must be a JSON object".to_string());
                        }
                        em.end(*index, v);
                        self.pos += done;
                        self.state = MistralState::Done;
                        return Ok(Some(self.pos));
                    }
                    return Ok(None);
                }
                MistralState::List(list) => match list.advance(rest, em)? {
                    Some(n) => {
                        self.pos += n;
                        self.state = MistralState::Done;
                        return Ok(Some(self.pos));
                    }
                    None => return Ok(None),
                },
                MistralState::Done => return Ok(Some(self.pos)),
            }
        }
    }

    fn finish(&mut self, _body: &str, _em: &mut Emitter) -> Result<(), String> {
        match &self.state {
            MistralState::Done => Ok(()),
            MistralState::List(l) if l.is_done() => Ok(()),
            _ => Err("[TOOL_CALLS] truncated before its arguments completed".to_string()),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Llama 3.x / 4

struct LlamaBody {
    pos: usize,
    state: LlamaState,
}

enum LlamaState {
    Decide,
    List(JsonCallList),
    Pythonic(PythonicBody),
    FunctionName,
    FunctionArgs {
        index: usize,
        scanner: JsonValueScanner,
        fed: usize,
    },
    FunctionClose,
    Done,
}

impl LlamaBody {
    fn new(opener: &str) -> LlamaBody {
        LlamaBody {
            pos: 0,
            state: if opener == "<function=" {
                LlamaState::FunctionName
            } else {
                LlamaState::Decide
            },
        }
    }
}

impl CallBody for LlamaBody {
    fn advance(&mut self, body: &str, em: &mut Emitter) -> Result<Option<usize>, String> {
        loop {
            let rest = &body[self.pos..];
            let ws = ws_len(rest);
            let r = &rest[ws..];
            match &mut self.state {
                LlamaState::Decide => {
                    let Some(first) = r.chars().next() else {
                        return Ok(None);
                    };
                    self.state = match first {
                        '{' => {
                            self.pos += ws;
                            LlamaState::List(JsonCallList::new(true, &["parameters", "arguments"]))
                        }
                        '[' => {
                            let inner = r[1..].trim_start();
                            let Some(second) = inner.chars().next() else {
                                return Ok(None);
                            };
                            self.pos += ws;
                            if second == '{' {
                                LlamaState::List(JsonCallList::new(
                                    false,
                                    &["parameters", "arguments"],
                                ))
                            } else {
                                LlamaState::Pythonic(PythonicBody::new(None))
                            }
                        }
                        c if c.is_ascii_alphabetic() || c == '_' => {
                            self.pos += ws;
                            LlamaState::Pythonic(PythonicBody::new(None))
                        }
                        other => return Err(format!("not a tool call (starts with {other:?})")),
                    };
                }
                LlamaState::List(list) => match list.advance(rest, em)? {
                    Some(n) => {
                        self.pos += n;
                        self.state = LlamaState::Done;
                        return Ok(Some(self.pos));
                    }
                    None => return Ok(None),
                },
                LlamaState::Pythonic(py) => match py.advance(rest, em)? {
                    Some(n) => {
                        self.pos += n;
                        self.state = LlamaState::Done;
                        return Ok(Some(self.pos));
                    }
                    None => return Ok(None),
                },
                LlamaState::FunctionName => {
                    let Some(gt) = rest.find('>') else {
                        return Ok(None);
                    };
                    let name = rest[..gt].trim();
                    if name.is_empty() {
                        return Err("empty function name".to_string());
                    }
                    let index = em.start(name, None);
                    self.pos += gt + 1;
                    self.state = LlamaState::FunctionArgs {
                        index,
                        scanner: JsonValueScanner::new(),
                        fed: 0,
                    };
                }
                LlamaState::FunctionArgs {
                    index,
                    scanner,
                    fed,
                } => {
                    let before = scanner.text.len();
                    scanner.feed(&rest[*fed..]);
                    *fed = rest.len();
                    if let Some(e) = &scanner.error {
                        return Err(format!("<function=…> body is not a JSON object: {e}"));
                    }
                    let end = scanner.done.unwrap_or(scanner.text.len());
                    if end > before {
                        em.delta(*index, &scanner.text[before..end]);
                    }
                    let Some(done) = scanner.done else {
                        return Ok(None);
                    };
                    let v: Value = serde_json::from_str(&scanner.text[..done])
                        .map_err(|e| format!("<function=…> body is not valid JSON: {e}"))?;
                    if !v.is_object() {
                        return Err("<function=…> body must be a JSON object".to_string());
                    }
                    em.end(*index, v);
                    self.pos += done;
                    self.state = LlamaState::FunctionClose;
                }
                LlamaState::FunctionClose => {
                    return match lenient_closer(rest, "</function>") {
                        Some(n) => {
                            self.pos += n;
                            self.state = LlamaState::Done;
                            Ok(Some(self.pos))
                        }
                        None => Ok(None),
                    };
                }
                LlamaState::Done => return Ok(Some(self.pos)),
            }
        }
    }

    fn finish(&mut self, body: &str, em: &mut Emitter) -> Result<(), String> {
        match &mut self.state {
            LlamaState::Done | LlamaState::FunctionClose => Ok(()),
            LlamaState::List(l) if l.is_done() => Ok(()),
            LlamaState::Pythonic(py) => py.finish(&body[self.pos..], em),
            _ => Err("Llama tool call truncated".to_string()),
        }
    }
}
