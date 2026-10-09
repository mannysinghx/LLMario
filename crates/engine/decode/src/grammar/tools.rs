//! Tool-call grammars per chat family (wire formats: `docs/engine/ARCHITECTURE.md` Appendix D).
//!
//! Every family grammar constrains the function name to the tool set and the arguments to that
//! tool's JSON Schema. JSON families (Hermes, Llama 3, Mistral, Harmony) wrap the arguments in a
//! `%json` sub-grammar; the tag families (Qwen XML, Gemma 4) spell the parameters out as rules in
//! schema order (required ones mandatory, optional ones skippable) with typed values.
//!
//! Control tokens that belong to a format (`<|python_tag|>`, `[TOOL_CALLS]`, `<|channel|>`, the
//! Gemma 4 `<|"|>` quote, ...) are referenced by token id (`<[id]>`) when the tokenizer has them
//! as control tokens and as plain text otherwise, so the same builder serves real and synthetic
//! vocabularies.

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use super::{with_json_options, GrammarError, GrammarTokenizer};

/// The chat families whose tool-call wire format the engine can enforce. Names mirror
/// `llmario-engine-chat`'s `TemplateFamily`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolFamily {
    /// `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` (Qwen3, SmolLM3, Granite 4).
    Hermes,
    /// `<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`
    /// (Qwen3.5, Qwen3-Coder).
    QwenXml,
    /// `<|tool_call>call:name{k:<|"|>v<|"|>}<tool_call|>` (Gemma 4).
    Gemma4,
    /// `<|python_tag|>{"name": ..., "parameters": {...}}` (Llama 3.x, single call).
    Llama3,
    /// `[TOOL_CALLS]name[ARGS]{json}` (Mistral Small 3.x, Ministral 3, Devstral 2).
    Mistral,
    /// `<|channel|>commentary to=functions.name <|constrain|>json<|message|>{json}<|call|>`
    /// (gpt-oss Harmony).
    Harmony,
}

/// One tool: its name and the JSON Schema of its arguments (OpenAI `function.parameters`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    /// JSON Schema of the arguments object; `{}`/missing means "any object".
    #[serde(default = "any_object")]
    pub parameters: Value,
}

fn any_object() -> Value {
    json!({"type": "object"})
}

impl ToolDef {
    pub fn new(name: impl Into<String>, parameters: Value) -> Self {
        ToolDef {
            name: name.into(),
            parameters,
        }
    }
}

/// OpenAI `tool_choice`. `none` is not a grammar: the caller simply attaches no processor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    /// Free text; if the family's opener appears, the call must be well-formed (lazy trigger on
    /// the opener).
    Auto,
    /// A call must come first (the grammar starts at the opener).
    Required,
    /// A call to this tool must come first.
    Named(String),
}

/// A tool-call grammar request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCallSpec {
    pub family: ToolFamily,
    pub tools: Vec<ToolDef>,
    pub choice: ToolChoice,
}

pub(crate) struct ToolGrammar {
    pub lark: String,
    pub triggers: Vec<Vec<u8>>,
}

/// Lark string literal (Lark uses JSON string syntax).
fn lit(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".to_string())
}

/// A format token: `<[id]>` when the tokenizer has it as a control token, else literal text.
fn tok(env: &GrammarTokenizer, text: &str) -> String {
    match env.special_token_id(text) {
        Some(id) => format!("<[{id}]>"),
        None => lit(text),
    }
}

/// Sub-schema `%json` reference (with the engine's bounded-whitespace option).
fn json_rule(schema: &Value) -> String {
    format!("%json {}", with_json_options(schema))
}

struct Lark {
    out: String,
    fresh: usize,
}

impl Lark {
    fn new() -> Self {
        let mut l = Lark {
            out: String::new(),
            fresh: 0,
        };
        // Bounded like the JSON whitespace (see `JSON_WHITESPACE_PATTERN`): a greedy decode must
        // not be able to emit blanks forever between calls.
        l.rule("ws", "/[ \\t\\r\\n]{1,40}/");
        l
    }

    fn rule(&mut self, name: &str, body: &str) {
        self.out.push_str(name);
        self.out.push_str(": ");
        self.out.push_str(body);
        self.out.push('\n');
    }

    fn rule_attr(&mut self, name: &str, attrs: &str, body: &str) {
        self.out.push_str(name);
        self.out.push('[');
        self.out.push_str(attrs);
        self.out.push_str("]: ");
        self.out.push_str(body);
        self.out.push('\n');
    }

    fn fresh(&mut self, prefix: &str) -> String {
        self.fresh += 1;
        format!("{prefix}_{}", self.fresh)
    }
}

/// A parameter of an object schema, in declaration order.
struct Param<'a> {
    name: &'a str,
    schema: &'a Value,
    required: bool,
}

fn object_params(schema: &Value) -> Vec<Param<'_>> {
    let Some(props) = schema.get("properties").and_then(Value::as_object) else {
        return Vec::new();
    };
    let required: Vec<&str> = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    props
        .iter()
        .map(|(k, v)| Param {
            name: k.as_str(),
            schema: v,
            required: required.contains(&k.as_str()),
        })
        .collect()
}

/// "Required ones in order, then each optional one at most once, in order", with `sep` between
/// entries. Returns the body of a rule (possibly empty).
fn ordered_params(l: &mut Lark, rules: &[(String, bool)], sep: &str) -> String {
    let required: Vec<&String> = rules.iter().filter(|(_, r)| *r).map(|(n, _)| n).collect();
    let optional: Vec<&String> = rules.iter().filter(|(_, r)| !*r).map(|(n, _)| n).collect();
    if rules.is_empty() {
        return String::new();
    }
    if !required.is_empty() {
        let mut body = String::new();
        for (i, r) in required.iter().enumerate() {
            if i > 0 {
                body.push_str(sep);
            }
            body.push_str(r);
        }
        for o in &optional {
            body.push_str(&format!(" ({sep} {o})?"));
        }
        return body;
    }
    // All optional: alternatives starting at each index.
    let mut alts = Vec::new();
    for k in 0..optional.len() {
        let mut body = optional[k].clone();
        for o in &optional[k + 1..] {
            body.push_str(&format!(" ({sep} {o})?"));
        }
        let name = l.fresh("opt");
        l.rule(&name, &body);
        alts.push(name);
    }
    format!("({})?", alts.join(" | "))
}

/// Wrapper schema for the JSON families: `{"name": <const>, "<args_key>": <parameters>}`.
fn json_call_schema(tools: &[&ToolDef], args_key: &str) -> Value {
    let alts: Vec<Value> = tools
        .iter()
        .map(|t| {
            let mut props = Map::new();
            props.insert("name".into(), json!({"const": t.name}));
            props.insert(args_key.to_string(), args_schema(&t.parameters));
            json!({
                "type": "object",
                "properties": Value::Object(props),
                "required": ["name", args_key],
                "additionalProperties": false
            })
        })
        .collect();
    if alts.len() == 1 {
        alts.into_iter().next().unwrap_or(Value::Null)
    } else {
        json!({ "anyOf": alts })
    }
}

/// The arguments schema as given, defaulting to "any object" when it is empty or not a schema.
fn args_schema(parameters: &Value) -> Value {
    match parameters {
        Value::Object(m) if !m.is_empty() => parameters.clone(),
        _ => any_object(),
    }
}

fn schema_types(schema: &Value) -> Vec<&str> {
    match schema.get("type") {
        Some(Value::String(s)) => vec![s.as_str()],
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    }
}

/// Gemma 4 value rule for a schema: strings between `<|"|>` quote tokens, scalars bare, arrays
/// and objects recursive; anything the builder does not model falls back to `%json`.
fn gemma_value(l: &mut Lark, env: &GrammarTokenizer, schema: &Value, depth: usize) -> String {
    let quote = tok(env, "<|\"|>");
    let quote_is_special = env.special_token_id("<|\"|>").is_some();
    if let Some(vals) = schema.get("enum").and_then(Value::as_array) {
        let alts: Vec<String> = vals
            .iter()
            .map(|v| match v {
                Value::String(s) => format!("{quote} {} {quote}", lit(s)),
                other => lit(&other.to_string()),
            })
            .collect();
        if !alts.is_empty() {
            let name = l.fresh("genum");
            l.rule(&name, &alts.join(" | "));
            return name;
        }
    }
    if let Some(c) = schema.get("const") {
        let name = l.fresh("gconst");
        let body = match c {
            Value::String(s) => format!("{quote} {} {quote}", lit(s)),
            other => lit(&other.to_string()),
        };
        l.rule(&name, &body);
        return name;
    }
    let types = schema_types(schema);
    if types.is_empty() || depth > 4 {
        let name = l.fresh("gjson");
        l.rule(&name, &json_rule(schema));
        return name;
    }
    let mut alts = Vec::new();
    for ty in types {
        match ty {
            "string" => {
                let name = l.fresh("gstr");
                if quote_is_special {
                    l.rule(&name, &format!("{quote} /(.|\\n)*/ {quote}"));
                } else {
                    let body = l.fresh("gstr_body");
                    l.rule_attr(&body, &format!("suffix={}", lit("<|\"|>")), "/(.|\\n)*/");
                    l.rule(&name, &format!("{quote} {body}"));
                }
                alts.push(name);
            }
            "integer" => alts.push("/-?(0|[1-9][0-9]*)/".to_string()),
            "number" => alts.push("/-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?/".to_string()),
            "boolean" => alts.push("(\"true\" | \"false\")".to_string()),
            "null" => alts.push("\"null\"".to_string()),
            "array" => {
                let item = match schema.get("items") {
                    Some(items) if items.is_object() => gemma_value(l, env, items, depth + 1),
                    _ => {
                        let n = l.fresh("gjson");
                        l.rule(&n, "%json {}");
                        n
                    }
                };
                let name = l.fresh("garr");
                l.rule(
                    &name,
                    &format!("\"[\" ws? ({item} (\",\" ws? {item})*)? ws? \"]\""),
                );
                alts.push(name);
            }
            "object" => {
                let params = object_params(schema);
                if params.is_empty() {
                    let n = l.fresh("gjson");
                    l.rule(&n, &json_rule(&any_object()));
                    alts.push(n);
                } else {
                    let mut rules = Vec::new();
                    for p in &params {
                        let v = gemma_value(l, env, p.schema, depth + 1);
                        let pr = l.fresh("gprop");
                        l.rule(&pr, &format!("{} \":\" {v}", lit(p.name)));
                        rules.push((pr, p.required));
                    }
                    let inner = ordered_params(l, &rules, "\",\" ws?");
                    let name = l.fresh("gobj");
                    l.rule(&name, &format!("\"{{\" ws? {inner} ws? \"}}\""));
                    alts.push(name);
                }
            }
            _ => {
                let n = l.fresh("gjson");
                l.rule(&n, &json_rule(schema));
                alts.push(n);
            }
        }
    }
    let name = l.fresh("gval");
    l.rule(&name, &alts.join(" | "));
    name
}

/// Qwen XML: `<parameter=k>` value `</parameter>`; strings are raw text up to the closing tag,
/// everything else is JSON.
fn qwen_param(l: &mut Lark, p: &Param<'_>, idx: usize) -> String {
    let name = format!("qp{idx}_{}", l.fresh("p"));
    let types = schema_types(p.schema);
    let is_plain_string = types == ["string"] && p.schema.get("enum").is_none();
    if is_plain_string {
        let body = l.fresh("qstr");
        l.rule_attr(
            &body,
            &format!("suffix={}", lit("</parameter>")),
            "/(.|\\n)*/",
        );
        l.rule(
            &name,
            &format!("{} {body}", lit(&format!("<parameter={}>", p.name))),
        );
    } else {
        l.rule(
            &name,
            &format!(
                "{} ws? {} ws? \"</parameter>\"",
                lit(&format!("<parameter={}>", p.name)),
                json_rule(p.schema)
            ),
        );
    }
    name
}

pub(crate) fn build(
    spec: &ToolCallSpec,
    env: &GrammarTokenizer,
) -> Result<ToolGrammar, GrammarError> {
    if spec.tools.is_empty() {
        return Err(GrammarError::InvalidSpec("tool set is empty".into()));
    }
    for t in &spec.tools {
        if t.name.is_empty() {
            return Err(GrammarError::InvalidSpec("tool with an empty name".into()));
        }
        if spec.tools.iter().filter(|o| o.name == t.name).count() > 1 {
            return Err(GrammarError::InvalidSpec(format!(
                "duplicate tool name {:?}",
                t.name
            )));
        }
    }
    let tools: Vec<&ToolDef> = match &spec.choice {
        ToolChoice::Named(name) => {
            let t = spec.tools.iter().find(|t| &t.name == name).ok_or_else(|| {
                GrammarError::InvalidSpec(format!("tool_choice names unknown tool {name:?}"))
            })?;
            vec![t]
        }
        _ => spec.tools.iter().collect(),
    };
    let lazy = spec.choice == ToolChoice::Auto;
    let mut l = Lark::new();
    let mut triggers = Vec::new();

    match spec.family {
        ToolFamily::Hermes => {
            let open = tok(env, "<tool_call>");
            let close = tok(env, "</tool_call>");
            l.rule("body", &json_rule(&json_call_schema(&tools, "arguments")));
            l.rule("call", &format!("{open} ws? body ws? {close}"));
            if lazy {
                triggers.push(b"<tool_call>".to_vec());
                l.rule("start", "ws? body ws? close_rest");
                l.rule("close_rest", &format!("{close} (ws? call)* ws?"));
            } else {
                l.rule("start", "call (ws? call)* ws?");
            }
        }
        ToolFamily::Llama3 => {
            let open = tok(env, "<|python_tag|>");
            l.rule("body", &json_rule(&json_call_schema(&tools, "parameters")));
            if lazy {
                triggers.push(b"<|python_tag|>".to_vec());
                l.rule("start", "ws? body ws?");
            } else {
                l.rule("start", &format!("{open} ws? body ws?"));
            }
        }
        ToolFamily::Mistral => {
            let open = tok(env, "[TOOL_CALLS]");
            let args = tok(env, "[ARGS]");
            let mut alts = Vec::new();
            for (i, t) in tools.iter().enumerate() {
                let name = format!("mcall{i}");
                l.rule(
                    &name,
                    &format!(
                        "{} {args} {}",
                        lit(&t.name),
                        json_rule(&args_schema(&t.parameters))
                    ),
                );
                alts.push(name);
            }
            l.rule("body", &alts.join(" | "));
            l.rule("call", &format!("{open} body"));
            if lazy {
                triggers.push(b"[TOOL_CALLS]".to_vec());
                l.rule("start", "body (ws? call)* ws?");
            } else {
                l.rule("start", "call (ws? call)* ws?");
            }
        }
        ToolFamily::Harmony => {
            let channel = tok(env, "<|channel|>");
            let constrain = tok(env, "<|constrain|>");
            let message = tok(env, "<|message|>");
            let call = tok(env, "<|call|>");
            let mut alts = Vec::new();
            for (i, t) in tools.iter().enumerate() {
                let name = format!("hcall{i}");
                l.rule(
                    &name,
                    &format!(
                        "{} (\" \" {constrain} \"json\")? {message} {} {call}",
                        lit(&t.name),
                        json_rule(&args_schema(&t.parameters))
                    ),
                );
                alts.push(name);
            }
            l.rule("body", &alts.join(" | "));
            let prefix = "commentary to=functions.";
            if lazy {
                let mut trig = b"<|channel|>".to_vec();
                trig.extend_from_slice(prefix.as_bytes());
                triggers.push(trig);
                l.rule("start", "body");
            } else {
                l.rule("start", &format!("{channel} {} body", lit(prefix)));
            }
        }
        ToolFamily::QwenXml => {
            let open = tok(env, "<tool_call>");
            let close = tok(env, "</tool_call>");
            let mut alts = Vec::new();
            for (i, t) in tools.iter().enumerate() {
                let params = object_params(&t.parameters);
                let mut rules = Vec::new();
                for p in &params {
                    let r = qwen_param(&mut l, p, i);
                    rules.push((r, p.required));
                }
                let inner = ordered_params(&mut l, &rules, "ws?");
                let name = format!("qfunc{i}");
                l.rule(
                    &name,
                    &format!(
                        "{} ws? {inner} ws? \"</function>\"",
                        lit(&format!("<function={}>", t.name))
                    ),
                );
                alts.push(name);
            }
            l.rule("func", &alts.join(" | "));
            l.rule("call", &format!("{open} ws? func ws? {close}"));
            if lazy {
                triggers.push(b"<tool_call>".to_vec());
                l.rule("start", &format!("ws? func ws? {close} (ws? call)* ws?"));
            } else {
                l.rule("start", "call (ws? call)* ws?");
            }
        }
        ToolFamily::Gemma4 => {
            let open = tok(env, "<|tool_call>");
            let close = tok(env, "<tool_call|>");
            let mut alts = Vec::new();
            for (i, t) in tools.iter().enumerate() {
                let params = object_params(&t.parameters);
                let mut rules = Vec::new();
                for p in &params {
                    let v = gemma_value(&mut l, env, p.schema, 1);
                    let pr = l.fresh("gparam");
                    l.rule(&pr, &format!("{} \":\" {v}", lit(p.name)));
                    rules.push((pr, p.required));
                }
                let inner = ordered_params(&mut l, &rules, "\",\" ws?");
                let name = format!("gfunc{i}");
                l.rule(
                    &name,
                    &format!(
                        "{} \"{{\" ws? {inner} ws? \"}}\"",
                        lit(&format!("call:{}", t.name))
                    ),
                );
                alts.push(name);
            }
            l.rule("func", &alts.join(" | "));
            l.rule("call", &format!("{open} func {close}"));
            if lazy {
                triggers.push(b"<|tool_call>".to_vec());
                l.rule("start", &format!("func {close} (ws? call)* ws?"));
            } else {
                l.rule("start", "call (ws? call)* ws?");
            }
        }
    }
    Ok(ToolGrammar {
        lark: l.out,
        triggers,
    })
}
