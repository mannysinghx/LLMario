//! Tool definitions as the parser sees them, JSON-Schema-driven typing of text argument
//! values, and a minimal validator with a repair-instruction generator.
//!
//! The validator covers `type` (string or list), `required`, `properties`, `enum`, `items`
//! and OpenAPI-style `nullable`; nothing else is interpreted. That is enough to catch what
//! small models actually get wrong (missing required fields, strings where numbers belong,
//! values outside an enum) without an external crate.

use super::scan::number_from_text;
use serde_json::{Map, Value};

/// One function tool: its name and `parameters` JSON Schema (an empty object when absent).
#[derive(Clone, Debug, PartialEq)]
pub struct ToolSpec {
    pub name: String,
    pub parameters: Value,
}

impl ToolSpec {
    /// The schema of top-level parameter `key`, if declared.
    pub fn property(&self, key: &str) -> Option<&Value> {
        self.parameters.get("properties")?.get(key)
    }
}

/// The tools a request carried, in OpenAI shape (`{"type": "function", "function": {...}}`,
/// bare `{"name": ..., "parameters": ...}` objects are accepted too).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ToolTable {
    specs: Vec<ToolSpec>,
}

impl ToolTable {
    pub fn from_tools(tools: Option<&Value>) -> ToolTable {
        let mut specs = Vec::new();
        if let Some(Value::Array(items)) = tools {
            for item in items {
                let function = item.get("function").unwrap_or(item);
                let Some(name) = function.get("name").and_then(Value::as_str) else {
                    continue;
                };
                let parameters = function
                    .get("parameters")
                    .cloned()
                    .unwrap_or_else(|| Value::Object(Map::new()));
                specs.push(ToolSpec {
                    name: name.to_string(),
                    parameters,
                });
            }
        }
        ToolTable { specs }
    }

    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&ToolSpec> {
        self.specs.iter().find(|s| s.name == name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.specs.iter().map(|s| s.name.as_str())
    }
}

/// The declared `type`s of a schema (`"integer"` or `["string", "null"]`), plus `"null"` when
/// `nullable` is true.
fn declared_types(schema: &Value) -> Vec<String> {
    let mut out = Vec::new();
    match schema.get("type") {
        Some(Value::String(t)) => out.push(t.clone()),
        Some(Value::Array(ts)) => out.extend(ts.iter().filter_map(Value::as_str).map(String::from)),
        _ => {}
    }
    if schema.get("nullable").and_then(Value::as_bool) == Some(true)
        && !out.iter().any(|t| t == "null")
    {
        out.push("null".to_string());
    }
    if out.is_empty() && schema.get("properties").is_some() {
        out.push("object".to_string());
    }
    out
}

/// Turn the text of a tagged-format argument (Qwen XML `<parameter>`, GLM `<arg_value>`) into
/// a typed JSON value.
///
/// With a schema, each declared type is tried in order: `null` accepts `null`/`None`;
/// `boolean` accepts `true`/`false`/`True`/`False`; `integer` and `number` parse numerals
/// (Python `str()` spellings such as `1e-05` and `1e+21` included); `object` and `array`
/// parse JSON text of that shape; `string` keeps the text. When nothing matches the text is
/// kept as a string. Without a schema the text is parsed as JSON when it is valid JSON and
/// kept as a string otherwise (the Qwen3-Coder reference parser's rule).
pub fn type_text(raw: &str, schema: Option<&Value>) -> Value {
    let trimmed = raw.trim();
    let Some(schema) = schema else {
        return serde_json::from_str::<Value>(trimmed)
            .unwrap_or_else(|_| Value::String(raw.to_string()));
    };
    let types = declared_types(schema);
    if types.is_empty() {
        return serde_json::from_str::<Value>(trimmed)
            .unwrap_or_else(|_| Value::String(raw.to_string()));
    }
    for t in &types {
        let converted = match t.as_str() {
            "null" => matches!(trimmed, "null" | "None").then_some(Value::Null),
            "boolean" => match trimmed {
                "true" | "True" => Some(Value::Bool(true)),
                "false" | "False" => Some(Value::Bool(false)),
                _ => None,
            },
            "integer" => number_from_text(trimmed).filter(|v| {
                v.as_i64().is_some()
                    || v.as_u64().is_some()
                    || v.as_f64().is_some_and(|f| f.fract() == 0.0)
            }),
            "number" => number_from_text(trimmed),
            "object" => serde_json::from_str::<Value>(trimmed)
                .ok()
                .filter(Value::is_object),
            "array" => serde_json::from_str::<Value>(trimmed)
                .ok()
                .filter(Value::is_array),
            "string" => Some(Value::String(raw.to_string())),
            _ => None,
        };
        if let Some(v) = converted {
            return v;
        }
    }
    Value::String(raw.to_string())
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() || n.as_f64().is_some_and(|f| f.fract() == 0.0) {
                "integer"
            } else {
                "number"
            }
        }
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn type_matches(declared: &str, v: &Value) -> bool {
    match declared {
        "number" => v.is_number(),
        "integer" => type_name(v) == "integer",
        other => type_name(v) == other,
    }
}

/// Validate `value` against `schema`, appending one human-readable violation per problem.
/// `path` names the value for the messages (`arguments`, `arguments.route.from`, ...).
pub fn validate(schema: &Value, value: &Value, path: &str, out: &mut Vec<String>) {
    let types = declared_types(schema);
    if !types.is_empty() && !types.iter().any(|t| type_matches(t, value)) {
        out.push(format!(
            "`{path}` must be of type {} (got {})",
            types.join(" or "),
            type_name(value)
        ));
        return;
    }
    if let Some(Value::Array(allowed)) = schema.get("enum") {
        if !allowed.contains(value) {
            let list: Vec<String> = allowed.iter().map(Value::to_string).collect();
            out.push(format!(
                "`{path}` must be one of {} (got {value})",
                list.join(", ")
            ));
        }
    }
    if let Value::Object(fields) = value {
        if let Some(Value::Array(required)) = schema.get("required") {
            for r in required.iter().filter_map(Value::as_str) {
                if !fields.contains_key(r) {
                    out.push(format!("`{path}.{r}` is required but missing"));
                }
            }
        }
        if let Some(Value::Object(props)) = schema.get("properties") {
            for (k, v) in fields {
                if let Some(sub) = props.get(k) {
                    validate(sub, v, &format!("{path}.{k}"), out);
                }
            }
        }
    }
    if let (Value::Array(items), Some(item_schema)) = (value, schema.get("items")) {
        if item_schema.is_object() {
            for (i, item) in items.iter().enumerate() {
                validate(item_schema, item, &format!("{path}[{i}]"), out);
            }
        }
    }
}

/// Safe, lossless coercions a model commonly needs: a string holding a numeral where a
/// number is declared (`"10"` → 10), a string `true`/`false` where a boolean is declared, a
/// string holding JSON where an object or array is declared, and `"null"`/`"None"` where
/// null is allowed. Everything else is returned unchanged. Recurses through `properties` and
/// `items`.
pub fn coerce(schema: &Value, value: Value) -> Value {
    let types = declared_types(schema);
    let value = match value {
        Value::String(s) if !types.is_empty() && !types.iter().any(|t| t == "string") => {
            let t = s.trim();
            let mut converted = None;
            for ty in &types {
                converted = match ty.as_str() {
                    "null" => matches!(t, "null" | "None").then_some(Value::Null),
                    "boolean" => match t {
                        "true" | "True" => Some(Value::Bool(true)),
                        "false" | "False" => Some(Value::Bool(false)),
                        _ => None,
                    },
                    "integer" | "number" => number_from_text(t).filter(|v| type_matches(ty, v)),
                    "object" => serde_json::from_str::<Value>(t)
                        .ok()
                        .filter(Value::is_object),
                    "array" => serde_json::from_str::<Value>(t)
                        .ok()
                        .filter(Value::is_array),
                    _ => None,
                };
                if converted.is_some() {
                    break;
                }
            }
            converted.unwrap_or(Value::String(s))
        }
        other => other,
    };
    match value {
        Value::Object(fields) => {
            let props = schema.get("properties").and_then(Value::as_object);
            let mut out = Map::new();
            for (k, v) in fields {
                let v = match props.and_then(|p| p.get(&k)) {
                    Some(sub) => coerce(sub, v),
                    None => v,
                };
                out.insert(k, v);
            }
            Value::Object(out)
        }
        Value::Array(items) => match schema.get("items") {
            Some(item_schema) if item_schema.is_object() => {
                Value::Array(items.into_iter().map(|i| coerce(item_schema, i)).collect())
            }
            _ => Value::Array(items),
        },
        other => other,
    }
}

/// Checks a finished tool call against its declared schema and, when it does not validate,
/// produces a short instruction the agent loop can feed back to the model.
#[derive(Clone, Debug, Default)]
pub struct ToolCallRepair {
    table: ToolTable,
}

/// Outcome of [`ToolCallRepair::check`].
#[derive(Clone, Debug, PartialEq)]
pub enum RepairVerdict {
    /// The arguments validate (possibly after coercion; the coerced arguments are returned).
    Valid(Value),
    /// The arguments do not validate; `instruction` is the text to send back to the model.
    Repair {
        violations: Vec<String>,
        instruction: String,
    },
    /// The tool is not declared at all.
    UnknownTool { instruction: String },
}

impl ToolCallRepair {
    pub fn new(tools: Option<&Value>) -> ToolCallRepair {
        ToolCallRepair {
            table: ToolTable::from_tools(tools),
        }
    }

    pub fn from_table(table: ToolTable) -> ToolCallRepair {
        ToolCallRepair { table }
    }

    /// Validate `arguments` for tool `name`. Coercion (see [`coerce`]) runs first, so a
    /// `"10"` where an integer is declared is repaired silently rather than bounced.
    pub fn check(&self, name: &str, arguments: &Value) -> RepairVerdict {
        let Some(spec) = self.table.get(name) else {
            let known: Vec<&str> = self.table.names().collect();
            return RepairVerdict::UnknownTool {
                instruction: format!(
                    "There is no tool named `{name}`. The available tools are: {}. Reply with a call to one of them, in the same format, or answer without calling a tool.",
                    if known.is_empty() { "(none)".to_string() } else { known.join(", ") }
                ),
            };
        };
        let coerced = coerce(&spec.parameters, arguments.clone());
        let mut violations = Vec::new();
        validate(&spec.parameters, &coerced, "arguments", &mut violations);
        if violations.is_empty() {
            return RepairVerdict::Valid(coerced);
        }
        let instruction = format!(
            "The call to `{name}` was invalid: {}. Reply with a corrected call to `{name}` in the same format, with every required parameter and values of the declared types.",
            violations.join("; ")
        );
        RepairVerdict::Repair {
            violations,
            instruction,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tools() -> Value {
        json!([{"type": "function", "function": {"name": "get_weather", "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                "days": {"type": "integer"},
                "max_price": {"type": "number", "nullable": true},
                "flag": {"type": "boolean"},
                "route": {"type": "object", "properties": {"from": {"type": "string"}, "via": {"type": "array", "items": {"type": "string"}}}, "required": ["from"]},
                "passengers": {"type": "array", "items": {"type": "object", "properties": {"age": {"type": "integer"}}, "required": ["age"]}}
            },
            "required": ["location"]
        }}}])
    }

    #[test]
    fn table_and_typing() {
        let t = ToolTable::from_tools(Some(&tools()));
        let spec = t.get("get_weather").unwrap();
        assert_eq!(
            type_text("Paris", spec.property("location")),
            json!("Paris")
        );
        assert_eq!(type_text("3", spec.property("days")), json!(3));
        assert_eq!(type_text("3.5", spec.property("days")), json!("3.5"));
        assert_eq!(type_text("899.5", spec.property("max_price")), json!(899.5));
        assert_eq!(type_text("None", spec.property("max_price")), json!(null));
        assert_eq!(type_text("1e-05", spec.property("max_price")), json!(1e-5));
        assert_eq!(type_text("True", spec.property("flag")), json!(true));
        assert_eq!(type_text("maybe", spec.property("flag")), json!("maybe"));
        assert_eq!(
            type_text("{\"from\": \"LHR\", \"via\": []}", spec.property("route")),
            json!({"from": "LHR", "via": []})
        );
        assert_eq!(
            type_text("[{\"age\": 1}]", spec.property("passengers")),
            json!([{"age": 1}])
        );
        assert_eq!(type_text("12", spec.property("location")), json!("12"));
        // No schema: JSON when valid, string otherwise.
        assert_eq!(type_text("12", None), json!(12));
        assert_eq!(type_text("\"q\"", None), json!("q"));
        assert_eq!(type_text("plain text", None), json!("plain text"));
        assert_eq!(type_text("7", spec.property("unknown")), json!(7));
        // Bare tool objects are accepted too.
        let bare = ToolTable::from_tools(Some(
            &json!([{"name": "f", "parameters": {"type": "object"}}]),
        ));
        assert!(bare.get("f").is_some());
        assert!(ToolTable::from_tools(None).is_empty());
    }

    #[test]
    fn validator_reports_each_problem() {
        let t = ToolTable::from_tools(Some(&tools()));
        let schema = &t.get("get_weather").unwrap().parameters;
        let mut v = Vec::new();
        validate(
            schema,
            &json!({"unit": "kelvin", "days": "3", "route": {"via": [1]}, "passengers": [{}]}),
            "arguments",
            &mut v,
        );
        assert_eq!(
            v,
            vec![
                "`arguments.location` is required but missing",
                "`arguments.unit` must be one of \"celsius\", \"fahrenheit\" (got \"kelvin\")",
                "`arguments.days` must be of type integer (got string)",
                "`arguments.route.from` is required but missing",
                "`arguments.route.via[0]` must be of type string (got integer)",
                "`arguments.passengers[0].age` is required but missing",
            ]
        );
        let mut ok = Vec::new();
        validate(
            schema,
            &json!({"location": "Paris", "days": 2.0, "max_price": null, "flag": false}),
            "arguments",
            &mut ok,
        );
        assert!(ok.is_empty(), "{ok:?}");
    }

    #[test]
    fn coercion_and_repair() {
        let r = ToolCallRepair::new(Some(&tools()));
        match r.check("get_weather", &json!({"location": "Paris", "days": "3", "flag": "true", "max_price": "None", "route": "{\"from\": \"A\"}"})) {
            RepairVerdict::Valid(v) => assert_eq!(
                v,
                json!({"location": "Paris", "days": 3, "flag": true, "max_price": null, "route": {"from": "A"}})
            ),
            other => panic!("{other:?}"),
        }
        match r.check("get_weather", &json!({"unit": "kelvin"})) {
            RepairVerdict::Repair {
                violations,
                instruction,
            } => {
                assert_eq!(violations.len(), 2);
                assert!(instruction.starts_with("The call to `get_weather` was invalid: `arguments.location` is required but missing; `arguments.unit` must be one of"));
                assert!(instruction.ends_with("in the same format, with every required parameter and values of the declared types."));
            }
            other => panic!("{other:?}"),
        }
        match r.check("nope", &json!({})) {
            RepairVerdict::UnknownTool { instruction } => {
                assert!(instruction.contains("The available tools are: get_weather."));
            }
            other => panic!("{other:?}"),
        }
    }
}
