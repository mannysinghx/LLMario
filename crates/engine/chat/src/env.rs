//! The Jinja environment, configured to match `transformers.utils.chat_template_utils`.
//!
//! transformers compiles every chat template with
//! `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
//! extensions=[AssistantTracker, loopcontrols])`, overrides the `tojson` filter with
//! `json.dumps(ensure_ascii=False, ...)` and adds the globals `raise_exception` and
//! `strftime_now`. Everything else is stock Jinja2 (Python semantics). This module reproduces
//! that: the knobs minijinja exposes directly, plus the Python behaviours it lacks.

use crate::pyfmt::{json_dumps, py_str, JsonOpts};
use chrono::{DateTime, FixedOffset, Local};
use minijinja::value::{Kwargs, Rest, Value, ValueKind};
use minijinja::{AutoEscape, Environment, Error, ErrorKind, UndefinedBehavior};
use std::fmt;

/// Marker attached (as the error source) to errors produced by `raise_exception`, so the
/// renderer can tell "the template refused this input" from "the template is broken".
#[derive(Debug)]
pub struct RaisedException(pub String);

impl fmt::Display for RaisedException {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RaisedException {}

/// Walk the error chain and return the message if the error came from `raise_exception`.
pub fn raised_message(err: &Error) -> Option<String> {
    let mut source: Option<&(dyn std::error::Error + 'static)> = std::error::Error::source(err);
    while let Some(s) = source {
        if let Some(raised) = s.downcast_ref::<RaisedException>() {
            return Some(raised.0.clone());
        }
        source = s.source();
    }
    None
}

/// Build the environment. `now` fixes `strftime_now`; `None` uses the wall clock (local time,
/// like Python's `datetime.now()`).
pub fn build_environment(now: Option<DateTime<FixedOffset>>) -> Environment<'static> {
    let mut env = Environment::new();
    // Environment knobs transformers sets explicitly.
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    // Jinja2 defaults that transformers leaves alone.
    env.set_keep_trailing_newline(false);
    env.set_undefined_behavior(UndefinedBehavior::Lenient);
    env.set_auto_escape_callback(|_| AutoEscape::None);
    // Python method calls on str/dict/list (`.strip()`, `.split()`, `.items()`, `.get()`, ...).
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    // `{{ x }}` prints with Python `str()` semantics (True/None/1e-05/{'k': 'v'}).
    env.set_formatter(|out, _state, value| {
        out.write_str(&py_str(value))
            .map_err(|e| Error::new(ErrorKind::WriteFailure, "write failed").with_source(e))
    });

    env.add_filter("tojson", tojson);
    env.add_filter("to_json", tojson);
    env.add_filter("string", string);
    env.add_filter("join", join);
    env.add_test("sequence", is_sequence);
    env.add_test("number", is_number);
    env.add_function("raise_exception", raise_exception);
    env.add_function("strftime_now", move |format: String| {
        strftime_now(now, &format)
    });
    env
}

/// transformers' `{% generation %}...{% endgeneration %}` extension marks assistant spans for
/// token masks; at render time the body passes through unchanged. minijinja has no custom
/// tags, so the tags are rewritten to `if true`/`endif`, which has identical whitespace-control
/// semantics (both are block tags) and renders the body unchanged.
pub fn rewrite_generation_tags(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(start) = rest.find("{%") {
        out.push_str(&rest[..start]);
        let after = &rest[start..];
        match after.find("%}") {
            None => {
                out.push_str(after);
                return out;
            }
            Some(end) => {
                let tag = &after[..end + 2];
                out.push_str(&rewrite_tag(tag));
                rest = &after[end + 2..];
            }
        }
    }
    out.push_str(rest);
    out
}

fn rewrite_tag(tag: &str) -> String {
    // tag is "{%" [-+] ws name ws [-+] "%}"
    let inner = &tag[2..tag.len() - 2];
    let lead = inner.chars().next().filter(|c| matches!(c, '-' | '+'));
    let trail = inner.chars().last().filter(|c| matches!(c, '-' | '+'));
    let body = inner
        .strip_prefix(['-', '+'])
        .unwrap_or(inner)
        .strip_suffix(['-', '+'])
        .unwrap_or(inner)
        .trim();
    let body = match lead {
        Some(_) => body.trim_start_matches(['-', '+']).trim(),
        None => body,
    };
    let replacement = match body {
        "generation" => "if true",
        "endgeneration" => "endif",
        _ => return tag.to_string(),
    };
    let mut s = String::from("{%");
    if let Some(c) = lead {
        s.push(c);
    }
    s.push(' ');
    s.push_str(replacement);
    s.push(' ');
    if let Some(c) = trail {
        s.push(c);
    }
    s.push_str("%}");
    s
}

fn raise_exception(message: Value) -> Result<Value, Error> {
    let text = py_str(&message);
    Err(Error::new(ErrorKind::InvalidOperation, text.clone()).with_source(RaisedException(text)))
}

fn strftime_now(now: Option<DateTime<FixedOffset>>, format: &str) -> Result<Value, Error> {
    let now = now.unwrap_or_else(|| Local::now().fixed_offset());
    let mut s = String::new();
    use std::fmt::Write as _;
    write!(s, "{}", now.format(format)).map_err(|_| {
        Error::new(
            ErrorKind::InvalidOperation,
            format!("strftime_now: unsupported format {format:?}"),
        )
    })?;
    Ok(Value::from(s))
}

/// `tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False)` as
/// transformers defines it (positional or keyword arguments, like the Python signature).
fn tojson(value: &Value, args: Rest<Value>, kwargs: Kwargs) -> Result<Value, Error> {
    let positional = &args.0;
    let pick = |idx: usize, name: &str| -> Result<Option<Value>, Error> {
        if let Some(v) = positional.get(idx) {
            return Ok(Some(v.clone()));
        }
        kwargs.get::<Option<Value>>(name)
    };
    let ensure_ascii = pick(0, "ensure_ascii")?
        .map(|v| v.is_true())
        .unwrap_or(false);
    let indent = match pick(1, "indent")? {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) => Some(match v.as_str() {
            Some(s) => s.to_string(),
            None => {
                let n = usize::try_from(v.clone())?;
                " ".repeat(n)
            }
        }),
    };
    let separators = match pick(2, "separators")? {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) => {
            let parts: Vec<Value> = v.try_iter()?.collect();
            if parts.len() != 2 {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "tojson: separators must be a (item, key) pair",
                ));
            }
            Some((py_str(&parts[0]), py_str(&parts[1])))
        }
    };
    let sort_keys = pick(3, "sort_keys")?.map(|v| v.is_true()).unwrap_or(false);
    kwargs.assert_all_used()?;
    let opts = JsonOpts::resolve(ensure_ascii, indent, separators, sort_keys);
    Ok(Value::from(json_dumps(value, &opts)?))
}

/// Jinja2's `string` filter is `str(value)`.
fn string(value: &Value) -> Value {
    if value.kind() == ValueKind::String {
        value.clone()
    } else {
        Value::from(py_str(value))
    }
}

/// Jinja2's `join(value, d='', attribute=None)`: items go through `str()`.
fn join(value: &Value, d: Option<&str>, kwargs: Kwargs) -> Result<Value, Error> {
    let attribute: Option<String> = kwargs.get("attribute")?;
    kwargs.assert_all_used()?;
    let joiner = d.unwrap_or("");
    let mut parts = Vec::new();
    for item in value.try_iter()? {
        let item = match &attribute {
            Some(attr) => item.get_attr(attr)?,
            None => item,
        };
        parts.push(py_str(&item));
    }
    Ok(Value::from(parts.join(joiner)))
}

/// Jinja2's `sequence` test is "has `len()` and `__getitem__`": true for lists, dicts and
/// strings alike.
fn is_sequence(value: &Value) -> bool {
    matches!(
        value.kind(),
        ValueKind::Seq | ValueKind::Map | ValueKind::String
    )
}

/// Jinja2's `number` test is `isinstance(value, Number)`, and Python's `bool` is a `Number`.
fn is_number(value: &Value) -> bool {
    matches!(value.kind(), ValueKind::Number | ValueKind::Bool)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_tags_become_if_blocks() {
        assert_eq!(
            rewrite_generation_tags("a{% generation %}b{% endgeneration %}c"),
            "a{% if true %}b{% endif %}c"
        );
        assert_eq!(
            rewrite_generation_tags("{%- generation -%}x{%+ endgeneration %}"),
            "{%- if true -%}x{%+ endif %}"
        );
        assert_eq!(
            rewrite_generation_tags("{% if x %}{{ y }}{% endif %}"),
            "{% if x %}{{ y }}{% endif %}"
        );
        assert_eq!(rewrite_generation_tags("{% broken"), "{% broken");
    }
}
