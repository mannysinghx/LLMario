//! Python-faithful text formatting of template values.
//!
//! HuggingFace renders chat templates with Python Jinja2, so every place a value becomes text
//! follows CPython rules: `{{ true }}` prints `True`, `{{ none }}` prints `None`, `1e-5` prints
//! `1e-05`, `str(dict)` uses single quotes and `tojson` is `json.dumps(ensure_ascii=False)`
//! with `, ` and `: ` separators. minijinja's own `Display` prints `true`/`none` and compact
//! JSON, so this module re-implements `str()`, `repr()`, `float.__repr__` and `json.dumps`
//! over [`minijinja::value::Value`].

use minijinja::value::{Value, ValueKind};
use minijinja::{Error, ErrorKind};
use std::fmt::Write as _;

/// `str(value)` as CPython prints it. Strings are returned unchanged, Jinja2's undefined prints
/// as the empty string, everything else goes through [`py_repr`].
pub fn py_str(v: &Value) -> String {
    match v.kind() {
        ValueKind::String => v.as_str().unwrap_or("").to_string(),
        ValueKind::Undefined => String::new(),
        _ => py_repr(v),
    }
}

/// `repr(value)` as CPython prints it (used for `str()` of containers).
pub fn py_repr(v: &Value) -> String {
    let mut out = String::new();
    write_repr(&mut out, v);
    out
}

fn write_repr(out: &mut String, v: &Value) {
    match v.kind() {
        // `repr(jinja2.Undefined())` is "Undefined"; `str()` of it is handled by `py_str`.
        ValueKind::Undefined => out.push_str("Undefined"),
        ValueKind::None => out.push_str("None"),
        ValueKind::Bool => out.push_str(if v.is_true() { "True" } else { "False" }),
        ValueKind::Number => write_number(out, v),
        ValueKind::String => write_str_repr(out, v.as_str().unwrap_or("")),
        ValueKind::Bytes => {
            out.push_str("b'");
            for b in v.as_bytes().unwrap_or(&[]) {
                match *b {
                    b'\\' => out.push_str("\\\\"),
                    b'\'' => out.push_str("\\'"),
                    b'\n' => out.push_str("\\n"),
                    b'\r' => out.push_str("\\r"),
                    b'\t' => out.push_str("\\t"),
                    0x20..=0x7e => out.push(*b as char),
                    _ => {
                        let _ = write!(out, "\\x{b:02x}");
                    }
                }
            }
            out.push('\'');
        }
        ValueKind::Seq | ValueKind::Iterable => {
            out.push('[');
            if let Ok(iter) = v.try_iter() {
                for (i, item) in iter.enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    write_repr(out, &item);
                }
            }
            out.push(']');
        }
        ValueKind::Map => {
            out.push('{');
            if let Ok(iter) = v.try_iter() {
                for (i, key) in iter.enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    write_repr(out, &key);
                    out.push_str(": ");
                    let item = v.get_item(&key).unwrap_or(Value::UNDEFINED);
                    write_repr(out, &item);
                }
            }
            out.push('}');
        }
        _ => out.push_str(&v.to_string()),
    }
}

fn write_number(out: &mut String, v: &Value) {
    if v.is_integer() {
        // minijinja prints integers (i64/u64/i128) plainly, exactly like Python.
        out.push_str(&v.to_string());
    } else if let Ok(f) = f64::try_from(v.clone()) {
        out.push_str(&py_float_repr(f));
    } else {
        out.push_str(&v.to_string());
    }
}

/// `float.__repr__`: shortest round-trip digits, positional notation for decimal exponents in
/// `-4 <= e < 16`, otherwise `d.ddde±XX` with a sign and at least two exponent digits.
pub fn py_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "nan".to_string();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    let (negative, digits, exp) = shortest_digits(f);
    let mut out = String::new();
    if negative {
        out.push('-');
    }
    if (-4..16).contains(&exp) {
        if exp >= 0 {
            let int_len = (exp + 1) as usize;
            if digits.len() <= int_len {
                out.push_str(&digits);
                for _ in digits.len()..int_len {
                    out.push('0');
                }
                out.push_str(".0");
            } else {
                out.push_str(&digits[..int_len]);
                out.push('.');
                out.push_str(&digits[int_len..]);
            }
        } else {
            out.push_str("0.");
            for _ in 0..(-exp - 1) {
                out.push('0');
            }
            out.push_str(&digits);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(out, "e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    }
    out
}

/// Decompose `f` into (sign, shortest round-trip digit string without leading/trailing zeros,
/// decimal exponent of the first digit). `0.0` yields `("0", 0)`.
fn shortest_digits(f: f64) -> (bool, String, i32) {
    // Rust's `{:e}` is shortest-round-trip, like CPython's repr.
    let s = format!("{:e}", f);
    let (mantissa, exp) = s.split_once('e').unwrap_or((&s, "0"));
    let negative = mantissa.starts_with('-');
    let mantissa = mantissa.trim_start_matches('-');
    let mut digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let mut exp: i32 = exp.parse().unwrap_or(0);
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    if digits.is_empty() || digits == "0" {
        digits = "0".to_string();
        exp = 0;
    }
    (negative, digits, exp)
}

/// Characters CPython's `str.isprintable()` rejects among the non-ASCII code points that
/// commonly show up in chat content. (Full Unicode category tables are out of scope; everything
/// else at or above U+00A1 is treated as printable, which matches CPython for letters, digits,
/// punctuation, symbols and emoji.)
fn is_py_printable(c: char) -> bool {
    let cp = c as u32;
    if cp < 0x20 || cp == 0x7f {
        return false;
    }
    if cp < 0x7f {
        return true;
    }
    !matches!(
        cp,
        0x80..=0xa0
            | 0xad
            | 0x600..=0x605
            | 0x61c
            | 0x6dd
            | 0x70f
            | 0x1680
            | 0x180e
            | 0x2000..=0x200f
            | 0x2028..=0x202f
            | 0x205f..=0x2064
            | 0x2066..=0x206f
            | 0x3000
            | 0xd800..=0xdfff
            | 0xe000..=0xf8ff
            | 0xfeff
            | 0xfff9..=0xfffb
            | 0xfffe
            | 0xffff
    )
}

/// `repr(str)`: single quotes unless the text has a single quote and no double quote.
fn write_str_repr(out: &mut String, s: &str) {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if is_py_printable(c) => out.push(c),
            c => {
                let cp = c as u32;
                if cp < 0x100 {
                    let _ = write!(out, "\\x{cp:02x}");
                } else if cp < 0x10000 {
                    let _ = write!(out, "\\u{cp:04x}");
                } else {
                    let _ = write!(out, "\\U{cp:08x}");
                }
            }
        }
    }
    out.push(quote);
}

/// Options of Python's `json.dumps` that chat templates use.
#[derive(Clone, Debug)]
pub struct JsonOpts {
    pub ensure_ascii: bool,
    pub indent: Option<String>,
    pub item_separator: String,
    pub key_separator: String,
    pub sort_keys: bool,
}

impl JsonOpts {
    /// Resolve the options the way `json.dumps` does: explicit `separators` win, otherwise
    /// `indent` switches the item separator from `", "` to `","`.
    pub fn resolve(
        ensure_ascii: bool,
        indent: Option<String>,
        separators: Option<(String, String)>,
        sort_keys: bool,
    ) -> JsonOpts {
        let (item_separator, key_separator) = match separators {
            Some(seps) => seps,
            None if indent.is_some() => (",".to_string(), ": ".to_string()),
            None => (", ".to_string(), ": ".to_string()),
        };
        JsonOpts {
            ensure_ascii,
            indent,
            item_separator,
            key_separator,
            sort_keys,
        }
    }
}

impl Default for JsonOpts {
    fn default() -> Self {
        JsonOpts::resolve(false, None, None, false)
    }
}

/// `json.dumps(value, **opts)`.
pub fn json_dumps(v: &Value, opts: &JsonOpts) -> Result<String, Error> {
    let mut out = String::new();
    write_json(&mut out, v, opts, 0)?;
    Ok(out)
}

fn not_serializable(what: &str) -> Error {
    Error::new(
        ErrorKind::InvalidOperation,
        format!("Object of type {what} is not JSON serializable"),
    )
}

fn newline_indent(out: &mut String, opts: &JsonOpts, level: usize) {
    if let Some(ind) = &opts.indent {
        out.push('\n');
        for _ in 0..level {
            out.push_str(ind);
        }
    }
}

fn write_json(out: &mut String, v: &Value, opts: &JsonOpts, level: usize) -> Result<(), Error> {
    match v.kind() {
        ValueKind::Undefined => Err(not_serializable("Undefined")),
        ValueKind::None => {
            out.push_str("null");
            Ok(())
        }
        ValueKind::Bool => {
            out.push_str(if v.is_true() { "true" } else { "false" });
            Ok(())
        }
        ValueKind::Number => {
            if v.is_integer() {
                out.push_str(&v.to_string());
            } else {
                let f = f64::try_from(v.clone()).map_err(|_| not_serializable("number"))?;
                if f.is_nan() {
                    out.push_str("NaN");
                } else if f.is_infinite() {
                    out.push_str(if f > 0.0 { "Infinity" } else { "-Infinity" });
                } else {
                    out.push_str(&py_float_repr(f));
                }
            }
            Ok(())
        }
        ValueKind::String => {
            write_json_string(out, v.as_str().unwrap_or(""), opts.ensure_ascii);
            Ok(())
        }
        ValueKind::Bytes => Err(not_serializable("bytes")),
        ValueKind::Seq | ValueKind::Iterable => {
            let items: Vec<Value> = v.try_iter()?.collect();
            if items.is_empty() {
                out.push_str("[]");
                return Ok(());
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(&opts.item_separator);
                }
                newline_indent(out, opts, level + 1);
                write_json(out, item, opts, level + 1)?;
            }
            newline_indent(out, opts, level);
            out.push(']');
            Ok(())
        }
        ValueKind::Map => {
            let mut pairs: Vec<(String, Value)> = Vec::new();
            for key in v.try_iter()? {
                let key_text = json_key(&key)?;
                let item = v.get_item(&key).unwrap_or(Value::UNDEFINED);
                pairs.push((key_text, item));
            }
            if opts.sort_keys {
                pairs.sort_by(|a, b| a.0.cmp(&b.0));
            }
            if pairs.is_empty() {
                out.push_str("{}");
                return Ok(());
            }
            out.push('{');
            for (i, (key, item)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push_str(&opts.item_separator);
                }
                newline_indent(out, opts, level + 1);
                write_json_string(out, key, opts.ensure_ascii);
                out.push_str(&opts.key_separator);
                write_json(out, item, opts, level + 1)?;
            }
            newline_indent(out, opts, level);
            out.push('}');
            Ok(())
        }
        other => Err(not_serializable(&other.to_string())),
    }
}

/// `json.dumps` coerces non-string keys: `True`→`"true"`, `None`→`"null"`, numbers→repr.
fn json_key(key: &Value) -> Result<String, Error> {
    match key.kind() {
        ValueKind::String => Ok(key.as_str().unwrap_or("").to_string()),
        ValueKind::Bool => Ok(if key.is_true() { "true" } else { "false" }.to_string()),
        ValueKind::None => Ok("null".to_string()),
        ValueKind::Number => {
            let mut s = String::new();
            write_number(&mut s, key);
            Ok(s)
        }
        other => Err(Error::new(
            ErrorKind::InvalidOperation,
            format!("keys must be str, int, float, bool or None, not {other}"),
        )),
    }
}

fn write_json_string(out: &mut String, s: &str, ensure_ascii: bool) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c if ensure_ascii && (c as u32) > 0x7e => {
                let cp = c as u32;
                if cp >= 0x10000 {
                    let v = cp - 0x10000;
                    let hi = 0xd800 | (v >> 10);
                    let lo = 0xdc00 | (v & 0x3ff);
                    let _ = write!(out, "\\u{hi:04x}\\u{lo:04x}");
                } else {
                    let _ = write!(out, "\\u{cp:04x}");
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_matches_cpython() {
        let cases: &[(f64, &str)] = &[
            (1.0, "1.0"),
            (0.5, "0.5"),
            (-0.0, "-0.0"),
            (0.0, "0.0"),
            (100.0, "100.0"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.5e16, "1.5e+16"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.23456e-7, "1.23456e-07"),
            (123456789.123, "123456789.123"),
            (1e100, "1e+100"),
            (0.1 + 0.2, "0.30000000000000004"),
            (2.5e-5, "2.5e-05"),
        ];
        for (f, want) in cases {
            assert_eq!(py_float_repr(*f), *want, "repr({f:?})");
        }
    }

    #[test]
    fn str_repr_quotes_like_cpython() {
        let mut s = String::new();
        write_str_repr(&mut s, "it's");
        assert_eq!(s, "\"it's\"");
        s.clear();
        write_str_repr(&mut s, "say \"hi\"");
        assert_eq!(s, "'say \"hi\"'");
        s.clear();
        write_str_repr(&mut s, "both ' and \"");
        assert_eq!(s, "'both \\' and \"'");
        s.clear();
        write_str_repr(&mut s, "tab\tnl\n\u{1}é\u{a0}");
        assert_eq!(s, "'tab\\tnl\\n\\x01é\\xa0'");
    }

    #[test]
    fn json_dumps_uses_python_spacing_and_escapes() {
        let v = Value::from_serialize(serde_json::json!({
            "name": "wx", "n": 1, "f": 0.5, "ok": true, "none": null,
            "list": [1, "two", {"k": "v"}], "uni": "héllo 🌍 <tag> / \"q\"\n"
        }));
        assert_eq!(
            json_dumps(&v, &JsonOpts::default()).unwrap(),
            "{\"name\": \"wx\", \"n\": 1, \"f\": 0.5, \"ok\": true, \"none\": null, \
             \"list\": [1, \"two\", {\"k\": \"v\"}], \"uni\": \"héllo 🌍 <tag> / \\\"q\\\"\\n\"}"
        );
        let ascii = JsonOpts::resolve(true, None, None, false);
        assert_eq!(
            json_dumps(&Value::from("é🌍"), &ascii).unwrap(),
            "\"\\u00e9\\ud83c\\udf0d\""
        );
        let indented = JsonOpts::resolve(false, Some("  ".to_string()), None, false);
        assert_eq!(
            json_dumps(
                &Value::from_serialize(serde_json::json!({"a": [1, 2], "b": {}})),
                &indented
            )
            .unwrap(),
            "{\n  \"a\": [\n    1,\n    2\n  ],\n  \"b\": {}\n}"
        );
        let sorted = JsonOpts::resolve(false, None, None, true);
        assert_eq!(
            json_dumps(
                &Value::from_serialize(serde_json::json!({"b": 1, "a": 2})),
                &sorted
            )
            .unwrap(),
            "{\"a\": 2, \"b\": 1}"
        );
    }

    #[test]
    fn py_str_prints_python_literals() {
        assert_eq!(py_str(&Value::from(true)), "True");
        assert_eq!(py_str(&Value::from(())), "None");
        assert_eq!(py_str(&Value::UNDEFINED), "");
        assert_eq!(py_str(&Value::from(2.0)), "2.0");
        assert_eq!(py_str(&Value::from("x")), "x");
        let v = Value::from_serialize(serde_json::json!({"a": ["x", 1, null, false]}));
        assert_eq!(py_str(&v), "{'a': ['x', 1, None, False]}");
    }
}
