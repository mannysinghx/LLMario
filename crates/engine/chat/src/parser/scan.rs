//! Low-level scanning helpers shared by the family parsers.
//!
//! * [`scan`]: find the earliest marker in a buffer, or report how much of the buffer can be
//!   released because it cannot be the start of a marker (llama.cpp's `NEED_MORE_INPUT`).
//! * [`JsonObjectScanner`] / [`JsonValueScanner`]: incremental, allocation-light scanners over
//!   JSON text that know where each top-level member starts and ends while the text is still
//!   arriving, so raw argument text can be streamed before the object is complete.
//! * [`parse_py_value`] / [`parse_gemma_value`]: resumable literal parsers for the pythonic
//!   (`f(a='b', n=1)`) and Gemma 4 (`{k:<|"|>v<|"|>}`) argument syntaxes. They return
//!   [`PErr::Incomplete`] when the input ends mid-value, which is how the call parsers wait for
//!   more text without ever guessing.

use serde_json::{Map, Number, Value};

/// A parse attempt over a buffer that is still growing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PErr {
    /// The input ended before the unit was complete: feed more text and retry.
    Incomplete,
    /// The input cannot be this unit.
    Syntax(String),
}

/// Result of [`scan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scan {
    /// `markers[which]` occupies `buf[start..end]`; `buf[..start]` is plain text.
    Found {
        start: usize,
        which: usize,
        end: usize,
    },
    /// No complete marker. `buf[..release]` cannot be part of a marker; `buf[release..]` is
    /// the longest suffix that is a prefix of some marker and must be held back.
    NotFound { release: usize },
}

/// Scan `buf` for the earliest complete marker. Ties go to the longest marker. A marker that
/// is found is reported only if no *longer* partial match starts before it and runs off the end
/// of the buffer (then the buffer is held back until that partial match resolves).
pub fn scan(buf: &str, markers: &[&str]) -> Scan {
    let mut best: Option<(usize, usize, usize)> = None;
    for (i, m) in markers.iter().enumerate() {
        if m.is_empty() {
            continue;
        }
        if let Some(pos) = buf.find(m) {
            let end = pos + m.len();
            let better = match best {
                None => true,
                Some((bs, _, be)) => pos < bs || (pos == bs && end > be),
            };
            if better {
                best = Some((pos, i, end));
            }
        }
    }
    let limit = best.map(|(s, _, _)| s).unwrap_or(buf.len());
    // Longest suffix of `buf` that is a proper prefix of some marker and starts before `limit`.
    let mut held_from = buf.len();
    for m in markers {
        let max = m.len().saturating_sub(1).min(buf.len());
        for k in (1..=max).rev() {
            let s = buf.len() - k;
            if s >= held_from || s >= limit {
                continue;
            }
            if buf.is_char_boundary(s) && m.as_bytes().starts_with(&buf.as_bytes()[s..]) {
                held_from = s;
                break;
            }
        }
    }
    match best {
        Some((start, which, end)) if held_from >= start || held_from == buf.len() => {
            Scan::Found { start, which, end }
        }
        _ => Scan::NotFound { release: held_from },
    }
}

/// Number of leading ASCII whitespace bytes.
pub fn ws_len(s: &str) -> usize {
    s.len() - s.trim_start().len()
}

/// `tag` at the start of `rest`: `Ok(tag.len())`, `Incomplete` when `rest` is a proper prefix
/// of `tag`, `Syntax` otherwise.
pub fn expect_tag(rest: &str, tag: &str) -> Result<usize, PErr> {
    if rest.starts_with(tag) {
        Ok(tag.len())
    } else if tag.starts_with(rest) {
        Err(PErr::Incomplete)
    } else {
        Err(PErr::Syntax(format!("expected {tag:?}")))
    }
}

/// Which of `tags` starts `rest`: `Ok(index)`; `Incomplete` when `rest` is a proper prefix of
/// one of them; `Syntax` otherwise.
pub fn expect_one_of(rest: &str, tags: &[&str]) -> Result<usize, PErr> {
    let mut incomplete = false;
    for (i, tag) in tags.iter().enumerate() {
        match expect_tag(rest, tag) {
            Ok(_) => return Ok(i),
            Err(PErr::Incomplete) => incomplete = true,
            Err(PErr::Syntax(_)) => {}
        }
    }
    if incomplete {
        Err(PErr::Incomplete)
    } else {
        Err(PErr::Syntax(format!("expected one of {tags:?}")))
    }
}

/// Position of `tag` in `rest`; `Incomplete` when absent (the tail may still begin it).
pub fn find_tag(rest: &str, tag: &str) -> Result<usize, PErr> {
    rest.find(tag).ok_or(PErr::Incomplete)
}

/// One top-level member of a JSON object seen by [`JsonObjectScanner`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Member {
    pub key: String,
    /// Byte offset of the value's first character in [`JsonObjectScanner::text`].
    pub value_start: usize,
    /// Byte offset just past the value once it is complete.
    pub value_end: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OState {
    Start,
    KeyOrEnd,
    InKey,
    Colon,
    Value,
    InScalar,
    InString,
    InComposite,
    CommaOrEnd,
    Done,
}

/// Incremental scanner over one JSON object. Feed text as it arrives; `members` lists the
/// top-level keys with the byte range of each value (the last one open while it streams),
/// `done` is set once the closing brace was seen, `error` when the text is not an object.
#[derive(Debug, Clone)]
pub struct JsonObjectScanner {
    pub text: String,
    state: OState,
    depth: usize,
    in_string: bool,
    escape: bool,
    key_raw: String,
    pending_key: Option<String>,
    pub members: Vec<Member>,
    pub done: Option<usize>,
    pub error: Option<String>,
}

impl Default for JsonObjectScanner {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonObjectScanner {
    pub fn new() -> Self {
        JsonObjectScanner {
            text: String::new(),
            state: OState::Start,
            depth: 0,
            in_string: false,
            escape: false,
            key_raw: String::new(),
            pending_key: None,
            members: Vec::new(),
            done: None,
            error: None,
        }
    }

    pub fn feed(&mut self, s: &str) {
        if self.done.is_some() || self.error.is_some() {
            return;
        }
        let base = self.text.len();
        self.text.push_str(s);
        for (i, ch) in s.char_indices() {
            let off = base + i;
            // A scalar ends at the first non-scalar character, which then needs reprocessing.
            if self.step(off, ch) {
                self.step(off, ch);
            }
            if self.done.is_some() || self.error.is_some() {
                return;
            }
        }
    }

    /// The text of member `i` once complete.
    pub fn value_text(&self, i: usize) -> Option<&str> {
        let m = self.members.get(i)?;
        Some(&self.text[m.value_start..m.value_end?])
    }

    fn fail(&mut self, off: usize, ch: char, what: &str) {
        self.error = Some(format!("{what} at byte {off} (got {ch:?})"));
    }

    /// Returns true when `ch` must be processed again in the new state.
    fn step(&mut self, off: usize, ch: char) -> bool {
        match self.state {
            OState::Start => {
                if ch.is_ascii_whitespace() {
                } else if ch == '{' {
                    self.state = OState::KeyOrEnd;
                } else {
                    self.fail(off, ch, "expected '{'");
                }
            }
            OState::KeyOrEnd => {
                if ch.is_ascii_whitespace() {
                } else if ch == '"' {
                    self.key_raw.clear();
                    self.state = OState::InKey;
                } else if ch == '}' {
                    self.done = Some(off + 1);
                    self.state = OState::Done;
                } else {
                    self.fail(off, ch, "expected a key");
                }
            }
            OState::InKey => {
                if self.escape {
                    self.key_raw.push('\\');
                    self.key_raw.push(ch);
                    self.escape = false;
                } else if ch == '\\' {
                    self.escape = true;
                } else if ch == '"' {
                    let quoted = format!("\"{}\"", self.key_raw);
                    let key = serde_json::from_str::<String>(&quoted)
                        .unwrap_or_else(|_| self.key_raw.clone());
                    self.pending_key = Some(key);
                    self.state = OState::Colon;
                } else {
                    self.key_raw.push(ch);
                }
            }
            OState::Colon => {
                if ch.is_ascii_whitespace() {
                } else if ch == ':' {
                    self.state = OState::Value;
                } else {
                    self.fail(off, ch, "expected ':'");
                }
            }
            OState::Value => {
                if ch.is_ascii_whitespace() {
                    return false;
                }
                let key = self.pending_key.take().unwrap_or_default();
                self.members.push(Member {
                    key,
                    value_start: off,
                    value_end: None,
                });
                match ch {
                    '{' | '[' => {
                        self.depth = 1;
                        self.in_string = false;
                        self.escape = false;
                        self.state = OState::InComposite;
                    }
                    '"' => {
                        self.escape = false;
                        self.state = OState::InString;
                    }
                    '-' | '0'..='9' | 't' | 'f' | 'n' => self.state = OState::InScalar,
                    _ => self.fail(off, ch, "expected a value"),
                }
            }
            OState::InString => {
                if self.escape {
                    self.escape = false;
                } else if ch == '\\' {
                    self.escape = true;
                } else if ch == '"' {
                    self.close_value(off + 1);
                }
            }
            OState::InComposite => {
                if self.in_string {
                    if self.escape {
                        self.escape = false;
                    } else if ch == '\\' {
                        self.escape = true;
                    } else if ch == '"' {
                        self.in_string = false;
                    }
                } else {
                    match ch {
                        '"' => self.in_string = true,
                        '{' | '[' => self.depth += 1,
                        '}' | ']' => {
                            self.depth -= 1;
                            if self.depth == 0 {
                                self.close_value(off + 1);
                            }
                        }
                        _ => {}
                    }
                }
            }
            OState::InScalar => {
                if ch.is_ascii_alphanumeric() || matches!(ch, '+' | '-' | '.') {
                } else {
                    self.close_value(off);
                    return true;
                }
            }
            OState::CommaOrEnd => {
                if ch.is_ascii_whitespace() {
                } else if ch == ',' {
                    self.state = OState::KeyOrEnd;
                } else if ch == '}' {
                    self.done = Some(off + 1);
                    self.state = OState::Done;
                } else {
                    self.fail(off, ch, "expected ',' or '}'");
                }
            }
            OState::Done => {}
        }
        false
    }

    fn close_value(&mut self, end: usize) {
        if let Some(m) = self.members.last_mut() {
            m.value_end = Some(end);
        }
        self.state = OState::CommaOrEnd;
    }
}

/// Incremental scanner over one JSON object or array (the shapes tool arguments take): reports
/// the byte offset just past the value once its brackets balance.
#[derive(Debug, Clone, Default)]
pub struct JsonValueScanner {
    pub text: String,
    started: bool,
    depth: usize,
    in_string: bool,
    escape: bool,
    pub done: Option<usize>,
    pub error: Option<String>,
}

impl JsonValueScanner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn feed(&mut self, s: &str) {
        if self.done.is_some() || self.error.is_some() {
            return;
        }
        let base = self.text.len();
        self.text.push_str(s);
        for (i, ch) in s.char_indices() {
            let off = base + i;
            if !self.started {
                if ch.is_ascii_whitespace() {
                    continue;
                }
                if ch == '{' || ch == '[' {
                    self.started = true;
                    self.depth = 1;
                    continue;
                }
                self.error = Some(format!("expected '{{' or '[' at byte {off} (got {ch:?})"));
                return;
            }
            if self.in_string {
                if self.escape {
                    self.escape = false;
                } else if ch == '\\' {
                    self.escape = true;
                } else if ch == '"' {
                    self.in_string = false;
                }
                continue;
            }
            match ch {
                '"' => self.in_string = true,
                '{' | '[' => self.depth += 1,
                '}' | ']' => {
                    self.depth -= 1;
                    if self.depth == 0 {
                        self.done = Some(off + 1);
                        return;
                    }
                }
                _ => {}
            }
        }
    }
}

/// A cursor over a borrowed string for the resumable literal parsers.
#[derive(Debug, Clone, Copy)]
pub struct Cursor<'a> {
    s: &'a str,
    pub pos: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(s: &'a str) -> Self {
        Cursor { s, pos: 0 }
    }

    pub fn rest(&self) -> &'a str {
        &self.s[self.pos..]
    }

    pub fn peek(&self) -> Option<char> {
        self.rest().chars().next()
    }

    pub fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    pub fn skip_ws(&mut self) {
        self.pos += ws_len(self.rest());
    }

    pub fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.pos += c.len_utf8();
            true
        } else {
            false
        }
    }

    pub fn at_end(&self) -> bool {
        self.pos >= self.s.len()
    }

    /// Consume an identifier (`[A-Za-z_][A-Za-z0-9_.-]*`). `Incomplete` at end of input: a
    /// longer identifier may still arrive.
    pub fn identifier(&mut self) -> Result<String, PErr> {
        let rest = self.rest();
        let mut len = 0;
        for (i, c) in rest.char_indices() {
            let ok = if i == 0 {
                c.is_ascii_alphabetic() || c == '_'
            } else {
                c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-')
            };
            if !ok {
                break;
            }
            len = i + c.len_utf8();
        }
        if len == rest.len() {
            return Err(PErr::Incomplete);
        }
        if len == 0 {
            return Err(PErr::Syntax(format!(
                "expected an identifier at byte {}",
                self.pos
            )));
        }
        self.pos += len;
        Ok(rest[..len].to_string())
    }
}

/// Parse a Python or JSON literal: quoted strings (both quote styles, backslash escapes),
/// numbers, `True`/`False`/`None`/`true`/`false`/`null`, lists, tuples and dicts. A bare
/// identifier is accepted as a string (models drop quotes). `Incomplete` when the input ends
/// mid-literal, including after a bare number or word (a longer token may follow).
pub fn parse_py_value(c: &mut Cursor<'_>) -> Result<Value, PErr> {
    c.skip_ws();
    let Some(ch) = c.peek() else {
        return Err(PErr::Incomplete);
    };
    match ch {
        '\'' | '"' => parse_py_string(c).map(Value::String),
        '[' | '(' => {
            let close = if ch == '[' { ']' } else { ')' };
            c.bump();
            let mut items = Vec::new();
            loop {
                c.skip_ws();
                match c.peek() {
                    None => return Err(PErr::Incomplete),
                    Some(x) if x == close => {
                        c.bump();
                        return Ok(Value::Array(items));
                    }
                    Some(_) => {}
                }
                items.push(parse_py_value(c)?);
                c.skip_ws();
                match c.peek() {
                    None => return Err(PErr::Incomplete),
                    Some(',') => {
                        c.bump();
                    }
                    Some(x) if x == close => {}
                    Some(x) => {
                        return Err(PErr::Syntax(format!(
                            "expected ',' or '{close}' at byte {} (got {x:?})",
                            c.pos
                        )))
                    }
                }
            }
        }
        '{' => {
            c.bump();
            let mut map = Map::new();
            loop {
                c.skip_ws();
                match c.peek() {
                    None => return Err(PErr::Incomplete),
                    Some('}') => {
                        c.bump();
                        return Ok(Value::Object(map));
                    }
                    Some(_) => {}
                }
                let key = match c.peek() {
                    Some('\'') | Some('"') => parse_py_string(c)?,
                    _ => c.identifier()?,
                };
                c.skip_ws();
                if !c.eat(':') {
                    return Err(if c.at_end() {
                        PErr::Incomplete
                    } else {
                        PErr::Syntax(format!("expected ':' at byte {}", c.pos))
                    });
                }
                let v = parse_py_value(c)?;
                map.insert(key, v);
                c.skip_ws();
                match c.peek() {
                    None => return Err(PErr::Incomplete),
                    Some(',') => {
                        c.bump();
                    }
                    Some('}') => {}
                    Some(x) => {
                        return Err(PErr::Syntax(format!(
                            "expected ',' or '}}' at byte {} (got {x:?})",
                            c.pos
                        )))
                    }
                }
            }
        }
        '-' | '+' | '0'..='9' | '.' => parse_number(c),
        _ => {
            let word = c.identifier()?;
            Ok(match word.as_str() {
                "True" | "true" => Value::Bool(true),
                "False" | "false" => Value::Bool(false),
                "None" | "null" => Value::Null,
                _ => Value::String(word),
            })
        }
    }
}

/// A quoted string with Python escapes (`\\`, `\'`, `\"`, `\n`, `\r`, `\t`, `\0`, `\xHH`,
/// `\uHHHH`); unknown escapes keep the backslash like Python does.
pub fn parse_py_string(c: &mut Cursor<'_>) -> Result<String, PErr> {
    let quote = c.bump().ok_or(PErr::Incomplete)?;
    let mut out = String::new();
    loop {
        let Some(ch) = c.bump() else {
            return Err(PErr::Incomplete);
        };
        if ch == quote {
            return Ok(out);
        }
        if ch != '\\' {
            out.push(ch);
            continue;
        }
        let Some(e) = c.bump() else {
            return Err(PErr::Incomplete);
        };
        match e {
            '\\' => out.push('\\'),
            '\'' => out.push('\''),
            '"' => out.push('"'),
            'n' => out.push('\n'),
            'r' => out.push('\r'),
            't' => out.push('\t'),
            '0' => out.push('\0'),
            'b' => out.push('\u{8}'),
            'f' => out.push('\u{c}'),
            '/' => out.push('/'),
            'x' | 'u' => {
                let n = if e == 'x' { 2 } else { 4 };
                let rest = c.rest();
                if rest.len() < n {
                    return Err(PErr::Incomplete);
                }
                match u32::from_str_radix(&rest[..n], 16)
                    .ok()
                    .and_then(char::from_u32)
                {
                    Some(decoded) => {
                        out.push(decoded);
                        c.pos += n;
                    }
                    None => {
                        out.push('\\');
                        out.push(e);
                    }
                }
            }
            other => {
                out.push('\\');
                out.push(other);
            }
        }
    }
}

fn parse_number(c: &mut Cursor<'_>) -> Result<Value, PErr> {
    let rest = c.rest();
    let mut len = 0;
    for (i, ch) in rest.char_indices() {
        if ch.is_ascii_digit() || matches!(ch, '+' | '-' | '.' | 'e' | 'E') {
            len = i + ch.len_utf8();
        } else {
            break;
        }
    }
    if len == rest.len() {
        return Err(PErr::Incomplete);
    }
    let text = &rest[..len];
    let value = number_from_text(text)
        .ok_or_else(|| PErr::Syntax(format!("bad number {text:?} at byte {}", c.pos)))?;
    c.pos += len;
    Ok(value)
}

/// `"12"` → 12, `"1.5"` / `"1e-05"` / `"1e+21"` → floats; anything else → `None`.
pub fn number_from_text(text: &str) -> Option<Value> {
    let t = text.strip_prefix('+').unwrap_or(text);
    if let Ok(i) = t.parse::<i64>() {
        return Some(Value::Number(i.into()));
    }
    if let Ok(u) = t.parse::<u64>() {
        return Some(Value::Number(u.into()));
    }
    let f = t.parse::<f64>().ok()?;
    Number::from_f64(f).map(Value::Number)
}

/// Gemma 4's string delimiter token.
pub const GEMMA_QUOTE: &str = "<|\"|>";

/// Parse a Gemma 4 argument literal: `<|"|>text<|"|>` strings (no escaping; the text runs to
/// the next delimiter), `null`, `true`, `false`, numbers, `{key:value,...}` with bare or
/// delimited keys, and `[...]` lists.
pub fn parse_gemma_value(c: &mut Cursor<'_>) -> Result<Value, PErr> {
    c.skip_ws();
    let rest = c.rest();
    if rest.is_empty() {
        return Err(PErr::Incomplete);
    }
    if rest.starts_with(GEMMA_QUOTE) {
        return parse_gemma_string(c).map(Value::String);
    }
    if GEMMA_QUOTE.starts_with(rest) {
        return Err(PErr::Incomplete);
    }
    let ch = c.peek().unwrap_or(' ');
    match ch {
        '{' => {
            c.bump();
            let mut map = Map::new();
            loop {
                c.skip_ws();
                match c.peek() {
                    None => return Err(PErr::Incomplete),
                    Some('}') => {
                        c.bump();
                        return Ok(Value::Object(map));
                    }
                    Some(_) => {}
                }
                let key = parse_gemma_key(c)?;
                c.skip_ws();
                if !c.eat(':') {
                    return Err(if c.at_end() {
                        PErr::Incomplete
                    } else {
                        PErr::Syntax(format!("expected ':' at byte {}", c.pos))
                    });
                }
                let v = parse_gemma_value(c)?;
                map.insert(key, v);
                c.skip_ws();
                match c.peek() {
                    None => return Err(PErr::Incomplete),
                    Some(',') => {
                        c.bump();
                    }
                    Some('}') => {}
                    Some(x) => {
                        return Err(PErr::Syntax(format!(
                            "expected ',' or '}}' at byte {} (got {x:?})",
                            c.pos
                        )))
                    }
                }
            }
        }
        '[' => {
            c.bump();
            let mut items = Vec::new();
            loop {
                c.skip_ws();
                match c.peek() {
                    None => return Err(PErr::Incomplete),
                    Some(']') => {
                        c.bump();
                        return Ok(Value::Array(items));
                    }
                    Some(_) => {}
                }
                items.push(parse_gemma_value(c)?);
                c.skip_ws();
                match c.peek() {
                    None => return Err(PErr::Incomplete),
                    Some(',') => {
                        c.bump();
                    }
                    Some(']') => {}
                    Some(x) => {
                        return Err(PErr::Syntax(format!(
                            "expected ',' or ']' at byte {} (got {x:?})",
                            c.pos
                        )))
                    }
                }
            }
        }
        '-' | '+' | '0'..='9' | '.' => parse_number(c),
        _ => {
            let word = c.identifier()?;
            Ok(match word.as_str() {
                "true" | "True" => Value::Bool(true),
                "false" | "False" => Value::Bool(false),
                "null" | "None" => Value::Null,
                _ => Value::String(word),
            })
        }
    }
}

/// A Gemma key: bare identifier or a delimited string.
pub fn parse_gemma_key(c: &mut Cursor<'_>) -> Result<String, PErr> {
    let rest = c.rest();
    if rest.starts_with(GEMMA_QUOTE) {
        return parse_gemma_string(c);
    }
    if GEMMA_QUOTE.starts_with(rest) {
        return Err(PErr::Incomplete);
    }
    c.identifier()
}

fn parse_gemma_string(c: &mut Cursor<'_>) -> Result<String, PErr> {
    let rest = c.rest();
    let body = &rest[GEMMA_QUOTE.len()..];
    let end = body.find(GEMMA_QUOTE).ok_or(PErr::Incomplete)?;
    c.pos += GEMMA_QUOTE.len() + end + GEMMA_QUOTE.len();
    Ok(body[..end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn scan_finds_earliest_and_holds_partials() {
        let m = &["<think>", "<tool_call>", "</think>"];
        assert_eq!(
            scan("abc<tool_call>x", m),
            Scan::Found {
                start: 3,
                which: 1,
                end: 14
            }
        );
        assert_eq!(scan("abc<too", m), Scan::NotFound { release: 3 });
        assert_eq!(scan("abc<", m), Scan::NotFound { release: 3 });
        assert_eq!(scan("abc", m), Scan::NotFound { release: 3 });
        assert_eq!(
            scan("<tool_c<think>", m),
            Scan::Found {
                start: 7,
                which: 0,
                end: 14
            }
        );
        // A longer partial starting before a shorter full match holds the buffer.
        assert_eq!(scan("ab", &["b", "abc"]), Scan::NotFound { release: 0 });
        assert_eq!(
            scan("abd", &["b", "abc"]),
            Scan::Found {
                start: 1,
                which: 0,
                end: 2
            }
        );
        // Non-ASCII text never confuses the boundary logic.
        assert_eq!(scan("héllo<", &["<x>"]), Scan::NotFound { release: 6 });
    }

    #[test]
    fn json_object_scanner_tracks_members_incrementally() {
        let mut s = JsonObjectScanner::new();
        s.feed("{\"name\": \"f\", \"arguments\": {\"a\": \"x}\", \"b\": [1, {");
        assert_eq!(s.members.len(), 2);
        assert_eq!(s.value_text(0), Some("\"f\""));
        assert!(s.members[1].value_end.is_none());
        assert!(s.done.is_none());
        s.feed("\"c\": null}]}, \"n\": 1}");
        assert_eq!(s.done, Some(s.text.len()));
        assert_eq!(
            s.value_text(1),
            Some("{\"a\": \"x}\", \"b\": [1, {\"c\": null}]}")
        );
        assert_eq!(s.value_text(2), Some("1"));
        assert!(s.error.is_none());

        // Scalar spelling is not validated here (serde does that on the complete value).
        let mut lenient = JsonObjectScanner::new();
        lenient.feed("{\"name\": f}");
        assert_eq!(lenient.value_text(0), Some("f"));
        let mut bad = JsonObjectScanner::new();
        bad.feed("{\"name\": @}");
        assert!(bad.error.is_some());
        let mut not_obj = JsonObjectScanner::new();
        not_obj.feed("[1]");
        assert!(not_obj.error.is_some());
    }

    #[test]
    fn json_value_scanner_balances_brackets() {
        let mut s = JsonValueScanner::new();
        s.feed(" {\"a\": \"]}\", \"b\": [1, 2");
        assert!(s.done.is_none());
        s.feed("]} trailing");
        assert_eq!(s.done, Some(25));
        let mut bad = JsonValueScanner::new();
        bad.feed("x");
        assert!(bad.error.is_some());
    }

    #[test]
    fn python_literals() {
        let parse = |t: &str| {
            let mut c = Cursor::new(t);
            parse_py_value(&mut c).map(|v| (v, c.pos))
        };
        assert_eq!(parse("'a\\'b\\n' ").unwrap().0, json!("a'b\n"));
        assert_eq!(parse("\"q\\\"x\" ").unwrap().0, json!("q\"x"));
        assert_eq!(parse("12,").unwrap().0, json!(12));
        assert_eq!(parse("-1.5e3)").unwrap().0, json!(-1500.0));
        assert_eq!(parse("True)").unwrap().0, json!(true));
        assert_eq!(parse("None)").unwrap().0, json!(null));
        assert_eq!(parse("null)").unwrap().0, json!(null));
        assert_eq!(parse("celsius)").unwrap().0, json!("celsius"));
        assert_eq!(
            parse("[1, 'a', (2, 3), {'k': [True], \"j\": {\"n\": null}}] ")
                .unwrap()
                .0,
            json!([1, "a", [2, 3], {"k": [true], "j": {"n": null}}])
        );
        assert_eq!(parse("12"), Err(PErr::Incomplete));
        assert_eq!(parse("True"), Err(PErr::Incomplete));
        assert_eq!(parse("'abc"), Err(PErr::Incomplete));
        assert_eq!(parse("[1, 2"), Err(PErr::Incomplete));
        assert!(matches!(parse("[1 2]"), Err(PErr::Syntax(_))));
        assert_eq!(parse("'\\x41\\u00e9'!").unwrap().0, json!("Aé"));
    }

    #[test]
    fn gemma_literals() {
        let parse = |t: &str| {
            let mut c = Cursor::new(t);
            parse_gemma_value(&mut c).map(|v| (v, c.pos))
        };
        assert_eq!(
            parse("<|\"|>Paris, France<|\"|>,").unwrap().0,
            json!("Paris, France")
        );
        assert_eq!(parse("<|\"|>a,}{<|\"|>}").unwrap().0, json!("a,}{"));
        assert_eq!(parse("null,").unwrap().0, json!(null));
        assert_eq!(parse("true}").unwrap().0, json!(true));
        assert_eq!(parse("899.5}").unwrap().0, json!(899.5));
        assert_eq!(
            parse("{from:<|\"|>LHR<|\"|>,via:[],n:1}}").unwrap().0,
            json!({"from": "LHR", "via": [], "n": 1})
        );
        assert_eq!(
            parse("{<|\"|>k<|\"|>:[{a:1}]}}").unwrap().0,
            json!({"k": [{"a": 1}]})
        );
        assert_eq!(parse("<|\"|>unterminated"), Err(PErr::Incomplete));
        assert_eq!(parse("<|\""), Err(PErr::Incomplete));
        assert_eq!(parse("{a:1"), Err(PErr::Incomplete));
        assert!(matches!(parse("{a 1}"), Err(PErr::Syntax(_))));
    }

    #[test]
    fn tag_helpers() {
        assert_eq!(expect_tag("<function=", "<function="), Ok(10));
        assert_eq!(expect_tag("<fun", "<function="), Err(PErr::Incomplete));
        assert!(matches!(
            expect_tag("<x", "<function="),
            Err(PErr::Syntax(_))
        ));
        assert_eq!(
            expect_one_of("</tool_call>", &["<parameter=", "</tool_call>"]),
            Ok(1)
        );
        assert_eq!(
            expect_one_of("</to", &["<parameter=", "</tool_call>"]),
            Err(PErr::Incomplete)
        );
        assert_eq!(find_tag("ab</p>", "</p>"), Ok(2));
        assert_eq!(find_tag("ab</", "</p>"), Err(PErr::Incomplete));
    }
}
