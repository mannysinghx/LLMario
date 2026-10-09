//! gpt-oss Harmony: `<|channel|>CHANNEL [to=RECIPIENT] [<|constrain|>TYPE]<|message|>BODY`
//! terminated by `<|end|>`, `<|call|>` (tool call) or `<|return|>` (final answer), with an
//! optional `<|start|>assistant` prefix before each header.
//!
//! * `analysis` → [`super::OutputEvent::Reasoning`] (never shown to end users).
//! * `final` → content. `commentary` without a recipient → content (a preamble).
//! * `commentary` (or `analysis`) with `to=functions.NAME` → a tool call named `NAME`; the body
//!   is the JSON arguments and is streamed as raw deltas. Other namespaces (`browser.search`,
//!   `python`) keep the full recipient as the name.
//!
//! Both header orders are accepted: the model's generation order
//! `<|channel|>commentary to=functions.f <|constrain|>json<|message|>` and the order the
//! chat template renders into history, `<|start|>assistant to=functions.f<|channel|>commentary
//! json<|message|>`. Sources: `tokenizer.chat_template` of gpt-oss-20b-MXFP4.gguf, and the
//! OpenAI Harmony guide (developers.openai.com/cookbook/articles/openai-harmony).
//!
//! If no `<|message|>` arrives within the first 256 bytes the output is treated as plain
//! content (the server stripped the control tokens, or the model skipped the header).

use super::scan::{scan, JsonValueScanner, Scan};
use super::Emitter;

const MESSAGE: &str = "<|message|>";
const BODY_MARKERS: &[&str] = &["<|end|>", "<|call|>", "<|return|>", "<|start|>"];
const HEADER_LIMIT: usize = 256;

enum Body {
    Reasoning,
    Content,
    Call {
        index: usize,
        scanner: JsonValueScanner,
    },
}

enum State {
    Header,
    Body(Body),
}

pub(crate) struct Harmony {
    buf: String,
    state: State,
}

impl Harmony {
    pub(crate) fn new() -> Harmony {
        Harmony {
            buf: String::new(),
            state: State::Header,
        }
    }

    pub(crate) fn push(&mut self, text: &str, em: &mut Emitter) {
        self.buf.push_str(text);
        self.run(em);
    }

    fn run(&mut self, em: &mut Emitter) {
        loop {
            match &mut self.state {
                State::Header => match scan(&self.buf, &[MESSAGE]) {
                    Scan::Found { start, end, .. } => {
                        let header = self.buf[..start].to_string();
                        self.buf.drain(..end);
                        self.state = State::Body(parse_header(&header, em));
                    }
                    Scan::NotFound { .. } => {
                        if self.buf.len() > HEADER_LIMIT {
                            self.state = State::Body(Body::Content);
                            continue;
                        }
                        return;
                    }
                },
                State::Body(body) => match scan(&self.buf, BODY_MARKERS) {
                    Scan::Found { start, end, .. } => {
                        let text = self.buf[..start].to_string();
                        deliver(body, &text, em);
                        self.buf.drain(..end);
                        let finished = std::mem::replace(&mut self.state, State::Header);
                        if let State::Body(b) = finished {
                            close_body(b, em);
                        }
                    }
                    Scan::NotFound { release } => {
                        if release > 0 {
                            let text: String = self.buf.drain(..release).collect();
                            deliver(body, &text, em);
                        }
                        return;
                    }
                },
            }
        }
    }

    pub(crate) fn finish(&mut self, em: &mut Emitter) {
        self.run(em);
        let rest = std::mem::take(&mut self.buf);
        match std::mem::replace(&mut self.state, State::Header) {
            State::Header => em.content(&rest),
            State::Body(mut body) => {
                deliver(&mut body, &rest, em);
                close_body(body, em);
            }
        }
    }
}

/// Split a header into channel, recipient and constraint.
fn parse_header(header: &str, em: &mut Emitter) -> Body {
    let mut channel = None;
    let mut recipient = None;
    for (i, _) in header.match_indices("<|channel|>") {
        channel = Some(token_after(&header[i + "<|channel|>".len()..]));
    }
    if let Some(i) = header.find("to=") {
        recipient = Some(token_after(&header[i + 3..]));
    }
    let channel = channel.unwrap_or_else(|| "final".to_string());
    match (channel.as_str(), recipient) {
        (_, Some(r)) if !r.is_empty() => {
            let name = r.strip_prefix("functions.").unwrap_or(&r).to_string();
            let index = em.start(&name, None);
            Body::Call {
                index,
                scanner: JsonValueScanner::new(),
            }
        }
        ("analysis", _) => Body::Reasoning,
        _ => Body::Content,
    }
}

/// The text up to the next whitespace or control token.
fn token_after(s: &str) -> String {
    let end = s
        .find(|c: char| c.is_whitespace() || c == '<')
        .unwrap_or(s.len());
    s[..end].to_string()
}

fn deliver(body: &mut Body, text: &str, em: &mut Emitter) {
    match body {
        Body::Reasoning => em.reasoning(text),
        Body::Content => em.content(text),
        Body::Call { index, scanner } => {
            if scanner.done.is_some() {
                return;
            }
            let before = scanner.text.len();
            scanner.feed(text);
            let end = scanner.done.unwrap_or(scanner.text.len());
            em.delta(*index, &scanner.text[before..end]);
        }
    }
}

fn close_body(body: Body, em: &mut Emitter) {
    if let Body::Call { index, scanner } = body {
        let parsed = scanner
            .done
            .and_then(|end| serde_json::from_str::<serde_json::Value>(&scanner.text[..end]).ok());
        match parsed {
            Some(v) => em.end(index, v),
            None => {
                em.cancel();
                let reason = match &scanner.error {
                    Some(e) => format!("harmony tool call arguments are not JSON: {e}"),
                    None => {
                        "harmony tool call truncated before its arguments completed".to_string()
                    }
                };
                em.invalid(reason);
                em.content(&scanner.text);
            }
        }
    }
}
