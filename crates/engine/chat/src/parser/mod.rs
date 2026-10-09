//! Streaming parser for model output: reasoning text, visible content and tool calls, per
//! template family (ARCHITECTURE.md 10.2, Appendix D).
//!
//! [`OutputParser::push`] takes the detokenised text as it is generated and returns
//! [`OutputEvent`]s; [`OutputParser::finish`] flushes at end of output. Text that could be the
//! start of a marker is held back until it either becomes one or cannot (llama.cpp's
//! `NEED_MORE_INPUT`), so partial tags never leak into the content stream and the stream never
//! stalls once the output ends.
//!
//! Argument deltas are JSON text: for JSON-native wire formats (Hermes, Harmony, Mistral,
//! Llama JSON) they are the model's own bytes, streamed as soon as the tool name is known; for
//! tagged and pythonic formats (Qwen XML, GLM, Gemma 4, LFM2, OLMo, Llama 4) each completed
//! parameter is converted and streamed as a JSON object fragment (`{`, `"k":v`, `,"k2":v2`,
//! `}`). Either way the concatenated deltas parse to the `arguments` object delivered with
//! [`OutputEvent::ToolCallEnd`].
//!
//! Per-family marker strings and their sources are tabulated in the crate README and in
//! [`families`].

mod bodies;
mod families;
mod harmony;
mod scan;
pub mod schema;

use crate::TemplateFamily;
use serde_json::{Map, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub use schema::{RepairVerdict, ToolCallRepair, ToolSpec, ToolTable};

/// One parsed unit of model output.
#[derive(Clone, Debug, PartialEq)]
pub enum OutputEvent {
    /// Text inside the family's thinking markers (`<think>…</think>`, Gemma 4's
    /// `<|channel>thought…<channel|>`, Harmony's `analysis` channel).
    Reasoning(String),
    /// Visible assistant text.
    Content(String),
    /// A tool call begins; its name is known. `index` counts calls within this output from 0.
    ToolCallStart {
        index: usize,
        id: String,
        name: String,
    },
    /// A piece of the call's arguments as JSON text.
    ToolCallArgumentsDelta { index: usize, delta: String },
    /// The call is complete; `arguments` is always a JSON object (map).
    ToolCallEnd { index: usize, arguments: Value },
    /// Something could not be parsed. Advisory: the raw text is still delivered as
    /// [`OutputEvent::Content`] right after it. An `Invalid` that follows a `ToolCallStart`
    /// with no `ToolCallEnd` in between cancels that call.
    Invalid { reason: String },
}

/// Parser knobs the server sets from what it rendered.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParserOptions {
    /// The prompt ended with the family's reasoning opener (Qwen3.5's `<think>\n`, GLM's
    /// `<think>`, Gemma 4's `<|channel>thought\n` after a tool response), so the output begins
    /// inside the reasoning block without repeating the opener.
    pub reasoning_open: bool,
}

/// Streaming output parser for one generation.
pub struct OutputParser {
    family: TemplateFamily,
    backend: Backend,
    em: Emitter,
}

enum Backend {
    Tagged(families::Tagged),
    Harmony(harmony::Harmony),
}

impl std::fmt::Debug for OutputParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OutputParser")
            .field("family", &self.family)
            .field("calls", &self.em.next_index)
            .finish()
    }
}

impl OutputParser {
    /// A parser for `family`. `tools` are the request's tool definitions (OpenAI shape); when
    /// given, unknown tool names are flagged with [`OutputEvent::Invalid`] and tagged-format
    /// argument values are typed from each tool's JSON Schema.
    pub fn new(family: TemplateFamily, tools: Option<&Value>) -> OutputParser {
        OutputParser::with_options(family, tools, ParserOptions::default())
    }

    pub fn with_options(
        family: TemplateFamily,
        tools: Option<&Value>,
        options: ParserOptions,
    ) -> OutputParser {
        let table = ToolTable::from_tools(tools);
        let em = Emitter::new(table, matches!(family, TemplateFamily::Mistral));
        let backend = match family {
            TemplateFamily::Harmony => Backend::Harmony(harmony::Harmony::new()),
            other => Backend::Tagged(families::Tagged::new(families::spec(other), options)),
        };
        OutputParser {
            family,
            backend,
            em,
        }
    }

    pub fn family(&self) -> TemplateFamily {
        self.family
    }

    /// Feed the next piece of output text.
    pub fn push(&mut self, text: &str) -> Vec<OutputEvent> {
        match &mut self.backend {
            Backend::Tagged(t) => t.push(text, &mut self.em),
            Backend::Harmony(h) => h.push(text, &mut self.em),
        }
        self.em.take()
    }

    /// The output ended (EOS, stop token, length limit or cancel): flush held text, close an
    /// open reasoning block, finish a call whose closer is missing, or report a truncated call.
    pub fn finish(&mut self) -> Vec<OutputEvent> {
        match &mut self.backend {
            Backend::Tagged(t) => t.finish(&mut self.em),
            Backend::Harmony(h) => h.finish(&mut self.em),
        }
        self.em.take()
    }

    /// The family's reasoning markers (`(opener, closer)`), if any. Harmony has none in this
    /// sense: its `analysis` channel is parsed structurally.
    pub fn reasoning_markers(family: TemplateFamily) -> Option<(&'static str, &'static str)> {
        let s = families::spec(family);
        (!s.think_open.is_empty()).then_some((s.think_open, s.think_close))
    }

    /// Whether a rendered prompt ends inside the reasoning block (the generation prompt
    /// emitted the opener, as Qwen3.5, GLM and Gemma 4 do), i.e. whether
    /// [`ParserOptions::reasoning_open`] should be set.
    pub fn prompt_opens_reasoning(family: TemplateFamily, prompt: &str) -> bool {
        let Some((open, close)) = OutputParser::reasoning_markers(family) else {
            return false;
        };
        let tail = prompt.trim_end_matches(['\n', ' ']);
        let open = open.trim_end_matches('\n');
        tail.ends_with(open) && !tail.ends_with(close)
    }
}

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Collects events and owns the call index / id bookkeeping.
pub(crate) struct Emitter {
    events: Vec<OutputEvent>,
    next_index: usize,
    seed: u64,
    mistral_ids: bool,
    tools: ToolTable,
    open: Option<usize>,
}

impl Emitter {
    fn new(tools: ToolTable, mistral_ids: bool) -> Emitter {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let n = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        // Mixed so that two parsers created in the same nanosecond still differ.
        let seed = (nanos ^ n.wrapping_mul(0x9E37_79B9_7F4A_7C15)) & 0xFFFF_FFFF;
        Emitter {
            events: Vec::new(),
            next_index: 0,
            seed,
            mistral_ids,
            tools,
            open: None,
        }
    }

    fn make_id(&self, index: usize) -> String {
        if self.mistral_ids {
            // Mistral tokenizers validate 9-character alphanumeric ids.
            let mut n = self.seed * 64 + (index as u64 % 64);
            let digits = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
            let mut out = Vec::new();
            while n > 0 {
                out.push(digits[(n % 62) as usize]);
                n /= 62;
            }
            while out.len() < 9 {
                out.push(b'0');
            }
            out.reverse();
            String::from_utf8(out).unwrap_or_default()
        } else {
            format!("call_{:08x}_{index}", self.seed)
        }
    }

    pub(crate) fn tools(&self) -> &ToolTable {
        &self.tools
    }

    /// Start call `name`; emits `Invalid` first when the name is not among the declared tools.
    pub(crate) fn start(&mut self, name: &str, explicit_id: Option<String>) -> usize {
        let index = self.next_index;
        self.next_index += 1;
        let name = name.trim();
        if !self.tools.is_empty() && self.tools.get(name).is_none() {
            self.invalid(format!(
                "tool call {index} names an undeclared tool {name:?}"
            ));
        }
        let id = explicit_id.unwrap_or_else(|| self.make_id(index));
        self.events.push(OutputEvent::ToolCallStart {
            index,
            id,
            name: name.to_string(),
        });
        self.open = Some(index);
        index
    }

    pub(crate) fn delta(&mut self, index: usize, delta: &str) {
        if !delta.is_empty() {
            self.events.push(OutputEvent::ToolCallArgumentsDelta {
                index,
                delta: delta.to_string(),
            });
        }
    }

    pub(crate) fn end(&mut self, index: usize, arguments: Value) {
        let arguments = match arguments {
            Value::Object(m) => Value::Object(m),
            Value::Null => Value::Object(Map::new()),
            other => {
                let mut m = Map::new();
                m.insert("value".to_string(), other);
                Value::Object(m)
            }
        };
        self.events
            .push(OutputEvent::ToolCallEnd { index, arguments });
        self.open = None;
    }

    /// Drop the bookkeeping for a call that started but cannot be completed.
    pub(crate) fn cancel(&mut self) {
        self.open = None;
    }

    pub(crate) fn content(&mut self, text: &str) {
        if !text.is_empty() {
            self.events.push(OutputEvent::Content(text.to_string()));
        }
    }

    pub(crate) fn reasoning(&mut self, text: &str) {
        if !text.is_empty() {
            self.events.push(OutputEvent::Reasoning(text.to_string()));
        }
    }

    pub(crate) fn invalid(&mut self, reason: String) {
        self.events.push(OutputEvent::Invalid { reason });
    }

    fn take(&mut self) -> Vec<OutputEvent> {
        std::mem::take(&mut self.events)
    }
}

/// Builds the argument object of a tagged/pythonic call one parameter at a time, streaming
/// the JSON fragments as it goes.
pub(crate) struct ArgAssembler {
    index: usize,
    first: bool,
    map: Map<String, Value>,
}

impl ArgAssembler {
    pub(crate) fn new(em: &mut Emitter, index: usize) -> ArgAssembler {
        em.delta(index, "{");
        ArgAssembler {
            index,
            first: true,
            map: Map::new(),
        }
    }

    pub(crate) fn arg(&mut self, em: &mut Emitter, key: &str, value: Value) {
        let key_json = serde_json::to_string(key).unwrap_or_else(|_| format!("{key:?}"));
        let value_json = serde_json::to_string(&value).unwrap_or_else(|_| "null".to_string());
        let frag = format!(
            "{}{key_json}:{value_json}",
            if self.first { "" } else { "," }
        );
        self.first = false;
        em.delta(self.index, &frag);
        self.map.insert(key.to_string(), value);
    }

    pub(crate) fn finish(self, em: &mut Emitter) {
        em.delta(self.index, "}");
        em.end(self.index, Value::Object(self.map));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique_per_parser_and_mistral_ids_are_nine_alnum() {
        let a = Emitter::new(ToolTable::default(), false);
        let b = Emitter::new(ToolTable::default(), false);
        assert_ne!(a.make_id(0), b.make_id(0));
        assert!(a.make_id(3).starts_with("call_"));
        let m = Emitter::new(ToolTable::default(), true);
        let id = m.make_id(5);
        assert_eq!(id.len(), 9);
        assert!(id.chars().all(|c| c.is_ascii_alphanumeric()));
        assert_ne!(m.make_id(5), m.make_id(6));
    }

    #[test]
    fn prompt_opens_reasoning_per_family() {
        assert!(OutputParser::prompt_opens_reasoning(
            TemplateFamily::QwenXml,
            "<|im_start|>assistant\n<think>\n"
        ));
        assert!(!OutputParser::prompt_opens_reasoning(
            TemplateFamily::QwenXml,
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ));
        assert!(OutputParser::prompt_opens_reasoning(
            TemplateFamily::Glm,
            "<|assistant|><think>"
        ));
        assert!(!OutputParser::prompt_opens_reasoning(
            TemplateFamily::Glm,
            "<|assistant|></think>"
        ));
        assert!(OutputParser::prompt_opens_reasoning(
            TemplateFamily::Gemma4,
            "<tool_response|><|channel>thought\n"
        ));
        assert!(!OutputParser::prompt_opens_reasoning(
            TemplateFamily::Gemma4,
            "<|turn>model\n<|channel>thought\n<channel|>"
        ));
        assert!(!OutputParser::prompt_opens_reasoning(
            TemplateFamily::Harmony,
            "<|start|>assistant"
        ));
        assert!(!OutputParser::prompt_opens_reasoning(
            TemplateFamily::Hermes,
            "<|im_start|>assistant\n"
        ));
    }

    #[test]
    fn end_always_delivers_an_object() {
        let mut em = Emitter::new(ToolTable::default(), false);
        let i = em.start("f", None);
        em.end(i, Value::Null);
        em.end(i, serde_json::json!([1]));
        let ev = em.take();
        assert_eq!(
            ev[1],
            OutputEvent::ToolCallEnd {
                index: 0,
                arguments: serde_json::json!({})
            }
        );
        assert_eq!(
            ev[2],
            OutputEvent::ToolCallEnd {
                index: 0,
                arguments: serde_json::json!({"value": [1]})
            }
        );
    }
}
