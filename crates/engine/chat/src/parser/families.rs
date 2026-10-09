//! The marker-driven text machine shared by every family except Harmony, and the per-family
//! marker table.
//!
//! | Family | Reasoning | Call opener(s) | Turn-end markers dropped |
//! |---|---|---|---|
//! | Hermes (Qwen3, SmolLM3, Granite) | `<think>` … `</think>` | `<tool_call>` | `<\|im_end\|>`, `<\|eot_id\|>`, `<\|end_of_text\|>` |
//! | QwenXml (Qwen3.5, Qwen3-Coder, Nemotron 3) | `<think>` … `</think>` | `<tool_call>` | `<\|im_end\|>` |
//! | Glm (GLM-4.7) | `<think>` … `</think>` | `<tool_call>` | `<\|user\|>`, `<\|observation\|>`, `<\|endoftext\|>` |
//! | Gemma4 | `<\|channel>thought\n` … `<channel\|>` | `<\|tool_call>` | `<turn\|>`, `<eos>`, `<\|tool_response>` |
//! | Llama3 (3.x JSON, 4 pythonic) | none | `<\|python_tag\|>`, `<function=`, or `{` / `[` at the start of the output | `<\|eom_id\|>`, `<\|eot_id\|>`, `<\|eot\|>` |
//! | Mistral | `[THINK]` … `[/THINK]` | `[TOOL_CALLS]` | `</s>` |
//! | Lfm2 | `<think>` … `</think>` | `<\|tool_call_start\|>` (closer `<\|tool_call_end\|>`) | `<\|im_end\|>` |
//! | Olmo3 | `<think>` … `</think>` | `<function_calls>` (closer `</function_calls>`) | `<\|im_end\|>`, `<\|endoftext\|>` |
//! | ChatMl, Unknown | `<think>` … `</think>` | `<tool_call>` (Hermes JSON, the de-facto generic format) | `<\|im_end\|>` |
//!
//! Sources: the `tokenizer.chat_template` of the local GGUFs (Qwen3-1.7B, Qwen3.5-0.8B,
//! gemma-4-12b-it, SmolLM3-3B, OLMo-3-7B), and the public templates / prompt-format documents
//! named in `bodies.rs` for GLM-4.7-Flash, Devstral Small 2, LFM2-1.2B, Llama 3.1 and Llama 4.

use super::bodies::{new_body, BodyKind, CallBody};
use super::scan::{scan, Scan};
use super::{Emitter, ParserOptions};
use crate::TemplateFamily;

/// Marker strings of one family.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FamilySpec {
    pub think_open: &'static str,
    pub think_close: &'static str,
    pub openers: &'static [&'static str],
    pub turn_end: &'static [&'static str],
    pub kind: BodyKind,
    /// A `{` or `[` as the first non-blank output opens a call (Llama 3 JSON without
    /// `<|python_tag|>`, Llama 4 pythonic lists).
    pub start_call: bool,
}

const HERMES: FamilySpec = FamilySpec {
    think_open: "<think>",
    think_close: "</think>",
    openers: &["<tool_call>"],
    turn_end: &["<|im_end|>", "<|eot_id|>", "<|end_of_text|>"],
    kind: BodyKind::Hermes,
    start_call: false,
};

pub(crate) fn spec(family: TemplateFamily) -> FamilySpec {
    match family {
        TemplateFamily::Hermes | TemplateFamily::ChatMl | TemplateFamily::Unknown => HERMES,
        TemplateFamily::QwenXml => FamilySpec {
            openers: &["<tool_call>"],
            turn_end: &["<|im_end|>"],
            kind: BodyKind::QwenXml,
            ..HERMES
        },
        TemplateFamily::Glm => FamilySpec {
            openers: &["<tool_call>"],
            turn_end: &["<|user|>", "<|observation|>", "<|endoftext|>"],
            kind: BodyKind::Glm,
            ..HERMES
        },
        TemplateFamily::Gemma4 => FamilySpec {
            think_open: "<|channel>thought\n",
            think_close: "<channel|>",
            openers: &["<|tool_call>"],
            // The template closes a turn that has tool calls with `<|tool_response>` (the result
            // follows in the same turn) rather than `<turn|>`.
            turn_end: &["<turn|>", "<eos>", "<|tool_response>"],
            kind: BodyKind::Gemma4,
            start_call: false,
        },
        TemplateFamily::Llama3 => FamilySpec {
            think_open: "",
            think_close: "",
            openers: &["<|python_tag|>", "<function="],
            turn_end: &["<|eom_id|>", "<|eot_id|>", "<|eot|>"],
            kind: BodyKind::Llama,
            start_call: true,
        },
        TemplateFamily::Mistral => FamilySpec {
            think_open: "[THINK]",
            think_close: "[/THINK]",
            openers: &["[TOOL_CALLS]"],
            turn_end: &["</s>"],
            kind: BodyKind::Mistral,
            start_call: false,
        },
        TemplateFamily::Lfm2 => FamilySpec {
            openers: &["<|tool_call_start|>"],
            turn_end: &["<|im_end|>"],
            kind: BodyKind::Pythonic {
                closer: Some("<|tool_call_end|>"),
            },
            ..HERMES
        },
        TemplateFamily::Olmo3 => FamilySpec {
            openers: &["<function_calls>"],
            turn_end: &["<|im_end|>", "<|endoftext|>"],
            kind: BodyKind::Pythonic {
                closer: Some("</function_calls>"),
            },
            ..HERMES
        },
        TemplateFamily::Harmony => HERMES, // never used: Harmony has its own machine
    }
}

enum Mode {
    Content,
    Reasoning,
    Call {
        opener: String,
        body: Box<dyn CallBody>,
    },
}

/// Text machine: content / reasoning / inside a call, with marker hold-back.
pub(crate) struct Tagged {
    spec: FamilySpec,
    buf: String,
    mode: Mode,
    /// Whether any content, reasoning or call has been produced (gates `start_call` and the
    /// swallowing of a repeated reasoning opener).
    emitted: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Marker {
    ThinkOpen,
    ThinkClose,
    Opener(usize),
    TurnEnd,
}

impl Tagged {
    pub(crate) fn new(spec: FamilySpec, options: ParserOptions) -> Tagged {
        let mode = if options.reasoning_open && !spec.think_close.is_empty() {
            Mode::Reasoning
        } else {
            Mode::Content
        };
        Tagged {
            spec,
            buf: String::new(),
            mode,
            emitted: false,
        }
    }

    /// Markers relevant in content mode, with their meaning.
    fn content_markers(&self) -> (Vec<&'static str>, Vec<Marker>) {
        let mut names = Vec::new();
        let mut kinds = Vec::new();
        if !self.spec.think_open.is_empty() {
            names.push(self.spec.think_open);
            kinds.push(Marker::ThinkOpen);
            names.push(self.spec.think_close);
            kinds.push(Marker::ThinkClose);
        }
        for (i, o) in self.spec.openers.iter().enumerate() {
            names.push(o);
            kinds.push(Marker::Opener(i));
        }
        for t in self.spec.turn_end {
            names.push(t);
            kinds.push(Marker::TurnEnd);
        }
        (names, kinds)
    }

    fn reasoning_markers(&self) -> (Vec<&'static str>, Vec<Marker>) {
        let mut names = vec![self.spec.think_close];
        let mut kinds = vec![Marker::ThinkClose];
        for t in self.spec.turn_end {
            names.push(t);
            kinds.push(Marker::TurnEnd);
        }
        names.push(self.spec.think_open);
        kinds.push(Marker::ThinkOpen);
        (names, kinds)
    }

    pub(crate) fn push(&mut self, text: &str, em: &mut Emitter) {
        self.buf.push_str(text);
        self.run(em);
    }

    fn run(&mut self, em: &mut Emitter) {
        loop {
            match &mut self.mode {
                Mode::Content => {
                    if self.spec.start_call && !self.emitted {
                        let trimmed = self.buf.trim_start();
                        if trimmed.is_empty() {
                            return;
                        }
                        if trimmed.starts_with('{') || trimmed.starts_with('[') {
                            let skip = self.buf.len() - trimmed.len();
                            self.buf.drain(..skip);
                            self.emitted = true;
                            self.mode = Mode::Call {
                                opener: String::new(),
                                body: new_body(self.spec.kind, ""),
                            };
                            continue;
                        }
                    }
                    let (names, kinds) = self.content_markers();
                    match scan(&self.buf, &names) {
                        Scan::Found { start, which, end } => {
                            let before = self.buf[..start].to_string();
                            if !before.is_empty() {
                                em.content(&before);
                                self.emitted = true;
                            }
                            let opener = names[which].to_string();
                            self.buf.drain(..end);
                            match kinds[which] {
                                Marker::ThinkOpen => self.mode = Mode::Reasoning,
                                Marker::ThinkClose | Marker::TurnEnd => {}
                                Marker::Opener(_) => {
                                    self.emitted = true;
                                    self.mode = Mode::Call {
                                        body: new_body(self.spec.kind, &opener),
                                        opener,
                                    };
                                }
                            }
                        }
                        Scan::NotFound { release } => {
                            if release > 0 {
                                let out: String = self.buf.drain(..release).collect();
                                em.content(&out);
                                self.emitted = true;
                            }
                            return;
                        }
                    }
                }
                Mode::Reasoning => {
                    let (names, kinds) = self.reasoning_markers();
                    match scan(&self.buf, &names) {
                        Scan::Found { start, which, end } => {
                            let before = self.buf[..start].to_string();
                            match kinds[which] {
                                Marker::ThinkOpen if start == 0 && !self.emitted => {
                                    // The prompt already opened the block and the model
                                    // repeated the opener: swallow it.
                                    self.buf.drain(..end);
                                }
                                Marker::ThinkOpen => {
                                    // A nested opener is just text.
                                    let literal: String = self.buf.drain(..end).collect();
                                    em.reasoning(&literal);
                                    self.emitted = true;
                                }
                                Marker::ThinkClose | Marker::TurnEnd => {
                                    if !before.is_empty() {
                                        em.reasoning(&before);
                                        self.emitted = true;
                                    }
                                    self.buf.drain(..end);
                                    self.mode = Mode::Content;
                                }
                                Marker::Opener(_) => unreachable!("not a reasoning marker"),
                            }
                        }
                        Scan::NotFound { release } => {
                            if release > 0 {
                                let out: String = self.buf.drain(..release).collect();
                                em.reasoning(&out);
                                self.emitted = true;
                            }
                            return;
                        }
                    }
                }
                Mode::Call { opener, body } => match body.advance(&self.buf, em) {
                    Ok(Some(consumed)) => {
                        self.buf.drain(..consumed);
                        self.mode = Mode::Content;
                    }
                    Ok(None) => return,
                    Err(reason) => {
                        em.cancel();
                        em.invalid(reason);
                        let raw = format!("{opener}{}", self.buf);
                        em.content(&raw);
                        self.buf.clear();
                        self.mode = Mode::Content;
                        return;
                    }
                },
            }
        }
    }

    pub(crate) fn finish(&mut self, em: &mut Emitter) {
        self.run(em);
        let rest = std::mem::take(&mut self.buf);
        match std::mem::replace(&mut self.mode, Mode::Content) {
            Mode::Content => {
                if !(self.spec.start_call && !self.emitted && rest.trim().is_empty()) {
                    em.content(&rest);
                }
            }
            Mode::Reasoning => em.reasoning(&rest),
            Mode::Call { opener, mut body } => {
                if let Err(reason) = body.finish(&rest, em) {
                    em.cancel();
                    em.invalid(reason);
                    em.content(&format!("{opener}{rest}"));
                }
            }
        }
    }
}
