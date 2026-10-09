//! Incremental detokenization that only ever yields complete UTF-8.

use crate::Tokenizer;

/// Feeds token ids one at a time and returns text as soon as it is complete UTF-8, holding back
/// the trailing bytes of a multi-byte sequence until the next token completes it.
///
/// Rendering rules are those of [`Tokenizer::decode`] (or [`Tokenizer::decode_with_special`] when
/// built with [`Detokenizer::with_special`]), including the SentencePiece leading-space removal.
pub struct Detokenizer<'a> {
    tok: &'a Tokenizer,
    buf: Vec<u8>,
    remove_space: bool,
    special: bool,
}

impl<'a> Detokenizer<'a> {
    /// Special (control/unknown) tokens render as nothing.
    pub fn new(tok: &'a Tokenizer) -> Self {
        Detokenizer {
            tok,
            buf: Vec::new(),
            remove_space: tok.add_space_prefix(),
            special: false,
        }
    }

    /// Special tokens render as their text.
    pub fn with_special(tok: &'a Tokenizer) -> Self {
        Detokenizer {
            special: true,
            ..Detokenizer::new(tok)
        }
    }

    /// Add one token; returns the newly completed text (possibly empty).
    pub fn push(&mut self, id: u32) -> String {
        let start = self.buf.len();
        self.tok.piece_into(id, self.special, &mut self.buf);
        if self.remove_space && self.buf.len() > start {
            if self.buf[start] == b' ' {
                self.buf.remove(start);
            }
            self.remove_space = false;
        }
        self.drain()
    }

    /// Text completed so far; bytes still pending are kept.
    fn drain(&mut self) -> String {
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.buf) {
                Ok(s) => {
                    out.push_str(s);
                    self.buf.clear();
                    return out;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    // Safe: the prefix was just validated.
                    out.push_str(std::str::from_utf8(&self.buf[..valid]).unwrap_or(""));
                    match e.error_len() {
                        None => {
                            // Incomplete sequence at the end: hold it back.
                            self.buf.drain(..valid);
                            return out;
                        }
                        Some(bad) => {
                            out.push('\u{FFFD}');
                            self.buf.drain(..valid + bad);
                        }
                    }
                }
            }
        }
    }

    /// Return whatever is still pending (lossily) and reset the pending buffer.
    pub fn flush(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.buf).into_owned();
        self.buf.clear();
        s
    }

    /// Bytes held back because they do not yet form a complete character.
    pub fn pending(&self) -> &[u8] {
        &self.buf
    }
}
