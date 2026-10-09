//! SentencePiece-style tokenization as llama.cpp's `llm_tokenizer_spm_session` does it: start from
//! UTF-8 characters, repeatedly merge the adjacent pair whose concatenation is the vocabulary
//! token with the highest score, then emit tokens, falling back to `<0xXX>` byte tokens.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::bpe::byte_token_text;
use crate::unicode::utf8_len;

#[derive(Clone, Copy, Debug)]
struct Sym {
    prev: i32,
    next: i32,
    start: usize,
    len: usize,
}

#[derive(Debug)]
struct Bigram {
    score: f32,
    left: i32,
    right: i32,
    size: usize,
}

impl PartialEq for Bigram {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Bigram {}

// Highest score first, then lowest left index (llama.cpp's comparator is
// `l.score < r.score || (l.score == r.score && l.left > r.left)`).
impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.left.cmp(&self.left))
    }
}
impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Reusable scratch state plus the vocabulary an SPM run needs.
pub struct SpmSession<'a> {
    token_to_id: &'a HashMap<Box<[u8]>, u32>,
    scores: &'a [f32],
    unk: Option<u32>,
    symbols: Vec<Sym>,
    heap: BinaryHeap<Bigram>,
}

impl<'a> SpmSession<'a> {
    pub fn new(
        token_to_id: &'a HashMap<Box<[u8]>, u32>,
        scores: &'a [f32],
        unk: Option<u32>,
    ) -> Self {
        SpmSession {
            token_to_id,
            scores,
            unk,
            symbols: Vec::new(),
            heap: BinaryHeap::new(),
        }
    }

    fn try_add_bigram(&mut self, text: &[u8], left: i32, right: i32) {
        if left == -1 || right == -1 {
            return;
        }
        let l = self.symbols[left as usize];
        let r = self.symbols[right as usize];
        let piece = &text[l.start..l.start + l.len + r.len];
        if let Some(&id) = self.token_to_id.get(piece) {
            let score = self.scores.get(id as usize).copied().unwrap_or(0.0);
            self.heap.push(Bigram {
                score,
                left,
                right,
                size: piece.len(),
            });
        }
    }

    /// Append the token ids of `text` (already space-escaped) to `out`.
    pub fn tokenize(&mut self, text: &[u8], out: &mut Vec<u32>) {
        self.symbols.clear();
        self.heap.clear();
        let mut offset = 0;
        let mut index: i32 = 0;
        while offset < text.len() {
            let len = utf8_len(text[offset]).min(text.len() - offset);
            self.symbols.push(Sym {
                prev: index - 1,
                next: if offset + len == text.len() {
                    -1
                } else {
                    index + 1
                },
                start: offset,
                len,
            });
            offset += len;
            index += 1;
        }
        for i in 1..self.symbols.len() as i32 {
            self.try_add_bigram(text, i - 1, i);
        }

        while let Some(bigram) = self.heap.pop() {
            let left = self.symbols[bigram.left as usize];
            let right = self.symbols[bigram.right as usize];
            if left.len == 0 || right.len == 0 || left.len + right.len != bigram.size {
                continue;
            }
            let merged_next = right.next;
            {
                let l = &mut self.symbols[bigram.left as usize];
                l.len += right.len;
                l.next = merged_next;
            }
            self.symbols[bigram.right as usize].len = 0;
            if merged_next >= 0 {
                self.symbols[merged_next as usize].prev = bigram.left;
            }
            let left_prev = self.symbols[bigram.left as usize].prev;
            self.try_add_bigram(text, left_prev, bigram.left);
            self.try_add_bigram(text, bigram.left, merged_next);
        }

        // Every merged symbol is a vocabulary token by construction (bigrams are only queued for
        // pieces that are tokens), so llama.cpp's `rev_merge` re-segmentation never fires; only
        // unmerged single characters can miss and those fall back to bytes.
        for i in 0..self.symbols.len() {
            let s = self.symbols[i];
            if s.len == 0 {
                continue;
            }
            let piece = &text[s.start..s.start + s.len];
            match self.token_to_id.get(piece) {
                Some(&id) => out.push(id),
                None => {
                    for &b in piece {
                        if let Some(id) = self.byte_to_token(b) {
                            out.push(id);
                        }
                    }
                }
            }
        }
    }

    /// `<0xXX>` token, else the single-byte token, else the unknown token (llama.cpp throws here).
    fn byte_to_token(&self, b: u8) -> Option<u32> {
        self.token_to_id
            .get(byte_token_text(b).as_slice())
            .or_else(|| self.token_to_id.get(&[b][..]))
            .copied()
            .or(self.unk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn highest_score_merges_first_and_bytes_fall_back() {
        let tokens = [
            "<unk>", "▁", "h", "e", "l", "o", "▁he", "he", "ll", "llo", "ello", "<0xF0>",
        ];
        let scores = [
            0.0, -1.0, -5.0, -5.0, -5.0, -5.0, -2.0, -3.0, -3.5, -2.5, -2.0, -20.0,
        ];
        let map: HashMap<Box<[u8]>, u32> = tokens
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_bytes().into(), i as u32))
            .collect();
        let mut s = SpmSession::new(&map, &scores, Some(0));
        let mut out = vec![];
        s.tokenize("▁hello".as_bytes(), &mut out);
        // "▁he" (-2.0) and "ello" (-2.0) compete; "▁h"? not a token. First merges: he(-3), ll(-3.5),
        // then ▁he(-2.0) beats... llama.cpp picks by score among queued bigrams: queue has
        // he(-3), ll(-3.5), lo? no, "lo" not a token. Pop he -> symbols ▁ he l l o; queue ▁he(-2).
        // Pop ▁he -> ▁he l l o; then ll -> ▁he ll o; "llo"(-2.5) queued -> ▁he llo.
        assert_eq!(out, [6, 9]);
        out.clear();
        s.tokenize("x\u{1F600}".as_bytes(), &mut out);
        // 'x' has no token -> unk; the emoji's first byte is <0xF0>, the rest fall to unk.
        assert_eq!(out, [0, 11, 0, 0, 0]);
    }
}
