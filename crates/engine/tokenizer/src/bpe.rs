//! Byte-pair merging of one pre-tokenized chunk, following llama.cpp's `llm_tokenizer_bpe_session`:
//! symbols start as UTF-8 characters, candidate bigrams sit in a priority queue ordered by merge
//! rank (ties broken by position), stale bigrams are skipped, and symbols that are not tokens
//! after merging fall back to byte tokens.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::unicode::utf8_len;

/// A symbol in the doubly linked chain over the chunk bytes. `len == 0` marks a merged-away symbol.
#[derive(Clone, Copy, Debug)]
struct Sym {
    prev: i32,
    next: i32,
    start: usize,
    len: usize,
}

#[derive(Debug, PartialEq, Eq)]
struct Bigram {
    rank: u32,
    left: i32,
    right: i32,
    size: usize,
}

// `BinaryHeap` is a max-heap: the "greatest" bigram is the one with the lowest rank, then the
// lowest left index (llama.cpp: `l.rank > r.rank || (l.rank == r.rank && l.left > r.left)`).
impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Per-chunk options.
#[derive(Clone, Copy, Debug, Default)]
pub struct BpeOpts {
    /// Emit the whole chunk as one token when it is in the vocabulary.
    pub ignore_merges: bool,
    /// Gemma 4: a chunk consisting only of newlines that is itself a token is emitted whole.
    pub gemma4_newlines: bool,
    /// Chunk bytes are GPT-2 byte-encoded (fallback looks up single bytes as tokens); otherwise
    /// fallback uses `<0xXX>` byte tokens.
    pub byte_encode: bool,
}

/// Reusable scratch state plus the vocabulary maps a BPE run needs.
pub struct BpeSession<'a> {
    token_to_id: &'a HashMap<Box<[u8]>, u32>,
    merges: &'a HashMap<Box<[u8]>, u32>,
    symbols: Vec<Sym>,
    heap: BinaryHeap<Bigram>,
    key: Vec<u8>,
}

impl<'a> BpeSession<'a> {
    pub fn new(
        token_to_id: &'a HashMap<Box<[u8]>, u32>,
        merges: &'a HashMap<Box<[u8]>, u32>,
    ) -> Self {
        BpeSession {
            token_to_id,
            merges,
            symbols: Vec::new(),
            heap: BinaryHeap::new(),
            key: Vec::new(),
        }
    }

    /// Rank of the merge `left right`, if it exists. Merge keys are `left + " " + right`, which is
    /// unambiguous because tokens of merge-based vocabularies never contain a raw space.
    fn rank(&mut self, word: &[u8], l: Sym, r: Sym) -> Option<u32> {
        self.key.clear();
        self.key.extend_from_slice(&word[l.start..l.start + l.len]);
        self.key.push(b' ');
        self.key.extend_from_slice(&word[r.start..r.start + r.len]);
        self.merges.get(self.key.as_slice()).copied()
    }

    fn add_bigram(&mut self, word: &[u8], left: i32, right: i32) {
        if left == -1 || right == -1 {
            return;
        }
        let l = self.symbols[left as usize];
        let r = self.symbols[right as usize];
        if let Some(rank) = self.rank(word, l, r) {
            self.heap.push(Bigram {
                rank,
                left,
                right,
                size: l.len + r.len,
            });
        }
    }

    /// Append the token ids of one chunk to `out`.
    pub fn tokenize_word(&mut self, word: &[u8], opts: BpeOpts, out: &mut Vec<u32>) {
        if word.is_empty() {
            return;
        }
        if opts.ignore_merges || (opts.gemma4_newlines && word.iter().all(|&b| b == b'\n')) {
            if let Some(&id) = self.token_to_id.get(word) {
                out.push(id);
                return;
            }
        }

        self.symbols.clear();
        self.heap.clear();
        let mut offset = 0;
        let mut index: i32 = 0;
        while offset < word.len() {
            let len = utf8_len(word[offset]).min(word.len() - offset);
            self.symbols.push(Sym {
                prev: index - 1,
                next: if offset + len == word.len() {
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
            self.add_bigram(word, i - 1, i);
        }

        while let Some(bigram) = self.heap.pop() {
            let left = self.symbols[bigram.left as usize];
            let right = self.symbols[bigram.right as usize];
            if left.len == 0 || right.len == 0 || left.len + right.len != bigram.size {
                continue; // stale: one side merged since this bigram was queued
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
            self.add_bigram(word, left_prev, bigram.left);
            self.add_bigram(word, bigram.left, merged_next);
        }

        for i in 0..self.symbols.len() {
            let s = self.symbols[i];
            if s.len == 0 {
                continue;
            }
            let piece = &word[s.start..s.start + s.len];
            match self.token_to_id.get(piece) {
                Some(&id) => out.push(id),
                None => {
                    for &b in piece {
                        let id = if opts.byte_encode {
                            self.token_to_id.get(&[b][..]).copied()
                        } else {
                            self.token_to_id.get(byte_token_text(b).as_slice()).copied()
                        };
                        if let Some(id) = id {
                            out.push(id);
                        }
                    }
                }
            }
        }
    }
}

/// `<0xXX>` text of a byte token.
pub fn byte_token_text(b: u8) -> [u8; 6] {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    [
        b'<',
        b'0',
        b'x',
        HEX[(b >> 4) as usize],
        HEX[(b & 15) as usize],
        b'>',
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    type Map = HashMap<Box<[u8]>, u32>;

    fn maps(tokens: &[&str], merges: &[&str]) -> (Map, Map) {
        let t = tokens
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_bytes().into(), i as u32))
            .collect();
        let m = merges
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_bytes().into(), i as u32))
            .collect();
        (t, m)
    }

    #[test]
    fn merges_follow_rank_order() {
        // ranks: "l o" < "lo w" < "e r" ; "lower" -> [low, er]
        let (t, m) = maps(
            &["l", "o", "w", "e", "r", "lo", "low", "er"],
            &["l o", "lo w", "e r"],
        );
        let mut s = BpeSession::new(&t, &m);
        let mut out = vec![];
        s.tokenize_word(
            b"lower",
            BpeOpts {
                byte_encode: true,
                ..Default::default()
            },
            &mut out,
        );
        assert_eq!(out, [6, 7]);
        // Ties on rank are impossible; but equal-rank order falls back to the leftmost pair.
    }

    #[test]
    fn ignore_merges_and_fallback() {
        let (t, m) = maps(&["a", "b", "ab", "abc"], &["a b"]);
        let mut s = BpeSession::new(&t, &m);
        let mut out = vec![];
        s.tokenize_word(
            b"abc",
            BpeOpts {
                ignore_merges: true,
                byte_encode: true,
                ..Default::default()
            },
            &mut out,
        );
        assert_eq!(out, [3]);
        out.clear();
        s.tokenize_word(
            b"abc",
            BpeOpts {
                byte_encode: true,
                ..Default::default()
            },
            &mut out,
        );
        assert_eq!(out, [2]); // "ab" merged, "c" unknown and no byte token -> dropped like llama.cpp
    }

    #[test]
    fn gemma4_newline_and_byte_tokens() {
        let (t, m) = maps(&["\n", "\n\n", "x", "<0x79>"], &[]);
        let mut s = BpeSession::new(&t, &m);
        let opts = BpeOpts {
            gemma4_newlines: true,
            byte_encode: false,
            ..Default::default()
        };
        let mut out = vec![];
        s.tokenize_word(b"\n\n", opts, &mut out);
        assert_eq!(out, [1]);
        out.clear();
        s.tokenize_word(b"xy", opts, &mut out);
        assert_eq!(out, [2, 3]);
        assert_eq!(&byte_token_text(0xAB), b"<0xAB>");
    }
}
