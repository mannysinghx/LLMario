//! Streaming stop-string matching.
//!
//! The server detokenises each generated token into a text fragment and pushes it here before
//! emitting it. Because a stop string can straddle fragment boundaries, the matcher tells the
//! server how much of the accumulated text is safe to release: everything, everything but a
//! held-back tail that might be the start of a stop string, or everything before a completed
//! stop string.
//!
//! All offsets are **byte offsets into the concatenation of every fragment pushed since
//! construction (or the last [`StopMatcher::reset`])**. When fragments are pushed as `&str`,
//! every released boundary is a UTF-8 character boundary of that concatenation: a held tail always
//! starts where a stop string's first byte matched, and stop strings start on a character
//! boundary.

/// Outcome of pushing one fragment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopResult {
    /// Nothing in the accumulated text can start a stop string: emit everything pushed so far.
    Flush,
    /// The last `n` bytes of the accumulated text may be the beginning of a stop string. Emit
    /// everything before them and hold them back until a later push decides.
    Hold(usize),
    /// A stop string completed. Emit the accumulated text up to byte `emit_up_to` (the text
    /// before the stop string); the stop string itself is never emitted. Generation should end.
    Matched { emit_up_to: usize },
}

/// Incremental matcher over a set of stop strings.
#[derive(Clone, Debug)]
pub struct StopMatcher {
    stops: Vec<Vec<u8>>,
    /// Longest stop string in bytes (0 when there are none).
    max_len: usize,
    /// Bytes pushed but not yet released; always a suffix of the stream.
    held: Vec<u8>,
    /// Total bytes pushed so far.
    total: usize,
    /// Bytes of the stream the caller may emit (a prefix).
    released: usize,
    /// `emit_up_to` once a stop string matched; the matcher is then finished.
    matched: Option<usize>,
}

/// First occurrence of `needle` in `hay`.
fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > hay.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

impl StopMatcher {
    /// Builds a matcher. Empty stop strings are ignored (they would match everywhere).
    pub fn new(stop_strings: Vec<String>) -> Self {
        let stops: Vec<Vec<u8>> = stop_strings
            .into_iter()
            .filter(|s| !s.is_empty())
            .map(String::into_bytes)
            .collect();
        let max_len = stops.iter().map(Vec::len).max().unwrap_or(0);
        Self {
            stops,
            max_len,
            held: Vec::new(),
            total: 0,
            released: 0,
            matched: None,
        }
    }

    /// `true` when there are no stop strings (every push is a [`StopResult::Flush`]).
    pub fn is_empty(&self) -> bool {
        self.stops.is_empty()
    }

    /// Total bytes pushed so far.
    pub fn total_len(&self) -> usize {
        self.total
    }

    /// Byte offset up to which the stream may currently be emitted. Equals `total_len() - n`
    /// after [`StopResult::Hold`]`(n)`, `total_len()` after a flush and `emit_up_to` after a
    /// match, so a server only has to track what it has already sent.
    pub fn released_len(&self) -> usize {
        self.released
    }

    /// The held-back tail (bytes pushed but not yet released).
    pub fn held(&self) -> &[u8] {
        &self.held
    }

    /// Whether a stop string has matched.
    pub fn is_matched(&self) -> bool {
        self.matched.is_some()
    }

    /// Pushes a text fragment and reports what may be emitted.
    pub fn push(&mut self, text_fragment: &str) -> StopResult {
        self.push_bytes(text_fragment.as_bytes())
    }

    /// Byte-level variant of [`StopMatcher::push`] for detokenisers that stream raw bytes. The
    /// character-boundary guarantee on released offsets then depends on the caller's fragments.
    pub fn push_bytes(&mut self, fragment: &[u8]) -> StopResult {
        if let Some(emit_up_to) = self.matched {
            return StopResult::Matched { emit_up_to };
        }
        self.total += fragment.len();
        if self.stops.is_empty() {
            self.released = self.total;
            return StopResult::Flush;
        }
        self.held.extend_from_slice(fragment);
        let base = self.total - self.held.len();
        let buf = &self.held;

        // Earliest-starting complete match.
        let mut complete: Option<usize> = None;
        for s in &self.stops {
            if let Some(pos) = find(buf, s) {
                complete = Some(complete.map_or(pos, |c| c.min(pos)));
            }
        }

        // Earliest-starting suffix that is a proper prefix of some stop string. Only the last
        // `max_len - 1` bytes can be one.
        let lo = buf.len().saturating_sub(self.max_len - 1);
        let mut partial: Option<usize> = None;
        'outer: for i in lo..buf.len() {
            let tail = &buf[i..];
            for s in &self.stops {
                if s.len() > tail.len() && s.starts_with(tail) {
                    partial = Some(i);
                    break 'outer;
                }
            }
        }

        match (complete, partial) {
            // A stop string that started earlier may still complete and would take precedence
            // (its text-before is shorter), so keep waiting.
            (Some(c), Some(p)) if p < c => self.hold_from(p),
            (Some(c), _) => {
                let emit_up_to = base + c;
                self.matched = Some(emit_up_to);
                self.released = emit_up_to;
                StopResult::Matched { emit_up_to }
            }
            (None, Some(p)) => self.hold_from(p),
            (None, None) => {
                self.held.clear();
                self.released = self.total;
                StopResult::Flush
            }
        }
    }

    /// Call when generation ends for another reason (EOS, `max_tokens`): a held tail that never
    /// became a stop string is released. Returns the final result (`Matched` if one had already
    /// matched, otherwise `Flush`).
    pub fn finish(&mut self) -> StopResult {
        if let Some(emit_up_to) = self.matched {
            return StopResult::Matched { emit_up_to };
        }
        self.held.clear();
        self.released = self.total;
        StopResult::Flush
    }

    /// Clears all state (offsets restart at 0); the stop strings are kept.
    pub fn reset(&mut self) {
        self.held.clear();
        self.total = 0;
        self.released = 0;
        self.matched = None;
    }

    fn hold_from(&mut self, start: usize) -> StopResult {
        self.held.drain(..start);
        self.released = self.total - self.held.len();
        StopResult::Hold(self.held.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matcher(stops: &[&str]) -> StopMatcher {
        StopMatcher::new(stops.iter().map(|s| s.to_string()).collect())
    }

    /// Drives the matcher like a server would: emits the released prefix and returns the emitted
    /// text plus whether a stop matched. Also checks every released offset is a char boundary.
    fn drive(stops: &[&str], fragments: &[&str]) -> (String, bool) {
        let mut m = matcher(stops);
        let mut stream = String::new();
        let mut sent = 0usize;
        let mut matched = false;
        for f in fragments {
            stream.push_str(f);
            let r = m.push(f);
            match r {
                StopResult::Flush => assert_eq!(m.released_len(), stream.len()),
                StopResult::Hold(n) => assert_eq!(m.released_len(), stream.len() - n),
                StopResult::Matched { emit_up_to } => {
                    assert_eq!(m.released_len(), emit_up_to);
                    matched = true;
                }
            }
            assert!(
                stream.is_char_boundary(m.released_len()),
                "{r:?} on {stream:?}"
            );
            assert!(m.released_len() >= sent);
            sent = m.released_len();
            if matched {
                break;
            }
        }
        if !matched {
            assert_eq!(m.finish(), StopResult::Flush);
            sent = m.released_len();
        }
        (stream[..sent].to_string(), matched)
    }

    #[test]
    fn no_stop_strings_always_flush() {
        let mut m = matcher(&[]);
        assert!(m.is_empty());
        assert_eq!(m.push("hello"), StopResult::Flush);
        assert_eq!(m.push("</s>"), StopResult::Flush);
        assert_eq!(m.released_len(), 9);
        assert_eq!(m.finish(), StopResult::Flush);
        // Empty stop strings are dropped.
        let m = matcher(&["", ""]);
        assert!(m.is_empty());
    }

    #[test]
    fn eos_string_byte_by_byte() {
        let mut m = matcher(&["</s>"]);
        assert_eq!(m.push("Hi"), StopResult::Flush);
        assert_eq!(m.push("<"), StopResult::Hold(1));
        assert_eq!(m.push("/"), StopResult::Hold(2));
        assert_eq!(m.push("s"), StopResult::Hold(3));
        assert_eq!(m.push(">"), StopResult::Matched { emit_up_to: 2 });
        assert!(m.is_matched());
        // Further pushes keep reporting the match.
        assert_eq!(m.push("more"), StopResult::Matched { emit_up_to: 2 });
        assert_eq!(m.finish(), StopResult::Matched { emit_up_to: 2 });
        assert_eq!(
            drive(&["</s>"], &["H", "i", "<", "/", "s", ">", "x"]),
            ("Hi".into(), true)
        );
    }

    #[test]
    fn false_start_is_released() {
        let mut m = matcher(&["</s>"]);
        assert_eq!(m.push("a<"), StopResult::Hold(1));
        assert_eq!(m.push("/b"), StopResult::Flush);
        assert_eq!(m.released_len(), 4);
        // A false start immediately followed by a real start.
        assert_eq!(m.push("<<"), StopResult::Hold(1));
        assert_eq!(m.held(), b"<");
        assert_eq!(m.push("/s>"), StopResult::Matched { emit_up_to: 5 });
        assert_eq!(
            drive(&["</s>"], &["a<", "/b", "<<", "/s>"]),
            ("a</b<".into(), true)
        );
    }

    #[test]
    fn stop_inside_one_fragment() {
        let mut m = matcher(&["STOP"]);
        assert_eq!(m.push("abcSTOPdef"), StopResult::Matched { emit_up_to: 3 });
        let mut m = matcher(&["STOP"]);
        assert_eq!(m.push("STOP"), StopResult::Matched { emit_up_to: 0 });
    }

    #[test]
    fn overlapping_stops_ab_abc() {
        // "ab" completes first; "abc" starts at the same place, so the match is immediate.
        let mut m = matcher(&["ab", "abc"]);
        assert_eq!(m.push("x"), StopResult::Flush);
        assert_eq!(m.push("a"), StopResult::Hold(1));
        assert_eq!(m.push("b"), StopResult::Matched { emit_up_to: 1 });
        // Same with the list order reversed.
        let mut m = matcher(&["abc", "ab"]);
        assert_eq!(m.push("xab"), StopResult::Matched { emit_up_to: 1 });
        assert_eq!(
            drive(&["abc", "ab"], &["x", "a", "b", "c"]),
            ("x".into(), true)
        );
    }

    #[test]
    fn earlier_partial_takes_precedence_over_later_complete() {
        // "bc" is complete at 1 but "abcd" could still complete from 0 -> hold.
        let mut m = matcher(&["abcd", "bc"]);
        assert_eq!(m.push("abc"), StopResult::Hold(3));
        let mut m2 = m.clone();
        assert_eq!(m.push("d"), StopResult::Matched { emit_up_to: 0 });
        // If the earlier candidate fails, the later complete match wins.
        assert_eq!(m2.push("x"), StopResult::Matched { emit_up_to: 1 });
        // Chunking does not change the outcome.
        assert_eq!(drive(&["abcd", "bc"], &["abcd"]), ("".into(), true));
        assert_eq!(
            drive(&["abcd", "bc"], &["a", "b", "c", "d"]),
            ("".into(), true)
        );
        assert_eq!(drive(&["abcd", "bc"], &["abcx"]), ("a".into(), true));
        assert_eq!(drive(&["abcd", "bc"], &["ab", "cx"]), ("a".into(), true));
    }

    #[test]
    fn multibyte_stop_split_across_fragments() {
        // Stop "。」" is 6 bytes (two 3-byte characters).
        let stop = "。」";
        let mut m = matcher(&[stop]);
        assert_eq!(m.push("終わり"), StopResult::Flush);
        assert_eq!(m.push("。"), StopResult::Hold(3));
        assert_eq!(m.push("」"), StopResult::Matched { emit_up_to: 9 });
        assert_eq!(
            drive(&[stop], &["終わり", "。", "」", "x"]),
            ("終わり".into(), true)
        );
        // A false start with a multi-byte first character.
        assert_eq!(
            drive(&[stop], &["終", "。", "終", "。」"]),
            ("終。終".into(), true)
        );
        // Emoji stop with a 4-byte character, fragments of varying sizes.
        assert_eq!(
            drive(&["🛑end"], &["go ", "🛑", "e", "nd", "!"]),
            ("go ".into(), true)
        );
        assert_eq!(
            drive(&["🛑end"], &["go ", "🛑", "e", "x"]),
            ("go 🛑ex".into(), false)
        );
    }

    #[test]
    fn repeated_prefix_characters() {
        // Stop "aab": the stream "aaab" must hold correctly through the repeated 'a's.
        let mut m = matcher(&["aab"]);
        assert_eq!(m.push("a"), StopResult::Hold(1));
        assert_eq!(m.push("a"), StopResult::Hold(2));
        // "aaa": "aa" (from index 1) is still a prefix of "aab"; index 0 "aaa" is not.
        assert_eq!(m.push("a"), StopResult::Hold(2));
        assert_eq!(m.released_len(), 1);
        assert_eq!(m.push("b"), StopResult::Matched { emit_up_to: 1 });
        assert_eq!(drive(&["aab"], &["a", "a", "a", "b"]), ("a".into(), true));
        assert_eq!(drive(&["aab"], &["aaab"]), ("a".into(), true));
    }

    #[test]
    fn finish_releases_held_tail_and_reset_clears() {
        let mut m = matcher(&["</s>"]);
        assert_eq!(m.push("done<"), StopResult::Hold(1));
        assert_eq!(m.finish(), StopResult::Flush);
        assert_eq!(m.released_len(), 5);
        assert!(m.held().is_empty());
        m.reset();
        assert_eq!(m.total_len(), 0);
        assert_eq!(m.released_len(), 0);
        assert_eq!(m.push("</s>"), StopResult::Matched { emit_up_to: 0 });
        m.reset();
        assert!(!m.is_matched());
        assert_eq!(m.push("ok"), StopResult::Flush);
    }

    #[test]
    fn many_stops_long_stream() {
        let stops = ["\n\n", "User:", "<|im_end|>", "###"];
        let text = "Hello there.\nThis is a reply with # and ## marks, <|im and User-ish words.\n\nUser: next";
        let frags: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        let frag_refs: Vec<&str> = frags.iter().map(String::as_str).collect();
        let (out, matched) = drive(&stops, &frag_refs);
        assert!(matched);
        assert_eq!(
            out,
            "Hello there.\nThis is a reply with # and ## marks, <|im and User-ish words."
        );
        // Whole-text push gives the same answer.
        assert_eq!(drive(&stops, &[text]), (out, true));
    }
}
