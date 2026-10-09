//! Codepoint classification and the GPT-2 byte <-> unicode mapping, transcribed from llama.cpp's
//! `unicode.cpp` (MIT) so that pre-tokenization matches it exactly, including the Unicode version
//! its tables were generated from (see [`crate::unicode_data`]).

use crate::unicode_data::{LOWERCASE, RANGES_FLAGS, WHITESPACE};

pub const UNDEFINED: u16 = 0x0001;
pub const NUMBER: u16 = 0x0002;
pub const LETTER: u16 = 0x0004;
pub const SEPARATOR: u16 = 0x0008;
pub const ACCENT_MARK: u16 = 0x0010;
pub const PUNCTUATION: u16 = 0x0020;
pub const SYMBOL: u16 = 0x0040;
pub const CONTROL: u16 = 0x0080;
pub const MASK_CATEGORIES: u16 = 0x00FF;
pub const WHITESPACE_FLAG: u16 = 0x0100;

/// Flags of one codepoint (llama.cpp's `unicode_cpt_flags`). The zero value means "no codepoint"
/// (used by the splitters for positions outside the current chunk).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Flags(pub u16);

impl Flags {
    pub fn is_number(self) -> bool {
        self.0 & NUMBER != 0
    }
    pub fn is_letter(self) -> bool {
        self.0 & LETTER != 0
    }
    pub fn is_accent_mark(self) -> bool {
        self.0 & ACCENT_MARK != 0
    }
    pub fn is_whitespace(self) -> bool {
        self.0 & WHITESPACE_FLAG != 0
    }
    /// General category bits only.
    pub fn category(self) -> u16 {
        self.0 & MASK_CATEGORIES
    }
    /// llama.cpp's `as_uint() != 0`: true for every real codepoint, false for "no codepoint".
    pub fn any(self) -> bool {
        self.0 != 0
    }
}

/// Flags of a codepoint. Values at or above `0x110000` are undefined.
pub fn flags(cpt: u32) -> Flags {
    if cpt >= 0x11_0000 {
        return Flags(UNDEFINED);
    }
    let idx = RANGES_FLAGS.partition_point(|&(start, _)| start <= cpt);
    // The first range starts at 0, so idx >= 1.
    let mut f = RANGES_FLAGS[idx - 1].1;
    if WHITESPACE.binary_search(&cpt).is_ok() {
        f |= WHITESPACE_FLAG;
    }
    Flags(f)
}

/// Lowercase mapping used by the contraction rules; identity when there is none.
pub fn tolower(cpt: u32) -> u32 {
    match LOWERCASE.binary_search_by_key(&cpt, |&(c, _)| c) {
        Ok(i) => LOWERCASE[i].1,
        Err(_) => cpt,
    }
}

/// Length of the UTF-8 sequence announced by its first byte, as llama.cpp's `unicode_len_utf8`
/// (continuation bytes count as 1 so malformed input still advances).
pub fn utf8_len(first: u8) -> usize {
    const LOOKUP: [u8; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4];
    LOOKUP[(first >> 4) as usize] as usize
}

/// Is `b` one of the bytes GPT-2 keeps as its own codepoint?
const fn printable(b: u8) -> bool {
    (0x21 <= b && b <= 0x7E) || (0xA1 <= b && b <= 0xAC) || 0xAE <= b
}

const fn build_byte_to_cpt() -> [u16; 256] {
    let mut table = [0u16; 256];
    let mut n: u16 = 0;
    let mut b: usize = 0;
    while b < 256 {
        if printable(b as u8) {
            table[b] = b as u16;
        } else {
            table[b] = 256 + n;
            n += 1;
        }
        b += 1;
    }
    table
}

/// Byte -> codepoint of the GPT-2 byte-level encoding (`unicode_byte_to_utf8`).
const BYTE_TO_CPT: [u16; 256] = build_byte_to_cpt();

/// Number of bytes that are not "printable" (they map to U+0100 and up).
const N_SHIFTED: usize = 68;

const fn build_shifted_to_byte() -> [u8; N_SHIFTED] {
    let mut table = [0u8; N_SHIFTED];
    let mut n = 0;
    let mut b: usize = 0;
    while b < 256 {
        if !printable(b as u8) {
            table[n] = b as u8;
            n += 1;
        }
        b += 1;
    }
    table
}

const SHIFTED_TO_BYTE: [u8; N_SHIFTED] = build_shifted_to_byte();

/// The GPT-2 byte-level mapping: every byte becomes one printable codepoint.
pub fn byte_to_char(b: u8) -> char {
    // Every value is below 0x144, so the conversion cannot fail.
    char::from_u32(BYTE_TO_CPT[b as usize] as u32).unwrap_or('\u{FFFD}')
}

/// Inverse of [`byte_to_char`]; `None` for codepoints outside the mapping.
pub fn char_to_byte(c: char) -> Option<u8> {
    let cpt = c as u32;
    if cpt < 256 {
        let b = cpt as u8;
        printable(b).then_some(b)
    } else if (256..256 + N_SHIFTED as u32).contains(&cpt) {
        Some(SHIFTED_TO_BYTE[(cpt - 256) as usize])
    } else {
        None
    }
}

/// Encode raw bytes with the GPT-2 byte-level mapping.
pub fn byte_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() + bytes.len() / 4);
    for &b in bytes {
        s.push(byte_to_char(b));
    }
    s
}

/// Decode a byte-level token text back to raw bytes. Codepoints outside the mapping (which a
/// well-formed byte-level vocabulary never contains) are copied through as their UTF-8 bytes.
pub fn byte_decode(text: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len());
    for c in String::from_utf8_lossy(text).chars() {
        match char_to_byte(c) {
            Some(b) => out.push(b),
            None => {
                let mut buf = [0u8; 4];
                out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_mapping_round_trips_and_matches_gpt2() {
        for b in 0..=255u8 {
            let c = byte_to_char(b);
            assert_eq!(char_to_byte(c), Some(b), "byte {b:#x}");
        }
        assert_eq!(byte_to_char(b' '), '\u{120}'); // Ġ
        assert_eq!(byte_to_char(b'\n'), '\u{10A}'); // Ċ
        assert_eq!(byte_to_char(b'!'), '!');
        assert_eq!(byte_to_char(0xAD), '\u{143}');
        assert_eq!(byte_to_char(0xFF), '\u{FF}');
        assert_eq!(byte_encode(b"Hi there"), "Hi\u{120}there");
        assert_eq!(byte_decode("Hi\u{120}there".as_bytes()), b"Hi there");
    }

    #[test]
    fn flags_follow_llama_cpp_tables() {
        assert!(flags('a' as u32).is_letter());
        assert!(flags('Z' as u32).is_letter());
        assert!(flags('5' as u32).is_number());
        assert!(flags(' ' as u32).is_whitespace());
        assert!(flags('\n' as u32).is_whitespace());
        assert!(flags(0xA0).is_whitespace());
        assert!(flags(0x3000).is_whitespace());
        assert!(!flags(0x1C).is_whitespace());
        assert_eq!(flags('!' as u32).category(), PUNCTUATION);
        assert_eq!(flags('$' as u32).category(), SYMBOL);
        assert_eq!(flags(0x0301).category(), ACCENT_MARK);
        assert!(flags('中' as u32).is_letter());
        assert!(flags('م' as u32).is_letter());
        assert_eq!(flags(0x1F600).category(), SYMBOL); // emoji
        assert_eq!(flags(0x0001).category(), CONTROL);
        assert_eq!(flags(0x11_0000), Flags(UNDEFINED));
        assert!(!Flags::default().any());
        assert!(flags(0xE000).any()); // private use is still a real codepoint
    }

    #[test]
    fn tolower_and_utf8_len() {
        assert_eq!(tolower('S' as u32), 's' as u32);
        assert_eq!(tolower('s' as u32), 's' as u32);
        assert_eq!(tolower('É' as u32), 'é' as u32);
        assert_eq!(tolower(0x1F600), 0x1F600);
        assert_eq!(utf8_len(b'a'), 1);
        assert_eq!(utf8_len(0xC3), 2);
        assert_eq!(utf8_len(0xE2), 3);
        assert_eq!(utf8_len(0xF0), 4);
        assert_eq!(utf8_len(0x80), 1);
    }
}
