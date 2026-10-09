//! The page model (after gpt-oss `simple_browser`): 80-column wrap, `L{i}:` numbered lines,
//! token-bounded view windows, `find`, numbered links `【id†text†domain】`, and a per-session
//! LRU page cache bounded by bytes.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use lru::LruCache;

/// Wrap width in characters.
pub const WRAP_COLUMNS: usize = 80;
/// Default view window.
pub const DEFAULT_WINDOW_TOKENS: usize = 1024;
/// Maximum `find` matches.
pub const MAX_FIND_MATCHES: usize = 50;

/// Counts tokens approximately. Pluggable: the engine can supply its tokenizer.
pub trait TokenCounter: Send + Sync {
    /// Token count of `text`.
    fn count(&self, text: &str) -> usize;
}

/// `ceil(chars / 4)`.
#[derive(Debug, Default, Clone, Copy)]
pub struct ApproxTokenCounter;

impl TokenCounter for ApproxTokenCounter {
    fn count(&self, text: &str) -> usize {
        text.chars().count().div_ceil(4)
    }
}

/// A numbered link.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Link {
    /// Id as rendered in the page text.
    pub id: usize,
    /// Anchor text (whitespace-collapsed).
    pub text: String,
    /// Absolute URL.
    pub url: String,
}

impl Link {
    /// `【id†text†domain】`.
    pub fn render(&self) -> String {
        format!("【{}†{}†{}】", self.id, self.text, domain_of(&self.url))
    }
}

/// A fetched, extracted and wrapped page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Page {
    /// Final URL.
    pub url: String,
    /// Title.
    pub title: String,
    /// Wrapped lines (`lines[i]` is `L{i}`).
    pub lines: Vec<String>,
    /// Links by id (ids are indices into this vector).
    pub links: Vec<Link>,
    /// When fetched.
    pub fetched_at: DateTime<Utc>,
}

impl Page {
    /// Build from Markdown: wraps to [`WRAP_COLUMNS`].
    pub fn from_markdown(
        url: impl Into<String>,
        title: impl Into<String>,
        markdown: &str,
        links: Vec<Link>,
    ) -> Self {
        Self {
            url: url.into(),
            title: title.into(),
            lines: wrap_text(markdown, WRAP_COLUMNS),
            links,
            fetched_at: Utc::now(),
        }
    }

    /// Number of lines.
    pub fn total_lines(&self) -> usize {
        self.lines.len()
    }

    /// Approximate memory footprint (for the cache budget).
    pub fn byte_size(&self) -> usize {
        self.url.len()
            + self.title.len()
            + self.lines.iter().map(|l| l.len() + 1).sum::<usize>()
            + self
                .links
                .iter()
                .map(|l| l.text.len() + l.url.len() + 16)
                .sum::<usize>()
    }

    /// Registrable domain-ish host of the page URL.
    pub fn domain(&self) -> String {
        domain_of(&self.url)
    }
}

/// Host of a URL without a `www.` prefix, or the input if it does not parse.
pub fn domain_of(url: &str) -> String {
    url::Url::parse(url)
        .ok()
        .and_then(|u| {
            u.host_str()
                .map(|h| h.trim_start_matches("www.").to_string())
        })
        .unwrap_or_else(|| url.to_string())
}

/// Split on whitespace, keeping `【…】` link markers (which may contain spaces) atomic.
fn split_words(line: &str) -> Vec<&str> {
    let mut words = Vec::new();
    let mut start: Option<usize> = None;
    let mut in_marker = false;
    for (i, c) in line.char_indices() {
        match c {
            '【' => {
                in_marker = true;
                if start.is_none() {
                    start = Some(i);
                }
            }
            '】' => in_marker = false,
            c if c.is_whitespace() && !in_marker => {
                if let Some(s) = start.take() {
                    words.push(&line[s..i]);
                }
            }
            _ => {
                if start.is_none() {
                    start = Some(i);
                }
            }
        }
    }
    if let Some(s) = start {
        words.push(&line[s..]);
    }
    words
}

/// Greedy word wrap on character count; words longer than `width` are hard-broken; blank lines
/// are preserved; leading indentation of a paragraph line is kept on its first output line;
/// `【id†text†domain】` markers are never split.
pub fn wrap_text(text: &str, width: usize) -> Vec<String> {
    let width = width.max(8);
    let mut out = Vec::new();
    for raw in text.lines() {
        let line = raw.trim_end();
        if line.chars().count() <= width {
            out.push(line.to_string());
            continue;
        }
        let indent: String = line.chars().take_while(|c| c.is_whitespace()).collect();
        let mut current = indent.clone();
        let mut current_len = indent.chars().count();
        for word in split_words(line) {
            let wlen = word.chars().count();
            let sep = if current_len > indent.chars().count() {
                1
            } else {
                0
            };
            if current_len + sep + wlen <= width {
                if sep == 1 {
                    current.push(' ');
                }
                current.push_str(word);
                current_len += sep + wlen;
            } else {
                if current_len > indent.chars().count() {
                    out.push(std::mem::take(&mut current));
                    current_len = 0;
                }
                if wlen <= width {
                    current = word.to_string();
                    current_len = wlen;
                } else {
                    // Hard-break an overlong word.
                    let chars: Vec<char> = word.chars().collect();
                    let mut start = 0;
                    while start < chars.len() {
                        let end = (start + width).min(chars.len());
                        let piece: String = chars[start..end].iter().collect();
                        if end == chars.len() {
                            current_len = piece.chars().count();
                            current = piece;
                        } else {
                            out.push(piece);
                        }
                        start = end;
                    }
                }
            }
        }
        if current_len > 0 {
            out.push(current);
        }
    }
    out
}

/// A rendered window of a page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageView {
    /// Page URL.
    pub url: String,
    /// Page title.
    pub title: String,
    /// First line shown.
    pub start: usize,
    /// Last line shown (inclusive).
    pub end: usize,
    /// Total lines in the page.
    pub total: usize,
    /// Text for the model: header + `L{i}: …` lines.
    pub text: String,
}

/// Render the window starting at `cursor` (line index) that fits `window_tokens`. Always shows
/// at least one line. A cursor past the end clamps to the last line.
pub fn view(
    page: &Page,
    cursor: usize,
    window_tokens: usize,
    counter: &dyn TokenCounter,
) -> PageView {
    let total = page.lines.len();
    let start = if total == 0 { 0 } else { cursor.min(total - 1) };
    let mut body = String::new();
    let mut end = start;
    let mut used = 0usize;
    for (i, line) in page.lines.iter().enumerate().skip(start) {
        let numbered = format!("L{i}: {line}\n");
        let cost = counter.count(&numbered);
        if i > start && used + cost > window_tokens {
            break;
        }
        body.push_str(&numbered);
        used += cost;
        end = i;
    }
    let last_index = total.saturating_sub(1);
    let header = format!(
        "{} ({})\n**viewing lines [{} - {}] of {}**\n\n",
        page.title,
        page.domain(),
        start,
        end,
        last_index
    );
    PageView {
        url: page.url.clone(),
        title: page.title.clone(),
        start,
        end,
        total,
        text: header + &body,
    }
}

/// One `find` hit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FindMatch {
    /// Matching line.
    pub line: usize,
    /// Up to four numbered lines starting at the match.
    pub snippet: String,
}

/// Case-insensitive substring search; at most [`MAX_FIND_MATCHES`] hits.
pub fn find(page: &Page, pattern: &str) -> Vec<FindMatch> {
    let needle = pattern.to_lowercase();
    if needle.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (i, line) in page.lines.iter().enumerate() {
        if line.to_lowercase().contains(&needle) {
            let snippet = page
                .lines
                .iter()
                .enumerate()
                .skip(i)
                .take(4)
                .map(|(j, l)| format!("L{j}: {l}"))
                .collect::<Vec<_>>()
                .join("\n");
            out.push(FindMatch { line: i, snippet });
            if out.len() >= MAX_FIND_MATCHES {
                break;
            }
        }
    }
    out
}

/// Render `find` results for the model.
pub fn render_find(page: &Page, pattern: &str, matches: &[FindMatch]) -> String {
    if matches.is_empty() {
        return format!(
            "No `find` results for pattern: `{pattern}` in {} ({})",
            page.title,
            page.domain()
        );
    }
    let mut s = format!(
        "Find results for pattern: `{pattern}` in {} ({})\n\n",
        page.title,
        page.domain()
    );
    for (n, m) in matches.iter().enumerate() {
        s.push_str(&format!(
            "# 【{}†match at L{}】\n{}\n\n",
            n, m.line, m.snippet
        ));
    }
    s
}

/// Per-session page cache: pages by URL and by cursor id, evicted LRU once the byte budget is
/// exceeded.
#[derive(Debug)]
pub struct PageCache {
    by_url: LruCache<String, Arc<Page>>,
    cursor_to_url: HashMap<usize, String>,
    url_to_cursor: HashMap<String, usize>,
    next_cursor: usize,
    bytes: usize,
    max_bytes: usize,
}

impl PageCache {
    /// Cache bounded by `max_bytes` of page content.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            by_url: LruCache::unbounded(),
            cursor_to_url: HashMap::new(),
            url_to_cursor: HashMap::new(),
            next_cursor: 0,
            bytes: 0,
            max_bytes,
        }
    }

    /// Insert (or replace) a page; returns its cursor id.
    pub fn insert(&mut self, page: Page) -> usize {
        let url = page.url.clone();
        let size = page.byte_size();
        if let Some(old) = self.by_url.put(url.clone(), Arc::new(page)) {
            self.bytes = self.bytes.saturating_sub(old.byte_size());
        }
        self.bytes += size;
        let cursor = match self.url_to_cursor.get(&url) {
            Some(c) => *c,
            None => {
                let c = self.next_cursor;
                self.next_cursor += 1;
                self.url_to_cursor.insert(url.clone(), c);
                self.cursor_to_url.insert(c, url);
                c
            }
        };
        self.evict();
        cursor
    }

    fn evict(&mut self) {
        while self.bytes > self.max_bytes && self.by_url.len() > 1 {
            let Some((url, page)) = self.by_url.pop_lru() else {
                break;
            };
            self.bytes = self.bytes.saturating_sub(page.byte_size());
            if let Some(c) = self.url_to_cursor.remove(&url) {
                self.cursor_to_url.remove(&c);
            }
        }
    }

    /// Page by URL (marks it recently used).
    pub fn get(&mut self, url: &str) -> Option<Arc<Page>> {
        self.by_url.get(url).cloned()
    }

    /// Page by cursor id.
    pub fn get_cursor(&mut self, cursor: usize) -> Option<Arc<Page>> {
        let url = self.cursor_to_url.get(&cursor)?.clone();
        self.by_url.get(&url).cloned()
    }

    /// Cursor id for a cached URL.
    pub fn cursor_of(&self, url: &str) -> Option<usize> {
        self.url_to_cursor.get(url).copied()
    }

    /// Cached pages.
    pub fn len(&self) -> usize {
        self.by_url.len()
    }

    /// True when empty.
    pub fn is_empty(&self) -> bool {
        self.by_url.is_empty()
    }

    /// Bytes currently held.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(n: usize) -> Page {
        let md: String = (0..n)
            .map(|i| format!("line {i} with some words"))
            .collect::<Vec<_>>()
            .join("\n");
        Page::from_markdown("https://www.example.com/a", "Title", &md, vec![])
    }

    #[test]
    fn wrap_respects_width_and_preserves_blank_lines() {
        let text = "short\n\n".to_string() + &"word ".repeat(40) + "\n" + &"x".repeat(200);
        let lines = wrap_text(&text, 80);
        assert_eq!(lines[0], "short");
        assert_eq!(lines[1], "");
        assert!(lines.iter().all(|l| l.chars().count() <= 80), "{lines:?}");
        assert_eq!(lines[2], "word ".repeat(16).trim_end());
        assert_eq!(lines.last().unwrap().len(), 40);
        assert_eq!(
            wrap_text("  indented ".to_string().as_str(), 80),
            vec!["  indented"]
        );
        let indented = "    ".to_string() + &"ab ".repeat(40);
        let w = wrap_text(&indented, 20);
        assert!(w[0].starts_with("    ab ab"));
        assert!(w.iter().all(|l| l.chars().count() <= 20));
    }

    #[test]
    fn wrap_keeps_link_markers_atomic() {
        let text = "x ".repeat(36) + "【3†a long link text†example.com】 tail";
        let lines = wrap_text(&text, 80);
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert_eq!(lines[1], "【3†a long link text†example.com】 tail");
        assert_eq!(
            split_words("a 【1†b c†d】 e"),
            vec!["a", "【1†b c†d】", "e"]
        );
        assert_eq!(split_words("【1†b c†d】x y"), vec!["【1†b c†d】x", "y"]);
        // An unterminated marker does not swallow the rest of the line forever: it is one word.
        assert_eq!(split_words("【oops a b"), vec!["【oops a b"]);
    }

    #[test]
    fn view_windows_by_tokens_and_numbers_lines() {
        let p = page(100);
        let v = view(&p, 0, 50, &ApproxTokenCounter);
        assert_eq!(v.start, 0);
        assert!(v.end < 99);
        assert!(v
            .text
            .starts_with("Title (example.com)\n**viewing lines [0 - "));
        assert!(v.text.contains("\nL0: line 0 with some words\n"));
        assert!(!v.text.contains(&format!("L{}:", v.end + 1)));
        let counted = ApproxTokenCounter.count(&v.text[v.text.find("\n\n").unwrap() + 2..]);
        assert!(counted <= 50, "{counted}");
        let v = view(&p, 98, 10_000, &ApproxTokenCounter);
        assert_eq!((v.start, v.end, v.total), (98, 99, 100));
        let v = view(&p, 500, 10_000, &ApproxTokenCounter);
        assert_eq!((v.start, v.end), (99, 99));
        // At least one line even when the budget is tiny.
        let v = view(&p, 3, 0, &ApproxTokenCounter);
        assert_eq!((v.start, v.end), (3, 3));
        let empty = Page::from_markdown("https://e.com", "E", "", vec![]);
        let v = view(&empty, 0, 10, &ApproxTokenCounter);
        assert_eq!(v.total, 0);
        assert!(v.text.contains("[0 - 0] of 0"));
    }

    #[test]
    fn find_is_case_insensitive_and_capped() {
        let p = page(200);
        let m = find(&p, "LINE 7");
        assert_eq!(m.len(), 11, "7, 70-79");
        assert_eq!(m[0].line, 7);
        assert!(m[0].snippet.starts_with("L7: line 7 with some words\nL8: "));
        assert_eq!(m[0].snippet.lines().count(), 4);
        assert_eq!(find(&p, "line").len(), MAX_FIND_MATCHES);
        assert!(find(&p, "").is_empty());
        let r = render_find(&p, "zzz", &find(&p, "zzz"));
        assert!(r.starts_with("No `find` results"));
        let r = render_find(&p, "LINE 7", &m);
        assert!(r.contains("# 【0†match at L7】"));
    }

    #[test]
    fn links_render_with_domain() {
        let l = Link {
            id: 3,
            text: "Docs".into(),
            url: "https://www.rust-lang.org/learn".into(),
        };
        assert_eq!(l.render(), "【3†Docs†rust-lang.org】");
        assert_eq!(domain_of("not a url"), "not a url");
    }

    #[test]
    fn cache_evicts_lru_by_bytes_and_keeps_cursors_stable() {
        let mut c = PageCache::new(page(10).byte_size() * 2 + 10);
        let a = Page {
            url: "https://a".into(),
            ..page(10)
        };
        let b = Page {
            url: "https://b".into(),
            ..page(10)
        };
        let d = Page {
            url: "https://d".into(),
            ..page(10)
        };
        assert_eq!(c.insert(a.clone()), 0);
        assert_eq!(c.insert(b), 1);
        assert!(c.get("https://a").is_some(), "touch a so b is LRU");
        assert_eq!(c.insert(d), 2);
        assert_eq!(c.len(), 2);
        assert!(c.get("https://b").is_none(), "b evicted");
        assert!(c.get_cursor(1).is_none());
        assert_eq!(c.get_cursor(0).unwrap().url, "https://a");
        assert_eq!(c.cursor_of("https://d"), Some(2));
        // Re-inserting a URL keeps its cursor and replaces the content.
        let a2 = Page {
            title: "new".into(),
            ..a
        };
        assert_eq!(c.insert(a2), 0);
        assert_eq!(c.get_cursor(0).unwrap().title, "new");
        assert!(c.bytes() <= c.max_bytes);
    }
}
