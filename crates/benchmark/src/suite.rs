use serde::Deserialize;

const DEFAULT_SUITE: &str = include_str!("../../../benchmarks/suites/default.toml");

#[derive(Deserialize, Clone, Debug)]
pub struct Case {
    pub id: String,
    pub max_tokens: u32,
    pub prompt: String,
    #[serde(default)]
    pub expect_any: Vec<String>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Suite {
    pub name: String,
    pub version: u32,
    #[serde(default)]
    pub perf: Vec<Case>,
    #[serde(default)]
    pub quality: Vec<Case>,
}

impl Suite {
    pub fn builtin() -> Self {
        Self::parse(DEFAULT_SUITE).expect("built-in suite is valid")
    }

    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        Self::parse(&std::fs::read_to_string(path)?)
    }

    pub fn parse(text: &str) -> anyhow::Result<Self> {
        let mut s: Suite = toml::from_str(text)?;
        for c in s.perf.iter_mut().chain(s.quality.iter_mut()) {
            c.prompt = expand(&c.prompt)?;
            c.expect_any = c
                .expect_any
                .iter()
                .map(|e| expand(e))
                .collect::<anyhow::Result<_>>()?;
        }
        Ok(s)
    }
}

pub fn locker_code(i: u32) -> u32 {
    (i.wrapping_mul(7919) % 9000) + 1000
}

/// Expand `{{records:N}}`, `{{log:N}}`, `{{code:K}}` deterministically.
pub fn expand(text: &str) -> anyhow::Result<String> {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("{{") {
        out.push_str(&rest[..start]);
        let end = rest[start..]
            .find("}}")
            .ok_or_else(|| anyhow::anyhow!("unclosed placeholder"))?
            + start;
        let inner = &rest[start + 2..end];
        let (kind, n) = inner
            .split_once(':')
            .ok_or_else(|| anyhow::anyhow!("bad placeholder {{{{{inner}}}}}"))?;
        let n: u32 = n.trim().parse()?;
        match kind.trim() {
            "records" => {
                for i in 1..=n {
                    out.push_str(&format!(
                        "Record {i}: the access code for locker {i} is {}.\n",
                        locker_code(i)
                    ));
                }
            }
            "log" => {
                let comps = [
                    "api-gateway",
                    "auth",
                    "billing",
                    "search",
                    "cache",
                    "scheduler",
                ];
                let levels = ["INFO", "INFO", "WARN", "INFO", "ERROR", "INFO", "DEBUG"];
                for i in 0..n {
                    let c = comps[(i as usize * 7) % comps.len()];
                    let l = levels[(i as usize * 3) % levels.len()];
                    let ms = 20 + (i * 37) % 900;
                    out.push_str(&format!(
                        "2026-09-28T10:{:02}:{:02}Z {l} {c}: request {} completed in {ms} ms (status {})\n",
                        (i / 60) % 60,
                        i % 60,
                        10_000 + i * 13,
                        if l == "ERROR" { 503 } else { 200 }
                    ));
                }
            }
            "code" => out.push_str(&locker_code(n).to_string()),
            other => anyhow::bail!("unknown placeholder kind '{other}'"),
        }
        rest = &rest[end + 2..];
    }
    out.push_str(rest);
    Ok(out)
}

/// True if `needle` occurs in `hay` as a whole word (not inside a longer alphanumeric run),
/// so an expected "42" does not match "142" or "tok42".
pub fn contains_word(hay: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    let is_word = |c: char| c.is_alphanumeric() || c == '_';
    hay.match_indices(needle).any(|(i, _)| {
        let before = hay[..i].chars().next_back();
        let after = hay[i + needle.len()..].chars().next();
        !before.is_some_and(is_word) && !after.is_some_and(is_word)
    })
}

/// Strip `<think>…</think>` blocks so checks look at the final answer.
pub fn final_answer(text: &str) -> String {
    let mut s = text.to_string();
    while let Some(a) = s.find("<think>") {
        match s[a..].find("</think>") {
            Some(b) => s.replace_range(a..a + b + "</think>".len(), ""),
            None => s.truncate(a),
        }
    }
    s.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_suite_expands() {
        let s = Suite::builtin();
        assert_eq!(s.perf.len(), 3);
        let long = s.quality.iter().find(|c| c.id == "long-retrieval").unwrap();
        assert!(long.prompt.contains("Record 220:"));
        assert!(!long.prompt.contains("{{"));
        assert_eq!(long.expect_any, vec![locker_code(137).to_string()]);
        assert!(long
            .prompt
            .contains(&format!("locker 137 is {}.", locker_code(137))));
    }

    #[test]
    fn speculative_suite_parses() {
        let s = Suite::parse(include_str!("../../../benchmarks/suites/speculative.toml")).unwrap();
        let ids: Vec<&str> = s.perf.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, ["prose", "code-edit", "quote"]);
        assert!(s.perf[1]
            .prompt
            .contains("def compute_theta(values, scale, offset):"));
        assert!(s.perf[2]
            .prompt
            .contains("Record 40: the access code for locker 40 is"));
        assert_eq!(s.quality.len(), 3);
    }

    #[test]
    fn deterministic() {
        assert_eq!(expand("{{log:3}}").unwrap(), expand("{{log:3}}").unwrap());
        assert!(expand("{{nope:1}}").is_err());
        assert!(expand("{{records:2").is_err());
    }

    #[test]
    fn word_matching() {
        assert!(contains_word("The answer is 42.", "42"));
        assert!(contains_word("42", "42"));
        assert!(contains_word("**Paris**", "Paris"));
        assert!(!contains_word("tok42 tok43", "42"));
        assert!(!contains_word("142", "42"));
        assert!(!contains_word("Parisian", "Paris"));
    }

    #[test]
    fn strips_thinking() {
        assert_eq!(final_answer("<think>hmm 41?</think>\n42"), "42");
        assert_eq!(final_answer("<think>never closed"), "");
        assert_eq!(final_answer("plain"), "plain");
    }
}
