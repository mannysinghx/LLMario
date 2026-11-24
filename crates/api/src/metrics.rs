//! In-process metrics rendered in Prometheus text format. Recorded once per request, never
//! per token, so the token path stays lock-free.

use llmario_supervisor::EngineInfo;
use std::collections::BTreeMap;
use std::fmt::Write;
use std::sync::Mutex;

const BUCKETS: &[f64] = &[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 120.0];

#[derive(Default, Clone)]
struct Histogram {
    counts: Vec<u64>,
    sum: f64,
    count: u64,
}

impl Histogram {
    fn observe(&mut self, v: f64) {
        if self.counts.is_empty() {
            self.counts = vec![0; BUCKETS.len()];
        }
        for (i, b) in BUCKETS.iter().enumerate() {
            if v <= *b {
                self.counts[i] += 1;
            }
        }
        self.sum += v;
        self.count += 1;
    }
}

#[derive(Default)]
struct Inner {
    requests: BTreeMap<(String, String), u64>,
    prompt_tokens: BTreeMap<String, u64>,
    completion_tokens: BTreeMap<String, u64>,
    ttft: BTreeMap<String, Histogram>,
    duration: BTreeMap<String, Histogram>,
}

#[derive(Default)]
pub struct Metrics {
    inner: Mutex<Inner>,
    /// In-flight and recent requests for the runtime monitor (`GET /llmario/monitor`).
    pub monitor: crate::monitor::Monitor,
}

/// Outcome of one request, recorded when its response finishes or is dropped.
pub struct RequestRecord<'a> {
    pub model: &'a str,
    /// `ok`, `cancelled`, or an error code.
    pub outcome: &'a str,
    pub ttft_s: Option<f64>,
    pub duration_s: f64,
    pub prompt_tokens: u64,
    /// Prompt tokens the engine served from its cache, when its usage says so.
    pub cached_tokens: Option<u64>,
    pub completion_tokens: u64,
    /// Decode speed as the engine timed it (`timings`), when it reports one.
    pub engine_decode_tok_s: Option<f64>,
}

/// Longest model label kept, and the most distinct labels the registry holds.
const LABEL_MAX_LEN: usize = 96;
const LABEL_MAX_SERIES: usize = 256;

/// A metric label (and log field) for a model name a client sent. The name is untrusted and
/// unbounded: it is cut to [`LABEL_MAX_LEN`] characters of `[A-Za-z0-9._:/@+-]` (anything else,
/// line breaks included, becomes `_`), so it can neither forge exposition or log lines nor grow
/// without limit.
pub fn label(model: &str) -> String {
    label_up_to(model, LABEL_MAX_LEN)
}

/// [`label`] with another length limit. The runtime monitor keeps names up to 256 characters,
/// so a long registered model id still matches the engine it ran on.
pub fn label_up_to(model: &str, max_len: usize) -> String {
    let s: String = model
        .chars()
        .take(max_len)
        .map(|c| {
            if c.is_ascii_alphanumeric() || "._:/@+-".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect();
    if s.is_empty() {
        "_".into()
    } else {
        s
    }
}

impl Metrics {
    pub fn record(&self, r: RequestRecord<'_>) {
        // The monitor's ring is bounded on its own (64 entries), so it keeps the real name
        // rather than the Prometheus series cap's "other" or the 96-character cut.
        self.monitor.finished(&RequestRecord {
            model: &label_up_to(r.model, crate::monitor::MODEL_NAME_MAX),
            ..r
        });
        let mut g = self.inner.lock().unwrap();
        // Names that are not served models still arrive (failed requests). Past the series
        // limit they are counted together, so a client cannot grow the maps without bound.
        let mut model = label(r.model);
        if !g.duration.contains_key(&model) && g.duration.len() >= LABEL_MAX_SERIES {
            model = "other".into();
        }
        let r = RequestRecord { model: &model, ..r };
        *g.requests
            .entry((r.model.to_string(), r.outcome.to_string()))
            .or_default() += 1;
        *g.prompt_tokens.entry(r.model.to_string()).or_default() += r.prompt_tokens;
        *g.completion_tokens.entry(r.model.to_string()).or_default() += r.completion_tokens;
        if let Some(t) = r.ttft_s {
            g.ttft.entry(r.model.to_string()).or_default().observe(t);
        }
        g.duration
            .entry(r.model.to_string())
            .or_default()
            .observe(r.duration_s);
    }

    pub fn render(&self, engines: &[EngineInfo]) -> String {
        let g = self.inner.lock().unwrap();
        let mut s = String::new();
        let esc = |v: &str| {
            v.replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', "\\n")
        };
        let _ = writeln!(
            s,
            "# HELP llmario_up Gateway is running.\n# TYPE llmario_up gauge\nllmario_up 1"
        );
        let _ = writeln!(s, "# HELP llmario_requests_total Chat requests by model and outcome.\n# TYPE llmario_requests_total counter");
        for ((m, o), n) in &g.requests {
            let _ = writeln!(
                s,
                "llmario_requests_total{{model=\"{}\",outcome=\"{}\"}} {n}",
                esc(m),
                esc(o)
            );
        }
        for (name, map) in [
            ("prompt", &g.prompt_tokens),
            ("completion", &g.completion_tokens),
        ] {
            let _ = writeln!(s, "# TYPE llmario_{name}_tokens_total counter");
            for (m, n) in map {
                let _ = writeln!(s, "llmario_{name}_tokens_total{{model=\"{}\"}} {n}", esc(m));
            }
        }
        for (name, map, help) in [
            ("llmario_ttft_seconds", &g.ttft, "Time to first token."),
            (
                "llmario_request_duration_seconds",
                &g.duration,
                "End-to-end request duration.",
            ),
        ] {
            let _ = writeln!(s, "# HELP {name} {help}\n# TYPE {name} histogram");
            for (m, h) in map {
                let m = esc(m);
                for (i, b) in BUCKETS.iter().enumerate() {
                    let _ = writeln!(
                        s,
                        "{name}_bucket{{model=\"{m}\",le=\"{b}\"}} {}",
                        h.counts[i]
                    );
                }
                let _ = writeln!(s, "{name}_bucket{{model=\"{m}\",le=\"+Inf\"}} {}", h.count);
                let _ = writeln!(s, "{name}_sum{{model=\"{m}\"}} {}", h.sum);
                let _ = writeln!(s, "{name}_count{{model=\"{m}\"}} {}", h.count);
            }
        }
        let _ = writeln!(
            s,
            "# TYPE llmario_loaded_models gauge\nllmario_loaded_models {}",
            engines.len()
        );
        let _ = writeln!(s, "# TYPE llmario_engine_active_requests gauge");
        for e in engines {
            let _ = writeln!(
                s,
                "llmario_engine_active_requests{{model=\"{}\",backend=\"{}\"}} {}",
                esc(&e.model),
                e.backend,
                e.active_requests
            );
        }
        let _ = writeln!(s, "# HELP llmario_engine_memory_bytes Engine memory: max(physical footprint, resident size) on macOS, RSS on Linux, working set on Windows.\n# TYPE llmario_engine_memory_bytes gauge");
        for e in engines {
            if let Some(b) = e.resident_bytes {
                let _ = writeln!(
                    s,
                    "llmario_engine_memory_bytes{{model=\"{}\"}} {b}",
                    esc(&e.model)
                );
            }
        }
        let _ = writeln!(s, "# TYPE llmario_engine_estimated_bytes gauge");
        for e in engines {
            let _ = writeln!(
                s,
                "llmario_engine_estimated_bytes{{model=\"{}\"}} {}",
                esc(&e.model),
                e.estimated_bytes
            );
        }
        s
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn client_model_names_cannot_forge_lines_or_grow_without_bound() {
        assert_eq!(super::label("qwen3-1.7b-gguf-q4km"), "qwen3-1.7b-gguf-q4km");
        assert_eq!(super::label("a\nllmario_up 0\n\"x"), "a_llmario_up_0__x");
        assert_eq!(super::label(""), "_");
        assert_eq!(super::label(&"x".repeat(5000)).len(), 96);
        let m = super::Metrics::default();
        for i in 0..1000 {
            m.record(super::RequestRecord {
                model: &format!("m{i}"),
                outcome: "invalid_request",
                ttft_s: None,
                duration_s: 0.1,
                prompt_tokens: 0,
                cached_tokens: None,
                completion_tokens: 0,
                engine_decode_tok_s: None,
            });
        }
        let g = m.inner.lock().unwrap();
        assert!(g.duration.len() <= 257 && g.duration.contains_key("other"));
        assert!(g.requests.len() <= 257);
    }

    use super::*;

    #[test]
    fn renders_prometheus_text() {
        let m = Metrics::default();
        m.record(RequestRecord {
            model: "a",
            outcome: "ok",
            ttft_s: Some(0.07),
            duration_s: 1.2,
            prompt_tokens: 10,
            cached_tokens: None,
            completion_tokens: 5,
            engine_decode_tok_s: None,
        });
        m.record(RequestRecord {
            model: "a",
            outcome: "cancelled",
            ttft_s: None,
            duration_s: 0.3,
            prompt_tokens: 3,
            cached_tokens: None,
            completion_tokens: 0,
            engine_decode_tok_s: None,
        });
        let t = m.render(&[]);
        assert!(t.contains("llmario_requests_total{model=\"a\",outcome=\"ok\"} 1"));
        assert!(t.contains("llmario_requests_total{model=\"a\",outcome=\"cancelled\"} 1"));
        assert!(t.contains("llmario_ttft_seconds_bucket{model=\"a\",le=\"0.1\"} 1"));
        assert!(t.contains("llmario_ttft_seconds_bucket{model=\"a\",le=\"0.05\"} 0"));
        assert!(t.contains("llmario_prompt_tokens_total{model=\"a\"} 13"));
        assert!(t.contains("llmario_request_duration_seconds_count{model=\"a\"} 2"));
    }
}
