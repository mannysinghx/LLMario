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
}

/// Outcome of one request, recorded when its response finishes or is dropped.
pub struct RequestRecord<'a> {
    pub model: &'a str,
    /// `ok`, `cancelled`, or an error code.
    pub outcome: &'a str,
    pub ttft_s: Option<f64>,
    pub duration_s: f64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

impl Metrics {
    pub fn record(&self, r: RequestRecord<'_>) {
        let mut g = self.inner.lock().unwrap();
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
        let esc = |v: &str| v.replace('\\', "\\\\").replace('"', "\\\"");
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
        let _ = writeln!(s, "# HELP llmario_engine_memory_bytes Engine physical footprint (macOS) or RSS (Linux).\n# TYPE llmario_engine_memory_bytes gauge");
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
            completion_tokens: 5,
        });
        m.record(RequestRecord {
            model: "a",
            outcome: "cancelled",
            ttft_s: None,
            duration_s: 0.3,
            prompt_tokens: 3,
            completion_tokens: 0,
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
