use crate::{LevelResult, QualityResult, Sample, Settings, Stat};
use serde::{Deserialize, Serialize};
use std::fmt::Write;

/// Everything needed to reproduce and judge a run. Written as JSON; rendered as Markdown.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Report {
    pub schema: u32,
    pub tool: String,
    pub timestamp: String,
    pub label: String,
    pub environment: Environment,
    pub settings: Settings,
    pub suite: String,
    pub suite_version: u32,
    /// Engine spawn → first warm-up token (llmario-managed targets only).
    pub cold_start_s: Option<f64>,
    /// llmario's pre-load memory estimate, for calibration against measured peak.
    pub estimated_memory_bytes: Option<u64>,
    pub idle_memory_bytes: Option<u64>,
    pub levels: Vec<LevelResult>,
    pub quality: Vec<QualityResult>,
    pub samples: Vec<Sample>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Environment {
    pub hardware_fingerprint: String,
    pub hardware: String,
    pub os: String,
    pub target: String,
    pub backend: Option<String>,
    pub backend_version: Option<String>,
    pub model: String,
    pub model_format: Option<String>,
    pub model_quantization: Option<String>,
    pub model_content_hash: Option<String>,
    pub model_source: Option<String>,
    pub profile: Option<serde_json::Value>,
    pub power_note: String,
    /// Bytes of weight files, for effective bandwidth (llmario-managed targets only).
    #[serde(default)]
    pub model_weight_bytes: Option<u64>,
    /// One-minute load average when the run started, and the CPU's logical cores.
    #[serde(default)]
    pub load_average_1m: Option<f64>,
    #[serde(default)]
    pub logical_cores: Option<usize>,
}

impl Environment {
    /// True when the machine was too busy for trustworthy timings (see [`crate::is_busy`]).
    pub fn busy(&self) -> bool {
        match (self.load_average_1m, self.logical_cores) {
            (Some(l), Some(c)) => crate::is_busy(l, c),
            _ => false,
        }
    }
}

fn f(s: &Option<Stat>, pick: fn(&Stat) -> f64, scale: f64, prec: usize) -> String {
    s.as_ref()
        .map(|s| format!("{:.*}", prec, pick(s) * scale))
        .unwrap_or_else(|| "–".into())
}

fn gib(b: Option<u64>) -> String {
    b.map(|b| format!("{:.2} GiB", b as f64 / (1u64 << 30) as f64))
        .unwrap_or_else(|| "–".into())
}

impl Report {
    pub fn markdown(&self) -> String {
        let e = &self.environment;
        let mut s = String::new();
        let _ = writeln!(s, "# Benchmark: {}\n", self.label);
        if e.busy() {
            let _ = writeln!(
                s,
                "> ⚠ **Busy machine**: these timings are not comparable (load average {:.1} on {} cores when the run started).\n",
                e.load_average_1m.unwrap_or_default(),
                e.logical_cores.unwrap_or_default()
            );
        }
        let _ = writeln!(s, "| | |\n|---|---|");
        let _ = writeln!(s, "| Date | {} |", self.timestamp);
        let _ = writeln!(
            s,
            "| Hardware | {} (fingerprint `{}`) |",
            e.hardware, e.hardware_fingerprint
        );
        let _ = writeln!(s, "| OS | {} |", e.os);
        let _ = writeln!(s, "| Target | {} |", e.target);
        let _ = writeln!(
            s,
            "| Backend | {} {} |",
            e.backend.as_deref().unwrap_or("–"),
            e.backend_version.as_deref().unwrap_or("")
        );
        let _ = writeln!(
            s,
            "| Model | `{}` {} {} (content hash `{}`) |",
            e.model,
            e.model_format.as_deref().unwrap_or(""),
            e.model_quantization.as_deref().unwrap_or(""),
            e.model_content_hash.as_deref().unwrap_or("–")
        );
        if let Some(src) = &e.model_source {
            let _ = writeln!(s, "| Source | {src} |");
        }
        if let Some(p) = &e.profile {
            let _ = writeln!(s, "| Profile | `{}` |", p);
        }
        let _ = writeln!(
            s,
            "| Settings | suite `{}` v{}, temperature {}, seed {}, {} run(s) per case, {} warm-up, {:?} prefix cache |",
            self.suite, self.suite_version, self.settings.temperature, self.settings.seed.map(|s| s.to_string()).unwrap_or_else(|| "none".into()), self.settings.runs, self.settings.warmup, self.settings.cache
        );
        let _ = writeln!(s, "| Power | {} |", e.power_note);
        if let (Some(l), Some(c)) = (e.load_average_1m, e.logical_cores) {
            let _ = writeln!(
                s,
                "| Machine load | {l:.1} (1-min load average) on {c} cores at start: {} |",
                if e.busy() { "busy" } else { "quiet" }
            );
        }
        if let Some(c) = self.cold_start_s {
            let _ = writeln!(s, "| Cold start | {c:.2} s (spawn → first warm-up token) |");
        }
        let _ = writeln!(
            s,
            "| Memory | estimate {}, idle after load {} |",
            gib(self.estimated_memory_bytes),
            gib(self.idle_memory_bytes)
        );

        let _ = writeln!(s, "\n## Throughput and latency by concurrency\n");
        let bw = self
            .levels
            .iter()
            .any(|l| l.effective_bandwidth_gbs.is_some());
        let draft = self.levels.iter().any(|l| l.draft_acceptance.is_some());
        let _ = writeln!(
            s,
            "| conc | req | err | TTFT p50 ms | TTFT p95 ms | E2E p50 s | E2E p95 s | decode tok/s (mean/req) | aggregate tok/s | peak mem |{}{}",
            if bw { " eff. GB/s |" } else { "" },
            if draft { " draft accepted |" } else { "" }
        );
        let _ = writeln!(
            s,
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|{}{}",
            if bw { "---:|" } else { "" },
            if draft { "---:|" } else { "" }
        );
        for l in &self.levels {
            let opt = |v: Option<f64>, fmt: &dyn Fn(f64) -> String| {
                v.map(fmt).unwrap_or_else(|| "–".into())
            };
            let extra = format!(
                "{}{}",
                if bw {
                    format!(
                        " {} |",
                        opt(l.effective_bandwidth_gbs, &|v| format!("{v:.0}"))
                    )
                } else {
                    String::new()
                },
                if draft {
                    format!(
                        " {} |",
                        opt(l.draft_acceptance, &|v| format!("{:.0}%", v * 100.0))
                    )
                } else {
                    String::new()
                }
            );
            let _ = writeln!(
                s,
                "| {} | {} | {} | {} | {} | {} | {} | {} | {:.1} | {} |{extra}",
                l.concurrency,
                l.requests,
                l.errors,
                f(&l.ttft_s, |s| s.p50, 1000.0, 0),
                f(&l.ttft_s, |s| s.p95, 1000.0, 0),
                f(&l.e2e_s, |s| s.p50, 1.0, 2),
                f(&l.e2e_s, |s| s.p95, 1.0, 2),
                f(&l.decode_tps, |s| s.mean, 1.0, 1),
                l.aggregate_decode_tps,
                gib(l.peak_memory_bytes)
            );
        }
        if let Some(l1) = self
            .levels
            .iter()
            .find(|l| l.concurrency == 1)
            .or(self.levels.first())
        {
            let _ = writeln!(s, "\n## Per prompt (concurrency {})\n", l1.concurrency);
            let _ = writeln!(
                s,
                "| case | prompt tokens | TTFT p50 ms | prefill tok/s | decode tok/s |"
            );
            let _ = writeln!(s, "|---|---:|---:|---:|---:|");
            for c in &l1.per_case {
                let _ = writeln!(
                    s,
                    "| {} | {} | {} | {} | {} |",
                    c.case,
                    c.prompt_tokens
                        .map(|p| p.to_string())
                        .unwrap_or_else(|| "–".into()),
                    f(&c.ttft_s, |s| s.p50, 1000.0, 0),
                    f(&c.prefill_tps, |s| s.p50, 1.0, 0),
                    f(&c.decode_tps, |s| s.p50, 1.0, 1)
                );
            }
        }
        if !self.quality.is_empty() {
            let passed = self.quality.iter().filter(|q| q.passed).count();
            let _ = writeln!(s, "\n## Quality: {passed}/{} passed\n", self.quality.len());
            let _ = writeln!(s, "| case | result | answer (preview) |\n|---|---|---|");
            for q in &self.quality {
                let _ = writeln!(
                    s,
                    "| {} | {} | {} |",
                    q.case,
                    if q.passed { "pass" } else { "FAIL" },
                    q.answer_preview
                        .replace('\n', " ")
                        .replace('|', "\\|")
                        .chars()
                        .take(80)
                        .collect::<String>()
                );
            }
        }
        let errs: Vec<&Sample> = self.samples.iter().filter(|s| !s.ok).collect();
        if !errs.is_empty() {
            let _ = writeln!(s, "\n## Errors ({})\n", errs.len());
            for e in errs.iter().take(10) {
                let _ = writeln!(
                    s,
                    "- `{}` @ conc {}: {}",
                    e.case,
                    e.concurrency,
                    e.error.as_deref().unwrap_or("")
                );
            }
        }
        let _ = writeln!(
            s,
            "\n_Method: client-side streaming timings; TTFT = request sent → first content/reasoning delta; decode = (completion tokens − 1) / (last − first delta); \
             token counts from server `usage` when reported. Effective bandwidth = weight bytes × mean decode tok/s (dense models: a lower bound, KV reads excluded; MoE models read only active experts, so it overstates). \
             Numbers are for this machine, model, and settings only._"
        );
        s
    }
}
