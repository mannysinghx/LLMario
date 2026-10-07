//! Benchmark harness. Talks to any OpenAI-compatible endpoint (llmario, Ollama, raw
//! llama-server, LM Studio…) with the same suite and settings, so results are comparable.
//! It measures from the client side; engine memory is sampled by PID when one is given.

pub mod report;
pub mod suite;

use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
pub use suite::Suite;

#[derive(Clone, Debug)]
pub struct Target {
    /// Base URL including `/v1`, e.g. `http://127.0.0.1:11500/v1`.
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    pub concurrency: Vec<usize>,
    /// Repetitions of each perf case at each concurrency level.
    pub runs: usize,
    pub warmup: usize,
    pub temperature: f64,
    /// Sent only when set. Note: a seed disables batching on some servers (mlx_lm.server).
    pub seed: Option<u64>,
    pub skip_quality: bool,
    /// `cold`: every request gets a unique prefix so prefix caches cannot hide prefill cost.
    /// `warm`: identical prompts repeat, measuring prefix-cache reuse.
    pub cache: CacheMode,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum CacheMode {
    #[default]
    Cold,
    Warm,
}

impl std::str::FromStr for CacheMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "cold" => Ok(Self::Cold),
            "warm" => Ok(Self::Warm),
            o => Err(format!("unknown cache mode '{o}' (cold|warm)")),
        }
    }
}

static NONCE: AtomicU64 = AtomicU64::new(0);

fn prompt_for(case: &suite::Case, mode: CacheMode) -> String {
    match mode {
        CacheMode::Warm => case.prompt.clone(),
        CacheMode::Cold => {
            let n = NONCE.fetch_add(1, Ordering::Relaxed);
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            format!("[benchmark request {t:x}-{n}]\n{}", case.prompt)
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sample {
    pub case: String,
    pub concurrency: usize,
    pub ok: bool,
    pub error: Option<String>,
    pub ttft_s: Option<f64>,
    pub e2e_s: f64,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: u64,
    /// `usage` when the server reported it, else `chunks` (one per content delta).
    pub token_count_source: String,
    pub prefill_tps: Option<f64>,
    pub decode_tps: Option<f64>,
    /// Speculative decoding, when the engine reports it (llama-server `timings.draft_n` /
    /// `timings.draft_n_accepted`): tokens proposed by the draft and tokens the model kept.
    #[serde(default)]
    pub draft_tokens: Option<u64>,
    #[serde(default)]
    pub draft_accepted: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QualityResult {
    pub case: String,
    pub passed: bool,
    pub expected_any: Vec<String>,
    /// First 200 characters of the final answer (suite prompts are synthetic, not user data).
    pub answer_preview: String,
}

/// Streams one chat completion and measures it.
pub async fn run_one(
    http: &reqwest::Client,
    t: &Target,
    case: &suite::Case,
    s: &Settings,
    concurrency: usize,
) -> (Sample, String) {
    let mut body = json!({
        "model": t.model,
        "messages": [{"role": "user", "content": prompt_for(case, s.cache)}],
        "max_tokens": case.max_tokens,
        "temperature": s.temperature,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    if let Some(seed) = s.seed {
        body["seed"] = json!(seed);
    }
    let start = Instant::now();
    let mut sample = Sample {
        case: case.id.clone(),
        concurrency,
        ok: false,
        error: None,
        ttft_s: None,
        e2e_s: 0.0,
        prompt_tokens: None,
        completion_tokens: 0,
        token_count_source: "chunks".into(),
        prefill_tps: None,
        decode_tps: None,
        draft_tokens: None,
        draft_accepted: None,
    };
    let mut rb = http
        .post(format!(
            "{}/chat/completions",
            t.base_url.trim_end_matches('/')
        ))
        .json(&body);
    if let Some(k) = &t.api_key {
        rb = rb.bearer_auth(k);
    }
    let resp = match rb.send().await {
        Ok(r) => r,
        Err(e) => {
            sample.error = Some(e.to_string());
            sample.e2e_s = start.elapsed().as_secs_f64();
            return (sample, String::new());
        }
    };
    if !resp.status().is_success() {
        let st = resp.status();
        sample.error = Some(format!(
            "HTTP {st}: {}",
            resp.text()
                .await
                .unwrap_or_default()
                .chars()
                .take(300)
                .collect::<String>()
        ));
        sample.e2e_s = start.elapsed().as_secs_f64();
        return (sample, String::new());
    }

    let mut text = String::new();
    let mut first: Option<Instant> = None;
    let mut last: Option<Instant> = None;
    let mut chunks = 0u64;
    let mut usage: Option<(u64, u64)> = None;
    let mut buf = Vec::new();
    let mut stream = resp.bytes_stream();
    let mut stream_err = None;
    'outer: while let Some(item) = stream.next().await {
        let bytes = match item {
            Ok(b) => b,
            Err(e) => {
                stream_err = Some(e.to_string());
                break;
            }
        };
        buf.extend_from_slice(&bytes);
        while let Some(pos) = buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = buf.drain(..=pos).collect();
            let line = String::from_utf8_lossy(&line);
            let Some(data) = line.trim().strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data == "[DONE]" {
                break 'outer;
            }
            let Ok(v) = serde_json::from_str::<Value>(data) else {
                continue;
            };
            if let Some(err) = v.get("error") {
                stream_err = Some(err.to_string());
                break 'outer;
            }
            if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
                if let (Some(p), Some(c)) = (
                    u.get("prompt_tokens").and_then(Value::as_u64),
                    u.get("completion_tokens").and_then(Value::as_u64),
                ) {
                    usage = Some((p, c));
                }
            }
            if let Some((n, a)) = draft_counts(&v) {
                sample.draft_tokens = Some(n);
                sample.draft_accepted = Some(a);
            }
            if let Some(delta) = v.pointer("/choices/0/delta") {
                let mut got = false;
                for k in ["reasoning_content", "reasoning", "content"] {
                    if let Some(s) = delta
                        .get(k)
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                    {
                        if k == "content" {
                            text.push_str(s);
                        }
                        got = true;
                    }
                }
                if got {
                    let now = Instant::now();
                    first.get_or_insert(now);
                    last = Some(now);
                    chunks += 1;
                }
            }
        }
    }
    sample.e2e_s = start.elapsed().as_secs_f64();
    if let Some(e) = stream_err {
        sample.error = Some(e);
        return (sample, text);
    }
    sample.ok = true;
    sample.ttft_s = first.map(|f| (f - start).as_secs_f64());
    let (p, c) = match usage {
        Some((p, c)) => {
            sample.token_count_source = "usage".into();
            (Some(p), c)
        }
        None => (None, chunks),
    };
    sample.prompt_tokens = p;
    sample.completion_tokens = c;
    if let (Some(p), Some(ttft)) = (p, sample.ttft_s) {
        if ttft > 0.0 {
            sample.prefill_tps = Some(p as f64 / ttft);
        }
    }
    if let (Some(f), Some(l)) = (first, last) {
        let dt = (l - f).as_secs_f64();
        if c > 1 && dt > 0.0 {
            sample.decode_tps = Some((c - 1) as f64 / dt);
        }
    }
    (sample, text)
}

/// Samples a process's memory every 100 ms and keeps the peak.
pub struct MemorySampler {
    peak: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    handle: tokio::task::JoinHandle<()>,
}

impl MemorySampler {
    pub fn start(pid: u32) -> Self {
        let peak = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (p, s) = (peak.clone(), stop.clone());
        let handle = tokio::spawn(async move {
            while !s.load(Ordering::Relaxed) {
                if let Some(b) = llmario_hardware::process_memory_bytes(pid) {
                    p.fetch_max(b, Ordering::Relaxed);
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        });
        Self { peak, stop, handle }
    }

    pub async fn finish(self) -> Option<u64> {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.handle.await;
        Some(self.peak.load(Ordering::Relaxed)).filter(|p| *p > 0)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stat {
    pub n: usize,
    pub mean: f64,
    pub p50: f64,
    pub p95: f64,
    pub min: f64,
    pub max: f64,
}

pub fn stat(values: &[f64]) -> Option<Stat> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| {
        let rank = (p * (v.len() - 1) as f64).round() as usize;
        v[rank.min(v.len() - 1)]
    };
    Some(Stat {
        n: v.len(),
        mean: v.iter().sum::<f64>() / v.len() as f64,
        p50: pct(0.50),
        p95: pct(0.95),
        min: v[0],
        max: v[v.len() - 1],
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LevelResult {
    pub concurrency: usize,
    pub requests: usize,
    pub errors: usize,
    pub wall_s: f64,
    /// Σ completion tokens / wall time.
    pub aggregate_decode_tps: f64,
    pub ttft_s: Option<Stat>,
    pub e2e_s: Option<Stat>,
    pub decode_tps: Option<Stat>,
    pub prefill_tps: Option<Stat>,
    pub peak_memory_bytes: Option<u64>,
    /// Σ accepted ÷ Σ drafted tokens, when the engine reports speculative decoding.
    #[serde(default)]
    pub draft_acceptance: Option<f64>,
    /// Weight bytes × mean decode tok/s, in GB/s (see [`add_effective_bandwidth`]).
    #[serde(default)]
    pub effective_bandwidth_gbs: Option<f64>,
    /// Per-case TTFT/prefill (prefill depends on prompt length).
    pub per_case: Vec<CaseResult>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CaseResult {
    pub case: String,
    pub prompt_tokens: Option<u64>,
    pub ttft_s: Option<Stat>,
    pub prefill_tps: Option<Stat>,
    pub decode_tps: Option<Stat>,
}

/// Run the perf cases at every concurrency level, then the quality cases once.
pub async fn run(
    t: &Target,
    suite: &Suite,
    s: &Settings,
    pid: Option<u32>,
    progress: impl Fn(&str),
) -> anyhow::Result<(Vec<LevelResult>, Vec<QualityResult>, Vec<Sample>)> {
    let http = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(1800))
        .build()?;

    // Each warm-up pass runs every perf case once, so the engine has seen every prompt length
    // (kernels compiled, buffers sized, GPU clocks up) before anything is measured.
    for i in 0..s.warmup {
        for c in &suite.perf {
            progress(&format!("warm-up {}/{}: {}", i + 1, s.warmup, c.id));
            let (smp, _) = run_one(&http, t, c, s, 1).await;
            if !smp.ok {
                anyhow::bail!("warm-up request failed: {}", smp.error.unwrap_or_default());
            }
        }
    }

    let mut levels = Vec::new();
    let mut all = Vec::new();
    for &c in &s.concurrency {
        let jobs: Vec<suite::Case> = (0..s.runs)
            .flat_map(|_| suite.perf.iter().cloned())
            .collect();
        progress(&format!("concurrency {c}: {} requests", jobs.len()));
        let sampler = pid.map(MemorySampler::start);
        let started = Instant::now();
        let samples: Vec<Sample> = futures::stream::iter(jobs.into_iter().map(|case| {
            let http = http.clone();
            async move { run_one(&http, t, &case, s, c).await.0 }
        }))
        .buffer_unordered(c.max(1))
        .collect()
        .await;
        let wall = started.elapsed().as_secs_f64();
        let peak = match sampler {
            Some(sm) => sm.finish().await,
            None => None,
        };
        levels.push(summarise(c, &samples, wall, peak, suite));
        all.extend(samples);
    }

    let mut quality = Vec::new();
    if !s.skip_quality {
        // Quality always uses the exact suite prompt: a cold-cache nonce would change the
        // prompt and make greedy answers differ between runs (observed: "42" vs "32").
        let qs = Settings {
            cache: CacheMode::Warm,
            ..s.clone()
        };
        for q in &suite.quality {
            progress(&format!("quality: {}", q.id));
            let (smp, text) = run_one(&http, t, q, &qs, 1).await;
            let answer = suite::final_answer(&text);
            let passed = smp.ok
                && q.expect_any
                    .iter()
                    .any(|e| suite::contains_word(&answer, e));
            quality.push(QualityResult {
                case: q.id.clone(),
                passed,
                expected_any: q.expect_any.clone(),
                answer_preview: if smp.ok && answer.is_empty() && smp.completion_tokens >= q.max_tokens as u64 {
                    format!(
                        "(no final answer: the {}-token budget was spent before an answer, e.g. in reasoning)",
                        q.max_tokens
                    )
                } else if smp.ok {
                    answer.chars().take(200).collect()
                } else {
                    smp.error.clone().unwrap_or_default()
                },
            });
        }
    }
    Ok((levels, quality, all))
}

fn summarise(
    c: usize,
    samples: &[Sample],
    wall: f64,
    peak: Option<u64>,
    suite: &Suite,
) -> LevelResult {
    let ok: Vec<&Sample> = samples.iter().filter(|s| s.ok).collect();
    let col = |f: &dyn Fn(&Sample) -> Option<f64>, set: &[&Sample]| -> Vec<f64> {
        set.iter().filter_map(|s| f(s)).collect()
    };
    let tokens: u64 = ok.iter().map(|s| s.completion_tokens).sum();
    let (drafted, accepted) = ok
        .iter()
        .filter_map(|s| Some((s.draft_tokens?, s.draft_accepted?)))
        .fold((0u64, 0u64), |(n, a), (dn, da)| (n + dn, a + da));
    let per_case = suite
        .perf
        .iter()
        .map(|case| {
            let set: Vec<&Sample> = ok.iter().copied().filter(|s| s.case == case.id).collect();
            CaseResult {
                case: case.id.clone(),
                prompt_tokens: set.iter().find_map(|s| s.prompt_tokens),
                ttft_s: stat(&col(&|s| s.ttft_s, &set)),
                prefill_tps: stat(&col(&|s| s.prefill_tps, &set)),
                decode_tps: stat(&col(&|s| s.decode_tps, &set)),
            }
        })
        .collect();
    LevelResult {
        concurrency: c,
        requests: samples.len(),
        errors: samples.len() - ok.len(),
        wall_s: wall,
        aggregate_decode_tps: if wall > 0.0 {
            tokens as f64 / wall
        } else {
            0.0
        },
        ttft_s: stat(&col(&|s| s.ttft_s, &ok)),
        e2e_s: stat(&col(&|s| Some(s.e2e_s), &ok)),
        decode_tps: stat(&col(&|s| s.decode_tps, &ok)),
        prefill_tps: stat(&col(&|s| s.prefill_tps, &ok)),
        peak_memory_bytes: peak,
        draft_acceptance: (drafted > 0).then(|| accepted as f64 / drafted as f64),
        effective_bandwidth_gbs: None,
        per_case,
    }
}

/// `(draft_n, draft_n_accepted)` from a streamed chunk's llama-server `timings`, if present.
fn draft_counts(v: &Value) -> Option<(u64, u64)> {
    let t = v.get("timings")?;
    Some((
        t.get("draft_n")?.as_u64()?,
        t.get("draft_n_accepted")?.as_u64()?,
    ))
}

/// Set each level's effective memory bandwidth: weight bytes × mean decode tok/s per request.
/// Every decode step reads the weights once and gives each running request one token, so
/// this holds at any concurrency. Dense models: a lower bound (KV-cache reads are left out).
/// MoE models read only their active experts, so for them it overstates the bandwidth.
pub fn add_effective_bandwidth(levels: &mut [LevelResult], weight_bytes: u64) {
    for l in levels {
        l.effective_bandwidth_gbs = l
            .decode_tps
            .as_ref()
            .map(|d| weight_bytes as f64 * d.mean / 1e9);
    }
}

/// Busy enough that timings would be noise: a one-minute load average above half the CPU
/// cores (at least 2). Measured: at load 26 on 16 cores the same run gave 92–178 tok/s
/// against a 217 tok/s quiet baseline.
pub fn is_busy(load_1m: f64, cores: usize) -> bool {
    load_1m > (cores as f64 * 0.5).max(2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_prompts_are_unique_warm_prompts_exact() {
        let c = suite::Case {
            id: "x".into(),
            max_tokens: 1,
            prompt: "P".into(),
            expect_any: vec![],
        };
        assert_eq!(prompt_for(&c, CacheMode::Warm), "P");
        let (a, b) = (
            prompt_for(&c, CacheMode::Cold),
            prompt_for(&c, CacheMode::Cold),
        );
        assert_ne!(a, b);
        assert!(a.ends_with("\nP"));
    }

    fn sample(case: &str, decode: f64, draft: Option<(u64, u64)>) -> Sample {
        Sample {
            case: case.into(),
            concurrency: 1,
            ok: true,
            error: None,
            ttft_s: Some(0.1),
            e2e_s: 1.0,
            prompt_tokens: Some(10),
            completion_tokens: 100,
            token_count_source: "usage".into(),
            prefill_tps: Some(100.0),
            decode_tps: Some(decode),
            draft_tokens: draft.map(|d| d.0),
            draft_accepted: draft.map(|d| d.1),
        }
    }

    #[test]
    fn reads_llama_server_draft_counts() {
        let v =
            serde_json::json!({"choices": [], "timings": {"draft_n": 48, "draft_n_accepted": 46}});
        assert_eq!(draft_counts(&v), Some((48, 46)));
        assert_eq!(
            draft_counts(&serde_json::json!({"timings": {"predicted_n": 3}})),
            None
        );
        assert_eq!(draft_counts(&serde_json::json!({"choices": []})), None);
    }

    #[test]
    fn acceptance_and_bandwidth() {
        let suite = Suite::builtin();
        let with = [
            sample("short", 100.0, Some((40, 30))),
            sample("short", 100.0, Some((10, 10))),
        ];
        let mut levels = vec![summarise(1, &with, 2.0, None, &suite)];
        assert_eq!(levels[0].draft_acceptance, Some(0.8));
        assert_eq!(levels[0].effective_bandwidth_gbs, None);
        add_effective_bandwidth(&mut levels, 4_610_000_000);
        let bw = levels[0].effective_bandwidth_gbs.unwrap();
        assert!((bw - 461.0).abs() < 1e-6, "{bw}");
        let without = [sample("short", 50.0, None)];
        assert_eq!(
            summarise(1, &without, 1.0, None, &suite).draft_acceptance,
            None
        );
    }

    #[test]
    fn busy_threshold() {
        assert!(
            is_busy(26.0, 16),
            "the noisy run in docs/PHASES_16GB_AND_SPEED.md"
        );
        assert!(!is_busy(3.5, 16));
        assert!(!is_busy(1.9, 2), "small machines: at least 2");
        assert!(is_busy(2.1, 2));
    }

    #[test]
    fn percentiles() {
        let s = stat(&[5.0, 1.0, 3.0, 2.0, 4.0]).unwrap();
        assert_eq!((s.min, s.p50, s.max, s.n), (1.0, 3.0, 5.0, 5));
        assert!((s.mean - 3.0).abs() < 1e-9);
        assert_eq!(stat(&[7.0]).unwrap().p95, 7.0);
        assert!(stat(&[]).is_none());
    }
}
