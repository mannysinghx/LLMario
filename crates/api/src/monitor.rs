//! Runtime monitor: one JSON snapshot of everything technical about the running models, served
//! on `GET /llmario/monitor` and drawn by `llmario top` and the desktop Monitor panel.
//!
//! The snapshot holds what the gateway sees (requests in flight and their phase, live decode
//! speed, recent requests), what each engine reports (`/engine/stats` from the built-in engine,
//! process memory, CPU and storage reads for every engine), and what the machine reports (CPU,
//! memory, swap, GPU). It never holds prompt or reply text.
//!
//! Switching the monitor off costs nothing: the machine and the engines are only read while a
//! client asks for snapshots. What is always kept is small and fixed: a ring of the last
//! [`RECENT_MAX`] finished requests (numbers only) and one entry per request in flight, touched
//! at its start, at its first token and at its end. Per-token work is one relaxed atomic add.

use crate::metrics::RequestRecord;
use crate::AppState;
use llmario_core::BackendKind;
use llmario_hardware::live::{LiveSampler, SystemLive};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Finished requests remembered for the "Recent requests" table.
pub const RECENT_MAX: usize = 64;
/// Longest model name kept in the ring (registered ids are charset-checked but not length-capped).
pub const MODEL_NAME_MAX: usize = 256;
/// A failed request keeps the state at `error` this long when nothing else is happening.
const ERROR_STATE_FOR: Duration = Duration::from_secs(15);
/// How long the monitor waits for an engine's own statistics.
const ENGINE_STATS_TIMEOUT: Duration = Duration::from_millis(800);
/// Snapshots closer together than this share one reading of the machine, so a client polling
/// fast cannot make the gateway run `ioreg` / `nvidia-smi` in a tight loop.
const MIN_SAMPLE_GAP: Duration = Duration::from_millis(250);

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// One finished request, as numbers.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct RecentRequest {
    pub at_unix_ms: u64,
    pub model: String,
    /// `ok`, `cancelled`, or an error code.
    pub outcome: String,
    pub prompt_tokens: u64,
    /// Prompt tokens the engine reused from its cache, when it says.
    pub cached_tokens: Option<u64>,
    pub completion_tokens: u64,
    pub ttft_ms: Option<f64>,
    pub duration_ms: f64,
    /// Generated tokens per second after the first token (or as the engine timed it).
    pub decode_tok_s: Option<f64>,
}

/// One request in flight.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct LiveView {
    pub model: String,
    /// Streaming requests: `reading` (the prompt is being processed) until the first token,
    /// then `generating`. Non-streaming requests are `running` throughout: the gateway sees no
    /// tokens until the reply is complete, so it cannot tell the two apart.
    pub phase: String,
    /// `false` when read from a gateway that predates the field.
    #[serde(default)]
    pub streaming: bool,
    pub elapsed_s: f64,
    pub ttft_ms: Option<f64>,
    /// Estimated from the prompt's length (characters / 4) until the engine reports usage.
    pub prompt_tokens_estimate: u64,
    /// Streamed pieces so far (about one token each).
    pub generated: u64,
    pub tok_s: Option<f64>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct Totals {
    pub requests: u64,
    pub ok: u64,
    pub cancelled: u64,
    pub failed: u64,
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub completion_tokens: u64,
}

/// A loaded model and the process running it.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct EngineView {
    pub model: String,
    pub backend: String,
    pub pid: u32,
    pub family: Option<String>,
    pub format: Option<String>,
    pub architecture: Option<String>,
    pub quantization: Option<String>,
    pub file_bytes: Option<u64>,
    pub layers: Option<u32>,
    pub context_max: Option<u32>,
    /// Context tokens each concurrent request gets, and how many run at once.
    pub context_per_slot: u32,
    pub parallel: u32,
    pub batch: u32,
    pub ubatch: u32,
    pub profile: String,
    pub active_requests: usize,
    /// Tokens of the window in use: the request in flight, else the last one that finished.
    pub context_used: Option<u64>,
    pub load_seconds: f64,
    pub idle_seconds: f64,
    pub estimated_bytes: u64,
    pub memory_bytes: Option<u64>,
    pub cpu_percent: Option<f32>,
    pub disk_read_bps: Option<u64>,
    pub disk_write_bps: Option<u64>,
    pub disk_read_total: Option<u64>,
    /// The built-in engine's own `/engine/stats` body (threads, kernels, device, prefill and
    /// decode speed, cache reuse, KV disk tier, memory pressure, paging). `None` for llama.cpp
    /// and MLX, which do not serve it.
    pub engine_stats: Option<Value>,
}

/// What does not change while the gateway runs.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct MachineView {
    pub cpu: String,
    pub os: String,
    pub arch: String,
    pub performance_cores: Option<usize>,
    pub efficiency_cores: Option<usize>,
    pub unified_memory: bool,
    pub memory_bandwidth_gbs: Option<f64>,
    pub gpu: Option<String>,
    pub gpu_api: Option<String>,
    pub profile: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Snapshot {
    pub version: String,
    pub at_unix_ms: u64,
    pub uptime_s: f64,
    /// `idle`, `reading`, `generating`, `running` (only non-streaming requests in flight, whose
    /// phase the gateway cannot see), `queued` (waiting for a slot or a model to load), or
    /// `error` (the last request failed in the past 15 seconds).
    pub state: String,
    /// Requests waiting for a slot or for their model to load.
    pub waiting: usize,
    /// Tokens per second across every request generating now.
    pub speed_tok_s: f64,
    pub live: Vec<LiveView>,
    pub engines: Vec<EngineView>,
    /// Newest first.
    pub recent: Vec<RecentRequest>,
    pub totals: Totals,
    pub machine: MachineView,
    /// The gateway process itself (in the desktop app, the app).
    pub gateway: Option<llmario_hardware::live::ProcessLive>,
    pub system: SystemLive,
}

struct LiveRequest {
    id: u64,
    model: String,
    started: Instant,
    approx_prompt: u64,
    streaming: bool,
    /// Microseconds from `started` to the first token, plus one (0: none yet).
    first_token_us: AtomicU64,
    chunks: AtomicU64,
}

#[derive(Default)]
pub struct Monitor {
    started: Mutex<Option<Instant>>,
    next_id: AtomicU64,
    live: Mutex<Vec<Arc<LiveRequest>>>,
    waiting: AtomicUsize,
    recent: Mutex<VecDeque<RecentRequest>>,
    totals: Mutex<Totals>,
    /// Shared with the blocking sampling task, which needs an owned handle.
    sampler: Arc<Mutex<Option<Sampling>>>,
    /// The latest machine reading, served while another sample is still being taken.
    last_system: Arc<Mutex<Option<SystemLive>>>,
}

/// The machine reader and its latest reading (when, for which processes, what).
struct Sampling {
    sampler: LiveSampler,
    last: Option<(Instant, Vec<u32>, SystemLive)>,
}

impl Sampling {
    fn sample(&mut self, pids: Vec<u32>) -> SystemLive {
        if let Some((at, p, sys)) = &self.last {
            if at.elapsed() < MIN_SAMPLE_GAP && *p == pids {
                return sys.clone();
            }
        }
        let sys = self.sampler.sample(&pids);
        self.last = Some((Instant::now(), pids, sys.clone()));
        sys
    }
}

/// Removes its request from the in-flight list when dropped.
pub struct LiveGuard {
    monitor: Arc<crate::Metrics>,
    req: Arc<LiveRequest>,
}

impl LiveGuard {
    /// One streamed piece of output (about a token).
    pub fn token(&self) {
        let r = &self.req;
        if r.first_token_us.load(Ordering::Relaxed) == 0 {
            let us = r.started.elapsed().as_micros() as u64 + 1;
            let _ = r
                .first_token_us
                .compare_exchange(0, us, Ordering::Relaxed, Ordering::Relaxed);
        }
        r.chunks.fetch_add(1, Ordering::Relaxed);
    }
}

impl Drop for LiveGuard {
    fn drop(&mut self) {
        let id = self.req.id;
        self.monitor
            .monitor
            .live
            .lock()
            .unwrap()
            .retain(|r| r.id != id);
    }
}

/// Counts a request waiting in admission while it lives.
pub struct WaitGuard<'a>(&'a Monitor);

impl Drop for WaitGuard<'_> {
    fn drop(&mut self) {
        self.0.waiting.fetch_sub(1, Ordering::Relaxed);
    }
}

impl Monitor {
    fn uptime(&self) -> Duration {
        self.started
            .lock()
            .unwrap()
            .get_or_insert_with(Instant::now)
            .elapsed()
    }

    /// Marks the gateway's start (uptime counts from the first call).
    pub fn touch(&self) {
        let _ = self.uptime();
    }

    pub fn waiting(&self) -> WaitGuard<'_> {
        self.waiting.fetch_add(1, Ordering::Relaxed);
        WaitGuard(self)
    }

    /// Records a finished request (called by [`crate::Metrics::record`]).
    pub(crate) fn finished(&self, r: &RequestRecord<'_>) {
        let decode = r.engine_decode_tok_s.or_else(|| {
            let ttft = r.ttft_s?;
            let gen = r.duration_s - ttft;
            // Tokens after the first, over the time after the first (as `run` and `bench` do).
            (r.completion_tokens > 1 && gen > 0.0).then(|| (r.completion_tokens - 1) as f64 / gen)
        });
        {
            let mut t = self.totals.lock().unwrap();
            t.requests += 1;
            match r.outcome {
                "ok" => t.ok += 1,
                "cancelled" => t.cancelled += 1,
                _ => t.failed += 1,
            }
            t.prompt_tokens += r.prompt_tokens;
            t.cached_tokens += r.cached_tokens.unwrap_or(0);
            t.completion_tokens += r.completion_tokens;
        }
        let mut q = self.recent.lock().unwrap();
        if q.len() == RECENT_MAX {
            q.pop_back();
        }
        q.push_front(RecentRequest {
            at_unix_ms: unix_ms(),
            model: r.model.to_string(),
            outcome: r.outcome.to_string(),
            prompt_tokens: r.prompt_tokens,
            cached_tokens: r.cached_tokens,
            completion_tokens: r.completion_tokens,
            ttft_ms: r.ttft_s.map(|t| t * 1000.0),
            duration_ms: r.duration_s * 1000.0,
            decode_tok_s: decode,
        });
    }

    fn live_views(&self) -> Vec<LiveView> {
        let now = Instant::now();
        self.live
            .lock()
            .unwrap()
            .iter()
            .map(|r| {
                let first = r.first_token_us.load(Ordering::Relaxed);
                let chunks = r.chunks.load(Ordering::Relaxed);
                let elapsed = now.duration_since(r.started).as_secs_f64();
                let ttft = (first > 0).then(|| (first - 1) as f64 / 1e6);
                let tok_s = ttft.and_then(|t| {
                    let gen = elapsed - t;
                    (gen > 0.05 && chunks > 1).then(|| (chunks - 1) as f64 / gen)
                });
                let phase = match (r.streaming, first > 0) {
                    (false, _) => "running",
                    (true, true) => "generating",
                    (true, false) => "reading",
                };
                LiveView {
                    model: r.model.clone(),
                    phase: phase.into(),
                    streaming: r.streaming,
                    elapsed_s: elapsed,
                    ttft_ms: ttft.map(|t| t * 1000.0),
                    prompt_tokens_estimate: r.approx_prompt,
                    generated: chunks,
                    tok_s,
                }
            })
            .collect()
    }
}

impl crate::Metrics {
    /// Registers a request in flight; it leaves the list when the guard drops.
    pub fn begin(self: &Arc<Self>, model: &str, approx_prompt: u64, streaming: bool) -> LiveGuard {
        let req = Arc::new(LiveRequest {
            id: self.monitor.next_id.fetch_add(1, Ordering::Relaxed),
            model: model.to_string(),
            started: Instant::now(),
            approx_prompt,
            streaming,
            first_token_us: AtomicU64::new(0),
            chunks: AtomicU64::new(0),
        });
        self.monitor.live.lock().unwrap().push(req.clone());
        LiveGuard {
            monitor: self.clone(),
            req,
        }
    }
}

/// The overall state shown in the state bar.
pub fn overall_state(live: &[LiveView], waiting: usize, recent: &[RecentRequest]) -> &'static str {
    if live.iter().any(|l| l.phase == "generating") {
        "generating"
    } else if live.iter().any(|l| l.phase == "reading") {
        "reading"
    } else if !live.is_empty() {
        "running"
    } else if waiting > 0 {
        "queued"
    } else if recent.first().is_some_and(|r| {
        r.outcome != "ok"
            && r.outcome != "cancelled"
            && unix_ms().saturating_sub(r.at_unix_ms) < ERROR_STATE_FOR.as_millis() as u64
    }) {
        "error"
    } else {
        "idle"
    }
}

async fn engine_stats(http: &reqwest::Client, port: u16) -> Option<Value> {
    let v: Value = http
        .get(format!("http://127.0.0.1:{port}/engine/stats"))
        .timeout(ENGINE_STATS_TIMEOUT)
        .send()
        .await
        .ok()?
        .error_for_status()
        .ok()?
        .json()
        .await
        .ok()?;
    v.is_object().then_some(v)
}

/// Builds one snapshot: reads the machine, every engine process, and the built-in engine's own
/// statistics. Takes a few milliseconds; nothing is read between calls.
pub async fn snapshot(st: &AppState) -> Snapshot {
    let sup = &st.sup;
    let mon = &st.metrics.monitor;
    let engines = sup.loaded().await;

    // The machine, the engines' processes and the gateway's own, read off the executor.
    let mut pids: Vec<u32> = engines.iter().map(|e| e.pid).collect();
    let me = std::process::id();
    pids.push(me);
    let sampler = mon.sampler.clone();
    let last_system = mon.last_system.clone();
    let sample = tokio::task::spawn_blocking(move || {
        // A sample already running (a slow GPU tool) is not waited for: callers get the last
        // reading instead of queueing blocking threads behind it.
        let Ok(mut g) = sampler.try_lock() else {
            return last_system.lock().unwrap().clone().unwrap_or_default();
        };
        let sys = g
            .get_or_insert_with(|| Sampling {
                sampler: LiveSampler::new(),
                last: None,
            })
            .sample(pids);
        *last_system.lock().unwrap() = Some(sys.clone());
        sys
    });

    let stats = futures::future::join_all(engines.iter().map(|e| async move {
        if e.backend == BackendKind::Native {
            engine_stats(sup.http(), e.port).await
        } else {
            None
        }
    }))
    .await;
    let mut system = sample.await.unwrap_or_default();

    let live = mon.live_views();
    let recent: Vec<RecentRequest> = mon.recent.lock().unwrap().iter().cloned().collect();
    let reg = sup.registry();
    let gateway_idx = system.processes.iter().position(|p| p.pid == me);
    let gateway = gateway_idx.map(|i| system.processes.remove(i));

    let views = engines
        .iter()
        .zip(stats)
        .map(|(e, engine_stats)| {
            let entry = reg.get(&e.model);
            let proc = system.processes.iter().find(|p| p.pid == e.pid);
            let shape = entry.and_then(|m| m.shape.as_ref());
            let context_used = live
                .iter()
                .filter(|l| l.model == e.model)
                .map(|l| l.prompt_tokens_estimate + l.generated)
                .max()
                .or_else(|| {
                    recent
                        .iter()
                        .find(|r| r.model == e.model && r.outcome == "ok")
                        .map(|r| r.prompt_tokens + r.completion_tokens)
                });
            EngineView {
                model: e.model.clone(),
                backend: e.backend.to_string(),
                pid: e.pid,
                family: entry.and_then(|m| m.family.clone()),
                format: entry.map(|m| m.format.to_string()),
                architecture: entry.and_then(|m| m.architecture.clone()),
                quantization: entry.and_then(|m| m.quantization.clone()),
                file_bytes: entry.map(|m| m.size_bytes),
                layers: shape.map(|s| s.n_layers),
                context_max: shape.and_then(|s| s.context_max),
                context_per_slot: e.profile.ctx_per_slot,
                parallel: e.profile.parallel,
                batch: e.profile.batch,
                ubatch: e.profile.ubatch,
                profile: e.profile.kind.to_string(),
                active_requests: e.active_requests,
                context_used,
                load_seconds: e.ready_seconds,
                idle_seconds: e.idle_seconds,
                estimated_bytes: e.estimated_bytes,
                memory_bytes: e.resident_bytes.or(proc.and_then(|p| p.memory_bytes)),
                cpu_percent: proc.and_then(|p| p.cpu_percent),
                disk_read_bps: proc.and_then(|p| p.disk_read_bps),
                disk_write_bps: proc.and_then(|p| p.disk_write_bps),
                disk_read_total: proc.map(|p| p.disk_read_total),
                engine_stats,
            }
        })
        .collect();

    let hw = &sup.hw;
    let gpu = hw.primary_gpu();
    Snapshot {
        version: llmario_core::VERSION.to_string(),
        at_unix_ms: unix_ms(),
        uptime_s: mon.uptime().as_secs_f64(),
        state: overall_state(&live, mon.waiting.load(Ordering::Relaxed), &recent).into(),
        waiting: mon.waiting.load(Ordering::Relaxed),
        speed_tok_s: live.iter().filter_map(|l| l.tok_s).fold(0.0, |a, b| a + b),
        live,
        engines: views,
        totals: mon.totals.lock().unwrap().clone(),
        recent,
        machine: MachineView {
            cpu: hw.cpu_brand.clone(),
            os: hw.os_version.clone(),
            arch: hw.arch.clone(),
            performance_cores: hw.performance_cores,
            efficiency_cores: hw.efficiency_cores,
            unified_memory: hw.unified_memory,
            memory_bandwidth_gbs: hw.memory_bandwidth_gbs,
            gpu: gpu.map(|g| g.name.clone()),
            gpu_api: gpu.map(|g| format!("{:?}", g.api).to_lowercase()),
            profile: sup.profile.to_string(),
        },
        gateway,
        system,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Metrics;

    fn rec<'a>(model: &'a str, outcome: &'a str, ttft: Option<f64>) -> RequestRecord<'a> {
        RequestRecord {
            model,
            outcome,
            ttft_s: ttft,
            duration_s: 2.0,
            prompt_tokens: 100,
            cached_tokens: Some(40),
            completion_tokens: 51,
            engine_decode_tok_s: None,
        }
    }

    #[test]
    fn fast_polls_share_one_machine_reading() {
        let mut s = Sampling {
            sampler: LiveSampler::new(),
            last: None,
        };
        let me = std::process::id();
        let a = s.sample(vec![me]);
        let at = s.last.as_ref().unwrap().0;
        let b = s.sample(vec![me]);
        assert_eq!(a, b);
        assert_eq!(s.last.as_ref().unwrap().0, at, "reused, not re-read");
        s.sample(vec![me, 1]);
        assert!(
            s.last.as_ref().unwrap().0 > at,
            "a new process set is read afresh"
        );
        std::thread::sleep(MIN_SAMPLE_GAP);
        let t = s.last.as_ref().unwrap().0;
        s.sample(vec![me, 1]);
        assert!(
            s.last.as_ref().unwrap().0 > t,
            "re-read once the gap has passed"
        );
    }

    #[test]
    fn finished_requests_land_newest_first_in_a_bounded_ring() {
        let m = Metrics::default();
        for i in 0..(RECENT_MAX + 10) {
            m.record(rec(if i % 2 == 0 { "a" } else { "b" }, "ok", Some(0.3)));
        }
        m.record(rec("c", "engine_crashed", None));
        let q = m.monitor.recent.lock().unwrap();
        assert_eq!(q.len(), RECENT_MAX);
        assert_eq!(q[0].model, "c");
        assert_eq!(q[0].decode_tok_s, None, "no first token, no decode speed");
        assert!((q[1].decode_tok_s.unwrap() - 50.0 / 1.7).abs() < 1e-9);
        assert_eq!(q[1].ttft_ms, Some(300.0));
        let t = m.monitor.totals.lock().unwrap();
        assert_eq!(
            (t.requests, t.ok, t.failed),
            (RECENT_MAX as u64 + 11, RECENT_MAX as u64 + 10, 1)
        );
        assert_eq!(t.cached_tokens, 40 * (RECENT_MAX as u64 + 11));
    }

    #[test]
    fn live_requests_move_from_reading_to_generating_and_leave_on_drop() {
        let m = Arc::new(Metrics::default());
        let g = m.begin("a", 1200, true);
        let v = m.monitor.live_views();
        assert_eq!(
            (v.len(), v[0].phase.as_str(), v[0].generated),
            (1, "reading", 0)
        );
        assert_eq!(overall_state(&v, 0, &[]), "reading");
        std::thread::sleep(Duration::from_millis(20));
        g.token();
        std::thread::sleep(Duration::from_millis(80));
        g.token();
        let v = m.monitor.live_views();
        assert_eq!((v[0].phase.as_str(), v[0].generated), ("generating", 2));
        assert!(v[0].ttft_ms.unwrap() >= 20.0);
        assert!(v[0].tok_s.unwrap() > 0.0);
        assert_eq!(overall_state(&v, 3, &[]), "generating");
        drop(g);
        assert!(m.monitor.live_views().is_empty());
    }

    #[test]
    fn non_streaming_requests_are_running_not_reading() {
        let m = Arc::new(Metrics::default());
        let g = m.begin("a", 50, false);
        let v = m.monitor.live_views();
        assert_eq!((v[0].phase.as_str(), v[0].streaming), ("running", false));
        assert_eq!(overall_state(&v, 0, &[]), "running");
        let s = m.begin("a", 50, true);
        assert_eq!(overall_state(&m.monitor.live_views(), 0, &[]), "reading");
        drop((g, s));
    }

    #[test]
    fn the_ring_keeps_long_model_names_past_the_prometheus_caps() {
        let m = Metrics::default();
        let long = "m".repeat(150);
        m.record(rec(&long, "ok", None));
        // Past the Prometheus series cap, names are counted as "other" there but not here.
        for i in 0..300 {
            m.record(rec(&format!("x{i}"), "invalid_request", None));
        }
        m.record(rec("late-model", "ok", None));
        let q = m.monitor.recent.lock().unwrap();
        assert_eq!(q[0].model, "late-model");
        drop(q);
        let m2 = Metrics::default();
        m2.record(rec(&long, "ok", None));
        assert_eq!(m2.monitor.recent.lock().unwrap()[0].model, long);
        let huge = "z\n".repeat(400);
        m2.record(rec(&huge, "invalid_request", None));
        let name = m2.monitor.recent.lock().unwrap()[0].model.clone();
        assert_eq!(name.chars().count(), MODEL_NAME_MAX);
        assert!(!name.contains('\n'));
    }

    #[test]
    fn state_is_queued_while_waiting_then_error_after_a_recent_failure() {
        let m = Metrics::default();
        {
            let _w = m.monitor.waiting();
            assert_eq!(m.monitor.waiting.load(Ordering::Relaxed), 1);
            assert_eq!(overall_state(&[], 1, &[]), "queued");
        }
        assert_eq!(m.monitor.waiting.load(Ordering::Relaxed), 0);
        m.record(rec("a", "context_overflow", None));
        let recent: Vec<_> = m.monitor.recent.lock().unwrap().iter().cloned().collect();
        assert_eq!(overall_state(&[], 0, &recent), "error");
        let mut old = recent.clone();
        old[0].at_unix_ms -= 60_000;
        assert_eq!(overall_state(&[], 0, &old), "idle");
        m.record(rec("a", "cancelled", None));
        let recent: Vec<_> = m.monitor.recent.lock().unwrap().iter().cloned().collect();
        assert_eq!(overall_state(&[], 0, &recent), "idle");
    }
}
