//! Live machine readings for the runtime monitor (`llmario top`, the desktop Monitor panel).
//!
//! Nothing here runs in the background: a [`LiveSampler`] reads the machine only when someone
//! asks for a sample, so the monitor costs nothing while it is switched off. Rates (CPU %, disk
//! bytes per second) are measured between two calls; the first call after creating a sampler
//! reports them as `None` rather than guessing.
//!
//! GPU readings come from tools that ship with the system or driver and need no privileges:
//! `ioreg` on Apple Silicon (utilisation and GPU-allocated memory; temperature and power need
//! `powermetrics` with root, so they stay `None`), `nvidia-smi` for NVIDIA. Anything a tool
//! does not report is `None`, never a made-up zero.

use llmario_core::os::background_command as command;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::process::{Output, Stdio};
use std::time::{Duration, Instant};
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

/// One process the monitor follows (an engine, or the gateway itself).
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct ProcessLive {
    pub pid: u32,
    /// Share of one core, summed over cores (200 = two cores busy). `None` on the first sample.
    pub cpu_percent: Option<f32>,
    /// Same measure as `llmario_engine_memory_bytes` (see [`crate::process_memory_bytes`]).
    pub memory_bytes: Option<u64>,
    /// Bytes per second read from and written to storage since the previous sample.
    pub disk_read_bps: Option<u64>,
    pub disk_write_bps: Option<u64>,
    /// Totals since the process started.
    pub disk_read_total: u64,
    pub disk_write_total: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct GpuLive {
    pub name: Option<String>,
    /// `ioreg` (Apple) or `nvidia-smi`.
    pub source: String,
    pub utilization_percent: Option<f32>,
    /// Discrete VRAM in use, or on Apple Silicon the system memory the GPU has in use.
    pub memory_used_bytes: Option<u64>,
    pub memory_total_bytes: Option<u64>,
    pub temperature_c: Option<f32>,
    pub power_w: Option<f32>,
    pub power_limit_w: Option<f32>,
    /// PCIe link, e.g. "Gen4 x16" (discrete GPUs only).
    pub pcie: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct SystemLive {
    /// Whole-machine CPU use, 0–100. `None` on the first sample.
    pub cpu_percent: Option<f32>,
    pub logical_cores: usize,
    pub physical_cores: usize,
    pub load_average_1m: Option<f64>,
    pub memory_total_bytes: u64,
    pub memory_used_bytes: u64,
    pub memory_available_bytes: u64,
    pub swap_total_bytes: u64,
    pub swap_used_bytes: u64,
    pub gpus: Vec<GpuLive>,
    pub processes: Vec<ProcessLive>,
}

/// A sample further than this from the previous one starts afresh: rates averaged over a long
/// pause (the panel closed for an hour) would look like current readings.
const STALE_AFTER: Duration = Duration::from_secs(5);
/// Longest a GPU tool may run before it is killed and GPU readings pause.
const TOOL_DEADLINE: Duration = Duration::from_secs(2);
/// After a GPU tool hangs, how long before it is tried again.
const TOOL_BACKOFF: Duration = Duration::from_secs(60);

/// Per-process counters at the previous sample.
#[derive(Clone, Copy)]
struct ProcPrev {
    at: Instant,
    /// CPU time used since the process started, in milliseconds.
    cpu_ms: u64,
    read: u64,
    written: u64,
}

/// Reads the machine on demand and remembers the previous reading for rates.
pub struct LiveSampler {
    sys: System,
    prev: HashMap<u32, ProcPrev>,
    /// Processes sysinfo still holds (an open `/proc` file on Linux, a handle on Windows) that
    /// are no longer asked for. They are refreshed until they exit so sysinfo can drop them.
    lingering: HashSet<u32>,
    last_sample: Option<Instant>,
    gpu: GpuProbe,
    gpu_retry_after: Option<Instant>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GpuProbe {
    Unknown,
    Apple,
    Nvidia,
    None,
}

impl Default for LiveSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl LiveSampler {
    pub fn new() -> Self {
        Self {
            sys: System::new(),
            prev: HashMap::new(),
            lingering: HashSet::new(),
            last_sample: None,
            gpu: GpuProbe::Unknown,
            gpu_retry_after: None,
        }
    }

    /// Takes one reading of the machine and of `pids`. Blocking (it may run `ioreg` or
    /// `nvidia-smi`, a few milliseconds each, killed after two seconds): call it off the async
    /// executor.
    pub fn sample(&mut self, pids: &[u32]) -> SystemLive {
        let now = Instant::now();
        // The first sample, or the first after a long pause, has no recent baseline.
        let fresh = self
            .last_sample
            .is_some_and(|t| now.duration_since(t) <= STALE_AFTER);
        self.last_sample = Some(now);
        if !fresh {
            self.prev.clear();
        }

        self.sys.refresh_cpu_usage();
        self.sys.refresh_memory();
        let cpu_percent = fresh
            .then(|| self.sys.global_cpu_usage())
            .filter(|v| v.is_finite());

        let wanted: HashSet<u32> = pids.iter().copied().collect();
        let refresh: Vec<Pid> = wanted
            .iter()
            .chain(self.lingering.iter())
            .map(|p| Pid::from_u32(*p))
            .collect();
        if !refresh.is_empty() {
            // CPU use comes from CPU-time deltas below, not sysinfo's `cpu_usage()`: on macOS
            // that keeps its last non-zero value while a process sleeps, and on Windows its
            // first value is a lifetime average.
            self.sys.refresh_processes_specifics(
                ProcessesToUpdate::Some(&refresh),
                true,
                ProcessRefreshKind::nothing()
                    .with_cpu()
                    .with_disk_usage()
                    .with_memory(),
            );
        }
        self.lingering = self
            .lingering
            .iter()
            .chain(self.prev.keys())
            .copied()
            .filter(|p| !wanted.contains(p) && self.sys.process(Pid::from_u32(*p)).is_some())
            .collect();

        let mut processes = Vec::with_capacity(pids.len());
        let mut seen = HashMap::new();
        for &pid in pids {
            let Some(p) = self.sys.process(Pid::from_u32(pid)) else {
                continue;
            };
            let d = p.disk_usage();
            let cur = ProcPrev {
                at: now,
                cpu_ms: p.accumulated_cpu_time(),
                read: d.total_read_bytes,
                written: d.total_written_bytes,
            };
            let prev = self.prev.get(&pid).copied();
            let secs = prev.map(|q| now.duration_since(q.at).as_secs_f64());
            let per_sec = |before: u64, after: u64| {
                secs.filter(|s| *s > 0.05)
                    .map(|s| after.saturating_sub(before) as f64 / s)
            };
            processes.push(ProcessLive {
                pid,
                cpu_percent: prev
                    .and_then(|q| per_sec(q.cpu_ms, cur.cpu_ms))
                    .map(|ms_per_s| (ms_per_s / 10.0) as f32),
                memory_bytes: crate::process_memory_bytes(pid).or(Some(p.memory())),
                disk_read_bps: prev
                    .and_then(|q| per_sec(q.read, cur.read))
                    .map(|v| v as u64),
                disk_write_bps: prev
                    .and_then(|q| per_sec(q.written, cur.written))
                    .map(|v| v as u64),
                disk_read_total: cur.read,
                disk_write_total: cur.written,
            });
            seen.insert(pid, cur);
        }
        // Forget processes that are gone, so a reused pid does not inherit old totals.
        self.prev = seen;

        SystemLive {
            cpu_percent,
            logical_cores: self.sys.cpus().len(),
            physical_cores: System::physical_core_count().unwrap_or(self.sys.cpus().len()),
            load_average_1m: crate::HardwareReport::load_average_1m(),
            memory_total_bytes: self.sys.total_memory(),
            memory_used_bytes: self.sys.used_memory(),
            memory_available_bytes: self.sys.available_memory(),
            swap_total_bytes: self.sys.total_swap(),
            swap_used_bytes: self.sys.used_swap(),
            gpus: self.gpus(),
            processes,
        }
    }

    fn gpus(&mut self) -> Vec<GpuLive> {
        if self.gpu_retry_after.is_some_and(|t| Instant::now() < t) {
            return Vec::new();
        }
        self.gpu_retry_after = None;
        match self.read_gpus() {
            Ok(g) => g,
            Err(Hung) => {
                // A wedged driver can make the tool hang for ever; do not queue behind it.
                self.gpu_retry_after = Some(Instant::now() + TOOL_BACKOFF);
                Vec::new()
            }
        }
    }

    fn read_gpus(&mut self) -> Result<Vec<GpuLive>, Hung> {
        if matches!(self.gpu, GpuProbe::Unknown | GpuProbe::Apple)
            && cfg!(all(target_os = "macos", target_arch = "aarch64"))
        {
            if let Some(g) = apple_gpu()? {
                self.gpu = GpuProbe::Apple;
                return Ok(vec![g]);
            }
        }
        if matches!(self.gpu, GpuProbe::Unknown | GpuProbe::Nvidia) {
            let g = nvidia_gpus()?;
            if !g.is_empty() {
                self.gpu = GpuProbe::Nvidia;
                return Ok(g);
            }
        }
        // Probe once; a machine without these tools does not get a process spawned per sample.
        if self.gpu == GpuProbe::Unknown {
            self.gpu = GpuProbe::None;
        }
        Ok(Vec::new())
    }
}

/// A GPU tool ran past [`TOOL_DEADLINE`] and was killed.
#[derive(Debug, PartialEq, Eq)]
struct Hung;

/// Runs a tool and returns its output, killing it at `deadline`. `Ok(None)`: the tool is
/// missing or could not start.
fn run_with_deadline(
    mut cmd: std::process::Command,
    deadline: Duration,
) -> Result<Option<Output>, Hung> {
    let Ok(mut child) = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    else {
        return Ok(None);
    };
    // Read on another thread so a chatty tool cannot fill the pipe and stall.
    let mut out = child.stdout.take();
    let reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(o) = out.as_mut() {
            let _ = o.read_to_end(&mut buf);
        }
        buf
    });
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let stdout = reader.join().unwrap_or_default();
                return Ok(Some(Output {
                    status,
                    stdout,
                    stderr: Vec::new(),
                }));
            }
            Ok(None) if started.elapsed() < deadline => {
                std::thread::sleep(Duration::from_millis(10))
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = reader.join();
                return Err(Hung);
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return Ok(None);
            }
        }
    }
}

fn apple_gpu() -> Result<Option<GpuLive>, Hung> {
    let mut cmd = command("ioreg");
    cmd.args(["-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"]);
    Ok(run_with_deadline(cmd, TOOL_DEADLINE)?
        .filter(|o| o.status.success())
        .and_then(|o| parse_ioreg(&String::from_utf8_lossy(&o.stdout))))
}

/// Reads `"PerformanceStatistics" = {...}` from `ioreg -c IOAccelerator` output.
fn parse_ioreg(text: &str) -> Option<GpuLive> {
    let start = text.find("\"PerformanceStatistics\" = {")?;
    let body = &text[start..];
    let body = &body[body.find('{')? + 1..body.find('}')?];
    let mut fields = HashMap::new();
    for part in body.split(',') {
        if let Some((k, v)) = part.split_once('=') {
            fields.insert(k.trim().trim_matches('"').to_string(), v.trim().to_string());
        }
    }
    let num = |k: &str| fields.get(k).and_then(|v| v.parse::<u64>().ok());
    let name = text
        .find("\"model\" = \"")
        .map(|i| &text[i + 11..])
        .and_then(|s| s.split('"').next())
        .map(str::to_string);
    Some(GpuLive {
        name,
        source: "ioreg".into(),
        utilization_percent: num("Device Utilization %").map(|v| v as f32),
        memory_used_bytes: num("In use system memory"),
        memory_total_bytes: None,
        temperature_c: None,
        power_w: None,
        power_limit_w: None,
        pcie: None,
    })
}

fn nvidia_gpus() -> Result<Vec<GpuLive>, Hung> {
    let mut cmd = command("nvidia-smi");
    cmd.args([
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,pcie.link.gen.current,pcie.link.width.current",
        "--format=csv,noheader,nounits",
    ]);
    Ok(run_with_deadline(cmd, TOOL_DEADLINE)?
        .filter(|o| o.status.success())
        .map(|o| parse_nvidia_live(&String::from_utf8_lossy(&o.stdout)))
        .unwrap_or_default())
}

fn parse_nvidia_live(text: &str) -> Vec<GpuLive> {
    const MIB: u64 = 1024 * 1024;
    text.lines()
        .filter_map(|line| {
            let f: Vec<&str> = line.split(',').map(str::trim).collect();
            if f.len() < 9 {
                return None;
            }
            // nvidia-smi prints "[N/A]" or "[Not Supported]" for fields a card does not report.
            let n = |s: &str| s.parse::<f32>().ok();
            let pcie = match (f[7].parse::<u32>(), f[8].parse::<u32>()) {
                (Ok(g), Ok(w)) => Some(format!("Gen{g} x{w}")),
                _ => None,
            };
            Some(GpuLive {
                name: Some(f[0].to_string()),
                source: "nvidia-smi".into(),
                utilization_percent: n(f[1]),
                memory_used_bytes: f[2].parse::<u64>().ok().map(|m| m * MIB),
                memory_total_bytes: f[3].parse::<u64>().ok().map(|m| m * MIB),
                temperature_c: n(f[4]),
                power_w: n(f[5]),
                power_limit_w: n(f[6]),
                pcie,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ioreg_performance_statistics() {
        let text = r#"+-o AGXAcceleratorG16X  <class AGXAcceleratorG16X>
    {
      "model" = "Apple M4 Max"
      "PerformanceStatistics" = {"In use system memory (driver)"=0,"Alloc system memory"=9999695872,"Tiler Utilization %"=2,"Renderer Utilization %"=95,"Device Utilization %"=97,"In use system memory"=1517174784}
    }"#;
        let g = parse_ioreg(text).unwrap();
        assert_eq!(g.name.as_deref(), Some("Apple M4 Max"));
        assert_eq!(g.utilization_percent, Some(97.0));
        assert_eq!(g.memory_used_bytes, Some(1517174784));
        assert_eq!(g.temperature_c, None);
        assert_eq!(parse_ioreg("no statistics here"), None);
    }

    #[test]
    fn parses_nvidia_live_csv_and_unsupported_fields() {
        let g = parse_nvidia_live(
            "NVIDIA GeForce RTX 3060, 100, 11674, 12288, 68, 159.20, 170.00, 3, 16\n\
             Tesla T4, [N/A], 10, 15360, [N/A], [Not Supported], 70.00, 3, 8\nbad line\n",
        );
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].utilization_percent, Some(100.0));
        assert_eq!(g[0].memory_used_bytes, Some(11674 * 1024 * 1024));
        assert_eq!(g[0].temperature_c, Some(68.0));
        assert_eq!(g[0].power_w, Some(159.2));
        assert_eq!(g[0].pcie.as_deref(), Some("Gen3 x16"));
        assert_eq!(g[1].utilization_percent, None);
        assert_eq!(g[1].power_w, None);
    }

    #[test]
    fn first_sample_has_no_rates_and_the_second_does() {
        let me = std::process::id();
        let mut s = LiveSampler::new();
        let a = s.sample(&[me, u32::MAX - 7]);
        assert_eq!(a.cpu_percent, None);
        assert_eq!(a.processes.len(), 1, "a pid that does not exist is skipped");
        assert_eq!(a.processes[0].disk_read_bps, None);
        assert!(a.memory_total_bytes > 0 && a.memory_used_bytes <= a.memory_total_bytes);
        std::thread::sleep(std::time::Duration::from_millis(250));
        let b = s.sample(&[me]);
        assert!(b.cpu_percent.is_some());
        assert!(b.processes[0].disk_read_bps.is_some());
        assert!(b.processes[0].cpu_percent.is_some());
    }

    #[cfg(unix)]
    #[test]
    fn process_cpu_drops_to_zero_when_it_sleeps() {
        // A child that burns CPU in its own process for 1-2 s, then sleeps.
        let mut child = std::process::Command::new("perl")
            .args(["-e", "my $t = time + 2; 1 while time < $t; sleep 30"])
            .spawn()
            .unwrap();
        let pid = child.id();
        let mut s = LiveSampler::new();
        s.sample(&[pid]);
        std::thread::sleep(Duration::from_millis(400));
        let busy = s.sample(&[pid]).processes[0].cpu_percent.unwrap();
        std::thread::sleep(Duration::from_millis(2600));
        s.sample(&[pid]);
        std::thread::sleep(Duration::from_millis(500));
        let idle = s.sample(&[pid]).processes[0].cpu_percent.unwrap();
        let _ = child.kill();
        let _ = child.wait();
        assert!(busy > 30.0, "busy loop measured at {busy}%");
        assert!(idle < 5.0, "sleeping process still shown at {idle}%");
    }

    #[test]
    fn a_long_pause_starts_afresh_instead_of_averaging() {
        let me = std::process::id();
        let mut s = LiveSampler::new();
        s.sample(&[me]);
        s.last_sample = Some(Instant::now() - STALE_AFTER - Duration::from_secs(1));
        let b = s.sample(&[me]);
        assert_eq!(b.cpu_percent, None);
        assert_eq!(b.processes[0].cpu_percent, None);
        assert_eq!(b.processes[0].disk_read_bps, None);
    }

    #[cfg(unix)]
    #[test]
    fn processes_no_longer_asked_for_are_followed_until_they_exit() {
        let mut child = std::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .unwrap();
        let pid = child.id();
        let mut s = LiveSampler::new();
        s.sample(&[pid]);
        s.sample(&[]);
        assert!(
            s.lingering.contains(&pid),
            "still alive, still tracked for cleanup"
        );
        let _ = child.kill();
        let _ = child.wait();
        s.sample(&[]);
        assert!(s.lingering.is_empty(), "dropped once it exited");
        assert!(s.sys.process(Pid::from_u32(pid)).is_none());
    }

    #[cfg(unix)]
    #[test]
    fn a_hanging_tool_is_killed_at_the_deadline() {
        let mut cmd = std::process::Command::new("sleep");
        cmd.arg("30");
        let t = Instant::now();
        assert_eq!(
            run_with_deadline(cmd, Duration::from_millis(200)),
            Err(Hung)
        );
        assert!(t.elapsed() < Duration::from_secs(5));
        let mut ok = std::process::Command::new("sh");
        ok.args(["-c", "echo hello"]);
        let out = run_with_deadline(ok, Duration::from_secs(5))
            .unwrap()
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "hello");
        let missing = std::process::Command::new("llmario-no-such-tool");
        assert_eq!(run_with_deadline(missing, Duration::from_secs(1)), Ok(None));
    }
}
