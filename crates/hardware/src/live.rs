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
use std::collections::HashMap;
use std::time::Instant;
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

/// Reads the machine on demand and remembers the previous reading for rates.
pub struct LiveSampler {
    sys: System,
    /// pid → (when, total bytes read, total bytes written) at the previous sample.
    prev_disk: HashMap<u32, (Instant, u64, u64)>,
    cpu_primed: bool,
    gpu: GpuProbe,
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
            prev_disk: HashMap::new(),
            cpu_primed: false,
            gpu: GpuProbe::Unknown,
        }
    }

    /// Takes one reading of the machine and of `pids`. Blocking (it may run `ioreg` or
    /// `nvidia-smi`, a few milliseconds each): call it off the async executor.
    pub fn sample(&mut self, pids: &[u32]) -> SystemLive {
        self.sys.refresh_cpu_usage();
        self.sys.refresh_memory();
        let cpu_percent = self
            .cpu_primed
            .then(|| self.sys.global_cpu_usage())
            .filter(|v| v.is_finite());
        self.cpu_primed = true;

        let sys_pids: Vec<Pid> = pids.iter().map(|p| Pid::from_u32(*p)).collect();
        if !sys_pids.is_empty() {
            self.sys.refresh_processes_specifics(
                ProcessesToUpdate::Some(&sys_pids),
                true,
                ProcessRefreshKind::nothing()
                    .with_cpu()
                    .with_disk_usage()
                    .with_memory(),
            );
        }
        let now = Instant::now();
        let mut processes = Vec::with_capacity(pids.len());
        let mut seen = HashMap::new();
        for &pid in pids {
            let Some(p) = self.sys.process(Pid::from_u32(pid)) else {
                continue;
            };
            let d = p.disk_usage();
            let prev = self.prev_disk.get(&pid).copied();
            let rate = |before: u64, after: u64, since: Instant| {
                let secs = now.duration_since(since).as_secs_f64();
                (secs > 0.05).then(|| (after.saturating_sub(before) as f64 / secs) as u64)
            };
            let had_prev = prev.is_some();
            processes.push(ProcessLive {
                pid,
                cpu_percent: had_prev.then(|| p.cpu_usage()).filter(|v| v.is_finite()),
                memory_bytes: crate::process_memory_bytes(pid).or(Some(p.memory())),
                disk_read_bps: prev.and_then(|(t, r, _)| rate(r, d.total_read_bytes, t)),
                disk_write_bps: prev.and_then(|(t, _, w)| rate(w, d.total_written_bytes, t)),
                disk_read_total: d.total_read_bytes,
                disk_write_total: d.total_written_bytes,
            });
            seen.insert(pid, (now, d.total_read_bytes, d.total_written_bytes));
        }
        // Forget processes that are gone, so a reused pid does not inherit old totals.
        self.prev_disk = seen;

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
        if matches!(self.gpu, GpuProbe::Unknown | GpuProbe::Apple)
            && cfg!(all(target_os = "macos", target_arch = "aarch64"))
        {
            if let Some(g) = apple_gpu() {
                self.gpu = GpuProbe::Apple;
                return vec![g];
            }
        }
        if matches!(self.gpu, GpuProbe::Unknown | GpuProbe::Nvidia) {
            let g = nvidia_gpus();
            if !g.is_empty() {
                self.gpu = GpuProbe::Nvidia;
                return g;
            }
        }
        // Probe once; a machine without these tools does not get a process spawned per sample.
        if self.gpu == GpuProbe::Unknown {
            self.gpu = GpuProbe::None;
        }
        Vec::new()
    }
}

fn apple_gpu() -> Option<GpuLive> {
    let out = command("ioreg")
        .args(["-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_ioreg(&String::from_utf8_lossy(&out.stdout))
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

fn nvidia_gpus() -> Vec<GpuLive> {
    let Ok(out) = command("nvidia-smi")
        .args([
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,pcie.link.gen.current,pcie.link.width.current",
            "--format=csv,noheader,nounits",
        ])
        .output()
    else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }
    parse_nvidia_live(&String::from_utf8_lossy(&out.stdout))
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
}
