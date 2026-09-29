//! Hardware discovery. Every value is either measured or explicitly marked as an estimate in
//! `notes`; we never claim an accelerator or capability we could not observe.

mod process;

pub use process::process_memory_bytes;

use llmario_core::os::background_command as command;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum GpuApi {
    Metal,
    Cuda,
    Rocm,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GpuInfo {
    pub vendor: String,
    pub name: String,
    pub api: GpuApi,
    /// Dedicated VRAM (discrete) or usable GPU working set (unified). `None` if unknown.
    pub memory_total_bytes: Option<u64>,
    pub memory_free_bytes: Option<u64>,
    pub driver: Option<String>,
    pub cores: Option<u32>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HardwareReport {
    pub os: String,
    pub os_version: String,
    pub arch: String,
    pub cpu_brand: String,
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub performance_cores: Option<usize>,
    pub efficiency_cores: Option<usize>,
    pub cpu_features: Vec<String>,
    pub total_memory_bytes: u64,
    pub available_memory_bytes: u64,
    pub apple_silicon: bool,
    /// CPU and GPU share one memory pool (Apple Silicon).
    pub unified_memory: bool,
    pub gpus: Vec<GpuInfo>,
    /// Caveats about how values were obtained (estimates, missing tools, unvalidated paths).
    pub notes: Vec<String>,
}

impl HardwareReport {
    pub fn detect() -> Self {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        sys.refresh_cpu_all();

        let arch = std::env::consts::ARCH.to_string();
        let os = std::env::consts::OS.to_string();
        let os_version = sysinfo::System::long_os_version().unwrap_or_else(|| "unknown".into());
        let cpu_brand = sys
            .cpus()
            .first()
            .map(|c| c.brand().trim().to_string())
            .filter(|b| !b.is_empty())
            .unwrap_or_else(|| "unknown".into());
        let logical_cores = sys.cpus().len();
        let physical_cores = sysinfo::System::physical_core_count().unwrap_or(logical_cores);
        let mut notes = Vec::new();

        let apple_silicon = os == "macos" && arch == "aarch64";
        let (performance_cores, efficiency_cores) = if os == "macos" {
            (
                sysctl_u64("hw.perflevel0.physicalcpu").map(|v| v as usize),
                sysctl_u64("hw.perflevel1.physicalcpu").map(|v| v as usize),
            )
        } else {
            (None, None)
        };

        let total_memory_bytes = sys.total_memory();
        let available_memory_bytes = sys.available_memory();

        let mut gpus = Vec::new();
        if apple_silicon {
            gpus.push(detect_apple_gpu(total_memory_bytes, &mut notes));
        }
        gpus.extend(detect_nvidia(&mut notes));
        if let Some(g) = detect_rocm(&mut notes) {
            gpus.push(g);
        }
        if gpus.is_empty() {
            notes.push("no supported GPU detected; CPU-only execution (llama.cpp)".into());
        }

        Self {
            os,
            os_version,
            arch,
            cpu_brand,
            physical_cores,
            logical_cores,
            performance_cores,
            efficiency_cores,
            cpu_features: cpu_features(),
            total_memory_bytes,
            available_memory_bytes,
            apple_silicon,
            unified_memory: apple_silicon,
            gpus,
            notes,
        }
    }

    /// Stable identifier of this machine's inference-relevant hardware (no serial numbers).
    /// Used to key benchmark results and future tuning caches.
    pub fn fingerprint(&self) -> String {
        let mut h = Sha256::new();
        h.update(self.arch.as_bytes());
        h.update(self.cpu_brand.as_bytes());
        h.update(self.total_memory_bytes.to_le_bytes());
        for g in &self.gpus {
            h.update(g.name.as_bytes());
            h.update(g.memory_total_bytes.unwrap_or(0).to_le_bytes());
        }
        hex::encode(&h.finalize()[..8])
    }

    /// The accelerator a model would be offloaded to, if any.
    pub fn primary_gpu(&self) -> Option<&GpuInfo> {
        self.gpus
            .iter()
            .find(|g| g.api == GpuApi::Metal)
            .or_else(|| {
                self.gpus
                    .iter()
                    .max_by_key(|g| g.memory_total_bytes.unwrap_or(0))
            })
    }

    /// Threads for CPU-side work: performance cores on hybrid CPUs, else physical cores.
    pub fn recommended_threads(&self) -> usize {
        self.performance_cores.unwrap_or(self.physical_cores).max(1)
    }

    pub fn summary_line(&self) -> String {
        let gpu = self
            .primary_gpu()
            .map(|g| g.name.clone())
            .unwrap_or_else(|| "no GPU".into());
        format!(
            "{} · {} · {:.0} GiB RAM · {}",
            self.cpu_brand,
            self.os_version,
            self.total_memory_bytes as f64 / GIB,
            gpu
        )
    }
}

fn detect_apple_gpu(total_mem: u64, notes: &mut Vec<String>) -> GpuInfo {
    // Metal can wire only part of unified memory for the GPU. `iogpu.wired_limit_mb` is the
    // user override; 0 means the OS default, which Apple does not publish as a sysctl. The
    // commonly observed default is ~2/3 of RAM up to 36 GiB and ~3/4 above; we use that and
    // say so.
    let wired = sysctl_u64("iogpu.wired_limit_mb").filter(|v| *v > 0);
    let working_set = match wired {
        Some(mb) => {
            notes.push(format!(
                "GPU working set from iogpu.wired_limit_mb = {mb} MiB"
            ));
            mb * 1024 * 1024
        }
        None => {
            let frac = if total_mem as f64 > 36.0 * GIB {
                0.75
            } else {
                2.0 / 3.0
            };
            notes.push(format!(
                "GPU working set estimated as {:.0}% of unified memory (macOS default; set iogpu.wired_limit_mb to change)",
                frac * 100.0
            ));
            (total_mem as f64 * frac) as u64
        }
    };
    let (name, cores) = apple_gpu_name().unwrap_or_else(|| ("Apple GPU".into(), None));
    GpuInfo {
        vendor: "apple".into(),
        name,
        api: GpuApi::Metal,
        memory_total_bytes: Some(working_set),
        memory_free_bytes: None,
        driver: None,
        cores,
    }
}

fn apple_gpu_name() -> Option<(String, Option<u32>)> {
    let out = command("system_profiler")
        .args(["SPDisplaysDataType", "-json"])
        .output()
        .ok()?;
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
    let first = v.get("SPDisplaysDataType")?.as_array()?.first()?;
    let name = first.get("sppci_model")?.as_str()?.to_string();
    let cores = first
        .get("sppci_cores")
        .and_then(|c| c.as_str())
        .and_then(|c| c.parse().ok());
    Some((name, cores))
}

fn detect_nvidia(notes: &mut Vec<String>) -> Vec<GpuInfo> {
    let out = match command("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        Ok(o) if o.status.success() => o,
        Ok(_) => {
            notes.push("nvidia-smi present but failed; NVIDIA GPUs not reported".into());
            return vec![];
        }
        Err(_) => return vec![],
    };
    parse_nvidia_smi(&String::from_utf8_lossy(&out.stdout))
}

fn parse_nvidia_smi(text: &str) -> Vec<GpuInfo> {
    const MIB: u64 = 1024 * 1024;
    text.lines()
        .filter_map(|line| {
            let f: Vec<&str> = line.split(',').map(str::trim).collect();
            if f.len() < 4 {
                return None;
            }
            Some(GpuInfo {
                vendor: "nvidia".into(),
                name: f[0].to_string(),
                api: GpuApi::Cuda,
                memory_total_bytes: f[1].parse::<u64>().ok().map(|m| m * MIB),
                memory_free_bytes: f[2].parse::<u64>().ok().map(|m| m * MIB),
                driver: Some(f[3].to_string()),
                cores: None,
            })
        })
        .collect()
}

fn detect_rocm(notes: &mut Vec<String>) -> Option<GpuInfo> {
    let out = command("rocm-smi").arg("--showproductname").output().ok()?;
    if !out.status.success() {
        return None;
    }
    notes.push(
        "AMD GPU detected via rocm-smi; VRAM is not measured and the ROCm path is unvalidated"
            .into(),
    );
    Some(GpuInfo {
        vendor: "amd".into(),
        name: "AMD GPU (rocm-smi)".into(),
        api: GpuApi::Rocm,
        memory_total_bytes: None,
        memory_free_bytes: None,
        driver: None,
        cores: None,
    })
}

fn sysctl_u64(name: &str) -> Option<u64> {
    let out = command("sysctl").args(["-n", name]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

#[allow(unused_mut)]
fn cpu_features() -> Vec<String> {
    let mut f: Vec<&str> = Vec::new();
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::is_aarch64_feature_detected as has;
        if has!("neon") {
            f.push("neon")
        }
        if has!("dotprod") {
            f.push("dotprod")
        }
        if has!("fp16") {
            f.push("fp16")
        }
        if has!("i8mm") {
            f.push("i8mm")
        }
        if has!("bf16") {
            f.push("bf16")
        }
        if has!("sve") {
            f.push("sve")
        }
        if has!("sve2") {
            f.push("sve2")
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::is_x86_feature_detected as has;
        if has!("sse4.2") {
            f.push("sse4.2")
        }
        if has!("avx") {
            f.push("avx")
        }
        if has!("avx2") {
            f.push("avx2")
        }
        if has!("fma") {
            f.push("fma")
        }
        if has!("f16c") {
            f.push("f16c")
        }
        if has!("avx512f") {
            f.push("avx512f")
        }
        if has!("avx512bw") {
            f.push("avx512bw")
        }
        if has!("avx512vnni") {
            f.push("avx512vnni")
        }
    }
    f.into_iter().map(String::from).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_is_self_consistent() {
        let r = HardwareReport::detect();
        assert!(r.total_memory_bytes > 0);
        assert!(r.available_memory_bytes <= r.total_memory_bytes);
        assert!(r.logical_cores >= 1 && r.recommended_threads() >= 1);
        assert_eq!(r.fingerprint(), r.fingerprint());
        assert_eq!(r.fingerprint().len(), 16);
        if r.apple_silicon {
            let g = r.primary_gpu().expect("apple silicon has a Metal GPU");
            assert!(g.memory_total_bytes.unwrap() < r.total_memory_bytes);
        }
    }

    #[test]
    fn parses_nvidia_smi_csv() {
        let g = parse_nvidia_smi("NVIDIA GeForce RTX 4090, 24564, 23000, 550.54.14\nbad line\n");
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].memory_total_bytes, Some(24564 * 1024 * 1024));
        assert_eq!(g[0].driver.as_deref(), Some("550.54.14"));
    }
}
