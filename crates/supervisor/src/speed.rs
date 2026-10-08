//! Decode-speed planning (Phase 6). Generating a token reads the weights it uses plus the caches
//! it attends over, so: tokens/s ≈ effective bandwidth ÷ bytes read per token.
//!
//! Bytes read per token = active weights (MoE experts at the share used per token; see
//! `ModelShape::bytes_per_token`) + the KV cache and state read at a typical working context.
//! Effective bandwidth = the median measured on this computer with this engine (`autotune`), or
//! else the published figure × a per-engine efficiency. A model measured on this exact setup
//! uses its measured speed.

use crate::autotune::Autotune;
use crate::memory::kv_plan;
use llmario_core::{BackendKind, Config, KvCacheType, ResolvedProfile, Speculative};
use llmario_hardware::HardwareReport;
use llmario_registry::ModelEntry;
use serde::Serialize;

/// Typical context while generating (prompt plus answer so far), for the cache part of the read.
pub const WORKING_CONTEXT: u64 = 1024;

/// Share of the published memory bandwidth plain decoding achieves, per engine, with
/// [`overhead_s`]. Calibrated on the M4 Max (546 GB/s) from Qwen3 1.7B and Gemma 4 12B (llama.cpp)
/// and Qwen3 1.7B and Qwen3.8 27B (MLX); held-out models came within ±8% (Qwen3.5 9B +7.7%,
/// gpt-oss-20b -5.1%, Qwen3 8B -7.0%, Llama 3.2 3B +1.7%), except below ~1B parameters, where the
/// prediction is conservative (Qwen3.5 0.8B -36%). Used until this computer has measurements.
pub fn efficiency(backend: BackendKind) -> f64 {
    match backend {
        BackendKind::Mlx => MLX_EFFICIENCY,
        _ => LLAMACPP_EFFICIENCY,
    }
}
pub const LLAMACPP_EFFICIENCY: f64 = 0.90;
pub const MLX_EFFICIENCY: f64 = 0.73;

/// Fixed time per generated token beyond reading memory (kernel launches, sampling, the HTTP
/// stream), per engine. Calibrated with [`efficiency`]; it dominates for very small models.
pub fn overhead_s(backend: BackendKind) -> f64 {
    match backend {
        BackendKind::Mlx => MLX_OVERHEAD_S,
        _ => LLAMACPP_OVERHEAD_S,
    }
}
pub const LLAMACPP_OVERHEAD_S: f64 = 0.004_65;
pub const MLX_OVERHEAD_S: f64 = 0.001_22;

/// Share of the published bandwidth the CPU reaches when it reads weights kept in RAM by a
/// GPU/CPU split (Phase 7). Measured on the M4 Max (llama.cpp build 11146, `--load-mode none`):
/// gpt-oss-20b with the experts of 8 of 24 layers on the CPU ran at 70.7 tok/s, which implies the
/// CPU read its 404 MiB per token at ~84 GB/s (15.5% of 546 GB/s); predicted 70.0 tok/s. One
/// model on one chip; unverified elsewhere.
pub const CPU_BANDWIDTH_SHARE: f64 = 0.15;

/// The GPU/CPU split an autotune measurement is keyed by: "" when the model is not split
/// (everything on the GPU, or everything on the CPU of a computer without a GPU).
pub fn placement_label(plan: &crate::memory::MemoryPlan, hw: &HardwareReport) -> String {
    if hw.primary_gpu().is_none() {
        return String::new();
    }
    match (plan.cpu_moe_layers, plan.gpu_layers) {
        (Some(n), _) => format!("cpu-moe:{n}"),
        (None, Some(g)) => format!("gpu-layers:{g}"),
        _ => String::new(),
    }
}

/// The KV cache type an autotune measurement is keyed by: "" for the default f16 cache.
pub fn kv_cache_label(backend: BackendKind, cfg: &Config) -> String {
    let kv = cfg.backends.llamacpp.kv_cache_type;
    if backend == BackendKind::LlamaCpp && kv != KvCacheType::F16 {
        kv.as_arg().to_string()
    } else {
        String::new()
    }
}

/// Bytes per token the CPU reads when a plan splits the model between GPU and CPU (0 without a
/// GPU: the CPU is then the only processor, and its speed is what the bandwidth describes).
pub fn cpu_bytes_per_token(
    model: &ModelEntry,
    plan: &crate::memory::MemoryPlan,
    hw: &HardwareReport,
) -> u64 {
    if placement_label(plan, hw).is_empty() {
        return 0;
    }
    let Some(s) = model.shape.as_ref() else {
        return 0;
    };
    let layers = s.n_layers.saturating_sub(s.mtp_layers).max(1) as u64;
    let weights = if s.bytes_per_token > 0 {
        s.bytes_per_token
    } else {
        model.weight_bytes()
    };
    match (plan.cpu_moe_layers, plan.gpu_layers) {
        (Some(n), _) => s.active_expert_bytes * n.min(layers as u32) as u64 / layers,
        (None, Some(g)) => weights * layers.saturating_sub(g as u64) / layers,
        _ => 0,
    }
}

#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct SpeedEstimate {
    pub tokens_per_second: f64,
    /// Measured for this model on this computer (otherwise predicted).
    pub measured: bool,
    /// What the number is based on, for display.
    pub basis: String,
    /// Bytes read per generated token behind a prediction (0 for measurements).
    #[serde(default)]
    pub bytes_per_token: u64,
}

/// The speculative-decoding label an autotune measurement is keyed by (the mode actually used).
pub fn speculative_label(
    model: &ModelEntry,
    backend: BackendKind,
    cfg: &Config,
    draft: Option<&str>,
) -> String {
    let has_mtp = model.shape.as_ref().is_some_and(|s| s.mtp_layers > 0);
    match backend {
        BackendKind::LlamaCpp => match cfg.backends.llamacpp.speculative.resolve(has_mtp) {
            Speculative::Off | Speculative::Auto => "off".into(),
            Speculative::Ngram => "ngram".into(),
            Speculative::Mtp => "mtp".into(),
            Speculative::Draft => draft
                .map(|d| format!("draft:{d}"))
                .unwrap_or_else(|| "off".into()),
        },
        BackendKind::Mlx => draft
            .map(|d| format!("draft:{d}"))
            .unwrap_or_else(|| "off".into()),
        _ => "off".into(),
    }
}

/// Bytes read to generate one token at the working context, one request at a time.
pub fn bytes_read_per_token(
    model: &ModelEntry,
    profile: &ResolvedProfile,
    backend: BackendKind,
    cfg: &Config,
) -> u64 {
    let weights = model
        .shape
        .as_ref()
        .map(|s| s.bytes_per_token)
        .filter(|b| *b > 0)
        .unwrap_or_else(|| model.weight_bytes());
    let working = ResolvedProfile {
        parallel: 1,
        ctx_per_slot: (profile.ctx_per_slot as u64).min(WORKING_CONTEXT) as u32,
        ..profile.clone()
    };
    weights + kv_plan(model, &working, backend, cfg).total
}

/// Decode speed for `model` on this computer: measured if this exact setup was benchmarked,
/// otherwise predicted. `None` when neither a measurement nor a bandwidth figure exists.
/// `placement` is the plan's GPU/CPU split ([`placement_label`]) and `cpu_bytes` the part of each
/// token's read done by the CPU after it ([`cpu_bytes_per_token`]; 0 when not split).
#[allow(clippy::too_many_arguments)]
pub fn estimate(
    model: &ModelEntry,
    profile: &ResolvedProfile,
    backend: BackendKind,
    hw: &HardwareReport,
    cfg: &Config,
    tuned: &Autotune,
    engine_version: &str,
    speculative: &str,
    placement: &str,
    cpu_bytes: u64,
) -> Option<SpeedEstimate> {
    let fp = hw.fingerprint();
    let hash = model.content_hash();
    let profile_name = profile.kind.to_string();
    let kv = kv_cache_label(backend, cfg);
    if let Some(m) = tuned.lookup(
        &fp,
        backend,
        engine_version,
        &model.id,
        hash.as_deref(),
        &profile_name,
        speculative,
        placement,
        &kv,
    ) {
        return Some(SpeedEstimate {
            tokens_per_second: m.decode_tps,
            measured: true,
            basis: "measured on this computer".into(),
            bytes_per_token: 0,
        });
    }
    let bytes = bytes_read_per_token(model, profile, backend, cfg);
    if bytes == 0 {
        return None;
    }
    let cpu = cpu_bytes.min(bytes);
    // A machine bandwidth measured end to end already includes per-token overhead.
    let (gbs, t0, basis) = match tuned.effective_bandwidth(&fp, backend, engine_version, &kv) {
        Some(e) => (
            e,
            0.0,
            format!("predicted from {e:.0} GB/s measured on this computer"),
        ),
        None => {
            let peak = hw.memory_bandwidth_gbs?;
            let eff = efficiency(backend);
            (
                peak * eff,
                overhead_s(backend),
                format!(
                    "predicted from {peak:.0} GB/s published × {:.0}% efficiency",
                    eff * 100.0
                ),
            )
        }
    };
    let cpu_gbs = hw.memory_bandwidth_gbs.unwrap_or(gbs) * CPU_BANDWIDTH_SHARE;
    let seconds = (bytes - cpu) as f64 / (gbs * 1e9) + cpu as f64 / (cpu_gbs * 1e9) + t0;
    let basis = if cpu > 0 {
        format!(
            "{basis}; {} read by the CPU after a GPU/CPU split",
            crate::memory::fmt_bytes(cpu)
        )
    } else {
        basis
    };
    Some(SpeedEstimate {
        tokens_per_second: 1.0 / seconds,
        measured: false,
        basis,
        bytes_per_token: bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autotune::Measurement;
    use crate::memory::tests::{hw, model};
    use llmario_core::ProfileKind;
    use llmario_hardware::{GpuApi, GpuInfo};

    #[test]
    fn predicts_from_bandwidth_and_prefers_measurements() {
        let mut h = hw(16, None);
        let cfg = Config::default();
        let p = ResolvedProfile::resolve(ProfileKind::Latency, None);
        let mut m = model(4.0);
        // MoE-style: only 2 GB of the 4 GiB file is read per token.
        m.shape.as_mut().unwrap().bytes_per_token = 2_000_000_000;
        let tuned = Autotune::default();
        assert!(
            estimate(
                &m,
                &p,
                BackendKind::LlamaCpp,
                &h,
                &cfg,
                &tuned,
                "v1",
                "off",
                "",
                0
            )
            .is_none(),
            "no bandwidth figure"
        );
        h.memory_bandwidth_gbs = Some(100.0);
        let e = estimate(
            &m,
            &p,
            BackendKind::LlamaCpp,
            &h,
            &cfg,
            &tuned,
            "v1",
            "off",
            "",
            0,
        )
        .unwrap();
        let bytes = bytes_read_per_token(&m, &p, BackendKind::LlamaCpp, &cfg);
        // 2 GB of weights + 36 layers × 8 heads × 128 × 2 × 2 bytes × 1024 tokens of KV.
        assert_eq!(bytes, 2_000_000_000 + 2 * 36 * 8 * 128 * 2 * 1024);
        assert!(
            (e.tokens_per_second
                - 1.0 / (bytes as f64 / (100.0 * LLAMACPP_EFFICIENCY * 1e9) + LLAMACPP_OVERHEAD_S))
                .abs()
                < 1e-6
        );
        assert!(!e.measured && e.basis.contains("published"));

        let mut tuned = Autotune::default();
        let meas = |model: &str, tps: f64, gbs: f64| Measurement {
            hardware: h.fingerprint(),
            backend: BackendKind::LlamaCpp,
            engine_version: "v1".into(),
            model: model.into(),
            model_hash: None,
            profile: "latency".into(),
            speculative: "off".into(),
            placement: String::new(),
            kv_cache: String::new(),
            decode_tps: tps,
            bytes_per_token: 1,
            effective_gbs: gbs,
            measured_at: String::new(),
        };
        tuned.record(meas("other", 40.0, 80.0));
        let e = estimate(
            &m,
            &p,
            BackendKind::LlamaCpp,
            &h,
            &cfg,
            &tuned,
            "v1",
            "off",
            "",
            0,
        )
        .unwrap();
        assert!(
            (e.tokens_per_second - 80.0e9 / bytes as f64).abs() < 1e-9,
            "machine bandwidth measured here"
        );
        assert!(e.basis.contains("measured on this computer") && !e.measured);
        tuned.record(meas("m", 42.0, 84.0));
        let e = estimate(
            &m,
            &p,
            BackendKind::LlamaCpp,
            &h,
            &cfg,
            &tuned,
            "v1",
            "off",
            "",
            0,
        )
        .unwrap();
        assert!(e.measured && e.tokens_per_second == 42.0);
        // A GPU/CPU split makes the CPU part slower.
        let fresh = Autotune::default();
        let all_gpu = estimate(
            &m,
            &p,
            BackendKind::LlamaCpp,
            &h,
            &cfg,
            &fresh,
            "v1",
            "off",
            "",
            0,
        )
        .unwrap();
        let split = estimate(
            &m,
            &p,
            BackendKind::LlamaCpp,
            &h,
            &cfg,
            &fresh,
            "v1",
            "off",
            "cpu-moe:8",
            500_000_000,
        )
        .unwrap();
        assert!(split.tokens_per_second < all_gpu.tokens_per_second && split.basis.contains("CPU"));
        assert_eq!(
            speculative_label(&m, BackendKind::Mlx, &cfg, Some("d")),
            "draft:d"
        );
        let mut auto = cfg.clone();
        auto.backends.llamacpp.speculative = Speculative::Auto;
        assert_eq!(
            speculative_label(&m, BackendKind::LlamaCpp, &auto, None),
            "off",
            "no MTP layers"
        );
        m.shape.as_mut().unwrap().mtp_layers = 1;
        assert_eq!(
            speculative_label(&m, BackendKind::LlamaCpp, &auto, None),
            "mtp"
        );
    }

    /// A benchmark of a split model must not become the machine's bandwidth: before this was
    /// keyed, one split run (gpt-oss-20b, 185 GB/s effective) dropped the prediction for Qwen3.5
    /// 9B from ~66 to ~36 tok/s (it runs at 61).
    #[test]
    fn a_split_run_is_its_own_setup() {
        let metal = GpuInfo {
            vendor: "Apple".into(),
            name: "Apple M4".into(),
            api: GpuApi::Metal,
            memory_total_bytes: None,
            memory_free_bytes: None,
            driver: None,
            cores: None,
        };
        let mut h = hw(64, Some(metal));
        h.memory_bandwidth_gbs = Some(546.0);
        let cfg = Config::default();
        let p = ResolvedProfile::resolve(ProfileKind::Latency, None);
        let m = model(4.0);
        let plain = |tuned: &Autotune| {
            estimate(
                &m,
                &p,
                BackendKind::LlamaCpp,
                &h,
                &cfg,
                tuned,
                "v1",
                "off",
                "",
                0,
            )
            .unwrap()
        };
        let before = plain(&Autotune::default());
        let mut tuned = Autotune::default();
        tuned.record(Measurement {
            hardware: h.fingerprint(),
            backend: BackendKind::LlamaCpp,
            engine_version: "v1".into(),
            model: "big-moe".into(),
            model_hash: None,
            profile: "latency".into(),
            speculative: "off".into(),
            placement: "cpu-moe:8".into(),
            kv_cache: String::new(),
            decode_tps: 70.7,
            bytes_per_token: 2_618_742_528,
            effective_gbs: 185.0,
            measured_at: String::new(),
        });
        let after = plain(&tuned);
        assert_eq!(after, before, "other models still use the published figure");
        // The same model measured split is not shown for an all-GPU plan, and vice versa.
        let mut split_m = m.clone();
        split_m.id = "big-moe".into();
        let all_gpu = estimate(
            &split_m,
            &p,
            BackendKind::LlamaCpp,
            &h,
            &cfg,
            &tuned,
            "v1",
            "off",
            "",
            0,
        )
        .unwrap();
        assert!(!all_gpu.measured);
        let split = estimate(
            &split_m,
            &p,
            BackendKind::LlamaCpp,
            &h,
            &cfg,
            &tuned,
            "v1",
            "off",
            "cpu-moe:8",
            404 * 1024 * 1024,
        )
        .unwrap();
        assert!(split.measured && split.tokens_per_second == 70.7);
        // An 8-bit KV cache is another setup too.
        let mut q8 = cfg.clone();
        q8.backends.llamacpp.kv_cache_type = KvCacheType::Q8_0;
        assert_eq!(kv_cache_label(BackendKind::LlamaCpp, &q8), "q8_0");
        assert_eq!(kv_cache_label(BackendKind::Mlx, &q8), "");
        assert_eq!(kv_cache_label(BackendKind::LlamaCpp, &cfg), "");
    }

    #[test]
    fn placement_follows_the_plan() {
        let metal = GpuInfo {
            vendor: "Apple".into(),
            name: "Apple M4".into(),
            api: GpuApi::Metal,
            memory_total_bytes: None,
            memory_free_bytes: None,
            driver: None,
            cores: None,
        };
        let gpu_hw = hw(16, Some(metal));
        let cpu_hw = hw(16, None);
        let cfg = Config::default();
        let p = ResolvedProfile::resolve(ProfileKind::Latency, None);
        let mut m = model(4.0);
        m.shape.as_mut().unwrap().active_expert_bytes = 1_200_000_000;
        let mut plan = crate::memory::estimate(&m, &p, BackendKind::LlamaCpp, &gpu_hw, &cfg, 0, 0);
        assert_eq!(placement_label(&plan, &gpu_hw), "");
        assert_eq!(cpu_bytes_per_token(&m, &plan, &gpu_hw), 0);
        plan.cpu_moe_layers = Some(9);
        assert_eq!(placement_label(&plan, &gpu_hw), "cpu-moe:9");
        assert_eq!(
            cpu_bytes_per_token(&m, &plan, &gpu_hw),
            1_200_000_000 * 9 / 36
        );
        plan.cpu_moe_layers = None;
        plan.gpu_layers = Some(20);
        assert_eq!(placement_label(&plan, &gpu_hw), "gpu-layers:20");
        // Without a GPU everything runs on the CPU: not a split, and no extra CPU penalty.
        let cpu_plan = crate::memory::estimate(&m, &p, BackendKind::LlamaCpp, &cpu_hw, &cfg, 0, 0);
        assert_eq!(cpu_plan.gpu_layers, Some(0));
        assert_eq!(placement_label(&cpu_plan, &cpu_hw), "");
        assert_eq!(cpu_bytes_per_token(&m, &cpu_plan, &cpu_hw), 0);
    }
}
