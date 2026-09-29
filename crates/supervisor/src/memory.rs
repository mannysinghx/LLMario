//! Conservative memory estimation and budget checks.
//!
//! total = weights + KV cache (all slots, f16) + runtime overhead + backend extras
//!
//! The estimate is deliberately pessimistic: an engine that fails to load after we said it
//! fits is worse than a refusal with a clear alternative. `llmario bench` records measured
//! peak footprint next to this estimate so the model can be calibrated.

use llmario_core::{Config, ResolvedProfile};
use llmario_hardware::{GpuApi, HardwareReport};
use llmario_registry::ModelEntry;
use serde::Serialize;

pub const GIB: u64 = 1024 * 1024 * 1024;
pub const MIB: u64 = 1024 * 1024;
/// f16 KV cache element width.
const KV_ELEM_BYTES: u64 = 2;

#[derive(Serialize, Clone, Debug)]
pub struct MemoryPlan {
    pub weights_bytes: u64,
    pub kv_cache_bytes: u64,
    pub overhead_bytes: u64,
    pub total_bytes: u64,
    /// Memory llmario may plan against after headroom and other loaded models.
    pub budget_bytes: u64,
    pub budget_source: String,
    pub fits: bool,
    pub ctx_total: u64,
    pub kv_bytes_per_token: u64,
    /// llama.cpp `-ngl` when only part of the model fits in dedicated VRAM.
    pub gpu_layers: Option<u32>,
    /// Largest per-request context that would fit with the same profile (multiple of 256).
    pub max_ctx_per_slot_that_fits: Option<u32>,
    pub notes: Vec<String>,
}

impl MemoryPlan {
    pub fn explain(&self) -> String {
        format!(
            "needs ~{} (weights {} + KV cache {} for {} tokens + overhead {}); budget {} ({})",
            fmt_bytes(self.total_bytes),
            fmt_bytes(self.weights_bytes),
            fmt_bytes(self.kv_cache_bytes),
            self.ctx_total,
            fmt_bytes(self.overhead_bytes),
            fmt_bytes(self.budget_bytes),
            self.budget_source
        )
    }

    /// Actionable text for a refusal.
    pub fn refusal(&self, model_id: &str, profile: &ResolvedProfile) -> String {
        let mut s = format!("model '{model_id}' does not fit: {}.", self.explain());
        match self.max_ctx_per_slot_that_fits {
            Some(c) if c >= 512 => s.push_str(&format!(
                " Try a smaller context (--context {c}) or a profile with fewer parallel slots (current: {} × {} tokens).",
                profile.parallel, profile.ctx_per_slot
            )),
            _ => s.push_str(" The weights alone exceed the budget; choose a smaller model or lower-bit quantization."),
        }
        s.push_str(" Other loaded models are unloaded first when idle; memory_headroom_gb / memory_limit_gb change the budget.");
        s
    }
}

pub fn fmt_bytes(b: u64) -> String {
    if b >= GIB {
        format!("{:.2} GiB", b as f64 / GIB as f64)
    } else {
        format!("{:.0} MiB", b as f64 / MIB as f64)
    }
}

pub fn headroom_bytes(hw: &HardwareReport, cfg: &Config) -> u64 {
    match cfg.runtime.memory_headroom_gb {
        Some(g) => (g * GIB as f64) as u64,
        None => (hw.total_memory_bytes / 10).max(2 * GIB),
    }
}

/// f16 KV-cache bytes per token; conservative fallback when the architecture is unknown.
pub fn kv_bytes_per_token(model: &ModelEntry) -> u64 {
    match &model.shape {
        Some(s) => s.kv_bytes_per_token(KV_ELEM_BYTES),
        None => (model.weight_bytes() / 2000).max(64 * 1024),
    }
}

/// Estimate memory for serving `model` with `profile`. `reserved_by_others` is memory held by
/// engines that will stay loaded; `backend_extra` comes from the adapter.
pub fn estimate(
    model: &ModelEntry,
    profile: &ResolvedProfile,
    hw: &HardwareReport,
    cfg: &Config,
    backend_extra: u64,
    reserved_by_others: u64,
) -> MemoryPlan {
    let mut notes = Vec::new();
    let weights = model.weight_bytes();
    let ctx_total = profile.total_ctx();
    if model.shape.is_none() {
        notes.push(
            "model shape unknown: KV cache estimated as weights/2000 per token (conservative)"
                .into(),
        );
    }
    let kv_per_token = kv_bytes_per_token(model);
    let kv = kv_per_token * ctx_total;
    // Compute buffers, tokenizer, runtime: fixed floor plus a share of weights.
    let overhead = 600 * MIB + weights / 12 + backend_extra;
    let total = weights + kv + overhead;

    let headroom = headroom_bytes(hw, cfg);
    let ram_budget = hw.total_memory_bytes.saturating_sub(headroom);
    let limit = cfg.runtime.memory_limit_gb.map(|g| (g * GIB as f64) as u64);

    let gpu = hw.primary_gpu();
    let (mut budget, mut source, mut gpu_layers) = match gpu {
        Some(g) if g.api == GpuApi::Metal => {
            let ws = g.memory_total_bytes.unwrap_or(ram_budget);
            (
                ws.min(ram_budget),
                format!(
                    "unified memory: min(GPU working set {}, RAM {} − headroom {})",
                    fmt_bytes(ws),
                    fmt_bytes(hw.total_memory_bytes),
                    fmt_bytes(headroom)
                ),
                None,
            )
        }
        Some(g) if g.api == GpuApi::Cuda && g.memory_total_bytes.is_some() => {
            let vram = g
                .memory_free_bytes
                .or(g.memory_total_bytes)
                .unwrap_or(0)
                .saturating_sub(512 * MIB);
            if total <= vram {
                (
                    vram,
                    format!("free VRAM on {} (minus 512 MiB)", g.name),
                    None,
                )
            } else {
                // Hybrid offload: KV + overhead stay on GPU, remaining VRAM takes weight layers.
                let n_layers = model.shape.as_ref().map(|s| s.n_layers).unwrap_or(0);
                let spare = vram.saturating_sub(kv + overhead);
                let layers = if weights == 0 || n_layers == 0 {
                    0
                } else {
                    ((spare as u128 * n_layers as u128) / weights as u128) as u32
                };
                notes.push(format!(
                    "partial GPU offload: {layers}/{n_layers} layers in VRAM, rest on CPU (expect lower decode speed; unvalidated path)"
                ));
                (
                    ram_budget,
                    format!(
                        "system RAM {} − headroom {} (hybrid CPU/GPU)",
                        fmt_bytes(hw.total_memory_bytes),
                        fmt_bytes(headroom)
                    ),
                    Some(layers),
                )
            }
        }
        _ => (
            ram_budget,
            format!(
                "system RAM {} − headroom {}",
                fmt_bytes(hw.total_memory_bytes),
                fmt_bytes(headroom)
            ),
            None,
        ),
    };
    if let Some(l) = limit {
        if l < budget {
            budget = l;
            source = format!("memory_limit_gb = {:.1}", l as f64 / GIB as f64);
        }
    }
    if reserved_by_others > 0 {
        budget = budget.saturating_sub(reserved_by_others);
        source.push_str(&format!(
            ", minus {} held by other loaded models",
            fmt_bytes(reserved_by_others)
        ));
    }
    if gpu.is_none() {
        gpu_layers = Some(0);
    }

    let fits = total <= budget;
    let max_ctx = {
        let avail = budget.saturating_sub(weights + overhead);
        let per_slot_tokens = avail / (kv_per_token.max(1) * profile.parallel.max(1) as u64);
        let c = (per_slot_tokens / 256 * 256).min(u32::MAX as u64) as u32;
        (c > 0).then_some(c)
    };
    if hw.available_memory_bytes + reserved_by_others < total && fits {
        notes.push(format!(
            "only {} is currently available; the OS may compress or swap other apps while loading",
            fmt_bytes(hw.available_memory_bytes)
        ));
    }

    MemoryPlan {
        weights_bytes: weights,
        kv_cache_bytes: kv,
        overhead_bytes: overhead,
        total_bytes: total,
        budget_bytes: budget,
        budget_source: source,
        fits,
        ctx_total,
        kv_bytes_per_token: kv_per_token,
        gpu_layers,
        max_ctx_per_slot_that_fits: max_ctx,
        notes,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use llmario_core::{ModelFormat, ProfileKind};
    use llmario_hardware::GpuInfo;
    use llmario_registry::{FileRecord, ModelShape};

    pub fn hw(total_gib: u64, gpu: Option<GpuInfo>) -> HardwareReport {
        HardwareReport {
            os: "test".into(),
            os_version: "test".into(),
            arch: "aarch64".into(),
            cpu_brand: "test".into(),
            physical_cores: 8,
            logical_cores: 8,
            performance_cores: None,
            efficiency_cores: None,
            cpu_features: vec![],
            total_memory_bytes: total_gib * GIB,
            available_memory_bytes: total_gib * GIB / 2,
            apple_silicon: false,
            unified_memory: false,
            gpus: gpu.into_iter().collect(),
            notes: vec![],
        }
    }

    pub fn model(weights_gib: f64) -> ModelEntry {
        ModelEntry {
            id: "m".into(),
            family: None,
            format: ModelFormat::Gguf,
            path: "/x.gguf".into(),
            managed: false,
            source: None,
            license: None,
            architecture: None,
            quantization: None,
            // Qwen3-8B-like: 36 layers × 8 KV heads × 128 dim → 144 KiB/token at f16.
            shape: Some(ModelShape {
                n_layers: 36,
                n_heads: 32,
                n_kv_heads: 8,
                head_dim: 128,
                hidden_size: 4096,
                context_max: Some(40960),
            }),
            chat_template: true,
            files: vec![FileRecord {
                name: "x.gguf".into(),
                size: (weights_gib * GIB as f64) as u64,
                sha256: None,
            }],
            size_bytes: (weights_gib * GIB as f64) as u64,
            added_at: String::new(),
        }
    }

    fn metal(ws_gib: u64) -> GpuInfo {
        GpuInfo {
            vendor: "apple".into(),
            name: "M".into(),
            api: GpuApi::Metal,
            memory_total_bytes: Some(ws_gib * GIB),
            memory_free_bytes: None,
            driver: None,
            cores: None,
        }
    }

    #[test]
    fn kv_scales_with_context_and_slots() {
        let h = hw(64, Some(metal(48)));
        let cfg = Config::default();
        let lat = estimate(
            &model(4.7),
            &ResolvedProfile::resolve(ProfileKind::Latency, None),
            &h,
            &cfg,
            0,
            0,
        );
        assert_eq!(lat.kv_bytes_per_token, 2 * 36 * 8 * 128 * 2);
        assert_eq!(lat.kv_cache_bytes, lat.kv_bytes_per_token * 8192);
        let bal = estimate(
            &model(4.7),
            &ResolvedProfile::resolve(ProfileKind::Balanced, None),
            &h,
            &cfg,
            0,
            0,
        );
        assert_eq!(bal.kv_cache_bytes, lat.kv_cache_bytes * 4);
        assert!(lat.fits && bal.fits);
        assert!(
            lat.budget_bytes <= 48 * GIB,
            "unified budget capped by GPU working set"
        );
    }

    #[test]
    fn refuses_when_too_big_and_suggests_context() {
        let h = hw(16, Some(metal(10)));
        let cfg = Config::default();
        let p = ResolvedProfile::resolve(ProfileKind::Throughput, Some(32768));
        let plan = estimate(&model(4.7), &p, &h, &cfg, 0, 0);
        assert!(!plan.fits);
        let c = plan.max_ctx_per_slot_that_fits.unwrap();
        assert!(c < 32768 && c.is_multiple_of(256));
        let retry = estimate(
            &model(4.7),
            &ResolvedProfile::resolve(ProfileKind::Throughput, Some(c)),
            &h,
            &cfg,
            0,
            0,
        );
        assert!(retry.fits, "the suggested context actually fits");
        assert!(plan.refusal("m", &p).contains("--context"));

        let huge = estimate(
            &model(40.0),
            &ResolvedProfile::resolve(ProfileKind::Latency, None),
            &h,
            &cfg,
            0,
            0,
        );
        assert!(!huge.fits);
        assert!(huge.refusal("m", &p).contains("smaller model"));
    }

    #[test]
    fn limits_and_other_models_shrink_budget() {
        let h = hw(64, Some(metal(48)));
        let mut cfg = Config::default();
        let p = ResolvedProfile::resolve(ProfileKind::Latency, None);
        let base = estimate(&model(4.7), &p, &h, &cfg, 0, 0).budget_bytes;
        assert_eq!(
            estimate(&model(4.7), &p, &h, &cfg, 0, 10 * GIB).budget_bytes,
            base - 10 * GIB
        );
        cfg.runtime.memory_limit_gb = Some(4.0);
        let capped = estimate(&model(4.7), &p, &h, &cfg, 0, 0);
        assert_eq!(capped.budget_bytes, 4 * GIB);
        assert!(!capped.fits);
    }

    #[test]
    fn cuda_partial_offload_when_vram_short() {
        let g = GpuInfo {
            vendor: "nvidia".into(),
            name: "RTX".into(),
            api: GpuApi::Cuda,
            memory_total_bytes: Some(4 * GIB),
            memory_free_bytes: Some(4 * GIB),
            driver: None,
            cores: None,
        };
        let h = hw(64, Some(g));
        let plan = estimate(
            &model(4.7),
            &ResolvedProfile::resolve(ProfileKind::Latency, None),
            &h,
            &Config::default(),
            0,
            0,
        );
        assert!(plan.fits, "falls back to hybrid offload within system RAM");
        let layers = plan.gpu_layers.unwrap();
        assert!(layers > 0 && layers < 36, "{layers}");
    }

    #[test]
    fn cpu_only_uses_ram() {
        let plan = estimate(
            &model(1.0),
            &ResolvedProfile::resolve(ProfileKind::Latency, None),
            &hw(8, None),
            &Config::default(),
            0,
            0,
        );
        assert!(plan.fits);
        assert_eq!(plan.gpu_layers, Some(0));
        assert_eq!(
            plan.budget_bytes,
            6 * GIB,
            "8 GiB minus 2 GiB minimum headroom"
        );
    }
}
