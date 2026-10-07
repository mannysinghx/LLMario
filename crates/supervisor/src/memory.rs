//! Conservative memory estimation and budget checks.
//!
//! total = weights + KV cache (all slots, f16) + runtime overhead + backend extras
//!
//! The estimate is deliberately pessimistic: an engine that fails to load after we said it
//! fits is worse than a refusal with a clear alternative. `llmario bench` records measured
//! peak memory next to this estimate so the model can be calibrated.
//!
//! The KV cache follows the model's per-layer layout when it is known ([`kv_plan`]): only
//! full-attention layers grow with the context. Unknown layouts count every layer as full
//! attention, so the estimate never drops below what the engine allocates.

use llmario_core::{BackendKind, Config, KvAccounting, ResolvedProfile};
use llmario_hardware::{GpuApi, HardwareReport};
use llmario_registry::ModelEntry;
use serde::Serialize;

pub const GIB: u64 = 1024 * 1024 * 1024;
pub const MIB: u64 = 1024 * 1024;
/// f16 KV cache element width.
const KV_ELEM_BYTES: u64 = 2;
/// Tokens `mlx_lm.server` processes at once while reading a prompt (`--prefill-step-size`,
/// default 2048; llmario does not change it).
const MLX_PREFILL_STEP: u64 = 2048;

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

/// f16 KV-cache bytes per token with every layer as full attention; conservative fallback
/// when the architecture is unknown.
pub fn kv_bytes_per_token(model: &ModelEntry) -> u64 {
    match &model.shape {
        Some(s) => s.kv_bytes_per_token(KV_ELEM_BYTES),
        None => (model.weight_bytes() / 2000).max(64 * 1024),
    }
}

/// The KV cache planned for one model, profile and engine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvPlan {
    /// Bytes per token of context in one sequence: full-attention layers only.
    pub growing_per_token: u64,
    /// Bytes that stop growing: sliding-window caches at their cap plus fixed state, all slots.
    pub bounded: u64,
    /// Total for the profile's context and slots.
    pub total: u64,
    /// The model's per-layer layout was used (else every layer counts as full attention).
    pub per_layer: bool,
    /// One-line description of the layout, for notes.
    pub layout: String,
}

/// Cells a sliding-window cache holds: the window of every sequence plus the tokens processed
/// at once, rounded up to 256. Measured on llama.cpp (8k context, 1 slot, micro-batch 512):
/// Gemma 4 12B window 1024 → 1536 cells, gpt-oss-20b window 128 → 768 cells. For several slots
/// this assumes one window per slot (not yet measured; it can only overestimate).
fn window_cells(window: u64, parallel: u64, batch: u64) -> u64 {
    (window * parallel + batch).div_ceil(256) * 256
}

/// Plan the KV cache: full-attention layers for the whole context, sliding-window layers up to
/// their window, recurrent / linear-attention layers as fixed state per sequence. Falls back to
/// every layer as full attention when the layout is unknown or `kv_accounting = "conservative"`.
pub fn kv_plan(
    model: &ModelEntry,
    profile: &ResolvedProfile,
    backend: BackendKind,
    cfg: &Config,
) -> KvPlan {
    let parallel = profile.parallel.max(1) as u64;
    let full_cells = profile.total_ctx();
    let Some(shape) = &model.shape else {
        let per = kv_bytes_per_token(model);
        return KvPlan {
            growing_per_token: per,
            bounded: 0,
            total: per * full_cells,
            per_layer: false,
            layout: "layout unknown".into(),
        };
    };
    let per_layer =
        cfg.runtime.kv_accounting == KvAccounting::PerLayer && !shape.kv_groups.is_empty();
    let groups = if per_layer {
        shape.kv_groups.clone()
    } else {
        shape.conservative_groups()
    };
    let batch = match backend {
        BackendKind::Mlx => MLX_PREFILL_STEP,
        _ => profile.ubatch as u64,
    };
    let (mut growing, mut bounded, mut total) = (0u64, 0u64, 0u64);
    let (mut full_layers, mut parts) = (0u32, Vec::new());
    for g in &groups {
        let per = g.bytes_per_token(KV_ELEM_BYTES);
        match g.window {
            None => {
                growing += per;
                total += per * full_cells;
                full_layers += g.layers;
            }
            Some(w) => {
                let cap = window_cells(w as u64, parallel, batch);
                bounded += per * cap;
                total += per * cap.min(full_cells);
                parts.push(format!("{} sliding-window (window {w})", g.layers));
            }
        }
    }
    let state = if per_layer {
        shape.state_bytes_per_seq * parallel
    } else {
        0
    };
    bounded += state;
    total += state;
    let mut layout = format!("{full_layers} of {} layers full attention", shape.n_layers);
    for p in parts {
        layout.push_str(&format!(", {p}"));
    }
    if state > 0 {
        layout.push_str(&format!(", fixed state {}", fmt_bytes(state)));
    }
    KvPlan {
        growing_per_token: growing,
        bounded,
        total,
        per_layer,
        layout,
    }
}

/// Estimate memory for serving `model` with `profile` on `backend`. `reserved_by_others` is
/// memory held by engines that will stay loaded; `backend_extra` comes from the adapter.
pub fn estimate(
    model: &ModelEntry,
    profile: &ResolvedProfile,
    backend: BackendKind,
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
    let kvp = kv_plan(model, profile, backend, cfg);
    if kvp.per_layer {
        notes.push(format!("KV cache per layer: {}", kvp.layout));
    }
    let kv_per_token = kvp.growing_per_token;
    let kv = kvp.total;
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
        // Only full-attention layers grow with the context; the rest is counted at its cap.
        let avail = budget.saturating_sub(weights + overhead + kvp.bounded);
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
                kv_groups: vec![],
                state_bytes_per_seq: 0,
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

    use llmario_registry::KvGroup;

    const MIB_: u64 = 1024 * 1024;

    fn group(layers: u32, heads: u32, dim: u32, window: Option<u32>) -> KvGroup {
        KvGroup {
            layers,
            n_kv_heads: heads,
            head_dim: dim,
            window,
        }
    }

    /// A model with the given per-layer layout (weights 1 GiB; only the KV cache is checked).
    fn layered(n_layers: u32, groups: Vec<KvGroup>, state: u64) -> ModelEntry {
        let mut m = model(1.0);
        let s = m.shape.as_mut().unwrap();
        s.n_layers = n_layers;
        s.kv_groups = groups;
        s.state_bytes_per_seq = state;
        m
    }

    fn latency() -> ResolvedProfile {
        ResolvedProfile::resolve(ProfileKind::Latency, None)
    }

    /// llama.cpp build 11146 with llmario's latency flags (8192 context, 1 slot, micro-batch 512)
    /// reported these allocations; the planner must match them exactly.
    #[test]
    fn matches_llama_cpp_allocations() {
        let cfg = Config::default();
        // Qwen3.5 9B: 8 of 32 layers full attention (4 KV heads × 256); recurrent state
        // R 2.25 MiB + S 48 MiB = 52_690_944 bytes. llama.cpp: KV 256 MiB, RS 50.25 MiB.
        let q = layered(32, vec![group(8, 4, 256, None)], 52_690_944);
        let p = kv_plan(&q, &latency(), BackendKind::LlamaCpp, &cfg);
        assert_eq!(p.total, 256 * MIB_ + 52_690_944);
        assert!(
            p.per_layer && p.layout.contains("8 of 32 layers"),
            "{}",
            p.layout
        );
        // Gemma 4 12B: 8 full layers (1 KV head × 512) + 40 sliding (8 × 256, window 1024).
        // llama.cpp: non-SWA 128 MiB (8192 cells) + SWA 480 MiB (1536 cells).
        let g = layered(
            48,
            vec![group(8, 1, 512, None), group(40, 8, 256, Some(1024))],
            0,
        );
        assert_eq!(
            kv_plan(&g, &latency(), BackendKind::LlamaCpp, &cfg).total,
            (128 + 480) * MIB_
        );
        // gpt-oss-20b: 12 full + 12 sliding (window 128), 8 KV heads × 64.
        // llama.cpp: non-SWA 192 MiB + SWA 18 MiB (768 cells).
        let o = layered(
            24,
            vec![group(12, 8, 64, None), group(12, 8, 64, Some(128))],
            0,
        );
        assert_eq!(
            kv_plan(&o, &latency(), BackendKind::LlamaCpp, &cfg).total,
            (192 + 18) * MIB_
        );
    }

    #[test]
    fn unknown_layout_and_conservative_switch_count_every_layer() {
        let g = layered(
            48,
            vec![group(8, 1, 512, None), group(40, 8, 256, Some(1024))],
            0,
        );
        let mut cfg = Config::default();
        cfg.runtime.kv_accounting = KvAccounting::Conservative;
        let every_layer = 2 * 48 * 8 * 128 * 2 * 8192;
        let p = kv_plan(&g, &latency(), BackendKind::LlamaCpp, &cfg);
        assert!(!p.per_layer);
        assert_eq!(p.total, every_layer);
        let unknown = layered(48, vec![], 0);
        assert_eq!(
            kv_plan(
                &unknown,
                &latency(),
                BackendKind::LlamaCpp,
                &Config::default()
            )
            .total,
            every_layer
        );
    }

    #[test]
    fn sliding_windows_stop_growing_and_mlx_holds_a_prefill_step() {
        let cfg = Config::default();
        let swa = layered(1, vec![group(1, 1, 128, Some(1024))], 0);
        let short = ResolvedProfile::resolve(ProfileKind::Latency, Some(512));
        let long = ResolvedProfile::resolve(ProfileKind::Latency, Some(65536));
        let per = 2 * 128 * 2;
        // Never more than the context, never more than window + batch.
        assert_eq!(
            kv_plan(&swa, &short, BackendKind::LlamaCpp, &cfg).total,
            per * 512
        );
        assert_eq!(
            kv_plan(&swa, &long, BackendKind::LlamaCpp, &cfg).total,
            per * 1536
        );
        assert_eq!(
            kv_plan(&swa, &long, BackendKind::Mlx, &cfg).total,
            per * 3072
        );
        assert_eq!(
            kv_plan(&swa, &long, BackendKind::Mlx, &cfg).growing_per_token,
            0
        );
    }

    #[test]
    fn gemma_4_12b_fits_a_16_gb_mac_with_its_real_layout() {
        // 16 GB Mac: GPU working set 2/3 of RAM. Weights as in the catalog (6.50 GiB, Q4_0).
        let h = hw(16, Some(metal(16 * 2 / 3)));
        let mut g = layered(
            48,
            vec![group(8, 1, 512, None), group(40, 8, 256, Some(1024))],
            0,
        );
        g.files[0].size = 6_979_321_856;
        g.size_bytes = 6_979_321_856;
        let shape = g.shape.as_mut().unwrap();
        shape.n_kv_heads = 8;
        shape.head_dim = 512;
        let cfg = Config::default();
        let extra = 1024 * MIB_; // llama.cpp --cache-ram
        let now = estimate(&g, &latency(), BackendKind::LlamaCpp, &h, &cfg, extra, 0);
        assert!(now.fits, "{}", now.explain());
        assert!(now.notes.iter().any(|n| n.contains("KV cache per layer")));
        let mut old = cfg.clone();
        old.runtime.kv_accounting = KvAccounting::Conservative;
        let before = estimate(&g, &latency(), BackendKind::LlamaCpp, &h, &old, extra, 0);
        assert!(
            !before.fits,
            "the old formula refused it: {}",
            before.explain()
        );
    }

    #[test]
    fn max_context_solves_only_for_the_growing_part() {
        let h = hw(16, Some(metal(10)));
        let cfg = Config::default();
        let g = layered(
            48,
            vec![group(8, 1, 512, None), group(40, 8, 256, Some(1024))],
            0,
        );
        let p = ResolvedProfile::resolve(ProfileKind::Latency, Some(1 << 20));
        let plan = estimate(&g, &p, BackendKind::LlamaCpp, &h, &cfg, 0, 0);
        let c = plan.max_ctx_per_slot_that_fits.unwrap();
        let retry = estimate(
            &g,
            &ResolvedProfile::resolve(ProfileKind::Latency, Some(c)),
            BackendKind::LlamaCpp,
            &h,
            &cfg,
            0,
            0,
        );
        assert!(
            retry.fits,
            "the suggested context fits: {}",
            retry.explain()
        );
        assert!(c > 100_000, "only 16 KiB/token grows: {c}");
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
            BackendKind::LlamaCpp,
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
            BackendKind::LlamaCpp,
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
        let plan = estimate(&model(4.7), &p, BackendKind::LlamaCpp, &h, &cfg, 0, 0);
        assert!(!plan.fits);
        let c = plan.max_ctx_per_slot_that_fits.unwrap();
        assert!(c < 32768 && c.is_multiple_of(256));
        let retry = estimate(
            &model(4.7),
            &ResolvedProfile::resolve(ProfileKind::Throughput, Some(c)),
            BackendKind::LlamaCpp,
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
            BackendKind::LlamaCpp,
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
        let base = estimate(&model(4.7), &p, BackendKind::LlamaCpp, &h, &cfg, 0, 0).budget_bytes;
        assert_eq!(
            estimate(
                &model(4.7),
                &p,
                BackendKind::LlamaCpp,
                &h,
                &cfg,
                0,
                10 * GIB
            )
            .budget_bytes,
            base - 10 * GIB
        );
        cfg.runtime.memory_limit_gb = Some(4.0);
        let capped = estimate(&model(4.7), &p, BackendKind::LlamaCpp, &h, &cfg, 0, 0);
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
            BackendKind::LlamaCpp,
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
            BackendKind::LlamaCpp,
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
