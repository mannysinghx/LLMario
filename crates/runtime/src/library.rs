//! The model library as front-ends show it: every catalog entry with its memory estimate for
//! this machine, whether the installed engine can load it, and one recommended variant per family.

use llmario_core::{BackendKind, ModelFormat};
use llmario_registry::Catalog;
use llmario_supervisor::planner::{self, Selection};
use llmario_supervisor::Supervisor;
use serde::Serialize;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogView {
    pub id: String,
    pub family: String,
    pub name: String,
    pub publisher: Option<String>,
    pub released: Option<String>,
    pub params: Option<String>,
    pub tasks: Vec<String>,
    pub description: String,
    pub notes: Option<String>,
    pub format: ModelFormat,
    pub backend: BackendKind,
    pub backend_available: bool,
    pub engine_version: Option<String>,
    /// Architecture support in the installed engine: `Some(false)` = it cannot load this model.
    pub supported: Option<bool>,
    pub architecture: Option<String>,
    pub quantization: Option<String>,
    pub context_max: Option<u32>,
    /// Exact Hugging Face repository, pinned commit and files.
    pub repo: String,
    pub revision: String,
    pub files: Vec<String>,
    pub license: String,
    pub gated: bool,
    pub approx_bytes: Option<u64>,
    /// Memory estimate with the current profile, before download.
    pub needs_bytes: u64,
    pub budget_bytes: u64,
    pub fits: bool,
    /// Fits, but above the comfortable target for a machine with 16 GB or less.
    pub tight: bool,
    /// A build with multi-token-prediction layers (faster with `speculative = "mtp"`).
    pub mtp: bool,
    /// A 3-bit build: smaller and faster than 4-bit, with some quality loss.
    pub low_bit: bool,
    /// Decode speed on this computer: measured, or predicted from memory bandwidth.
    pub speed: Option<llmario_supervisor::speed::SpeedEstimate>,
    /// The variant this machine should use for the family: supported, fits, and preferred engine.
    pub recommended: bool,
    pub installed: bool,
}

/// Build the library view for this machine and runtime profile.
pub fn catalog_views(sup: &Supervisor) -> Vec<CatalogView> {
    let reg = sup.registry();
    let cat = Catalog::builtin();
    let tuned = sup.autotune();
    let mut views: Vec<CatalogView> = cat
        .models
        .iter()
        .map(|c| {
            let backend = planner::backend_for(c.format);
            let status = sup.statuses().get(&backend);
            let entry = c.planning_entry();
            let profile = planner::effective_profile(sup.profile, sup.cfg.runtime.context, &entry);
            let sel = Selection {
                model: entry,
                backend,
                profile,
                reason: String::new(),
            };
            let plan = sup.plan_memory(&sel, 0);
            CatalogView {
                id: c.id.clone(),
                family: c.family.clone(),
                name: c.display_name().to_string(),
                publisher: c.publisher.clone(),
                released: c.released.clone(),
                params: c.params.clone(),
                tasks: c.tasks.clone(),
                description: c.description.clone(),
                notes: c.notes.clone(),
                format: c.format,
                backend,
                backend_available: status.is_some_and(|s| s.available),
                engine_version: status.and_then(|s| s.version.clone()),
                supported: c
                    .architecture
                    .as_deref()
                    .and_then(|a| status.and_then(|s| s.supports_architecture(a))),
                architecture: c.architecture.clone(),
                quantization: c.quantization.clone(),
                context_max: c.context_max,
                repo: c.repo.clone(),
                revision: c.revision.clone(),
                files: c.files.clone(),
                license: c.license.clone(),
                gated: c.gated,
                approx_bytes: c.approx_bytes,
                needs_bytes: plan.total_bytes,
                budget_bytes: plan.budget_bytes,
                fits: plan.fits,
                tight: plan.tight,
                mtp: c.shape.as_ref().is_some_and(|s| s.mtp_layers > 0),
                low_bit: is_low_bit(c.quantization.as_deref()),
                speed: sup.speed(&sel, &tuned),
                recommended: false,
                installed: reg.get(&c.id).is_some(),
            }
        })
        .collect();

    recommend(&mut views, sup.hw.apple_silicon);
    views
}

/// Whether a catalog quantization label is a 3-bit (or lower) build.
fn is_low_bit(q: Option<&str>) -> bool {
    q.is_some_and(|q| {
        let q = q.to_ascii_uppercase();
        q.starts_with("Q3")
            || q.starts_with("IQ3")
            || q.starts_with("Q2")
            || q.starts_with("IQ2")
            || q.starts_with("3-BIT")
    })
}

/// Slowest decode speed a recommended variant may have (about reading speed), when a measured
/// or predicted speed exists.
pub const MIN_RECOMMENDED_TPS: f64 = 10.0;

/// One recommended variant per family: usable here, comfortable (not tight on machines with
/// 16 GB or less) and at least [`MIN_RECOMMENDED_TPS`] when its speed is known; then 4-bit before
/// 3-bit, the preferred engine for this machine, and the plain build before the MTP one. Families
/// with no such variant get no recommendation.
fn recommend(views: &mut [CatalogView], prefer_mlx: bool) {
    let mut best: std::collections::HashMap<String, ((u8, u8, u8), usize)> = Default::default();
    for (i, v) in views.iter().enumerate() {
        let fast_enough = v
            .speed
            .as_ref()
            .is_none_or(|s| s.tokens_per_second >= MIN_RECOMMENDED_TPS);
        if !(v.backend_available && v.fits && !v.tight && fast_enough && v.supported != Some(false))
        {
            continue;
        }
        let engine = match (v.backend, prefer_mlx) {
            (BackendKind::Mlx, true) | (BackendKind::LlamaCpp, false) => 0,
            _ => 1,
        };
        let rank = (v.low_bit as u8, engine, v.mtp as u8);
        let e = best.entry(v.family.clone()).or_insert(((u8::MAX, 0, 0), i));
        if rank < e.0 {
            *e = (rank, i);
        }
    }
    for (_, (_, i)) in best {
        views[i].recommended = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(id: &str, backend: BackendKind, tight: bool, low_bit: bool, mtp: bool) -> CatalogView {
        CatalogView {
            id: id.into(),
            family: "fam".into(),
            name: "Fam".into(),
            publisher: None,
            released: None,
            params: None,
            tasks: vec![],
            description: String::new(),
            notes: None,
            format: if backend == BackendKind::Mlx {
                ModelFormat::Mlx
            } else {
                ModelFormat::Gguf
            },
            backend,
            backend_available: true,
            engine_version: None,
            supported: Some(true),
            architecture: None,
            quantization: None,
            context_max: None,
            repo: String::new(),
            revision: String::new(),
            files: vec![],
            license: String::new(),
            gated: false,
            approx_bytes: None,
            needs_bytes: 0,
            budget_bytes: 0,
            fits: true,
            tight,
            mtp,
            low_bit,
            speed: None,
            recommended: false,
            installed: false,
        }
    }

    fn pick(mut v: Vec<CatalogView>, prefer_mlx: bool) -> Option<String> {
        recommend(&mut v, prefer_mlx);
        v.into_iter().find(|v| v.recommended).map(|v| v.id)
    }

    #[test]
    fn recommends_comfortable_then_quality_then_engine() {
        use BackendKind::{LlamaCpp as L, Mlx as M};
        // 4-bit beats 3-bit even on the less preferred engine.
        let v = vec![
            view("q3-mlx", M, false, true, false),
            view("q4-gguf", L, false, false, false),
        ];
        assert_eq!(pick(v, true).as_deref(), Some("q4-gguf"));
        // A tight 4-bit loses to a comfortable 3-bit.
        let v = vec![
            view("q4-mlx", M, true, false, false),
            view("q3-gguf", L, false, true, false),
        ];
        assert_eq!(pick(v, true).as_deref(), Some("q3-gguf"));
        // Same quality: the preferred engine, then the plain build before MTP.
        let same = || {
            vec![
                view("q4-gguf-mtp", L, false, false, true),
                view("q4-gguf", L, false, false, false),
                view("q4-mlx", M, false, false, false),
            ]
        };
        assert_eq!(pick(same(), true).as_deref(), Some("q4-mlx"));
        assert_eq!(pick(same(), false).as_deref(), Some("q4-gguf"));
        // Only tight variants: no recommendation.
        assert_eq!(pick(vec![view("q4", L, true, false, false)], false), None);
        // Too slow on this computer: no recommendation; unknown speed does not block.
        let mut slow = view("q4", L, false, false, false);
        slow.speed = Some(llmario_supervisor::speed::SpeedEstimate {
            tokens_per_second: 6.0,
            measured: false,
            basis: String::new(),
            bytes_per_token: 0,
        });
        assert_eq!(pick(vec![slow], false), None);
        assert!(is_low_bit(Some("Q3_K_M (3-bit: …)")) && is_low_bit(Some("3-bit (group 64): …")));
        assert!(
            !is_low_bit(Some("Q4_K_M"))
                && !is_low_bit(Some("4-bit (group 64)"))
                && !is_low_bit(None)
        );
    }
}
