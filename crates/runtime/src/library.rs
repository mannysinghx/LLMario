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
    /// The variant this machine should use for the family: supported, fits, and preferred engine.
    pub recommended: bool,
    pub installed: bool,
}

/// Build the library view for this machine and runtime profile.
pub fn catalog_views(sup: &Supervisor) -> Vec<CatalogView> {
    let reg = sup.registry();
    let cat = Catalog::builtin();
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
                recommended: false,
                installed: reg.get(&c.id).is_some(),
            }
        })
        .collect();

    // One recommended variant per family: usable here, then the preferred engine for this machine.
    let prefer_mlx = sup.hw.apple_silicon;
    let mut best: std::collections::HashMap<String, (u8, usize)> = Default::default();
    for (i, v) in views.iter().enumerate() {
        if !(v.backend_available && v.fits && v.supported != Some(false)) {
            continue;
        }
        let rank = match (v.backend, prefer_mlx) {
            (BackendKind::Mlx, true) | (BackendKind::LlamaCpp, false) => 0,
            _ => 1,
        };
        let e = best.entry(v.family.clone()).or_insert((u8::MAX, i));
        if rank < e.0 {
            *e = (rank, i);
        }
    }
    for (_, (_, i)) in best {
        views[i].recommended = true;
    }
    views
}
