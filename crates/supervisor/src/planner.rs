//! Execution-path selection: which installed variant and which backend serve a request.
//!
//! Rules (no silent substitution):
//! - An exact model id is served by the backend for its format, or refused with the reason.
//! - A family name (e.g. `qwen3-1.7b`) picks among installed variants: the configured
//!   `backends.prefer` first, then MLX on Apple Silicon, then llama.cpp. The chosen concrete id
//!   is reported back in the response `model` field and headers.

use crate::adapter::BackendStatus;
use llmario_core::{BackendKind, Config, ModelFormat, ProfileKind, ResolvedProfile, RuntimeError};
use llmario_hardware::HardwareReport;
use llmario_registry::{ModelEntry, Registry};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Selection {
    pub model: ModelEntry,
    pub backend: BackendKind,
    pub profile: ResolvedProfile,
    pub reason: String,
}

pub fn backend_for(format: ModelFormat) -> BackendKind {
    match format {
        ModelFormat::Gguf => BackendKind::LlamaCpp,
        ModelFormat::Mlx => BackendKind::Mlx,
        ModelFormat::Mock => BackendKind::Mock,
    }
}

/// Clamp the profile to what the model supports.
pub fn effective_profile(
    kind: ProfileKind,
    ctx_override: Option<u32>,
    model: &ModelEntry,
) -> ResolvedProfile {
    let mut p = ResolvedProfile::resolve(kind, ctx_override);
    if let Some(max) = model.shape.as_ref().and_then(|s| s.context_max) {
        if max > 0 && p.ctx_per_slot > max {
            p.ctx_per_slot = max;
        }
    }
    p
}

pub fn select(
    name: &str,
    registry: &Registry,
    statuses: &HashMap<BackendKind, BackendStatus>,
    hw: &HardwareReport,
    cfg: &Config,
    profile: ProfileKind,
) -> Result<Selection, RuntimeError> {
    let candidates = registry.resolve(name);
    if candidates.is_empty() {
        return Err(RuntimeError::ModelNotFound(name.to_string()));
    }
    let exact = candidates.len() == 1 && candidates[0].id == name;

    let rank = |m: &ModelEntry| -> u8 {
        let b = backend_for(m.format);
        if cfg.backends.prefer == Some(b) {
            0
        } else if hw.apple_silicon && b == BackendKind::Mlx {
            1
        } else if b == BackendKind::LlamaCpp {
            2
        } else {
            3
        }
    };
    let mut sorted: Vec<&ModelEntry> = candidates;
    sorted.sort_by_key(|m| (rank(m), m.id.clone()));

    let mut rejected = Vec::new();
    for m in sorted {
        let backend = backend_for(m.format);
        match statuses.get(&backend) {
            Some(s) if s.available => {
                let reason = if exact {
                    format!("{} requested explicitly → {backend}", m.id)
                } else {
                    format!(
                        "family '{name}' → {} via {backend} (ranked by preference/hardware)",
                        m.id
                    )
                };
                return Ok(Selection {
                    profile: effective_profile(profile, cfg.runtime.context, m),
                    model: m.clone(),
                    backend,
                    reason,
                });
            }
            Some(s) => rejected.push(format!("{} needs {backend}: {}", m.id, s.detail)),
            None => rejected.push(format!(
                "{} needs {backend}, which this build does not include",
                m.id
            )),
        }
    }
    Err(RuntimeError::BackendUnavailable(format!(
        "no available backend can serve '{name}': {}. Run `llmario doctor`.",
        rejected.join("; ")
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::tests::{hw, model};

    fn status(kind: BackendKind, available: bool) -> (BackendKind, BackendStatus) {
        (
            kind,
            BackendStatus {
                kind,
                available,
                path: None,
                version: None,
                tested_version: String::new(),
                detail: if available {
                    "ok".into()
                } else {
                    "not installed".into()
                },
            },
        )
    }

    fn reg() -> Registry {
        let mut g = model(1.0);
        g.id = "q-gguf".into();
        g.family = Some("q".into());
        let mut x = model(1.0);
        x.id = "q-mlx".into();
        x.family = Some("q".into());
        x.format = ModelFormat::Mlx;
        Registry::in_memory(vec![g, x])
    }

    #[test]
    fn family_prefers_mlx_on_apple_silicon() {
        let mut h = hw(64, None);
        h.apple_silicon = true;
        let st: HashMap<_, _> = [
            status(BackendKind::Mlx, true),
            status(BackendKind::LlamaCpp, true),
        ]
        .into();
        let s = select(
            "q",
            &reg(),
            &st,
            &h,
            &Config::default(),
            ProfileKind::Latency,
        )
        .unwrap();
        assert_eq!(s.model.id, "q-mlx");
        h.apple_silicon = false;
        let s = select(
            "q",
            &reg(),
            &st,
            &h,
            &Config::default(),
            ProfileKind::Latency,
        )
        .unwrap();
        assert_eq!(s.model.id, "q-gguf");
    }

    #[test]
    fn config_preference_wins() {
        let mut h = hw(64, None);
        h.apple_silicon = true;
        let mut cfg = Config::default();
        cfg.backends.prefer = Some(BackendKind::LlamaCpp);
        let st: HashMap<_, _> = [
            status(BackendKind::Mlx, true),
            status(BackendKind::LlamaCpp, true),
        ]
        .into();
        assert_eq!(
            select("q", &reg(), &st, &h, &cfg, ProfileKind::Latency)
                .unwrap()
                .model
                .id,
            "q-gguf"
        );
    }

    #[test]
    fn exact_id_never_substituted() {
        let h = hw(64, None);
        let st: HashMap<_, _> = [
            status(BackendKind::Mlx, false),
            status(BackendKind::LlamaCpp, true),
        ]
        .into();
        let err = select(
            "q-mlx",
            &reg(),
            &st,
            &h,
            &Config::default(),
            ProfileKind::Latency,
        )
        .unwrap_err();
        assert_eq!(err.code(), "backend_unavailable");
        // The family still resolves to the working variant.
        assert_eq!(
            select(
                "q",
                &reg(),
                &st,
                &h,
                &Config::default(),
                ProfileKind::Latency
            )
            .unwrap()
            .model
            .id,
            "q-gguf"
        );
        assert_eq!(
            select(
                "nope",
                &reg(),
                &st,
                &h,
                &Config::default(),
                ProfileKind::Latency
            )
            .unwrap_err()
            .code(),
            "model_not_found"
        );
    }

    #[test]
    fn context_clamped_to_model_max() {
        let mut m = model(1.0);
        m.shape.as_mut().unwrap().context_max = Some(2048);
        assert_eq!(
            effective_profile(ProfileKind::Latency, None, &m).ctx_per_slot,
            2048
        );
        assert_eq!(
            effective_profile(ProfileKind::Latency, Some(1024), &m).ctx_per_slot,
            1024
        );
    }
}
