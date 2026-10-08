//! Measured speeds on this computer (`$LLMARIO_HOME/autotune.json`), written by `bench` and read
//! by the speed planner. Keyed by hardware fingerprint, engine and version, model (id and content
//! hash), profile and speculative mode, so a result only applies to the setup it was measured on.

use llmario_core::BackendKind;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Measurement {
    /// `HardwareReport::fingerprint()`.
    pub hardware: String,
    pub backend: BackendKind,
    /// Engine version as probed (e.g. "build 11146"); a different version invalidates the result.
    pub engine_version: String,
    pub model: String,
    #[serde(default)]
    pub model_hash: Option<String>,
    pub profile: String,
    /// Speculative decoding in effect: "off", "ngram", "mtp" or "draft:<model id>".
    pub speculative: String,
    /// GPU/CPU split in effect: "" when the model is not split, else "cpu-moe:<layers>" or
    /// "gpu-layers:<layers>" (see `speed::placement_label`). A split reads part of the model at
    /// CPU speed, so it is a different setup and never stands for the machine's bandwidth.
    #[serde(default)]
    pub placement: String,
    /// KV cache type when not the default f16 (llama.cpp "q8_0"); it changes decode speed
    /// (gpt-oss-20b: 114 tok/s f16, 102 tok/s q8_0 on the M4 Max).
    #[serde(default)]
    pub kv_cache: String,
    /// Mean decode speed of one request at a time.
    pub decode_tps: f64,
    /// Weight and cache bytes read per generated token (the speed planner's model).
    pub bytes_per_token: u64,
    /// decode_tps × bytes_per_token, in GB/s.
    pub effective_gbs: f64,
    pub measured_at: String,
}

impl Measurement {
    fn same_setup(&self, o: &Measurement) -> bool {
        self.hardware == o.hardware
            && self.backend == o.backend
            && self.engine_version == o.engine_version
            && self.model == o.model
            && self.model_hash == o.model_hash
            && self.profile == o.profile
            && self.speculative == o.speculative
            && self.placement == o.placement
            && self.kv_cache == o.kv_cache
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct Autotune {
    pub measurements: Vec<Measurement>,
}

impl Autotune {
    /// Load the cache; a missing or unreadable file is an empty cache.
    pub fn load(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(self)?)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Add a measurement, replacing an older one for the same setup.
    pub fn record(&mut self, m: Measurement) {
        self.measurements.retain(|o| !o.same_setup(&m));
        self.measurements.push(m);
    }

    /// The measurement for exactly this setup, if any.
    #[allow(clippy::too_many_arguments)]
    pub fn lookup(
        &self,
        hardware: &str,
        backend: BackendKind,
        engine_version: &str,
        model: &str,
        model_hash: Option<&str>,
        profile: &str,
        speculative: &str,
        placement: &str,
        kv_cache: &str,
    ) -> Option<&Measurement> {
        self.measurements.iter().rev().find(|m| {
            m.hardware == hardware
                && m.backend == backend
                && m.engine_version == engine_version
                && m.model == model
                && m.model_hash.as_deref() == model_hash
                && m.profile == profile
                && m.speculative == speculative
                && m.placement == placement
                && m.kv_cache == kv_cache
        })
    }

    /// Median effective bandwidth of plain decoding (no speculation, no GPU/CPU split, the same KV
    /// cache type) measured on this hardware with this engine and version: the speed planner's
    /// best estimate for other models.
    pub fn effective_bandwidth(
        &self,
        hardware: &str,
        backend: BackendKind,
        engine_version: &str,
        kv_cache: &str,
    ) -> Option<f64> {
        let mut v: Vec<f64> = self
            .measurements
            .iter()
            .filter(|m| {
                m.hardware == hardware
                    && m.backend == backend
                    && m.engine_version == engine_version
                    && m.speculative == "off"
                    && m.placement.is_empty()
                    && m.kv_cache == kv_cache
                    && m.effective_gbs > 0.0
            })
            .map(|m| m.effective_gbs)
            .collect();
        if v.is_empty() {
            return None;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some(v[v.len() / 2])
    }

    /// Measured decode speed per backend for a model family on this hardware (plain decoding),
    /// used to choose between engines when a family is installed in several formats.
    pub fn family_speed(
        &self,
        hardware: &str,
        model_ids: &[&str],
        backend: BackendKind,
    ) -> Option<f64> {
        self.measurements
            .iter()
            .rev()
            .find(|m| {
                m.hardware == hardware
                    && m.backend == backend
                    && m.speculative == "off"
                    && model_ids.contains(&m.model.as_str())
            })
            .map(|m| m.decode_tps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(model: &str, backend: BackendKind, spec: &str, tps: f64, gbs: f64) -> Measurement {
        Measurement {
            hardware: "hw1".into(),
            backend,
            engine_version: "v1".into(),
            model: model.into(),
            model_hash: None,
            profile: "latency".into(),
            speculative: spec.into(),
            placement: String::new(),
            kv_cache: String::new(),
            decode_tps: tps,
            bytes_per_token: (gbs * 1e9 / tps) as u64,
            effective_gbs: gbs,
            measured_at: "2026-10-07T00:00:00Z".into(),
        }
    }

    #[test]
    fn record_replaces_same_setup_and_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("autotune.json");
        assert_eq!(
            Autotune::load(&path),
            Autotune::default(),
            "missing file is empty"
        );
        let mut a = Autotune::default();
        a.record(m("a", BackendKind::LlamaCpp, "off", 50.0, 300.0));
        a.record(m("a", BackendKind::LlamaCpp, "off", 60.0, 360.0));
        a.record(m("a", BackendKind::LlamaCpp, "mtp", 70.0, 420.0));
        a.record(Measurement {
            placement: "cpu-moe:8".into(),
            ..m("a", BackendKind::LlamaCpp, "off", 40.0, 240.0)
        });
        a.record(Measurement {
            kv_cache: "q8_0".into(),
            ..m("a", BackendKind::LlamaCpp, "off", 55.0, 300.0)
        });
        assert_eq!(
            a.measurements.len(),
            4,
            "same setup replaced; other mode, split and cache type kept"
        );
        a.save(&path).unwrap();
        let b = Autotune::load(&path);
        assert_eq!(a, b);
        let hit = b.lookup(
            "hw1",
            BackendKind::LlamaCpp,
            "v1",
            "a",
            None,
            "latency",
            "off",
            "",
            "",
        );
        assert_eq!(hit.unwrap().decode_tps, 60.0);
        let split = b.lookup(
            "hw1",
            BackendKind::LlamaCpp,
            "v1",
            "a",
            None,
            "latency",
            "off",
            "cpu-moe:8",
            "",
        );
        assert_eq!(split.unwrap().decode_tps, 40.0, "a split is its own setup");
        assert!(
            b.lookup(
                "hw1",
                BackendKind::LlamaCpp,
                "v2",
                "a",
                None,
                "latency",
                "off",
                "",
                ""
            )
            .is_none(),
            "other engine version"
        );
        std::fs::write(&path, "not json").unwrap();
        assert_eq!(
            Autotune::load(&path),
            Autotune::default(),
            "unreadable file is empty"
        );
    }

    #[test]
    fn effective_bandwidth_is_the_median_of_plain_runs() {
        let mut a = Autotune::default();
        for (id, gbs) in [("a", 300.0), ("b", 400.0), ("c", 350.0)] {
            a.record(m(id, BackendKind::LlamaCpp, "off", 50.0, gbs));
        }
        a.record(m("d", BackendKind::LlamaCpp, "ngram", 50.0, 900.0));
        a.record(m("e", BackendKind::Mlx, "off", 50.0, 450.0));
        // A GPU/CPU split reads part of the model at CPU speed: not the machine's bandwidth.
        for (id, gbs) in [("f", 185.0), ("g", 150.0)] {
            a.record(Measurement {
                placement: "cpu-moe:8".into(),
                ..m(id, BackendKind::LlamaCpp, "off", 70.0, gbs)
            });
        }
        a.record(Measurement {
            kv_cache: "q8_0".into(),
            ..m("h", BackendKind::LlamaCpp, "off", 50.0, 280.0)
        });
        assert_eq!(
            a.effective_bandwidth("hw1", BackendKind::LlamaCpp, "v1", ""),
            Some(350.0)
        );
        assert_eq!(
            a.effective_bandwidth("hw1", BackendKind::LlamaCpp, "v1", "q8_0"),
            Some(280.0),
            "same cache type only"
        );
        assert_eq!(
            a.effective_bandwidth("hw1", BackendKind::Mlx, "v1", ""),
            Some(450.0)
        );
        assert_eq!(
            a.effective_bandwidth("hw2", BackendKind::Mlx, "v1", ""),
            None
        );
        assert_eq!(
            a.family_speed("hw1", &["e", "a"], BackendKind::Mlx),
            Some(50.0)
        );
    }
}
