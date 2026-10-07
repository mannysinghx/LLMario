//! The other LLMario edition on this computer ([`llmario_core::SIBLING`]): which of its engines
//! are running and how much memory they hold. Its files are only read. Its memory is not part of
//! this edition's plan, so loading models in both apps can make the computer swap.

use crate::engine::{self, RecordedEngine};
use crate::memory::fmt_bytes;
use llmario_core::Paths;
use serde::Serialize;

#[derive(Serialize, Clone, Debug)]
pub struct SiblingUsage {
    pub name: &'static str,
    pub engines: Vec<RecordedEngine>,
}

impl SiblingUsage {
    pub fn resident_bytes(&self) -> u64 {
        self.engines.iter().filter_map(|e| e.resident_bytes).sum()
    }

    /// One-line warning for people.
    pub fn warning(&self) -> String {
        let n = self.engines.len();
        let mut what: Vec<String> = self.engines.iter().map(|e| e.model.clone()).collect();
        if self.resident_bytes() > 0 {
            what.push(fmt_bytes(self.resident_bytes()));
        }
        format!(
            "{name} is also running {n} model{s} ({what}). That memory is not counted here; \
             loading models in both apps can make this computer swap. Unload it in {name} if \
             loading is slow.",
            name = self.name,
            s = if n == 1 { "" } else { "s" },
            what = what.join(", "),
        )
    }
}

/// Running engines of the sibling edition; `None` when it is not installed or has none.
pub fn sibling_usage(own: &Paths) -> Option<SiblingUsage> {
    let name = llmario_core::SIBLING.as_ref()?.name;
    let engines = engine::running_engines(&own.sibling()?.run_dir());
    (!engines.is_empty()).then_some(SiblingUsage { name, engines })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warning_names_models_and_memory() {
        let mut u = SiblingUsage {
            name: "LLMario",
            engines: vec![RecordedEngine {
                pid: 1,
                model: "qwen3-8b".into(),
                resident_bytes: Some(5 * 1024 * 1024 * 1024),
            }],
        };
        let w = u.warning();
        assert!(
            w.starts_with("LLMario is also running 1 model (qwen3-8b, 5.00 GiB)."),
            "{w}"
        );
        assert!(w.contains("Unload it in LLMario"), "{w}");
        u.engines[0].resident_bytes = None;
        u.engines.push(u.engines[0].clone());
        assert!(
            u.warning()
                .starts_with("LLMario is also running 2 models (qwen3-8b, qwen3-8b)."),
            "unknown memory is left out"
        );
    }
}
