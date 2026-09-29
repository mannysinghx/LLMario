use llmario_core::ModelFormat;
use serde::Deserialize;

const BUILTIN: &str = include_str!("../catalog.toml");

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct CatalogEntry {
    pub id: String,
    pub family: String,
    pub format: ModelFormat,
    pub repo: String,
    pub revision: String,
    /// Exact files to fetch. Empty = every allowed file in the repo (MLX directories).
    #[serde(default)]
    pub files: Vec<String>,
    pub license: String,
    pub description: String,
    /// Approximate download size, for display and pre-download fit checks.
    #[serde(default)]
    pub approx_bytes: Option<u64>,
}

#[derive(Deserialize, Debug)]
pub struct Catalog {
    pub models: Vec<CatalogEntry>,
}

impl Catalog {
    pub fn builtin() -> Self {
        toml::from_str(BUILTIN).expect("built-in catalog.toml is valid")
    }

    pub fn get(&self, id: &str) -> Option<&CatalogEntry> {
        self.models.iter().find(|m| m.id == id)
    }

    pub fn family(&self, family: &str) -> Vec<&CatalogEntry> {
        self.models.iter().filter(|m| m.family == family).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_catalog_is_consistent() {
        let c = Catalog::builtin();
        assert!(!c.models.is_empty());
        let mut ids: Vec<_> = c.models.iter().map(|m| &m.id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), c.models.len(), "catalog ids are unique");
        for m in &c.models {
            assert!(m.repo.contains('/'), "{}: repo must be org/name", m.id);
            if m.format == ModelFormat::Gguf {
                assert!(
                    !m.files.is_empty(),
                    "{}: GGUF entries must pin a file",
                    m.id
                );
            }
            for f in &m.files {
                crate::validate_relative_name(f).unwrap();
            }
        }
        assert_eq!(c.family("qwen3-1.7b").len(), 2);
    }
}
