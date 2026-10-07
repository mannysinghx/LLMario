use llmario_core::ModelFormat;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Architecture facts needed for memory planning.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct ModelShape {
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub hidden_size: u32,
    /// Maximum trained context, if the model declares one.
    pub context_max: Option<u32>,
    /// Which layers keep a KV cache and how big it is, when known. Empty = unknown: every layer
    /// is planned as full attention at `n_kv_heads` × `head_dim` (the conservative layout).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub kv_groups: Vec<KvGroup>,
    /// Fixed per-sequence state of recurrent / linear-attention layers, in bytes.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub state_bytes_per_seq: u64,
}

fn is_zero(v: &u64) -> bool {
    *v == 0
}

/// Layers that share one KV-cache shape.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct KvGroup {
    pub layers: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    /// Sliding-window size in tokens; `None` = full attention (the cache grows with the context).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<u32>,
}

impl KvGroup {
    /// K and V bytes per cached token at the given element width (2 = f16/bf16).
    pub fn bytes_per_token(&self, bytes_per_elem: u64) -> u64 {
        2 * self.layers as u64 * self.n_kv_heads as u64 * self.head_dim as u64 * bytes_per_elem
    }
}

impl ModelShape {
    /// KV-cache bytes per token at the given element width (2 = f16/bf16), counting every layer
    /// as full attention (the conservative layout).
    pub fn kv_bytes_per_token(&self, bytes_per_elem: u64) -> u64 {
        2 * self.n_layers as u64 * self.n_kv_heads as u64 * self.head_dim as u64 * bytes_per_elem
    }

    /// Every layer as one full-attention group (used when the real layout is unknown).
    pub fn conservative_groups(&self) -> Vec<KvGroup> {
        vec![KvGroup {
            layers: self.n_layers,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            window: None,
        }]
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct ModelSource {
    /// Hugging Face repository id, e.g. `mlx-community/Qwen3-1.7B-4bit`.
    pub repo: String,
    /// Resolved commit SHA the files were downloaded from.
    pub revision: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct FileRecord {
    /// Path relative to the model root (directory for MLX, parent dir for GGUF).
    pub name: String,
    pub size: u64,
    pub sha256: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ModelEntry {
    /// Unique local name, e.g. `qwen3-1.7b-mlx-4bit`.
    pub id: String,
    /// Model family shared by format variants, e.g. `qwen3-1.7b`. Requests may use it.
    #[serde(default)]
    pub family: Option<String>,
    pub format: ModelFormat,
    /// GGUF: the `.gguf` file (first shard). MLX: the model directory.
    pub path: PathBuf,
    /// True when llmario owns the files (downloaded into the models dir) and may delete them.
    pub managed: bool,
    #[serde(default)]
    pub source: Option<ModelSource>,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub architecture: Option<String>,
    #[serde(default)]
    pub quantization: Option<String>,
    #[serde(default)]
    pub shape: Option<ModelShape>,
    /// Whether the model ships a chat template (required for chat completions).
    #[serde(default)]
    pub chat_template: bool,
    #[serde(default)]
    pub files: Vec<FileRecord>,
    /// Total bytes of weight + tokenizer files.
    pub size_bytes: u64,
    pub added_at: String,
}

impl ModelEntry {
    /// Stable content hash over file names and SHA-256s (used to key benchmark/tuning data).
    /// `None` when any file lacks a recorded checksum.
    pub fn content_hash(&self) -> Option<String> {
        if self.files.is_empty() || self.files.iter().any(|f| f.sha256.is_none()) {
            return None;
        }
        let mut files: Vec<_> = self.files.iter().collect();
        files.sort_by(|a, b| a.name.cmp(&b.name));
        let mut h = Sha256::new();
        for f in files {
            h.update(f.name.as_bytes());
            h.update(b":");
            h.update(f.sha256.as_deref().unwrap_or_default().as_bytes());
            h.update(b"\n");
        }
        Some(hex::encode(&h.finalize()[..8]))
    }

    /// Bytes of weights only (safetensors / gguf), used as the memory estimate base.
    pub fn weight_bytes(&self) -> u64 {
        let w: u64 = self
            .files
            .iter()
            .filter(|f| f.name.ends_with(".safetensors") || f.name.ends_with(".gguf"))
            .map(|f| f.size)
            .sum();
        if w == 0 {
            self.size_bytes
        } else {
            w
        }
    }
}

/// The installed-model manifest, persisted as TOML at `$LLMARIO_HOME/registry.toml`.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Registry {
    #[serde(default)]
    pub models: Vec<ModelEntry>,
    #[serde(skip)]
    file: Option<PathBuf>,
}

impl Registry {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let mut reg: Registry = if path.exists() {
            let text = std::fs::read_to_string(path)?;
            toml::from_str(&text).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?
        } else {
            Registry::default()
        };
        reg.file = Some(path.to_path_buf());
        Ok(reg)
    }

    pub fn in_memory(models: Vec<ModelEntry>) -> Self {
        Self { models, file: None }
    }

    /// Atomic save: write a sibling temp file then rename over the manifest.
    pub fn save(&self) -> anyhow::Result<()> {
        let Some(path) = &self.file else {
            return Ok(());
        };
        let text = toml::to_string_pretty(self)?;
        let tmp = path.with_extension(format!("toml.tmp.{}", std::process::id()));
        std::fs::write(&tmp, text)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn get(&self, id: &str) -> Option<&ModelEntry> {
        self.models.iter().find(|m| m.id == id)
    }

    /// Exact id match, else every installed variant of a family.
    pub fn resolve(&self, name: &str) -> Vec<&ModelEntry> {
        if let Some(m) = self.get(name) {
            return vec![m];
        }
        self.models
            .iter()
            .filter(|m| m.family.as_deref() == Some(name))
            .collect()
    }

    pub fn insert(&mut self, entry: ModelEntry) -> anyhow::Result<()> {
        if self.get(&entry.id).is_some() {
            anyhow::bail!("a model named '{}' is already registered", entry.id);
        }
        self.models.push(entry);
        self.models.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(())
    }

    pub fn remove(&mut self, id: &str) -> Option<ModelEntry> {
        let idx = self.models.iter().position(|m| m.id == id)?;
        Some(self.models.remove(idx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn entry(id: &str, family: Option<&str>, format: ModelFormat) -> ModelEntry {
        ModelEntry {
            id: id.into(),
            family: family.map(Into::into),
            format,
            path: "/nonexistent".into(),
            managed: false,
            source: None,
            license: Some("Apache-2.0".into()),
            architecture: Some("qwen3".into()),
            quantization: None,
            shape: None,
            chat_template: true,
            files: vec![FileRecord {
                name: "model.safetensors".into(),
                size: 10,
                sha256: Some("ab".into()),
            }],
            size_bytes: 12,
            added_at: "2026-09-28T00:00:00Z".into(),
        }
    }

    #[test]
    fn save_load_round_trip_and_resolve() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.toml");
        let mut r = Registry::load(&path).unwrap();
        r.insert(entry("q-mlx", Some("q"), ModelFormat::Mlx))
            .unwrap();
        r.insert(entry("q-gguf", Some("q"), ModelFormat::Gguf))
            .unwrap();
        assert!(
            r.insert(entry("q-mlx", None, ModelFormat::Mlx)).is_err(),
            "duplicate ids rejected"
        );
        r.save().unwrap();

        let r2 = Registry::load(&path).unwrap();
        assert_eq!(r2.models.len(), 2);
        assert_eq!(r2.resolve("q-gguf").len(), 1);
        assert_eq!(
            r2.resolve("q").len(),
            2,
            "family name resolves to all variants"
        );
        assert!(r2.resolve("missing").is_empty());
        assert_eq!(r2.get("q-mlx").unwrap(), r.get("q-mlx").unwrap());
    }

    #[test]
    fn content_hash_requires_all_checksums() {
        let mut e = entry("a", None, ModelFormat::Mlx);
        let h = e.content_hash().unwrap();
        assert_eq!(h.len(), 16);
        e.files.push(FileRecord {
            name: "x.json".into(),
            size: 1,
            sha256: None,
        });
        assert!(e.content_hash().is_none());
    }

    #[test]
    fn kv_bytes() {
        let s = ModelShape {
            n_layers: 28,
            n_heads: 16,
            n_kv_heads: 8,
            head_dim: 128,
            hidden_size: 2048,
            context_max: None,
            kv_groups: vec![],
            state_bytes_per_seq: 0,
        };
        assert_eq!(s.kv_bytes_per_token(2), 2 * 28 * 8 * 128 * 2);
        assert_eq!(
            s.conservative_groups()[0].bytes_per_token(2),
            s.kv_bytes_per_token(2)
        );
        // Older registries and catalogs have no layout fields; newer ones round-trip them.
        let old: ModelShape = toml::from_str(
            "n_layers = 2\nn_heads = 2\nn_kv_heads = 1\nhead_dim = 8\nhidden_size = 16",
        )
        .unwrap();
        assert!(old.kv_groups.is_empty() && old.state_bytes_per_seq == 0);
        let mut new = old.clone();
        new.kv_groups = vec![KvGroup {
            layers: 1,
            n_kv_heads: 1,
            head_dim: 8,
            window: Some(1024),
        }];
        new.state_bytes_per_seq = 7;
        let back: ModelShape = toml::from_str(&toml::to_string(&new).unwrap()).unwrap();
        assert_eq!(back, new);
        assert!(!toml::to_string(&old).unwrap().contains("kv_groups"));
    }
}
