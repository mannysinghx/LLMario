//! Inspect a local model (GGUF file or MLX directory) and extract what the planner needs.

use crate::gguf;
use crate::manifest::{FileRecord, ModelShape};
use llmario_core::ModelFormat;
use serde_json::Value;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Inspected {
    pub format: ModelFormat,
    /// File for GGUF (first shard), directory for MLX.
    pub path: PathBuf,
    pub architecture: Option<String>,
    pub quantization: Option<String>,
    pub shape: Option<ModelShape>,
    pub chat_template: bool,
    pub files: Vec<FileRecord>,
    pub size_bytes: u64,
    pub notes: Vec<String>,
}

/// File extensions accepted in an MLX model directory. Python files are deliberately absent:
/// llmario never runs model-repository code.
pub const MLX_ALLOWED_EXT: &[&str] = &["json", "safetensors", "txt", "model", "jinja", "tiktoken"];

pub fn inspect(path: &Path, hash: bool) -> anyhow::Result<Inspected> {
    let path = path
        .canonicalize()
        .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    // GGUF is recognised by extension or by magic bytes (Ollama stores GGUF blobs as
    // `sha256-<hex>` files without an extension).
    if path.is_file()
        && (path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
            || has_gguf_magic(&path))
    {
        inspect_gguf(&path, hash)
    } else if path.is_dir() && path.join("config.json").is_file() {
        inspect_mlx(&path, hash)
    } else if path.is_dir() {
        // A directory holding exactly one GGUF (or one split set) is accepted too.
        let mut ggufs: Vec<PathBuf> = std::fs::read_dir(&path)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
            .collect();
        ggufs.sort();
        match ggufs.first() {
            Some(first) if ggufs.len() == 1 || is_first_shard(first) => inspect_gguf(first, hash),
            Some(_) => anyhow::bail!(
                "{} contains several GGUF files; pass the one to use",
                path.display()
            ),
            None => anyhow::bail!(
                "{} has neither config.json (MLX) nor a .gguf file",
                path.display()
            ),
        }
    } else {
        anyhow::bail!(
            "{} is not a .gguf file or a model directory",
            path.display()
        )
    }
}

fn has_gguf_magic(p: &Path) -> bool {
    use std::io::Read;
    let mut magic = [0u8; 4];
    std::fs::File::open(p)
        .and_then(|mut f| f.read_exact(&mut magic))
        .is_ok()
        && &magic == b"GGUF"
}

fn is_first_shard(p: &Path) -> bool {
    shard_info(p).is_some_and(|(_, idx, _)| idx == 1)
}

/// `name-00001-of-00003.gguf` → ("name", 1, 3)
fn shard_info(p: &Path) -> Option<(String, u32, u32)> {
    let stem = p.file_stem()?.to_str()?;
    let (head, total) = stem.rsplit_once("-of-")?;
    let (base, idx) = head.rsplit_once('-')?;
    Some((base.to_string(), idx.parse().ok()?, total.parse().ok()?))
}

fn file_record(root: &Path, file: &Path, hash: bool) -> anyhow::Result<FileRecord> {
    let meta = std::fs::metadata(file)?; // follows symlinks (HF cache snapshots)
    let name = file
        .strip_prefix(root)
        .unwrap_or(file)
        .to_string_lossy()
        .replace('\\', "/");
    Ok(FileRecord {
        name,
        size: meta.len(),
        sha256: if hash {
            Some(crate::sha256_file(file)?)
        } else {
            None
        },
    })
}

fn inspect_gguf(path: &Path, hash: bool) -> anyhow::Result<Inspected> {
    let md = gguf::read_metadata(path)?;
    let root = path.parent().unwrap_or(Path::new("/")).to_path_buf();
    let mut notes = Vec::new();

    let mut shard_paths = vec![path.to_path_buf()];
    if let Some((base, 1, total)) = shard_info(path) {
        for i in 2..=total {
            let p = root.join(format!("{base}-{i:05}-of-{total:05}.gguf"));
            if !p.is_file() {
                anyhow::bail!("missing GGUF shard {}", p.display());
            }
            shard_paths.push(p);
        }
        notes.push(format!("split GGUF with {total} shards"));
    }
    let files = shard_paths
        .iter()
        .map(|p| file_record(&root, p, hash))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let size_bytes = files.iter().map(|f| f.size).sum();

    let arch = md.architecture().map(String::from);
    let shape = (|| {
        let n_layers = md.arch_u64("block_count")? as u32;
        let n_heads = md.arch_u64("attention.head_count")? as u32;
        let n_kv_heads = md
            .arch_u64("attention.head_count_kv")
            .map(|v| v as u32)
            .unwrap_or(n_heads);
        let hidden_size = md.arch_u64("embedding_length")? as u32;
        let head_dim = md
            .arch_u64("attention.key_length")
            .map(|v| v as u32)
            .unwrap_or(hidden_size / n_heads.max(1));
        Some(ModelShape {
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            hidden_size,
            context_max: md.arch_u64("context_length").map(|v| v as u32),
        })
    })();
    if shape.is_none() {
        notes.push(
            "could not read layer/head counts; memory estimate will use a conservative fallback"
                .into(),
        );
    }
    let quantization = md
        .get("general.file_type")
        .and_then(|v| v.as_u64())
        .map(gguf::file_type_name);
    Ok(Inspected {
        format: ModelFormat::Gguf,
        path: path.to_path_buf(),
        architecture: arch,
        quantization,
        shape,
        chat_template: md.get("tokenizer.chat_template").is_some(),
        files,
        size_bytes,
        notes,
    })
}

fn inspect_mlx(dir: &Path, hash: bool) -> anyhow::Result<Inspected> {
    let cfg: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)
        .map_err(|e| anyhow::anyhow!("config.json: {e}"))?;
    let mut notes = Vec::new();

    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let p = entry?.path();
        if !p.is_file() {
            continue;
        }
        let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext == "py" {
            notes.push(format!(
                "{} contains custom Python code; llmario never executes it (no trust_remote_code)",
                p.file_name().unwrap().to_string_lossy()
            ));
            continue;
        }
        if MLX_ALLOWED_EXT.contains(&ext) {
            files.push(file_record(dir, &p, hash)?);
        }
    }
    files.sort_by(|a, b| a.name.cmp(&b.name));
    if !files.iter().any(|f| f.name.ends_with(".safetensors")) {
        anyhow::bail!("{} has no .safetensors weights", dir.display());
    }
    let size_bytes = files.iter().map(|f| f.size).sum();

    // Multimodal configs nest the language model under `text_config`.
    let tc = cfg
        .get("text_config")
        .filter(|v| v.is_object())
        .unwrap_or(&cfg);
    let num = |k: &str| {
        tc.get(k)
            .or_else(|| cfg.get(k))
            .and_then(Value::as_u64)
            .map(|v| v as u32)
    };
    let shape = (|| {
        let n_layers = num("num_hidden_layers")?;
        let n_heads = num("num_attention_heads")?;
        let hidden_size = num("hidden_size")?;
        Some(ModelShape {
            n_layers,
            n_heads,
            n_kv_heads: num("num_key_value_heads").unwrap_or(n_heads),
            head_dim: num("head_dim").unwrap_or(hidden_size / n_heads.max(1)),
            hidden_size,
            context_max: num("max_position_embeddings"),
        })
    })();
    if shape.is_none() {
        notes.push(
            "config.json lacks layer/head counts; memory estimate will use a conservative fallback"
                .into(),
        );
    }
    let quantization = cfg
        .get("quantization")
        .or_else(|| cfg.get("quantization_config"))
        .and_then(|q| {
            let bits = q.get("bits")?.as_u64()?;
            let gs = q.get("group_size").and_then(Value::as_u64);
            Some(match gs {
                Some(g) => format!("{bits}-bit (group {g})"),
                None => format!("{bits}-bit"),
            })
        })
        .or_else(|| {
            cfg.get("torch_dtype")
                .and_then(Value::as_str)
                .map(String::from)
        });

    let chat_template = dir.join("chat_template.jinja").is_file()
        || dir.join("chat_template.json").is_file()
        || std::fs::read_to_string(dir.join("tokenizer_config.json"))
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
            .is_some_and(|v| v.get("chat_template").is_some());

    Ok(Inspected {
        format: ModelFormat::Mlx,
        path: dir.to_path_buf(),
        architecture: cfg
            .get("model_type")
            .and_then(Value::as_str)
            .map(String::from),
        quantization,
        shape,
        chat_template,
        files,
        size_bytes,
        notes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inspects_mlx_dir() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(
            d.path().join("config.json"),
            r#"{"model_type":"qwen3","num_hidden_layers":28,"num_attention_heads":16,
                "num_key_value_heads":8,"hidden_size":2048,"head_dim":128,
                "max_position_embeddings":40960,"quantization":{"bits":4,"group_size":64}}"#,
        )
        .unwrap();
        std::fs::write(d.path().join("model.safetensors"), vec![0u8; 100]).unwrap();
        std::fs::write(
            d.path().join("tokenizer_config.json"),
            r#"{"chat_template":"x"}"#,
        )
        .unwrap();
        std::fs::write(d.path().join("modeling_evil.py"), "import os").unwrap();

        let i = inspect(d.path(), true).unwrap();
        assert_eq!(i.format, ModelFormat::Mlx);
        assert_eq!(i.architecture.as_deref(), Some("qwen3"));
        assert_eq!(i.quantization.as_deref(), Some("4-bit (group 64)"));
        assert_eq!(i.shape.as_ref().unwrap().n_kv_heads, 8);
        assert!(i.chat_template);
        assert!(
            !i.files.iter().any(|f| f.name.ends_with(".py")),
            "python never recorded"
        );
        assert!(i.notes.iter().any(|n| n.contains("custom Python")));
        assert!(i.files.iter().all(|f| f.sha256.is_some()));
    }

    #[test]
    fn inspects_gguf_file() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("m.gguf");
        std::fs::write(&p, crate::gguf::tests::sample_gguf()).unwrap();
        let i = inspect(&p, false).unwrap();
        assert_eq!(i.format, ModelFormat::Gguf);
        assert_eq!(i.quantization.as_deref(), Some("Q4_K_M"));
        let s = i.shape.unwrap();
        assert_eq!(
            (s.n_layers, s.n_heads, s.n_kv_heads, s.head_dim),
            (28, 24, 8, 128)
        );
        assert!(i.chat_template);
    }

    #[test]
    fn gguf_without_extension_by_magic() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("sha256-abcdef");
        std::fs::write(&p, crate::gguf::tests::sample_gguf()).unwrap();
        assert_eq!(inspect(&p, false).unwrap().format, ModelFormat::Gguf);
        std::fs::write(&p, b"not a model").unwrap();
        assert!(inspect(&p, false).is_err());
    }

    #[test]
    fn shard_names() {
        assert_eq!(
            shard_info(Path::new("/m/Big-Q4_K_M-00001-of-00003.gguf")),
            Some(("Big-Q4_K_M".into(), 1, 3))
        );
        assert_eq!(shard_info(Path::new("single.gguf")), None);
    }

    #[test]
    fn rejects_non_models() {
        let d = tempfile::tempdir().unwrap();
        assert!(inspect(d.path(), false).is_err());
    }
}
