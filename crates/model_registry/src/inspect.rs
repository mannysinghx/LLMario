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

/// macOS metadata files that are never part of a model: AppleDouble sidecars ("._name", written
/// next to files on exFAT/FAT drives) and Finder's ".DS_Store".
pub fn is_os_sidecar(name: &str) -> bool {
    name.starts_with("._") || name == ".DS_Store"
}

fn sidecar(p: &Path) -> bool {
    p.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(is_os_sidecar)
}

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
            .filter(|p| p.extension().is_some_and(|e| e == "gguf") && !sidecar(p))
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
            kv_groups: vec![],
            state_bytes_per_seq: 0,
            mtp_layers: 0,
            bytes_per_token: 0,
            expert_bytes: 0,
            active_expert_bytes: 0,
        })
    })()
    .map(|mut s| {
        (s.kv_groups, s.state_bytes_per_seq) =
            crate::layout::from_gguf(&md, s.n_layers, s.n_kv_heads, s.head_dim);
        s.mtp_layers = md.arch_u64("nextn_predict_layers").unwrap_or(0) as u32;
        // Split GGUFs keep each shard's tensors in that shard's own header; only whole files.
        if files.len() == 1 {
            s.bytes_per_token = md.bytes_per_token(files[0].size).unwrap_or(0);
            (s.expert_bytes, s.active_expert_bytes) =
                md.expert_bytes(files[0].size).unwrap_or((0, 0));
        }
        s
    });
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

/// Largest safetensors header read (the JSON index at the start of each file).
const MAX_SAFETENSORS_HEADER: u64 = 100 * 1024 * 1024;

/// Tensor names and sizes from a safetensors file's header (never its data).
pub fn safetensors_sizes(path: &Path) -> anyhow::Result<Vec<(String, u64)>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let len = u64::from_le_bytes(len);
    anyhow::ensure!(
        len <= MAX_SAFETENSORS_HEADER,
        "safetensors header of {len} bytes"
    );
    let mut buf = vec![0u8; len as usize];
    f.read_exact(&mut buf)?;
    let header: serde_json::Map<String, Value> = serde_json::from_slice(&buf)?;
    Ok(header
        .into_iter()
        .filter(|(k, _)| k != "__metadata__")
        .filter_map(|(k, v)| {
            let o = v.get("data_offsets")?.as_array()?;
            Some((k, o.get(1)?.as_u64()?.checked_sub(o.first()?.as_u64()?)?))
        })
        .collect())
}

/// `(weight bytes read per token, total expert bytes, expert bytes read per token)` for an MLX
/// model directory, from its
/// safetensors headers: experts (`switch_mlp` / `experts`) at the share used per token, the
/// embedding table left out when a separate `lm_head` exists, and vision/audio towers left out
/// (not loaded for text).
fn mlx_bytes_per_token(
    dir: &Path,
    files: &[FileRecord],
    tc: &Value,
    cfg: &Value,
) -> Option<(u64, u64, u64)> {
    let num = |k: &str| tc.get(k).or_else(|| cfg.get(k)).and_then(Value::as_u64);
    // Experts used per token: Qwen/Mixtral-style and Gemma-style config keys.
    let used = num("num_experts_per_tok")
        .or_else(|| num("top_k_experts"))
        .or_else(|| num("moe_top_k"))
        .or_else(|| num("num_experts_per_token"));
    let count = num("num_local_experts")
        .or_else(|| num("num_experts"))
        .or_else(|| num("n_routed_experts"))
        .filter(|c| *c > 0);
    let mut tensors = Vec::new();
    for f in files.iter().filter(|f| f.name.ends_with(".safetensors")) {
        tensors.extend(safetensors_sizes(&dir.join(&f.name)).ok()?);
    }
    let skip = |n: &str| {
        [
            "vision",
            "visual",
            "audio",
            "multi_modal_projector",
            "embed_vision",
            "embed_audio",
        ]
        .iter()
        .any(|k| n.contains(k))
    };
    let has_head = tensors.iter().any(|(n, _)| n.contains("lm_head."));
    let (mut total, mut experts, mut active) = (0u64, 0u64, 0u64);
    for (name, size) in &tensors {
        if skip(name) || (has_head && name.contains("embed_tokens.")) {
            continue;
        }
        let expert = name.contains(".switch_mlp.") || name.contains(".experts.");
        let read = match (expert, used, count) {
            (true, Some(u), Some(c)) => (*size as u128 * u.min(c) as u128 / c as u128) as u64,
            _ => *size,
        };
        if expert {
            experts += size;
            active += read;
        }
        total += read;
    }
    (total > 0).then_some((total, experts, active))
}

fn inspect_mlx(dir: &Path, hash: bool) -> anyhow::Result<Inspected> {
    let cfg: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)
        .map_err(|e| anyhow::anyhow!("config.json: {e}"))?;
    let mut notes = Vec::new();

    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let p = entry?.path();
        if !p.is_file() || sidecar(&p) {
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
            kv_groups: vec![],
            state_bytes_per_seq: 0,
            mtp_layers: 0,
            bytes_per_token: 0,
            expert_bytes: 0,
            active_expert_bytes: 0,
        })
    })()
    .map(|mut s| {
        (s.kv_groups, s.state_bytes_per_seq) = crate::layout::from_config(tc);
        if let Some((per_token, experts, active)) = mlx_bytes_per_token(dir, &files, tc, &cfg) {
            s.bytes_per_token = per_token;
            s.expert_bytes = experts;
            s.active_expert_bytes = active;
        }
        s
    });
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

    /// MLX bytes read per token from safetensors headers (experts at the used share, embedding
    /// skipped next to `lm_head`, vision tower skipped); macOS sidecars are never model files.
    #[test]
    fn mlx_bytes_per_token_and_sidecars() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(
            d.path().join("config.json"),
            r#"{"model_type":"qwen3_moe","num_hidden_layers":1,"num_attention_heads":4,
                "num_key_value_heads":2,"hidden_size":64,"head_dim":16,
                "num_experts_per_tok":2,"num_experts":8}"#,
        )
        .unwrap();
        let tensors = [
            ("model.embed_tokens.weight", 1000u64),
            ("lm_head.weight", 1000),
            ("model.layers.0.self_attn.q_proj.weight", 400),
            ("model.layers.0.mlp.switch_mlp.up_proj.weight", 3200),
            ("vision_tower.patch.weight", 400),
        ];
        let mut header = serde_json::Map::new();
        let mut off = 0u64;
        for (name, size) in tensors {
            header.insert(
                name.into(),
                serde_json::json!({"dtype": "U8", "shape": [size], "data_offsets": [off, off + size]}),
            );
            off += size;
        }
        let h = serde_json::to_vec(&header).unwrap();
        let mut file = (h.len() as u64).to_le_bytes().to_vec();
        file.extend(&h);
        file.extend(vec![0u8; off as usize]);
        std::fs::write(d.path().join("model.safetensors"), file).unwrap();
        // AppleDouble sidecars, as macOS writes them on exFAT drives: not valid model files.
        std::fs::write(d.path().join("._model.safetensors"), [0u8, 5, 22, 7, 0xb0]).unwrap();
        std::fs::write(d.path().join("._config.json"), [0u8, 5, 22, 7]).unwrap();
        std::fs::write(d.path().join(".DS_Store"), [0u8; 8]).unwrap();
        let i = inspect(d.path(), false).unwrap();
        assert!(
            i.files.iter().all(|f| !is_os_sidecar(&f.name)),
            "{:?}",
            i.files
        );
        // 1000 (lm_head) + 400 (attention) + 3200 × 2/8 (experts); embedding and vision skipped.
        let s = i.shape.unwrap();
        assert_eq!(s.bytes_per_token, 2200);
        assert_eq!((s.expert_bytes, s.active_expert_bytes), (3200, 800));
        assert!(
            is_os_sidecar("._x.gguf") && is_os_sidecar(".DS_Store") && !is_os_sidecar("model.gguf")
        );
    }

    /// macOS sidecars on exFAT drives ("._name", ".DS_Store") are never model files.
    #[test]
    fn mlx_dir_ignores_os_sidecars() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(
            d.path().join("config.json"),
            r#"{"model_type":"qwen3","num_hidden_layers":2,"num_attention_heads":4,
                "num_key_value_heads":2,"hidden_size":64,"head_dim":16}"#,
        )
        .unwrap();
        std::fs::write(d.path().join("model.safetensors"), vec![0u8; 100]).unwrap();
        std::fs::write(d.path().join("._model.safetensors"), [0u8, 5, 22, 7, 0xb0]).unwrap();
        std::fs::write(d.path().join("._config.json"), [0u8, 5, 22, 7]).unwrap();
        std::fs::write(d.path().join(".DS_Store"), [0u8; 8]).unwrap();
        let i = inspect(d.path(), true).unwrap();
        let names: Vec<&str> = i.files.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["config.json", "model.safetensors"]);
        assert!(
            is_os_sidecar("._x.gguf") && is_os_sidecar(".DS_Store") && !is_os_sidecar("model.gguf")
        );
    }

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
