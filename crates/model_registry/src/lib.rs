//! Local model registry: manifest of installed models, the built-in catalog of pullable
//! models, metadata inspection (GGUF header / MLX `config.json`), and checksum-verified
//! downloads from the Hugging Face Hub. Model weights are never stored in this repository.

pub mod catalog;
pub mod download;
pub mod gguf;
pub mod inspect;
pub mod layout;
pub mod manifest;

pub use catalog::{Catalog, CatalogEntry};
pub use manifest::{FileRecord, KvGroup, ModelEntry, ModelShape, ModelSource, Registry};

use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;

/// Streaming SHA-256 of a file.
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex::encode(h.finalize()))
}

/// Reject remote file names that could escape the destination directory.
pub fn validate_relative_name(name: &str) -> anyhow::Result<()> {
    let bad = name.is_empty()
        || name.starts_with('/')
        || name.contains('\\')
        || name.contains('\0')
        || name
            .split('/')
            .any(|seg| seg.is_empty() || seg == "." || seg == "..")
        || Path::new(name).is_absolute();
    if bad {
        anyhow::bail!("unsafe file name in model repository: {name:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_path_traversal() {
        for bad in [
            "../x",
            "a/../b",
            "/etc/passwd",
            "a//b",
            "a\\b",
            "",
            "./a",
            "a/",
        ] {
            assert!(
                validate_relative_name(bad).is_err(),
                "{bad:?} should be rejected"
            );
        }
        for ok in [
            "model.safetensors",
            "sub/dir/file.json",
            "Qwen3-1.7B-Q4_K_M.gguf",
        ] {
            validate_relative_name(ok).unwrap();
        }
    }
}
