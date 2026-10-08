//! Checksum-verified model download from the Hugging Face Hub.
//!
//! Safety properties:
//! - HTTPS only (plain HTTP allowed solely for a loopback test endpoint).
//! - Every file is verified before it becomes visible: LFS files against the Hub's SHA-256,
//!   small files against their git blob SHA-1.
//! - Remote file names are validated against path traversal.
//! - Files are staged in a private directory and renamed into place atomically.
//! - Free disk space is checked up front.
//! - Blobs already present in the local Hugging Face cache (same hash) are hard-linked or
//!   copied instead of downloaded, then verified like any download.

use crate::catalog::CatalogEntry;
use crate::inspect::{self, MLX_ALLOWED_EXT};
use crate::manifest::{ModelEntry, ModelSource, Registry};
use crate::{sha256_file, validate_relative_name};
use futures::StreamExt;
use llmario_core::{ModelFormat, Paths};
use serde::Deserialize;
use sha1::Sha1;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

#[derive(Deserialize, Debug, Clone)]
pub struct Sibling {
    pub rfilename: String,
    #[serde(rename = "blobId")]
    pub blob_id: Option<String>,
    pub size: Option<u64>,
    pub lfs: Option<LfsInfo>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct LfsInfo {
    pub sha256: String,
    pub size: u64,
}

#[derive(Deserialize, Debug)]
pub struct RepoInfo {
    pub sha: String,
    pub siblings: Vec<Sibling>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expected {
    Sha256(String),
    GitSha1(String),
}

impl Sibling {
    pub fn expected(&self) -> anyhow::Result<Expected> {
        if let Some(l) = &self.lfs {
            return Ok(Expected::Sha256(l.sha256.to_ascii_lowercase()));
        }
        match &self.blob_id {
            Some(b) => Ok(Expected::GitSha1(b.to_ascii_lowercase())),
            None => anyhow::bail!(
                "{}: Hub returned no checksum; refusing unverifiable file",
                self.rfilename
            ),
        }
    }
    pub fn expected_size(&self) -> Option<u64> {
        self.lfs.as_ref().map(|l| l.size).or(self.size)
    }
}

pub struct PullOptions {
    pub force: bool,
    /// Terminal progress bars (CLI).
    pub show_progress: bool,
    pub use_hf_cache: bool,
    /// Progress callback for GUIs, called at most every ~100 ms per file and once per finished file.
    pub on_progress: Option<ProgressFn>,
}

/// Download progress across all files of one pull.
#[derive(serde::Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct PullProgress {
    pub file: String,
    pub done_bytes: u64,
    pub total_bytes: u64,
    /// True when the file came from the local Hugging Face cache instead of the network.
    pub reused: bool,
}

pub type ProgressFn = std::sync::Arc<dyn Fn(PullProgress) + Send + Sync>;

pub struct PullOutcome {
    pub entry: ModelEntry,
    pub downloaded_bytes: u64,
    pub reused_bytes: u64,
}

pub struct HubClient {
    http: reqwest::Client,
    endpoint: String,
    token: Option<String>,
}

impl HubClient {
    pub fn from_env() -> anyhow::Result<Self> {
        let endpoint =
            std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into());
        Self::new(
            &endpoint,
            std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty()),
        )
    }

    pub fn new(endpoint: &str, token: Option<String>) -> anyhow::Result<Self> {
        let endpoint = endpoint.trim_end_matches('/').to_string();
        let loopback_http =
            endpoint.starts_with("http://127.0.0.1") || endpoint.starts_with("http://localhost");
        if !endpoint.starts_with("https://") && !loopback_http {
            anyhow::bail!("refusing non-HTTPS model endpoint {endpoint}");
        }
        let http = reqwest::Client::builder()
            .user_agent(llmario_core::user_agent())
            .connect_timeout(std::time::Duration::from_secs(30))
            .read_timeout(std::time::Duration::from_secs(120))
            .https_only(!loopback_http)
            .build()?;
        Ok(Self {
            http,
            endpoint,
            token,
        })
    }

    fn get(&self, url: &str) -> reqwest::RequestBuilder {
        let rb = self.http.get(url);
        match &self.token {
            Some(t) => rb.bearer_auth(t),
            None => rb,
        }
    }

    pub async fn repo_info(&self, repo: &str, revision: &str) -> anyhow::Result<RepoInfo> {
        let url = format!(
            "{}/api/models/{repo}/revision/{revision}?blobs=true",
            self.endpoint
        );
        let resp = self.get(&url).send().await?;
        let status = resp.status();
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            anyhow::bail!("{repo} is gated or private (HTTP {status}); accept its license on huggingface.co and set HF_TOKEN");
        }
        if !status.is_success() {
            anyhow::bail!("Hub API {url} returned HTTP {status}");
        }
        Ok(resp.json().await?)
    }

    /// Stream `repo@commit/file` into `dest`, hashing while writing. Returns sha256 hex.
    async fn download(
        &self,
        repo: &str,
        commit: &str,
        sib: &Sibling,
        dest: &Path,
        pb: &indicatif::ProgressBar,
        on_bytes: &(dyn Fn(u64) + Send + Sync),
    ) -> anyhow::Result<String> {
        let url = format!(
            "{}/{repo}/resolve/{commit}/{}",
            self.endpoint, sib.rfilename
        );
        let resp = self.get(&url).send().await?;
        if !resp.status().is_success() {
            anyhow::bail!("GET {url}: HTTP {}", resp.status());
        }
        let mut file = tokio::fs::File::create(dest).await?;
        let mut sha256 = Sha256::new();
        let mut sha1 = Sha1::new();
        let size = sib.expected_size().unwrap_or(0);
        sha1.update(format!("blob {size}\0").as_bytes());
        let mut written = 0u64;
        let mut last_report = std::time::Instant::now();
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            sha256.update(&chunk);
            sha1.update(&chunk);
            file.write_all(&chunk).await?;
            written += chunk.len() as u64;
            pb.set_position(written);
            if last_report.elapsed() >= std::time::Duration::from_millis(100) {
                on_bytes(written);
                last_report = std::time::Instant::now();
            }
        }
        on_bytes(written);
        file.flush().await?;
        file.sync_all().await?;
        let got256 = hex::encode(sha256.finalize());
        let got1 = hex::encode(sha1.finalize());
        check(sib, written, &got256, &got1)?;
        Ok(got256)
    }
}

fn check(sib: &Sibling, size: u64, sha256: &str, git_sha1: &str) -> anyhow::Result<()> {
    if let Some(exp) = sib.expected_size() {
        if exp != size {
            anyhow::bail!(
                "{}: size mismatch (expected {exp}, got {size})",
                sib.rfilename
            );
        }
    }
    match sib.expected()? {
        Expected::Sha256(e) if e != sha256 => anyhow::bail!("{}: SHA-256 mismatch", sib.rfilename),
        Expected::GitSha1(e) if e != git_sha1 => {
            anyhow::bail!("{}: git blob SHA-1 mismatch", sib.rfilename)
        }
        _ => Ok(()),
    }
}

/// Hash an existing file: SHA-256 always (recorded in the manifest), git blob SHA-1 only
/// when `with_sha1` (small non-LFS files are verified against it).
fn hash_local(path: &Path, with_sha1: bool) -> anyhow::Result<(u64, String, String)> {
    let size = std::fs::metadata(path)?.len();
    let mut f = std::fs::File::open(path)?;
    let mut s256 = Sha256::new();
    let mut s1 = with_sha1.then(Sha1::new);
    if let Some(s1) = s1.as_mut() {
        s1.update(format!("blob {size}\0").as_bytes());
    }
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        s256.update(&buf[..n]);
        if let Some(s1) = s1.as_mut() {
            s1.update(&buf[..n]);
        }
    }
    Ok((
        size,
        hex::encode(s256.finalize()),
        s1.map(|s| hex::encode(s.finalize())).unwrap_or_default(),
    ))
}

/// Which repository files to fetch for a catalog entry.
pub fn select_files(entry: &CatalogEntry, siblings: &[Sibling]) -> anyhow::Result<Vec<Sibling>> {
    let chosen: Vec<Sibling> = if entry.files.is_empty() {
        siblings
            .iter()
            .filter(|s| {
                !s.rfilename.contains('/')
                    && Path::new(&s.rfilename)
                        .extension()
                        .and_then(|e| e.to_str())
                        .is_some_and(|e| MLX_ALLOWED_EXT.contains(&e))
            })
            .cloned()
            .collect()
    } else {
        entry
            .files
            .iter()
            .map(|f| {
                siblings
                    .iter()
                    .find(|s| &s.rfilename == f)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("{} not found in {}", f, entry.repo))
            })
            .collect::<anyhow::Result<_>>()?
    };
    if chosen.is_empty() {
        anyhow::bail!("no downloadable files selected from {}", entry.repo);
    }
    for s in &chosen {
        validate_relative_name(&s.rfilename)?;
    }
    Ok(chosen)
}

pub fn hf_cache_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("HF_HUB_CACHE") {
        return Some(PathBuf::from(p));
    }
    if let Ok(p) = std::env::var("HF_HOME") {
        return Some(PathBuf::from(p).join("hub"));
    }
    dirs::home_dir().map(|h| h.join(".cache/huggingface/hub"))
}

fn cached_blob(repo: &str, sib: &Sibling) -> Option<PathBuf> {
    let name = match sib.expected().ok()? {
        Expected::Sha256(h) | Expected::GitSha1(h) => h,
    };
    let p = hf_cache_dir()?
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("blobs")
        .join(name);
    p.is_file().then_some(p)
}

pub use llmario_core::os::free_space_bytes;

/// Pull a catalog model into the managed models directory and register it.
pub async fn pull(
    hub: &HubClient,
    entry: &CatalogEntry,
    paths: &Paths,
    registry: &mut Registry,
    opts: &PullOptions,
) -> anyhow::Result<PullOutcome> {
    if let Some(existing) = registry.get(&entry.id) {
        if !opts.force {
            anyhow::bail!(
                "'{}' is already installed at {} (use --force to re-download)",
                entry.id,
                existing.path.display()
            );
        }
        if !existing.managed {
            anyhow::bail!(
                "'{}' is registered from a user path; `model remove` it first",
                entry.id
            );
        }
    }

    let info = hub.repo_info(&entry.repo, &entry.revision).await?;
    let files = select_files(entry, &info.siblings)?;
    let total: u64 = files.iter().filter_map(Sibling::expected_size).sum();

    let models_dir = paths.models_dir();
    std::fs::create_dir_all(&models_dir)?;
    if let Some(free) = free_space_bytes(&models_dir) {
        let need = total + 512 * 1024 * 1024;
        if free < need {
            anyhow::bail!(
                "not enough disk space: need {:.2} GiB, {:.2} GiB free in {}",
                need as f64 / GIB,
                free as f64 / GIB,
                models_dir.display()
            );
        }
    }

    let staging = models_dir.join(format!(".staging-{}-{}", entry.id, std::process::id()));
    if staging.exists() {
        std::fs::remove_dir_all(&staging)?;
    }
    std::fs::create_dir_all(&staging)?;
    let result = fetch_all(hub, entry, &info.sha, &files, &staging, opts).await;
    let (hashes, downloaded, reused) = match result {
        Ok(v) => v,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&staging);
            return Err(e);
        }
    };

    let final_dir = models_dir.join(&entry.id);
    if final_dir.exists() {
        // Only ever delete inside our own models directory.
        anyhow::ensure!(
            final_dir.starts_with(&models_dir),
            "refusing to replace {}",
            final_dir.display()
        );
        std::fs::remove_dir_all(&final_dir)?;
    }
    std::fs::rename(&staging, &final_dir)?;

    let model_path = match entry.format {
        ModelFormat::Gguf => final_dir.join(&files[0].rfilename),
        _ => final_dir.clone(),
    };
    let mut ins = inspect::inspect(&model_path, false)?;
    for f in ins.files.iter_mut() {
        f.sha256 = hashes.get(&f.name).cloned();
    }
    let model = ModelEntry {
        id: entry.id.clone(),
        family: Some(entry.family.clone()),
        format: entry.format,
        path: ins.path,
        managed: true,
        source: Some(ModelSource {
            repo: entry.repo.clone(),
            revision: info.sha.clone(),
        }),
        license: Some(entry.license.clone()),
        architecture: ins.architecture,
        quantization: ins.quantization,
        shape: ins.shape,
        chat_template: ins.chat_template,
        files: ins.files,
        size_bytes: ins.size_bytes,
        added_at: chrono::Utc::now().to_rfc3339(),
    };
    registry.remove(&entry.id);
    registry.insert(model.clone())?;
    registry.save()?;
    Ok(PullOutcome {
        entry: model,
        downloaded_bytes: downloaded,
        reused_bytes: reused,
    })
}

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

async fn fetch_all(
    hub: &HubClient,
    entry: &CatalogEntry,
    commit: &str,
    files: &[Sibling],
    staging: &Path,
    opts: &PullOptions,
) -> anyhow::Result<(HashMap<String, String>, u64, u64)> {
    let mut hashes = HashMap::new();
    let (mut downloaded, mut reused) = (0u64, 0u64);
    let total: u64 = files.iter().filter_map(Sibling::expected_size).sum();
    let report = |file: &str, done: u64, from_cache: bool| {
        if let Some(cb) = &opts.on_progress {
            cb(PullProgress {
                file: file.to_string(),
                done_bytes: done,
                total_bytes: total,
                reused: from_cache,
            });
        }
    };
    for sib in files {
        let dest = staging.join(&sib.rfilename);
        // Some repos keep large models in quantization subfolders (e.g. `Q4_K_M/…-00001-of-00003.gguf`).
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let size = sib.expected_size().unwrap_or(0);

        if opts.use_hf_cache {
            if let Some(blob) = cached_blob(&entry.repo, sib) {
                if std::fs::hard_link(&blob, &dest).is_err() {
                    std::fs::copy(&blob, &dest)?;
                }
                let (sz, s256, s1) = hash_local(&dest, sib.lfs.is_none())?;
                if check(sib, sz, &s256, &s1).is_ok() {
                    tracing::info!(file = %sib.rfilename, "reused verified blob from Hugging Face cache");
                    if opts.show_progress {
                        eprintln!("  ✓ {} (from local HF cache, verified)", sib.rfilename);
                    }
                    hashes.insert(sib.rfilename.clone(), s256);
                    reused += sz;
                    report(&sib.rfilename, downloaded + reused, true);
                    continue;
                }
                std::fs::remove_file(&dest)?;
            }
        }

        let pb = if opts.show_progress {
            let pb = indicatif::ProgressBar::new(size);
            pb.set_style(
                indicatif::ProgressStyle::with_template(
                    "  {msg:40!} {bar:30} {bytes}/{total_bytes} {bytes_per_sec} eta {eta}",
                )
                .unwrap(),
            );
            pb.set_message(sib.rfilename.clone());
            pb
        } else {
            indicatif::ProgressBar::hidden()
        };
        let mut last_err = None;
        for attempt in 1..=3 {
            let before = downloaded + reused;
            let on_bytes = |n: u64| report(&sib.rfilename, before + n, false);
            match hub
                .download(&entry.repo, commit, sib, &dest, &pb, &on_bytes)
                .await
            {
                Ok(h) => {
                    hashes.insert(sib.rfilename.clone(), h);
                    last_err = None;
                    break;
                }
                Err(e) => {
                    tracing::warn!(file = %sib.rfilename, attempt, error = %e, "download failed");
                    let _ = std::fs::remove_file(&dest);
                    last_err = Some(e);
                }
            }
        }
        pb.finish();
        if let Some(e) = last_err {
            return Err(e);
        }
        downloaded += size;
    }
    Ok((hashes, downloaded, reused))
}

/// Register a model the user dropped onto the app (or pointed at) with an automatic id.
/// Refuses a path that is already registered. Files are hashed in place, never copied.
pub fn add_local_auto(path: &Path, registry: &mut Registry) -> anyhow::Result<ModelEntry> {
    let probe = inspect::inspect(path, false)?;
    if let Some(m) = registry.models.iter().find(|m| m.path == probe.path) {
        anyhow::bail!("this model is already added as '{}'", m.id);
    }
    let id = unique_id(registry, &suggested_id(&probe));
    add_local(path, Some(&id), registry)
}

/// A readable id: the file/folder name, or the GGUF `general.name` when the file is named by a
/// hash (e.g. `sha256-…` blobs from other runtimes' caches), plus the quantization if missing.
fn suggested_id(ins: &inspect::Inspected) -> String {
    let stem = default_id(&ins.path);
    let hashy = stem.starts_with("sha256")
        || (stem.len() >= 32 && stem.chars().all(|c| c.is_ascii_hexdigit() || c == '-'));
    let mut base = stem.clone();
    if hashy && ins.format == ModelFormat::Gguf {
        if let Some(name) = crate::gguf::read_metadata(&ins.path).ok().and_then(|m| {
            m.get("general.name")
                .and_then(|v| v.as_str())
                .map(String::from)
        }) {
            base = name;
        }
    }
    let mut id = slug(&base);
    if let Some(q) = ins.quantization.as_deref().map(slug) {
        if !q.is_empty() && !id.contains(&q) && q.len() <= 16 {
            id = format!("{id}-{q}");
        }
    }
    id
}

/// Lowercase id with only `[a-z0-9._-]`, at most 64 characters.
pub fn slug(s: &str) -> String {
    let mut out = String::new();
    for c in s.trim().to_lowercase().chars() {
        let c = if c.is_ascii_alphanumeric() || "._".contains(c) {
            c
        } else {
            '-'
        };
        if !(c == '-' && out.ends_with('-')) {
            out.push(c);
        }
    }
    let out: String = out.trim_matches('-').chars().take(64).collect();
    if out.is_empty() {
        "model".into()
    } else {
        out
    }
}

/// `base`, or `base-2`, `base-3`… so the id is new and never shadows a family name (which would
/// hijack requests for that family).
pub fn unique_id(registry: &Registry, base: &str) -> String {
    let catalog = crate::Catalog::builtin();
    let taken = |id: &str| {
        registry.get(id).is_some()
            || registry
                .models
                .iter()
                .any(|m| m.family.as_deref() == Some(id))
            || !catalog.family(id).is_empty()
    };
    if !taken(base) {
        return base.to_string();
    }
    (2..)
        .map(|n| format!("{base}-{n}"))
        .find(|c| !taken(c))
        .expect("unbounded")
}

/// Register a model that already exists on disk. Files are hashed but not copied; `remove`
/// will never delete them.
pub fn add_local(
    path: &Path,
    id: Option<&str>,
    registry: &mut Registry,
) -> anyhow::Result<ModelEntry> {
    let ins = inspect::inspect(path, true)?;
    let id = match id {
        Some(i) => i.to_string(),
        None => default_id(&ins.path),
    };
    anyhow::ensure!(
        !id.is_empty()
            && id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "-_.:".contains(c)),
        "model id may only contain letters, digits, '-', '_', '.', ':'"
    );
    let entry = ModelEntry {
        id,
        family: None,
        format: ins.format,
        path: ins.path,
        managed: false,
        source: None,
        license: None,
        architecture: ins.architecture,
        quantization: ins.quantization,
        shape: ins.shape,
        chat_template: ins.chat_template,
        files: ins.files,
        size_bytes: ins.size_bytes,
        added_at: chrono::Utc::now().to_rfc3339(),
    };
    for n in &ins.notes {
        tracing::warn!("{n}");
    }
    registry.insert(entry.clone())?;
    registry.save()?;
    Ok(entry)
}

fn default_id(path: &Path) -> String {
    // HF cache snapshot: .../models--org--name/snapshots/<sha>  → name
    let comps: Vec<String> = path
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    if let Some(i) = comps.iter().position(|c| c == "snapshots") {
        if i > 0 {
            if let Some(name) = comps[i - 1].rsplit("--").next() {
                return name.to_ascii_lowercase();
            }
        }
    }
    path.file_stem()
        .map(|s| s.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default()
}

/// Delete a registered model. Managed files are removed; user paths are only unregistered.
pub fn remove(id: &str, paths: &Paths, registry: &mut Registry) -> anyhow::Result<ModelEntry> {
    let entry = registry
        .get(id)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("model '{id}' is not registered"))?;
    if entry.managed {
        let dir = paths.models_dir().join(&entry.id);
        if dir.exists() {
            // Compare real locations: the models folder may be a link to another disk, and
            // recorded paths are canonical. `models/<id>` must lead to a folder inside it.
            let real = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
            let (root, real_dir) = (real(&paths.models_dir()), real(&dir));
            anyhow::ensure!(
                real_dir.starts_with(&root)
                    && real_dir != root
                    && real(&entry.path).starts_with(&real_dir),
                "refusing to delete {}: not inside the managed models dir ({})",
                entry.path.display(),
                root.display()
            );
            std::fs::remove_dir_all(&dir)?;
        }
        // Nothing on disk (already deleted, or its drive is disconnected): only unregister.
    }
    registry.remove(id);
    registry.save()?;
    Ok(entry)
}

/// Verify an installed model's files against the recorded SHA-256s.
pub fn verify(entry: &ModelEntry) -> anyhow::Result<Vec<String>> {
    let root = match entry.format {
        ModelFormat::Gguf => entry.path.parent().unwrap_or(Path::new("/")).to_path_buf(),
        _ => entry.path.clone(),
    };
    let mut problems = Vec::new();
    for f in &entry.files {
        let p = root.join(&f.name);
        match (&f.sha256, sha256_file(&p)) {
            (_, Err(e)) => problems.push(format!("{}: {e}", f.name)),
            (Some(exp), Ok(got)) if *exp != got => {
                problems.push(format!("{}: checksum mismatch", f.name))
            }
            (None, Ok(_)) => problems.push(format!("{}: no recorded checksum", f.name)),
            _ => {}
        }
    }
    Ok(problems)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sib(name: &str, lfs: Option<&str>, blob: Option<&str>, size: u64) -> Sibling {
        Sibling {
            rfilename: name.into(),
            blob_id: blob.map(Into::into),
            size: Some(size),
            lfs: lfs.map(|h| LfsInfo {
                sha256: h.into(),
                size,
            }),
        }
    }

    #[test]
    fn git_blob_sha1_matches_git() {
        // `printf 'hello\n' | git hash-object --stdin`
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("f");
        std::fs::write(&p, "hello\n").unwrap();
        let (_, _, s1) = hash_local(&p, true).unwrap();
        assert_eq!(s1, "ce013625030ba8dba906f756967f9e9ca394464a");
    }

    #[test]
    fn check_detects_mismatches() {
        let s = sib(
            "a.json",
            None,
            Some("ce013625030ba8dba906f756967f9e9ca394464a"),
            6,
        );
        check(&s, 6, "x", "ce013625030ba8dba906f756967f9e9ca394464a").unwrap();
        assert!(check(&s, 6, "x", "deadbeef").is_err());
        assert!(check(&s, 7, "x", "ce013625030ba8dba906f756967f9e9ca394464a").is_err());
        let l = sib("w.safetensors", Some("abc"), Some("zzz"), 3);
        check(&l, 3, "abc", "ignored").unwrap();
        assert!(check(&l, 3, "abd", "zzz").is_err());
        let none = Sibling {
            rfilename: "x".into(),
            blob_id: None,
            size: None,
            lfs: None,
        };
        assert!(none.expected().is_err(), "unverifiable files are refused");
    }

    #[test]
    fn selects_mlx_allowlist_and_pinned_gguf() {
        let sibs = vec![
            sib("README.md", None, Some("1"), 1),
            sib(".gitattributes", None, Some("2"), 1),
            sib("config.json", None, Some("3"), 1),
            sib("model.safetensors", Some("4"), None, 1),
            sib("modeling.py", None, Some("5"), 1),
            sib("x.gguf", Some("6"), None, 1),
        ];
        let mut e = crate::Catalog::builtin()
            .get("qwen3-1.7b-mlx-4bit")
            .unwrap()
            .clone();
        let names: Vec<_> = select_files(&e, &sibs)
            .unwrap()
            .into_iter()
            .map(|s| s.rfilename)
            .collect();
        assert_eq!(names, vec!["config.json", "model.safetensors"]);
        e.files = vec!["x.gguf".into()];
        assert_eq!(select_files(&e, &sibs).unwrap().len(), 1);
        e.files = vec!["missing.gguf".into()];
        assert!(select_files(&e, &sibs).is_err());
    }

    #[test]
    fn default_ids() {
        assert_eq!(
            default_id(Path::new(
                "/x/hub/models--mlx-community--Qwen3-1.7B-4bit/snapshots/abc"
            )),
            "qwen3-1.7b-4bit"
        );
        assert_eq!(default_id(Path::new("/m/Foo-Q4_K_M.gguf")), "foo-q4_k_m");
    }

    #[test]
    fn rejects_plain_http_remote() {
        assert!(HubClient::new("http://example.com", None).is_err());
        assert!(HubClient::new("http://127.0.0.1:9", None).is_ok());
    }

    #[test]
    fn slugs_and_unique_ids() {
        assert_eq!(slug("Qwen3 1.7B (Instruct)!"), "qwen3-1.7b-instruct");
        assert_eq!(slug("  "), "model");
        let mut reg = Registry::in_memory(vec![]);
        assert_eq!(unique_id(&reg, "mine"), "mine");
        assert_eq!(
            unique_id(&reg, "qwen3-1.7b"),
            "qwen3-1.7b-2",
            "catalog family names are reserved"
        );
        let d = tempfile::tempdir().unwrap();
        let g = d.path().join("mine.gguf");
        std::fs::write(&g, crate::gguf::tests::sample_gguf()).unwrap();
        add_local(&g, Some("mine"), &mut reg).unwrap();
        assert_eq!(unique_id(&reg, "mine"), "mine-2");
    }

    #[test]
    fn add_local_auto_names_blobs_and_refuses_duplicates() {
        let home = tempfile::tempdir().unwrap();
        let paths = Paths::at(home.path());
        paths.ensure().unwrap();
        let mut reg = Registry::load(&paths.registry_file()).unwrap();
        let d = tempfile::tempdir().unwrap();

        // An Ollama-style blob: no extension, hash name, real name in the header.
        let blob = d.path().join("sha256-0123456789abcdef0123456789abcdef");
        std::fs::write(
            &blob,
            crate::gguf::tests::sample_gguf_named(Some("Tiny Llama 1B")),
        )
        .unwrap();
        let e = add_local_auto(&blob, &mut reg).unwrap();
        assert_eq!(e.id, "tiny-llama-1b-q4_k_m");
        assert!(!e.managed);
        let err = add_local_auto(&blob, &mut reg).unwrap_err().to_string();
        assert!(
            err.contains("already added as 'tiny-llama-1b-q4_k_m'"),
            "{err}"
        );

        // A normal file whose name already carries the quantization.
        let f = d.path().join("Mistral-7B-Instruct-Q4_K_M.gguf");
        std::fs::write(&f, crate::gguf::tests::sample_gguf()).unwrap();
        assert_eq!(
            add_local_auto(&f, &mut reg).unwrap().id,
            "mistral-7b-instruct-q4_k_m"
        );

        // Not a model.
        let junk = d.path().join("notes.txt");
        std::fs::write(&junk, "hello").unwrap();
        assert!(add_local_auto(&junk, &mut reg).is_err());
        assert!(
            Registry::load(&paths.registry_file()).unwrap().models.len() == 2,
            "persisted"
        );
    }

    /// A downloaded model, recorded the way `pull` records it: `managed`, canonical path.
    #[cfg(unix)]
    fn managed_entry(reg: &mut Registry, id: &str, file: &Path) {
        let mut e = add_local(file, Some(id), reg).unwrap();
        reg.remove(id);
        e.managed = true;
        e.path = std::fs::canonicalize(file).unwrap();
        reg.insert(e).unwrap();
        reg.save().unwrap();
    }

    /// The models folder is a link to another disk (as with a beta home on an external drive):
    /// removal deletes the model's files there and keeps the folder itself.
    #[cfg(unix)]
    #[test]
    fn remove_managed_through_a_linked_models_folder() {
        let home = tempfile::tempdir().unwrap();
        let store = tempfile::tempdir().unwrap();
        let paths = Paths::at(home.path());
        paths.ensure().unwrap();
        std::fs::remove_dir(paths.models_dir()).unwrap();
        std::os::unix::fs::symlink(store.path(), paths.models_dir()).unwrap();
        std::fs::create_dir(store.path().join("tiny")).unwrap();
        let gguf = store.path().join("tiny/tiny.gguf");
        std::fs::write(&gguf, crate::gguf::tests::sample_gguf()).unwrap();
        let mut reg = Registry::load(&paths.registry_file()).unwrap();
        managed_entry(&mut reg, "tiny", &gguf);
        remove("tiny", &paths, &mut reg).unwrap();
        assert!(
            !store.path().join("tiny").exists(),
            "the model's files are deleted"
        );
        assert!(
            store.path().exists() && paths.models_dir().exists(),
            "the folder stays"
        );
        assert!(Registry::load(&paths.registry_file())
            .unwrap()
            .get("tiny")
            .is_none());
    }

    /// `models/<id>` that leads outside the models folder is never deleted.
    #[cfg(unix)]
    #[test]
    fn remove_refuses_a_model_folder_that_leads_outside() {
        let home = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let paths = Paths::at(home.path());
        paths.ensure().unwrap();
        let gguf = outside.path().join("tiny.gguf");
        std::fs::write(&gguf, crate::gguf::tests::sample_gguf()).unwrap();
        std::os::unix::fs::symlink(outside.path(), paths.models_dir().join("tiny")).unwrap();
        let mut reg = Registry::load(&paths.registry_file()).unwrap();
        managed_entry(&mut reg, "tiny", &gguf);
        let err = remove("tiny", &paths, &mut reg).unwrap_err().to_string();
        assert!(err.contains("not inside the managed models dir"), "{err}");
        assert!(gguf.exists(), "files outside are untouched");
        assert!(Registry::load(&paths.registry_file())
            .unwrap()
            .get("tiny")
            .is_some());
    }

    /// Files already gone (or the drive is disconnected): the entry is unregistered.
    #[cfg(unix)]
    #[test]
    fn remove_managed_with_missing_files_unregisters() {
        let home = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        let paths = Paths::at(home.path());
        paths.ensure().unwrap();
        let gguf = elsewhere.path().join("tiny.gguf");
        std::fs::write(&gguf, crate::gguf::tests::sample_gguf()).unwrap();
        let mut reg = Registry::load(&paths.registry_file()).unwrap();
        managed_entry(&mut reg, "tiny", &gguf);
        remove("tiny", &paths, &mut reg).unwrap();
        assert!(
            gguf.exists(),
            "nothing outside the models folder is touched"
        );
        assert!(Registry::load(&paths.registry_file())
            .unwrap()
            .get("tiny")
            .is_none());
    }

    #[test]
    fn add_and_remove_unmanaged_keeps_files() {
        let home = tempfile::tempdir().unwrap();
        let paths = Paths::at(home.path());
        paths.ensure().unwrap();
        let model = tempfile::tempdir().unwrap();
        let gguf = model.path().join("tiny.gguf");
        std::fs::write(&gguf, crate::gguf::tests::sample_gguf()).unwrap();
        let mut reg = Registry::load(&paths.registry_file()).unwrap();
        let e = add_local(&gguf, None, &mut reg).unwrap();
        assert_eq!(e.id, "tiny");
        assert!(!e.managed);
        assert!(verify(&e).unwrap().is_empty());
        remove("tiny", &paths, &mut reg).unwrap();
        assert!(gguf.exists(), "unmanaged files are never deleted");
        assert!(Registry::load(&paths.registry_file())
            .unwrap()
            .get("tiny")
            .is_none());
    }
}
