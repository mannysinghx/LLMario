//! The disk tier of the KV cache (Architecture §8.6).
//!
//! When a slot's cached conversation is about to be dropped (a new request takes the slot, or
//! the shared pool runs out), its state is written to a file; a later request that continues the
//! conversation reads it back instead of recomputing the prompt. Reading a few hundred MiB from
//! an SSD takes a fraction of the time the prefill it replaces does, and the RAM it used is free
//! for other requests meanwhile.
//!
//! - Files are written straight from the cache memory (no second copy in RAM) under a temporary
//!   name; a background thread makes the data durable and only then renames the file into place,
//!   so a crash or power loss never leaves a torn file under a final name.
//! - Each file carries the model key (engine version, architecture, tensor table and sampled
//!   weight bytes) and the cache-layout fingerprint; a file is only restored into the same model
//!   and layout.
//! - The directory has a byte budget: the least recently used files leave first (files of other
//!   models in the same directory count too). Writes are skipped when the disk is nearly full.
//! - Files hold conversation state, so the directory is owner-only (0700) and files are 0600.
//!   Deleting the directory removes everything; nothing else refers to it.
//!
//! Restores follow the cache's rewind rules: a snapshot of plain paged layers can be cut back
//! to the prompt's common prefix; one with recurrent state or window rings is restored only when
//! all its tokens are a prefix of the prompt (it cannot be rewound).

use llmario_engine_formats::GgufFile;
use llmario_engine_model::StableHasher;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::time::{Duration, SystemTime};

const MAGIC: &[u8; 8] = b"LLMKVC01";
const VERSION: u32 = 1;
const HEADER: usize = 64;
/// The payload starts on a page boundary so the mapped bytes are page-aligned.
const PAGE: usize = 4096;
/// Keep this much disk free beyond a file being written.
const DISK_RESERVE: u64 = 2 << 30;
/// Temporary files older than this are left over from a crash and removed at startup.
const STALE_TMP: Duration = Duration::from_secs(3600);
const FLAG_TRIMMABLE: u32 = 1;

/// A key for everything the cached state of `file` depends on besides the cache layout: the
/// engine version, the architecture, the tensor table and a sample of the weight bytes (so two
/// fine-tunes of one base model differ). Stable across runs.
pub fn model_key(file: &GgufFile) -> u64 {
    let mut h = StableHasher::default();
    env!("CARGO_PKG_VERSION").hash(&mut h);
    file.architecture().unwrap_or("").hash(&mut h);
    file.tensors.len().hash(&mut h);
    for t in &file.tensors {
        t.name.hash(&mut h);
        format!("{:?}", t.dtype).hash(&mut h);
        t.shape.dims().hash(&mut h);
        t.span.len.hash(&mut h);
    }
    let n = file.tensors.len();
    let step = (n / 16).max(1);
    for t in file.tensors.iter().step_by(step) {
        let b = file.tensor_bytes(t);
        let k = b.len().min(64);
        b[..k].hash(&mut h);
        b[b.len() - k..].hash(&mut h);
    }
    h.finish()
}

/// What a file holds, from its header.
struct Header {
    trimmable: bool,
    model_key: u64,
    fingerprint: u64,
    len: usize,
    payload_bytes: usize,
    payload_off: usize,
}

impl Header {
    fn payload_off(len: usize) -> usize {
        (HEADER + len * 4).div_ceil(PAGE) * PAGE
    }

    fn encode(&self) -> [u8; HEADER] {
        let mut b = [0u8; HEADER];
        b[0..8].copy_from_slice(MAGIC);
        b[8..12].copy_from_slice(&VERSION.to_le_bytes());
        let flags = if self.trimmable { FLAG_TRIMMABLE } else { 0 };
        b[12..16].copy_from_slice(&flags.to_le_bytes());
        b[16..24].copy_from_slice(&self.model_key.to_le_bytes());
        b[24..32].copy_from_slice(&self.fingerprint.to_le_bytes());
        b[32..40].copy_from_slice(&(self.len as u64).to_le_bytes());
        b[40..48].copy_from_slice(&(self.payload_bytes as u64).to_le_bytes());
        b[48..56].copy_from_slice(&(self.payload_off as u64).to_le_bytes());
        b
    }

    /// Parse and check a header against the file size; `None` for anything malformed.
    fn decode(b: &[u8], file_len: u64) -> Option<Header> {
        if b.len() < HEADER || &b[0..8] != MAGIC {
            return None;
        }
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        if u32_at(8) != VERSION {
            return None;
        }
        let len = usize::try_from(u64_at(32)).ok()?;
        let payload_bytes = usize::try_from(u64_at(40)).ok()?;
        let payload_off = usize::try_from(u64_at(48)).ok()?;
        if len == 0
            || len > (1 << 26)
            || payload_off != Header::payload_off(len)
            || (payload_off as u64).checked_add(payload_bytes as u64)? != file_len
        {
            return None;
        }
        Some(Header {
            trimmable: u32_at(12) & FLAG_TRIMMABLE != 0,
            model_key: u64_at(16),
            fingerprint: u64_at(24),
            len,
            payload_bytes,
            payload_off,
        })
    }
}

/// One file in the directory.
struct Entry {
    name: String,
    bytes: u64,
    last_used: SystemTime,
    /// The tokens a file of this model and layout covers (`None`: another model's file, kept
    /// only for the budget).
    ours: Option<Ours>,
    /// Durable and under its final name.
    ready: bool,
}

struct Ours {
    tokens: Vec<u32>,
    trimmable: bool,
}

struct Index {
    entries: Vec<Entry>,
}

impl Index {
    fn bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.bytes).sum()
    }
}

struct Shared {
    dir: PathBuf,
    budget: u64,
    index: Mutex<Index>,
    ready: Condvar,
}

impl Shared {
    /// Remove least recently used ready files until the directory fits the budget.
    fn enforce_budget(&self, index: &mut Index) {
        while index.bytes() > self.budget {
            let victim = index
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.ready)
                .min_by_key(|(_, e)| e.last_used)
                .map(|(i, _)| i);
            let Some(i) = victim else { break };
            let e = index.entries.swap_remove(i);
            let _ = fs::remove_file(self.dir.join(&e.name));
        }
    }
}

/// A file found for a prompt: `usable` of its tokens can be kept after the restore.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    name: String,
    pub usable: usize,
}

/// A restored file, mapped: the payload is read straight from the page cache.
pub struct Loaded {
    pub tokens: Vec<u32>,
    map: memmap2::Mmap,
    off: usize,
    bytes: usize,
}

impl Loaded {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
    pub fn payload(&self) -> &[u8] {
        &self.map[self.off..self.off + self.bytes]
    }
}

pub struct DiskTier {
    shared: Arc<Shared>,
    model_key: u64,
    fingerprint: u64,
    seq: u64,
    syncer: Option<mpsc::Sender<String>>,
    thread: Option<std::thread::JoinHandle<()>>,
    warned_full: bool,
}

impl DiskTier {
    /// Open (creating) `dir` for a model and cache layout, index the files already there and
    /// start the background syncer.
    pub fn open(dir: &Path, budget: u64, model_key: u64, fingerprint: u64) -> io::Result<DiskTier> {
        if !cfg!(unix) {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "the KV disk tier runs on macOS and Linux",
            ));
        }
        fs::create_dir_all(dir)?;
        set_mode(dir, 0o700)?;
        let prefix = format!("{model_key:016x}-{fingerprint:016x}-");
        let now = SystemTime::now();
        let mut entries = Vec::new();
        for de in fs::read_dir(dir)? {
            let de = de?;
            let name = de.file_name().to_string_lossy().into_owned();
            let meta = match de.metadata() {
                Ok(m) if m.is_file() => m,
                _ => continue,
            };
            let mtime = meta.modified().unwrap_or(now);
            if name.ends_with(".kv.tmp") {
                if now.duration_since(mtime).unwrap_or_default() > STALE_TMP {
                    let _ = fs::remove_file(de.path());
                }
                continue;
            }
            if !name.ends_with(".kv") {
                continue;
            }
            let ours = if name.starts_with(&prefix) {
                match read_head(&de.path(), meta.len()) {
                    Some((h, tokens))
                        if h.model_key == model_key && h.fingerprint == fingerprint =>
                    {
                        Some(Ours {
                            tokens,
                            trimmable: h.trimmable,
                        })
                    }
                    _ => {
                        tracing::warn!(file = %name, "removing an unreadable KV cache file");
                        let _ = fs::remove_file(de.path());
                        continue;
                    }
                }
            } else {
                None
            };
            entries.push(Entry {
                name,
                bytes: meta.len(),
                last_used: mtime,
                ours,
                ready: true,
            });
        }
        let shared = Arc::new(Shared {
            dir: dir.to_path_buf(),
            budget,
            index: Mutex::new(Index { entries }),
            ready: Condvar::new(),
        });
        shared.enforce_budget(&mut shared.index.lock().unwrap());
        let (tx, rx) = mpsc::channel::<String>();
        let s2 = shared.clone();
        let thread = std::thread::Builder::new()
            .name("llmario-kv-disk".into())
            .spawn(move || {
                for name in rx {
                    s2.finish_write(&name);
                }
            })?;
        Ok(DiskTier {
            shared,
            model_key,
            fingerprint,
            seq: 0,
            syncer: Some(tx),
            thread: Some(thread),
            warned_full: false,
        })
    }

    /// Bytes and files of this model in the directory.
    pub fn usage(&self) -> (u64, usize) {
        let ix = self.shared.index.lock().unwrap();
        let ours = ix.entries.iter().filter(|e| e.ours.is_some());
        ours.fold((0, 0), |(b, n), e| (b + e.bytes, n + 1))
    }

    /// Whether a file already holds `tokens` (exactly, or a longer cuttable file that starts
    /// with them); refreshes that file's place in the LRU order.
    pub fn covers(&self, tokens: &[u32]) -> bool {
        let mut ix = self.shared.index.lock().unwrap();
        for e in &mut ix.entries {
            if let Some(o) = &e.ours {
                if o.tokens == tokens || (o.trimmable && o.tokens.starts_with(tokens)) {
                    e.last_used = SystemTime::now();
                    return true;
                }
            }
        }
        false
    }

    /// Write a file for `tokens` whose `payload_bytes` bytes come from `write`. `Ok(false)` when
    /// it was skipped (too large for the budget, or the disk is nearly full).
    pub fn save(
        &mut self,
        tokens: &[u32],
        trimmable: bool,
        payload_bytes: usize,
        write: impl FnOnce(&mut dyn Write) -> io::Result<()>,
    ) -> io::Result<bool> {
        let len = tokens.len();
        let off = Header::payload_off(len);
        let total = (off + payload_bytes) as u64;
        if len == 0 || total > self.shared.budget / 2 {
            return Ok(false);
        }
        if let Some(free) = available_bytes(&self.shared.dir) {
            if free < total + DISK_RESERVE {
                if !self.warned_full {
                    tracing::warn!(
                        free_mib = free >> 20,
                        "disk nearly full; KV cache files are not written"
                    );
                    self.warned_full = true;
                }
                return Ok(false);
            }
        }
        self.seq += 1;
        let stamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let name = format!(
            "{:016x}-{:016x}-{stamp:x}-{}-{}.kv",
            self.model_key,
            self.fingerprint,
            std::process::id(),
            self.seq
        );
        let tmp = self.shared.dir.join(format!("{name}.tmp"));
        let written = (|| -> io::Result<()> {
            let f = create_private(&tmp)?;
            let mut w = BufWriter::with_capacity(1 << 20, f);
            let h = Header {
                trimmable,
                model_key: self.model_key,
                fingerprint: self.fingerprint,
                len,
                payload_bytes,
                payload_off: off,
            };
            w.write_all(&h.encode())?;
            for t in tokens {
                w.write_all(&t.to_le_bytes())?;
            }
            w.write_all(&vec![0u8; off - HEADER - len * 4])?;
            let mut counted = Counted { w: &mut w, n: 0 };
            write(&mut counted)?;
            if counted.n != payload_bytes {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "snapshot wrote {} bytes, expected {payload_bytes}",
                        counted.n
                    ),
                ));
            }
            w.flush()
        })();
        if let Err(e) = written {
            let _ = fs::remove_file(&tmp);
            return Err(e);
        }
        {
            let mut ix = self.shared.index.lock().unwrap();
            // A cuttable file this one extends is redundant now.
            if trimmable {
                let dir = &self.shared.dir;
                ix.entries.retain(|e| match &e.ours {
                    Some(o) if e.ready && o.trimmable && tokens.starts_with(&o.tokens) => {
                        let _ = fs::remove_file(dir.join(&e.name));
                        false
                    }
                    _ => true,
                });
            }
            ix.entries.push(Entry {
                name: name.clone(),
                bytes: total,
                last_used: SystemTime::now(),
                ours: Some(Ours {
                    tokens: tokens.to_vec(),
                    trimmable,
                }),
                ready: false,
            });
        }
        match &self.syncer {
            Some(tx) if tx.send(name.clone()).is_ok() => {}
            _ => self.shared.finish_write(&name),
        }
        Ok(true)
    }

    /// The file that keeps the most tokens of `prompt` (at least one prompt token is always left
    /// to compute).
    pub fn best(&self, prompt: &[u32]) -> Option<Hit> {
        let limit = prompt.len().checked_sub(1)?;
        let ix = self.shared.index.lock().unwrap();
        let mut best: Option<(usize, SystemTime, &Entry)> = None;
        for e in &ix.entries {
            let Some(o) = &e.ours else { continue };
            let usable = if o.trimmable {
                let common = o
                    .tokens
                    .iter()
                    .zip(prompt)
                    .take_while(|(a, b)| a == b)
                    .count();
                common.min(limit)
            } else if o.tokens.len() <= limit && prompt.starts_with(&o.tokens) {
                o.tokens.len()
            } else {
                0
            };
            let better = match &best {
                None => true,
                Some(b) => (usable, e.last_used) > (b.0, b.1),
            };
            if usable > 0 && better {
                best = Some((usable, e.last_used, e));
            }
        }
        best.map(|(usable, _, e)| Hit {
            name: e.name.clone(),
            usable,
        })
    }

    /// Map the file of `hit` (waiting for it to become durable if it was just written) and check
    /// it belongs to this model and layout.
    pub fn load(&self, hit: &Hit) -> io::Result<Loaded> {
        let deadline = std::time::Instant::now() + Duration::from_secs(60);
        let mut ix = self.shared.index.lock().unwrap();
        loop {
            match ix.entries.iter().find(|e| e.name == hit.name) {
                None => return Err(io::ErrorKind::NotFound.into()),
                Some(e) if e.ready => break,
                Some(_) => {
                    let wait = deadline.saturating_duration_since(std::time::Instant::now());
                    if wait.is_zero() {
                        return Err(io::ErrorKind::TimedOut.into());
                    }
                    ix = self.shared.ready.wait_timeout(ix, wait).unwrap().0;
                }
            }
        }
        let path = self.shared.dir.join(&hit.name);
        let loaded = (|| -> io::Result<Loaded> {
            let f = File::open(&path)?;
            // SAFETY: the file is only ever replaced by rename, never written in place, so the
            // mapping cannot change under us (an unlink keeps the mapped inode alive).
            let map = unsafe { memmap2::Mmap::map(&f)? };
            let bad = || io::Error::new(io::ErrorKind::InvalidData, "malformed KV cache file");
            let h = Header::decode(&map, map.len() as u64).ok_or_else(bad)?;
            if h.model_key != self.model_key || h.fingerprint != self.fingerprint {
                return Err(bad());
            }
            let tokens = map[HEADER..HEADER + h.len * 4]
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let _ = f.set_modified(SystemTime::now());
            Ok(Loaded {
                tokens,
                map,
                off: h.payload_off,
                bytes: h.payload_bytes,
            })
        })();
        match &loaded {
            Ok(_) => {
                if let Some(e) = ix.entries.iter_mut().find(|e| e.name == hit.name) {
                    e.last_used = SystemTime::now();
                }
            }
            Err(e) => {
                tracing::warn!(file = %hit.name, error = %e, "dropping a KV cache file");
                ix.entries.retain(|x| x.name != hit.name);
                let _ = fs::remove_file(&path);
            }
        }
        loaded
    }

    /// Wait until every file written so far is durable (tests and shutdown).
    pub fn flush(&mut self) {
        if let Some(tx) = self.syncer.take() {
            drop(tx);
            if let Some(t) = self.thread.take() {
                let _ = t.join();
            }
        }
    }
}

impl Drop for DiskTier {
    fn drop(&mut self) {
        self.flush();
    }
}

impl Shared {
    /// Make `name`'s temporary file durable, rename it into place and mark it ready; then keep
    /// the directory within the budget.
    fn finish_write(&self, name: &str) {
        let tmp = self.dir.join(format!("{name}.tmp"));
        let done = File::open(&tmp)
            .and_then(|f| f.sync_all())
            .and_then(|_| fs::rename(&tmp, self.dir.join(name)));
        let mut ix = self.index.lock().unwrap();
        match done {
            Ok(()) => {
                if let Some(e) = ix.entries.iter_mut().find(|e| e.name == name) {
                    e.ready = true;
                }
            }
            Err(e) => {
                tracing::warn!(file = %name, error = %e, "KV cache file not kept");
                let _ = fs::remove_file(&tmp);
                ix.entries.retain(|x| x.name != name);
            }
        }
        self.enforce_budget(&mut ix);
        self.ready.notify_all();
    }
}

/// Counts bytes passed through to the file (the payload must be exactly what the header says).
struct Counted<'a, W: Write> {
    w: &'a mut W,
    n: usize,
}

impl<W: Write> Write for Counted<'_, W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let k = self.w.write(buf)?;
        self.n += k;
        Ok(k)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.w.flush()
    }
}

/// Header and tokens of a file, checked against its size.
fn read_head(path: &Path, file_len: u64) -> Option<(Header, Vec<u32>)> {
    let mut f = File::open(path).ok()?;
    let mut b = [0u8; HEADER];
    f.read_exact(&mut b).ok()?;
    let h = Header::decode(&b, file_len)?;
    let mut t = vec![0u8; h.len * 4];
    f.read_exact(&mut t).ok()?;
    let tokens = t
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Some((h, tokens))
}

#[cfg(unix)]
fn create_private(path: &Path) -> io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
}

#[cfg(not(unix))]
fn create_private(path: &Path) -> io::Result<File> {
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
}

#[cfg(unix)]
fn set_mode(path: &Path, mode: u32) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(mode))
}

#[cfg(not(unix))]
fn set_mode(_path: &Path, _mode: u32) -> io::Result<()> {
    Ok(())
}

/// Bytes free for an unprivileged writer on the file system holding `dir`.
#[cfg(unix)]
fn available_bytes(dir: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let c = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
    let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: `c` is a valid C string and `s` a writable statvfs.
    if unsafe { libc::statvfs(c.as_ptr(), &mut s) } != 0 {
        return None;
    }
    // The field types differ by platform (u32 on macOS, u64 on Linux).
    #[allow(clippy::unnecessary_cast)]
    Some(s.f_bavail as u64 * s.f_frsize as u64)
}

#[cfg(not(unix))]
fn available_bytes(_dir: &Path) -> Option<u64> {
    None
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    const MIB: u64 = 1 << 20;

    fn payload(n: usize, seed: u8) -> Vec<u8> {
        (0..n).map(|i| (i as u8).wrapping_mul(31) ^ seed).collect()
    }

    fn save(t: &mut DiskTier, tokens: &[u32], trimmable: bool, p: &[u8]) -> bool {
        t.save(tokens, trimmable, p.len(), |w| w.write_all(p))
            .unwrap()
    }

    #[test]
    fn save_then_restore_round_trips_and_files_are_private() {
        let dir = tempfile::tempdir().unwrap();
        let kv = dir.path().join("kv");
        let mut t = DiskTier::open(&kv, 64 * MIB, 1, 2).unwrap();
        let tokens: Vec<u32> = (0..300).collect();
        let p = payload(12_345, 7);
        assert!(save(&mut t, &tokens, true, &p));
        let mut prompt = tokens[..250].to_vec();
        prompt.extend([9999, 9998]);
        let hit = t.best(&prompt).unwrap();
        assert_eq!(hit.usable, 250);
        let l = t.load(&hit).unwrap();
        assert_eq!(l.tokens, tokens);
        assert_eq!(l.payload(), &p[..]);
        assert_eq!(
            fs::metadata(&kv).unwrap().permissions().mode() & 0o777,
            0o700
        );
        for f in fs::read_dir(&kv).unwrap() {
            let m = f.unwrap().metadata().unwrap();
            assert_eq!(m.permissions().mode() & 0o777, 0o600);
        }
        // Covered: the same tokens or a prefix of a cuttable file.
        assert!(t.covers(&tokens));
        assert!(t.covers(&tokens[..100]));
        assert!(!t.covers(&prompt));
        assert_eq!(t.usage().1, 1);
    }

    #[test]
    fn files_survive_a_restart_and_other_layouts_are_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let tokens: Vec<u32> = (0..64).collect();
        let p = payload(5000, 1);
        {
            let mut t = DiskTier::open(dir.path(), 64 * MIB, 1, 2).unwrap();
            assert!(save(&mut t, &tokens, false, &p));
        }
        let mut prompt = tokens.clone();
        prompt.push(5);
        let t = DiskTier::open(dir.path(), 64 * MIB, 1, 2).unwrap();
        let hit = t.best(&prompt).unwrap();
        assert_eq!(hit.usable, 64);
        assert_eq!(t.load(&hit).unwrap().payload(), &p[..]);
        // Another layout of the same model, or another model, never sees it.
        assert!(DiskTier::open(dir.path(), 64 * MIB, 1, 3)
            .unwrap()
            .best(&prompt)
            .is_none());
        assert!(DiskTier::open(dir.path(), 64 * MIB, 4, 2)
            .unwrap()
            .best(&prompt)
            .is_none());
    }

    #[test]
    fn uncuttable_files_restore_only_as_a_whole_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let mut t = DiskTier::open(dir.path(), 64 * MIB, 1, 2).unwrap();
        let tokens: Vec<u32> = (0..100).collect();
        assert!(save(&mut t, &tokens, false, &payload(100, 0)));
        // Diverges inside the file: a recurrent state cannot be rewound.
        let mut diverged = tokens[..80].to_vec();
        diverged.extend([7777; 40]);
        assert!(t.best(&diverged).is_none());
        // The same prompt: one token must be recomputed, so the whole file cannot be used.
        assert!(t.best(&tokens).is_none());
        let mut longer = tokens.clone();
        longer.push(1);
        assert_eq!(t.best(&longer).unwrap().usable, 100);
        // A cuttable file is cut to the common prefix, leaving one prompt token.
        let dir2 = tempfile::tempdir().unwrap();
        let mut t2 = DiskTier::open(dir2.path(), 64 * MIB, 1, 2).unwrap();
        assert!(save(&mut t2, &tokens, true, &payload(100, 0)));
        assert_eq!(t2.best(&diverged).unwrap().usable, 80);
        assert_eq!(t2.best(&tokens).unwrap().usable, 99);
    }

    #[test]
    fn budget_evicts_least_recently_used_and_extensions_replace_prefixes() {
        let dir = tempfile::tempdir().unwrap();
        // Three files of ~1 MiB in a 2.5 MiB budget.
        let mut t = DiskTier::open(dir.path(), 5 * MIB / 2, 1, 2).unwrap();
        let a: Vec<u32> = (0..10).collect();
        let b: Vec<u32> = (100..110).collect();
        let c: Vec<u32> = (200..210).collect();
        let p = payload(MIB as usize, 3);
        assert!(save(&mut t, &a, false, &p));
        assert!(save(&mut t, &b, false, &p));
        t.flush();
        // Touch `a`, so `b` is now the least recently used.
        let mut t = DiskTier::open(dir.path(), 5 * MIB / 2, 1, 2).unwrap();
        let mut pa = a.clone();
        pa.push(0);
        t.load(&t.best(&pa).unwrap()).unwrap();
        std::thread::sleep(Duration::from_millis(20));
        assert!(save(&mut t, &c, false, &p));
        t.flush();
        let t = DiskTier::open(dir.path(), 5 * MIB / 2, 1, 2).unwrap();
        let has = |x: &[u32]| {
            let mut q = x.to_vec();
            q.push(0);
            t.best(&q).is_some()
        };
        assert!(has(&a) && !has(&b) && has(&c));
        // A file larger than half the budget is not written.
        let mut t = t;
        assert!(!save(
            &mut t,
            &[1, 2, 3],
            false,
            &payload(2 * MIB as usize, 0)
        ));
        // A cuttable file that extends another replaces it.
        let dir = tempfile::tempdir().unwrap();
        let mut t = DiskTier::open(dir.path(), 64 * MIB, 1, 2).unwrap();
        let short: Vec<u32> = (0..50).collect();
        let long: Vec<u32> = (0..90).collect();
        assert!(save(&mut t, &short, true, &payload(10, 0)));
        t.flush();
        let mut t = DiskTier::open(dir.path(), 64 * MIB, 1, 2).unwrap();
        assert!(save(&mut t, &long, true, &payload(20, 0)));
        t.flush();
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn damaged_files_are_removed_and_crash_leftovers_cleaned() {
        let dir = tempfile::tempdir().unwrap();
        let tokens: Vec<u32> = (0..40).collect();
        let name;
        {
            let mut t = DiskTier::open(dir.path(), 64 * MIB, 1, 2).unwrap();
            assert!(save(&mut t, &tokens, true, &payload(4096, 0)));
            t.flush();
            name = fs::read_dir(dir.path())
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .path();
        }
        // Truncated payload: the size no longer matches the header.
        let f = fs::OpenOptions::new().write(true).open(&name).unwrap();
        f.set_len(f.metadata().unwrap().len() - 1).unwrap();
        drop(f);
        // A stale temporary file from a crash, and a fresh one another process is writing.
        let stale = dir.path().join("x.kv.tmp");
        let fresh = dir.path().join("y.kv.tmp");
        fs::write(&stale, b"partial").unwrap();
        fs::write(&fresh, b"partial").unwrap();
        let old = SystemTime::now() - Duration::from_secs(7200);
        File::options()
            .write(true)
            .open(&stale)
            .unwrap()
            .set_modified(old)
            .unwrap();
        let t = DiskTier::open(dir.path(), 64 * MIB, 1, 2).unwrap();
        let mut prompt = tokens.clone();
        prompt.push(1);
        assert!(t.best(&prompt).is_none());
        assert!(!name.exists() && !stale.exists() && fresh.exists());
    }

    #[test]
    fn a_short_payload_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let mut t = DiskTier::open(dir.path(), 64 * MIB, 1, 2).unwrap();
        let err = t
            .save(&[1, 2, 3], true, 100, |w| w.write_all(&[0u8; 99]))
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
    }
}
