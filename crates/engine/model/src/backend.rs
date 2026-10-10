//! The execution-backend boundary (Architecture §7).
//!
//! A [`ModelBackend`] owns a loaded model's device state (weights views or device buffers, the
//! KV cache, scratch) and executes forward passes. The server's inference thread talks only to
//! this trait, so the CPU backend here and the Metal/Vulkan backends in their own crates are
//! interchangeable. Sequence state is explicit and per slot: `seq_len(s)` tokens are cached for
//! slot `s`, `truncate_seq` keeps a prefix (prefix reuse), `clear_seq` resets. One
//! [`ModelBackend::forward_batch`] call appends tokens to several slots at once (continuous
//! batching); the single-slot methods are shorthands for slot 0.

use crate::forward::{Scratch, SeqTokens};
use crate::kv::{KvFull, KvOptions, KvType, SeqSnapshot};
use crate::{ArchSpec, KvCache, Model};
use llmario_engine_cpu::ThreadPool;
use llmario_engine_formats::GgufFile;

pub trait ModelBackend: Send {
    fn spec(&self) -> &ArchSpec;
    fn name(&self) -> &'static str;
    /// Maximum tokens one slot's cache holds.
    fn max_ctx(&self) -> usize;
    /// Slots this backend keeps state for (`forward_batch` may carry up to this many).
    fn n_seqs(&self) -> usize {
        1
    }
    /// Tokens cached for slot `s`.
    fn seq_len(&self, s: usize) -> usize;
    /// Keep the first `n` tokens of slot `s` (backends with recurrent state may keep less; read
    /// `seq_len` afterwards).
    fn truncate_seq(&mut self, s: usize, n: usize);
    fn clear_seq(&mut self, s: usize);
    /// Largest total number of tokens one forward call accepts.
    fn max_batch(&self) -> usize;
    /// Append each entry's tokens to its slot and return the logits of each entry's last token,
    /// `[batch.len()][n_vocab]` in batch order. Fails without changing anything when the KV
    /// cache cannot hold the new tokens.
    fn forward_batch(&mut self, batch: &[SeqTokens]) -> Result<&[f32], KvFull>;
    /// Bytes this backend reserved beyond the mapped weights (KV at full use + scratch), for
    /// the ledger.
    fn reserved_bytes(&self) -> u64;
    /// Bytes of KV memory backed right now (grows with the cached tokens).
    fn kv_in_use_bytes(&self) -> u64;
    /// Bytes of KV memory the backend can ever hold (what the plan charges for the cache).
    fn kv_reserved_bytes(&self) -> u64;
    /// Element type of the cached K/V rows.
    fn kv_type(&self) -> KvType {
        KvType::F16
    }
    /// Positions the shared KV pool can still hand out (free blocks × block size); `usize::MAX`
    /// for a backend without a pool limit.
    fn kv_free_tokens(&self) -> usize {
        usize::MAX
    }
    /// Hash of the cache layout, stable across runs: snapshots move only between equal
    /// fingerprints (0 = this backend cannot snapshot).
    fn kv_fingerprint(&self) -> u64 {
        0
    }
    /// Bytes of a snapshot of `len` cached tokens.
    fn snapshot_bytes(&self, _len: usize) -> usize {
        0
    }
    /// Whether a restored snapshot can be cut back to a shorter prefix.
    fn snapshot_trimmable(&self) -> bool {
        false
    }
    /// Write slot `s`'s cached state (`snapshot_bytes(seq_len(s))` bytes) to `w`.
    fn write_seq(&self, _s: usize, _w: &mut dyn std::io::Write) -> std::io::Result<()> {
        Err(std::io::ErrorKind::Unsupported.into())
    }
    /// Replace slot `s`'s state with `len` tokens' worth of snapshot bytes. `false` (nothing
    /// changed) when unsupported, when the size does not match, or when the pool cannot hold it.
    fn read_seq(&mut self, _s: usize, _len: usize, _bytes: &[u8]) -> bool {
        false
    }
    /// [`ModelBackend::write_seq`] into memory.
    fn export_seq(&self, s: usize) -> Option<SeqSnapshot> {
        let len = self.seq_len(s);
        let mut bytes = Vec::with_capacity(self.snapshot_bytes(len));
        self.write_seq(s, &mut bytes).ok()?;
        Some(SeqSnapshot { len, bytes })
    }
    /// [`ModelBackend::read_seq`] from memory.
    fn import_seq(&mut self, s: usize, snap: &SeqSnapshot) -> bool {
        self.read_seq(s, snap.len, &snap.bytes)
    }

    // Single-slot shorthands (slot 0).
    fn kv_len(&self) -> usize {
        self.seq_len(0)
    }
    fn truncate(&mut self, n: usize) {
        self.truncate_seq(0, n)
    }
    fn clear(&mut self) {
        self.clear_seq(0)
    }
    /// Run `tokens` on slot 0 at positions `kv_len()..` and return the last token's logits.
    /// Panics when the cache cannot grow (a context overflow for a single-slot backend).
    fn forward(&mut self, tokens: &[u32]) -> &[f32] {
        match self.forward_batch(&[SeqTokens { seq: 0, tokens }]) {
            Ok(l) => l,
            Err(e) => panic!("{e}"),
        }
    }
}

/// Options of the CPU backend's cache and scratch.
#[derive(Clone, Copy, Debug)]
pub struct CpuOptions {
    pub threads: usize,
    /// Positions per slot.
    pub max_ctx: usize,
    /// Tokens per forward call (all slots together).
    pub n_batch: usize,
    /// Slots.
    pub n_seqs: usize,
    pub kv_type: KvType,
    /// Positions the shared pool holds across slots (`None`: every slot can hold `max_ctx`).
    pub pool_tokens: Option<usize>,
}

impl CpuOptions {
    pub fn new(threads: usize, max_ctx: usize, n_batch: usize) -> CpuOptions {
        CpuOptions {
            threads,
            max_ctx,
            n_batch,
            n_seqs: 1,
            kv_type: KvType::F16,
            pool_tokens: None,
        }
    }

    fn kv_options(&self) -> KvOptions {
        let mut o = KvOptions::new(self.max_ctx)
            .seqs(self.n_seqs)
            .kv_type(self.kv_type);
        if let Some(t) = self.pool_tokens {
            o = o.pool_tokens(t);
        }
        o
    }

    /// Bytes the backend reserves for `spec` (KV at full use + scratch), without building it.
    pub fn reserved_bytes(&self, spec: &ArchSpec) -> u64 {
        crate::kv::KvLayout::new(spec, &self.kv_options()).reserved_bytes()
            + Scratch::bytes_with_seqs(spec, self.n_batch.max(1), self.n_seqs.max(1))
    }
}

/// The CPU backend: the forward pass over mapped weights with a parked thread pool and the
/// paged KV cache.
pub struct CpuBackend<'a> {
    model: Model<'a>,
    pool: ThreadPool,
    kv: KvCache,
    scratch: Scratch,
    n_batch: usize,
    reserved: u64,
}

impl<'a> CpuBackend<'a> {
    /// A single-slot f16 backend.
    pub fn new(
        file: &'a GgufFile,
        threads: usize,
        max_ctx: usize,
        n_batch: usize,
    ) -> crate::Result<CpuBackend<'a>> {
        Self::with_options(file, CpuOptions::new(threads, max_ctx, n_batch))
    }

    pub fn with_options(file: &'a GgufFile, o: CpuOptions) -> crate::Result<CpuBackend<'a>> {
        let model = Model::load(file)?;
        let spec = &model.spec;
        for l in 0..spec.n_layer as usize {
            let g = spec.attn_geom(l);
            for hd in [g.head_dim, g.head_dim_v] {
                if !o.kv_type.supports_head_dim(hd as usize) {
                    return Err(crate::ModelError::Engine(
                        llmario_engine_core::EngineError::Format(format!(
                            "KV cache type {} needs head widths that are multiples of 32 \
                             (layer {l} has {hd})",
                            o.kv_type.name()
                        )),
                    ));
                }
            }
        }
        let n_batch = o.n_batch.max(1);
        let n_seqs = o.n_seqs.max(1).min(n_batch);
        let o = CpuOptions {
            n_batch,
            n_seqs,
            ..o
        };
        let reserved = o.reserved_bytes(spec);
        let kv = KvCache::with_options(spec, o.kv_options());
        let scratch = Scratch::with_seqs(spec, n_batch, n_seqs);
        Ok(CpuBackend {
            pool: ThreadPool::new(o.threads.max(1)),
            model,
            kv,
            scratch,
            n_batch,
            reserved,
        })
    }

    pub fn threads(&self) -> usize {
        self.pool.n_threads()
    }

    pub fn kv(&self) -> &KvCache {
        &self.kv
    }
}

impl ModelBackend for CpuBackend<'_> {
    fn spec(&self) -> &ArchSpec {
        &self.model.spec
    }
    fn name(&self) -> &'static str {
        "cpu"
    }
    fn max_ctx(&self) -> usize {
        self.kv.max_ctx()
    }
    fn n_seqs(&self) -> usize {
        self.kv.n_seqs()
    }
    fn seq_len(&self, s: usize) -> usize {
        self.kv.len(s)
    }
    fn truncate_seq(&mut self, s: usize, n: usize) {
        self.kv.truncate(s, n);
    }
    fn clear_seq(&mut self, s: usize) {
        self.kv.clear(s);
    }
    fn max_batch(&self) -> usize {
        self.n_batch
    }
    fn forward_batch(&mut self, batch: &[SeqTokens]) -> Result<&[f32], KvFull> {
        self.model
            .forward_batch(&self.pool, &mut self.kv, batch, &mut self.scratch)
    }
    fn reserved_bytes(&self) -> u64 {
        self.reserved
    }
    fn kv_in_use_bytes(&self) -> u64 {
        self.kv.in_use_bytes()
    }
    fn kv_reserved_bytes(&self) -> u64 {
        self.kv.layout.reserved_bytes()
    }
    fn kv_type(&self) -> KvType {
        self.kv.kv_type()
    }
    fn kv_free_tokens(&self) -> usize {
        if self.kv.layout.block_bytes == 0 {
            return usize::MAX;
        }
        self.kv.pool().free_blocks() * self.kv.layout.block_tokens
    }
    fn kv_fingerprint(&self) -> u64 {
        self.kv.fingerprint()
    }
    fn snapshot_bytes(&self, len: usize) -> usize {
        self.kv.snapshot_bytes(len)
    }
    fn snapshot_trimmable(&self) -> bool {
        self.kv.snapshot_trimmable()
    }
    fn write_seq(&self, s: usize, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        self.kv.write_seq(s, w)
    }
    fn read_seq(&mut self, s: usize, len: usize, bytes: &[u8]) -> bool {
        len <= self.kv.max_ctx()
            && bytes.len() == self.kv.snapshot_bytes(len)
            && self.kv.read_seq(s, len, bytes).is_ok()
    }
}
