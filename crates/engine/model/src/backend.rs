//! The execution-backend boundary (Architecture §7).
//!
//! A [`ModelBackend`] owns a loaded model's device state (weights views or device buffers, the
//! KV cache, scratch) and executes forward passes. The server's inference thread talks only to
//! this trait, so the CPU backend here and the Metal/Vulkan backends in their own crates are
//! interchangeable. Sequence state is explicit: `kv_len()` tokens are cached, `truncate` keeps a
//! prefix (prefix reuse), `clear` resets.

use crate::forward::Scratch;
use crate::{ArchSpec, KvCache, Model};
use llmario_engine_cpu::ThreadPool;
use llmario_engine_formats::GgufFile;

pub trait ModelBackend: Send {
    fn spec(&self) -> &ArchSpec;
    fn name(&self) -> &'static str;
    /// Maximum tokens the KV cache holds.
    fn max_ctx(&self) -> usize;
    fn kv_len(&self) -> usize;
    fn truncate(&mut self, n: usize);
    fn clear(&mut self);
    /// Largest `tokens.len()` a single [`ModelBackend::forward`] call accepts.
    fn max_batch(&self) -> usize;
    /// Run `tokens` at positions `kv_len()..`, append their K/V, return the last token's logits.
    fn forward(&mut self, tokens: &[u32]) -> &[f32];
    /// Bytes this backend reserved beyond the mapped weights (KV + scratch), for the ledger.
    fn reserved_bytes(&self) -> u64;
}

/// The CPU backend: the dense forward pass over mapped weights with a parked thread pool.
pub struct CpuBackend<'a> {
    model: Model<'a>,
    pool: ThreadPool,
    kv: KvCache,
    prefill: Scratch,
    decode: Scratch,
    n_batch: usize,
    reserved: u64,
}

impl<'a> CpuBackend<'a> {
    pub fn new(
        file: &'a GgufFile,
        threads: usize,
        max_ctx: usize,
        n_batch: usize,
    ) -> crate::Result<CpuBackend<'a>> {
        let model = Model::load(file)?;
        let spec = &model.spec;
        let n_batch = n_batch.max(1);
        let reserved =
            KvCache::bytes(spec, max_ctx) + Scratch::bytes(spec, n_batch) + Scratch::bytes(spec, 1);
        let kv = KvCache::new(spec, max_ctx);
        let prefill = Scratch::new(spec, n_batch);
        let decode = Scratch::new(spec, 1);
        Ok(CpuBackend {
            pool: ThreadPool::new(threads.max(1)),
            model,
            kv,
            prefill,
            decode,
            n_batch,
            reserved,
        })
    }

    pub fn threads(&self) -> usize {
        self.pool.n_threads()
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
        self.kv.max_ctx
    }
    fn kv_len(&self) -> usize {
        self.kv.len
    }
    fn truncate(&mut self, n: usize) {
        self.kv.truncate(n);
    }
    fn clear(&mut self) {
        self.kv.clear();
    }
    fn max_batch(&self) -> usize {
        self.n_batch
    }
    fn forward(&mut self, tokens: &[u32]) -> &[f32] {
        if tokens.len() == 1 {
            self.model
                .forward(&self.pool, &mut self.kv, tokens, &mut self.decode)
        } else {
            self.model
                .forward(&self.pool, &mut self.kv, tokens, &mut self.prefill)
        }
    }
    fn reserved_bytes(&self) -> u64 {
        self.reserved
    }
}
