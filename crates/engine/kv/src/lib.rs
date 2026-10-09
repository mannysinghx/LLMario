//! Paged KV arena (Architecture §8.2) and prefix cache (§8.5).
//!
//! The arena is reserved **once** (`n_blocks × block_bytes`) and carved into fixed blocks of
//! `block_tokens` positions. Sequences own block tables; blocks are reference counted so several
//! sequences can share a cached prefix; freed blocks keep their content hash and sit in an LRU
//! free queue, so a later request with the same prefix can reclaim them without recomputation.
//! Nothing here grows by reallocation, and `free_blocks() × block_bytes` is exactly what
//! admission can hand out.
//!
//! Hashes chain per block: `SHA-256(parent_hash ‖ block_tokens ‖ extra)` where `extra` carries the
//! model digest, KV types, RoPE parameters, template hash and the per-client salt, so two API
//! keys never share blocks (CVE-2025-46570 class). Only full blocks are hashed; a partial tail
//! block is never shared.
//!
//! Storage layout is the backend's business: the arena hands out block indices, and a backend
//! places K/V (or latent) bytes for layer group `g`, block `b` at `g * n_blocks * block_bytes +
//! b * block_bytes` inside the region it allocated from the plan. A recurrent-state pool is a
//! second [`BlockPool`] with its own block size (§8.1).

use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};

pub type BlockId = u32;
pub type Hash = [u8; 32];

/// Per-block metadata; the arena bytes live elsewhere.
#[derive(Clone, Debug, Default)]
struct BlockMeta {
    refs: u32,
    /// Content hash of a full block (None while partial or after eviction).
    hash: Option<Hash>,
}

/// A fixed-size pool of KV blocks with reference counting and hash-addressed reuse.
#[derive(Debug)]
pub struct BlockPool {
    block_tokens: usize,
    block_bytes: u64,
    meta: Vec<BlockMeta>,
    /// Free blocks in LRU order (front = evict first). A free block may still carry a hash.
    free: VecDeque<BlockId>,
    /// Hash → block, for every block (in use or free) whose content is a hashed full block.
    by_hash: HashMap<Hash, BlockId>,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl BlockPool {
    pub fn new(n_blocks: usize, block_tokens: usize, block_bytes: u64) -> BlockPool {
        assert!(block_tokens > 0);
        BlockPool {
            block_tokens,
            block_bytes,
            meta: vec![BlockMeta::default(); n_blocks],
            free: (0..n_blocks as BlockId).collect(),
            by_hash: HashMap::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    pub fn n_blocks(&self) -> usize {
        self.meta.len()
    }
    pub fn block_tokens(&self) -> usize {
        self.block_tokens
    }
    pub fn block_bytes(&self) -> u64 {
        self.block_bytes
    }
    /// Blocks that can be handed out right now (free, whether or not they hold cached content).
    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            n_blocks: self.meta.len(),
            free: self.free.len(),
            cached_free: self.free.iter().filter(|&&b| self.meta[b as usize].hash.is_some()).count(),
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
        }
    }

    /// Take `n` free blocks (evicting cached content LRU-first). `None` if fewer are free.
    pub fn alloc(&mut self, n: usize) -> Option<Vec<BlockId>> {
        if self.free.len() < n {
            return None;
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let b = self.free.pop_front().unwrap();
            let m = &mut self.meta[b as usize];
            if let Some(h) = m.hash.take() {
                self.by_hash.remove(&h);
                self.evictions += 1;
            }
            m.refs = 1;
            out.push(b);
        }
        Some(out)
    }

    /// Record that block `b` now holds a full block with content hash `h` (callable once the
    /// backend has written it). Replaces any earlier mapping of the same hash.
    pub fn set_hash(&mut self, b: BlockId, h: Hash) {
        if let Some(old) = self.meta[b as usize].hash.replace(h) {
            self.by_hash.remove(&old);
        }
        self.by_hash.insert(h, b);
    }

    /// Release one reference; at zero the block joins the LRU free queue (keeping its hash).
    pub fn release(&mut self, b: BlockId) {
        let m = &mut self.meta[b as usize];
        assert!(m.refs > 0, "double release of block {b}");
        m.refs -= 1;
        if m.refs == 0 {
            self.free.push_back(b);
        }
    }

    /// Reclaim the block holding `h`, if any: a free cached block is pulled out of the free
    /// queue, an in-use block gains a reference. Returns the block id.
    pub fn reclaim(&mut self, h: &Hash) -> Option<BlockId> {
        let b = *self.by_hash.get(h)?;
        let m = &mut self.meta[b as usize];
        if m.refs == 0 {
            if let Some(pos) = self.free.iter().position(|&x| x == b) {
                self.free.remove(pos);
            }
        }
        m.refs += 1;
        self.hits += 1;
        Some(b)
    }

    pub fn note_miss(&mut self) {
        self.misses += 1;
    }

    pub fn refs(&self, b: BlockId) -> u32 {
        self.meta[b as usize].refs
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolStats {
    pub n_blocks: usize,
    pub free: usize,
    pub cached_free: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

/// Everything besides the tokens that must match for a cached block to be reusable.
#[derive(Clone, Debug)]
pub struct CacheKey {
    pub model_digest: Vec<u8>,
    pub kv_types: String,
    pub rope_params: String,
    pub template_hash: String,
    /// Per-client salt (derived from the API key); empty = shared.
    pub salt: Vec<u8>,
}

impl CacheKey {
    fn extra(&self) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&self.model_digest);
        v.push(0);
        v.extend_from_slice(self.kv_types.as_bytes());
        v.push(0);
        v.extend_from_slice(self.rope_params.as_bytes());
        v.push(0);
        v.extend_from_slice(self.template_hash.as_bytes());
        v.push(0);
        v.extend_from_slice(&self.salt);
        v
    }
}

/// Hash of one full block given its parent's hash.
pub fn block_hash(parent: Option<&Hash>, tokens: &[u32], key: &CacheKey) -> Hash {
    let mut h = Sha256::new();
    match parent {
        Some(p) => h.update(p),
        None => h.update([0u8; 32]),
    }
    for t in tokens {
        h.update(t.to_le_bytes());
    }
    h.update(key.extra());
    h.finalize().into()
}

/// Hash chain for a token sequence: one hash per full block.
pub fn chain_hashes(tokens: &[u32], block_tokens: usize, key: &CacheKey) -> Vec<Hash> {
    let mut out = Vec::new();
    let mut parent: Option<Hash> = None;
    for chunk in tokens.chunks_exact(block_tokens) {
        let h = block_hash(parent.as_ref(), chunk, key);
        out.push(h);
        parent = Some(h);
    }
    out
}

/// A sequence's view: its blocks (in position order) and how many tokens are filled.
#[derive(Clone, Debug, Default)]
pub struct BlockTable {
    pub blocks: Vec<BlockId>,
    pub n_tokens: usize,
}

impl BlockTable {
    /// Block and in-block offset of position `pos`.
    pub fn locate(&self, pos: usize, block_tokens: usize) -> (BlockId, usize) {
        (self.blocks[pos / block_tokens], pos % block_tokens)
    }
}

/// Outcome of [`Sequence::acquire`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Acquired {
    /// Tokens already present in the cache (a multiple of the block size); recompute from here.
    pub cached_tokens: usize,
}

/// Admission helper: a sequence that needs room for `n_tokens` reclaims every cached full
/// block of its prefix, then allocates the rest.
pub struct Sequence {
    pub table: BlockTable,
}

impl Sequence {
    /// Reserve blocks for `tokens` (prompt + planned generation length). Reclaims cached prefix
    /// blocks by hash; allocates fresh blocks for the remainder. On failure nothing is held.
    pub fn acquire(
        pool: &mut BlockPool,
        tokens: &[u32],
        reserve_tokens: usize,
        key: &CacheKey,
    ) -> Option<(Sequence, Acquired)> {
        let bt = pool.block_tokens();
        let hashes = chain_hashes(tokens, bt, key);
        let mut blocks = Vec::new();
        let mut cached = 0;
        for h in &hashes {
            match pool.reclaim(h) {
                Some(b) => {
                    blocks.push(b);
                    cached += bt;
                }
                None => {
                    pool.note_miss();
                    break;
                }
            }
        }
        let needed_blocks = reserve_tokens.max(tokens.len()).div_ceil(bt);
        let fresh = needed_blocks.saturating_sub(blocks.len());
        match pool.alloc(fresh) {
            Some(mut v) => blocks.append(&mut v),
            None => {
                for b in blocks {
                    pool.release(b);
                }
                return None;
            }
        }
        Some((
            Sequence {
                table: BlockTable {
                    blocks,
                    n_tokens: cached,
                },
            },
            Acquired {
                cached_tokens: cached,
            },
        ))
    }

    /// After the backend wrote positions up to `n_tokens`, publish the hashes of the newly
    /// completed full blocks so other sequences can reuse them.
    pub fn publish(&mut self, pool: &mut BlockPool, tokens: &[u32], key: &CacheKey) {
        let bt = pool.block_tokens();
        let full = tokens.len() / bt;
        let hashes = chain_hashes(&tokens[..full * bt], bt, key);
        for (i, h) in hashes.iter().enumerate() {
            pool.set_hash(self.table.blocks[i], *h);
        }
        self.table.n_tokens = tokens.len();
    }

    pub fn release(self, pool: &mut BlockPool) {
        for b in self.table.blocks {
            pool.release(b);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(salt: &[u8]) -> CacheKey {
        CacheKey {
            model_digest: vec![1, 2, 3],
            kv_types: "f16/f16".into(),
            rope_params: "theta=1e6".into(),
            template_hash: "abc".into(),
            salt: salt.to_vec(),
        }
    }

    #[test]
    fn alloc_release_and_capacity() {
        let mut p = BlockPool::new(4, 16, 1024);
        assert_eq!(p.free_blocks(), 4);
        let a = p.alloc(3).unwrap();
        assert_eq!(p.free_blocks(), 1);
        assert!(p.alloc(2).is_none());
        for b in a {
            p.release(b);
        }
        assert_eq!(p.free_blocks(), 4);
    }

    #[test]
    fn prefix_is_reclaimed_from_cached_free_blocks() {
        let mut p = BlockPool::new(8, 4, 64);
        let k = key(b"");
        let toks: Vec<u32> = (0..10).collect(); // 2 full blocks + 2 tail
        let (mut s, acq) = Sequence::acquire(&mut p, &toks, 16, &k).unwrap();
        assert_eq!(acq.cached_tokens, 0);
        assert_eq!(s.table.blocks.len(), 4);
        s.publish(&mut p, &toks, &k);
        s.release(&mut p);
        assert_eq!(p.stats().cached_free, 2);
        // Same prefix, longer prompt: the two full blocks come back without recompute.
        let toks2: Vec<u32> = (0..13).collect();
        let (s2, acq2) = Sequence::acquire(&mut p, &toks2, 13, &k).unwrap();
        assert_eq!(acq2.cached_tokens, 8);
        assert_eq!(p.stats().hits, 2);
        assert_eq!(p.refs(s2.table.blocks[0]), 1);
        s2.release(&mut p);
    }

    #[test]
    fn different_salt_never_shares() {
        let mut p = BlockPool::new(8, 4, 64);
        let toks: Vec<u32> = (0..8).collect();
        let (mut s, _) = Sequence::acquire(&mut p, &toks, 8, &key(b"client-a")).unwrap();
        s.publish(&mut p, &toks, &key(b"client-a"));
        s.release(&mut p);
        let (_, acq) = Sequence::acquire(&mut p, &toks, 8, &key(b"client-b")).unwrap();
        assert_eq!(acq.cached_tokens, 0);
    }

    #[test]
    fn shared_prefix_is_refcounted_and_eviction_is_lru() {
        let mut p = BlockPool::new(3, 4, 64);
        let k = key(b"");
        let toks: Vec<u32> = (0..4).collect();
        let (mut a, _) = Sequence::acquire(&mut p, &toks, 4, &k).unwrap();
        a.publish(&mut p, &toks, &k);
        let (b, acq) = Sequence::acquire(&mut p, &toks, 4, &k).unwrap();
        assert_eq!(acq.cached_tokens, 4);
        assert_eq!(p.refs(a.table.blocks[0]), 2);
        assert_eq!(a.table.blocks[0], b.table.blocks[0]);
        a.release(&mut p);
        b.release(&mut p);
        assert_eq!(p.free_blocks(), 3);
        // Allocate all three: the cached block is evicted last (LRU: it was freed last).
        let all = p.alloc(3).unwrap();
        assert_eq!(all.last(), Some(&0));
        assert_eq!(p.stats().evictions, 1);
    }

    #[test]
    fn hash_chain_depends_on_parent_and_extra() {
        let k = key(b"");
        let h1 = chain_hashes(&[1, 2, 3, 4, 5, 6, 7, 8], 4, &k);
        let h2 = chain_hashes(&[9, 2, 3, 4, 5, 6, 7, 8], 4, &k);
        assert_eq!(h1.len(), 2);
        assert_ne!(h1[0], h2[0]);
        assert_ne!(h1[1], h2[1], "a change in block 0 changes block 1's chained hash");
        let k2 = CacheKey {
            kv_types: "q8_0/q4_0".into(),
            ..key(b"")
        };
        assert_ne!(chain_hashes(&[1, 2, 3, 4], 4, &k2)[0], h1[0]);
    }

    #[test]
    fn acquire_fails_atomically() {
        let mut p = BlockPool::new(2, 4, 64);
        let k = key(b"");
        assert!(Sequence::acquire(&mut p, &[1, 2, 3, 4, 5, 6, 7, 8, 9], 12, &k).is_none());
        assert_eq!(p.free_blocks(), 2);
    }
}
