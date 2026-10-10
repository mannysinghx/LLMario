//! The paged KV cache of the Metal backend (Architecture §8.2).
//!
//! Same layout and pool rules as the CPU cache (`llmario_engine_model::kv`): blocks of
//! [`BLOCK_TOKENS`] positions holding every layer's K and V rows, block ids / reference counts /
//! LRU / prefix hashes from [`BlockPool`], copy-on-write for a shared partial block. A block is a
//! shared-storage `MTLBuffer` created when a sequence first reaches it and released when no
//! sequence uses it, so GPU memory follows the tokens actually cached (Metal charges a buffer's
//! full size to the process at creation, so one up-front buffer for the admitted context cost
//! 3.9 GiB for Qwen3-1.7B at 32K before any request). Kernels reach blocks through a table of GPU
//! addresses (one row of `blocks_per_seq` entries per sequence), so the blocks are kept resident
//! through a dynamic residency set (macOS 15+) or declared on each command encoder.

use crate::device::{Buf, DynResidency, Gpu};
use crate::Result;
use llmario_engine_kv::{BlockId, BlockPool};
use llmario_engine_model::kv::{KvLayout, KvOptions, KvType};
use llmario_engine_model::{ArchSpec, KvFull};

/// Positions per block: the prefill kernel's key tile (`FA_BK`), so a tile never straddles
/// blocks.
pub const BLOCK_TOKENS: usize = 32;

struct Seq {
    blocks: Vec<BlockId>,
    len: usize,
}

pub(crate) struct MetalKv {
    pub layout: KvLayout,
    pool: BlockPool,
    backing: Vec<Option<Buf>>,
    seqs: Vec<Seq>,
    /// `[n_seqs][blocks_per_seq]` u64 GPU addresses of each sequence's blocks.
    pub table: Buf,
    resid: Option<DynResidency>,
}

impl MetalKv {
    pub fn new(
        gpu: &Gpu,
        spec: &ArchSpec,
        max_ctx: usize,
        n_seqs: usize,
        kv_type: KvType,
    ) -> Result<MetalKv> {
        let layout = KvLayout::new(
            spec,
            &KvOptions::new(max_ctx)
                .seqs(n_seqs)
                .kv_type(kv_type)
                .block_tokens(BLOCK_TOKENS),
        );
        let n_blocks = layout.pool_blocks;
        let table = gpu.alloc(layout.n_seqs * layout.blocks_per_seq.max(1) * 8)?;
        Ok(MetalKv {
            pool: BlockPool::new(n_blocks, BLOCK_TOKENS, layout.block_bytes as u64),
            backing: (0..n_blocks).map(|_| None).collect(),
            seqs: (0..layout.n_seqs)
                .map(|_| Seq {
                    blocks: Vec::new(),
                    len: 0,
                })
                .collect(),
            resid: gpu.dynamic_residency("llmario-kv-blocks"),
            table,
            layout,
        })
    }

    pub fn n_seqs(&self) -> usize {
        self.seqs.len()
    }
    pub fn len(&self, s: usize) -> usize {
        self.seqs[s].len
    }
    pub fn set_len(&mut self, s: usize, n: usize) {
        self.seqs[s].len = n;
    }
    pub fn free_tokens(&self) -> usize {
        self.pool.free_blocks() * BLOCK_TOKENS
    }
    pub fn resident_blocks(&self) -> usize {
        self.backing.iter().filter(|b| b.is_some()).count()
    }
    pub fn in_use_bytes(&self) -> u64 {
        (self.resident_blocks() * self.layout.block_bytes) as u64 + self.table.len() as u64
    }
    pub fn reserved_bytes(&self) -> u64 {
        self.layout.reserved_bytes() + self.table.len() as u64
    }

    fn write_entry(&self, s: usize, i: usize, b: BlockId) {
        let addr = self.backing[b as usize]
            .as_ref()
            .expect("backed block")
            .gpu_address();
        let off = (s * self.layout.blocks_per_seq + i) * 8;
        self.table.write_bytes(off, &addr.to_le_bytes());
    }

    fn back(&mut self, gpu: &Gpu, b: BlockId) -> Result<()> {
        if self.backing[b as usize].is_none() {
            let buf = gpu.alloc(self.layout.block_bytes)?;
            if let Some(r) = &mut self.resid {
                r.add(&buf);
            }
            self.backing[b as usize] = Some(buf);
        }
        Ok(())
    }

    fn release_block(&mut self, b: BlockId) {
        self.pool.release(b);
        if self.pool.refs(b) == 0 && !self.pool.is_cached(b) {
            if let Some(buf) = self.backing[b as usize].take() {
                if let Some(r) = &mut self.resid {
                    r.remove(&buf);
                }
            }
        }
    }

    /// Free blocks `reserve(s, new_len)` would take (see the CPU cache).
    pub fn blocks_needed(&self, s: usize, new_len: usize) -> usize {
        let seq = &self.seqs[s];
        let missing = new_len
            .div_ceil(BLOCK_TOKENS)
            .saturating_sub(seq.blocks.len());
        let cow = seq.len % BLOCK_TOKENS != 0
            && seq.len < new_len
            && self.pool.refs(seq.blocks[seq.len / BLOCK_TOKENS]) > 1;
        missing + cow as usize
    }

    pub fn free_blocks(&self) -> usize {
        self.pool.free_blocks()
    }

    /// Make room to append up to `new_len` positions to sequence `s` (no GPU work may be in
    /// flight: copy-on-write copies through the CPU view of the shared buffers).
    pub fn reserve(
        &mut self,
        gpu: &Gpu,
        s: usize,
        new_len: usize,
    ) -> std::result::Result<Result<()>, KvFull> {
        let len = self.seqs[s].len;
        let missing = new_len
            .div_ceil(BLOCK_TOKENS)
            .saturating_sub(self.seqs[s].blocks.len());
        let wanted = self.blocks_needed(s, new_len);
        if self.pool.free_blocks() < wanted {
            return Err(KvFull {
                needed: wanted,
                free: self.pool.free_blocks(),
            });
        }
        Ok(self.reserve_checked(gpu, s, len, new_len, missing, wanted > missing))
    }

    fn reserve_checked(
        &mut self,
        gpu: &Gpu,
        s: usize,
        len: usize,
        new_len: usize,
        missing: usize,
        cow: bool,
    ) -> Result<()> {
        if len % BLOCK_TOKENS != 0 && len < new_len {
            let bi = len / BLOCK_TOKENS;
            let old = self.seqs[s].blocks[bi];
            if cow {
                let nb = self.pool.alloc(1).expect("checked")[0];
                self.back(gpu, nb)?;
                let src = self.backing[old as usize].as_ref().unwrap();
                let dst = self.backing[nb as usize].as_ref().unwrap();
                dst.write_bytes(0, src.as_slice());
                self.seqs[s].blocks[bi] = nb;
                self.write_entry(s, bi, nb);
                self.release_block(old);
            } else {
                self.pool.clear_hash(old);
            }
        }
        if missing > 0 {
            let fresh = self.pool.alloc(missing).expect("checked");
            for b in fresh {
                self.back(gpu, b)?;
                let i = self.seqs[s].blocks.len();
                self.seqs[s].blocks.push(b);
                self.write_entry(s, i, b);
            }
        }
        Ok(())
    }

    pub fn truncate(&mut self, s: usize, n: usize) {
        let n = n.min(self.seqs[s].len);
        self.seqs[s].len = n;
        let keep = n.div_ceil(BLOCK_TOKENS);
        while self.seqs[s].blocks.len() > keep {
            let b = self.seqs[s].blocks.pop().unwrap();
            self.release_block(b);
        }
    }

    pub fn clear(&mut self, s: usize) {
        self.truncate(s, 0);
    }

    /// Hash of the layout (as the CPU cache's: stable across runs, so a snapshot only moves
    /// between equal layouts).
    pub fn fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = llmario_engine_model::StableHasher::default();
        "metal".hash(&mut h);
        cfg!(target_endian = "little").hash(&mut h);
        self.layout.kv_type.hash(&mut h);
        self.layout.block_tokens.hash(&mut h);
        self.layout.block_bytes.hash(&mut h);
        self.layout.max_ctx.hash(&mut h);
        for l in &self.layout.layers {
            (l.kv_dim, l.v_dim, l.k_row, l.v_row, l.k_base(), l.v_base()).hash(&mut h);
        }
        h.finish()
    }

    /// Bytes of a snapshot of `len` tokens (blocks only: Metal runs the dense families).
    pub fn snapshot_bytes(&self, len: usize) -> usize {
        len.div_ceil(BLOCK_TOKENS) * self.layout.block_bytes
    }

    /// Write sequence `s`'s blocks straight from the shared buffers (no GPU work may be in
    /// flight).
    pub fn write_seq(&self, s: usize, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        let nb = self.seqs[s].len.div_ceil(BLOCK_TOKENS);
        for &b in &self.seqs[s].blocks[..nb] {
            w.write_all(
                self.backing[b as usize]
                    .as_ref()
                    .expect("backed")
                    .as_slice(),
            )?;
        }
        Ok(())
    }

    /// Replace sequence `s`'s state with `len` tokens of snapshot bytes; `Ok(false)` (nothing
    /// changed) when the size is wrong or the pool cannot hold it.
    pub fn read_seq(&mut self, gpu: &Gpu, s: usize, len: usize, bytes: &[u8]) -> Result<bool> {
        let bb = self.layout.block_bytes;
        let nb = len.div_ceil(BLOCK_TOKENS);
        if len > self.layout.max_ctx || bytes.len() != nb * bb {
            return Ok(false);
        }
        let own = self.seqs[s]
            .blocks
            .iter()
            .filter(|&&b| self.pool.refs(b) == 1)
            .count();
        if nb > self.pool.free_blocks() + own {
            return Ok(false);
        }
        self.clear(s);
        match self.reserve(gpu, s, len) {
            Ok(r) => r?,
            Err(_) => return Ok(false),
        }
        for i in 0..nb {
            let b = self.seqs[s].blocks[i];
            self.backing[b as usize]
                .as_ref()
                .expect("reserved")
                .write_bytes(0, &bytes[i * bb..(i + 1) * bb]);
        }
        self.seqs[s].len = len;
        Ok(true)
    }

    /// Apply residency changes before encoding (macOS 15+), or return the buffers the command
    /// must declare itself.
    pub fn prepare(&mut self) -> Vec<&Buf> {
        match &mut self.resid {
            Some(r) => {
                r.commit();
                Vec::new()
            }
            None => self.backing.iter().flatten().collect(),
        }
    }
}
