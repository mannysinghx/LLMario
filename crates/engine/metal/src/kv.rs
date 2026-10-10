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
//!
//! Sliding-window layers (Gemma 4) keep their rows in a second pool of blocks of the same 32
//! positions, addressed by absolute block number through their own table. A window block is
//! released once no future query can see it (every position in it is at least `window` behind the
//! next token), so the window layers hold at most `window + batch` positions and a short
//! conversation holds only what it wrote; a ring buffer would be charged in full at creation.

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
    /// Window blocks by absolute block number (`None`: released or never written).
    wblocks: Vec<Option<BlockId>>,
    len: usize,
}

/// The sliding-window layers' block pool.
pub(crate) struct WindowPool {
    pub window: usize,
    block_bytes: usize,
    /// Per attention layer: `(k_base, v_base)` inside a window block (`None` for paged layers).
    bases: Vec<Option<(usize, usize)>>,
    pool: BlockPool,
    backing: Vec<Option<Buf>>,
    /// `[n_seqs][bps]` u64 GPU addresses by absolute block number.
    pub table: Buf,
    pub bps: usize,
}

pub(crate) struct MetalKv {
    pub layout: KvLayout,
    pool: BlockPool,
    backing: Vec<Option<Buf>>,
    seqs: Vec<Seq>,
    /// `[n_seqs][blocks_per_seq]` u64 GPU addresses of each sequence's blocks.
    pub table: Buf,
    pub win: Option<WindowPool>,
    /// Recurrent state (Gated DeltaNet layers), one buffer per sequence created on its first
    /// token and freed when it is cleared; `rs_table[seq]` is its GPU address.
    rs: Vec<Option<Buf>>,
    pub rs_table: Buf,
    resid: Option<DynResidency>,
}

impl MetalKv {
    /// `n_batch`: most positions one call appends (bounds the live window blocks).
    pub fn new(
        gpu: &Gpu,
        spec: &ArchSpec,
        max_ctx: usize,
        n_seqs: usize,
        n_batch: usize,
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
        let win = Self::window_pool(gpu, &layout, n_batch)?;
        let rs_table = gpu.alloc(layout.n_seqs * 8)?;
        Ok(MetalKv {
            rs: (0..layout.n_seqs).map(|_| None).collect(),
            rs_table,
            pool: BlockPool::new(n_blocks, BLOCK_TOKENS, layout.block_bytes as u64),
            backing: (0..n_blocks).map(|_| None).collect(),
            seqs: (0..layout.n_seqs)
                .map(|_| Seq {
                    blocks: Vec::new(),
                    wblocks: Vec::new(),
                    len: 0,
                })
                .collect(),
            resid: gpu.dynamic_residency("llmario-kv-blocks"),
            table,
            win,
            layout,
        })
    }

    /// The window layers' pool, laid out like a paged block (each window layer's K rows then V
    /// rows, 32 positions); `None` without window layers.
    fn window_pool(gpu: &Gpu, layout: &KvLayout, n_batch: usize) -> Result<Option<WindowPool>> {
        let mut window = None;
        let mut off = 0;
        let mut bases = Vec::with_capacity(layout.layers.len());
        for l in &layout.layers {
            match l.window {
                Some(w) => {
                    if window.is_some_and(|x| x != w) {
                        return Err(crate::MetalError::Unsupported(
                            "sliding-window layers with different widths".into(),
                        ));
                    }
                    window = Some(w);
                    let k = off;
                    off += BLOCK_TOKENS * l.k_row;
                    bases.push(Some((k, off)));
                    off += BLOCK_TOKENS * l.v_row;
                }
                None => bases.push(None),
            }
        }
        let Some(window) = window else {
            return Ok(None);
        };
        let bps = layout.max_ctx.div_ceil(BLOCK_TOKENS);
        // Live blocks of one sequence: the window behind the first new token plus one call's
        // tokens, and a partial block at each end.
        let per_seq = ((window + n_batch).div_ceil(BLOCK_TOKENS) + 1).min(bps);
        let n = layout.n_seqs * per_seq;
        Ok(Some(WindowPool {
            window,
            block_bytes: off,
            bases,
            pool: BlockPool::new(n, BLOCK_TOKENS, off as u64),
            backing: (0..n).map(|_| None).collect(),
            table: gpu.alloc(layout.n_seqs * bps * 8)?,
            bps,
        }))
    }

    /// `(k_base, v_base)` of window layer `l` inside a window block.
    pub fn window_bases(&self, l: usize) -> Option<(usize, usize)> {
        self.win.as_ref().and_then(|w| w.bases[l])
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
        let rs = (self.rs.iter().filter(|b| b.is_some()).count() * self.layout.recurrent_bytes)
            as u64
            + self.rs_table.len() as u64;
        let win = rs
            + self.win.as_ref().map_or(0, |w| {
                (w.backing.iter().filter(|b| b.is_some()).count() * w.block_bytes + w.table.len())
                    as u64
            });
        (self.resident_blocks() * self.layout.block_bytes) as u64 + self.table.len() as u64 + win
    }
    pub fn reserved_bytes(&self) -> u64 {
        // The window layers live in the window pool, not in rings.
        let rings = (self.layout.n_seqs * self.layout.ring_bytes) as u64;
        let win = self.win.as_ref().map_or(0, |w| {
            (w.backing.len() * w.block_bytes + w.table.len()) as u64
        });
        self.layout.reserved_bytes() - rings
            + self.table.len() as u64
            + win
            + self.rs_table.len() as u64
    }

    /// Create sequence `s`'s recurrent buffer (zeroed) if the model has recurrent layers.
    fn ensure_recurrent(&mut self, gpu: &Gpu, s: usize) -> Result<()> {
        let bytes = self.layout.recurrent_bytes;
        if bytes == 0 || self.rs[s].is_some() {
            return Ok(());
        }
        let buf = gpu.alloc(bytes)?;
        buf.write_bytes(0, &vec![0u8; bytes]);
        self.rs_table
            .write_bytes(s * 8, &buf.gpu_address().to_le_bytes());
        if let Some(r) = &mut self.resid {
            r.add(&buf);
        }
        self.rs[s] = Some(buf);
        Ok(())
    }

    fn drop_recurrent(&mut self, s: usize) {
        if let Some(buf) = self.rs[s].take() {
            if let Some(r) = &mut self.resid {
                r.remove(&buf);
            }
        }
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
        if let Err(e) = self.reserve_window(gpu, s, len, new_len) {
            return Ok(Err(e));
        }
        if new_len > 0 {
            if let Err(e) = self.ensure_recurrent(gpu, s) {
                return Ok(Err(e));
            }
        }
        Ok(self.reserve_checked(gpu, s, len, new_len, missing, wanted > missing))
    }

    /// Absolute window blocks `[first, end)` a query after `len` cached positions can see.
    fn window_span(&self, len: usize) -> (usize, usize) {
        match &self.win {
            Some(w) => {
                let first = (len + 1).saturating_sub(w.window) / BLOCK_TOKENS;
                (first, len.div_ceil(BLOCK_TOKENS).max(first))
            }
            None => (0, 0),
        }
    }

    /// Release the window blocks no query from position `len` on can see, then back the blocks
    /// up to `new_len`.
    fn reserve_window(&mut self, gpu: &Gpu, s: usize, len: usize, new_len: usize) -> Result<()> {
        let Some(w) = self.win.as_mut() else {
            return Ok(());
        };
        // The query at `len` sees keys p > len − window.
        let first_live = (len + 1).saturating_sub(w.window) / BLOCK_TOKENS;
        let seq = &mut self.seqs[s];
        for slot in seq.wblocks.iter_mut().take(first_live) {
            if let Some(b) = slot.take() {
                w.pool.release(b);
                if let Some(buf) = w.backing[b as usize].take() {
                    if let Some(r) = &mut self.resid {
                        r.remove(&buf);
                    }
                }
            }
        }
        let need = new_len.div_ceil(BLOCK_TOKENS);
        while seq.wblocks.len() < need {
            let i = seq.wblocks.len();
            let b = w
                .pool
                .alloc(1)
                .ok_or_else(|| crate::MetalError::Device("window block pool exhausted".into()))?[0];
            if w.backing[b as usize].is_none() {
                let buf = gpu.alloc(w.block_bytes)?;
                if let Some(r) = &mut self.resid {
                    r.add(&buf);
                }
                w.backing[b as usize] = Some(buf);
            }
            let addr = w.backing[b as usize].as_ref().unwrap().gpu_address();
            w.table
                .write_bytes((s * w.bps + i) * 8, &addr.to_le_bytes());
            seq.wblocks.push(Some(b));
        }
        Ok(())
    }

    /// Release sequence `s`'s window blocks from absolute block `from` on.
    fn release_window_from(&mut self, s: usize, from: usize) {
        let Some(w) = self.win.as_mut() else {
            return;
        };
        let seq = &mut self.seqs[s];
        while seq.wblocks.len() > from {
            if let Some(b) = seq.wblocks.pop().unwrap() {
                w.pool.release(b);
                if let Some(buf) = w.backing[b as usize].take() {
                    if let Some(r) = &mut self.resid {
                        r.remove(&buf);
                    }
                }
            }
        }
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

    /// Keep the first `n` positions of sequence `s`. When a window block the next query would
    /// need was already released, the whole sequence is cleared instead (read `len` afterwards).
    pub fn truncate(&mut self, s: usize, mut n: usize) {
        n = n.min(self.seqs[s].len);
        // A recurrent state cannot be rewound: any cut resets the sequence.
        if self.layout.recurrent_bytes > 0 && n < self.seqs[s].len {
            n = 0;
        }
        if let Some(w) = &self.win {
            let first_needed = (n + 1).saturating_sub(w.window) / BLOCK_TOKENS;
            let last = n.div_ceil(BLOCK_TOKENS);
            if n > 0
                && (first_needed..last)
                    .any(|i| self.seqs[s].wblocks.get(i).map_or(true, |b| b.is_none()))
            {
                n = 0;
            }
        }
        self.seqs[s].len = n;
        let keep = n.div_ceil(BLOCK_TOKENS);
        while self.seqs[s].blocks.len() > keep {
            let b = self.seqs[s].blocks.pop().unwrap();
            self.release_block(b);
        }
        self.release_window_from(s, keep);
        if n == 0 {
            self.drop_recurrent(s);
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
        if let Some(w) = &self.win {
            (w.window, w.block_bytes, &w.bases).hash(&mut h);
        }
        self.layout.recurrent_bytes.hash(&mut h);
        h.finish()
    }

    /// Bytes of a snapshot of `len` tokens: the paged blocks, then the window blocks a query
    /// after `len` can still see.
    pub fn snapshot_bytes(&self, len: usize) -> usize {
        let (first, end) = self.window_span(len);
        let wb = self.win.as_ref().map_or(0, |w| w.block_bytes);
        if len == 0 {
            return 0;
        }
        len.div_ceil(BLOCK_TOKENS) * self.layout.block_bytes
            + (end - first) * wb
            + self.layout.recurrent_bytes
    }

    /// Write sequence `s`'s blocks straight from the shared buffers (no GPU work may be in
    /// flight).
    pub fn write_seq(&self, s: usize, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        let len = self.seqs[s].len;
        let nb = len.div_ceil(BLOCK_TOKENS);
        if self.layout.block_bytes > 0 {
            for &b in &self.seqs[s].blocks[..nb] {
                w.write_all(
                    self.backing[b as usize]
                        .as_ref()
                        .expect("backed")
                        .as_slice(),
                )?;
            }
        }
        if let Some(win) = &self.win {
            let (first, end) = self.window_span(len);
            for i in first..end {
                let b = self.seqs[s].wblocks[i].expect("a visible window block is live");
                w.write_all(win.backing[b as usize].as_ref().expect("backed").as_slice())?;
            }
        }
        if len > 0 {
            if let Some(rs) = &self.rs[s] {
                w.write_all(rs.as_slice())?;
            }
        }
        Ok(())
    }

    /// Replace sequence `s`'s state with `len` tokens of snapshot bytes; `Ok(false)` (nothing
    /// changed) when the size is wrong or the pool cannot hold it.
    pub fn read_seq(&mut self, gpu: &Gpu, s: usize, len: usize, bytes: &[u8]) -> Result<bool> {
        let bb = self.layout.block_bytes;
        let nb = if bb > 0 {
            len.div_ceil(BLOCK_TOKENS)
        } else {
            0
        };
        if len > self.layout.max_ctx || bytes.len() != self.snapshot_bytes(len) {
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
        // Paged blocks only: the window blocks are placed below, at their absolute numbers.
        self.reserve_checked(gpu, s, 0, len, nb, false)?;
        for i in 0..nb {
            let b = self.seqs[s].blocks[i];
            self.backing[b as usize]
                .as_ref()
                .expect("reserved")
                .write_bytes(0, &bytes[i * bb..(i + 1) * bb]);
        }
        let (first, end) = self.window_span(len);
        if let Some(w) = self.win.as_mut() {
            let wb = w.block_bytes;
            let seq = &mut self.seqs[s];
            seq.wblocks = vec![None; first];
            for (k, i) in (first..end).enumerate() {
                let b = w.pool.alloc(1).ok_or_else(|| {
                    crate::MetalError::Device("window block pool exhausted".into())
                })?[0];
                if w.backing[b as usize].is_none() {
                    let buf = gpu.alloc(wb)?;
                    if let Some(r) = &mut self.resid {
                        r.add(&buf);
                    }
                    w.backing[b as usize] = Some(buf);
                }
                let buf = w.backing[b as usize].as_ref().unwrap();
                let at = nb * bb + k * wb;
                buf.write_bytes(0, &bytes[at..at + wb]);
                w.table
                    .write_bytes((s * w.bps + i) * 8, &buf.gpu_address().to_le_bytes());
                seq.wblocks.push(Some(b));
            }
        }
        if len > 0 && self.layout.recurrent_bytes > 0 {
            self.ensure_recurrent(gpu, s)?;
            let at = bytes.len() - self.layout.recurrent_bytes;
            self.rs[s].as_ref().unwrap().write_bytes(0, &bytes[at..]);
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
            None => {
                let win = self.win.iter().flat_map(|w| w.backing.iter().flatten());
                let rs = self.rs.iter().flatten();
                self.backing.iter().flatten().chain(win).chain(rs).collect()
            }
        }
    }
}
