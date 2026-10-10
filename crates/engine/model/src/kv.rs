//! Paged KV cache for the CPU backend (Architecture §8.1–§8.3).
//!
//! # Storage classes
//!
//! - **Full-attention layers are paged.** Positions live in blocks of [`KvLayout::block_tokens`]
//!   positions; one block holds those positions for *every* full-attention layer (each layer's K
//!   rows, then its V rows). A sequence owns a block table. A block is backed by memory only
//!   while a sequence uses it (or while it holds published prefix-cache content): it is mapped
//!   when a sequence first reaches it and returned to the OS when released, so the memory held
//!   follows the tokens actually cached, not the context the plan admitted. Block ids,
//!   reference counts, the LRU free queue and content hashes come from
//!   [`llmario_engine_kv::BlockPool`]; this module adds the backing memory.
//! - **Sliding-window layers** (the "Window" class) keep one ring of
//!   `cap = min(max_ctx, n_swa + n_batch)` positions per sequence: position `p` lives in slot
//!   `p % cap`. The `n_batch` headroom exists because a prefill chunk of `n` tokens at
//!   `pos0..pos0 + n` needs positions `pos0 − n_swa + 1 ..= pos0 + n − 1` alive at once, so a
//!   forward pass appends at most [`KvCache::max_batch`] tokens per sequence (llama.cpp's iSWA
//!   cache uses the same `n_swa + n_batch` sizing, research note KQ1, PR #13194). Rings live in
//!   a lazily backed region per sequence, so a short conversation touches only the slots it
//!   wrote.
//! - **Gated DeltaNet layers** keep fp32 recurrent state per sequence ([`RecurrentState`]).
//!
//! # Element types
//!
//! [`KvType::F16`] (llama.cpp's default cache type) or [`KvType::Q8_0`]: ggml's `block_q8_0`,
//! an f16 scale and 32 signed bytes per 32 values (34 bytes per 32 values, 53 % of f16),
//! quantised exactly as ggml-cpu's SIMD `quantize_row_q8_0` does (`d = amax / 127`, values
//! `round_ties_even(x / d)`), the near-lossless 8-bit cache of Architecture §8.3.
//!
//! # Lazily backed memory
//!
//! Blocks and rings are anonymous, zero-filled mappings ([`Region`]): the OS supplies a page on
//! the first write and takes the whole mapping back on release. Nothing is zero-filled up front
//! (an eager `vec![f16::ZERO; n]` writes every page and made the whole admitted context
//! resident at load: 3.6 GiB for Qwen3-1.7B at 32K before any request).

use crate::arch::{ArchSpec, BlockKind};
use half::f16;
use half::slice::HalfFloatSliceExt;
use llmario_engine_kv::{BlockId, BlockPool};
use serde::{Deserialize, Serialize};
use std::ptr::NonNull;

/// Default prefill headroom added to a sliding-window ring (`cap = n_swa + SWA_RING_BATCH`).
pub const SWA_RING_BATCH: usize = 512;

/// Default block size of the paged layers (Architecture §8.2: 32 tokens on CPU and Metal).
pub const BLOCK_TOKENS: usize = 32;

/// Bytes per cached K/V element of the default cache type (f16); see [`KvType::bytes_per_elem`].
pub const KV_ELEM_BYTES: f64 = 2.0;

/// Element type of the cached K/V rows.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
pub enum KvType {
    #[default]
    #[serde(rename = "f16")]
    F16,
    #[serde(rename = "q8_0")]
    Q8_0,
}

/// Values per q8_0 block and bytes per block (f16 scale + 32 × i8).
pub const Q8_0_BLOCK: usize = 32;
pub const Q8_0_BYTES: usize = 34;

impl KvType {
    pub fn name(self) -> &'static str {
        match self {
            KvType::F16 => "f16",
            KvType::Q8_0 => "q8_0",
        }
    }

    pub fn parse(s: &str) -> Option<KvType> {
        match s.to_ascii_lowercase().as_str() {
            "f16" => Some(KvType::F16),
            "q8_0" | "q8" => Some(KvType::Q8_0),
            _ => None,
        }
    }

    /// Average bytes per element (2 for f16, 34/32 for q8_0).
    pub fn bytes_per_elem(self) -> f64 {
        match self {
            KvType::F16 => 2.0,
            KvType::Q8_0 => Q8_0_BYTES as f64 / Q8_0_BLOCK as f64,
        }
    }

    /// Bytes of one cached row of `n` elements.
    pub fn row_bytes(self, n: usize) -> usize {
        match self {
            KvType::F16 => 2 * n,
            KvType::Q8_0 => n.div_ceil(Q8_0_BLOCK) * Q8_0_BYTES,
        }
    }

    /// Whether rows whose heads are `head_dim` wide can use this type (q8_0 blocks must not
    /// straddle heads, because attention reads one head of a row at a time).
    pub fn supports_head_dim(self, head_dim: usize) -> bool {
        match self {
            KvType::F16 => true,
            KvType::Q8_0 => head_dim % Q8_0_BLOCK == 0,
        }
    }

    /// Encode `src` into `dst` (`row_bytes(src.len())` bytes).
    #[inline]
    pub fn store(self, src: &[f32], dst: &mut [u8]) {
        match self {
            KvType::F16 => {
                debug_assert_eq!(dst.len(), 2 * src.len());
                debug_assert_eq!(dst.as_ptr() as usize % 2, 0);
                // SAFETY: rows start at even offsets inside page-aligned regions (checked above),
                // so the bytes are a valid, aligned `[f16]` of `src.len()` elements.
                let d = unsafe {
                    std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f16, src.len())
                };
                d.convert_from_f32_slice(src);
            }
            KvType::Q8_0 => quantize_q8_0(src, dst),
        }
    }

    /// Decode elements `start..start + out.len()` of the row in `src` (for q8_0, `start` and
    /// `out.len()` are multiples of 32).
    #[inline]
    pub fn load(self, src: &[u8], start: usize, out: &mut [f32]) {
        match self {
            KvType::F16 => {
                debug_assert_eq!(src.as_ptr() as usize % 2, 0);
                // SAFETY: see `store`; the slice covers `start + out.len()` f16 elements.
                let s = unsafe {
                    std::slice::from_raw_parts(src.as_ptr().add(2 * start) as *const f16, out.len())
                };
                s.convert_to_f32_slice(out);
            }
            KvType::Q8_0 => {
                debug_assert_eq!(start % Q8_0_BLOCK, 0);
                debug_assert_eq!(out.len() % Q8_0_BLOCK, 0);
                let b0 = start / Q8_0_BLOCK;
                dequantize_q8_0(
                    &src[b0 * Q8_0_BYTES..(b0 + out.len() / Q8_0_BLOCK) * Q8_0_BYTES],
                    out,
                );
            }
        }
    }
}

/// ggml-cpu `quantize_row_q8_0` (NEON / AVX2 paths): per 32 values `d = amax / 127`,
/// `id = d ? 1/d : 0`, `qs = round_ties_even(x · id)`, `d` stored as f16.
pub fn quantize_q8_0(src: &[f32], dst: &mut [u8]) {
    debug_assert_eq!(src.len() % Q8_0_BLOCK, 0);
    debug_assert_eq!(dst.len(), src.len() / Q8_0_BLOCK * Q8_0_BYTES);
    for (blk, out) in src
        .chunks_exact(Q8_0_BLOCK)
        .zip(dst.chunks_exact_mut(Q8_0_BYTES))
    {
        let amax = blk.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out[..2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        for (q, &x) in out[2..].iter_mut().zip(blk) {
            *q = ((x * id).round_ties_even() as i32) as i8 as u8;
        }
    }
}

/// Inverse of [`quantize_q8_0`]: `x = qs · d`.
pub fn dequantize_q8_0(src: &[u8], out: &mut [f32]) {
    for (blk, o) in src
        .chunks_exact(Q8_0_BYTES)
        .zip(out.chunks_exact_mut(Q8_0_BLOCK))
    {
        let d = f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
        for (v, &q) in o.iter_mut().zip(&blk[2..]) {
            *v = (q as i8) as f32 * d;
        }
    }
}

/// Anonymous, zero-filled memory whose pages the OS supplies on first write and reclaims when
/// the region is dropped (`mmap`/`munmap` on Unix; a zeroed allocation elsewhere).
pub struct Region {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: the region is plain memory owned by this value; access is governed by the borrow of
// the owning `KvCache` (writes need `&mut`).
unsafe impl Send for Region {}
unsafe impl Sync for Region {}

impl Region {
    pub fn new(len: usize) -> Region {
        if len == 0 {
            return Region {
                ptr: NonNull::dangling(),
                len: 0,
            };
        }
        #[cfg(unix)]
        {
            // SAFETY: a fresh private anonymous mapping; the kernel zero-fills pages on demand.
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANON,
                    -1,
                    0,
                )
            };
            if p == libc::MAP_FAILED {
                std::alloc::handle_alloc_error(
                    std::alloc::Layout::from_size_align(len, 4096).unwrap(),
                );
            }
            Region {
                // SAFETY: mmap returned a non-null address on success.
                ptr: unsafe { NonNull::new_unchecked(p as *mut u8) },
                len,
            }
        }
        #[cfg(not(unix))]
        {
            let layout = std::alloc::Layout::from_size_align(len, 64).unwrap();
            // SAFETY: non-zero size layout.
            let p = unsafe { std::alloc::alloc_zeroed(layout) };
            Region {
                ptr: NonNull::new(p).unwrap_or_else(|| std::alloc::handle_alloc_error(layout)),
                len,
            }
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[inline]
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }
        #[cfg(unix)]
        // SAFETY: `ptr..ptr+len` is exactly the mapping created in `new`.
        unsafe {
            libc::munmap(self.ptr.as_ptr() as *mut libc::c_void, self.len);
        }
        #[cfg(not(unix))]
        // SAFETY: allocated in `new` with this layout.
        unsafe {
            std::alloc::dealloc(
                self.ptr.as_ptr(),
                std::alloc::Layout::from_size_align(self.len, 64).unwrap(),
            );
        }
    }
}

/// The recurrent state of one Gated DeltaNet layer (see `gdn.rs` for the layouts).
pub struct RecurrentState {
    /// Causal-conv history: `[conv_dim][d_conv - 1]`, the last `d_conv - 1` pre-activation inputs
    /// per channel, oldest first.
    pub conv: Vec<f32>,
    /// Delta-rule state: `[n_v_heads][head_k][head_v]`.
    pub state: Vec<f32>,
}

/// One attention layer's storage geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvLayer {
    pub kv_dim: usize,
    pub v_dim: usize,
    /// Positions a sequence's storage for this layer holds: `max_ctx` for a paged layer, the
    /// ring size for a window layer.
    pub cap: usize,
    /// `Some(n_swa)` for a sliding-window layer (stored in the per-sequence ring).
    pub window: Option<usize>,
    /// Bytes of one K / V row.
    pub k_row: usize,
    pub v_row: usize,
    /// Byte offset of this layer's K rows inside a block (paged) or a sequence's ring region
    /// (window); its V rows follow the K rows.
    off: usize,
    /// Rows this layer has per block (paged) or per ring (window).
    rows: usize,
}

impl KvLayer {
    pub fn paged(&self) -> bool {
        self.window.is_none()
    }
    #[inline]
    fn k_off(&self, slot: usize) -> usize {
        self.off + slot * self.k_row
    }
    #[inline]
    fn v_off(&self, slot: usize) -> usize {
        self.off + self.rows * self.k_row + slot * self.v_row
    }
}

/// Construction options for a [`KvCache`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvOptions {
    /// Positions per sequence.
    pub max_ctx: usize,
    /// Sequences (slots) that can hold state at once.
    pub n_seqs: usize,
    /// Prefill headroom of the window rings.
    pub ring_batch: usize,
    pub kv_type: KvType,
    /// Block size of the paged layers (a power of two).
    pub block_tokens: usize,
    /// Positions the shared pool holds across all sequences (`None`: every sequence can hold
    /// `max_ctx` at once). A smaller pool lets sequences share one budget: any one of them may
    /// still grow to `max_ctx` while the others are short.
    pub pool_tokens: Option<usize>,
}

impl KvOptions {
    pub fn new(max_ctx: usize) -> KvOptions {
        KvOptions {
            max_ctx,
            n_seqs: 1,
            ring_batch: SWA_RING_BATCH,
            kv_type: KvType::F16,
            block_tokens: BLOCK_TOKENS,
            pool_tokens: None,
        }
    }
    pub fn seqs(mut self, n: usize) -> Self {
        self.n_seqs = n.max(1);
        self
    }
    pub fn ring_batch(mut self, n: usize) -> Self {
        self.ring_batch = n;
        self
    }
    pub fn kv_type(mut self, t: KvType) -> Self {
        self.kv_type = t;
        self
    }
    pub fn block_tokens(mut self, b: usize) -> Self {
        self.block_tokens = b;
        self
    }
    pub fn pool_tokens(mut self, n: usize) -> Self {
        self.pool_tokens = Some(n);
        self
    }
}

/// Geometry and byte sizes of the cache for one model and option set (what the plan charges).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvLayout {
    pub kv_type: KvType,
    pub max_ctx: usize,
    pub block_tokens: usize,
    /// One entry per attention layer, in attention-layer order.
    pub layers: Vec<KvLayer>,
    /// Bytes of one block (all paged layers, K and V, `block_tokens` positions); 0 when the
    /// model has no full-attention layer.
    pub block_bytes: usize,
    /// Bytes of one sequence's ring region (all window layers).
    pub ring_bytes: usize,
    /// Bytes of one sequence's fp32 recurrent state.
    pub recurrent_bytes: usize,
    /// Blocks one sequence needs to hold `max_ctx` positions.
    pub blocks_per_seq: usize,
    /// Sequences (slots).
    pub n_seqs: usize,
    /// Blocks in the shared pool (0 when the model has no paged layer): enough for the pool's
    /// positions plus one copy-on-write spare per sequence.
    pub pool_blocks: usize,
}

impl KvLayout {
    pub fn new(spec: &ArchSpec, o: &KvOptions) -> KvLayout {
        assert!(
            o.block_tokens.is_power_of_two(),
            "block size must be a power of two"
        );
        let max_ctx = o.max_ctx.max(1);
        let mut layers = Vec::new();
        let (mut block_off, mut ring_off) = (0usize, 0usize);
        for l in 0..spec.n_layer as usize {
            if spec.blocks[l] != BlockKind::Attention {
                continue;
            }
            let g = spec.attn_geom(l);
            assert!(
                o.kv_type.supports_head_dim(g.head_dim as usize)
                    && o.kv_type.supports_head_dim(g.head_dim_v as usize),
                "KV cache type {} cannot hold layer {l}'s {}/{}-wide heads",
                o.kv_type.name(),
                g.head_dim,
                g.head_dim_v
            );
            let (kv_dim, v_dim) = (g.kv_dim() as usize, g.v_dim() as usize);
            let (k_row, v_row) = (o.kv_type.row_bytes(kv_dim), o.kv_type.row_bytes(v_dim));
            let layer = match g.window {
                Some(w) => {
                    let cap = max_ctx.min(w as usize + o.ring_batch);
                    let lay = KvLayer {
                        kv_dim,
                        v_dim,
                        cap,
                        window: Some(w as usize),
                        k_row,
                        v_row,
                        off: ring_off,
                        rows: cap,
                    };
                    ring_off += cap * (k_row + v_row);
                    lay
                }
                None => {
                    let lay = KvLayer {
                        kv_dim,
                        v_dim,
                        cap: max_ctx,
                        window: None,
                        k_row,
                        v_row,
                        off: block_off,
                        rows: o.block_tokens,
                    };
                    block_off += o.block_tokens * (k_row + v_row);
                    lay
                }
            };
            layers.push(layer);
        }
        let recurrent_bytes = match &spec.gdn {
            Some(g) => {
                (g.conv_state_len() as usize + g.state_len() as usize)
                    * 4
                    * spec.n_recurrent_layers() as usize
            }
            None => 0,
        };
        let n_seqs = o.n_seqs.max(1);
        let blocks_per_seq = if block_off > 0 {
            max_ctx.div_ceil(o.block_tokens)
        } else {
            0
        };
        let pool_blocks = if block_off > 0 {
            match o.pool_tokens {
                Some(t) => {
                    t.max(1)
                        .div_ceil(o.block_tokens)
                        .min(n_seqs * blocks_per_seq)
                        + n_seqs
                }
                None => n_seqs * (blocks_per_seq + 1),
            }
        } else {
            0
        };
        KvLayout {
            kv_type: o.kv_type,
            max_ctx,
            block_tokens: o.block_tokens,
            layers,
            block_bytes: block_off,
            ring_bytes: ring_off,
            recurrent_bytes,
            blocks_per_seq,
            n_seqs,
            pool_blocks,
        }
    }

    /// Paged bytes per cached position (all full-attention layers).
    pub fn paged_bytes_per_token(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| l.paged())
            .map(|l| l.k_row + l.v_row)
            .sum()
    }

    /// Bytes of the shared block pool when every block is backed.
    pub fn pool_bytes(&self) -> u64 {
        (self.pool_blocks * self.block_bytes) as u64
    }

    /// Everything the cache can ever hold: every pool block backed, plus each sequence's window
    /// rings and recurrent state (what the plan charges; the cache never exceeds it).
    pub fn reserved_bytes(&self) -> u64 {
        self.pool_bytes() + (self.n_seqs * (self.ring_bytes + self.recurrent_bytes)) as u64
    }

    /// Largest number of tokens one forward call may append to one sequence without a window
    /// ring overwriting a position a token of the same batch still needs (`usize::MAX` when no
    /// ring wraps).
    pub fn max_batch(&self) -> usize {
        self.layers
            .iter()
            .filter_map(|l| {
                let w = l.window?;
                (l.cap < self.max_ctx).then(|| (l.cap + 1).saturating_sub(w).max(1))
            })
            .min()
            .unwrap_or(usize::MAX)
    }
}

/// One sequence's cache state.
pub struct SeqKv {
    /// Block table of the paged layers, in position order.
    blocks: Vec<BlockId>,
    /// Tokens cached (positions `0..len`, modulo each ring's eviction).
    pub len: usize,
    /// Window-layer rings (lazily backed).
    ring: Region,
    /// One entry per Gated DeltaNet layer, in layer order (empty for the other families).
    pub rs: Vec<RecurrentState>,
}

impl SeqKv {
    pub fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }
}

/// The cache could not grow: every block of the plan's arena is in use.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("KV cache full: {needed} more blocks needed, {free} free")]
pub struct KvFull {
    pub needed: usize,
    pub free: usize,
}

/// Paged KV cache shared by a fixed set of sequences (slots).
pub struct KvCache {
    pub layout: KvLayout,
    pool: BlockPool,
    /// Backing memory per block id (`None` while the block holds nothing).
    backing: Vec<Option<Region>>,
    seqs: Vec<SeqKv>,
}

impl KvCache {
    /// A single-sequence f16 cache for `max_ctx` positions (rings with [`SWA_RING_BATCH`]
    /// headroom).
    pub fn new(spec: &ArchSpec, max_ctx: usize) -> KvCache {
        Self::with_options(spec, KvOptions::new(max_ctx))
    }

    pub fn with_options(spec: &ArchSpec, o: KvOptions) -> KvCache {
        let layout = KvLayout::new(spec, &o);
        let n_seqs = layout.n_seqs;
        let n_blocks = layout.pool_blocks;
        let pool = BlockPool::new(n_blocks, layout.block_tokens, layout.block_bytes as u64);
        let seqs = (0..n_seqs)
            .map(|_| SeqKv {
                blocks: Vec::new(),
                len: 0,
                ring: Region::new(layout.ring_bytes),
                rs: match &spec.gdn {
                    Some(g) => (0..spec.n_recurrent_layers())
                        .map(|_| RecurrentState {
                            conv: vec![0f32; g.conv_state_len() as usize],
                            state: vec![0f32; g.state_len() as usize],
                        })
                        .collect(),
                    None => Vec::new(),
                },
            })
            .collect();
        KvCache {
            backing: (0..n_blocks).map(|_| None).collect(),
            layout,
            pool,
            seqs,
        }
    }

    /// Bytes a single-sequence f16 cache for `max_ctx` positions reserves (paged blocks, rings,
    /// recurrent state).
    pub fn bytes(spec: &ArchSpec, max_ctx: usize) -> u64 {
        KvLayout::new(spec, &KvOptions::new(max_ctx)).reserved_bytes()
    }

    /// Bytes of fp32 recurrent state per sequence (0 for the dense families).
    pub fn recurrent_bytes(spec: &ArchSpec) -> u64 {
        KvLayout::new(spec, &KvOptions::new(1)).recurrent_bytes as u64
    }

    pub fn n_seqs(&self) -> usize {
        self.seqs.len()
    }
    pub fn max_ctx(&self) -> usize {
        self.layout.max_ctx
    }
    pub fn kv_type(&self) -> KvType {
        self.layout.kv_type
    }
    /// Number of attention layers.
    pub fn n_layers(&self) -> usize {
        self.layout.layers.len()
    }
    pub fn layer(&self, l: usize) -> &KvLayer {
        &self.layout.layers[l]
    }
    pub fn len(&self, s: usize) -> usize {
        self.seqs[s].len
    }
    pub fn is_empty(&self, s: usize) -> bool {
        self.seqs[s].len == 0
    }
    pub fn seq(&self, s: usize) -> &SeqKv {
        &self.seqs[s]
    }
    pub fn seq_mut(&mut self, s: usize) -> &mut SeqKv {
        &mut self.seqs[s]
    }
    pub fn pool(&self) -> &BlockPool {
        &self.pool
    }
    pub fn pool_mut(&mut self) -> &mut BlockPool {
        &mut self.pool
    }
    pub fn max_batch(&self) -> usize {
        self.layout.max_batch()
    }

    /// Blocks currently backed by memory (in use or holding cached content).
    pub fn resident_blocks(&self) -> usize {
        self.backing.iter().filter(|b| b.is_some()).count()
    }

    /// Bytes the cache holds right now: backed blocks, plus every sequence's ring and recurrent
    /// state counted in full (rings are lazily backed, so this is an upper bound for them).
    pub fn in_use_bytes(&self) -> u64 {
        (self.resident_blocks() * self.layout.block_bytes) as u64
            + (self.seqs.len() * (self.layout.ring_bytes + self.layout.recurrent_bytes)) as u64
    }

    fn back(&mut self, b: BlockId) {
        let slot = &mut self.backing[b as usize];
        if slot.is_none() {
            *slot = Some(Region::new(self.layout.block_bytes));
        }
    }

    /// Return block `b`'s memory to the OS when nothing references or caches it any more.
    fn maybe_unback(&mut self, b: BlockId) {
        if self.pool.refs(b) == 0 && !self.pool.is_cached(b) {
            self.backing[b as usize] = None;
        }
    }

    fn release_block(&mut self, b: BlockId) {
        self.pool.release(b);
        self.maybe_unback(b);
    }

    /// Free blocks [`KvCache::reserve`] would take to grow sequence `s` to `new_len` positions
    /// (missing blocks, plus one when the partially filled boundary block is shared).
    pub fn blocks_needed(&self, s: usize, new_len: usize) -> usize {
        if self.layout.block_bytes == 0 {
            return 0;
        }
        let bt = self.layout.block_tokens;
        let seq = &self.seqs[s];
        let missing = new_len.div_ceil(bt).saturating_sub(seq.blocks.len());
        let cow =
            seq.len % bt != 0 && seq.len < new_len && self.pool.refs(seq.blocks[seq.len / bt]) > 1;
        missing + cow as usize
    }

    /// Make room to append up to `new_len` positions to sequence `s`: allocate the missing
    /// blocks and make the block that will receive position `len` private (copy-on-write when
    /// it is shared; its published hash is withdrawn because its content is about to change).
    pub fn reserve(&mut self, s: usize, new_len: usize) -> Result<(), KvFull> {
        assert!(
            new_len <= self.layout.max_ctx,
            "context overflow: {new_len} > {}",
            self.layout.max_ctx
        );
        if self.layout.block_bytes == 0 {
            return Ok(());
        }
        let bt = self.layout.block_tokens;
        let len = self.seqs[s].len;
        let missing = new_len
            .div_ceil(bt)
            .saturating_sub(self.seqs[s].blocks.len());
        let wanted = self.blocks_needed(s, new_len);
        // Copy-on-write of the partially filled boundary block.
        let cow = wanted > missing;
        if self.pool.free_blocks() < wanted {
            return Err(KvFull {
                needed: wanted,
                free: self.pool.free_blocks(),
            });
        }
        if len % bt != 0 && len < new_len {
            let bi = len / bt;
            let old = self.seqs[s].blocks[bi];
            if cow {
                let nb = self.pool.alloc(1).expect("checked above")[0];
                self.back(nb);
                let n = self.layout.block_bytes;
                // SAFETY: two distinct backed blocks of `block_bytes` each.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.backing[old as usize].as_ref().unwrap().as_ptr(),
                        self.backing[nb as usize].as_ref().unwrap().as_ptr(),
                        n,
                    );
                }
                self.seqs[s].blocks[bi] = nb;
                self.release_block(old);
            } else {
                self.pool.clear_hash(old);
            }
        }
        if missing > 0 {
            let fresh = self.pool.alloc(missing).expect("checked above");
            for b in fresh {
                self.back(b);
                self.seqs[s].blocks.push(b);
            }
        }
        Ok(())
    }

    /// Keep the first `n` tokens of sequence `s` (prefix reuse / retry). A recurrent state
    /// cannot be rewound without checkpoints, so for hybrid families any `n < len` resets the
    /// whole sequence (`len == 0` afterwards); likewise when a window ring has already evicted a
    /// position the rewound sequence would need (`len − n > cap − n_swa + 1`). Callers re-read
    /// `len` and recompute. Blocks wholly past `n` are released.
    pub fn truncate(&mut self, s: usize, n: usize) {
        let len = self.seqs[s].len;
        if n < len {
            if !self.seqs[s].rs.is_empty() {
                self.clear(s);
                return;
            }
            let max_ctx = self.layout.max_ctx;
            let evicted = self.layout.layers.iter().any(|l| match l.window {
                Some(w) if l.cap < max_ctx => len - n > (l.cap + 1).saturating_sub(w),
                _ => false,
            });
            if evicted {
                self.clear(s);
                return;
            }
        }
        let n = n.min(len);
        self.seqs[s].len = n;
        let keep = n.div_ceil(self.layout.block_tokens.max(1));
        while self.seqs[s].blocks.len() > keep {
            let b = self.seqs[s].blocks.pop().unwrap();
            self.release_block(b);
        }
    }

    /// Forget sequence `s` entirely (blocks released, rings and recurrent state reset).
    pub fn clear(&mut self, s: usize) {
        self.seqs[s].len = 0;
        while let Some(b) = self.seqs[s].blocks.pop() {
            self.release_block(b);
        }
        // Drop the ring's pages too (a fresh lazily backed region).
        if self.layout.ring_bytes > 0 {
            self.seqs[s].ring = Region::new(self.layout.ring_bytes);
        }
        for r in &mut self.seqs[s].rs {
            r.conv.iter_mut().for_each(|v| *v = 0.0);
            r.state.iter_mut().for_each(|v| *v = 0.0);
        }
    }

    /// Slot of position `pos` in layer `l`'s storage (the ring slot for a window layer, the
    /// position itself for a paged layer).
    #[inline]
    pub fn slot(&self, l: usize, pos: usize) -> usize {
        pos % self.layout.layers[l].cap
    }

    /// Byte address of the K (`v == false`) or V row of position `pos` in layer `l` of
    /// sequence `s`. The block must have been reserved.
    #[inline]
    fn row_ptr(&self, s: usize, l: usize, pos: usize, v: bool) -> *mut u8 {
        let lay = &self.layout.layers[l];
        let seq = &self.seqs[s];
        if lay.paged() {
            let bt = self.layout.block_tokens;
            let b = seq.blocks[pos / bt];
            let base = self.backing[b as usize]
                .as_ref()
                .expect("reserved block")
                .as_ptr();
            let off = if v {
                lay.v_off(pos % bt)
            } else {
                lay.k_off(pos % bt)
            };
            // SAFETY: `off + row` lies inside the block (layout invariant).
            unsafe { base.add(off) }
        } else {
            let slot = pos % lay.cap;
            let off = if v { lay.v_off(slot) } else { lay.k_off(slot) };
            // SAFETY: `off + row` lies inside the ring region (layout invariant).
            unsafe { seq.ring.as_ptr().add(off) }
        }
    }

    /// Store the key row of position `pos` (layer `l`, sequence `s`), encoded as the cache type.
    #[inline]
    pub fn store_k(&mut self, s: usize, l: usize, pos: usize, k: &[f32]) {
        let lay = self.layout.layers[l];
        debug_assert_eq!(k.len(), lay.kv_dim);
        let p = self.row_ptr(s, l, pos, false);
        // SAFETY: `&mut self` gives exclusive access; the row is `k_row` bytes inside a region
        // this cache owns.
        let dst = unsafe { std::slice::from_raw_parts_mut(p, lay.k_row) };
        self.layout.kv_type.store(k, dst);
    }

    /// Store the value row of position `pos`.
    #[inline]
    pub fn store_v(&mut self, s: usize, l: usize, pos: usize, v: &[f32]) {
        let lay = self.layout.layers[l];
        debug_assert_eq!(v.len(), lay.v_dim);
        let p = self.row_ptr(s, l, pos, true);
        // SAFETY: as in `store_k`.
        let dst = unsafe { std::slice::from_raw_parts_mut(p, lay.v_row) };
        self.layout.kv_type.store(v, dst);
    }

    /// Decode elements `start..start + out.len()` of the cached K row of `pos`.
    #[inline]
    pub fn load_k(&self, s: usize, l: usize, pos: usize, start: usize, out: &mut [f32]) {
        let lay = &self.layout.layers[l];
        let p = self.row_ptr(s, l, pos, false);
        // SAFETY: shared borrow; the row is `k_row` bytes inside an owned region.
        let row = unsafe { std::slice::from_raw_parts(p, lay.k_row) };
        self.layout.kv_type.load(row, start, out);
    }

    #[inline]
    pub fn load_v(&self, s: usize, l: usize, pos: usize, start: usize, out: &mut [f32]) {
        let lay = &self.layout.layers[l];
        let p = self.row_ptr(s, l, pos, true);
        // SAFETY: as in `load_k`.
        let row = unsafe { std::slice::from_raw_parts(p, lay.v_row) };
        self.layout.kv_type.load(row, start, out);
    }

    /// A read-only view of one (sequence, layer) for the attention loop: block base addresses
    /// resolved once, so each row lookup is a shift, a mask and an add.
    pub(crate) fn reader(&self, s: usize, l: usize) -> LayerReader {
        let lay = self.layout.layers[l];
        let seq = &self.seqs[s];
        let bases: Vec<usize> = if lay.paged() {
            seq.blocks
                .iter()
                .map(|&b| {
                    self.backing[b as usize]
                        .as_ref()
                        .expect("reserved block")
                        .as_ptr() as usize
                })
                .collect()
        } else {
            vec![seq.ring.as_ptr() as usize]
        };
        LayerReader {
            kv_type: self.layout.kv_type,
            lay,
            bases,
            shift: self.layout.block_tokens.trailing_zeros(),
            mask: self.layout.block_tokens - 1,
        }
    }
}

/// See [`KvCache::reader`]. Holds raw addresses into regions the cache owns; it is only built
/// and used while the cache is borrowed immutably (no writes can happen meanwhile).
pub(crate) struct LayerReader {
    kv_type: KvType,
    lay: KvLayer,
    /// Block base addresses (paged) or the ring base (window).
    bases: Vec<usize>,
    shift: u32,
    mask: usize,
}

impl LayerReader {
    #[inline]
    fn row(&self, pos: usize, v: bool) -> &[u8] {
        let (base, slot) = if self.lay.paged() {
            (self.bases[pos >> self.shift], pos & self.mask)
        } else {
            (self.bases[0], pos % self.lay.cap)
        };
        let (off, len) = if v {
            (self.lay.v_off(slot), self.lay.v_row)
        } else {
            (self.lay.k_off(slot), self.lay.k_row)
        };
        // SAFETY: the address range is a row inside a live region of the borrowed cache.
        unsafe { std::slice::from_raw_parts((base + off) as *const u8, len) }
    }
    #[inline]
    pub fn load_k(&self, pos: usize, start: usize, out: &mut [f32]) {
        self.kv_type.load(self.row(pos, false), start, out);
    }
    #[inline]
    pub fn load_v(&self, pos: usize, start: usize, out: &mut [f32]) {
        self.kv_type.load(self.row(pos, true), start, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_round_trip_matches_ggml_rules() {
        let src: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.37).sin() * 3.0).collect();
        let mut q = vec![0u8; KvType::Q8_0.row_bytes(64)];
        assert_eq!(q.len(), 68);
        quantize_q8_0(&src, &mut q);
        let mut back = vec![0f32; 64];
        dequantize_q8_0(&q, &mut back);
        for blk in 0..2 {
            let s = &src[blk * 32..(blk + 1) * 32];
            let amax = s.iter().fold(0f32, |m, &x| m.max(x.abs()));
            let d = f16::from_f32(amax / 127.0).to_f32();
            for i in 0..32 {
                // Error at most half a quantisation step plus the f16 rounding of the scale.
                let err = (back[blk * 32 + i] - s[i]).abs();
                assert!(
                    err <= 0.5 * d + 127.0 * (d - amax / 127.0).abs() + 1e-6,
                    "{err}"
                );
            }
            // The largest value maps to ±127.
            let imax = s
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                .unwrap()
                .0;
            assert_eq!((q[blk * 34 + 2 + imax] as i8).unsigned_abs(), 127);
        }
        // Partial decode (second block only).
        let mut tail = vec![0f32; 32];
        KvType::Q8_0.load(&q, 32, &mut tail);
        assert_eq!(&tail[..], &back[32..]);
        // All-zero block: scale 0, values 0.
        let mut z = vec![0u8; 34];
        quantize_q8_0(&[0.0; 32], &mut z);
        assert!(z.iter().all(|&b| b == 0));
    }

    #[test]
    fn ties_round_to_even_like_the_simd_quantiser() {
        // 127 · (k + 0.5) / 127 hits exact .5 multiples of the step for some k.
        let mut src = vec![0f32; 32];
        src[0] = 127.0;
        src[1] = 2.5;
        src[2] = 3.5;
        src[3] = -2.5;
        let mut q = vec![0u8; 34];
        quantize_q8_0(&src, &mut q);
        assert_eq!(q[2] as i8, 127);
        assert_eq!(q[3] as i8, 2);
        assert_eq!(q[4] as i8, 4);
        assert_eq!(q[5] as i8, -2);
    }

    #[test]
    fn region_is_zeroed_and_writable() {
        let r = Region::new(1 << 20);
        // SAFETY: the region is 1 MiB of owned memory.
        let s = unsafe { std::slice::from_raw_parts_mut(r.as_ptr(), r.len()) };
        assert!(s.iter().step_by(4096).all(|&b| b == 0));
        s[12345] = 7;
        assert_eq!(s[12345], 7);
    }
}
