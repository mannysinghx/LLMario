//! Per-sequence cache state for the M1/M2 families: one contiguous f32 K/V slab per *attention*
//! layer, sized once from the admitted context (the paged arena with cache classes replaces the
//! storage in M3 behind the same accessors), plus the fp32 recurrent state of every Gated DeltaNet
//! layer (Architecture §8.1: recurrent state is per sequence, fixed size, fp32).
//!
//! Hybrid families index the K/V slabs by the layer's ordinal among the attention layers, so a
//! 24-layer Qwen3.5 model with six attention layers reserves six K/V slabs, not 24.
//!
//! # Sliding-window layers (Architecture §8.1 "Window" class)
//!
//! A layer whose [`crate::arch::AttnGeom::window`] is `Some(n_swa)` only ever attends to the last
//! `n_swa` positions, so its slab is a **ring** of `cap = min(max_ctx, n_swa + n_batch)` positions:
//! position `p` lives in slot `p % cap` ([`KvCache::slot`]). The `n_batch` headroom exists
//! because a prefill chunk of `n` tokens at positions `pos0..pos0 + n` needs positions
//! `pos0 − n_swa + 1 ..= pos0 + n − 1` alive at once (`n_swa + n − 1` distinct slots); the
//! forward pass therefore splits batches to at most [`KvCache::max_batch`] tokens. This is the
//! same `n_swa + n_batch` sizing llama.cpp's iSWA cache uses (research note KQ1, PR #13194). When
//! `cap == max_ctx` the ring never wraps and no batch limit applies. Full-attention layers have
//! `cap == max_ctx` and `slot(p) == p`.

use crate::arch::{ArchSpec, BlockKind};

/// Default prefill headroom added to a sliding-window ring (`cap = n_swa + SWA_RING_BATCH`) by
/// [`KvCache::new`] and [`KvCache::bytes`]; [`KvCache::with_batch`] takes an explicit value.
pub const SWA_RING_BATCH: usize = 512;

/// The recurrent state of one Gated DeltaNet layer (see `gdn.rs` for the layouts).
pub struct RecurrentState {
    /// Causal-conv history: `[conv_dim][d_conv - 1]`, the last `d_conv - 1` pre-activation inputs
    /// per channel, oldest first.
    pub conv: Vec<f32>,
    /// Delta-rule state: `[n_v_heads][head_k][head_v]`.
    pub state: Vec<f32>,
}

/// One attention layer's slab geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvLayer {
    pub kv_dim: usize,
    pub v_dim: usize,
    /// Positions the slab holds (`max_ctx`, or the ring size of a window layer).
    pub cap: usize,
    /// `Some(n_swa)` for a sliding-window layer.
    pub window: Option<usize>,
    k_off: usize,
    v_off: usize,
}

pub struct KvCache {
    /// Number of attention layers (K/V slabs).
    pub n_layer: usize,
    pub max_ctx: usize,
    /// K / V row widths of the full-attention layers (`ArchSpec::kv_dim()` / `v_dim()`); the
    /// per-layer widths are in [`KvCache::layers`].
    pub kv_dim: usize,
    pub v_dim: usize,
    /// Slab geometry per attention layer.
    pub layers: Vec<KvLayer>,
    /// All K slabs back to back: `[cap][kv_dim]` per layer.
    k: Vec<f32>,
    /// All V slabs back to back: `[cap][v_dim]` per layer.
    v: Vec<f32>,
    /// Tokens currently cached (positions `0..len` are valid, modulo each ring's eviction).
    pub len: usize,
    /// One entry per Gated DeltaNet layer, in layer order (empty for the dense families).
    pub rs: Vec<RecurrentState>,
}

impl KvCache {
    /// A cache for `max_ctx` positions; window rings get [`SWA_RING_BATCH`] tokens of headroom.
    pub fn new(spec: &ArchSpec, max_ctx: usize) -> KvCache {
        Self::with_batch(spec, max_ctx, SWA_RING_BATCH)
    }

    /// A cache for `max_ctx` positions whose window rings hold `n_swa + n_batch` positions.
    pub fn with_batch(spec: &ArchSpec, max_ctx: usize, n_batch: usize) -> KvCache {
        let mut layers = Vec::new();
        let (mut k_off, mut v_off) = (0usize, 0usize);
        for l in 0..spec.n_layer as usize {
            if spec.blocks[l] != BlockKind::Attention {
                continue;
            }
            let g = spec.attn_geom(l);
            let cap = g
                .window
                .map(|w| max_ctx.min(w as usize + n_batch))
                .unwrap_or(max_ctx);
            let layer = KvLayer {
                kv_dim: g.kv_dim() as usize,
                v_dim: g.v_dim() as usize,
                cap,
                window: g.window.map(|w| w as usize),
                k_off,
                v_off,
            };
            k_off += cap * layer.kv_dim;
            v_off += cap * layer.v_dim;
            layers.push(layer);
        }
        let rs = match &spec.gdn {
            Some(g) => (0..spec.n_recurrent_layers())
                .map(|_| RecurrentState {
                    conv: vec![0f32; g.conv_state_len() as usize],
                    state: vec![0f32; g.state_len() as usize],
                })
                .collect(),
            None => Vec::new(),
        };
        KvCache {
            n_layer: layers.len(),
            max_ctx,
            kv_dim: spec.kv_dim() as usize,
            v_dim: spec.v_dim() as usize,
            layers,
            k: vec![0f32; k_off],
            v: vec![0f32; v_off],
            len: 0,
            rs,
        }
    }

    /// Bytes this cache reserves (what the plan charges): K/V for the attention layers (window
    /// layers at their ring size) plus the fp32 recurrent state of the DeltaNet layers.
    pub fn bytes(spec: &ArchSpec, max_ctx: usize) -> u64 {
        Self::bytes_with_batch(spec, max_ctx, SWA_RING_BATCH)
    }

    pub fn bytes_with_batch(spec: &ArchSpec, max_ctx: usize, n_batch: usize) -> u64 {
        spec.kv_bytes_per_token(4.0) * max_ctx as u64
            + spec.window_bytes(4.0, max_ctx, n_batch)
            + Self::recurrent_bytes(spec)
    }

    /// Bytes of fp32 recurrent state per sequence (0 for the dense families).
    pub fn recurrent_bytes(spec: &ArchSpec) -> u64 {
        match &spec.gdn {
            Some(g) => {
                (g.conv_state_len() as u64 + g.state_len() as u64)
                    * 4
                    * spec.n_recurrent_layers() as u64
            }
            None => 0,
        }
    }

    /// Largest number of tokens one forward call may append without a window ring overwriting a
    /// position a token of the same batch still needs (`usize::MAX` when no ring wraps).
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

    /// Slot of position `pos` in layer `layer`'s slab.
    #[inline]
    pub fn slot(&self, layer: usize, pos: usize) -> usize {
        pos % self.layers[layer].cap
    }

    #[inline]
    pub fn k_row(&self, layer: usize, pos: usize) -> &[f32] {
        let l = &self.layers[layer];
        let o = l.k_off + (pos % l.cap) * l.kv_dim;
        &self.k[o..o + l.kv_dim]
    }
    #[inline]
    pub fn v_row(&self, layer: usize, pos: usize) -> &[f32] {
        let l = &self.layers[layer];
        let o = l.v_off + (pos % l.cap) * l.v_dim;
        &self.v[o..o + l.v_dim]
    }
    #[inline]
    pub fn k_row_mut(&mut self, layer: usize, pos: usize) -> &mut [f32] {
        let l = self.layers[layer];
        let o = l.k_off + (pos % l.cap) * l.kv_dim;
        &mut self.k[o..o + l.kv_dim]
    }
    #[inline]
    pub fn v_row_mut(&mut self, layer: usize, pos: usize) -> &mut [f32] {
        let l = self.layers[layer];
        let o = l.v_off + (pos % l.cap) * l.v_dim;
        &mut self.v[o..o + l.v_dim]
    }
    /// The whole K slab of one layer, `[cap][kv_dim]`, indexed by [`KvCache::slot`].
    #[inline]
    pub fn k_slab(&self, layer: usize) -> &[f32] {
        let l = &self.layers[layer];
        &self.k[l.k_off..l.k_off + l.cap * l.kv_dim]
    }
    #[inline]
    pub fn v_slab(&self, layer: usize) -> &[f32] {
        let l = &self.layers[layer];
        &self.v[l.v_off..l.v_off + l.cap * l.v_dim]
    }
    /// All keys of one layer for positions `0..n` as one contiguous `[n][kv_dim]` slice
    /// (full-attention layers, or a ring that has not wrapped: `n ≤ cap`).
    #[inline]
    pub fn k_layer(&self, layer: usize, n: usize) -> &[f32] {
        let l = &self.layers[layer];
        assert!(
            n <= l.cap,
            "k_layer: {n} positions exceed the slab ({})",
            l.cap
        );
        &self.k[l.k_off..l.k_off + n * l.kv_dim]
    }
    #[inline]
    pub fn v_layer(&self, layer: usize, n: usize) -> &[f32] {
        let l = &self.layers[layer];
        assert!(
            n <= l.cap,
            "v_layer: {n} positions exceed the slab ({})",
            l.cap
        );
        &self.v[l.v_off..l.v_off + n * l.v_dim]
    }

    /// Drop everything after `n` tokens (prefix reuse / retry). A recurrent state cannot be
    /// rewound without checkpoints (a later milestone), so for hybrid families any `n < len`
    /// resets the whole sequence (`len == 0` afterwards); likewise when a window ring has already
    /// evicted a position the rewound sequence would need (`len − n > cap − n_swa + 1`). Callers
    /// re-read `len` and recompute.
    pub fn truncate(&mut self, n: usize) {
        if n < self.len {
            if !self.rs.is_empty() {
                self.clear();
                return;
            }
            let evicted = self.layers.iter().any(|l| match l.window {
                Some(w) if l.cap < self.max_ctx => self.len - n > (l.cap + 1).saturating_sub(w),
                _ => false,
            });
            if evicted {
                self.clear();
                return;
            }
        }
        self.len = self.len.min(n);
    }
    pub fn clear(&mut self) {
        self.len = 0;
        for r in &mut self.rs {
            r.conv.iter_mut().for_each(|v| *v = 0.0);
            r.state.iter_mut().for_each(|v| *v = 0.0);
        }
    }
}
