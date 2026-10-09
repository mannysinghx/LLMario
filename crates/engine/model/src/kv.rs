//! Per-sequence cache state for the M1/M2 families: one contiguous f32 K/V region per *attention*
//! layer, sized once from the admitted context (the paged arena with cache classes replaces the
//! storage in M3 behind the same accessors), plus the fp32 recurrent state of every Gated DeltaNet
//! layer (Architecture §8.1: recurrent state is per sequence, fixed size, fp32).
//!
//! Hybrid families index the K/V region by the layer's ordinal among the attention layers, so a
//! 24-layer Qwen3.5 model with six attention layers reserves six K/V slabs, not 24.

use crate::arch::ArchSpec;

/// The recurrent state of one Gated DeltaNet layer (see `gdn.rs` for the layouts).
pub struct RecurrentState {
    /// Causal-conv history: `[conv_dim][d_conv - 1]`, the last `d_conv - 1` pre-activation inputs
    /// per channel, oldest first.
    pub conv: Vec<f32>,
    /// Delta-rule state: `[n_v_heads][head_k][head_v]`.
    pub state: Vec<f32>,
}

pub struct KvCache {
    /// Number of attention layers (K/V slabs).
    pub n_layer: usize,
    pub max_ctx: usize,
    pub kv_dim: usize,
    pub v_dim: usize,
    /// `[layer][pos][kv_dim]`
    k: Vec<f32>,
    /// `[layer][pos][v_dim]`
    v: Vec<f32>,
    /// Tokens currently cached (positions `0..len` are valid).
    pub len: usize,
    /// One entry per Gated DeltaNet layer, in layer order (empty for the dense families).
    pub rs: Vec<RecurrentState>,
}

impl KvCache {
    pub fn new(spec: &ArchSpec, max_ctx: usize) -> KvCache {
        let n_layer = spec.n_attn_layers() as usize;
        let kv_dim = spec.kv_dim() as usize;
        let v_dim = spec.v_dim() as usize;
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
            n_layer,
            max_ctx,
            kv_dim,
            v_dim,
            k: vec![0f32; n_layer * max_ctx * kv_dim],
            v: vec![0f32; n_layer * max_ctx * v_dim],
            len: 0,
            rs,
        }
    }

    /// Bytes this cache reserves (what the plan charges): K/V for the attention layers plus the
    /// fp32 recurrent state of the DeltaNet layers.
    pub fn bytes(spec: &ArchSpec, max_ctx: usize) -> u64 {
        let per_tok = (spec.kv_dim() + spec.v_dim()) as u64 * 4;
        per_tok * spec.n_attn_layers() as u64 * max_ctx as u64 + Self::recurrent_bytes(spec)
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

    #[inline]
    pub fn k_row(&self, layer: usize, pos: usize) -> &[f32] {
        let o = (layer * self.max_ctx + pos) * self.kv_dim;
        &self.k[o..o + self.kv_dim]
    }
    #[inline]
    pub fn v_row(&self, layer: usize, pos: usize) -> &[f32] {
        let o = (layer * self.max_ctx + pos) * self.v_dim;
        &self.v[o..o + self.v_dim]
    }
    #[inline]
    pub fn k_row_mut(&mut self, layer: usize, pos: usize) -> &mut [f32] {
        let o = (layer * self.max_ctx + pos) * self.kv_dim;
        &mut self.k[o..o + self.kv_dim]
    }
    #[inline]
    pub fn v_row_mut(&mut self, layer: usize, pos: usize) -> &mut [f32] {
        let o = (layer * self.max_ctx + pos) * self.v_dim;
        &mut self.v[o..o + self.v_dim]
    }
    /// All keys of one layer for positions `0..n` as one contiguous `[n][kv_dim]` slice.
    #[inline]
    pub fn k_layer(&self, layer: usize, n: usize) -> &[f32] {
        let o = layer * self.max_ctx * self.kv_dim;
        &self.k[o..o + n * self.kv_dim]
    }
    #[inline]
    pub fn v_layer(&self, layer: usize, n: usize) -> &[f32] {
        let o = layer * self.max_ctx * self.v_dim;
        &self.v[o..o + n * self.v_dim]
    }

    /// Drop everything after `n` tokens (prefix reuse / retry). A recurrent state cannot be
    /// rewound without checkpoints (a later milestone), so for hybrid families any `n < len`
    /// resets the whole sequence (`len == 0` afterwards); callers re-read `len` and recompute.
    pub fn truncate(&mut self, n: usize) {
        if n < self.len && !self.rs.is_empty() {
            self.clear();
            return;
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
