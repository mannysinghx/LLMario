//! Per-sequence KV cache for the dense families (M1 shape: one contiguous f32 region per layer,
//! sized once from the admitted context; the paged arena with cache classes replaces the storage
//! in M3 behind the same accessors).

use crate::arch::ArchSpec;

pub struct KvCache {
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
}

impl KvCache {
    pub fn new(spec: &ArchSpec, max_ctx: usize) -> KvCache {
        let n_layer = spec.n_layer as usize;
        let kv_dim = spec.kv_dim() as usize;
        let v_dim = spec.v_dim() as usize;
        KvCache {
            n_layer,
            max_ctx,
            kv_dim,
            v_dim,
            k: vec![0f32; n_layer * max_ctx * kv_dim],
            v: vec![0f32; n_layer * max_ctx * v_dim],
            len: 0,
        }
    }

    /// Bytes this cache reserves (what the plan charges).
    pub fn bytes(spec: &ArchSpec, max_ctx: usize) -> u64 {
        let per_tok = (spec.kv_dim() + spec.v_dim()) as u64 * 4;
        per_tok * spec.n_layer as u64 * max_ctx as u64
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

    /// Drop everything after `n` tokens (prefix reuse / retry).
    pub fn truncate(&mut self, n: usize) {
        self.len = self.len.min(n);
    }
    pub fn clear(&mut self) {
        self.len = 0;
    }
}
