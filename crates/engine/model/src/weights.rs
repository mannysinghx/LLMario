//! Zero-copy weight views resolved from a GGUF file for the dense families (the hybrid family's
//! layer views are in `hybrid.rs` and hang off [`Weights::hybrid`]).

use crate::arch::{ArchSpec, Family};
use crate::hybrid::HybridWeights;
use crate::{ModelError, Result};
use llmario_engine_core::dequant::dequantize_row;
use llmario_engine_core::GgmlType;
use llmario_engine_cpu::QMat;
use llmario_engine_formats::GgufFile;

/// One transformer block's tensors.
pub struct LayerWeights<'a> {
    pub attn_norm: Vec<f32>,
    pub wq: QMat<'a>,
    pub wk: QMat<'a>,
    pub wv: QMat<'a>,
    pub wo: QMat<'a>,
    pub bq: Option<Vec<f32>>,
    pub bk: Option<Vec<f32>>,
    pub bv: Option<Vec<f32>>,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub ffn_norm: Vec<f32>,
    pub w_gate: QMat<'a>,
    pub w_up: QMat<'a>,
    pub w_down: QMat<'a>,
}

pub struct Weights<'a> {
    pub token_embd: QMat<'a>,
    pub output_norm: Vec<f32>,
    /// `None` when the output head is tied to the embedding.
    pub output: Option<QMat<'a>>,
    /// Dense-family layers (empty for the hybrid family).
    pub layers: Vec<LayerWeights<'a>>,
    /// Hybrid-family layers (`Some` only for [`Family::Qwen35`]).
    pub hybrid: Option<HybridWeights<'a>>,
}

pub(crate) fn mat<'a>(f: &'a GgufFile, name: &str, cols: u32, rows: u32) -> Result<QMat<'a>> {
    let t = f
        .tensor(name)
        .ok_or_else(|| ModelError::MissingTensor(name.into()))?;
    let dims = t.shape.dims();
    if dims.len() != 2 || dims[0] != cols as u64 || dims[1] != rows as u64 {
        return Err(ModelError::BadShape {
            name: name.into(),
            shape: t.shape.to_string(),
            expected: format!("[{cols}, {rows}]"),
        });
    }
    Ok(QMat::new(
        t.dtype,
        rows as usize,
        cols as usize,
        f.tensor_bytes(t),
    ))
}

/// Load a float tensor of any rank with `len` elements, row-major, into an f32 vector.
pub(crate) fn vecn(f: &GgufFile, name: &str, len: u32) -> Result<Vec<f32>> {
    vec1(f, name, len)
}

/// Load a 1-D float tensor (norm weights, biases) into an f32 vector.
pub(crate) fn vec1(f: &GgufFile, name: &str, len: u32) -> Result<Vec<f32>> {
    let t = f
        .tensor(name)
        .ok_or_else(|| ModelError::MissingTensor(name.into()))?;
    if t.shape.numel() != len as u64 {
        return Err(ModelError::BadShape {
            name: name.into(),
            shape: t.shape.to_string(),
            expected: format!("[{len}]"),
        });
    }
    let mut v = vec![0f32; len as usize];
    dequantize_row(t.dtype, f.tensor_bytes(t), &mut v)?;
    Ok(v)
}

fn vec1_opt(f: &GgufFile, name: &str, len: u32) -> Result<Option<Vec<f32>>> {
    if f.tensor(name).is_some() {
        Ok(Some(vec1(f, name, len)?))
    } else {
        Ok(None)
    }
}

impl<'a> Weights<'a> {
    pub fn load(f: &'a GgufFile, spec: &ArchSpec) -> Result<Weights<'a>> {
        let token_embd = mat(f, "token_embd.weight", spec.d_model, spec.n_vocab)?;
        let output_norm = vec1(f, "output_norm.weight", spec.d_model)?;
        let output = if spec.tied_embeddings {
            None
        } else {
            Some(mat(f, "output.weight", spec.d_model, spec.n_vocab)?)
        };
        if spec.family == Family::Qwen35 {
            return Ok(Weights {
                token_embd,
                output_norm,
                output,
                layers: Vec::new(),
                hybrid: Some(HybridWeights::load(f, spec)?),
            });
        }
        let mut layers = Vec::with_capacity(spec.n_layer as usize);
        for l in 0..spec.n_layer {
            let p = |s: &str| format!("blk.{l}.{s}");
            layers.push(LayerWeights {
                attn_norm: vec1(f, &p("attn_norm.weight"), spec.d_model)?,
                wq: mat(f, &p("attn_q.weight"), spec.d_model, spec.q_dim())?,
                wk: mat(f, &p("attn_k.weight"), spec.d_model, spec.kv_dim())?,
                wv: mat(f, &p("attn_v.weight"), spec.d_model, spec.v_dim())?,
                wo: mat(
                    f,
                    &p("attn_output.weight"),
                    spec.n_head * spec.head_dim_v,
                    spec.d_model,
                )?,
                bq: vec1_opt(f, &p("attn_q.bias"), spec.q_dim())?,
                bk: vec1_opt(f, &p("attn_k.bias"), spec.kv_dim())?,
                bv: vec1_opt(f, &p("attn_v.bias"), spec.v_dim())?,
                q_norm: vec1_opt(f, &p("attn_q_norm.weight"), spec.head_dim)?,
                k_norm: vec1_opt(f, &p("attn_k_norm.weight"), spec.head_dim)?,
                ffn_norm: vec1(f, &p("ffn_norm.weight"), spec.d_model)?,
                w_gate: mat(f, &p("ffn_gate.weight"), spec.d_model, spec.n_ff)?,
                w_up: mat(f, &p("ffn_up.weight"), spec.d_model, spec.n_ff)?,
                w_down: mat(f, &p("ffn_down.weight"), spec.n_ff, spec.d_model)?,
            });
        }
        Ok(Weights {
            token_embd,
            output_norm,
            output,
            layers,
            hybrid: None,
        })
    }

    /// Types present in the weights, for the plan's kernel-coverage check.
    pub fn dtypes(&self) -> Vec<GgmlType> {
        let mut v = vec![self.token_embd.dtype];
        if let Some(o) = &self.output {
            v.push(o.dtype);
        }
        for l in &self.layers {
            for m in [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down] {
                v.push(m.dtype);
            }
        }
        if let Some(h) = &self.hybrid {
            h.push_dtypes(&mut v);
        }
        v.sort();
        v.dedup();
        v
    }
}
