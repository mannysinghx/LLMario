//! Zero-copy weight views resolved from a GGUF file for the dense families (the hybrid family's
//! layer views are in `hybrid.rs` and hang off [`Weights::hybrid`]; Gemma 4's are in `gemma4.rs`
//! and hang off [`Weights::gemma4`]).

use crate::arch::{ArchSpec, Family};
use crate::gemma4::Gemma4Weights;
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
    /// Dense FFN (empty views on a mixture-of-experts layer, which uses `moe`).
    pub w_gate: QMat<'a>,
    pub w_up: QMat<'a>,
    pub w_down: QMat<'a>,
    /// Routed experts (`Some` only for the MoE families).
    pub moe: Option<MoeWeights<'a>>,
}

/// One 3-D expert tensor (`[n_expert][rows][cols]`, ggml `ne = [cols, rows, n_expert]`): every
/// expert's matrix is a contiguous slice of the mapping.
#[derive(Clone, Copy, Debug)]
pub struct ExpertMats<'a> {
    pub dtype: GgmlType,
    pub rows: usize,
    pub cols: usize,
    pub n_expert: usize,
    data: &'a [u8],
}

impl<'a> ExpertMats<'a> {
    /// Expert `e`'s `rows × cols` matrix.
    #[inline]
    pub fn expert(&self, e: usize) -> QMat<'a> {
        let per = self.rows * self.dtype.row_bytes(self.cols);
        QMat::new(
            self.dtype,
            self.rows,
            self.cols,
            &self.data[e * per..(e + 1) * per],
        )
    }
    /// Bytes of one expert's matrix.
    pub fn expert_bytes(&self) -> usize {
        self.rows * self.dtype.row_bytes(self.cols)
    }
    /// Every expert's matrix as one `(n_expert · rows) × cols` view (expert `e` starts at row
    /// `e · rows`), for backends that address experts by offset.
    pub fn all(&self) -> QMat<'a> {
        QMat::new(self.dtype, self.n_expert * self.rows, self.cols, self.data)
    }
}

/// A routed mixture-of-experts FFN: the router and the gate / up / down expert tensors.
pub struct MoeWeights<'a> {
    /// `[n_expert][d_model]` router (`ffn_gate_inp`).
    pub gate_inp: QMat<'a>,
    pub gate: ExpertMats<'a>,
    pub up: ExpertMats<'a>,
    pub down: ExpertMats<'a>,
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
    /// Gemma 4 layers (`Some` only for [`Family::Gemma4`]).
    pub gemma4: Option<Gemma4Weights<'a>>,
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

/// A 3-D expert tensor with ggml shape `[cols, rows, n_expert]`.
pub(crate) fn mat3<'a>(
    f: &'a GgufFile,
    name: &str,
    cols: u32,
    rows: u32,
    n_expert: u32,
) -> Result<ExpertMats<'a>> {
    let t = f
        .tensor(name)
        .ok_or_else(|| ModelError::MissingTensor(name.into()))?;
    let dims = t.shape.dims();
    if dims.len() != 3
        || dims[0] != cols as u64
        || dims[1] != rows as u64
        || dims[2] != n_expert as u64
    {
        return Err(ModelError::BadShape {
            name: name.into(),
            shape: t.shape.to_string(),
            expected: format!("[{cols}, {rows}, {n_expert}]"),
        });
    }
    Ok(ExpertMats {
        dtype: t.dtype,
        rows: rows as usize,
        cols: cols as usize,
        n_expert: n_expert as usize,
        data: f.tensor_bytes(t),
    })
}

/// An empty matrix view (the dense FFN slot of a mixture-of-experts layer).
fn empty_mat<'a>() -> QMat<'a> {
    QMat::new(GgmlType::F32, 0, 0, &[])
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
                gemma4: None,
            });
        }
        if spec.family == Family::Gemma4 {
            return Ok(Weights {
                token_embd,
                output_norm,
                output,
                layers: Vec::new(),
                hybrid: None,
                gemma4: Some(Gemma4Weights::load(f, spec)?),
            });
        }
        let mut layers = Vec::with_capacity(spec.n_layer as usize);
        for l in 0..spec.n_layer {
            let p = |s: &str| format!("blk.{l}.{s}");
            let (w_gate, w_up, w_down, moe) = match &spec.moe {
                Some(m) => (
                    empty_mat(),
                    empty_mat(),
                    empty_mat(),
                    Some(MoeWeights {
                        gate_inp: mat(f, &p("ffn_gate_inp.weight"), spec.d_model, m.n_expert)?,
                        gate: mat3(
                            f,
                            &p("ffn_gate_exps.weight"),
                            spec.d_model,
                            m.n_ff_exp,
                            m.n_expert,
                        )?,
                        up: mat3(
                            f,
                            &p("ffn_up_exps.weight"),
                            spec.d_model,
                            m.n_ff_exp,
                            m.n_expert,
                        )?,
                        down: mat3(
                            f,
                            &p("ffn_down_exps.weight"),
                            m.n_ff_exp,
                            spec.d_model,
                            m.n_expert,
                        )?,
                    }),
                ),
                None => (
                    mat(f, &p("ffn_gate.weight"), spec.d_model, spec.n_ff)?,
                    mat(f, &p("ffn_up.weight"), spec.d_model, spec.n_ff)?,
                    mat(f, &p("ffn_down.weight"), spec.n_ff, spec.d_model)?,
                    None,
                ),
            };
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
                w_gate,
                w_up,
                w_down,
                moe,
            });
        }
        Ok(Weights {
            token_embd,
            output_norm,
            output,
            layers,
            hybrid: None,
            gemma4: None,
        })
    }

    /// Types present in the weights, for the plan's kernel-coverage check.
    pub fn dtypes(&self) -> Vec<GgmlType> {
        let mut v = vec![self.token_embd.dtype];
        if let Some(o) = &self.output {
            v.push(o.dtype);
        }
        for l in &self.layers {
            for m in [&l.wq, &l.wk, &l.wv, &l.wo] {
                v.push(m.dtype);
            }
            match &l.moe {
                Some(m) => v.extend([m.gate_inp.dtype, m.gate.dtype, m.up.dtype, m.down.dtype]),
                None => v.extend([l.w_gate.dtype, l.w_up.dtype, l.w_down.dtype]),
            }
        }
        if let Some(h) = &self.hybrid {
            h.push_dtypes(&mut v);
        }
        if let Some(g) = &self.gemma4 {
            g.push_dtypes(&mut v);
        }
        v.sort();
        v.dedup();
        v
    }
}
