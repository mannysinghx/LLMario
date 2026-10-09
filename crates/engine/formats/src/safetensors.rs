//! Safetensors folder reader (Hugging Face and MLX layouts).
//!
//! A folder holds `config.json` (architecture hyper-parameters, and for MLX exports the
//! `quantization` block), `tokenizer.json`, `tokenizer_config.json` (chat template), optional
//! `generation_config.json`, and one `model.safetensors` or several `model-NNNNN-of-MMMMM.safetensors`
//! shards listed by `model.safetensors.index.json`.
//!
//! Each `.safetensors` file is: an 8-byte little-endian `u64` header length `N`, `N` bytes of UTF-8
//! JSON (starts with `{`, may be padded with spaces), then the byte buffer. The JSON maps tensor
//! names to `{dtype, shape, data_offsets: [begin, end)}` with offsets relative to the buffer, plus an
//! optional `__metadata__` string map (MLX writes `{"format": "mlx"}`). Tensors are row-major and
//! little-endian (safetensors README, Apache-2.0).
//!
//! Every shard is memory-mapped read-only and parsed once; tensors are zero-copy views
//! ([`TensorInfo`] with `span.source` = shard index). The data region is only byte-aligned, so a
//! view is never reinterpreted as `&[u32]`; `dequant::mlx_affine_row_le_bytes` reads the words.
//!
//! Shapes are converted to ggml order (`ne[0]` is the contiguous dimension): an HF `[out, in]`
//! matrix becomes `[in, out]`, the same orientation GGUF uses for the same tensor, so the model
//! layer can treat both readers alike. Tensors with more than four dimensions keep their three
//! innermost dimensions and fold the rest into `ne[3]`; [`SafetensorsFolder::hf_dims`] returns the
//! original row-major dimensions.
//!
//! MLX quantised linears are `{module}.weight` (U32, `[rows, cols*bits/32]`), `{module}.scales` and
//! `{module}.biases` (`[rows, cols/group_size]`, F16/BF16/F32); [`SafetensorsFolder::mlx_quant_view`]
//! resolves the triplet and [`SafetensorsFolder::dequantize_mlx_rows`] expands rows to f32.

use llmario_engine_core::dequant::{dequantize_row, mlx_affine_row_le_bytes, MLX_AFFINE_BITS};
use llmario_engine_core::tensor::ByteSpan;
use llmario_engine_core::{EngineError, GgmlType, Result, Shape, TensorInfo};
use memmap2::Mmap;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::{Path, PathBuf};

/// The safetensors header limit (100 MB in the reference implementation).
const MAX_HEADER: u64 = 100 << 20;

pub const INDEX_FILE: &str = "model.safetensors.index.json";
pub const SINGLE_FILE: &str = "model.safetensors";

/// Map a safetensors dtype tag to an engine element type. Sub-byte and FP8 tags are refused.
pub fn dtype_from_tag(tag: &str) -> Option<GgmlType> {
    Some(match tag {
        "F32" => GgmlType::F32,
        "F16" => GgmlType::F16,
        "BF16" => GgmlType::BF16,
        "F64" => GgmlType::F64,
        "U8" => GgmlType::U8,
        "I8" => GgmlType::I8,
        "I16" => GgmlType::I16,
        "I32" => GgmlType::I32,
        "I64" => GgmlType::I64,
        "U32" => GgmlType::U32,
        _ => return None,
    })
}

/// The safetensors tag for an engine element type (None for block types).
pub fn dtype_tag(t: GgmlType) -> Option<&'static str> {
    Some(match t {
        GgmlType::F32 => "F32",
        GgmlType::F16 => "F16",
        GgmlType::BF16 => "BF16",
        GgmlType::F64 => "F64",
        GgmlType::U8 => "U8",
        GgmlType::I8 => "I8",
        GgmlType::I16 => "I16",
        GgmlType::I32 => "I32",
        GgmlType::I64 => "I64",
        GgmlType::U32 => "U32",
        _ => return None,
    })
}

/// One mapped `.safetensors` file.
#[derive(Debug)]
pub struct Shard {
    pub path: PathBuf,
    pub mmap: Mmap,
    /// Where the byte buffer starts (8 + header length).
    pub data_offset: u64,
    /// The header's `__metadata__` map.
    pub metadata: BTreeMap<String, String>,
}

/// Bits and group size of one MLX affine-quantised module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlxQuantParams {
    pub bits: u32,
    pub group_size: u32,
    /// `"affine"` unless the export says otherwise (`mxfp4`, `mxfp8`, `nvfp4` are not readable).
    pub mode: String,
}

impl MlxQuantParams {
    pub fn is_affine(&self) -> bool {
        self.mode == "affine"
    }
}

/// The `quantization` block of an MLX `config.json`: `{"bits": 4, "group_size": 64, "mode":
/// "affine"}` plus, for mixed-precision exports, one entry per module path whose value is either
/// `false` (left in full precision) or its own `{bits, group_size}` (mlx-lm `quantize_model`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlxQuantization {
    pub default: MlxQuantParams,
    pub overrides: BTreeMap<String, Option<MlxQuantParams>>,
}

impl MlxQuantization {
    /// Parse a `quantization` / `quantization_config` object. Returns None for objects that are
    /// not MLX's (an HF `quantization_config` carries `quant_method`; MLX's never does).
    pub fn from_value(v: &Value) -> Option<MlxQuantization> {
        let obj = v.as_object()?;
        if obj.contains_key("quant_method") {
            return None;
        }
        let default = Self::params(obj, None)?;
        let mut overrides = BTreeMap::new();
        for (k, v) in obj {
            if matches!(k.as_str(), "bits" | "group_size" | "mode") {
                continue;
            }
            let p = match v {
                Value::Bool(false) => None,
                Value::Bool(true) => Some(default.clone()),
                Value::Object(o) => Some(Self::params(o, Some(&default))?),
                _ => continue,
            };
            overrides.insert(k.clone(), p);
        }
        Some(MlxQuantization { default, overrides })
    }

    fn params(
        o: &serde_json::Map<String, Value>,
        fallback: Option<&MlxQuantParams>,
    ) -> Option<MlxQuantParams> {
        let bits = o
            .get("bits")
            .and_then(Value::as_u64)
            .or(fallback.map(|f| f.bits as u64))?;
        let group_size = o
            .get("group_size")
            .and_then(Value::as_u64)
            .or(fallback.map(|f| f.group_size as u64))?;
        let mode = o
            .get("mode")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| fallback.map(|f| f.mode.clone()))
            .unwrap_or_else(|| "affine".to_string());
        Some(MlxQuantParams {
            bits: u32::try_from(bits).ok()?,
            group_size: u32::try_from(group_size).ok()?,
            mode,
        })
    }

    /// Parameters for a module path such as `model.layers.3.mlp.gate_proj`; None when the
    /// export left that module unquantised.
    pub fn params_for(&self, module: &str) -> Option<&MlxQuantParams> {
        match self.overrides.get(module) {
            Some(p) => p.as_ref(),
            None => Some(&self.default),
        }
    }
}

/// An MLX affine-quantised matrix: the packed weight plus its per-group scales and biases.
/// `rows` and `cols` are the logical HF shape (`[out_features, in_features]`); each row has
/// `cols / group_size` groups.
#[derive(Clone, Debug, PartialEq)]
pub struct MlxQuantView {
    pub module: String,
    pub weight: TensorInfo,
    pub scales: TensorInfo,
    pub biases: TensorInfo,
    pub bits: u32,
    pub group_size: u32,
    pub rows: u64,
    pub cols: u64,
    /// F16, BF16 or F32 (the dtype of both `scales` and `biases`).
    pub scale_dtype: GgmlType,
}

impl MlxQuantView {
    /// Packed bytes per row of the weight.
    pub fn weight_row_bytes(&self) -> usize {
        self.weight.row_bytes() as usize
    }
    /// Bytes per row of `scales` (and of `biases`).
    pub fn scales_row_bytes(&self) -> usize {
        self.scales.row_bytes() as usize
    }
    /// Bits per weight including scales and biases.
    pub fn bits_per_weight(&self) -> f64 {
        let bytes = self.weight.span.len + self.scales.span.len + self.biases.span.len;
        bytes as f64 * 8.0 / (self.rows * self.cols) as f64
    }
}

/// A parsed safetensors model folder.
#[derive(Debug)]
pub struct SafetensorsFolder {
    pub dir: PathBuf,
    /// `config.json` as parsed.
    pub config: Value,
    /// `generation_config.json`, when present.
    pub generation_config: Option<Value>,
    /// `tokenizer_config.json`, when present.
    pub tokenizer_config: Option<Value>,
    /// `metadata.total_size` from the index file, when sharded.
    pub index_total_size: Option<u64>,
    pub shards: Vec<Shard>,
    pub tensors: Vec<TensorInfo>,
    hf_dims: Vec<Vec<u64>>,
    by_name: BTreeMap<String, usize>,
    quantization: Option<MlxQuantization>,
}

impl SafetensorsFolder {
    /// Open a model folder: `config.json` is required; the weights are `model.safetensors`, or the
    /// shards named by `model.safetensors.index.json`, or failing both every `*.safetensors` file.
    pub fn open(dir: &Path) -> Result<SafetensorsFolder> {
        let config = read_json(&dir.join("config.json")).map_err(|e| {
            EngineError::Format(format!("{}: not a model folder: {e}", dir.display()))
        })?;
        let generation_config = read_json_opt(&dir.join("generation_config.json"))?;
        let tokenizer_config = read_json_opt(&dir.join("tokenizer_config.json"))?;

        // Which files hold the weights.
        let mut index_total_size = None;
        let mut weight_map: Option<BTreeMap<String, String>> = None;
        let index_path = dir.join(INDEX_FILE);
        let files: Vec<PathBuf> = if index_path.is_file() {
            let idx = read_json(&index_path)?;
            index_total_size = idx
                .get("metadata")
                .and_then(|m| m.get("total_size"))
                .and_then(Value::as_u64);
            let map = idx
                .get("weight_map")
                .and_then(Value::as_object)
                .ok_or_else(|| EngineError::Format(format!("{INDEX_FILE}: no weight_map")))?;
            let mut wm = BTreeMap::new();
            let mut names = BTreeSet::new();
            for (k, v) in map {
                let f = v.as_str().ok_or_else(|| {
                    EngineError::Format(format!("{INDEX_FILE}: weight_map[{k}] is not a string"))
                })?;
                names.insert(f.to_string());
                wm.insert(k.clone(), f.to_string());
            }
            weight_map = Some(wm);
            names.into_iter().map(|f| dir.join(f)).collect()
        } else if dir.join(SINGLE_FILE).is_file() {
            vec![dir.join(SINGLE_FILE)]
        } else {
            let mut v: Vec<PathBuf> = std::fs::read_dir(dir)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
                .collect();
            v.sort();
            v
        };
        if files.is_empty() {
            return Err(EngineError::Format(format!(
                "{}: no .safetensors files",
                dir.display()
            )));
        }

        let mut shards = Vec::with_capacity(files.len());
        let mut tensors = Vec::new();
        let mut hf_dims = Vec::new();
        let mut by_name = BTreeMap::new();
        for (i, path) in files.iter().enumerate() {
            let (shard, parsed) = open_shard(path)?;
            for (mut t, dims) in parsed {
                t.span.source = i as u32;
                if by_name.insert(t.name.clone(), tensors.len()).is_some() {
                    return Err(EngineError::Format(format!(
                        "duplicate tensor {} (in {})",
                        t.name,
                        path.display()
                    )));
                }
                tensors.push(t);
                hf_dims.push(dims);
            }
            shards.push(shard);
        }
        if let Some(wm) = &weight_map {
            for (name, file) in wm {
                let Some(&i) = by_name.get(name) else {
                    return Err(EngineError::Format(format!(
                        "{INDEX_FILE} lists {name} in {file} but the shard does not contain it"
                    )));
                };
                let actual = &files[tensors[i].span.source as usize];
                if actual.file_name().and_then(|f| f.to_str()) != Some(file.as_str()) {
                    return Err(EngineError::Format(format!(
                        "{INDEX_FILE} lists {name} in {file} but it is in {}",
                        actual.display()
                    )));
                }
            }
        }

        let quantization = config
            .get("quantization")
            .and_then(MlxQuantization::from_value)
            .or_else(|| {
                config
                    .get("quantization_config")
                    .and_then(MlxQuantization::from_value)
            });

        Ok(SafetensorsFolder {
            dir: dir.to_path_buf(),
            config,
            generation_config,
            tokenizer_config,
            index_total_size,
            shards,
            tensors,
            hf_dims,
            by_name,
            quantization,
        })
    }

    // ----- tensors -------------------------------------------------------------------------

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.by_name.get(name).map(|&i| &self.tensors[i])
    }

    /// The tensor's original row-major dimensions as written in the header.
    pub fn hf_dims(&self, name: &str) -> Option<&[u64]> {
        self.by_name.get(name).map(|&i| self.hf_dims[i].as_slice())
    }

    /// The bytes of a tensor (zero-copy slice of its shard's mapping).
    pub fn tensor_bytes(&self, t: &TensorInfo) -> &[u8] {
        let shard = &self.shards[t.span.source as usize];
        let start = (shard.data_offset + t.span.offset) as usize;
        &shard.mmap[start..start + t.span.len as usize]
    }

    /// Total tensor bytes across shards.
    pub fn tensor_bytes_total(&self) -> u64 {
        self.tensors.iter().map(|t| t.span.len).sum()
    }

    /// Bytes per element type.
    pub fn bytes_by_type(&self) -> BTreeMap<GgmlType, u64> {
        let mut m = BTreeMap::new();
        for t in &self.tensors {
            *m.entry(t.dtype).or_insert(0) += t.span.len;
        }
        m
    }

    /// True when any shard's `__metadata__.format` is `mlx` or `config.json` carries an MLX
    /// `quantization` block.
    pub fn is_mlx(&self) -> bool {
        self.quantization.is_some()
            || self
                .shards
                .iter()
                .any(|s| s.metadata.get("format").map(String::as_str) == Some("mlx"))
    }

    /// Dequantise `n_rows` rows starting at `row0` of a plain float tensor (F32/F16/BF16) into
    /// `out` (`n_rows * ne[0]` values).
    pub fn dequantize_rows(
        &self,
        t: &TensorInfo,
        row0: usize,
        n_rows: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let ne0 = t.shape.row_len() as usize;
        if row0 + n_rows > t.shape.rows() as usize || out.len() != n_rows * ne0 {
            return Err(EngineError::InvalidShape(format!(
                "{}: rows {row0}..{} of {} do not fit {} outputs",
                t.name,
                row0 + n_rows,
                t.shape,
                out.len()
            )));
        }
        let rb = t.row_bytes() as usize;
        let bytes = &self.tensor_bytes(t)[row0 * rb..(row0 + n_rows) * rb];
        dequantize_row(t.dtype, bytes, out)
    }

    // ----- MLX quantisation ----------------------------------------------------------------

    pub fn quantization(&self) -> Option<&MlxQuantization> {
        self.quantization.as_ref()
    }

    /// The raw HF `quantization_config` (AWQ/GPTQ/compressed-tensors; not readable yet).
    pub fn hf_quantization_config(&self) -> Option<&Value> {
        self.config
            .get("quantization_config")
            .filter(|v| v.get("quant_method").is_some())
    }

    /// Resolve `{module}.weight/.scales/.biases` into a view. `Ok(None)` when the module is not
    /// quantised (no `.scales`); an error when the triplet is inconsistent with `config.json`.
    pub fn mlx_quant_view(&self, module: &str) -> Result<Option<MlxQuantView>> {
        let Some(scales) = self.tensor(&format!("{module}.scales")) else {
            return Ok(None);
        };
        let bad = |msg: String| EngineError::Format(format!("{module}: {msg}"));
        let q = self
            .quantization
            .as_ref()
            .ok_or_else(|| bad("has .scales but config.json has no quantization block".into()))?;
        let params = q.params_for(module).ok_or_else(|| {
            bad("has .scales but config.json marks it unquantised (false)".into())
        })?;
        if !params.is_affine() {
            return Err(bad(format!(
                "quantization mode {} is not supported (affine only)",
                params.mode
            )));
        }
        if !MLX_AFFINE_BITS.contains(&params.bits) {
            return Err(bad(format!(
                "{} bits is not an MLX affine width",
                params.bits
            )));
        }
        let weight = self
            .tensor(&format!("{module}.weight"))
            .ok_or_else(|| bad("has .scales but no .weight".into()))?;
        let biases = self
            .tensor(&format!("{module}.biases"))
            .ok_or_else(|| bad("has .scales but no .biases (affine needs both)".into()))?;
        if weight.dtype != GgmlType::U32 {
            return Err(bad(format!(
                "packed weight must be U32, found {}",
                weight.dtype
            )));
        }
        if !matches!(scales.dtype, GgmlType::F16 | GgmlType::BF16 | GgmlType::F32)
            || biases.dtype != scales.dtype
        {
            return Err(bad(format!(
                "scales {} / biases {} must be the same float type",
                scales.dtype, biases.dtype
            )));
        }
        let wd = self.hf_dims(&weight.name).unwrap();
        let sd = self.hf_dims(&scales.name).unwrap();
        let bd = self.hf_dims(&biases.name).unwrap();
        if wd.len() != 2 || sd.len() != 2 || bd != sd {
            return Err(bad(format!(
                "expected 2-D weight/scales/biases, found {wd:?} / {sd:?} / {bd:?}"
            )));
        }
        let (rows, packed_cols) = (wd[0], wd[1]);
        let bits = params.bits as u64;
        let gs = params.group_size as u64;
        if (packed_cols * 32) % bits != 0 {
            return Err(bad(format!(
                "{packed_cols} packed words do not hold whole {bits}-bit elements"
            )));
        }
        let cols = packed_cols * 32 / bits;
        if cols % gs != 0 || sd[0] != rows || sd[1] != cols / gs {
            return Err(bad(format!(
                "scales shape {sd:?} does not match [{rows}, {cols}/{gs}] for {bits}-bit weight {wd:?}"
            )));
        }
        Ok(Some(MlxQuantView {
            module: module.to_string(),
            weight: weight.clone(),
            scales: scales.clone(),
            biases: biases.clone(),
            bits: params.bits,
            group_size: params.group_size,
            rows,
            cols,
            scale_dtype: scales.dtype,
        }))
    }

    /// Every quantised module in the folder (each `*.scales` tensor's module), in name order.
    pub fn mlx_quant_views(&self) -> Result<Vec<MlxQuantView>> {
        let mut out = Vec::new();
        for name in self.by_name.keys() {
            if let Some(module) = name.strip_suffix(".scales") {
                if let Some(v) = self.mlx_quant_view(module)? {
                    out.push(v);
                }
            }
        }
        Ok(out)
    }

    /// Dequantise rows `row0..row0+n_rows` of an MLX view into `out` (`n_rows * cols` values).
    pub fn dequantize_mlx_rows(
        &self,
        v: &MlxQuantView,
        row0: usize,
        n_rows: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let cols = v.cols as usize;
        if row0 + n_rows > v.rows as usize || out.len() != n_rows * cols {
            return Err(EngineError::InvalidShape(format!(
                "{}: rows {row0}..{} of {} do not fit {} outputs",
                v.module,
                row0 + n_rows,
                v.rows,
                out.len()
            )));
        }
        let wrb = v.weight_row_bytes();
        let srb = v.scales_row_bytes();
        let w = self.tensor_bytes(&v.weight);
        let s = self.tensor_bytes(&v.scales);
        let b = self.tensor_bytes(&v.biases);
        for (i, y) in out.chunks_exact_mut(cols).enumerate() {
            let r = row0 + i;
            mlx_affine_row_le_bytes(
                v.bits,
                v.group_size as usize,
                &w[r * wrb..(r + 1) * wrb],
                v.scale_dtype,
                &s[r * srb..(r + 1) * srb],
                &b[r * srb..(r + 1) * srb],
                y,
            )?;
        }
        Ok(())
    }

    // ----- config.json ---------------------------------------------------------------------

    /// A config key, looked up at the top level and then under `text_config` (multimodal
    /// configs such as Qwen3.5's nest the language model's hyper-parameters there).
    pub fn cfg(&self, key: &str) -> Option<&Value> {
        self.config
            .get(key)
            .filter(|v| !v.is_null())
            .or_else(|| self.config.get("text_config")?.get(key))
            .filter(|v| !v.is_null())
    }
    pub fn cfg_u64(&self, key: &str) -> Option<u64> {
        self.cfg(key).and_then(Value::as_u64)
    }
    pub fn cfg_f64(&self, key: &str) -> Option<f64> {
        self.cfg(key).and_then(Value::as_f64)
    }
    pub fn cfg_str(&self, key: &str) -> Option<&str> {
        self.cfg(key).and_then(Value::as_str)
    }
    pub fn cfg_bool(&self, key: &str) -> Option<bool> {
        self.cfg(key).and_then(Value::as_bool)
    }

    /// `model_type` (`qwen3`, `llama`, `qwen3_5`, ...). Multimodal configs name the text model
    /// under `text_config.model_type` (`qwen3_5_text`); the top-level value wins.
    pub fn model_type(&self) -> Option<&str> {
        self.cfg_str("model_type")
    }
    pub fn architectures(&self) -> Vec<&str> {
        self.config
            .get("architectures")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default()
    }
    pub fn hidden_size(&self) -> Option<u64> {
        self.cfg_u64("hidden_size")
    }
    pub fn num_hidden_layers(&self) -> Option<u64> {
        self.cfg_u64("num_hidden_layers")
    }
    pub fn num_attention_heads(&self) -> Option<u64> {
        self.cfg_u64("num_attention_heads")
    }
    /// Defaults to `num_attention_heads` (no GQA) when absent, as transformers does.
    pub fn num_key_value_heads(&self) -> Option<u64> {
        self.cfg_u64("num_key_value_heads")
            .or_else(|| self.num_attention_heads())
    }
    /// `head_dim` when written, else `hidden_size / num_attention_heads`.
    pub fn head_dim(&self) -> Option<u64> {
        self.cfg_u64("head_dim")
            .or_else(|| Some(self.hidden_size()? / self.num_attention_heads()?))
    }
    pub fn intermediate_size(&self) -> Option<u64> {
        self.cfg_u64("intermediate_size")
    }
    pub fn vocab_size(&self) -> Option<u64> {
        self.cfg_u64("vocab_size")
    }
    pub fn max_position_embeddings(&self) -> Option<u64> {
        self.cfg_u64("max_position_embeddings")
    }
    pub fn rms_norm_eps(&self) -> Option<f64> {
        self.cfg_f64("rms_norm_eps")
    }
    /// `rope_theta`, also found under `rope_parameters` (transformers v5 configs).
    pub fn rope_theta(&self) -> Option<f64> {
        self.cfg_f64("rope_theta").or_else(|| {
            self.cfg("rope_parameters")?
                .get("rope_theta")
                .and_then(Value::as_f64)
        })
    }
    /// The `rope_scaling` object when present and non-null, else `rope_parameters` (which in v5
    /// configs carries `rope_type`, `factor`, `mrope_section`, ... alongside `rope_theta`).
    pub fn rope_scaling(&self) -> Option<&Value> {
        self.cfg("rope_scaling")
            .filter(|v| v.is_object())
            .or_else(|| self.cfg("rope_parameters").filter(|v| v.is_object()))
    }
    pub fn tie_word_embeddings(&self) -> Option<bool> {
        self.cfg_bool("tie_word_embeddings")
    }
    pub fn bos_token_id(&self) -> Option<u64> {
        self.cfg_u64("bos_token_id")
    }
    /// `eos_token_id` (an id or a list) from `config.json`, merged with `generation_config.json`.
    pub fn eos_token_ids(&self) -> Vec<u64> {
        let mut ids = Vec::new();
        let mut push = |v: Option<&Value>| match v {
            Some(Value::Number(n)) => ids.extend(n.as_u64()),
            Some(Value::Array(a)) => ids.extend(a.iter().filter_map(Value::as_u64)),
            _ => {}
        };
        push(self.cfg("eos_token_id"));
        push(
            self.generation_config
                .as_ref()
                .and_then(|g| g.get("eos_token_id")),
        );
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    // ----- tokenizer and chat template -----------------------------------------------------

    /// `tokenizer.json` (HF tokenizers format) when present.
    pub fn tokenizer_json_path(&self) -> Option<PathBuf> {
        let p = self.dir.join("tokenizer.json");
        p.is_file().then_some(p)
    }

    /// The chat template: `chat_template.jinja`, else `chat_template.json`, else
    /// `tokenizer_config.json["chat_template"]` (a string, or a list of `{name, template}` from
    /// which `default` or the first entry is taken).
    pub fn chat_template(&self) -> Option<String> {
        if let Ok(s) = std::fs::read_to_string(self.dir.join("chat_template.jinja")) {
            return Some(s);
        }
        if let Ok(Some(v)) = read_json_opt(&self.dir.join("chat_template.json")) {
            if let Some(t) = template_value(v.get("chat_template")) {
                return Some(t);
            }
        }
        template_value(self.tokenizer_config.as_ref()?.get("chat_template"))
    }
}

fn template_value(v: Option<&Value>) -> Option<String> {
    match v? {
        Value::String(s) => Some(s.clone()),
        Value::Array(a) => {
            let pick = a
                .iter()
                .find(|e| e.get("name").and_then(Value::as_str) == Some("default"))
                .or_else(|| a.first())?;
            pick.get("template")
                .and_then(Value::as_str)
                .map(str::to_string)
        }
        _ => None,
    }
}

fn read_json(path: &Path) -> Result<Value> {
    let s = std::fs::read_to_string(path)
        .map_err(|e| EngineError::Format(format!("cannot read {}: {e}", path.display())))?;
    serde_json::from_str(&s)
        .map_err(|e| EngineError::Format(format!("{}: invalid JSON: {e}", path.display())))
}

fn read_json_opt(path: &Path) -> Result<Option<Value>> {
    if path.is_file() {
        read_json(path).map(Some)
    } else {
        Ok(None)
    }
}

/// Tensors of one shard (with `span.source` unset) and their row-major dims.
type ShardTensors = Vec<(TensorInfo, Vec<u64>)>;

/// Open and validate one shard.
fn open_shard(path: &Path) -> Result<(Shard, ShardTensors)> {
    let file = File::open(path)
        .map_err(|e| EngineError::Format(format!("cannot open {}: {e}", path.display())))?;
    // SAFETY: read-only private mapping of a regular file; the file may change underneath us
    // (any mmap has that property), which can only produce garbage tensors, never UB in safe
    // code paths that bounds-check offsets against the mapping length.
    let mmap = unsafe { Mmap::map(&file)? };
    let parsed = parse_header(&mmap[..])
        .map_err(|e| EngineError::Format(format!("{}: {e}", path.display())))?;
    let shard = Shard {
        path: path.to_path_buf(),
        mmap,
        data_offset: parsed.data_offset,
        metadata: parsed.metadata,
    };
    Ok((shard, parsed.tensors))
}

#[derive(Debug)]
pub struct ParsedHeader {
    pub metadata: BTreeMap<String, String>,
    pub tensors: ShardTensors,
    pub data_offset: u64,
}

/// Parse and validate a safetensors file image: header length, JSON, every tensor's dtype, shape
/// and byte range (inside the buffer, matching the shape, not overlapping any other tensor).
pub fn parse_header(buf: &[u8]) -> Result<ParsedHeader> {
    let fmt = |m: String| EngineError::Format(m);
    if buf.len() < 8 {
        return Err(fmt("file shorter than the 8-byte header length".into()));
    }
    let n = u64::from_le_bytes(buf[..8].try_into().unwrap());
    if n > MAX_HEADER {
        return Err(fmt(format!("header of {n} bytes exceeds the 100 MB limit")));
    }
    let n = n as usize;
    if buf.len() - 8 < n {
        return Err(fmt(format!(
            "header length {n} exceeds the file ({} bytes)",
            buf.len()
        )));
    }
    let header = &buf[8..8 + n];
    if header.first() != Some(&b'{') {
        return Err(fmt("header JSON does not start with '{'".into()));
    }
    let map: serde_json::Map<String, Value> = serde_json::from_slice(header)
        .map_err(|e| fmt(format!("header is not a JSON object: {e}")))?;
    let data_len = (buf.len() - 8 - n) as u64;
    let data_offset = 8 + n as u64;

    let mut metadata = BTreeMap::new();
    let mut tensors = Vec::with_capacity(map.len());
    let mut ranges: Vec<(u64, u64, usize)> = Vec::with_capacity(map.len());
    for (name, v) in &map {
        if name == "__metadata__" {
            if let Some(o) = v.as_object() {
                for (k, v) in o {
                    if let Some(s) = v.as_str() {
                        metadata.insert(k.clone(), s.to_string());
                    }
                }
            }
            continue;
        }
        let bad = |m: String| fmt(format!("tensor {name}: {m}"));
        let obj = v
            .as_object()
            .ok_or_else(|| bad("entry is not an object".into()))?;
        let tag = obj
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| bad("missing dtype".into()))?;
        let dtype = dtype_from_tag(tag).ok_or_else(|| bad(format!("unsupported dtype {tag}")))?;
        let dims: Vec<u64> = obj
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("missing shape".into()))?
            .iter()
            .map(|d| d.as_u64().ok_or_else(|| bad(format!("bad dimension {d}"))))
            .collect::<Result<_>>()?;
        let offs = obj
            .get("data_offsets")
            .and_then(Value::as_array)
            .filter(|a| a.len() == 2)
            .ok_or_else(|| bad("missing data_offsets [begin, end]".into()))?;
        let begin = offs[0]
            .as_u64()
            .ok_or_else(|| bad("bad data_offsets".into()))?;
        let end = offs[1]
            .as_u64()
            .ok_or_else(|| bad("bad data_offsets".into()))?;
        if begin > end || end > data_len {
            return Err(bad(format!(
                "data_offsets [{begin}, {end}) outside the {data_len}-byte buffer"
            )));
        }
        let numel = dims
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| bad("shape overflows".into()))?;
        let expected = numel
            .checked_mul(dtype.block_bytes() as u64)
            .ok_or_else(|| bad("size overflows".into()))?;
        if end - begin != expected {
            return Err(bad(format!(
                "{tag} {dims:?} needs {expected} bytes but data_offsets span {}",
                end - begin
            )));
        }
        let shape = ggml_shape(&dims)?;
        ranges.push((begin, end, tensors.len()));
        tensors.push((
            TensorInfo {
                name: name.clone(),
                dtype,
                shape,
                span: ByteSpan {
                    source: 0,
                    offset: begin,
                    len: end - begin,
                },
            },
            dims,
        ));
    }
    // No two non-empty tensors may share bytes.
    ranges.retain(|r| r.1 > r.0);
    ranges.sort_unstable();
    for w in ranges.windows(2) {
        let (a, b) = (w[0], w[1]);
        if b.0 < a.1 {
            return Err(fmt(format!(
                "tensors {} and {} overlap ([{}, {}) and [{}, {}))",
                tensors[a.2].0.name, tensors[b.2].0.name, a.0, a.1, b.0, b.1
            )));
        }
    }
    Ok(ParsedHeader {
        metadata,
        tensors,
        data_offset,
    })
}

/// Row-major HF dims → ggml order (innermost first). Scalars become `[1]`; more than four dims
/// fold the outer ones into `ne[3]`.
fn ggml_shape(dims: &[u64]) -> Result<Shape> {
    if dims.is_empty() {
        return Shape::new(&[1]);
    }
    let mut ne: Vec<u64> = dims.iter().rev().copied().collect();
    if ne.len() > 4 {
        let outer: u64 = ne[3..].iter().product();
        ne.truncate(3);
        ne.push(outer);
    }
    Shape::new(&ne)
}

/// Minimal safetensors writer used by tests and by `testkit` to build synthetic folders.
pub mod writer {
    use super::INDEX_FILE;
    use serde_json::{json, Value};
    use std::collections::BTreeMap;
    use std::path::Path;

    /// `(name, dtype tag, row-major dims, little-endian bytes)`.
    type Entry = (String, &'static str, Vec<u64>, Vec<u8>);

    #[derive(Default)]
    pub struct SafetensorsWriter {
        metadata: BTreeMap<String, String>,
        tensors: Vec<Entry>,
    }

    impl SafetensorsWriter {
        pub fn new() -> Self {
            Self::default()
        }
        /// MLX exports write `{"format": "mlx"}`.
        pub fn metadata(&mut self, k: &str, v: &str) -> &mut Self {
            self.metadata.insert(k.into(), v.into());
            self
        }
        /// Add a tensor with its safetensors dtype tag (`"F32"`, `"BF16"`, `"U32"`, ...) and
        /// row-major dims; `data` must be `numel * elem_size` little-endian bytes.
        pub fn tensor(
            &mut self,
            name: &str,
            dtype: &'static str,
            dims: &[u64],
            data: Vec<u8>,
        ) -> &mut Self {
            self.tensors
                .push((name.to_string(), dtype, dims.to_vec(), data));
            self
        }
        pub fn names(&self) -> Vec<String> {
            self.tensors.iter().map(|t| t.0.clone()).collect()
        }
        pub fn to_bytes(&self) -> Vec<u8> {
            let mut header = serde_json::Map::new();
            if !self.metadata.is_empty() {
                header.insert("__metadata__".into(), json!(self.metadata));
            }
            let mut off = 0u64;
            for (name, dtype, dims, data) in &self.tensors {
                let end = off + data.len() as u64;
                header.insert(
                    name.clone(),
                    json!({"dtype": dtype, "shape": dims, "data_offsets": [off, end]}),
                );
                off = end;
            }
            let mut hj = serde_json::to_vec(&Value::Object(header)).unwrap();
            // Pad with spaces to a multiple of 8 like the reference writer does.
            while hj.len() % 8 != 0 {
                hj.push(b' ');
            }
            let mut out = Vec::with_capacity(8 + hj.len() + off as usize);
            out.extend_from_slice(&(hj.len() as u64).to_le_bytes());
            out.extend_from_slice(&hj);
            for (_, _, _, data) in &self.tensors {
                out.extend_from_slice(data);
            }
            out
        }
    }

    /// Write a folder: `config.json`, and either a single `model.safetensors` or the named
    /// shards plus `model.safetensors.index.json`.
    pub fn write_folder(
        dir: &Path,
        config: &Value,
        shards: &[(&str, &SafetensorsWriter)],
    ) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec_pretty(config).unwrap(),
        )?;
        let mut total = 0u64;
        let mut weight_map = serde_json::Map::new();
        for (file, w) in shards {
            let bytes = w.to_bytes();
            total += bytes.len() as u64;
            std::fs::write(dir.join(file), bytes)?;
            for n in w.names() {
                weight_map.insert(n, json!(file));
            }
        }
        if shards.len() > 1 {
            let idx = json!({"metadata": {"total_size": total}, "weight_map": weight_map});
            std::fs::write(
                dir.join(INDEX_FILE),
                serde_json::to_vec_pretty(&idx).unwrap(),
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::writer::{write_folder, SafetensorsWriter};
    use super::*;
    use llmario_engine_core::dequant::mlx_affine_pack;
    use serde_json::json;

    fn f32s(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
    fn f16s(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
            .collect()
    }
    fn bf16s(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }
    fn u32s(v: &[u32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn qwen_config() -> Value {
        json!({
            "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
            "hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 96,
            "vocab_size": 32, "rms_norm_eps": 1e-6, "rope_theta": 1000000,
            "rope_scaling": null, "tie_word_embeddings": true, "max_position_embeddings": 4096,
            "eos_token_id": [7, 9], "bos_token_id": 1
        })
    }

    #[test]
    fn single_file_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = SafetensorsWriter::new();
        w.metadata("format", "mlx")
            .tensor("a", "F32", &[2, 3], f32s(&[1., 2., 3., 4., 5., 6.]))
            .tensor("norm", "BF16", &[4], bf16s(&[1., 2., 3., 4.]))
            .tensor("packed", "U32", &[2, 8], u32s(&[7; 16]))
            .tensor("scalar", "F32", &[], f32s(&[9.]))
            .tensor("empty", "F16", &[0, 8], vec![])
            .tensor("five_d", "I8", &[2, 3, 4, 5, 6], vec![1u8; 720]);
        write_folder(dir.path(), &qwen_config(), &[(SINGLE_FILE, &w)]).unwrap();
        let f = SafetensorsFolder::open(dir.path()).unwrap();
        assert_eq!(f.shards.len(), 1);
        assert_eq!(f.shards[0].metadata["format"], "mlx");
        assert!(f.is_mlx());
        assert_eq!(f.tensors.len(), 6);
        let a = f.tensor("a").unwrap();
        assert_eq!(a.dtype, GgmlType::F32);
        assert_eq!(a.shape.dims(), &[3, 2], "ggml order: innermost first");
        assert_eq!(f.hf_dims("a").unwrap(), &[2, 3]);
        assert_eq!(f.tensor_bytes(a), f32s(&[1., 2., 3., 4., 5., 6.]));
        let mut out = vec![0f32; 3];
        f.dequantize_rows(a, 1, 1, &mut out).unwrap();
        assert_eq!(out, [4., 5., 6.]);
        let p = f.tensor("packed").unwrap();
        assert_eq!(p.dtype, GgmlType::U32);
        assert_eq!(p.span.len, 64);
        assert_eq!(p.row_bytes(), 32);
        assert_eq!(f.tensor("scalar").unwrap().shape.dims(), &[1]);
        assert_eq!(f.tensor("empty").unwrap().span.len, 0);
        assert_eq!(f.tensor("five_d").unwrap().shape.dims(), &[6, 5, 4, 6]);
        assert_eq!(f.tensor_bytes_total(), 24 + 8 + 64 + 4 + 720);
        assert_eq!(f.bytes_by_type()[&GgmlType::F32], 28);
        assert!(f.tensor("missing").is_none());
        assert_eq!(f.index_total_size, None);
        // Config accessors.
        assert_eq!(f.model_type(), Some("qwen3"));
        assert_eq!(f.architectures(), ["Qwen3ForCausalLM"]);
        assert_eq!(f.hidden_size(), Some(64));
        assert_eq!(f.num_key_value_heads(), Some(2));
        assert_eq!(f.head_dim(), Some(16));
        assert_eq!(f.rope_theta(), Some(1000000.0));
        assert!(f.rope_scaling().is_none(), "null rope_scaling is None");
        assert_eq!(f.tie_word_embeddings(), Some(true));
        assert_eq!(f.eos_token_ids(), [7, 9]);
        assert_eq!(f.bos_token_id(), Some(1));
        assert!(f.quantization().is_none());
        assert!(f.tokenizer_json_path().is_none());
        assert!(f.chat_template().is_none());
    }

    #[test]
    fn sharded_with_index() {
        let dir = tempfile::tempdir().unwrap();
        let mut s1 = SafetensorsWriter::new();
        s1.tensor("model.norm.weight", "F16", &[4], f16s(&[1., 1., 1., 1.]));
        let mut s2 = SafetensorsWriter::new();
        s2.tensor("lm_head.weight", "F16", &[2, 4], f16s(&[0.; 8]));
        write_folder(
            dir.path(),
            &qwen_config(),
            &[
                ("model-00001-of-00002.safetensors", &s1),
                ("model-00002-of-00002.safetensors", &s2),
            ],
        )
        .unwrap();
        let f = SafetensorsFolder::open(dir.path()).unwrap();
        assert_eq!(f.shards.len(), 2);
        assert_eq!(f.tensor("model.norm.weight").unwrap().span.source, 0);
        assert_eq!(f.tensor("lm_head.weight").unwrap().span.source, 1);
        assert_eq!(
            f.index_total_size,
            Some(f.shards.iter().map(|s| s.mmap.len() as u64).sum())
        );
        let mut out = vec![0f32; 4];
        f.dequantize_rows(f.tensor("model.norm.weight").unwrap(), 0, 1, &mut out)
            .unwrap();
        assert_eq!(out, [1., 1., 1., 1.]);

        // Index pointing at the wrong shard, or at a tensor that does not exist.
        let idx_path = dir.path().join(INDEX_FILE);
        let mut idx: Value = serde_json::from_slice(&std::fs::read(&idx_path).unwrap()).unwrap();
        idx["weight_map"]["lm_head.weight"] = json!("model-00001-of-00002.safetensors");
        std::fs::write(&idx_path, idx.to_string()).unwrap();
        let e = SafetensorsFolder::open(dir.path()).unwrap_err().to_string();
        assert!(e.contains("lm_head.weight"), "{e}");
        idx["weight_map"]["lm_head.weight"] = json!("model-00002-of-00002.safetensors");
        idx["weight_map"]["ghost"] = json!("model-00002-of-00002.safetensors");
        std::fs::write(&idx_path, idx.to_string()).unwrap();
        let e = SafetensorsFolder::open(dir.path()).unwrap_err().to_string();
        assert!(e.contains("ghost"), "{e}");
        idx["weight_map"]["ghost"] = json!("model-00009-of-00002.safetensors");
        std::fs::write(&idx_path, idx.to_string()).unwrap();
        assert!(SafetensorsFolder::open(dir.path()).is_err());

        // Without an index and without model.safetensors, every *.safetensors file is read.
        std::fs::remove_file(&idx_path).unwrap();
        let f = SafetensorsFolder::open(dir.path()).unwrap();
        assert_eq!(f.tensors.len(), 2);
        assert_eq!(f.index_total_size, None);

        // Duplicate names across shards are refused.
        let mut s3 = SafetensorsWriter::new();
        s3.tensor("model.norm.weight", "F16", &[4], f16s(&[2.; 4]));
        std::fs::write(dir.path().join("extra.safetensors"), s3.to_bytes()).unwrap();
        let e = SafetensorsFolder::open(dir.path()).unwrap_err().to_string();
        assert!(e.contains("duplicate"), "{e}");
    }

    fn raw(header: &str, data_len: usize) -> Vec<u8> {
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out.extend(std::iter::repeat(0u8).take(data_len));
        out
    }

    #[test]
    fn rejects_malformed_headers() {
        let ok = r#"{"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"b":{"dtype":"F16","shape":[4],"data_offsets":[16,24]}}"#;
        let h = parse_header(&raw(ok, 24)).unwrap();
        assert_eq!(h.tensors.len(), 2);
        assert_eq!(h.data_offset, 8 + ok.len() as u64);

        let err = |hdr: &str, data: usize| parse_header(&raw(hdr, data)).unwrap_err().to_string();
        // Header length beyond the file.
        let mut trunc = raw(ok, 24);
        trunc.truncate(40);
        assert!(parse_header(&trunc)
            .unwrap_err()
            .to_string()
            .contains("exceeds"));
        assert!(parse_header(&[1, 2, 3]).is_err());
        // Not a JSON object.
        assert!(err("[1,2]", 0).contains("start with"));
        assert!(err("{not json", 0).contains("not a JSON object"));
        // Range past the buffer, inverted range.
        assert!(err(
            r#"{"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#,
            8
        )
        .contains("outside"));
        assert!(err(
            r#"{"a":{"dtype":"F32","shape":[0],"data_offsets":[8,4]}}"#,
            8
        )
        .contains("outside"));
        // Size disagrees with the shape.
        assert!(err(
            r#"{"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,12]}}"#,
            16
        )
        .contains("needs 16 bytes"));
        // Overlap.
        let e = err(
            r#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},"b":{"dtype":"F16","shape":[4],"data_offsets":[8,16]}}"#,
            16,
        );
        assert!(e.contains("overlap"), "{e}");
        // Unknown / unsupported dtype.
        assert!(err(
            r#"{"a":{"dtype":"F8_E4M3","shape":[4],"data_offsets":[0,4]}}"#,
            4
        )
        .contains("unsupported dtype F8_E4M3"));
        assert!(err(
            r#"{"a":{"dtype":"BOOL","shape":[4],"data_offsets":[0,4]}}"#,
            4
        )
        .contains("unsupported dtype"));
        // Missing fields and bad dims.
        assert!(err(r#"{"a":{"shape":[4],"data_offsets":[0,4]}}"#, 4).contains("missing dtype"));
        assert!(err(r#"{"a":{"dtype":"F32","data_offsets":[0,4]}}"#, 4).contains("missing shape"));
        assert!(err(
            r#"{"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}"#,
            4
        )
        .contains("bad dimension"));
        assert!(
            err(r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0]}}"#, 4)
                .contains("data_offsets")
        );
        assert!(err(r#"{"a":5}"#, 4).contains("not an object"));
        // Oversized header length field.
        let mut huge = raw(ok, 24);
        huge[..8].copy_from_slice(&(MAX_HEADER + 1).to_le_bytes());
        assert!(parse_header(&huge)
            .unwrap_err()
            .to_string()
            .contains("100 MB"));
        // Empty tensors never overlap anything; metadata is kept; trailing header padding is fine.
        let padded = r#"{"__metadata__":{"format":"mlx","x":"y"},"e":{"dtype":"F32","shape":[0,4],"data_offsets":[4,4]},"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}   "#;
        let h = parse_header(&raw(padded, 8)).unwrap();
        assert_eq!(h.metadata["format"], "mlx");
        assert_eq!(h.tensors.len(), 2);
    }

    #[test]
    fn text_config_fallback_and_rope_parameters() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text", "hidden_size": 5120, "num_hidden_layers": 64,
                "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
                "intermediate_size": 17408, "vocab_size": 248320, "rms_norm_eps": 1e-6,
                "max_position_embeddings": 262144, "tie_word_embeddings": false,
                "rope_parameters": {"rope_theta": 10000000, "rope_type": "default", "partial_rotary_factor": 0.25}
            },
            "tie_word_embeddings": false,
            "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}
        });
        let mut w = SafetensorsWriter::new();
        w.tensor("x", "F32", &[1], f32s(&[0.]));
        write_folder(dir.path(), &cfg, &[(SINGLE_FILE, &w)]).unwrap();
        let f = SafetensorsFolder::open(dir.path()).unwrap();
        assert_eq!(f.model_type(), Some("qwen3_5"), "top level wins");
        assert_eq!(f.hidden_size(), Some(5120));
        assert_eq!(f.num_hidden_layers(), Some(64));
        assert_eq!(f.head_dim(), Some(256));
        assert_eq!(f.vocab_size(), Some(248320));
        assert_eq!(f.rope_theta(), Some(10000000.0));
        assert_eq!(f.rope_scaling().unwrap()["rope_type"], "default");
        assert_eq!(f.max_position_embeddings(), Some(262144));
        let q = f.quantization().unwrap();
        assert_eq!(
            q.default,
            MlxQuantParams {
                bits: 4,
                group_size: 64,
                mode: "affine".into()
            }
        );
        assert!(q.overrides.is_empty());
        // No head_dim: derived.
        let f2 = {
            let dir2 = tempfile::tempdir().unwrap();
            let cfg =
                json!({"model_type": "llama", "hidden_size": 3072, "num_attention_heads": 24});
            write_folder(dir2.path(), &cfg, &[(SINGLE_FILE, &w)]).unwrap();
            let f = SafetensorsFolder::open(dir2.path()).unwrap();
            (f.head_dim(), f.num_key_value_heads(), f.rope_theta())
        };
        assert_eq!(f2, (Some(128), Some(24), None));
    }

    #[test]
    fn mlx_quantization_overrides_and_hf_configs() {
        let v = json!({"bits": 4, "group_size": 64,
            "model.layers.0.mlp.gate_proj": {"bits": 8, "group_size": 32},
            "lm_head": false, "model.embed_tokens": true, "note": "ignored"});
        let q = MlxQuantization::from_value(&v).unwrap();
        assert_eq!(q.default.mode, "affine");
        assert_eq!(q.params_for("model.layers.1.mlp.up_proj").unwrap().bits, 4);
        let g = q.params_for("model.layers.0.mlp.gate_proj").unwrap();
        assert_eq!((g.bits, g.group_size), (8, 32));
        assert!(q.params_for("lm_head").is_none());
        assert_eq!(q.params_for("model.embed_tokens").unwrap().bits, 4);
        assert_eq!(q.overrides.len(), 3);
        // Override with only bits inherits the group size.
        let v = json!({"bits": 4, "group_size": 64, "a": {"bits": 6}});
        let q = MlxQuantization::from_value(&v).unwrap();
        assert_eq!(q.params_for("a").unwrap().group_size, 64);
        // HF quantization_config (GPTQ/AWQ) is not MLX.
        let hf = json!({"quant_method": "awq", "bits": 4, "group_size": 128});
        assert!(MlxQuantization::from_value(&hf).is_none());
        assert!(MlxQuantization::from_value(&json!({"bits": 4})).is_none());
        assert!(MlxQuantization::from_value(&json!(null)).is_none());
    }

    /// Build a folder with one 4-bit and one 3-bit quantised module plus a norm.
    fn mlx_folder(dir: &Path, scale_tag: &'static str) -> (Vec<u32>, Vec<u32>) {
        let cfg = json!({
            "model_type": "qwen3", "hidden_size": 64, "num_attention_heads": 2,
            "quantization": {"group_size": 32, "bits": 4,
                             "model.layers.0.mlp.down_proj": {"group_size": 32, "bits": 3},
                             "model.layers.0.mlp.up_proj": false},
            "quantization_config": {"group_size": 32, "bits": 4}
        });
        let (rows, cols) = (4u64, 64u64);
        let q4: Vec<u32> = (0..rows * cols).map(|i| (i * 7 + 3) as u32 % 16).collect();
        let q3: Vec<u32> = (0..rows * cols).map(|i| (i * 5 + 1) as u32 % 8).collect();
        let pack = |bits: u32, q: &[u32]| -> Vec<u8> {
            q.chunks_exact(cols as usize)
                .flat_map(|row| u32s(&mlx_affine_pack(bits, row)))
                .collect()
        };
        let groups = (rows * cols / 32) as usize;
        let scales: Vec<f32> = (0..groups).map(|g| 0.25 + g as f32 * 0.125).collect();
        let biases: Vec<f32> = (0..groups).map(|g| -1.0 + g as f32 * 0.5).collect();
        let enc = |v: &[f32]| match scale_tag {
            "F16" => f16s(v),
            "BF16" => bf16s(v),
            _ => f32s(v),
        };
        let mut w = SafetensorsWriter::new();
        w.metadata("format", "mlx")
            .tensor(
                "model.layers.0.self_attn.q_proj.weight",
                "U32",
                &[rows, cols * 4 / 32],
                pack(4, &q4),
            )
            .tensor(
                "model.layers.0.self_attn.q_proj.scales",
                scale_tag,
                &[rows, cols / 32],
                enc(&scales),
            )
            .tensor(
                "model.layers.0.self_attn.q_proj.biases",
                scale_tag,
                &[rows, cols / 32],
                enc(&biases),
            )
            .tensor(
                "model.layers.0.mlp.down_proj.weight",
                "U32",
                &[rows, cols * 3 / 32],
                pack(3, &q3),
            )
            .tensor(
                "model.layers.0.mlp.down_proj.scales",
                scale_tag,
                &[rows, cols / 32],
                enc(&scales),
            )
            .tensor(
                "model.layers.0.mlp.down_proj.biases",
                scale_tag,
                &[rows, cols / 32],
                enc(&biases),
            )
            .tensor(
                "model.layers.0.mlp.up_proj.weight",
                "BF16",
                &[rows, cols],
                bf16s(&[0.5; 256]),
            )
            .tensor("model.norm.weight", "BF16", &[64], bf16s(&[1.0; 64]))
            // Inconsistent triplets: scales of the wrong shape; weight not U32; biases missing.
            .tensor("bad_shape.weight", "U32", &[rows, 8], u32s(&[0; 32]))
            .tensor("bad_shape.scales", scale_tag, &[rows, 4], enc(&[1.0; 16]))
            .tensor("bad_shape.biases", scale_tag, &[rows, 4], enc(&[0.0; 16]))
            .tensor("bad_dtype.weight", "I32", &[rows, 8], u32s(&[0; 32]))
            .tensor("bad_dtype.scales", scale_tag, &[rows, 2], enc(&[1.0; 8]))
            .tensor("bad_dtype.biases", scale_tag, &[rows, 2], enc(&[0.0; 8]))
            .tensor("no_biases.weight", "U32", &[rows, 8], u32s(&[0; 32]))
            .tensor("no_biases.scales", scale_tag, &[rows, 2], enc(&[1.0; 8]));
        write_folder(dir, &cfg, &[(SINGLE_FILE, &w)]).unwrap();
        (q4, q3)
    }

    #[test]
    fn mlx_views_and_dequantisation() {
        for tag in ["F16", "BF16", "F32"] {
            let dir = tempfile::tempdir().unwrap();
            let (q4, q3) = mlx_folder(dir.path(), tag);
            let f = SafetensorsFolder::open(dir.path()).unwrap();
            assert!(f.is_mlx());
            let v = f
                .mlx_quant_view("model.layers.0.self_attn.q_proj")
                .unwrap()
                .unwrap();
            assert_eq!((v.bits, v.group_size, v.rows, v.cols), (4, 32, 4, 64));
            assert_eq!(v.weight.dtype, GgmlType::U32);
            assert_eq!(v.scale_dtype, dtype_from_tag(tag).unwrap());
            assert_eq!(v.weight.shape.dims(), &[8, 4]);
            assert_eq!(v.weight_row_bytes(), 32);
            assert_eq!(v.scales_row_bytes(), 2 * v.scale_dtype.block_bytes());
            let bpw = 4.0 + 2.0 * v.scale_dtype.block_bytes() as f64 * 8.0 / 32.0;
            assert!((v.bits_per_weight() - bpw).abs() < 1e-9);
            let mut out = vec![0f32; 4 * 64];
            f.dequantize_mlx_rows(&v, 0, 4, &mut out).unwrap();
            for (i, y) in out.iter().enumerate() {
                let g = i / 32; // groups are numbered row-major across the matrix
                let s = 0.25 + g as f32 * 0.125;
                let b = -1.0 + g as f32 * 0.5;
                assert_eq!(*y, s * q4[i] as f32 + b, "{tag} 4-bit element {i}");
            }
            // Row subsets.
            let mut two = vec![0f32; 2 * 64];
            f.dequantize_mlx_rows(&v, 1, 2, &mut two).unwrap();
            assert_eq!(two, out[64..192]);
            assert!(f.dequantize_mlx_rows(&v, 3, 2, &mut two).is_err());
            assert!(f.dequantize_mlx_rows(&v, 0, 1, &mut two).is_err());

            // The 3-bit override.
            let d = f
                .mlx_quant_view("model.layers.0.mlp.down_proj")
                .unwrap()
                .unwrap();
            assert_eq!((d.bits, d.group_size, d.cols), (3, 32, 64));
            assert_eq!(d.weight.shape.dims(), &[6, 4]);
            let mut out3 = vec![0f32; 4 * 64];
            f.dequantize_mlx_rows(&d, 0, 4, &mut out3).unwrap();
            for (i, y) in out3.iter().enumerate() {
                let g = i / 32;
                let s = 0.25 + g as f32 * 0.125;
                let b = -1.0 + g as f32 * 0.5;
                assert_eq!(*y, s * q3[i] as f32 + b, "{tag} 3-bit element {i}");
            }

            // Unquantised modules and inconsistent triplets.
            assert!(f
                .mlx_quant_view("model.layers.0.mlp.up_proj")
                .unwrap()
                .is_none());
            assert!(f.mlx_quant_view("model.norm").unwrap().is_none());
            assert!(f.mlx_quant_view("nope").unwrap().is_none());
            for m in ["bad_shape", "bad_dtype", "no_biases"] {
                assert!(f.mlx_quant_view(m).is_err(), "{m} must be rejected");
            }
            assert!(
                f.mlx_quant_views().is_err(),
                "the bad triplets poison the sweep"
            );
        }
    }

    #[test]
    fn mlx_views_sweep_and_missing_config() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = json!({"model_type": "llama", "quantization": {"group_size": 64, "bits": 8}});
        let rows = 2u64;
        let q: Vec<u32> = (0..rows * 64).map(|i| i as u32 % 256).collect();
        let packed: Vec<u8> = q
            .chunks_exact(64)
            .flat_map(|r| u32s(&mlx_affine_pack(8, r)))
            .collect();
        let mut w = SafetensorsWriter::new();
        w.tensor("lm_head.weight", "U32", &[rows, 16], packed)
            .tensor("lm_head.scales", "F16", &[rows, 1], f16s(&[2.0, 0.5]))
            .tensor("lm_head.biases", "F16", &[rows, 1], f16s(&[0.0, 1.0]))
            .tensor("model.norm.weight", "F16", &[64], f16s(&[1.0; 64]));
        write_folder(dir.path(), &cfg, &[(SINGLE_FILE, &w)]).unwrap();
        let f = SafetensorsFolder::open(dir.path()).unwrap();
        let views = f.mlx_quant_views().unwrap();
        assert_eq!(views.len(), 1);
        assert_eq!(views[0].module, "lm_head");
        let mut out = vec![0f32; 128];
        f.dequantize_mlx_rows(&views[0], 0, 2, &mut out).unwrap();
        assert_eq!(out[5], 2.0 * 5.0);
        assert_eq!(out[64 + 5], 0.5 * 69.0 + 1.0);

        // Same tensors, no quantization block: the triplet cannot be interpreted.
        let dir2 = tempfile::tempdir().unwrap();
        write_folder(
            dir2.path(),
            &json!({"model_type": "llama"}),
            &[(SINGLE_FILE, &w)],
        )
        .unwrap();
        let f2 = SafetensorsFolder::open(dir2.path()).unwrap();
        assert!(f2.quantization().is_none());
        assert!(f2.mlx_quant_view("lm_head").is_err());
        // HF quantization_config is exposed raw and not mistaken for MLX.
        let dir3 = tempfile::tempdir().unwrap();
        let cfg3 = json!({"model_type": "llama", "quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128}});
        write_folder(dir3.path(), &cfg3, &[(SINGLE_FILE, &w)]).unwrap();
        let f3 = SafetensorsFolder::open(dir3.path()).unwrap();
        assert!(f3.quantization().is_none());
        assert_eq!(f3.hf_quantization_config().unwrap()["quant_method"], "gptq");
        assert!(!f3.is_mlx());
    }

    #[test]
    fn chat_template_sources() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = SafetensorsWriter::new();
        w.tensor("x", "F32", &[1], f32s(&[0.]));
        write_folder(dir.path(), &qwen_config(), &[(SINGLE_FILE, &w)]).unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            json!({"chat_template": "from tokenizer_config", "eos_token": "<|im_end|>"})
                .to_string(),
        )
        .unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        let f = SafetensorsFolder::open(dir.path()).unwrap();
        assert_eq!(f.chat_template().as_deref(), Some("from tokenizer_config"));
        assert_eq!(
            f.tokenizer_json_path().unwrap(),
            dir.path().join("tokenizer.json")
        );
        assert_eq!(
            f.tokenizer_config.as_ref().unwrap()["eos_token"],
            "<|im_end|>"
        );
        // A list of named templates: `default` wins.
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            json!({"chat_template": [{"name": "tool_use", "template": "t"}, {"name": "default", "template": "d"}]}).to_string(),
        )
        .unwrap();
        let f = SafetensorsFolder::open(dir.path()).unwrap();
        assert_eq!(f.chat_template().as_deref(), Some("d"));
        std::fs::write(
            dir.path().join("chat_template.json"),
            json!({"chat_template": "from json"}).to_string(),
        )
        .unwrap();
        assert_eq!(f.chat_template().as_deref(), Some("from json"));
        std::fs::write(dir.path().join("chat_template.jinja"), "from jinja").unwrap();
        assert_eq!(f.chat_template().as_deref(), Some("from jinja"));
    }

    #[test]
    fn open_errors() {
        let dir = tempfile::tempdir().unwrap();
        let e = SafetensorsFolder::open(dir.path()).unwrap_err().to_string();
        assert!(e.contains("not a model folder"), "{e}");
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        let e = SafetensorsFolder::open(dir.path()).unwrap_err().to_string();
        assert!(e.contains("no .safetensors"), "{e}");
        std::fs::write(dir.path().join(SINGLE_FILE), b"garbage").unwrap();
        let e = SafetensorsFolder::open(dir.path()).unwrap_err().to_string();
        assert!(e.contains(SINGLE_FILE), "{e}");
        assert_eq!(dtype_tag(GgmlType::U32), Some("U32"));
        assert_eq!(dtype_tag(GgmlType::Q4_K), None);
        assert_eq!(dtype_from_tag("U32"), Some(GgmlType::U32));
    }
}
