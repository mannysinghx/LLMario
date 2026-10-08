//! Minimal, bounded GGUF metadata reader. Reads only the header (key/value section and the
//! tensor index: names and offsets) — never tensor data — and enforces size limits so a
//! malformed file cannot exhaust memory.
//! Format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

use std::collections::BTreeMap;
use std::io::{BufReader, Read};
use std::path::Path;

const MAX_KEY_LEN: u64 = 64 * 1024;
const MAX_STR_LEN: u64 = 32 * 1024 * 1024;
const MAX_ARRAY_LEN: u64 = 16 * 1024 * 1024;
const MAX_KV: u64 = 1 << 20;
const MAX_TENSORS: u64 = 1 << 20;
const MAX_DIMS: u32 = 8;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Uint(u64),
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    /// Arrays keep their length; small numeric arrays keep their values (e.g. per-layer
    /// `head_count_kv`). String arrays such as the vocabulary are skipped.
    Array {
        len: u64,
        numbers: Option<Vec<i64>>,
    },
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Uint(v) => Some(*v),
            Value::Int(v) if *v >= 0 => Some(*v as u64),
            // Per-layer arrays: the planner needs the worst case.
            Value::Array {
                numbers: Some(n), ..
            } => n.iter().copied().max().map(|m| m.max(0) as u64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Debug, Default)]
pub struct GgufMetadata {
    pub version: u32,
    pub tensor_count: u64,
    pub kv: BTreeMap<String, Value>,
    /// Tensor names and their offsets within the data section, in header order. Empty if the
    /// tensor index could not be read.
    pub tensors: Vec<(String, u64)>,
    /// Byte offset of the data section in the file (the header length, aligned).
    pub data_offset: u64,
}

impl GgufMetadata {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.kv.get(key)
    }
    pub fn architecture(&self) -> Option<&str> {
        self.get("general.architecture").and_then(Value::as_str)
    }
    /// Look up `<arch>.<suffix>`.
    pub fn arch_u64(&self, suffix: &str) -> Option<u64> {
        let arch = self.architecture()?;
        self.get(&format!("{arch}.{suffix}"))
            .and_then(Value::as_u64)
    }

    /// Size of every tensor in bytes, from the gaps between their offsets (the last runs to the
    /// end of the file). `file_len` is the whole file's size. `None` if the index is missing.
    pub fn tensor_sizes(&self, file_len: u64) -> Option<Vec<(&str, u64)>> {
        if self.tensors.is_empty() || file_len <= self.data_offset {
            return None;
        }
        let mut by_offset: Vec<&(String, u64)> = self.tensors.iter().collect();
        by_offset.sort_by_key(|(_, off)| *off);
        let data_len = file_len - self.data_offset;
        let mut out = Vec::with_capacity(by_offset.len());
        for (i, (name, off)) in by_offset.iter().enumerate() {
            let end = by_offset.get(i + 1).map(|(_, o)| *o).unwrap_or(data_len);
            out.push((name.as_str(), end.checked_sub(*off)?));
        }
        Some(out)
    }

    /// `(total, read per token)` bytes of MoE expert tensors (`*_exps`); `None` if the tensor
    /// index is missing. Dense models give `(0, 0)`.
    pub fn expert_bytes(&self, file_len: u64) -> Option<(u64, u64)> {
        let total: u64 = self
            .tensor_sizes(file_len)?
            .iter()
            .filter(|(n, _)| n.contains("_exps"))
            .map(|(_, s)| s)
            .sum();
        let active = match (
            self.arch_u64("expert_used_count"),
            self.arch_u64("expert_count"),
        ) {
            (Some(u), Some(c)) if c > 0 => (total as u128 * u.min(c) as u128 / c as u128) as u64,
            _ => total,
        };
        Some((total, active))
    }

    /// Weight bytes read to generate one token: every tensor, except that MoE expert tensors
    /// (`*_exps`) count at the share of experts used per token, and the token-embedding table is
    /// left out when a separate output head exists (only one row of it is read per token).
    /// `None` if the tensor index is missing.
    pub fn bytes_per_token(&self, file_len: u64) -> Option<u64> {
        let sizes = self.tensor_sizes(file_len)?;
        let has_output = sizes.iter().any(|(n, _)| *n == "output.weight");
        let used = self.arch_u64("expert_used_count");
        let count = self.arch_u64("expert_count").filter(|c| *c > 0);
        let mut total = 0u64;
        for (name, size) in sizes {
            if has_output && name.starts_with("token_embd.") {
                continue;
            }
            total += match (name.contains("_exps"), used, count) {
                (true, Some(u), Some(c)) => (size as u128 * u.min(c) as u128 / c as u128) as u64,
                _ => size,
            };
        }
        Some(total)
    }
}

/// Counts bytes read, so the parser knows where the header ends.
struct Counting<R> {
    inner: R,
    pos: u64,
}

impl<R: Read> Read for Counting<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.pos += n as u64;
        Ok(n)
    }
}

pub fn read_metadata(path: &Path) -> anyhow::Result<GgufMetadata> {
    let f = std::fs::File::open(path)?;
    parse(&mut BufReader::with_capacity(1 << 16, f))
}

pub fn parse(r: &mut impl Read) -> anyhow::Result<GgufMetadata> {
    let r = &mut Counting { inner: r, pos: 0 };
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        anyhow::bail!("not a GGUF file (bad magic)");
    }
    let version = read_u32(r)?;
    if !(2..=3).contains(&version) {
        anyhow::bail!("unsupported GGUF version {version} (supported: 2, 3)");
    }
    let tensor_count = read_u64(r)?;
    let kv_count = read_u64(r)?;
    if kv_count > MAX_KV {
        anyhow::bail!("GGUF declares {kv_count} metadata entries; refusing");
    }
    let mut kv = BTreeMap::new();
    for _ in 0..kv_count {
        let key = read_string(r, MAX_KEY_LEN)?;
        let ty = read_u32(r)?;
        let v = read_value(r, ty, true)?;
        kv.insert(key, v);
    }
    let mut md = GgufMetadata {
        version,
        tensor_count,
        kv,
        ..Default::default()
    };
    // The tensor index is optional for callers that only need metadata: on any problem, keep
    // the metadata and leave the index empty.
    if let Ok(tensors) = read_tensor_index(r, tensor_count) {
        let align = md
            .get("general.alignment")
            .and_then(Value::as_u64)
            .filter(|a| *a > 0)
            .unwrap_or(32);
        md.data_offset = r.pos.div_ceil(align) * align;
        md.tensors = tensors;
    }
    Ok(md)
}

fn read_tensor_index(r: &mut impl Read, count: u64) -> anyhow::Result<Vec<(String, u64)>> {
    if count > MAX_TENSORS {
        anyhow::bail!("GGUF declares {count} tensors; refusing");
    }
    let mut out = Vec::with_capacity(count.min(4096) as usize);
    for _ in 0..count {
        let name = read_string(r, MAX_KEY_LEN)?;
        let n_dims = read_u32(r)?;
        if n_dims > MAX_DIMS {
            anyhow::bail!("GGUF tensor with {n_dims} dimensions");
        }
        for _ in 0..n_dims {
            read_u64(r)?;
        }
        let _ty = read_u32(r)?;
        out.push((name, read_u64(r)?));
    }
    Ok(out)
}

fn read_value(r: &mut impl Read, ty: u32, allow_array: bool) -> anyhow::Result<Value> {
    Ok(match ty {
        0 => Value::Uint(read_n::<1>(r)?[0] as u64),
        1 => Value::Int(read_n::<1>(r)?[0] as i8 as i64),
        2 => Value::Uint(u16::from_le_bytes(read_n::<2>(r)?) as u64),
        3 => Value::Int(i16::from_le_bytes(read_n::<2>(r)?) as i64),
        4 => Value::Uint(read_u32(r)? as u64),
        5 => Value::Int(i32::from_le_bytes(read_n::<4>(r)?) as i64),
        6 => Value::Float(f32::from_le_bytes(read_n::<4>(r)?) as f64),
        7 => Value::Bool(read_n::<1>(r)?[0] != 0),
        8 => Value::Str(read_string(r, MAX_STR_LEN)?),
        9 if allow_array => {
            let elem_ty = read_u32(r)?;
            let len = read_u64(r)?;
            if len > MAX_ARRAY_LEN {
                anyhow::bail!("GGUF array of {len} elements exceeds limit");
            }
            let keep = elem_ty != 8 && len <= 4096;
            let mut numbers = keep.then(Vec::new);
            for _ in 0..len {
                let v = read_value(r, elem_ty, false)?;
                if let Some(n) = numbers.as_mut() {
                    match v {
                        Value::Uint(u) => n.push(u as i64),
                        Value::Int(i) => n.push(i),
                        Value::Bool(b) => n.push(b as i64),
                        _ => {}
                    }
                }
            }
            Value::Array { len, numbers }
        }
        10 => Value::Uint(read_u64(r)?),
        11 => Value::Int(i64::from_le_bytes(read_n::<8>(r)?)),
        12 => Value::Float(f64::from_le_bytes(read_n::<8>(r)?)),
        other => anyhow::bail!("unknown or nested GGUF value type {other}"),
    })
}

fn read_n<const N: usize>(r: &mut impl Read) -> anyhow::Result<[u8; N]> {
    let mut b = [0u8; N];
    r.read_exact(&mut b)?;
    Ok(b)
}
fn read_u32(r: &mut impl Read) -> anyhow::Result<u32> {
    Ok(u32::from_le_bytes(read_n::<4>(r)?))
}
fn read_u64(r: &mut impl Read) -> anyhow::Result<u64> {
    Ok(u64::from_le_bytes(read_n::<8>(r)?))
}
fn read_string(r: &mut impl Read, max: u64) -> anyhow::Result<String> {
    let len = read_u64(r)?;
    if len > max {
        anyhow::bail!("GGUF string of {len} bytes exceeds limit");
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Human name for `general.file_type` (llama.cpp `llama_ftype`).
pub fn file_type_name(ft: u64) -> String {
    let s = match ft {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "MXFP4_MOE",
        _ => return format!("ftype-{ft}"),
    };
    s.to_string()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Build a tiny GGUF v3 header for tests.
    pub fn sample_gguf() -> Vec<u8> {
        sample_gguf_named(None)
    }

    /// Same header, optionally with `general.name` (as Ollama-style blobs carry).
    pub fn sample_gguf_named(name: Option<&str>) -> Vec<u8> {
        fn s(out: &mut Vec<u8>, v: &str) {
            out.extend((v.len() as u64).to_le_bytes());
            out.extend(v.as_bytes());
        }
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(0u64.to_le_bytes()); // tensors
        b.extend((8u64 + name.is_some() as u64).to_le_bytes()); // kv count
        s(&mut b, "general.architecture");
        b.extend(8u32.to_le_bytes());
        s(&mut b, "llama");
        s(&mut b, "llama.block_count");
        b.extend(4u32.to_le_bytes());
        b.extend(28u32.to_le_bytes());
        s(&mut b, "llama.attention.head_count");
        b.extend(4u32.to_le_bytes());
        b.extend(24u32.to_le_bytes());
        s(&mut b, "llama.attention.head_count_kv");
        b.extend(9u32.to_le_bytes()); // array of u32 (per layer)
        b.extend(4u32.to_le_bytes());
        b.extend(3u64.to_le_bytes());
        for v in [8u32, 8, 4] {
            b.extend(v.to_le_bytes());
        }
        s(&mut b, "llama.embedding_length");
        b.extend(4u32.to_le_bytes());
        b.extend(3072u32.to_le_bytes());
        s(&mut b, "general.file_type");
        b.extend(4u32.to_le_bytes());
        b.extend(15u32.to_le_bytes());
        s(&mut b, "tokenizer.ggml.tokens");
        b.extend(9u32.to_le_bytes()); // array of strings: skipped
        b.extend(8u32.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        s(&mut b, "a");
        s(&mut b, "b");
        s(&mut b, "tokenizer.chat_template");
        b.extend(8u32.to_le_bytes());
        s(&mut b, "{% for m in messages %}{{ m.content }}{% endfor %}");
        if let Some(n) = name {
            s(&mut b, "general.name");
            b.extend(8u32.to_le_bytes());
            s(&mut b, n);
        }
        b
    }

    #[test]
    fn parses_header() {
        let m = parse(&mut sample_gguf().as_slice()).unwrap();
        assert_eq!(m.version, 3);
        assert_eq!(m.architecture(), Some("llama"));
        assert_eq!(m.arch_u64("block_count"), Some(28));
        assert_eq!(
            m.arch_u64("attention.head_count_kv"),
            Some(8),
            "max over per-layer array"
        );
        assert_eq!(
            file_type_name(m.get("general.file_type").unwrap().as_u64().unwrap()),
            "Q4_K_M"
        );
        assert!(matches!(
            m.get("tokenizer.ggml.tokens"),
            Some(Value::Array {
                len: 2,
                numbers: None
            })
        ));
        assert!(m.get("tokenizer.chat_template").is_some());
    }

    #[test]
    fn rejects_garbage_and_truncation() {
        assert!(parse(&mut &b"GGML\0\0\0\0"[..]).is_err());
        let full = sample_gguf();
        assert!(parse(&mut &full[..full.len() - 5]).is_err());
        let mut huge = b"GGUF".to_vec();
        huge.extend(3u32.to_le_bytes());
        huge.extend(0u64.to_le_bytes());
        huge.extend(u64::MAX.to_le_bytes());
        assert!(
            parse(&mut huge.as_slice()).is_err(),
            "absurd kv count refused"
        );
    }

    /// A header with an expert count and a tensor index: sizes come from offset gaps, experts
    /// count at used/total, and the embedding table is skipped next to a separate output head.
    #[test]
    fn tensor_index_gives_bytes_per_token() {
        fn s(out: &mut Vec<u8>, v: &str) {
            out.extend((v.len() as u64).to_le_bytes());
            out.extend(v.as_bytes());
        }
        let tensors = [
            ("token_embd.weight", 1000u64),
            ("blk.0.attn_q.weight", 400),
            ("blk.0.ffn_up_exps.weight", 3200),
            ("output.weight", 1000),
        ];
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend((tensors.len() as u64).to_le_bytes());
        b.extend(3u64.to_le_bytes()); // kv count
        s(&mut b, "general.architecture");
        b.extend(8u32.to_le_bytes());
        s(&mut b, "moe");
        s(&mut b, "moe.expert_count");
        b.extend(4u32.to_le_bytes());
        b.extend(8u32.to_le_bytes());
        s(&mut b, "moe.expert_used_count");
        b.extend(4u32.to_le_bytes());
        b.extend(2u32.to_le_bytes());
        let mut off = 0u64;
        for (name, size) in tensors {
            s(&mut b, name);
            b.extend(1u32.to_le_bytes()); // 1 dimension
            b.extend(size.to_le_bytes());
            b.extend(0u32.to_le_bytes()); // f32
            b.extend(off.to_le_bytes());
            off += size;
        }
        let header_len = b.len() as u64;
        let md = parse(&mut b.as_slice()).unwrap();
        assert_eq!(md.tensors.len(), 4);
        assert_eq!(md.data_offset, header_len.div_ceil(32) * 32);
        let file_len = md.data_offset + off;
        let sizes = md.tensor_sizes(file_len).unwrap();
        assert_eq!(sizes.iter().map(|(_, s)| *s).sum::<u64>(), 5600);
        // 400 (attention) + 3200 * 2/8 (experts) + 1000 (output); embedding skipped.
        assert_eq!(md.bytes_per_token(file_len), Some(400 + 800 + 1000));
        assert_eq!(md.expert_bytes(file_len), Some((3200, 800)));
        // Without a separate output head the (tied) embedding table is read in full.
        let mut tied = md;
        tied.tensors.retain(|(n, _)| n != "output.weight");
        assert_eq!(
            tied.bytes_per_token(file_len - 1000),
            Some(1000 + 400 + 800)
        );
        // Metadata-only buffers still parse, with an empty index.
        let md = parse(&mut sample_gguf().as_slice()).unwrap();
        assert!(md.tensors.is_empty() && md.bytes_per_token(1 << 20).is_none());
    }
}
