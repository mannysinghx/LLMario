//! GGUF v3 reader (spec: ggml/docs/gguf.md, MIT).
//!
//! Layout: `GGUF` magic, u32 version, u64 tensor count, u64 metadata count, metadata key/value
//! pairs, tensor infos (name, dims, type, offset), padding to `general.alignment` (default 32),
//! then tensor data. Offsets in tensor infos are relative to the data start. Little-endian only.

use llmario_engine_core::tensor::ByteSpan;
use llmario_engine_core::{EngineError, GgmlType, Result, Shape, TensorInfo};
use memmap2::Mmap;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;
/// Hard cap on a single string or array so a corrupt header cannot make us allocate the world.
const MAX_STRING: u64 = 1 << 28;
const MAX_ARRAY: u64 = 1 << 26;

#[derive(Clone, Debug, PartialEq)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Array(Vec<MetaValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl MetaValue {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match self {
            MetaValue::U8(v) => *v as u64,
            MetaValue::U16(v) => *v as u64,
            MetaValue::U32(v) => *v as u64,
            MetaValue::U64(v) => *v,
            MetaValue::I8(v) if *v >= 0 => *v as u64,
            MetaValue::I16(v) if *v >= 0 => *v as u64,
            MetaValue::I32(v) if *v >= 0 => *v as u64,
            MetaValue::I64(v) if *v >= 0 => *v as u64,
            _ => return None,
        })
    }
    pub fn as_i64(&self) -> Option<i64> {
        Some(match self {
            MetaValue::U8(v) => *v as i64,
            MetaValue::U16(v) => *v as i64,
            MetaValue::U32(v) => *v as i64,
            MetaValue::U64(v) => i64::try_from(*v).ok()?,
            MetaValue::I8(v) => *v as i64,
            MetaValue::I16(v) => *v as i64,
            MetaValue::I32(v) => *v as i64,
            MetaValue::I64(v) => *v,
            _ => return None,
        })
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            MetaValue::F32(v) => Some(*v as f64),
            MetaValue::F64(v) => Some(*v),
            other => other.as_i64().map(|v| v as f64),
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            MetaValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[MetaValue]> {
        match self {
            MetaValue::Array(a) => Some(a),
            _ => None,
        }
    }
    /// Short description for logs (long arrays are summarised).
    pub fn summary(&self) -> String {
        match self {
            MetaValue::Str(s) if s.len() > 60 => format!("{:?}… ({} bytes)", &s[..60], s.len()),
            MetaValue::Str(s) => format!("{s:?}"),
            MetaValue::Array(a) => format!("[{} items]", a.len()),
            MetaValue::F32(v) => format!("{v}"),
            MetaValue::F64(v) => format!("{v}"),
            MetaValue::Bool(v) => format!("{v}"),
            other => other
                .as_i64()
                .map(|v| v.to_string())
                .unwrap_or_else(|| format!("{other:?}")),
        }
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| EngineError::Format("truncated GGUF header".into()))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String> {
        let len = self.u64()?;
        if len > MAX_STRING {
            return Err(EngineError::Format(format!("string of {len} bytes")));
        }
        let bytes = self.take(len as usize)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }
    fn value(&mut self, ty: u32) -> Result<MetaValue> {
        Ok(match ty {
            0 => MetaValue::U8(self.u8()?),
            1 => MetaValue::I8(self.u8()? as i8),
            2 => MetaValue::U16(self.u16()?),
            3 => MetaValue::I16(self.u16()? as i16),
            4 => MetaValue::U32(self.u32()?),
            5 => MetaValue::I32(self.u32()? as i32),
            6 => MetaValue::F32(self.f32()?),
            7 => MetaValue::Bool(self.u8()? != 0),
            8 => MetaValue::Str(self.string()?),
            9 => {
                let ety = self.u32()?;
                let len = self.u64()?;
                if len > MAX_ARRAY {
                    return Err(EngineError::Format(format!("array of {len} items")));
                }
                let mut v = Vec::with_capacity(len.min(1 << 20) as usize);
                for _ in 0..len {
                    v.push(self.value(ety)?);
                }
                MetaValue::Array(v)
            }
            10 => MetaValue::U64(self.u64()?),
            11 => MetaValue::I64(self.u64()? as i64),
            12 => MetaValue::F64(self.f64()?),
            other => {
                return Err(EngineError::Format(format!(
                    "unknown metadata value type {other}"
                )))
            }
        })
    }
}

/// One mapped part of a GGUF model (a split set has several).
pub struct Part {
    pub path: PathBuf,
    pub mmap: Mmap,
    pub data_offset: u64,
}

/// A parsed GGUF model: metadata from the first part, tensors from every part.
pub struct GgufFile {
    pub version: u32,
    pub alignment: u64,
    pub metadata: BTreeMap<String, MetaValue>,
    pub tensors: Vec<TensorInfo>,
    pub parts: Vec<Part>,
    by_name: BTreeMap<String, usize>,
}

impl GgufFile {
    /// Open a GGUF file. If it is the first part of a split set (`split.count` > 1 and the file
    /// name ends in `-00001-of-NNNNN.gguf`), the sibling parts are opened too.
    pub fn open(path: &Path) -> Result<GgufFile> {
        let first = open_part(path)?;
        let count = first
            .metadata
            .get("split.count")
            .and_then(|v| v.as_u64())
            .unwrap_or(1);
        let mut parsed = vec![first];
        if count > 1 {
            for part_no in 2..=count {
                let sibling = split_sibling(path, part_no, count).ok_or_else(|| {
                    EngineError::Format(format!(
                        "{} declares {count} split parts but is not named -00001-of-{count:05}.gguf",
                        path.display()
                    ))
                })?;
                parsed.push(open_part(&sibling)?);
            }
        }
        let mut tensors = Vec::new();
        let mut parts = Vec::new();
        let mut by_name = BTreeMap::new();
        let mut version = 0;
        let mut alignment = DEFAULT_ALIGNMENT;
        let mut metadata = BTreeMap::new();
        for (i, p) in parsed.into_iter().enumerate() {
            if i == 0 {
                version = p.version;
                alignment = p.alignment;
                metadata = p.metadata;
            }
            for mut t in p.tensors {
                t.span.source = i as u32;
                if by_name.insert(t.name.clone(), tensors.len()).is_some() {
                    return Err(EngineError::Format(format!("duplicate tensor {}", t.name)));
                }
                tensors.push(t);
            }
            parts.push(p.part);
        }
        Ok(GgufFile {
            version,
            alignment,
            metadata,
            tensors,
            parts,
            by_name,
        })
    }

    pub fn architecture(&self) -> Option<&str> {
        self.get_str("general.architecture")
    }

    pub fn get(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(|v| v.as_str())
    }
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key).and_then(|v| v.as_u64())
    }
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.get_u64(key).and_then(|v| u32::try_from(v).ok())
    }
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.metadata
            .get(key)
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
    }
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.metadata.get(key).and_then(|v| v.as_bool())
    }
    pub fn get_array(&self, key: &str) -> Option<&[MetaValue]> {
        self.metadata.get(key).and_then(|v| v.as_array())
    }
    /// Architecture-prefixed key, e.g. `arch_key("block_count")` → `llama.block_count`.
    pub fn arch_key(&self, suffix: &str) -> String {
        format!("{}.{}", self.architecture().unwrap_or(""), suffix)
    }
    pub fn get_arch_u32(&self, suffix: &str) -> Option<u32> {
        self.get_u32(&self.arch_key(suffix))
    }
    pub fn get_arch_f32(&self, suffix: &str) -> Option<f32> {
        self.get_f32(&self.arch_key(suffix))
    }
    pub fn get_arch_str(&self, suffix: &str) -> Option<&str> {
        self.get_str(&self.arch_key(suffix))
    }
    pub fn get_arch_array(&self, suffix: &str) -> Option<&[MetaValue]> {
        self.get_array(&self.arch_key(suffix))
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.by_name.get(name).map(|&i| &self.tensors[i])
    }

    /// The bytes of a tensor (zero-copy slice of the mapping).
    pub fn tensor_bytes(&self, t: &TensorInfo) -> &[u8] {
        let part = &self.parts[t.span.source as usize];
        let start = (part.data_offset + t.span.offset) as usize;
        &part.mmap[start..start + t.span.len as usize]
    }

    /// Total tensor bytes (what "weights" means in the plan).
    pub fn tensor_bytes_total(&self) -> u64 {
        self.tensors.iter().map(|t| t.span.len).sum()
    }

    /// Bytes per type, for the plan's quantisation summary.
    pub fn bytes_by_type(&self) -> BTreeMap<GgmlType, u64> {
        let mut m = BTreeMap::new();
        for t in &self.tensors {
            *m.entry(t.dtype).or_insert(0) += t.span.len;
        }
        m
    }
}

struct ParsedPart {
    version: u32,
    alignment: u64,
    metadata: BTreeMap<String, MetaValue>,
    tensors: Vec<TensorInfo>,
    part: Part,
}

fn open_part(path: &Path) -> Result<ParsedPart> {
    let file = File::open(path)
        .map_err(|e| EngineError::Format(format!("cannot open {}: {e}", path.display())))?;
    // SAFETY: read-only private mapping of a regular file; the file may change underneath us
    // (any mmap has that property), which can only produce garbage tensors, never UB in safe
    // code paths that bounds-check offsets against the mapping length.
    let mmap = unsafe { Mmap::map(&file)? };
    let parsed = parse_header(&mmap[..])?;
    let part = Part {
        path: path.to_path_buf(),
        data_offset: parsed.data_offset,
        mmap,
    };
    // Validate every tensor against the part size.
    let total = part.mmap.len() as u64;
    for t in &parsed.tensors {
        let end = part
            .data_offset
            .checked_add(t.span.offset)
            .and_then(|s| s.checked_add(t.span.len));
        match end {
            Some(e) if e <= total => {}
            _ => {
                return Err(EngineError::Format(format!(
                    "tensor {} extends past the end of {} ({} bytes)",
                    t.name,
                    path.display(),
                    total
                )))
            }
        }
    }
    Ok(ParsedPart {
        version: parsed.version,
        alignment: parsed.alignment,
        metadata: parsed.metadata,
        tensors: parsed.tensors,
        part,
    })
}

struct ParsedHeader {
    version: u32,
    alignment: u64,
    metadata: BTreeMap<String, MetaValue>,
    tensors: Vec<TensorInfo>,
    data_offset: u64,
}

/// Parse a GGUF header from a byte buffer (tensor data need not be present; offsets are not
/// validated here).
fn parse_header(buf: &[u8]) -> Result<ParsedHeader> {
    let mut r = Reader { buf, pos: 0 };
    if r.take(4)? != MAGIC {
        return Err(EngineError::Format("not a GGUF file (bad magic)".into()));
    }
    let version = r.u32()?;
    if !(2..=3).contains(&version) {
        return Err(EngineError::Format(format!(
            "GGUF version {version} is not supported (need 2 or 3)"
        )));
    }
    let n_tensors = r.u64()?;
    let n_kv = r.u64()?;
    if n_tensors > 1 << 20 || n_kv > 1 << 20 {
        return Err(EngineError::Format("implausible GGUF header counts".into()));
    }
    let mut metadata = BTreeMap::new();
    for _ in 0..n_kv {
        let key = r.string()?;
        let ty = r.u32()?;
        let val = r.value(ty)?;
        metadata.insert(key, val);
    }
    let alignment = metadata
        .get("general.alignment")
        .and_then(|v| v.as_u64())
        .unwrap_or(DEFAULT_ALIGNMENT);
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(EngineError::Format(format!(
            "general.alignment {alignment} is not a power of two"
        )));
    }
    let mut tensors = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = r.string()?;
        let n_dims = r.u32()?;
        if n_dims == 0 || n_dims > 4 {
            return Err(EngineError::Format(format!("tensor {name}: {n_dims} dims")));
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(r.u64()?);
        }
        let ty = r.u32()?;
        let offset = r.u64()?;
        let dtype = GgmlType::from_id(ty).ok_or(EngineError::UnknownType(ty))?;
        let shape = Shape::new(&dims)?;
        let len = TensorInfo::expected_bytes(dtype, &shape)
            .map_err(|e| EngineError::Format(format!("tensor {name}: {e}")))?;
        if offset % alignment != 0 {
            return Err(EngineError::Format(format!(
                "tensor {name}: offset {offset} not aligned to {alignment}"
            )));
        }
        tensors.push(TensorInfo {
            name,
            dtype,
            shape,
            span: ByteSpan {
                source: 0,
                offset,
                len,
            },
        });
    }
    let data_offset = (r.pos as u64).div_ceil(alignment) * alignment;
    Ok(ParsedHeader {
        version,
        alignment,
        metadata,
        tensors,
        data_offset,
    })
}

/// `model-00001-of-00003.gguf` + part 2 → `model-00002-of-00003.gguf`.
fn split_sibling(first: &Path, part_no: u64, count: u64) -> Option<PathBuf> {
    let name = first.file_name()?.to_str()?;
    let suffix = format!("-00001-of-{count:05}.gguf");
    let stem = name.strip_suffix(&suffix)?;
    Some(first.with_file_name(format!("{stem}-{part_no:05}-of-{count:05}.gguf")))
}

/// Minimal GGUF writer used by tests and by `testkit` to build synthetic models.
pub mod writer {
    use super::MetaValue;
    use llmario_engine_core::GgmlType;
    use std::io::Write;

    pub struct GgufWriter {
        pub alignment: u64,
        meta: Vec<(String, MetaValue)>,
        tensors: Vec<(String, Vec<u64>, GgmlType, Vec<u8>)>,
    }

    impl Default for GgufWriter {
        fn default() -> Self {
            Self::new()
        }
    }

    impl GgufWriter {
        pub fn new() -> Self {
            GgufWriter {
                alignment: 32,
                meta: vec![],
                tensors: vec![],
            }
        }
        pub fn meta(&mut self, key: &str, v: MetaValue) -> &mut Self {
            self.meta.push((key.to_string(), v));
            self
        }
        pub fn tensor(
            &mut self,
            name: &str,
            dims: &[u64],
            ty: GgmlType,
            data: Vec<u8>,
        ) -> &mut Self {
            self.tensors
                .push((name.to_string(), dims.to_vec(), ty, data));
            self
        }
        fn write_string(out: &mut Vec<u8>, s: &str) {
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        fn type_id(v: &MetaValue) -> u32 {
            match v {
                MetaValue::U8(_) => 0,
                MetaValue::I8(_) => 1,
                MetaValue::U16(_) => 2,
                MetaValue::I16(_) => 3,
                MetaValue::U32(_) => 4,
                MetaValue::I32(_) => 5,
                MetaValue::F32(_) => 6,
                MetaValue::Bool(_) => 7,
                MetaValue::Str(_) => 8,
                MetaValue::Array(_) => 9,
                MetaValue::U64(_) => 10,
                MetaValue::I64(_) => 11,
                MetaValue::F64(_) => 12,
            }
        }
        fn write_value(out: &mut Vec<u8>, v: &MetaValue) {
            match v {
                MetaValue::U8(x) => out.push(*x),
                MetaValue::I8(x) => out.push(*x as u8),
                MetaValue::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
                MetaValue::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
                MetaValue::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
                MetaValue::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
                MetaValue::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
                MetaValue::Bool(x) => out.push(*x as u8),
                MetaValue::Str(s) => Self::write_string(out, s),
                MetaValue::Array(a) => {
                    let ety = a.first().map(Self::type_id).unwrap_or(4);
                    out.extend_from_slice(&ety.to_le_bytes());
                    out.extend_from_slice(&(a.len() as u64).to_le_bytes());
                    for e in a {
                        Self::write_value(out, e);
                    }
                }
                MetaValue::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
                MetaValue::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
                MetaValue::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
            }
        }
        pub fn to_bytes(&self) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend_from_slice(b"GGUF");
            out.extend_from_slice(&3u32.to_le_bytes());
            out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
            let has_align = self.meta.iter().any(|(k, _)| k == "general.alignment");
            let n_kv = self.meta.len() as u64 + if has_align { 0 } else { 1 };
            out.extend_from_slice(&n_kv.to_le_bytes());
            if !has_align {
                Self::write_string(&mut out, "general.alignment");
                out.extend_from_slice(&4u32.to_le_bytes());
                out.extend_from_slice(&(self.alignment as u32).to_le_bytes());
            }
            for (k, v) in &self.meta {
                Self::write_string(&mut out, k);
                out.extend_from_slice(&Self::type_id(v).to_le_bytes());
                Self::write_value(&mut out, v);
            }
            let mut offset = 0u64;
            let mut blobs = Vec::new();
            for (name, dims, ty, data) in &self.tensors {
                Self::write_string(&mut out, name);
                out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
                for d in dims {
                    out.extend_from_slice(&d.to_le_bytes());
                }
                out.extend_from_slice(&ty.id().to_le_bytes());
                out.extend_from_slice(&offset.to_le_bytes());
                blobs.push(data);
                offset += (data.len() as u64).div_ceil(self.alignment) * self.alignment;
            }
            let pad =
                (out.len() as u64).div_ceil(self.alignment) * self.alignment - out.len() as u64;
            out.extend(std::iter::repeat(0u8).take(pad as usize));
            for data in blobs {
                out.write_all(data).unwrap();
                let pad = (data.len() as u64).div_ceil(self.alignment) * self.alignment
                    - data.len() as u64;
                out.extend(std::iter::repeat(0u8).take(pad as usize));
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::writer::GgufWriter;
    use super::*;

    fn sample() -> Vec<u8> {
        let mut w = GgufWriter::new();
        w.meta("general.architecture", MetaValue::Str("llama".into()))
            .meta("llama.block_count", MetaValue::U32(2))
            .meta("llama.rope.freq_base", MetaValue::F32(10000.0))
            .meta(
                "tokenizer.ggml.tokens",
                MetaValue::Array(vec![
                    MetaValue::Str("<s>".into()),
                    MetaValue::Str("a".into()),
                ]),
            )
            .tensor("tok", &[32, 2], GgmlType::Q4_0, vec![1u8; 2 * 18])
            .tensor("norm", &[4], GgmlType::F32, vec![0u8; 16]);
        w.to_bytes()
    }

    #[test]
    fn parses_header_and_tensors() {
        let bytes = sample();
        let h = parse_header(&bytes).unwrap();
        assert_eq!(h.version, 3);
        assert_eq!(h.alignment, 32);
        assert_eq!(h.metadata["general.architecture"].as_str(), Some("llama"));
        assert_eq!(h.metadata["llama.block_count"].as_u64(), Some(2));
        assert_eq!(h.tensors.len(), 2);
        assert_eq!(h.tensors[0].dtype, GgmlType::Q4_0);
        assert_eq!(h.tensors[0].span.len, 36);
        assert_eq!(h.tensors[1].span.offset, 64);
        assert_eq!(h.data_offset % 32, 0);
    }

    #[test]
    fn opens_from_disk_and_reads_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("m.gguf");
        std::fs::write(&p, sample()).unwrap();
        let f = GgufFile::open(&p).unwrap();
        assert_eq!(f.architecture(), Some("llama"));
        assert_eq!(f.get_arch_u32("block_count"), Some(2));
        assert_eq!(f.get_arch_f32("rope.freq_base"), Some(10000.0));
        let t = f.tensor("tok").unwrap();
        assert_eq!(f.tensor_bytes(t), &[1u8; 36]);
        assert_eq!(f.tensor_bytes_total(), 36 + 16);
        assert_eq!(f.get_array("tokenizer.ggml.tokens").unwrap().len(), 2);
    }

    #[test]
    fn rejects_truncated_and_bad_magic() {
        let bytes = sample();
        assert!(parse_header(&bytes[..40]).is_err());
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert!(parse_header(&bad).is_err());
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("t.gguf");
        std::fs::write(&p, &bytes[..bytes.len() - 24]).unwrap();
        assert!(GgufFile::open(&p).is_err(), "tensor past EOF must fail");
    }

    #[test]
    fn split_names() {
        let p = Path::new("/x/Big-00001-of-00003.gguf");
        assert_eq!(
            split_sibling(p, 2, 3).unwrap(),
            PathBuf::from("/x/Big-00002-of-00003.gguf")
        );
        assert!(split_sibling(Path::new("/x/Big.gguf"), 2, 3).is_none());
    }
}
