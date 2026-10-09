//! Accuracy of the int8 / SIMD kernels on real Q4_K and Q6_K rows from a GGUF file.
//!
//! Gated on `LLMARIO_TEST_GGUF=<path>` (for example the Qwen3-1.7B Q4_K_M file). The reader here
//! is the minimum needed to find a tensor's bytes (GGUF v2/v3 header, KV skipping, tensor
//! infos, alignment); the engine's real reader lives in the `formats` crate.
//!
//! Run with `LLMARIO_TEST_GGUF=... cargo test -p llmario-engine-cpu --release -- --nocapture gguf`.

use super::tests::exact_quantised_reference;
use super::*;
use crate::{dequant_row, QMat, ThreadPool};
use llmario_engine_core::GgmlType;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

struct TensorInfo {
    name: String,
    dims: Vec<u64>,
    dtype: GgmlType,
    offset: u64,
}

struct Gguf {
    file: File,
    tensors: Vec<TensorInfo>,
    data_start: u64,
}

fn rd<const N: usize>(f: &mut File) -> [u8; N] {
    let mut b = [0u8; N];
    f.read_exact(&mut b).unwrap();
    b
}
fn u32_(f: &mut File) -> u32 {
    u32::from_le_bytes(rd(f))
}
fn u64_(f: &mut File) -> u64 {
    u64::from_le_bytes(rd(f))
}
fn string(f: &mut File) -> String {
    let n = u64_(f) as usize;
    let mut b = vec![0u8; n];
    f.read_exact(&mut b).unwrap();
    String::from_utf8_lossy(&b).into_owned()
}

/// Skip (or, for `general.alignment`, return) one KV value of GGUF type `ty`.
fn skip_value(f: &mut File, ty: u32) -> Option<u64> {
    match ty {
        0 | 1 | 7 => {
            f.seek(SeekFrom::Current(1)).unwrap();
        }
        2 | 3 => {
            f.seek(SeekFrom::Current(2)).unwrap();
        }
        4 | 5 => return Some(u32_(f) as u64),
        6 => {
            f.seek(SeekFrom::Current(4)).unwrap();
        }
        8 => {
            let n = u64_(f);
            f.seek(SeekFrom::Current(n as i64)).unwrap();
        }
        9 => {
            let et = u32_(f);
            let n = u64_(f);
            for _ in 0..n {
                skip_value(f, et);
            }
        }
        10..=12 => {
            f.seek(SeekFrom::Current(8)).unwrap();
        }
        other => panic!("unknown GGUF value type {other}"),
    }
    None
}

fn open(path: &str) -> Gguf {
    let mut f = File::open(path).unwrap();
    assert_eq!(&rd::<4>(&mut f), b"GGUF");
    let version = u32_(&mut f);
    assert!(version == 2 || version == 3, "GGUF version {version}");
    let n_tensors = u64_(&mut f);
    let n_kv = u64_(&mut f);
    let mut alignment = 32u64;
    for _ in 0..n_kv {
        let key = string(&mut f);
        let ty = u32_(&mut f);
        if let Some(v) = skip_value(&mut f, ty) {
            if key == "general.alignment" {
                alignment = v;
            }
        }
    }
    let mut tensors = Vec::new();
    for _ in 0..n_tensors {
        let name = string(&mut f);
        let nd = u32_(&mut f);
        let dims: Vec<u64> = (0..nd).map(|_| u64_(&mut f)).collect();
        let dtype = GgmlType::from_id(u32_(&mut f)).expect("known ggml type");
        let offset = u64_(&mut f);
        tensors.push(TensorInfo {
            name,
            dims,
            dtype,
            offset,
        });
    }
    let pos = f.stream_position().unwrap();
    let data_start = pos.div_ceil(alignment) * alignment;
    Gguf {
        file: f,
        tensors,
        data_start,
    }
}

impl Gguf {
    /// The first `max_rows` rows of the 2-D tensor `name` (bytes, dtype, cols).
    fn rows(&mut self, name: &str, max_rows: usize) -> (Vec<u8>, GgmlType, usize, usize) {
        let t = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("tensor {name} not in file"));
        assert_eq!(t.dims.len(), 2, "{name} is not a matrix");
        let cols = t.dims[0] as usize;
        let rows = (t.dims[1] as usize).min(max_rows);
        let rb = t.dtype.row_bytes(cols);
        let mut data = vec![0u8; rows * rb];
        let (dtype, off) = (t.dtype, t.offset);
        self.file
            .seek(SeekFrom::Start(self.data_start + off))
            .unwrap();
        self.file.read_exact(&mut data).unwrap();
        (data, dtype, rows, cols)
    }
}

/// xorshift activations: uniform in [-1, 1), optionally with a sparse set of 20× outliers (real
/// hidden states have a few large channels, which is what makes per-block int8 lossy).
fn activation(cols: usize, seed: u64, outliers: bool) -> Vec<f32> {
    let mut s = seed;
    (0..cols)
        .map(|i| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let u =
                (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 23) as f32 - 1.0;
            if outliers && i % 97 == 3 {
                u * 20.0
            } else {
                u
            }
        })
        .collect()
}

#[test]
fn real_gguf_rows_match_the_f32_reference_within_activation_quantisation() {
    let Ok(path) = std::env::var("LLMARIO_TEST_GGUF") else {
        return;
    };
    let mut g = open(&path);
    let pool = ThreadPool::new(4);
    let names = [
        "blk.0.attn_q.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.attn_v.weight",
        "blk.0.ffn_down.weight",
        "blk.27.ffn_down.weight",
        "output.weight",
        "token_embd.weight",
    ];
    println!("\nreal rows from {path}");
    for name in names {
        if !g.tensors.iter().any(|t| t.name == name) {
            println!("  {name}: not present, skipped");
            continue;
        }
        let (data, dtype, rows, cols) = g.rows(name, 1024);
        let w = QMat::new(dtype, rows, cols, &data);
        for outliers in [false, true] {
            let x = activation(cols, 0x1234_5678 + outliers as u64, outliers);
            let mut reference = vec![0f32; rows];
            SCALAR.matvec(&pool, &w, &x, &mut reference);
            let (exact, exact_scale) = exact_quantised_reference(&w, &x, 1);
            // Predicted error from activation quantisation alone: each Q8_K block rounds its
            // elements to a grid of step d8 = max|x|/127, so the per-row dot error has standard
            // deviation d8/√12 · ‖w‖₂ (summed over blocks).
            let mut wrow = vec![0f32; cols];
            let mut l2 = vec![0f32; rows];
            let mut predicted = vec![0f64; rows];
            let steps: Vec<f64> = x
                .chunks(if dtype.block_elems() == 256 { 256 } else { 32 })
                .map(|blk| {
                    let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs())) as f64;
                    amax / 127.0
                })
                .collect();
            let blk = if dtype.block_elems() == 256 { 256 } else { 32 };
            for r in 0..rows {
                dequant_row(&w, r, &mut wrow);
                l2[r] = wrow
                    .iter()
                    .zip(&x)
                    .map(|(a, b)| (a * b) * (a * b))
                    .sum::<f32>()
                    .sqrt();
                let mut var = 0f64;
                for (b, wb) in wrow.chunks(blk).enumerate() {
                    let ww: f64 = wb.iter().map(|v| (*v as f64) * (*v as f64)).sum();
                    var += ww * steps[b] * steps[b] / 12.0;
                }
                predicted[r] = var.sqrt();
            }
            for set in available().into_iter().skip(1) {
                let mut out = vec![0f32; rows];
                set.matvec(&pool, &w, &x, &mut out);
                let mut max_rel_l2 = 0f32; // |err| / ‖w∘x‖₂
                let mut sum_sq_rel = 0f64;
                let mut sum_sq_err = 0f64;
                let mut sum_sq_pred = 0f64;
                let mut max_oracle = 0f64; // |out − W·Q(x)| / Σ|w·Q(x)|
                for r in 0..rows {
                    let err = (out[r] - reference[r]).abs();
                    max_rel_l2 = max_rel_l2.max(err / l2[r]);
                    sum_sq_rel += (err / l2[r]) as f64 * (err / l2[r]) as f64;
                    sum_sq_err += err as f64 * err as f64;
                    sum_sq_pred += predicted[r] * predicted[r];
                    max_oracle =
                        max_oracle.max((out[r] as f64 - exact[r]).abs() / (exact_scale[r] + 1e-30));
                }
                let rms_rel = (sum_sq_rel / rows as f64).sqrt();
                let measured_over_predicted = (sum_sq_err / sum_sq_pred).sqrt();
                println!(
                    "  {name:<24} {dtype:<5} {rows}x{cols} {:<8} {:<5}: rms|err|/‖w∘x‖₂ = {:.2e}, max = {:.2e}, measured/predicted quantisation error = {:.2}, vs W·Q(x): {:.1e}",
                    if outliers { "outliers" } else { "uniform" },
                    set.name,
                    rms_rel,
                    max_rel_l2,
                    measured_over_predicted,
                    max_oracle
                );
                assert!(
                    max_oracle < 2e-5,
                    "{name} {}: kernel is not W·Q(x)",
                    set.name
                );
                assert!(
                    (outliers || rms_rel < 1e-2) && (max_rel_l2 as f64) < 5.0 * rms_rel.max(1e-3),
                    "{name} {}: error beyond activation quantisation",
                    set.name
                );
                assert!(
                    (0.7..1.3).contains(&measured_over_predicted),
                    "{name} {}: error does not match the quantisation model",
                    set.name
                );
            }
        }
    }
}
