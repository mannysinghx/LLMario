//! Compares the engine's reference dequantizers with gguf-py on real model files.
//!
//! Runs only when `LLMARIO_DEQUANT_REF` names one or more JSON files (comma-separated) produced
//! by `scripts/engine/dequant_ref.py`; each names its GGUF and holds the first and last `n`
//! dequantised values of up to two tensors per type.

use llmario_engine_core::dequant::{dequant_supported, dequantize_row};
use llmario_engine_formats::GgufFile;
use std::path::Path;

#[test]
fn dequant_matches_gguf_py() {
    let Ok(refs) = std::env::var("LLMARIO_DEQUANT_REF") else {
        eprintln!("LLMARIO_DEQUANT_REF not set; skipping");
        return;
    };
    for ref_path in refs.split(',').filter(|s| !s.is_empty()) {
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(ref_path).unwrap()).unwrap();
        let gguf_path = json["gguf"].as_str().unwrap();
        let n = json["n"].as_u64().unwrap() as usize;
        let f = GgufFile::open(Path::new(gguf_path)).unwrap();
        let mut checked = 0;
        for t in json["tensors"].as_array().unwrap() {
            let name = t["name"].as_str().unwrap();
            let info = f.tensor(name).unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(
                info.dtype.name().to_ascii_uppercase(),
                t["type"].as_str().unwrap().to_ascii_uppercase(),
                "{name} type"
            );
            if !dequant_supported(info.dtype) {
                continue;
            }
            let numel = info.shape.numel() as usize;
            assert_eq!(numel, t["numel"].as_u64().unwrap() as usize, "{name} numel");
            let ne0 = info.shape.row_len() as usize;
            let rows = info.shape.rows() as usize;
            let bytes = f.tensor_bytes(info);
            let row_bytes = info.row_bytes() as usize;
            let rows_needed = n.div_ceil(ne0).min(rows);
            let head_ref: Vec<f32> = t["head"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let tail_ref: Vec<f32> = t["tail"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let mut head = vec![0f32; rows_needed * ne0];
            dequantize_row(info.dtype, &bytes[..rows_needed * row_bytes], &mut head).unwrap();
            let mut tail = vec![0f32; rows_needed * ne0];
            dequantize_row(
                info.dtype,
                &bytes[(rows - rows_needed) * row_bytes..],
                &mut tail,
            )
            .unwrap();
            let tail_start = numel - tail_ref.len();
            let tail_local = tail_start - (rows - rows_needed) * ne0;
            for (i, r) in head_ref.iter().enumerate() {
                let got = head[i];
                assert!(
                    (got - r).abs() <= 1e-6 * r.abs().max(1.0),
                    "{name}[{i}] head: got {got}, gguf-py {r}"
                );
            }
            for (i, r) in tail_ref.iter().enumerate() {
                let got = tail[tail_local + i];
                assert!(
                    (got - r).abs() <= 1e-6 * r.abs().max(1.0),
                    "{name} tail[{i}]: got {got}, gguf-py {r}"
                );
            }
            checked += 1;
            eprintln!("ok {name} {} {}", info.dtype, info.shape);
        }
        assert!(checked > 0, "nothing checked in {ref_path}");
    }
}
