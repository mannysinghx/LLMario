//! Reads real MLX safetensors folders and compares the engine's MLX affine dequantiser with the
//! numpy/mlx reference produced by `scripts/engine/mlx_ref.py`.
//!
//! - `LLMARIO_TEST_MLX=<folder>[,<folder>...]`: open each folder, check the tensor table against
//!   `model.safetensors.index.json`, resolve every quantised module and dequantise one row of each
//!   (header parsing and consistency only; no reference values needed).
//! - `LLMARIO_TEST_MLX_REF=<json>[,<json>...]`: for each reference JSON (which names its folder),
//!   dequantise the same rows and compare value by value.
//!
//! Both are skipped when the variable is unset so `cargo test` stays hermetic.

use llmario_engine_formats::SafetensorsFolder;
use std::path::Path;

fn folders(var: &str) -> Vec<String> {
    std::env::var(var)
        .map(|v| {
            v.split(',')
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

#[test]
fn mlx_folders_parse_and_resolve() {
    let dirs = folders("LLMARIO_TEST_MLX");
    if dirs.is_empty() {
        eprintln!("LLMARIO_TEST_MLX not set; skipping");
        return;
    }
    for dir in dirs {
        let dir = Path::new(&dir);
        let t0 = std::time::Instant::now();
        let f = SafetensorsFolder::open(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        let open_ms = t0.elapsed().as_millis();
        assert!(f.is_mlx(), "{}: not detected as MLX", dir.display());
        let q = f.quantization().expect("quantization block");
        assert!(f.hidden_size().is_some() && f.num_hidden_layers().is_some());
        assert!(f.vocab_size().is_some());
        assert!(f.tokenizer_json_path().is_some(), "tokenizer.json");
        assert!(f.chat_template().is_some(), "chat template");

        // The index file agrees with what we mapped.
        let idx = dir.join("model.safetensors.index.json");
        if idx.is_file() {
            let idx: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(idx).unwrap()).unwrap();
            let wm = idx["weight_map"].as_object().unwrap();
            assert_eq!(f.tensors.len(), wm.len(), "tensor count vs index");
            assert_eq!(
                f.index_total_size.unwrap(),
                idx["metadata"]["total_size"].as_u64().unwrap()
            );
            assert_eq!(
                f.tensor_bytes_total(),
                f.index_total_size.unwrap(),
                "tensor bytes vs index total_size"
            );
        }
        for s in &f.shards {
            assert_eq!(
                s.metadata.get("format").map(String::as_str),
                Some("mlx"),
                "{}",
                s.path.display()
            );
        }

        // Every quantised module resolves, has the configured geometry, and dequantises.
        let views = f.mlx_quant_views().unwrap();
        assert!(!views.is_empty());
        let mut packed_bytes = 0u64;
        for v in &views {
            let p = q.params_for(&v.module).unwrap();
            assert_eq!(
                (v.bits, v.group_size),
                (p.bits, p.group_size),
                "{}",
                v.module
            );
            assert_eq!(v.cols % v.group_size as u64, 0);
            packed_bytes += v.weight.span.len + v.scales.span.len + v.biases.span.len;
            let mut row = vec![0f32; v.cols as usize];
            f.dequantize_mlx_rows(v, 0, 1, &mut row).unwrap();
            assert!(
                row.iter().all(|x| x.is_finite()),
                "{}: non-finite",
                v.module
            );
            let mut last = vec![0f32; v.cols as usize];
            f.dequantize_mlx_rows(v, v.rows as usize - 1, 1, &mut last)
                .unwrap();
        }
        // Each .weight that has a .scales sibling is a view; nothing is left dangling.
        let scales = f
            .tensors
            .iter()
            .filter(|t| t.name.ends_with(".scales"))
            .count();
        assert_eq!(scales, views.len());
        let bpw = views[0].bits_per_weight();
        eprintln!(
            "{}: {} tensors in {} shards, {} quantised modules ({} bits g{}, {:.3} bpw incl. scales/biases), {} MiB packed of {} MiB, opened in {open_ms} ms",
            dir.display(),
            f.tensors.len(),
            f.shards.len(),
            views.len(),
            q.default.bits,
            q.default.group_size,
            bpw,
            packed_bytes >> 20,
            f.tensor_bytes_total() >> 20
        );
    }
}

#[test]
fn mlx_dequant_matches_reference() {
    let refs = folders("LLMARIO_TEST_MLX_REF");
    if refs.is_empty() {
        eprintln!("LLMARIO_TEST_MLX_REF not set; skipping");
        return;
    }
    for ref_path in refs {
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&ref_path).unwrap()).unwrap();
        let dir = Path::new(json["folder"].as_str().unwrap());
        let f = SafetensorsFolder::open(dir).unwrap();
        assert_eq!(
            f.tensors.len() as u64,
            json["tensor_count"].as_u64().unwrap()
        );
        assert_eq!(
            f.tensor_bytes_total(),
            json["total_bytes"].as_u64().unwrap()
        );
        let mut compared = 0usize;
        let mut exact = 0usize;
        let mut max_diff = 0f32;
        let values = |v: &serde_json::Value| -> Vec<f32> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        };
        for t in json["tensors"].as_array().unwrap() {
            match t["kind"].as_str().unwrap() {
                "mlx" => {
                    let module = t["module"].as_str().unwrap();
                    let v = f
                        .mlx_quant_view(module)
                        .unwrap()
                        .unwrap_or_else(|| panic!("{module}: no view"));
                    assert_eq!(v.bits as u64, t["bits"].as_u64().unwrap(), "{module} bits");
                    assert_eq!(v.group_size as u64, t["group_size"].as_u64().unwrap());
                    assert_eq!(v.rows, t["rows"].as_u64().unwrap(), "{module} rows");
                    assert_eq!(v.cols, t["cols"].as_u64().unwrap(), "{module} cols");
                    assert_eq!(
                        v.scale_dtype.name().to_ascii_uppercase(),
                        t["scale_dtype"].as_str().unwrap()
                    );
                    for label in ["head", "tail"] {
                        let rows = t[format!("{label}_rows")].as_array().unwrap();
                        let (row0, n) = (
                            rows[0].as_u64().unwrap() as usize,
                            rows[1].as_u64().unwrap() as usize,
                        );
                        let mut out = vec![0f32; n * v.cols as usize];
                        f.dequantize_mlx_rows(&v, row0, n, &mut out).unwrap();
                        let want = values(&t[label]);
                        for (i, (a, b)) in out.iter().zip(&want).enumerate() {
                            let d = (a - b).abs();
                            max_diff = max_diff.max(d);
                            if d == 0.0 {
                                exact += 1;
                            }
                            assert!(
                                d <= 1e-6 * b.abs().max(1.0),
                                "{module} {label}[{i}]: engine {a} vs reference {b}"
                            );
                            compared += 1;
                        }
                        if let Some(m) = t.get(format!("{label}_mlx")) {
                            if m["available"].as_bool() == Some(true) {
                                assert_eq!(
                                    m["exact_f32"].as_bool(),
                                    Some(true),
                                    "{module}: numpy vs mlx"
                                );
                            }
                        }
                    }
                }
                "float" => {
                    let name = t["name"].as_str().unwrap();
                    let info = f.tensor(name).unwrap();
                    assert_eq!(
                        info.dtype.name().to_ascii_uppercase(),
                        t["dtype"].as_str().unwrap()
                    );
                    assert_eq!(info.shape.numel(), t["numel"].as_u64().unwrap());
                    let numel = info.shape.numel() as usize;
                    let mut all = vec![0f32; numel];
                    f.dequantize_rows(info, 0, info.shape.rows() as usize, &mut all)
                        .unwrap();
                    let head = values(&t["head"]);
                    let tail = values(&t["tail"]);
                    assert_eq!(&all[..head.len()], &head[..], "{name} head");
                    assert_eq!(&all[numel - tail.len()..], &tail[..], "{name} tail");
                    compared += head.len() + tail.len();
                    exact += head.len() + tail.len();
                }
                other => panic!("unknown kind {other}"),
            }
        }
        eprintln!(
            "{}: {compared} values compared, {exact} bit-exact, max |diff| {max_diff:e}",
            dir.display()
        );
        assert!(compared > 0);
    }
}
