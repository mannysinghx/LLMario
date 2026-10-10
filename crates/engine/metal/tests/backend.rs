//! Correctness of the Metal backend against the CPU reference.
//!
//! Every test skips (passes) when no usable Metal device is present. The real-model tests also
//! need `LLMARIO_TEST_GGUF=<path to a dense GGUF>` (the maintainers use Qwen3-1.7B-Q4_K_M).

#![cfg(target_os = "macos")]

use llmario_engine_core::dequant::dequantize_row;
use llmario_engine_core::GgmlType;
use llmario_engine_formats::gguf::writer::GgufWriter;
use llmario_engine_formats::{GgufFile, MetaValue};
use llmario_engine_metal::device::{groups, Gpu};
use llmario_engine_metal::MetalBackend;
use llmario_engine_model::{CpuBackend, ModelBackend};

fn metal() -> bool {
    if MetalBackend::is_available() {
        // The CPU reference must be the exact f32 path: the default CPU kernel set quantises
        // activations to int8 (Q8_K / Q8_0) and its logits differ from the f32 math by ~0.7 on
        // Qwen3-1.7B, far more than the Metal kernels do. `kernels()` reads this once per process.
        std::env::set_var("LLMARIO_CPU_KERNELS", "scalar");
        true
    } else {
        eprintln!("no usable Metal device; skipping");
        false
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// A tiny random llama-family model written as a GGUF (same generator as the CPU forward tests).
fn tiny_model(dir: &std::path::Path) -> std::path::PathBuf {
    tiny_model_d(dir, 32)
}

fn tiny_model_d(dir: &std::path::Path, d: u64) -> std::path::PathBuf {
    tiny_model_dh(dir, d, 8, 64)
}

/// Variant with `hd`-wide heads and a `ctx`-position context (hd 64 exercises the prefill flash
/// kernel and q8_0 rows; contexts past 32 span several KV blocks).
fn tiny_model_dh(dir: &std::path::Path, d: u64, hd: u64, ctx: u32) -> std::path::PathBuf {
    let n_head = 4u64;
    let n_kv = 2u64;
    let n_ff = 48u64;
    let vocab = 64u64;
    let n_layer = 2u64;
    let f32s = |n: u64, seed: u64| -> Vec<u8> {
        (0..n)
            .flat_map(|i| {
                let v = (((i * 2654435761 + seed * 97) % 1000) as f32 / 1000.0 - 0.5) * 0.2;
                v.to_le_bytes()
            })
            .collect()
    };
    let ones = |n: u64| -> Vec<u8> { (0..n).flat_map(|_| 1.0f32.to_le_bytes()).collect() };
    let mut w = GgufWriter::new();
    w.meta("general.architecture", MetaValue::Str("llama".into()))
        .meta("llama.block_count", MetaValue::U32(n_layer as u32))
        .meta("llama.embedding_length", MetaValue::U32(d as u32))
        .meta("llama.attention.head_count", MetaValue::U32(n_head as u32))
        .meta("llama.attention.head_count_kv", MetaValue::U32(n_kv as u32))
        .meta("llama.attention.key_length", MetaValue::U32(hd as u32))
        .meta("llama.feed_forward_length", MetaValue::U32(n_ff as u32))
        .meta("llama.vocab_size", MetaValue::U32(vocab as u32))
        .meta("llama.context_length", MetaValue::U32(ctx))
        .meta("llama.rope.freq_base", MetaValue::F32(10000.0))
        .meta(
            "llama.attention.layer_norm_rms_epsilon",
            MetaValue::F32(1e-5),
        );
    w.tensor(
        "token_embd.weight",
        &[d, vocab],
        GgmlType::F32,
        f32s(d * vocab, 1),
    );
    w.tensor("output_norm.weight", &[d], GgmlType::F32, ones(d));
    w.tensor(
        "output.weight",
        &[d, vocab],
        GgmlType::F32,
        f32s(d * vocab, 2),
    );
    for l in 0..n_layer {
        let p = |s: &str| format!("blk.{l}.{s}");
        w.tensor(&p("attn_norm.weight"), &[d], GgmlType::F32, ones(d));
        w.tensor(
            &p("attn_q.weight"),
            &[d, n_head * hd],
            GgmlType::F32,
            f32s(d * n_head * hd, 10 + l),
        );
        w.tensor(
            &p("attn_k.weight"),
            &[d, n_kv * hd],
            GgmlType::F32,
            f32s(d * n_kv * hd, 20 + l),
        );
        w.tensor(
            &p("attn_v.weight"),
            &[d, n_kv * hd],
            GgmlType::F32,
            f32s(d * n_kv * hd, 30 + l),
        );
        w.tensor(
            &p("attn_output.weight"),
            &[n_head * hd, d],
            GgmlType::F32,
            f32s(n_head * hd * d, 40 + l),
        );
        w.tensor(&p("ffn_norm.weight"), &[d], GgmlType::F32, ones(d));
        w.tensor(
            &p("ffn_gate.weight"),
            &[d, n_ff],
            GgmlType::F32,
            f32s(d * n_ff, 50 + l),
        );
        w.tensor(
            &p("ffn_up.weight"),
            &[d, n_ff],
            GgmlType::F32,
            f32s(d * n_ff, 60 + l),
        );
        w.tensor(
            &p("ffn_down.weight"),
            &[n_ff, d],
            GgmlType::F32,
            f32s(n_ff * d, 70 + l),
        );
    }
    let p = dir.join(format!("tiny-{d}-{hd}.gguf"));
    std::fs::write(&p, w.to_bytes()).unwrap();
    p
}

/// A geometry the kernels do not cover is refused with `Unsupported` (the server's `auto` device
/// turns that into a logged CPU fallback instead of a failed load).
#[test]
fn unsupported_geometry_is_refused_not_wrong() {
    if !metal() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = tiny_model_d(dir.path(), 24); // d_model not a multiple of 16
    let f = GgufFile::open(&path).unwrap();
    let err = MetalBackend::new(&f, 16, 8).err().expect("refused");
    assert!(
        matches!(err, llmario_engine_metal::MetalError::Unsupported(_)),
        "{err}"
    );
    // The CPU backend still runs it.
    let mut cpu = CpuBackend::new(&f, 1, 16, 8).unwrap();
    assert_eq!(cpu.forward(&[1, 2, 3]).len(), 64);
}

#[test]
fn kernels_compile() {
    if !metal() {
        return;
    }
    let gpu = Gpu::get().expect("kernel library compiles");
    assert!(gpu.info.has_unified_memory);
    assert!(gpu.info.recommended_max_working_set > 0);
    let info = MetalBackend::device_info().unwrap();
    assert_eq!(info.name, gpu.info.name);
}

/// Mirror of the private parameter structs the kernels take (same `#[repr(C)]` layout).
#[repr(C)]
#[derive(Clone, Copy)]
struct GemvParams {
    rows: u32,
    cols: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct GemmParams {
    rows: u32,
    cols: u32,
    n: u32,
    row_bytes: u32,
    accumulate: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct EmbedParams {
    cols: u32,
    n_tokens: u32,
    row_bytes: u32,
}

/// Deterministic pseudo-random bytes.
fn lcg_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as u8
        })
        .collect()
}

/// Random weight bytes of `dtype` for `rows × cols`, with sane (small, finite) f16 scales.
fn random_matrix(dtype: GgmlType, rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let rb = dtype.row_bytes(cols);
    let mut bytes = lcg_bytes(rows * rb, seed);
    let bb = dtype.block_bytes();
    let scale_bytes = |v: f32| half::f16::from_f32(v).to_le_bytes();
    match dtype {
        GgmlType::F32 => {
            for (i, chunk) in bytes.chunks_exact_mut(4).enumerate() {
                let v = (((i * 7919 + seed as usize) % 2001) as f32 / 1000.0 - 1.0) * 0.5;
                chunk.copy_from_slice(&v.to_le_bytes());
            }
        }
        GgmlType::F16 => {
            for (i, chunk) in bytes.chunks_exact_mut(2).enumerate() {
                let v = (((i * 7919 + seed as usize) % 2001) as f32 / 1000.0 - 1.0) * 0.5;
                chunk.copy_from_slice(&scale_bytes(v));
            }
        }
        GgmlType::Q4_0 | GgmlType::Q8_0 => {
            for (i, blk) in bytes.chunks_exact_mut(bb).enumerate() {
                let d = 0.01 + (i % 13) as f32 * 0.003;
                blk[0..2].copy_from_slice(&scale_bytes(d));
            }
        }
        GgmlType::Q4_K | GgmlType::Q5_K => {
            for (i, blk) in bytes.chunks_exact_mut(bb).enumerate() {
                let d = 0.002 + (i % 11) as f32 * 0.001;
                let dmin = 0.001 + (i % 7) as f32 * 0.0005;
                blk[0..2].copy_from_slice(&scale_bytes(d));
                blk[2..4].copy_from_slice(&scale_bytes(dmin));
            }
        }
        GgmlType::Q6_K => {
            for (i, blk) in bytes.chunks_exact_mut(bb).enumerate() {
                let d = 0.001 + (i % 11) as f32 * 0.0005;
                blk[208..210].copy_from_slice(&scale_bytes(d));
            }
        }
        _ => unreachable!(),
    }
    bytes
}

fn reference_matmul(
    dtype: GgmlType,
    w: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    n: usize,
) -> Vec<f32> {
    let rb = dtype.row_bytes(cols);
    let mut row = vec![0f32; cols];
    let mut y = vec![0f32; n * rows];
    for r in 0..rows {
        dequantize_row(dtype, &w[r * rb..(r + 1) * rb], &mut row).unwrap();
        for t in 0..n {
            let xt = &x[t * cols..(t + 1) * cols];
            y[t * rows + r] = row.iter().zip(xt).map(|(a, b)| a * b).sum();
        }
    }
    y
}

fn kernel_name(kind: &str, dtype: GgmlType) -> &'static str {
    let suffix = match dtype {
        GgmlType::F32 => "f32",
        GgmlType::F16 => "f16",
        GgmlType::Q4_0 => "q4_0",
        GgmlType::Q8_0 => "q8_0",
        GgmlType::Q4_K => "q4_k",
        GgmlType::Q5_K => "q5_k",
        GgmlType::Q6_K => "q6_k",
        _ => unreachable!(),
    };
    // `dispatch` wants a `'static` name; leak once per (kind, type) in the test.
    Box::leak(format!("{kind}_{suffix}").into_boxed_str())
}

/// GEMV, GEMM and the embedding gather of every supported type against the scalar dequantizer.
#[test]
fn matmul_kernels_match_scalar_reference() {
    if !metal() {
        return;
    }
    let gpu = Gpu::get().unwrap();
    let rows = 70; // not a multiple of the 64-row GEMM tile or the 8-row GEMV group
    let cols = 512;
    let n = 37; // not a multiple of the 32-token tile
    let x: Vec<f32> = (0..n * cols)
        .map(|i| ((i * 48271 % 1000) as f32 / 1000.0 - 0.5) * 2.0)
        .collect();
    let xbuf = gpu.alloc(x.len() * 4).unwrap();
    xbuf.write_f32(0, &x);
    for dtype in [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::Q4_0,
        GgmlType::Q8_0,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
    ] {
        let w = random_matrix(dtype, rows, cols, 7 + dtype.id() as u64);
        let wbuf = gpu.alloc(w.len()).unwrap();
        wbuf.write_bytes(0, &w);
        let expect = reference_matmul(dtype, &w, rows, cols, &x, n);
        let scale = expect.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);

        // GEMV on the first token.
        let ybuf = gpu.alloc(n * rows * 4).unwrap();
        let cmd = gpu.begin().unwrap();
        cmd.dispatch(
            kernel_name("gemv", dtype),
            &[(0, &wbuf, 0), (1, &xbuf, 0), (2, &ybuf, 0), (4, &wbuf, 0)],
            3,
            &GemvParams {
                rows: rows as u32,
                cols: cols as u32,
            },
            (groups(rows, 4), 1, 1),
            (64, 1, 1),
        )
        .unwrap();
        cmd.finish().unwrap();
        let mut got = vec![0f32; rows];
        ybuf.read_f32(0, &mut got);
        let diff = max_abs_diff(&got, &expect[..rows]);
        assert!(
            diff <= 2e-3 * scale,
            "{dtype} gemv: max abs diff {diff} (scale {scale})"
        );

        // GEMM over all n tokens (edge tiles in both dimensions).
        let cmd = gpu.begin().unwrap();
        cmd.dispatch(
            kernel_name("gemm", dtype),
            &[(0, &wbuf, 0), (1, &xbuf, 0), (2, &ybuf, 0)],
            3,
            &GemmParams {
                rows: rows as u32,
                cols: cols as u32,
                n: n as u32,
                row_bytes: dtype.row_bytes(cols) as u32,
                accumulate: 0,
            },
            (groups(n, 32), groups(rows, 64), 1),
            (128, 1, 1),
        )
        .unwrap();
        cmd.finish().unwrap();
        let mut got = vec![0f32; n * rows];
        ybuf.read_f32(0, &mut got);
        let diff = max_abs_diff(&got, &expect);
        // f16 operands in the simdgroup path: ~1e-3 relative.
        assert!(
            diff <= 6e-3 * scale,
            "{dtype} gemm: max abs diff {diff} (scale {scale})"
        );

        // Embedding gather of rows 3 and 69.
        let toks = [3u32, 69];
        let tbuf = gpu.alloc(8).unwrap();
        tbuf.write_bytes(
            0,
            &toks
                .iter()
                .flat_map(|t| t.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let ebuf = gpu.alloc(2 * cols * 4).unwrap();
        let cmd = gpu.begin().unwrap();
        cmd.dispatch(
            kernel_name("embed", dtype),
            &[(0, &wbuf, 0), (1, &tbuf, 0), (2, &ebuf, 0)],
            3,
            &EmbedParams {
                cols: cols as u32,
                n_tokens: 2,
                row_bytes: dtype.row_bytes(cols) as u32,
            },
            (groups(2 * cols / 16, 64), 1, 1),
            (64, 1, 1),
        )
        .unwrap();
        cmd.finish().unwrap();
        let mut got = vec![0f32; 2 * cols];
        ebuf.read_f32(0, &mut got);
        let rb = dtype.row_bytes(cols);
        for (i, &t) in toks.iter().enumerate() {
            let mut row = vec![0f32; cols];
            dequantize_row(dtype, &w[t as usize * rb..(t as usize + 1) * rb], &mut row).unwrap();
            let diff = max_abs_diff(&got[i * cols..(i + 1) * cols], &row);
            assert!(diff <= 1e-6, "{dtype} embed row {t}: max abs diff {diff}");
        }
    }
}

#[test]
fn tiny_model_matches_cpu_and_is_prefill_consistent() {
    if !metal() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = tiny_model(dir.path());
    let f = GgufFile::open(&path).unwrap();
    let toks = [3u32, 17, 5, 42, 9];

    let mut cpu = CpuBackend::new(&f, 2, 16, 8).unwrap();
    let mut gpu = MetalBackend::new(&f, 16, 8).unwrap();
    assert_eq!(gpu.name(), "metal");

    // Prefill all five, then four decode steps; compare every step with the CPU.
    let lc = cpu.forward(&toks).to_vec();
    let lg = gpu.forward(&toks).to_vec();
    assert!(max_abs_diff(&lc, &lg) < 2e-3, "prefill logits differ");
    assert_eq!(argmax(&lc), argmax(&lg));
    let mut next = argmax(&lc) as u32;
    for _ in 0..4 {
        let lc = cpu.forward(&[next]).to_vec();
        let lg = gpu.forward(&[next]).to_vec();
        assert!(max_abs_diff(&lc, &lg) < 2e-3, "decode logits differ");
        assert_eq!(argmax(&lc), argmax(&lg));
        next = argmax(&lc) as u32;
    }
    assert_eq!(gpu.kv_len(), 9);

    // Prefill of five at once equals one token at a time (same backend, cleared between).
    gpu.clear();
    let a = gpu.forward(&toks).to_vec();
    gpu.clear();
    let mut b = Vec::new();
    for &t in &toks {
        b = gpu.forward(&[t]).to_vec();
    }
    assert!(
        max_abs_diff(&a, &b) < 2e-3,
        "prefill vs token-by-token differ"
    );
    assert_eq!(gpu.kv_len(), 5);

    // Truncate keeps a prefix.
    gpu.truncate(3);
    assert_eq!(gpu.kv_len(), 3);
    let c = gpu.forward(&toks[3..]).to_vec();
    assert!(max_abs_diff(&a, &c) < 2e-3, "truncate + recompute differ");
}

fn test_gguf() -> Option<GgufFile> {
    let path = std::env::var_os("LLMARIO_TEST_GGUF")?;
    Some(GgufFile::open(std::path::Path::new(&path)).expect("open LLMARIO_TEST_GGUF"))
}

/// Qwen3-1.7B (or any dense GGUF): prefill 8 tokens then 8 decode steps against the CPU backend.
#[test]
fn real_model_matches_cpu() {
    if !metal() {
        return;
    }
    let Some(f) = test_gguf() else {
        eprintln!("LLMARIO_TEST_GGUF not set; skipping");
        return;
    };
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let mut cpu = CpuBackend::new(&f, threads, 64, 16).unwrap();
    let mut gpu = MetalBackend::new(&f, 64, 16).unwrap();
    // "The capital of France is" for the Qwen tokenizer, otherwise just a fixed id sequence.
    let prompt = [785u32, 6722, 315, 9625, 374, 12095, 13, 576];
    let lc = cpu.forward(&prompt).to_vec();
    let lg = gpu.forward(&prompt).to_vec();
    let d = max_abs_diff(&lc, &lg);
    eprintln!(
        "prefill: max abs diff {d:.4}, argmax cpu {} gpu {}",
        argmax(&lc),
        argmax(&lg)
    );
    assert!(d <= 0.05, "prefill logits differ by {d}");
    assert_eq!(argmax(&lc), argmax(&lg));
    let mut next = argmax(&lc) as u32;
    for step in 0..8 {
        let lc = cpu.forward(&[next]).to_vec();
        let lg = gpu.forward(&[next]).to_vec();
        let d = max_abs_diff(&lc, &lg);
        eprintln!(
            "decode {step}: max abs diff {d:.4}, argmax cpu {} gpu {}",
            argmax(&lc),
            argmax(&lg)
        );
        assert!(d <= 0.05, "decode step {step} logits differ by {d}");
        assert_eq!(argmax(&lc), argmax(&lg), "argmax differs at step {step}");
        next = argmax(&lc) as u32;
    }
}

/// Prefill five tokens at once vs one at a time on the real model.
#[test]
fn real_model_prefill_consistency() {
    if !metal() {
        return;
    }
    let Some(f) = test_gguf() else {
        eprintln!("LLMARIO_TEST_GGUF not set; skipping");
        return;
    };
    let mut gpu = MetalBackend::new(&f, 64, 48).unwrap();
    // 40 tokens: the batch goes through the simdgroup-matrix prefill attention and the tiled
    // GEMM; token by token goes through the decode attention and the GEMV path.
    let prompt: Vec<u32> = (0..40u32).map(|i| 785 + (i * 7919) % 5000).collect();
    let a = gpu.forward(&prompt).to_vec();
    gpu.clear();
    let mut b = Vec::new();
    for &t in &prompt {
        b = gpu.forward(&[t]).to_vec();
    }
    let d = max_abs_diff(&a, &b);
    eprintln!("prefill vs token-by-token: max abs diff {d:.4}");
    assert!(d <= 0.05, "logits differ by {d}");
    assert_eq!(argmax(&a), argmax(&b));
}

/// The paged cache on Metal against the CPU's, for f16 and q8_0 rows: a 40-token prompt (the
/// flash kernel, two KV blocks) then decode steps across the block boundary; prefill of all 40
/// at once equals token by token.
#[test]
fn paged_kv_matches_cpu_f16_and_q8_0() {
    use llmario_engine_metal::MetalOptions;
    use llmario_engine_model::{CpuOptions, KvType};
    if !metal() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = tiny_model_dh(dir.path(), 64, 64, 128);
    let f = GgufFile::open(&path).unwrap();
    let toks: Vec<u32> = (0..40).map(|i| (i * 7 + 3) % 64).collect();
    for kv_type in [KvType::F16, KvType::Q8_0] {
        let mut cpu = CpuBackend::with_options(
            &f,
            CpuOptions {
                kv_type,
                ..CpuOptions::new(2, 128, 64)
            },
        )
        .unwrap();
        let mut gpu = MetalBackend::with_options(
            &f,
            MetalOptions {
                kv_type,
                ..MetalOptions::new(128, 64)
            },
        )
        .unwrap();
        assert_eq!(gpu.kv_type(), kv_type);
        let lc = cpu.forward(&toks).to_vec();
        let lg = gpu.forward(&toks).to_vec();
        let tol = if kv_type == KvType::Q8_0 { 2e-2 } else { 2e-3 };
        assert!(
            max_abs_diff(&lc, &lg) < tol,
            "{kv_type:?} prefill differs: {}",
            max_abs_diff(&lc, &lg)
        );
        assert_eq!(argmax(&lc), argmax(&lg));
        let mut next = argmax(&lc) as u32;
        for _ in 0..30 {
            let lc = cpu.forward(&[next]).to_vec();
            let lg = gpu.forward(&[next]).to_vec();
            assert!(max_abs_diff(&lc, &lg) < tol, "{kv_type:?} decode differs");
            assert_eq!(argmax(&lc), argmax(&lg));
            next = argmax(&lc) as u32;
        }
        assert_eq!(gpu.kv_len(), 70);
        // Prefill vs token by token on the GPU.
        gpu.clear();
        let a = gpu.forward(&toks).to_vec();
        gpu.clear();
        let mut b = Vec::new();
        for &t in &toks {
            b = gpu.forward(&[t]).to_vec();
        }
        assert!(
            max_abs_diff(&a, &b) < tol,
            "{kv_type:?} prefill vs token-by-token differ"
        );
    }
}

/// Several sequences in one Metal call (a long prompt through the flash kernel, a single token
/// and a short prompt through the decode kernel, in mixed order) give each sequence the logits
/// it gets alone, then a batched decode step does too.
#[test]
fn batched_sequences_match_separate_runs() {
    use llmario_engine_metal::MetalOptions;
    use llmario_engine_model::SeqTokens;
    if !metal() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = tiny_model_dh(dir.path(), 64, 64, 128);
    let f = GgufFile::open(&path).unwrap();
    let long: Vec<u32> = (0..40).map(|i| (i * 5 + 1) % 64).collect();
    let prompts: [&[u32]; 3] = [&[9], &long, &[4, 8, 15, 16, 23]];
    let v = 64usize;
    let mut alone = Vec::new();
    for p in prompts {
        let mut gpu = MetalBackend::new(&f, 128, 64).unwrap();
        let a = gpu.forward(p).to_vec();
        let b = gpu.forward(&[7]).to_vec();
        alone.push((a, b));
    }
    let mut gpu = MetalBackend::with_options(
        &f,
        MetalOptions {
            n_seqs: 3,
            ..MetalOptions::new(128, 64)
        },
    )
    .unwrap();
    assert_eq!(gpu.n_seqs(), 3);
    let batch: Vec<SeqTokens> = prompts
        .iter()
        .enumerate()
        .map(|(i, p)| SeqTokens { seq: i, tokens: p })
        .collect();
    let first = gpu.forward_batch(&batch).unwrap().to_vec();
    let dec: Vec<SeqTokens> = (0..3)
        .map(|i| SeqTokens {
            seq: i,
            tokens: &[7],
        })
        .collect();
    let next = gpu.forward_batch(&dec).unwrap().to_vec();
    for (i, (a, b)) in alone.iter().enumerate() {
        let d1 = max_abs_diff(a, &first[i * v..(i + 1) * v]);
        let d2 = max_abs_diff(b, &next[i * v..(i + 1) * v]);
        assert!(d1 < 2e-3 && d2 < 2e-3, "seq {i}: prefill {d1}, decode {d2}");
        assert_eq!(gpu.seq_len(i), prompts[i].len() + 1);
    }
}

/// Metal KV blocks exist only while used: none after load, two after 40 tokens, one after a
/// truncation to 10, none after clearing.
#[test]
fn metal_kv_blocks_follow_use() {
    if !metal() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = tiny_model_dh(dir.path(), 64, 64, 128);
    let f = GgufFile::open(&path).unwrap();
    let mut gpu = MetalBackend::new(&f, 128, 64).unwrap();
    let table = gpu.kv_in_use_bytes();
    let toks: Vec<u32> = (0..40).collect();
    gpu.forward(&toks);
    let two = gpu.kv_in_use_bytes() - table;
    assert!(two > 0);
    gpu.truncate(10);
    assert_eq!(gpu.kv_in_use_bytes() - table, two / 2);
    gpu.clear();
    assert_eq!(gpu.kv_in_use_bytes(), table);
    // The reservation is the whole context (4 blocks of 32 + the copy-on-write spare).
    assert_eq!(gpu.kv_reserved_bytes() - table, 5 * (two / 2));
}
