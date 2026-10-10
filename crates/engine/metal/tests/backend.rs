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

/// A Metal sequence exported after a 40-token prompt and imported into another backend's second
/// slot continues exactly as the original does (f16 and q8_0); the CPU layout has a different
/// fingerprint, so a snapshot never crosses backends.
#[test]
fn metal_snapshot_round_trip() {
    use llmario_engine_metal::MetalOptions;
    use llmario_engine_model::{KvType, SeqTokens};
    if !metal() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = tiny_model_dh(dir.path(), 64, 64, 128);
    let f = GgufFile::open(&path).unwrap();
    let toks: Vec<u32> = (0..40).map(|i| (i * 3 + 2) % 64).collect();
    for kv_type in [KvType::F16, KvType::Q8_0] {
        let opts = MetalOptions {
            kv_type,
            ..MetalOptions::new(128, 64)
        };
        let mut a = MetalBackend::with_options(&f, opts).unwrap();
        a.forward(&toks);
        let snap = a.export_seq(0).unwrap();
        assert_eq!(snap.len, 40);
        let want = a.forward(&[5]).to_vec();
        let mut b = MetalBackend::with_options(&f, MetalOptions { n_seqs: 2, ..opts }).unwrap();
        assert_eq!(a.kv_fingerprint(), b.kv_fingerprint());
        b.forward(&[1, 2, 3]);
        assert!(b.import_seq(1, &snap));
        assert_eq!(b.seq_len(1), 40);
        assert_eq!(b.seq_len(0), 3);
        let got = b
            .forward_batch(&[SeqTokens {
                seq: 1,
                tokens: &[5],
            }])
            .unwrap()
            .to_vec();
        assert!(max_abs_diff(&want, &got) < 1e-5, "{kv_type:?}");
        let cpu = CpuBackend::new(&f, 1, 128, 64).unwrap();
        assert_ne!(cpu.kv_fingerprint(), a.kv_fingerprint());
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MoeRouteParams {
    n_expert: u32,
    k: u32,
    norm: u32,
    pad: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct GemvIdParams {
    rows: u32,
    cols: u32,
    expert_bytes: u32,
    k: u32,
    x_per_pair: u32,
    pad: [u32; 3],
}
#[repr(C)]
#[derive(Clone, Copy)]
struct MoeGroupParams {
    n_pairs: u32,
    n_expert: u32,
    cap: u32,
    pad: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct GemmIdParams {
    rows: u32,
    cols: u32,
    row_bytes: u32,
    expert_bytes: u32,
    k: u32,
    x_per_pair: u32,
    cap: u32,
    pad: u32,
}

/// The CPU routing rule (moe.rs `route`): softmax, top-k descending with ties to the lower
/// expert, optional renormalisation with the sum clamped at 6.1035156e-5.
fn route_ref(logits: &[f32], k: usize, norm: bool) -> (Vec<u32>, Vec<f32>) {
    let m = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let e: Vec<f64> = logits.iter().map(|&l| ((l - m) as f64).exp()).collect();
    let z: f64 = e.iter().sum();
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.sort_by(|&a, &b| {
        logits[b as usize]
            .total_cmp(&logits[a as usize])
            .then(a.cmp(&b))
    });
    let sel = idx[..k].to_vec();
    let mut w: Vec<f32> = sel.iter().map(|&i| (e[i as usize] / z) as f32).collect();
    if norm {
        let s = w.iter().sum::<f32>().max(6.103_515_6e-5);
        for v in &mut w {
            *v /= s;
        }
    }
    (sel, w)
}

/// `moe_route` picks the CPU's experts in the CPU's order (including ties and expert counts
/// that are not a multiple of the simdgroup width) with the same weights.
#[test]
fn moe_route_matches_the_cpu_rule() {
    if !metal() {
        return;
    }
    let gpu = Gpu::get().unwrap();
    for (ne, k, norm) in [
        (128usize, 8usize, true),
        (40, 3, true),
        (5, 5, false),
        (64, 1, true),
    ] {
        let n = 6;
        let mut logits: Vec<f32> = (0..n * ne)
            .map(|i| (((i * 7919) % 1013) as f32 / 1013.0 - 0.5) * 6.0)
            .collect();
        // Row 0: a three-way tie at the top; row 1: every logit equal.
        if ne >= 8 {
            for e in [2, 5, 7] {
                logits[e] = 9.0;
            }
        }
        for v in &mut logits[ne..2 * ne] {
            *v = 0.25;
        }
        let lbuf = gpu.alloc(logits.len() * 4).unwrap();
        lbuf.write_f32(0, &logits);
        let sbuf = gpu.alloc(n * k * 4).unwrap();
        let wbuf = gpu.alloc(n * k * 4).unwrap();
        let cmd = gpu.begin().unwrap();
        cmd.dispatch(
            "moe_route",
            &[(0, &lbuf, 0), (1, &sbuf, 0), (2, &wbuf, 0)],
            3,
            &MoeRouteParams {
                n_expert: ne as u32,
                k: k as u32,
                norm: norm as u32,
                pad: 0,
            },
            (n, 1, 1),
            (32, 1, 1),
        )
        .unwrap();
        cmd.finish().unwrap();
        let sel: Vec<u32> = sbuf.as_slice()[..n * k * 4]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let mut w = vec![0f32; n * k];
        wbuf.read_f32(0, &mut w);
        for t in 0..n {
            let (rs, rw) = route_ref(&logits[t * ne..(t + 1) * ne], k, norm);
            assert_eq!(&sel[t * k..(t + 1) * k], &rs[..], "ne {ne} k {k} row {t}");
            for j in 0..k {
                let (a, b) = (w[t * k + j], rw[j]);
                assert!(
                    (a - b).abs() <= 1e-6 * b.abs().max(1.0),
                    "row {t} slot {j}: {a} vs {b}"
                );
            }
        }
    }
}

/// The expert matvec (plain and gate/up SwiGLU) and the grouped expert GEMM of every supported
/// weight type against the scalar dequantizer, with experts shared between rows, rows per token
/// (`pair / k`) and rows per pair.
#[test]
fn moe_expert_kernels_match_scalar_reference() {
    if !metal() {
        return;
    }
    let gpu = Gpu::get().unwrap();
    let (ne, rows, cols, n, k) = (5usize, 70usize, 512usize, 7usize, 2usize);
    let pairs = n * k;
    // Expert choices per (token, slot): expert 4 by most tokens, expert 1 by none.
    let sel: Vec<u32> = (0..pairs)
        .map(|p| [4u32, 0, 4, 2, 3, 4, 0][p / k] ^ (p % k) as u32)
        .map(|e| if e == 1 { 4 } else { e % ne as u32 })
        .collect();
    let sel_bytes: Vec<u8> = sel.iter().flat_map(|e| e.to_le_bytes()).collect();
    let selbuf = gpu.alloc(sel_bytes.len()).unwrap();
    selbuf.write_bytes(0, &sel_bytes);
    let xt: Vec<f32> = (0..n * cols)
        .map(|i| ((i * 48271 % 1000) as f32 / 1000.0 - 0.5) * 2.0)
        .collect();
    let xp: Vec<f32> = (0..pairs * cols)
        .map(|i| ((i * 69621 % 1000) as f32 / 1000.0 - 0.5) * 2.0)
        .collect();
    let xtbuf = gpu.alloc(xt.len() * 4).unwrap();
    xtbuf.write_f32(0, &xt);
    let xpbuf = gpu.alloc(xp.len() * 4).unwrap();
    xpbuf.write_f32(0, &xp);
    let ybuf = gpu.alloc(pairs * rows * 4).unwrap();
    let counts = gpu.alloc(ne * 4).unwrap();
    let ids = gpu.alloc(ne * n * 4).unwrap();
    // Group once (the expert lists do not depend on the weight type).
    let cmd = gpu.begin().unwrap();
    cmd.dispatch(
        "moe_group",
        &[(0, &selbuf, 0), (1, &counts, 0), (2, &ids, 0)],
        3,
        &MoeGroupParams {
            n_pairs: pairs as u32,
            n_expert: ne as u32,
            cap: n as u32,
            pad: 0,
        },
        (1, 1, 1),
        (256, 1, 1),
    )
    .unwrap();
    cmd.finish().unwrap();
    for dtype in [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::Q4_0,
        GgmlType::Q8_0,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
    ] {
        let eb = dtype.row_bytes(cols) * rows;
        let w = random_matrix(dtype, ne * rows, cols, 11 + dtype.id() as u64);
        let w2 = random_matrix(dtype, ne * rows, cols, 23 + dtype.id() as u64);
        let wbuf = gpu.alloc(w.len()).unwrap();
        wbuf.write_bytes(0, &w);
        let w2buf = gpu.alloc(w2.len()).unwrap();
        w2buf.write_bytes(0, &w2);
        // Reference per pair: expert sel[p] times its token row (or its own row).
        let expect = |wb: &[u8], x: &[f32], per_pair: bool| -> Vec<f32> {
            let mut out = vec![0f32; pairs * rows];
            for p in 0..pairs {
                let e = sel[p] as usize;
                let xr = if per_pair { p } else { p / k };
                let y = reference_matmul(
                    dtype,
                    &wb[e * eb..(e + 1) * eb],
                    rows,
                    cols,
                    &x[xr * cols..(xr + 1) * cols],
                    1,
                );
                out[p * rows..(p + 1) * rows].copy_from_slice(&y);
            }
            out
        };
        let read = || {
            let mut v = vec![0f32; pairs * rows];
            ybuf.read_f32(0, &mut v);
            v
        };
        for per_pair in [false, true] {
            let (x, xbuf) = if per_pair {
                (&xp, &xpbuf)
            } else {
                (&xt, &xtbuf)
            };
            let want = expect(&w, x, per_pair);
            let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
            let gp = GemvIdParams {
                rows: rows as u32,
                cols: cols as u32,
                expert_bytes: eb as u32,
                k: k as u32,
                x_per_pair: per_pair as u32,
                pad: [0; 3],
            };
            let cmd = gpu.begin().unwrap();
            cmd.dispatch(
                kernel_name("gemv_id", dtype),
                &[
                    (0, &wbuf, 0),
                    (1, xbuf, 0),
                    (2, &ybuf, 0),
                    (4, &wbuf, 0),
                    (5, &selbuf, 0),
                ],
                3,
                &gp,
                (groups(rows, 4), pairs, 1),
                (64, 1, 1),
            )
            .unwrap();
            cmd.finish().unwrap();
            let diff = max_abs_diff(&read(), &want);
            assert!(
                diff <= 2e-3 * scale,
                "{dtype} gemv_id (per pair {per_pair}): {diff} (scale {scale})"
            );
            let mp = GemmIdParams {
                rows: rows as u32,
                cols: cols as u32,
                row_bytes: dtype.row_bytes(cols) as u32,
                expert_bytes: eb as u32,
                k: k as u32,
                x_per_pair: per_pair as u32,
                cap: n as u32,
                pad: 0,
            };
            let cmd = gpu.begin().unwrap();
            cmd.dispatch(
                kernel_name("gemm_id", dtype),
                &[
                    (0, &wbuf, 0),
                    (1, xbuf, 0),
                    (2, &ybuf, 0),
                    (4, &counts, 0),
                    (5, &ids, 0),
                ],
                3,
                &mp,
                (groups(n, 32), groups(rows, 64), ne),
                (128, 1, 1),
            )
            .unwrap();
            cmd.finish().unwrap();
            let diff = max_abs_diff(&read(), &want);
            assert!(
                diff <= 6e-3 * scale,
                "{dtype} gemm_id (per pair {per_pair}): {diff} (scale {scale})"
            );
        }
        // Gate/up SwiGLU per pair.
        let g = expect(&w, &xt, false);
        let u = expect(&w2, &xt, false);
        let want: Vec<f32> = g
            .iter()
            .zip(&u)
            .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
        let cmd = gpu.begin().unwrap();
        cmd.dispatch(
            kernel_name("gemv_glu_id", dtype),
            &[
                (0, &wbuf, 0),
                (1, &xtbuf, 0),
                (2, &ybuf, 0),
                (4, &w2buf, 0),
                (5, &selbuf, 0),
            ],
            3,
            &GemvIdParams {
                rows: rows as u32,
                cols: cols as u32,
                expert_bytes: eb as u32,
                k: k as u32,
                x_per_pair: 0,
                pad: [0; 3],
            },
            (groups(rows, 4), pairs, 1),
            (64, 1, 1),
        )
        .unwrap();
        cmd.finish().unwrap();
        let diff = max_abs_diff(&read(), &want);
        assert!(
            diff <= 4e-3 * scale,
            "{dtype} gemv_glu_id: {diff} (scale {scale})"
        );
    }
}

/// A tiny random `qwen3moe` model (two layers, 6 experts, 2 per token) for the CPU-vs-Metal
/// comparison; head width 64 so prompts take the flash attention kernel.
fn tiny_moe(dir: &std::path::Path) -> std::path::PathBuf {
    let (d, n_head, n_kv, hd, vocab, n_layer) = (64u64, 4u64, 2u64, 64u64, 64u64, 2u64);
    let (ne, k, ff) = (6u64, 2u64, 48u64);
    let f32s = |n: u64, seed: u64| -> Vec<u8> {
        (0..n)
            .flat_map(|i| {
                let v = (((i * 2654435761 + seed * 97) % 1000) as f32 / 1000.0 - 0.5) * 0.3;
                v.to_le_bytes()
            })
            .collect()
    };
    let ones = |n: u64| -> Vec<u8> { (0..n).flat_map(|_| 1.0f32.to_le_bytes()).collect() };
    let a = "qwen3moe";
    let mut w = GgufWriter::new();
    w.meta("general.architecture", MetaValue::Str(a.into()))
        .meta(&format!("{a}.block_count"), MetaValue::U32(n_layer as u32))
        .meta(&format!("{a}.embedding_length"), MetaValue::U32(d as u32))
        .meta(
            &format!("{a}.attention.head_count"),
            MetaValue::U32(n_head as u32),
        )
        .meta(
            &format!("{a}.attention.head_count_kv"),
            MetaValue::U32(n_kv as u32),
        )
        .meta(
            &format!("{a}.attention.key_length"),
            MetaValue::U32(hd as u32),
        )
        .meta(
            &format!("{a}.attention.value_length"),
            MetaValue::U32(hd as u32),
        )
        .meta(&format!("{a}.expert_count"), MetaValue::U32(ne as u32))
        .meta(&format!("{a}.expert_used_count"), MetaValue::U32(k as u32))
        .meta(
            &format!("{a}.expert_feed_forward_length"),
            MetaValue::U32(ff as u32),
        )
        .meta(&format!("{a}.vocab_size"), MetaValue::U32(vocab as u32))
        .meta(&format!("{a}.context_length"), MetaValue::U32(128))
        .meta(&format!("{a}.rope.freq_base"), MetaValue::F32(1_000_000.0))
        .meta(
            &format!("{a}.attention.layer_norm_rms_epsilon"),
            MetaValue::F32(1e-6),
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
        for (name, rows, seed) in [
            ("attn_q.weight", n_head * hd, 10),
            ("attn_k.weight", n_kv * hd, 20),
            ("attn_v.weight", n_kv * hd, 30),
        ] {
            w.tensor(
                &p(name),
                &[d, rows],
                GgmlType::F32,
                f32s(d * rows, seed + l),
            );
        }
        w.tensor(
            &p("attn_output.weight"),
            &[n_head * hd, d],
            GgmlType::F32,
            f32s(n_head * hd * d, 40 + l),
        );
        w.tensor(&p("attn_q_norm.weight"), &[hd], GgmlType::F32, ones(hd));
        w.tensor(&p("attn_k_norm.weight"), &[hd], GgmlType::F32, ones(hd));
        w.tensor(&p("ffn_norm.weight"), &[d], GgmlType::F32, ones(d));
        w.tensor(
            &p("ffn_gate_inp.weight"),
            &[d, ne],
            GgmlType::F32,
            f32s(d * ne, 50 + l),
        );
        w.tensor(
            &p("ffn_gate_exps.weight"),
            &[d, ff, ne],
            GgmlType::F32,
            f32s(d * ff * ne, 60 + l),
        );
        w.tensor(
            &p("ffn_up_exps.weight"),
            &[d, ff, ne],
            GgmlType::F32,
            f32s(d * ff * ne, 70 + l),
        );
        w.tensor(
            &p("ffn_down_exps.weight"),
            &[ff, d, ne],
            GgmlType::F32,
            f32s(ff * d * ne, 80 + l),
        );
    }
    let p = dir.join("tiny_moe.gguf");
    std::fs::write(&p, w.to_bytes()).unwrap();
    p
}

/// A `qwen3moe` model on Metal gives the CPU's logits: a 40-token prompt (grouped expert GEMMs),
/// decode steps (expert matvecs), a 3-token prompt (matvec path with several rows), and three
/// sequences in one call.
#[test]
fn moe_model_matches_cpu() {
    use llmario_engine_metal::MetalOptions;
    use llmario_engine_model::SeqTokens;
    if !metal() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = tiny_moe(dir.path());
    let f = GgufFile::open(&path).unwrap();
    let toks: Vec<u32> = (0..40).map(|i| (i * 7 + 3) % 64).collect();
    let mut cpu = CpuBackend::new(&f, 2, 128, 64).unwrap();
    let mut gpu = MetalBackend::new(&f, 128, 64).unwrap();
    let tol = 2e-3;
    let lc = cpu.forward(&toks).to_vec();
    let lg = gpu.forward(&toks).to_vec();
    let spread =
        lc.iter().fold(f32::MIN, |a, &b| a.max(b)) - lc.iter().fold(f32::MAX, |a, &b| a.min(b));
    eprintln!(
        "prefill: logit spread {spread}, max diff {}",
        max_abs_diff(&lc, &lg)
    );
    // The tolerance must be small against the logits for the comparison to mean anything.
    assert!(spread > 100.0 * tol, "logit spread {spread}");
    assert!(
        max_abs_diff(&lc, &lg) < tol,
        "prefill: {}",
        max_abs_diff(&lc, &lg)
    );
    assert_eq!(argmax(&lc), argmax(&lg));
    let mut next = argmax(&lc) as u32;
    for step in 0..8 {
        let lc = cpu.forward(&[next]).to_vec();
        let lg = gpu.forward(&[next]).to_vec();
        assert!(
            max_abs_diff(&lc, &lg) < tol,
            "decode step {step}: {}",
            max_abs_diff(&lc, &lg)
        );
        next = argmax(&lc) as u32;
    }
    let lc = cpu.forward(&[5, 9, 13]).to_vec();
    let lg = gpu.forward(&[5, 9, 13]).to_vec();
    assert!(max_abs_diff(&lc, &lg) < tol, "3-token prompt");

    // Three sequences in one Metal call equal the CPU running each alone.
    let prompts: [&[u32]; 3] = [&toks, &[9], &[4, 8, 15]];
    let mut gpu = MetalBackend::with_options(
        &f,
        MetalOptions {
            n_seqs: 3,
            ..MetalOptions::new(128, 64)
        },
    )
    .unwrap();
    let batch: Vec<SeqTokens> = prompts
        .iter()
        .enumerate()
        .map(|(i, p)| SeqTokens { seq: i, tokens: p })
        .collect();
    let got = gpu.forward_batch(&batch).unwrap().to_vec();
    let v = 64;
    for (i, p) in prompts.iter().enumerate() {
        let mut cpu = CpuBackend::new(&f, 2, 128, 64).unwrap();
        let want = cpu.forward(p).to_vec();
        let diff = max_abs_diff(&want, &got[i * v..(i + 1) * v]);
        assert!(diff < tol, "sequence {i}: {diff}");
    }
}

/// A tiny random Gemma 4 model (four layers: sliding, sliding, global K=V, sliding) with
/// `hd_swa` / `hd_full` head widths and an `n_swa`-position window (CPU forward tests' generator).
fn tiny_gemma4(dir: &std::path::Path, hd_swa: u64, hd_full: u64, n_swa: u32) -> std::path::PathBuf {
    let (d, n_head, n_ff, vocab, n_layer) = (64u64, 4u64, 48u64, 64u64, 4u64);
    let swa = [true, true, false, true];
    let (n_kv_swa, n_kv_full) = (2u64, 1u64);
    let f32s = |n: u64, seed: u64| -> Vec<u8> {
        (0..n)
            .flat_map(|i| {
                let v = (((i * 2654435761 + seed * 97) % 1000) as f32 / 1000.0 - 0.5) * 0.3;
                v.to_le_bytes()
            })
            .collect()
    };
    let consts = |n: u64, c: f32| -> Vec<u8> { (0..n).flat_map(|_| c.to_le_bytes()).collect() };
    let a = "gemma4";
    let u = |v: u64| MetaValue::U32(v as u32);
    let mut w = GgufWriter::new();
    w.meta("general.architecture", MetaValue::Str(a.into()))
        .meta(&format!("{a}.block_count"), u(n_layer))
        .meta(&format!("{a}.embedding_length"), u(d))
        .meta(&format!("{a}.attention.head_count"), u(n_head))
        .meta(
            &format!("{a}.attention.head_count_kv"),
            MetaValue::Array(
                swa.iter()
                    .map(|&s| u(if s { n_kv_swa } else { n_kv_full }))
                    .collect(),
            ),
        )
        .meta(
            &format!("{a}.attention.sliding_window_pattern"),
            MetaValue::Array(swa.iter().map(|&s| MetaValue::Bool(s)).collect()),
        )
        .meta(
            &format!("{a}.attention.sliding_window"),
            MetaValue::U32(n_swa),
        )
        .meta(&format!("{a}.attention.key_length"), u(hd_full))
        .meta(&format!("{a}.attention.value_length"), u(hd_full))
        .meta(&format!("{a}.attention.key_length_swa"), u(hd_swa))
        .meta(&format!("{a}.attention.value_length_swa"), u(hd_swa))
        .meta(&format!("{a}.attention.shared_kv_layers"), u(0))
        .meta(&format!("{a}.embedding_length_per_layer_input"), u(0))
        .meta(&format!("{a}.feed_forward_length"), u(n_ff))
        .meta(&format!("{a}.vocab_size"), u(vocab))
        .meta(&format!("{a}.context_length"), u(256))
        .meta(&format!("{a}.rope.freq_base"), MetaValue::F32(1_000_000.0))
        .meta(&format!("{a}.rope.freq_base_swa"), MetaValue::F32(10_000.0))
        .meta(&format!("{a}.rope.dimension_count"), u(hd_full))
        .meta(&format!("{a}.rope.dimension_count_swa"), u(hd_swa))
        .meta(
            &format!("{a}.final_logit_softcapping"),
            MetaValue::F32(30.0),
        )
        .meta(
            &format!("{a}.attention.layer_norm_rms_epsilon"),
            MetaValue::F32(1e-6),
        );
    w.tensor(
        "token_embd.weight",
        &[d, vocab],
        GgmlType::F32,
        f32s(d * vocab, 1),
    );
    w.tensor("output_norm.weight", &[d], GgmlType::F32, consts(d, 1.0));
    // Proportional RoPE on the global layers: the first quarter of the pairs rotate.
    let ff: Vec<u8> = (0..hd_full / 2)
        .flat_map(|i| (if i < hd_full / 8 { 1.0f32 } else { 1e30 }).to_le_bytes())
        .collect();
    w.tensor("rope_freqs.weight", &[hd_full / 2], GgmlType::F32, ff);
    for l in 0..n_layer {
        let p = |s: &str| format!("blk.{l}.{s}");
        let (hd, n_kv) = if swa[l as usize] {
            (hd_swa, n_kv_swa)
        } else {
            (hd_full, n_kv_full)
        };
        w.tensor(&p("attn_norm.weight"), &[d], GgmlType::F32, consts(d, 1.0));
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
        if swa[l as usize] {
            w.tensor(
                &p("attn_v.weight"),
                &[d, n_kv * hd],
                GgmlType::F32,
                f32s(d * n_kv * hd, 30 + l),
            );
        }
        w.tensor(
            &p("attn_q_norm.weight"),
            &[hd],
            GgmlType::F32,
            f32s(hd, 80 + l),
        );
        w.tensor(
            &p("attn_k_norm.weight"),
            &[hd],
            GgmlType::F32,
            f32s(hd, 90 + l),
        );
        w.tensor(
            &p("attn_output.weight"),
            &[n_head * hd, d],
            GgmlType::F32,
            f32s(n_head * hd * d, 40 + l),
        );
        w.tensor(
            &p("post_attention_norm.weight"),
            &[d],
            GgmlType::F32,
            consts(d, 1.0),
        );
        w.tensor(&p("ffn_norm.weight"), &[d], GgmlType::F32, consts(d, 1.0));
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
        w.tensor(
            &p("post_ffw_norm.weight"),
            &[d],
            GgmlType::F32,
            consts(d, 1.0),
        );
        w.tensor(
            &p("layer_output_scale.weight"),
            &[1],
            GgmlType::F32,
            consts(1, 0.9),
        );
    }
    let p = dir.join(format!("tiny_gemma4_{hd_swa}_{hd_full}_{n_swa}.gguf"));
    std::fs::write(&p, w.to_bytes()).unwrap();
    p
}

/// Gemma 4 on Metal gives the CPU's logits (f16 and q8_0 caches).
///
/// - Token by token for 80 positions past a 9-position window (window blocks are released as the
///   window slides): the GPU's single-token path (f32 activations) against the CPU, tightly.
/// - A 40-token prompt at once: the batched GEMM path rounds activations to f16 inside its tiles
///   (as llama.cpp's Metal kernels do), so the tolerance is wider and the top token must agree.
/// - Three sequences in one call against the CPU running each alone.
#[test]
fn gemma4_matches_cpu() {
    use llmario_engine_metal::MetalOptions;
    use llmario_engine_model::{CpuOptions, KvType, SeqTokens};
    if !metal() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = tiny_gemma4(dir.path(), 32, 64, 9);
    let f = GgufFile::open(&path).unwrap();
    let toks: Vec<u32> = (0..40).map(|i| (i * 7 + 3) % 64).collect();
    for kv_type in [KvType::F16, KvType::Q8_0] {
        let cpu_opts = CpuOptions {
            kv_type,
            ..CpuOptions::new(2, 256, 64)
        };
        let gpu_opts = MetalOptions {
            kv_type,
            ..MetalOptions::new(256, 64)
        };
        let q8 = kv_type == KvType::Q8_0;
        // Single-token path.
        let mut cpu = CpuBackend::with_options(&f, cpu_opts).unwrap();
        let mut gpu = MetalBackend::with_options(&f, gpu_opts).unwrap();
        let tol = if q8 { 2e-2 } else { 1e-3 };
        let mut next = 3u32;
        let mut spread = 0f32;
        for step in 0..80 {
            let lc = cpu.forward(&[next]).to_vec();
            let lg = gpu.forward(&[next]).to_vec();
            spread = spread.max(
                lc.iter().fold(f32::MIN, |a, &b| a.max(b))
                    - lc.iter().fold(f32::MAX, |a, &b| a.min(b)),
            );
            let diff = max_abs_diff(&lc, &lg);
            assert!(diff < tol, "{kv_type:?} step {step}: {diff}");
            next = argmax(&lc) as u32;
        }
        assert!(spread > 100.0 * tol, "logit spread {spread}");
        assert_eq!(gpu.kv_len(), 80);
        // Prompt at once.
        let mut cpu = CpuBackend::with_options(&f, cpu_opts).unwrap();
        let mut gpu = MetalBackend::with_options(&f, gpu_opts).unwrap();
        let lc = cpu.forward(&toks).to_vec();
        let lg = gpu.forward(&toks).to_vec();
        let ptol = if q8 { 2e-2 } else { 6e-3 };
        let diff = max_abs_diff(&lc, &lg);
        eprintln!("{kv_type:?}: prompt at once differs by {diff}");
        assert!(diff < ptol, "{kv_type:?} prompt: {diff}");
        assert_eq!(argmax(&lc), argmax(&lg));
    }

    // Three sequences in one call equal the CPU running each alone.
    let prompts: [&[u32]; 3] = [&toks, &[9], &[4, 8, 15, 16, 23]];
    let mut gpu = MetalBackend::with_options(
        &f,
        MetalOptions {
            n_seqs: 3,
            ..MetalOptions::new(256, 64)
        },
    )
    .unwrap();
    let batch: Vec<SeqTokens> = prompts
        .iter()
        .enumerate()
        .map(|(i, p)| SeqTokens { seq: i, tokens: p })
        .collect();
    let got = gpu.forward_batch(&batch).unwrap().to_vec();
    for (i, p) in prompts.iter().enumerate() {
        let mut cpu = CpuBackend::new(&f, 2, 256, 64).unwrap();
        let want = cpu.forward(p).to_vec();
        let diff = max_abs_diff(&want, &got[i * 64..(i + 1) * 64]);
        assert!(diff < 6e-3, "sequence {i}: {diff}");
        assert_eq!(argmax(&want), argmax(&got[i * 64..(i + 1) * 64]));
    }
}

/// Window blocks: memory stops growing once the window slides past whole blocks; cutting back
/// within the live window keeps the prefix, cutting behind it resets the sequence.
#[test]
fn gemma4_window_blocks_follow_the_window() {
    if !metal() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = tiny_gemma4(dir.path(), 32, 64, 9);
    let f = GgufFile::open(&path).unwrap();
    let mut gpu = MetalBackend::new(&f, 256, 64).unwrap();
    let base = gpu.kv_in_use_bytes();
    gpu.forward(&(0..32).collect::<Vec<u32>>());
    let one = gpu.kv_in_use_bytes() - base; // one global block + one window block
    for t in 0..96u32 {
        gpu.forward(&[t % 64]);
    }
    // 128 positions: 4 global blocks, but the window still holds at most 2 window blocks.
    let used = gpu.kv_in_use_bytes() - base;
    assert!(
        used < 4 * one,
        "window blocks must be released: {used} bytes vs {one} per block pair"
    );
    // Cut back by 3 positions: the window behind the new end is still there.
    gpu.truncate(125);
    assert_eq!(gpu.kv_len(), 125);
    // Cut back by 100 positions: the window blocks for position 28 were released.
    gpu.truncate(28);
    assert_eq!(gpu.kv_len(), 0, "a released window block forces a reset");
    assert_eq!(gpu.kv_in_use_bytes(), base);
}

/// A Gemma 4 sequence whose early window blocks were already released snapshots (paged blocks
/// plus the visible window blocks) and continues identically in another backend's second slot.
#[test]
fn gemma4_snapshot_round_trip() {
    use llmario_engine_metal::MetalOptions;
    use llmario_engine_model::SeqTokens;
    if !metal() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = tiny_gemma4(dir.path(), 32, 64, 9);
    let f = GgufFile::open(&path).unwrap();
    let mut a = MetalBackend::new(&f, 256, 64).unwrap();
    for t in 0..80u32 {
        a.forward(&[(t * 5 + 1) % 64]);
    }
    let snap = a.export_seq(0).unwrap();
    assert_eq!(snap.len, 80);
    assert!(!a.snapshot_trimmable());
    let want = a.forward(&[7]).to_vec();
    let mut b = MetalBackend::with_options(
        &f,
        MetalOptions {
            n_seqs: 2,
            ..MetalOptions::new(256, 64)
        },
    )
    .unwrap();
    assert_eq!(a.kv_fingerprint(), b.kv_fingerprint());
    b.forward(&[1, 2, 3]);
    assert!(b.import_seq(1, &snap));
    let got = b
        .forward_batch(&[SeqTokens {
            seq: 1,
            tokens: &[7],
        }])
        .unwrap()
        .to_vec();
    assert!(
        max_abs_diff(&want, &got) < 1e-5,
        "{}",
        max_abs_diff(&want, &got)
    );
}
