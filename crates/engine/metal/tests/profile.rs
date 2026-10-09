//! Kernel profile of one decode step and one 512-token prefill on a real model (prints a table;
//! never asserts on speed). Needs a Metal device and `LLMARIO_TEST_GGUF`.

#![cfg(target_os = "macos")]

use llmario_engine_formats::GgufFile;
use llmario_engine_metal::MetalBackend;
use llmario_engine_model::ModelBackend;
use std::collections::BTreeMap;
use std::time::Instant;

fn report(label: &str, prof: &[(&'static str, f64)], wall: f64, gpu: f64) {
    let mut by: BTreeMap<&str, (usize, f64)> = BTreeMap::new();
    for (k, s) in prof {
        let e = by.entry(k).or_insert((0, 0.0));
        e.0 += 1;
        e.1 += s;
    }
    let total: f64 = prof.iter().map(|(_, s)| s).sum();
    let mut rows: Vec<_> = by.into_iter().collect();
    rows.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap());
    eprintln!("== {label}: normal forward wall {:.2} ms, GPU {:.2} ms; profiled (one command buffer per dispatch) sum {:.2} ms", wall * 1e3, gpu * 1e3, total * 1e3);
    eprintln!("{:<14} {:>6} {:>10} {:>8}", "kernel", "n", "ms", "%");
    for (k, (n, s)) in rows {
        eprintln!(
            "{k:<14} {n:>6} {:>10.3} {:>7.1}%",
            s * 1e3,
            s / total * 100.0
        );
    }
}

#[test]
fn kernel_profile() {
    if !MetalBackend::is_available() {
        return;
    }
    let Some(path) = std::env::var_os("LLMARIO_TEST_GGUF") else {
        eprintln!("LLMARIO_TEST_GGUF not set; skipping");
        return;
    };
    let f = GgufFile::open(std::path::Path::new(&path)).unwrap();
    let mut gpu = MetalBackend::new(&f, 1024, 512).unwrap();
    let prompt: Vec<u32> = (0..512u32).map(|i| (i * 7919 + 13) % 150000).collect();

    // Warm up, then time a normal prefill and a normal decode step.
    gpu.forward(&prompt);
    gpu.clear();
    let t = Instant::now();
    gpu.forward(&prompt);
    let pre_wall = t.elapsed().as_secs_f64();
    let pre_gpu = gpu.last_gpu_secs();
    let t = Instant::now();
    gpu.forward(&[42]);
    let dec_wall = t.elapsed().as_secs_f64();
    let dec_gpu = gpu.last_gpu_secs();

    let dec = gpu.profile_forward(&[43]).unwrap();
    report("decode (pos ~513)", &dec, dec_wall, dec_gpu);
    gpu.clear();
    let pre = gpu.profile_forward(&prompt).unwrap();
    report("prefill 512", &pre, pre_wall, pre_gpu);
}
