//! Greedy-fidelity test against llama.cpp (Architecture §12, "Golden fidelity").
//!
//! Runs only when `LLMARIO_TEST_GGUF` lists GGUF files (comma-separated) and `llama-completion`
//! is on `PATH`. For each model and prompt it compares our greedy continuation (our tokenizer,
//! our forward pass, argmax) with llama.cpp's raw greedy completion of the same prompt, as text.
//! Numeric differences between two correct implementations can flip a near-tie, so the test
//! requires an exact match for the first 16 characters and reports (without failing) where the
//! continuations diverge after that.

use std::path::Path;
use std::process::Command;

const PROMPTS: &[&str] = &[
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "Once upon a time, in a small village by the sea,",
];
const N: usize = 32;
const MUST_MATCH: usize = 16;

fn llama_completion(model: &Path, prompt: &str, n: usize) -> Option<String> {
    let out = Command::new("llama-completion")
        .args([
            "-m",
            model.to_str()?,
            "-p",
            prompt,
            "-n",
            &n.to_string(),
            "--temp",
            "0",
            "-no-cnv",
            "--no-display-prompt",
            "-t",
            "8",
            "-ngl",
            "0",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).to_string())
}

#[test]
fn greedy_matches_llama_cpp() {
    let Ok(models) = std::env::var("LLMARIO_TEST_GGUF") else {
        eprintln!("LLMARIO_TEST_GGUF not set; skipping");
        return;
    };
    if Command::new("llama-completion")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("llama-completion not on PATH; skipping");
        return;
    }
    let engine = env!("CARGO_BIN_EXE_llmario-engine");
    // Devices to compare against llama.cpp's CPU output: `cpu` by default; add `auto` (Metal where
    // the backend covers the family, CPU otherwise) with LLMARIO_GOLDEN_DEVICES=cpu,auto.
    let devices: Vec<String> = std::env::var("LLMARIO_GOLDEN_DEVICES")
        .unwrap_or_else(|_| "cpu".into())
        .split(',')
        .filter(|d| !d.is_empty())
        .map(str::to_string)
        .collect();
    // The CPU leg uses the f32 reference kernels by default. The fast kernels quantise activations
    // to 8 bits (as llama.cpp's own CPU path does, with different rounding), which can flip a
    // greedy near-tie and make this a test of rounding luck rather than model correctness.
    // LLMARIO_GOLDEN_CPU_KERNELS=int8 (or neon, avx2) compares the fast kernels instead.
    let kernels = std::env::var("LLMARIO_GOLDEN_CPU_KERNELS").unwrap_or_else(|_| "scalar".into());
    for m in models.split(',').filter(|s| !s.is_empty()) {
        let model = Path::new(m);
        for prompt in PROMPTS {
            let Some(theirs) = llama_completion(model, prompt, N) else {
                eprintln!("llama-completion failed on {m}; skipping prompt");
                continue;
            };
            for device in &devices {
                // Our side: tokenize with the engine's tokenizer, then raw greedy run.
                let tok = Command::new(engine)
                    .args(["tokenize", "--model", m, "--text", prompt])
                    .output()
                    .expect("run tokenize");
                assert!(
                    tok.status.success(),
                    "tokenize failed: {}",
                    String::from_utf8_lossy(&tok.stderr)
                );
                let ids_line = String::from_utf8_lossy(&tok.stdout)
                    .lines()
                    .next()
                    .unwrap_or_default()
                    .to_string();
                let run = Command::new(engine)
                    .env("LLMARIO_CPU_KERNELS", &kernels)
                    .args([
                        "raw-run",
                        "--model",
                        m,
                        "--tokens",
                        &ids_line,
                        "--n",
                        &N.to_string(),
                        "--threads",
                        "8",
                        "--device",
                        device,
                    ])
                    .output()
                    .expect("run raw-run");
                assert!(
                    run.status.success(),
                    "raw-run failed: {}",
                    String::from_utf8_lossy(&run.stderr)
                );
                let stderr = String::from_utf8_lossy(&run.stderr);
                let ours = stderr
                    .lines()
                    .find_map(|l| l.strip_prefix("decoded: "))
                    .map(|s| serde_json::from_str::<String>(s).unwrap_or_default())
                    .unwrap_or_default();
                // Compare token-wise via whitespace-insensitive character prefix: llama.cpp prints
                // raw text; both sides decode the same ids to the same bytes.
                let theirs_t = theirs.trim_end_matches('\n');
                assert!(
                    !theirs_t.is_empty(),
                    "llama-completion printed nothing for {m} / {prompt:?}"
                );
                let ours_t = ours.trim_end_matches('\n');
                let common = theirs_t
                    .chars()
                    .zip(ours_t.chars())
                    .take_while(|(a, b)| a == b)
                    .count();
                let prefix_ok = common
                    >= MUST_MATCH
                        .min(theirs_t.chars().count())
                        .min(ours_t.chars().count());
                eprintln!(
                "[{} · {device}] {:?}\n  llama.cpp: {:?}\n  ours:      {:?}\n  common chars: {common}",
                model.file_name().unwrap().to_string_lossy(),
                prompt,
                theirs_t,
                ours_t
            );
                assert!(
                prefix_ok,
                "greedy continuation diverged within the first {MUST_MATCH} characters for {m} / {prompt:?} on {device}"
            );
            }
        }
    }
}
