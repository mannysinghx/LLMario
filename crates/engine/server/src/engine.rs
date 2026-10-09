//! The inference thread: owns the model, the KV cache and the scratch buffers; executes one
//! generation job at a time from a queue; streams events back over a channel.

use crate::{Device, ServeOptions};
use llmario_engine_chat::ChatTemplate;
use llmario_engine_core::ledger::{DeviceId, Ledger};
use llmario_engine_decode::{
    GrammarCache, GrammarProcessor, GrammarSpec, GrammarTokenizer, LazyTrigger, Sampler,
    SamplingParams, StopMatcher, StopResult,
};
use llmario_engine_formats::GgufFile;
use llmario_engine_model::{ArchSpec, CpuBackend, ModelBackend};
use llmario_engine_plan::Plan;
use llmario_engine_tokenizer::{Detokenizer, Tokenizer};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;

/// A generation request.
pub struct Job {
    pub prompt: Vec<u32>,
    pub params: SamplingParams,
    pub max_tokens: usize,
    pub stop: Vec<String>,
    /// Constrained decoding (`response_format`, tool grammars); `None` = free text.
    pub grammar: Option<GrammarSpec>,
    /// Text that must be generated before the grammar applies (e.g. `</think>`).
    pub lazy_trigger: Option<String>,
    /// Receives events; a closed receiver cancels the job.
    pub events: tokio::sync::mpsc::Sender<Event>,
}

#[derive(Debug, Clone)]
pub enum Event {
    /// A piece of generated text (already past the stop-string filter).
    Text(String),
    Done(Finish),
    Error(String),
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Finish {
    pub reason: &'static str,
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    pub prompt_ms: f64,
    pub decode_ms: f64,
}

/// Counters for `/engine/stats`.
#[derive(Default)]
pub struct Stats {
    pub requests: AtomicU64,
    pub prompt_tokens: AtomicU64,
    pub cached_prompt_tokens: AtomicU64,
    pub completion_tokens: AtomicU64,
    pub prompt_micros: AtomicU64,
    pub decode_micros: AtomicU64,
    pub cancelled: AtomicU64,
}

impl Stats {
    pub fn snapshot(&self) -> serde_json::Value {
        let pt = self.prompt_tokens.load(Ordering::Relaxed);
        let ct = self.completion_tokens.load(Ordering::Relaxed);
        let pm = self.prompt_micros.load(Ordering::Relaxed) as f64 / 1e6;
        let dm = self.decode_micros.load(Ordering::Relaxed) as f64 / 1e6;
        serde_json::json!({
            "requests": self.requests.load(Ordering::Relaxed),
            "cancelled": self.cancelled.load(Ordering::Relaxed),
            "prompt_tokens": pt,
            "cached_prompt_tokens": self.cached_prompt_tokens.load(Ordering::Relaxed),
            "completion_tokens": ct,
            "prefill_tok_s": if pm > 0.0 { pt as f64 / pm } else { 0.0 },
            "decode_tok_s": if dm > 0.0 { ct as f64 / dm } else { 0.0 },
        })
    }
}

/// Handle to the inference thread.
#[derive(Clone)]
pub struct EngineRuntime {
    tx: mpsc::Sender<Job>,
    pub spec: ArchSpec,
    pub tokenizer: Arc<Tokenizer>,
    pub template: Option<Arc<ChatTemplate>>,
    pub stats: Arc<Stats>,
    pub kernels: &'static str,
    pub threads: usize,
    /// `"cpu"` or `"metal"`.
    pub backend: &'static str,
}

impl EngineRuntime {
    /// Load the model on the inference thread and run a one-token warm-up before returning.
    pub async fn start(
        file: Arc<GgufFile>,
        opts: ServeOptions,
        plan: Plan,
        ledger: Arc<Ledger>,
    ) -> anyhow::Result<EngineRuntime> {
        let (tx, rx) = mpsc::channel::<Job>();
        let stats = Arc::new(Stats::default());
        let stats2 = stats.clone();
        type Ready = anyhow::Result<(
            ArchSpec,
            Arc<Tokenizer>,
            Option<Arc<ChatTemplate>>,
            usize,
            &'static str,
        )>;
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Ready>();
        std::thread::Builder::new()
            .name("llmario-infer".into())
            .spawn(move || {
                let loaded = load_worker(&file, &opts, &plan, &ledger, stats2);
                match loaded {
                    Ok((mut worker, template)) => {
                        let spec = worker.backend.spec().clone();
                        let tokenizer = worker.tokenizer.clone();
                        let threads = opts.threads.max(1);
                        let backend = worker.backend.name();
                        let _ = ready_tx.send(Ok((spec, tokenizer, template, threads, backend)));
                        for job in rx {
                            worker.run(job);
                        }
                        tracing::info!("inference thread: queue closed");
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "model load failed");
                        let _ = ready_tx.send(Err(e));
                    }
                }
            })?;
        let (spec, tokenizer, template, threads, backend) = ready_rx
            .await
            .map_err(|_| anyhow::anyhow!("inference thread exited during load"))??;
        Ok(EngineRuntime {
            tx,
            spec,
            tokenizer,
            template,
            stats,
            kernels: if backend == "metal" {
                "metal"
            } else {
                llmario_engine_cpu::simd::kernels().name
            },
            threads,
            backend,
        })
    }

    pub fn submit(&self, job: Job) -> anyhow::Result<()> {
        self.tx
            .send(job)
            .map_err(|_| anyhow::anyhow!("inference thread is gone"))
    }
}

#[allow(clippy::type_complexity)]
fn load_worker<'a>(
    file: &'a GgufFile,
    opts: &ServeOptions,
    plan: &Plan,
    ledger: &Arc<Ledger>,
    stats: Arc<Stats>,
) -> anyhow::Result<(Worker<'a>, Option<Arc<ChatTemplate>>)> {
    let ledger_handle = ledger.clone();
    let t0 = Instant::now();
    let tokenizer = Arc::new(Tokenizer::from_gguf(file)?);
    let template = match tokenizer.chat_template() {
        Some(src) => {
            let bos = tokenizer
                .bos()
                .map(|id| tokenizer.decode_with_special(&[id]));
            let eos = tokenizer
                .eos()
                .map(|id| tokenizer.decode_with_special(&[id]));
            match ChatTemplate::new(src, bos.as_deref(), eos.as_deref()) {
                Ok(t) => Some(Arc::new(t)),
                Err(e) => {
                    tracing::warn!(error = %e, "chat template failed to compile; chat requests will be refused");
                    None
                }
            }
        }
        None => None,
    };
    let threads = opts.threads.max(1);
    let ctx = plan.ctx_per_slot as usize;
    let backend = open_backend(file, opts, threads, ctx, plan.n_batch as usize)?;
    ledger.update(DeviceId::Host, |d| {
        d.weights_mapped = file.tensor_bytes_total();
        d.runtime_fixed = plan.requested.runtime_fixed;
    });
    match backend.name() {
        // Metal on unified memory: the weights stay a host mapping; the KV cache and scratch
        // are GPU buffers (and the weight views are wired by the residency sets when present).
        "metal" => {
            let kv = backend.kv_bytes(ctx);
            let reserved = backend.reserved_bytes();
            ledger.update(DeviceId::Gpu(0), |d| {
                d.kv_arena_reserved = kv;
                d.scratch_reserved = reserved.saturating_sub(kv);
            });
        }
        _ => ledger.update(DeviceId::Host, |d| {
            d.kv_arena_reserved = llmario_engine_model::KvCache::bytes(backend.spec(), ctx);
            d.scratch_reserved = backend.reserved_bytes() - d.kv_arena_reserved;
        }),
    }
    let grammar_env = Arc::new(
        GrammarTokenizer::from_tokenizer(&tokenizer)
            .map_err(|e| anyhow::anyhow!("grammar tokenizer: {e}"))?,
    );
    let mut worker = Worker {
        backend,
        cached: Vec::new(),
        tokenizer,
        stats,
        ledger: ledger_handle,
        grammar_env,
        grammar_cache: GrammarCache::new(64),
    };
    // Warm-up: one token through every kernel so the first request pays nothing.
    let bos = worker.tokenizer.bos().unwrap_or(0);
    worker.backend.forward(&[bos]);
    worker.backend.clear();
    if let Some(m) = crate::footprint::current() {
        ledger.record_measured_peak(m);
        let planned = plan.planned_peak.max(1);
        tracing::info!(
            measured_mib = m / (1024 * 1024),
            planned_mib = planned / (1024 * 1024),
            ratio = format!("{:.2}", planned as f64 / m as f64),
            "memory after load: measured vs planned"
        );
        if m > planned {
            tracing::warn!(
                "measured memory exceeds the plan; the plan's runtime constant is too small"
            );
        }
    }
    tracing::info!(
        secs = t0.elapsed().as_secs_f32(),
        threads,
        kernels = llmario_engine_cpu::simd::kernels().name,
        backend = worker.backend.name(),
        device = ?opts.device,
        "model loaded and warmed up"
    );
    Ok((worker, template))
}

/// Choose and construct the backend for `opts.device` (Architecture §7.8: `Auto` prefers the GPU
/// when one is usable, otherwise the CPU; `Metal` on a build or machine without it is an error).
fn open_backend<'a>(
    file: &'a GgufFile,
    opts: &ServeOptions,
    threads: usize,
    ctx: usize,
    n_batch: usize,
) -> anyhow::Result<Box<dyn ModelBackend + 'a>> {
    let want_metal = match opts.device {
        Device::Cpu => false,
        Device::Metal => true,
        Device::Auto => metal_available(),
    };
    if want_metal {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            match llmario_engine_metal::MetalBackend::new(file, ctx, n_batch) {
                Ok(backend) => return Ok(Box::new(backend)),
                // `Auto` falls back to the CPU (e.g. a family the Metal kernels do not cover
                // yet); an explicit `--device metal` is an error.
                Err(e) if opts.device == Device::Auto => {
                    tracing::warn!(error = %e, "Metal backend unavailable for this model; using the CPU backend");
                }
                Err(e) => return Err(e.into()),
            }
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        {
            anyhow::bail!("this build has no Metal backend (use --device cpu)");
        }
    }
    Ok(Box::new(CpuBackend::new(file, threads, ctx, n_batch)?))
}

/// True when the build has the Metal backend and a usable GPU is present.
pub fn metal_available() -> bool {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        llmario_engine_metal::MetalBackend::is_available()
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        false
    }
}

/// Bytes of K/V the backend holds for `ctx` tokens (f16 on Metal, f32 on the CPU).
trait KvBytes {
    fn kv_bytes(&self, ctx: usize) -> u64;
}

impl KvBytes for dyn ModelBackend + '_ {
    fn kv_bytes(&self, ctx: usize) -> u64 {
        let spec = self.spec();
        let per_elem = if self.name() == "metal" { 2.0 } else { 4.0 };
        spec.kv_bytes_per_token(per_elem) * ctx as u64
    }
}

struct Worker<'a> {
    backend: Box<dyn ModelBackend + 'a>,
    /// Tokens whose K/V are in the cache (prefix reuse between requests).
    cached: Vec<u32>,
    tokenizer: Arc<Tokenizer>,
    stats: Arc<Stats>,
    ledger: Arc<Ledger>,
    grammar_env: Arc<GrammarTokenizer>,
    grammar_cache: GrammarCache,
}

impl Worker<'_> {
    fn run(&mut self, job: Job) {
        self.stats.requests.fetch_add(1, Ordering::Relaxed);
        let max_ctx = self.backend.max_ctx();
        let n_batch = self.backend.max_batch();
        if job.prompt.is_empty() {
            let _ = job
                .events
                .blocking_send(Event::Error("empty prompt".into()));
            return;
        }
        if job.prompt.len() >= max_ctx {
            let _ = job.events.blocking_send(Event::Error(format!(
                "prompt has {} tokens but the context is {max_ctx}",
                job.prompt.len()
            )));
            return;
        }
        // Prefix reuse: keep the longest common prefix with the cache, but always recompute at
        // least the last prompt token so the logits are fresh.
        let common = self
            .cached
            .iter()
            .zip(&job.prompt)
            .take_while(|(a, b)| a == b)
            .count()
            .min(job.prompt.len() - 1);
        self.backend.truncate(common);
        // Backends with recurrent state may keep less than asked (they reset instead of
        // trimming); always recompute from what the backend actually kept.
        let common = self.backend.kv_len().min(common);
        self.cached.truncate(common);
        let t0 = Instant::now();
        let mut logits_owned: Vec<f32> = Vec::new();
        let rest = &job.prompt[common..];
        for chunk in rest.chunks(n_batch) {
            let logits = self.backend.forward(chunk);
            logits_owned.clear();
            logits_owned.extend_from_slice(logits);
            self.cached.extend_from_slice(chunk);
        }
        let prompt_ms = t0.elapsed().as_secs_f64() * 1e3;
        self.stats
            .prompt_tokens
            .fetch_add(rest.len() as u64, Ordering::Relaxed);
        self.stats
            .cached_prompt_tokens
            .fetch_add(common as u64, Ordering::Relaxed);
        self.stats
            .prompt_micros
            .fetch_add((prompt_ms * 1e3) as u64, Ordering::Relaxed);

        let mut sampler = Sampler::new(job.params.clone(), self.backend.spec().n_vocab as usize);
        for &t in &job.prompt {
            sampler.accept_prompt(t);
        }
        if let Some(spec) = &job.grammar {
            match self.grammar_cache.get_or_compile(spec, &self.grammar_env) {
                Ok(compiled) => {
                    let lazy = job.lazy_trigger.as_deref().map(LazyTrigger::new);
                    sampler.set_processor(Some(Box::new(GrammarProcessor::new(compiled, lazy))));
                }
                Err(e) => {
                    let _ = job
                        .events
                        .blocking_send(Event::Error(format!("grammar: {e}")));
                    return;
                }
            }
        }
        let mut detok = Detokenizer::new(&self.tokenizer);
        let mut stop = StopMatcher::new(job.stop.clone());
        let mut text_all = String::new();
        let mut emitted = 0usize;
        let mut n_gen = 0usize;
        let mut reason = "length";
        let t1 = Instant::now();
        loop {
            let next = sampler.sample(&logits_owned);
            n_gen += 1;
            if self.tokenizer.is_eog(next) {
                reason = "stop";
                break;
            }
            sampler.accept(next);
            let piece = detok.push(next);
            if !piece.is_empty() {
                text_all.push_str(&piece);
                let r = stop.push(&piece);
                let target = match r {
                    StopResult::Flush => text_all.len(),
                    StopResult::Hold(n) => text_all.len().saturating_sub(n),
                    StopResult::Matched { emit_up_to } => emit_up_to,
                };
                if target > emitted && text_all.is_char_boundary(target) {
                    let out = text_all[emitted..target].to_string();
                    emitted = target;
                    if job.events.blocking_send(Event::Text(out)).is_err() {
                        self.stats.cancelled.fetch_add(1, Ordering::Relaxed);
                        return;
                    }
                }
                if matches!(r, StopResult::Matched { .. }) {
                    reason = "stop";
                    break;
                }
            }
            if n_gen >= job.max_tokens {
                break;
            }
            if self.backend.kv_len() + 1 >= max_ctx {
                reason = "length";
                break;
            }
            let logits = self.backend.forward(&[next]);
            logits_owned.clear();
            logits_owned.extend_from_slice(logits);
            self.cached.push(next);
        }
        // Flush held-back bytes that turned out not to be a stop string.
        let tail = detok.flush();
        if !tail.is_empty() {
            text_all.push_str(&tail);
        }
        if reason == "length"
            && text_all.len() > emitted
            && text_all.is_char_boundary(text_all.len())
        {
            let out = text_all[emitted..].to_string();
            if !out.is_empty() && job.events.blocking_send(Event::Text(out)).is_err() {
                self.stats.cancelled.fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        let decode_ms = t1.elapsed().as_secs_f64() * 1e3;
        self.stats
            .completion_tokens
            .fetch_add(n_gen as u64, Ordering::Relaxed);
        self.stats
            .decode_micros
            .fetch_add((decode_ms * 1e3) as u64, Ordering::Relaxed);
        if let Some(m) = crate::footprint::current() {
            self.ledger.record_measured_peak(m);
        }
        let _ = job.events.blocking_send(Event::Done(Finish {
            reason,
            prompt_tokens: job.prompt.len(),
            cached_tokens: common,
            completion_tokens: n_gen,
            prompt_ms,
            decode_ms,
        }));
    }
}
