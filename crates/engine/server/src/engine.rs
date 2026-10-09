//! The inference thread: owns the model, the KV cache and the scratch buffers; executes one
//! generation job at a time from a queue; streams events back over a channel.

use crate::ServeOptions;
use llmario_engine_chat::ChatTemplate;
use llmario_engine_core::ledger::{DeviceId, Ledger};
use llmario_engine_cpu::ThreadPool;
use llmario_engine_decode::{Sampler, SamplingParams, StopMatcher, StopResult};
use llmario_engine_formats::GgufFile;
use llmario_engine_model::forward::Scratch;
use llmario_engine_model::{ArchSpec, KvCache, Model};
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
        type Ready = anyhow::Result<(ArchSpec, Arc<Tokenizer>, Option<Arc<ChatTemplate>>, usize)>;
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Ready>();
        std::thread::Builder::new()
            .name("llmario-infer".into())
            .spawn(move || {
                let loaded = load_worker(&file, &opts, &plan, &ledger, stats2);
                match loaded {
                    Ok((mut worker, template)) => {
                        let spec = worker.model.spec.clone();
                        let tokenizer = worker.tokenizer.clone();
                        let threads = worker.pool.n_threads();
                        let _ = ready_tx.send(Ok((spec, tokenizer, template, threads)));
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
        let (spec, tokenizer, template, threads) = ready_rx
            .await
            .map_err(|_| anyhow::anyhow!("inference thread exited during load"))??;
        Ok(EngineRuntime {
            tx,
            spec,
            tokenizer,
            template,
            stats,
            kernels: llmario_engine_cpu::simd::kernels().name,
            threads,
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
    ledger: &Ledger,
    stats: Arc<Stats>,
) -> anyhow::Result<(Worker<'a>, Option<Arc<ChatTemplate>>)> {
    let t0 = Instant::now();
    let model = Model::load(file)?;
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
    let pool = ThreadPool::new(threads);
    let ctx = plan.ctx_per_slot as usize;
    let kv = KvCache::new(&model.spec, ctx);
    let prefill = Scratch::new(&model.spec, plan.n_batch as usize);
    let decode = Scratch::new(&model.spec, 1);
    ledger.update(DeviceId::Host, |d| {
        d.weights_mapped = file.tensor_bytes_total();
        d.kv_arena_reserved = KvCache::bytes(&model.spec, ctx);
        d.scratch_reserved =
            Scratch::bytes(&model.spec, plan.n_batch as usize) + Scratch::bytes(&model.spec, 1);
        d.runtime_fixed = plan.requested.runtime_fixed;
    });
    let mut worker = Worker {
        model,
        pool,
        kv,
        prefill,
        decode,
        n_batch: plan.n_batch as usize,
        cached: Vec::new(),
        tokenizer,
        stats,
    };
    // Warm-up: one token through every kernel so the first request pays nothing.
    let bos = worker.tokenizer.bos().unwrap_or(0);
    worker
        .model
        .forward(&worker.pool, &mut worker.kv, &[bos], &mut worker.decode);
    worker.kv.clear();
    tracing::info!(
        secs = t0.elapsed().as_secs_f32(),
        threads,
        kernels = llmario_engine_cpu::simd::kernels().name,
        "model loaded and warmed up"
    );
    Ok((worker, template))
}

struct Worker<'a> {
    model: Model<'a>,
    pool: ThreadPool,
    kv: KvCache,
    prefill: Scratch,
    decode: Scratch,
    n_batch: usize,
    /// Tokens whose K/V are in the cache (prefix reuse between requests).
    cached: Vec<u32>,
    tokenizer: Arc<Tokenizer>,
    stats: Arc<Stats>,
}

impl Worker<'_> {
    fn run(&mut self, job: Job) {
        self.stats.requests.fetch_add(1, Ordering::Relaxed);
        let max_ctx = self.kv.max_ctx;
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
        self.kv.truncate(common);
        self.cached.truncate(common);
        let t0 = Instant::now();
        let mut logits_owned: Vec<f32> = Vec::new();
        let rest = &job.prompt[common..];
        for chunk in rest.chunks(self.n_batch) {
            let logits = self
                .model
                .forward(&self.pool, &mut self.kv, chunk, &mut self.prefill);
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

        let mut sampler = Sampler::new(job.params.clone(), self.model.spec.n_vocab as usize);
        for &t in &job.prompt {
            sampler.accept(t);
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
            if self.kv.len + 1 >= max_ctx {
                reason = "length";
                break;
            }
            let logits = self
                .model
                .forward(&self.pool, &mut self.kv, &[next], &mut self.decode);
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
