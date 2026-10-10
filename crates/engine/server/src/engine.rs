//! The inference thread: owns the model, the KV cache and the scratch buffers, and runs the step
//! loop of Architecture §9.1. Jobs from the queue take free slots (the slot whose cached tokens
//! share the longest prefix with the prompt first); every step runs one batched forward that
//! carries each decoding job's next token and then prompt chunks of the oldest prefilling jobs
//! within the token budget, samples per job and streams events back. With one slot this is the
//! sequential engine it replaced.

use crate::{Device, ServeOptions};
use llmario_engine_chat::ChatTemplate;
use llmario_engine_core::ledger::{DeviceId, Ledger};
use llmario_engine_decode::{
    GrammarCache, GrammarProcessor, GrammarSpec, GrammarTokenizer, LazyTrigger, Sampler,
    SamplingParams, StopMatcher, StopResult,
};
use llmario_engine_formats::GgufFile;
use llmario_engine_model::{ArchSpec, CpuBackend, CpuOptions, KvType, ModelBackend, SeqTokens};
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
    /// Batched forward steps run.
    pub steps: AtomicU64,
    /// Most jobs one step has carried (continuous batching at work when > 1).
    pub max_seqs_in_step: AtomicU64,
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
            "steps": self.steps.load(Ordering::Relaxed),
            "max_seqs_in_step": self.max_seqs_in_step.load(Ordering::Relaxed),
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
    /// Element type of the KV cache.
    pub kv_type: KvType,
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
            KvType,
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
                        let kv_type = worker.backend.kv_type();
                        let _ = ready_tx
                            .send(Ok((spec, tokenizer, template, threads, backend, kv_type)));
                        worker.serve(rx);
                        tracing::info!("inference thread: queue closed");
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "model load failed");
                        let _ = ready_tx.send(Err(e));
                    }
                }
            })?;
        let (spec, tokenizer, template, threads, backend, kv_type) = ready_rx
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
            kv_type,
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
    let backend = open_backend(file, opts, threads, plan)?;
    ledger.update(DeviceId::Host, |d| {
        d.weights_mapped = file.tensor_bytes_total();
        d.runtime_fixed = plan.requested.runtime_fixed;
    });
    // Metal on unified memory: the weights stay a host mapping; the KV blocks and scratch are GPU
    // buffers (and the weight views are wired by the residency sets when present).
    let kv_device = kv_device(backend.as_ref());
    let kv_reserved = backend.kv_reserved_bytes();
    let reserved = backend.reserved_bytes();
    ledger.update(kv_device, |d| {
        d.kv_arena_reserved = kv_reserved;
        d.kv_arena_in_use = backend.kv_in_use_bytes();
        d.scratch_reserved = reserved.saturating_sub(kv_reserved);
    });
    let grammar_env = Arc::new(
        GrammarTokenizer::from_tokenizer(&tokenizer)
            .map_err(|e| anyhow::anyhow!("grammar tokenizer: {e}"))?,
    );
    let mut worker = Worker {
        backend,
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

/// The ledger device that holds the backend's KV cache.
fn kv_device(b: &(dyn ModelBackend + '_)) -> DeviceId {
    if b.name() == "metal" {
        DeviceId::Gpu(0)
    } else {
        DeviceId::Host
    }
}

/// Choose and construct the backend for `opts.device` with the plan's context, batch, slots and
/// KV type (Architecture §7.8: `Auto` prefers the GPU when one is usable, otherwise the CPU;
/// `Metal` on a build or machine without it is an error).
fn open_backend<'a>(
    file: &'a GgufFile,
    opts: &ServeOptions,
    threads: usize,
    plan: &Plan,
) -> anyhow::Result<Box<dyn ModelBackend + 'a>> {
    let ctx = plan.ctx_per_slot as usize;
    let n_batch = plan.n_batch as usize;
    let n_seqs = plan.slots as usize;
    let kv_type = plan.kv_type();
    let want_metal = match opts.device {
        Device::Cpu => false,
        Device::Metal => true,
        Device::Auto => metal_available(),
    };
    if want_metal {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            let mo = llmario_engine_metal::MetalOptions {
                max_ctx: ctx,
                n_batch,
                n_seqs,
                kv_type,
            };
            match llmario_engine_metal::MetalBackend::with_options(file, mo) {
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
    Ok(Box::new(CpuBackend::with_options(
        file,
        CpuOptions {
            n_seqs,
            kv_type,
            ..CpuOptions::new(threads, ctx, n_batch)
        },
    )?))
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

struct Worker<'a> {
    backend: Box<dyn ModelBackend + 'a>,
    tokenizer: Arc<Tokenizer>,
    stats: Arc<Stats>,
    ledger: Arc<Ledger>,
    grammar_env: Arc<GrammarTokenizer>,
    grammar_cache: GrammarCache,
}

/// What a slot's KV cache holds (for prefix reuse between jobs) and whether a job owns it.
struct Slot {
    cached: Vec<u32>,
    busy: bool,
}

/// Events of one job not yet accepted by its channel (sent without blocking, so one slow client
/// never stalls the other slots).
struct Outbox {
    events: tokio::sync::mpsc::Sender<Event>,
    queue: std::collections::VecDeque<Event>,
    closed: bool,
}

impl Outbox {
    fn push(&mut self, e: Event) {
        self.queue.push_back(e);
        self.flush();
    }
    /// Send what the channel accepts; `false` once the receiver is gone.
    fn flush(&mut self) -> bool {
        use tokio::sync::mpsc::error::TrySendError;
        while let Some(e) = self.queue.pop_front() {
            match self.events.try_send(e) {
                Ok(()) => {}
                Err(TrySendError::Full(e)) => {
                    self.queue.push_front(e);
                    break;
                }
                Err(TrySendError::Closed(_)) => {
                    self.closed = true;
                    self.queue.clear();
                }
            }
        }
        !self.closed && !self.events.is_closed()
    }
}

enum Phase {
    /// `prompt[fed..]` still has to go through the model.
    Prefill,
    /// Waiting for the forward of `last` (the token sampled most recently).
    Decode { last: u32 },
}

/// One running job.
struct Active<'t> {
    slot: usize,
    prompt: Vec<u32>,
    /// Prompt tokens already in the slot's cache.
    fed: usize,
    /// Prompt tokens reused from the cache at admission.
    common: usize,
    phase: Phase,
    max_tokens: usize,
    sampler: Sampler,
    detok: Detokenizer<'t>,
    stop: StopMatcher,
    text_all: String,
    emitted: usize,
    n_gen: usize,
    out: Outbox,
    t0: Instant,
    t1: Option<Instant>,
    prompt_ms: f64,
}

/// How a job ended.
enum Done {
    Finished(&'static str),
    Cancelled,
    Failed(String),
}

impl Worker<'_> {
    /// The step loop: admit queued jobs into free slots, run one batched forward over every
    /// active job, sample and stream, repeat; block on the queue only when nothing is running.
    fn serve(&mut self, rx: mpsc::Receiver<Job>) {
        let tok = self.tokenizer.clone();
        let tok: &Tokenizer = &tok;
        let n_slots = self.backend.n_seqs().max(1);
        let mut slots: Vec<Slot> = (0..n_slots)
            .map(|_| Slot {
                cached: Vec::new(),
                busy: false,
            })
            .collect();
        let mut active: Vec<Active> = Vec::new();
        let mut waiting: std::collections::VecDeque<Job> = std::collections::VecDeque::new();
        // Finished jobs whose last events the client has not taken yet.
        let mut draining: Vec<Outbox> = Vec::new();
        let mut open = true;
        loop {
            draining.retain_mut(|o| o.flush() && !o.queue.is_empty());
            if active.is_empty() && waiting.is_empty() {
                if !open {
                    return;
                }
                let next = if draining.is_empty() {
                    rx.recv().ok()
                } else {
                    match rx.recv_timeout(std::time::Duration::from_millis(10)) {
                        Ok(j) => Some(j),
                        Err(mpsc::RecvTimeoutError::Timeout) => continue,
                        Err(mpsc::RecvTimeoutError::Disconnected) => None,
                    }
                };
                match next {
                    Some(j) => waiting.push_back(j),
                    None => {
                        open = false;
                        continue;
                    }
                }
            }
            loop {
                match rx.try_recv() {
                    Ok(j) => waiting.push_back(j),
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        open = false;
                        break;
                    }
                }
            }
            // Admit waiting jobs while slots are free.
            while slots.iter().any(|s| !s.busy) {
                let Some(job) = waiting.pop_front() else {
                    break;
                };
                if let Some(a) = self.admit(job, &mut slots, tok) {
                    active.push(a);
                }
            }
            if active.is_empty() {
                continue;
            }
            // Cancelled jobs (client gone) leave before the step.
            let mut i = 0;
            while i < active.len() {
                if !active[i].out.flush() {
                    let a = active.swap_remove(i);
                    self.finish(a, Done::Cancelled, &mut slots, &mut draining);
                } else {
                    i += 1;
                }
            }
            if active.is_empty() {
                continue;
            }
            self.step(&mut active, &mut slots, &mut draining);
        }
    }

    /// Validate `job`, give it the free slot sharing the longest prefix with its prompt, reuse
    /// that prefix, and set up sampling. Jobs that fail validation are answered here.
    fn admit<'t>(
        &mut self,
        job: Job,
        slots: &mut [Slot],
        tok: &'t Tokenizer,
    ) -> Option<Active<'t>> {
        self.stats.requests.fetch_add(1, Ordering::Relaxed);
        let max_ctx = self.backend.max_ctx();
        if job.prompt.is_empty() {
            let _ = job.events.try_send(Event::Error("empty prompt".into()));
            return None;
        }
        if job.prompt.len() >= max_ctx {
            let _ = job.events.try_send(Event::Error(format!(
                "prompt has {} tokens but the context is {max_ctx}",
                job.prompt.len()
            )));
            return None;
        }
        let prefix = |c: &[u32]| {
            c.iter()
                .zip(&job.prompt)
                .take_while(|(a, b)| a == b)
                .count()
        };
        let slot = (0..slots.len())
            .filter(|&i| !slots[i].busy)
            .max_by_key(|&i| (prefix(&slots[i].cached), std::cmp::Reverse(i)))
            .expect("a free slot");
        // Keep the longest common prefix, but always recompute at least the last prompt token so
        // its logits are fresh.
        let common = prefix(&slots[slot].cached).min(job.prompt.len() - 1);
        self.backend.truncate_seq(slot, common);
        // Backends with recurrent state may keep less than asked (they reset instead of
        // trimming); always continue from what the backend actually kept.
        let common = self.backend.seq_len(slot).min(common);
        slots[slot].cached.truncate(common);
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
                    let _ = job.events.try_send(Event::Error(format!("grammar: {e}")));
                    return None;
                }
            }
        }
        slots[slot].busy = true;
        self.stats
            .cached_prompt_tokens
            .fetch_add(common as u64, Ordering::Relaxed);
        Some(Active {
            slot,
            fed: common,
            common,
            phase: Phase::Prefill,
            max_tokens: job.max_tokens,
            sampler,
            // Special tokens are rendered: tool-call and channel markers are control tokens in
            // several families, and the output parser downstream needs to see them.
            detok: Detokenizer::with_special(tok),
            stop: StopMatcher::new(job.stop.clone()),
            text_all: String::new(),
            emitted: 0,
            n_gen: 0,
            out: Outbox {
                events: job.events,
                queue: Default::default(),
                closed: false,
            },
            prompt: job.prompt,
            t0: Instant::now(),
            t1: None,
            prompt_ms: 0.0,
        })
    }

    /// One batched forward over the active jobs, then sampling and streaming per job.
    fn step(&mut self, active: &mut Vec<Active>, slots: &mut [Slot], draining: &mut Vec<Outbox>) {
        let budget = self.backend.max_batch().max(1);
        let max_ctx = self.backend.max_ctx();
        // Decode tokens first, then prompt chunks of the oldest prefilling jobs (`active` is in
        // admission order).
        let mut plan: Vec<(usize, usize)> = Vec::new(); // (active index, tokens this step)
        let mut used = 0;
        for (i, a) in active.iter().enumerate() {
            if matches!(a.phase, Phase::Decode { .. }) && used < budget {
                plan.push((i, 1));
                used += 1;
            }
        }
        for (i, a) in active.iter().enumerate() {
            if matches!(a.phase, Phase::Prefill) && used < budget {
                let n = (a.prompt.len() - a.fed).min(budget - used);
                plan.push((i, n));
                used += n;
            }
        }
        let lasts: Vec<u32> = plan
            .iter()
            .map(|&(i, _)| match active[i].phase {
                Phase::Decode { last } => last,
                Phase::Prefill => 0,
            })
            .collect();
        let batch: Vec<SeqTokens> = plan
            .iter()
            .zip(&lasts)
            .map(|(&(i, n), last)| {
                let a = &active[i];
                SeqTokens {
                    seq: a.slot,
                    tokens: match a.phase {
                        Phase::Decode { .. } => std::slice::from_ref(last),
                        Phase::Prefill => &a.prompt[a.fed..a.fed + n],
                    },
                }
            })
            .collect();
        let vocab = self.backend.spec().n_vocab as usize;
        let mut done: Vec<(usize, Done)> = Vec::new();
        self.stats.steps.fetch_add(1, Ordering::Relaxed);
        self.stats
            .max_seqs_in_step
            .fetch_max(batch.len() as u64, Ordering::Relaxed);
        let result = self.backend.forward_batch(&batch);
        let logits = match result {
            Ok(l) => l,
            Err(full) => {
                drop(batch);
                // The pool is shared: first drop what idle slots keep for prefix reuse, then, if
                // that is not enough, end the most recently admitted job.
                let idle: Vec<usize> = (0..slots.len()).filter(|&s| !slots[s].busy).collect();
                if idle.iter().any(|&s| !slots[s].cached.is_empty()) {
                    for s in idle {
                        self.backend.clear_seq(s);
                        slots[s].cached.clear();
                    }
                    return;
                }
                let a = active.pop().expect("an active job");
                self.finish(
                    a,
                    Done::Failed(format!("{full}; retry when fewer requests are running")),
                    slots,
                    draining,
                );
                return;
            }
        };
        for (k, &(i, n)) in plan.iter().enumerate() {
            let row = &logits[k * vocab..(k + 1) * vocab];
            let a = &mut active[i];
            match a.phase {
                Phase::Prefill => {
                    slots[a.slot]
                        .cached
                        .extend_from_slice(&a.prompt[a.fed..a.fed + n]);
                    a.fed += n;
                    if a.fed < a.prompt.len() {
                        continue;
                    }
                    a.prompt_ms = a.t0.elapsed().as_secs_f64() * 1e3;
                    a.t1 = Some(Instant::now());
                }
                Phase::Decode { last } => slots[a.slot].cached.push(last),
            }
            if let Some(d) =
                sample_and_emit(a, row, &self.tokenizer, max_ctx, slots[a.slot].cached.len())
            {
                done.push((i, d));
            }
        }
        // Remove finished jobs (highest index first so the others keep their positions).
        done.sort_by_key(|(i, _)| std::cmp::Reverse(*i));
        for (i, d) in done {
            let a = active.remove(i);
            self.finish(a, d, slots, draining);
        }
    }

    /// Account and answer a job that ended; its slot becomes free (its cache stays for reuse).
    fn finish(&mut self, mut a: Active, how: Done, slots: &mut [Slot], draining: &mut Vec<Outbox>) {
        slots[a.slot].busy = false;
        let prompt_fed = a.fed - a.common;
        self.stats
            .prompt_tokens
            .fetch_add(prompt_fed as u64, Ordering::Relaxed);
        let prompt_ms = if a.t1.is_some() {
            a.prompt_ms
        } else {
            a.t0.elapsed().as_secs_f64() * 1e3
        };
        self.stats
            .prompt_micros
            .fetch_add((prompt_ms * 1e3) as u64, Ordering::Relaxed);
        let decode_ms = a.t1.map(|t| t.elapsed().as_secs_f64() * 1e3).unwrap_or(0.0);
        self.stats
            .completion_tokens
            .fetch_add(a.n_gen as u64, Ordering::Relaxed);
        self.stats
            .decode_micros
            .fetch_add((decode_ms * 1e3) as u64, Ordering::Relaxed);
        if let Some(m) = crate::footprint::current() {
            self.ledger.record_measured_peak(m);
        }
        let in_use = self.backend.kv_in_use_bytes();
        self.ledger.update(kv_device(self.backend.as_ref()), |d| {
            d.kv_arena_in_use = in_use
        });
        match how {
            Done::Cancelled => {
                self.stats.cancelled.fetch_add(1, Ordering::Relaxed);
                return;
            }
            Done::Failed(e) => a.out.push(Event::Error(e)),
            Done::Finished(reason) => {
                // Flush held-back bytes that turned out not to be a stop string.
                let tail = a.detok.flush();
                if !tail.is_empty() {
                    a.text_all.push_str(&tail);
                }
                if reason == "length"
                    && a.text_all.len() > a.emitted
                    && a.text_all.is_char_boundary(a.text_all.len())
                {
                    let out = a.text_all[a.emitted..].to_string();
                    if !out.is_empty() {
                        a.out.push(Event::Text(out));
                    }
                }
                a.out.push(Event::Done(Finish {
                    reason,
                    prompt_tokens: a.prompt.len(),
                    cached_tokens: a.common,
                    completion_tokens: a.n_gen,
                    prompt_ms,
                    decode_ms,
                }));
            }
        }
        if a.out.flush() && !a.out.queue.is_empty() {
            draining.push(a.out);
        } else if a.out.closed {
            self.stats.cancelled.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Sample the next token of `a` from `logits`, run it through the detokenizer and the stop
/// matcher, stream what is final, and decide whether the job ends (`cached` = tokens in its
/// slot's cache).
fn sample_and_emit(
    a: &mut Active,
    logits: &[f32],
    tok: &Tokenizer,
    max_ctx: usize,
    cached: usize,
) -> Option<Done> {
    let next = a.sampler.sample(logits);
    a.n_gen += 1;
    if tok.is_eog(next) {
        return Some(Done::Finished("stop"));
    }
    a.sampler.accept(next);
    let piece = a.detok.push(next);
    if !piece.is_empty() {
        a.text_all.push_str(&piece);
        let r = a.stop.push(&piece);
        let target = match r {
            StopResult::Flush => a.text_all.len(),
            StopResult::Hold(n) => a.text_all.len().saturating_sub(n),
            StopResult::Matched { emit_up_to } => emit_up_to,
        };
        if target > a.emitted && a.text_all.is_char_boundary(target) {
            let out = a.text_all[a.emitted..target].to_string();
            a.emitted = target;
            a.out.push(Event::Text(out));
            if a.out.closed {
                return Some(Done::Cancelled);
            }
        }
        if matches!(r, StopResult::Matched { .. }) {
            return Some(Done::Finished("stop"));
        }
    }
    if a.n_gen >= a.max_tokens {
        return Some(Done::Finished("length"));
    }
    if cached + 1 >= max_ctx {
        return Some(Done::Finished("length"));
    }
    a.phase = Phase::Decode { last: next };
    None
}
