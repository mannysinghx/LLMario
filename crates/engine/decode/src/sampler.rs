//! The token sampler: llama.cpp's default sampler chain over one logits vector.
//!
//! Per call: copy logits → [`LogitProcessor`] (grammar hook) → penalties → top-k → top-p → min-p →
//! temperature → softmax → multinomial draw; or argmax when greedy. Everything is O(n_vocab) plus a
//! partial selection for top-k; the only full sort happens when top-k is disabled but top-p or a
//! sorted min-p pass still needs descending order.

use std::cmp::Ordering;
use std::collections::VecDeque;

use crate::params::SamplingParams;
use crate::rng::Xoshiro256StarStar;
use crate::Token;

/// Hook applied to the raw logits before penalties: constrained decoding (grammar masks), logit
/// bias, or anything else that rewrites logits in place. `accept` is called for every token the
/// sampler accepts so a stateful processor (a grammar) can advance.
pub trait LogitProcessor: Send {
    /// Rewrite `logits` in place (set entries to `f32::NEG_INFINITY` to forbid them).
    fn process(&mut self, logits: &mut [f32]);
    /// Advance internal state after `token` was accepted.
    fn accept(&mut self, token: Token);
}

#[derive(Clone, Copy, Debug)]
struct Cand {
    logit: f32,
    id: Token,
}

/// Descending by logit, ascending by token id on ties, so filtering is fully deterministic.
fn cand_desc(a: &Cand, b: &Cand) -> Ordering {
    b.logit.total_cmp(&a.logit).then(a.id.cmp(&b.id))
}

/// Index of the maximum logit (first one on ties). Panics on an empty slice.
fn argmax(logits: &[f32]) -> Token {
    let mut best = 0usize;
    for (i, &l) in logits.iter().enumerate().skip(1) {
        if l > logits[best] {
            best = i;
        }
    }
    best as Token
}

/// Largest `top_k` for which the heap scan is used instead of a partial selection.
const HEAP_TOP_K_MAX: usize = 1024;

/// Min-heap ordering: the root is the *worst* candidate under `cand_desc`.
struct HeapCand(Cand);

impl PartialEq for HeapCand {
    fn eq(&self, other: &Self) -> bool {
        cand_desc(&self.0, &other.0) == Ordering::Equal
    }
}
impl Eq for HeapCand {}
impl PartialOrd for HeapCand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapCand {
    fn cmp(&self, other: &Self) -> Ordering {
        // `cand_desc` orders better candidates first (`Less`), so under it the worst candidate
        // is the greatest: `BinaryHeap` (a max-heap) then keeps the worst at the root.
        cand_desc(&self.0, &other.0)
    }
}

/// Writes the `k` best candidates of `logits` (unordered) into `out`. `k < logits.len()`.
fn top_k_heap(logits: &[f32], k: usize, out: &mut Vec<Cand>) {
    let mut heap: std::collections::BinaryHeap<HeapCand> =
        std::collections::BinaryHeap::with_capacity(k + 1);
    let mut iter = logits.iter().enumerate();
    for (i, &logit) in iter.by_ref().take(k) {
        heap.push(HeapCand(Cand {
            logit,
            id: i as Token,
        }));
    }
    // Root = current k-th best. An entry only enters if it beats it (ties go to the lower id,
    // which is already in the heap, so `>` on the logit alone is the common-case rejection).
    let mut worst = heap.peek().map(|h| h.0.logit).unwrap_or(f32::NEG_INFINITY);
    for (i, &logit) in iter {
        if logit > worst {
            heap.pop();
            heap.push(HeapCand(Cand {
                logit,
                id: i as Token,
            }));
            worst = heap.peek().map(|h| h.0.logit).unwrap_or(f32::NEG_INFINITY);
        }
    }
    out.clear();
    out.extend(heap.into_iter().map(|h| h.0));
}

/// Softmax of `cands` into `probs`. Returns `false` when the distribution is degenerate (every
/// logit is `-inf` or NaN), in which case `probs` must not be used.
fn softmax_into(cands: &[Cand], probs: &mut Vec<f32>) -> bool {
    probs.clear();
    let max = cands
        .iter()
        .map(|c| c.logit)
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return false;
    }
    let mut sum = 0.0f32;
    probs.extend(cands.iter().map(|c| {
        let e = (c.logit - max).exp();
        sum += e;
        e
    }));
    if !sum.is_finite() || sum <= 0.0 {
        return false;
    }
    let inv = 1.0 / sum;
    for p in probs.iter_mut() {
        *p *= inv;
    }
    true
}

/// Draws a seed from OS-backed entropy without an external crate: `RandomState` keys are
/// randomly initialised per process/thread by the standard library.
fn entropy_seed() -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::BuildHasher;
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    RandomState::new().hash_one(nanos) ^ nanos
}

/// Stateful sampler for one decode slot.
///
/// Holds the penalty window (the last `repeat_last_n` accepted tokens), the PRNG, an optional
/// [`LogitProcessor`], and reusable scratch buffers sized to the vocabulary. Not `Clone`: the
/// processor is a boxed trait object.
pub struct Sampler {
    params: SamplingParams,
    n_vocab: usize,
    seed: u64,
    rng: Xoshiro256StarStar,
    processor: Option<Box<dyn LogitProcessor>>,
    /// Ring of the last `repeat_last_n` accepted tokens.
    recent: VecDeque<Token>,
    /// Occurrence count of each token in `recent` (indexed by token id).
    counts: Vec<u32>,
    /// Tokens with `counts > 0`, in arbitrary order (the set the penalties iterate over).
    active: Vec<Token>,
    /// Position of each token inside `active`, or `u32::MAX` when absent.
    active_pos: Vec<u32>,
    scratch: Vec<f32>,
    cands: Vec<Cand>,
    probs: Vec<f32>,
}

impl std::fmt::Debug for Sampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sampler")
            .field("params", &self.params)
            .field("n_vocab", &self.n_vocab)
            .field("seed", &self.seed)
            .field("has_processor", &self.processor.is_some())
            .field("window_len", &self.recent.len())
            .finish()
    }
}

impl Sampler {
    /// Creates a sampler for a vocabulary of `n_vocab` entries. When `params.seed` is `None` a
    /// seed is drawn from entropy; [`Sampler::seed`] reports the seed in use either way.
    pub fn new(params: SamplingParams, n_vocab: usize) -> Self {
        let seed = params.seed.unwrap_or_else(entropy_seed);
        Self {
            rng: Xoshiro256StarStar::seed_from_u64(seed),
            seed,
            processor: None,
            recent: VecDeque::with_capacity(params.repeat_last_n.min(1 << 16)),
            counts: vec![0; n_vocab],
            active: Vec::new(),
            active_pos: vec![u32::MAX; n_vocab],
            scratch: Vec::with_capacity(n_vocab),
            cands: Vec::with_capacity(n_vocab),
            probs: Vec::new(),
            params,
            n_vocab,
        }
    }

    /// Attaches a logit processor (grammar / logit bias) applied before penalties.
    pub fn with_processor(mut self, processor: Box<dyn LogitProcessor>) -> Self {
        self.processor = Some(processor);
        self
    }

    /// Replaces (or removes, with `None`) the logit processor.
    pub fn set_processor(&mut self, processor: Option<Box<dyn LogitProcessor>>) {
        self.processor = processor;
    }

    /// Detaches and returns the logit processor, if any.
    pub fn take_processor(&mut self) -> Option<Box<dyn LogitProcessor>> {
        self.processor.take()
    }

    /// The parameters this sampler was built with.
    pub fn params(&self) -> &SamplingParams {
        &self.params
    }

    /// Vocabulary size the logits must have.
    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    /// The PRNG seed in use (the requested one, or the entropy-drawn one).
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Tokens currently inside the penalty window, oldest first.
    pub fn window(&self) -> impl Iterator<Item = Token> + '_ {
        self.recent.iter().copied()
    }

    /// Samples one token from `logits` (length must equal `n_vocab`). Does not record the token;
    /// call [`Sampler::accept`] once the engine commits it.
    pub fn sample(&mut self, logits: &[f32]) -> Token {
        assert_eq!(
            logits.len(),
            self.n_vocab,
            "logits length must equal the sampler's n_vocab"
        );
        self.scratch.clear();
        self.scratch.extend_from_slice(logits);
        if let Some(p) = self.processor.as_mut() {
            p.process(&mut self.scratch);
        }
        apply_penalties(&self.params, &self.active, &self.counts, &mut self.scratch);

        if self.params.is_greedy() {
            return argmax(&self.scratch);
        }

        let mut sorted = false;
        let k = self.params.top_k;
        if k > 0 && (k as usize) < self.scratch.len() {
            // top-k. Small k: one pass over the logits with a bounded min-heap (one comparison
            // per entry once the heap is full; the candidate vector is never materialised).
            // Large k: build all candidates and partially select.
            let k = k as usize;
            if k <= HEAP_TOP_K_MAX {
                top_k_heap(&self.scratch, k, &mut self.cands);
            } else {
                self.cands.clear();
                self.cands
                    .extend(self.scratch.iter().enumerate().map(|(i, &logit)| Cand {
                        logit,
                        id: i as Token,
                    }));
                self.cands.select_nth_unstable_by(k - 1, cand_desc);
                self.cands.truncate(k);
            }
            self.cands.sort_unstable_by(cand_desc);
            sorted = true;
        } else {
            self.cands.clear();
            self.cands
                .extend(self.scratch.iter().enumerate().map(|(i, &logit)| Cand {
                    logit,
                    id: i as Token,
                }));
        }

        // top-p: nucleus over the candidates that survived top-k, at temperature 1 (llama.cpp
        // applies temperature after the truncation samplers).
        let top_p = self.params.top_p;
        if top_p < 1.0 && self.cands.len() > 1 {
            if !sorted {
                self.cands.sort_unstable_by(cand_desc);
                sorted = true;
            }
            if softmax_into(&self.cands, &mut self.probs) {
                let mut cum = 0.0f32;
                let mut last = self.cands.len();
                for (i, p) in self.probs.iter().enumerate() {
                    cum += p;
                    if cum >= top_p {
                        last = i + 1;
                        break;
                    }
                }
                self.cands.truncate(last);
            }
        }

        // min-p: relative to the maximum logit; `p_i >= min_p * p_max  <=>  l_i >= l_max + ln(min_p)`.
        let min_p = self.params.min_p;
        if min_p > 0.0 && self.cands.len() > 1 {
            let max = if sorted {
                self.cands[0].logit
            } else {
                self.cands
                    .iter()
                    .map(|c| c.logit)
                    .fold(f32::NEG_INFINITY, f32::max)
            };
            if max.is_finite() {
                let min_logit = max + min_p.ln();
                if sorted {
                    let mut keep = 1;
                    while keep < self.cands.len() && self.cands[keep].logit >= min_logit {
                        keep += 1;
                    }
                    self.cands.truncate(keep);
                } else {
                    self.cands.retain(|c| c.logit >= min_logit);
                }
            }
        }

        // temperature
        let temp = self.params.temperature;
        if temp != 1.0 {
            for c in self.cands.iter_mut() {
                c.logit /= temp;
            }
        }

        // softmax + multinomial draw
        if !softmax_into(&self.cands, &mut self.probs) {
            // Degenerate distribution (everything masked): fall back to the best candidate.
            return self
                .cands
                .iter()
                .min_by(|a, b| cand_desc(a, b))
                .map(|c| c.id)
                .unwrap_or(0);
        }
        let u = self.rng.next_f64();
        let mut cum = 0.0f64;
        for (c, &p) in self.cands.iter().zip(self.probs.iter()) {
            cum += p as f64;
            if u < cum {
                return c.id;
            }
        }
        // Rounding left `u` above the accumulated mass: the last candidate with non-zero mass.
        self.cands
            .iter()
            .zip(self.probs.iter())
            .rev()
            .find(|(_, &p)| p > 0.0)
            .map(|(c, _)| c.id)
            .unwrap_or(self.cands[0].id)
    }

    /// Records `token` as generated: advances the logit processor and the penalty window.
    pub fn accept(&mut self, token: Token) {
        if let Some(p) = self.processor.as_mut() {
            p.accept(token);
        }
        self.accept_prompt(token);
    }

    /// Records a *prompt* token: it enters the penalty window (llama-server penalises the prompt
    /// too) but is not shown to the logit processor, whose grammar only covers generated text.
    pub fn accept_prompt(&mut self, token: Token) {
        let n = self.params.repeat_last_n;
        if n == 0 || (token as usize) >= self.n_vocab {
            return;
        }
        if self.recent.len() >= n {
            if let Some(old) = self.recent.pop_front() {
                self.window_remove(old);
            }
        }
        self.recent.push_back(token);
        self.window_add(token);
    }

    /// Clears the penalty window and reseeds the PRNG so the same sequence of draws repeats.
    /// The logit processor, if any, is left untouched (replace it with
    /// [`Sampler::set_processor`]).
    pub fn reset(&mut self) {
        for &t in &self.active {
            self.counts[t as usize] = 0;
            self.active_pos[t as usize] = u32::MAX;
        }
        self.active.clear();
        self.recent.clear();
        self.rng = Xoshiro256StarStar::seed_from_u64(self.seed);
    }

    /// Applies the repetition / frequency / presence penalties to `logits` in place using the
    /// current window. Exposed so a device-side sampler can reuse the host-side window.
    pub fn apply_penalties(&self, logits: &mut [f32]) {
        apply_penalties(&self.params, &self.active, &self.counts, logits);
    }

    /// Log-probabilities of the `n` most likely tokens after temperature scaling (no processor,
    /// penalties or truncation: the model's own distribution, as the API's `logprobs` reports).
    /// Sorted descending; ties broken by token id. With greedy decoding the raw logits are used.
    pub fn top_logprobs(&self, logits: &[f32], n: usize) -> Vec<(Token, f32)> {
        if n == 0 || logits.is_empty() {
            return Vec::new();
        }
        let scale = self.logprob_scale();
        let log_z = log_sum_exp(logits, scale);
        let mut cands: Vec<Cand> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| Cand {
                logit: l * scale,
                id: i as Token,
            })
            .collect();
        if n < cands.len() {
            cands.select_nth_unstable_by(n - 1, cand_desc);
            cands.truncate(n);
        }
        cands.sort_unstable_by(cand_desc);
        cands.into_iter().map(|c| (c.id, c.logit - log_z)).collect()
    }

    /// Log-probability of one token under the same distribution as [`Sampler::top_logprobs`].
    pub fn token_logprob(&self, logits: &[f32], token: Token) -> f32 {
        let scale = self.logprob_scale();
        logits[token as usize] * scale - log_sum_exp(logits, scale)
    }

    fn logprob_scale(&self) -> f32 {
        if self.params.is_greedy() {
            1.0
        } else {
            1.0 / self.params.temperature
        }
    }

    fn window_add(&mut self, token: Token) {
        let t = token as usize;
        self.counts[t] += 1;
        if self.counts[t] == 1 {
            self.active_pos[t] = self.active.len() as u32;
            self.active.push(token);
        }
    }

    fn window_remove(&mut self, token: Token) {
        let t = token as usize;
        self.counts[t] -= 1;
        if self.counts[t] == 0 {
            let pos = self.active_pos[t] as usize;
            self.active.swap_remove(pos);
            if pos < self.active.len() {
                self.active_pos[self.active[pos] as usize] = pos as u32;
            }
            self.active_pos[t] = u32::MAX;
        }
    }
}

/// `log(sum(exp(l * scale)))` with the max subtracted for stability. `-inf` when every logit is
/// `-inf`.
fn log_sum_exp(logits: &[f32], scale: f32) -> f32 {
    let max = logits
        .iter()
        .map(|&l| l * scale)
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return max;
    }
    let sum: f32 = logits.iter().map(|&l| (l * scale - max).exp()).sum();
    max + sum.ln()
}

/// llama.cpp `llama_sampler_penalties`: for every token in the window, divide a positive logit
/// by `repeat_penalty` (multiply a non-positive one), then subtract `count * frequency_penalty +
/// presence_penalty`.
fn apply_penalties(params: &SamplingParams, active: &[Token], counts: &[u32], logits: &mut [f32]) {
    if !params.penalties_active() {
        return;
    }
    let repeat = params.repeat_penalty;
    let freq = params.frequency_penalty;
    let present = params.presence_penalty;
    for &t in active {
        let t = t as usize;
        let count = counts[t];
        if count == 0 {
            continue;
        }
        let l = &mut logits[t];
        if *l <= 0.0 {
            *l *= repeat;
        } else {
            *l /= repeat;
        }
        *l -= count as f32 * freq + present;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(f: impl FnOnce(&mut SamplingParams)) -> SamplingParams {
        let mut p = SamplingParams {
            seed: Some(1234),
            ..Default::default()
        };
        f(&mut p);
        p
    }

    /// Everything off: pure multinomial sampling at temperature 1.
    fn plain() -> SamplingParams {
        params(|p| {
            p.top_k = 0;
            p.top_p = 1.0;
            p.min_p = 0.0;
        })
    }

    fn histogram(sampler: &mut Sampler, logits: &[f32], draws: usize) -> Vec<usize> {
        let mut h = vec![0usize; logits.len()];
        for _ in 0..draws {
            h[sampler.sample(logits) as usize] += 1;
        }
        h
    }

    #[test]
    fn same_seed_same_sequence() {
        let logits: Vec<f32> = (0..100).map(|i| ((i * 7919) % 13) as f32 * 0.3).collect();
        let mut a = Sampler::new(params(|p| p.seed = Some(99)), 100);
        let mut b = Sampler::new(params(|p| p.seed = Some(99)), 100);
        let mut c = Sampler::new(params(|p| p.seed = Some(100)), 100);
        let seq_a: Vec<Token> = (0..64)
            .map(|_| {
                let t = a.sample(&logits);
                a.accept(t);
                t
            })
            .collect();
        let seq_b: Vec<Token> = (0..64)
            .map(|_| {
                let t = b.sample(&logits);
                b.accept(t);
                t
            })
            .collect();
        let seq_c: Vec<Token> = (0..64)
            .map(|_| {
                let t = c.sample(&logits);
                c.accept(t);
                t
            })
            .collect();
        assert_eq!(seq_a, seq_b);
        assert_ne!(
            seq_a, seq_c,
            "different seeds should (overwhelmingly) differ"
        );
        // reset() replays the same stream.
        a.reset();
        let replay: Vec<Token> = (0..64)
            .map(|_| {
                let t = a.sample(&logits);
                a.accept(t);
                t
            })
            .collect();
        assert_eq!(replay, seq_a);
    }

    #[test]
    fn entropy_seed_is_reported_and_varies() {
        let a = Sampler::new(SamplingParams::default(), 8);
        let b = Sampler::new(SamplingParams::default(), 8);
        assert_eq!(a.params().seed, None);
        // Two entropy draws colliding is astronomically unlikely.
        assert_ne!(a.seed(), b.seed());
        let c = Sampler::new(params(|p| p.seed = Some(5)), 8);
        assert_eq!(c.seed(), 5);
    }

    #[test]
    fn greedy_picks_argmax_and_ignores_rng() {
        let logits = [0.1, 3.0, -2.0, 3.0, 2.9];
        let mut s = Sampler::new(params(|p| p.temperature = 0.0), 5);
        for _ in 0..10 {
            assert_eq!(s.sample(&logits), 1, "first maximal index wins ties");
        }
        let mut s = Sampler::new(params(|p| p.temperature = -1.0), 5);
        assert_eq!(s.sample(&logits), 1);
    }

    #[test]
    fn top_k_keeps_only_k_best() {
        // Token 9 is the best, then 8, 7, ...; with top_k = 3 only {9, 8, 7} may ever be drawn.
        let logits: Vec<f32> = (0..10).map(|i| i as f32 * 0.5).collect();
        let mut s = Sampler::new(
            params(|p| {
                p.top_k = 3;
                p.top_p = 1.0;
                p.min_p = 0.0;
            }),
            10,
        );
        let h = histogram(&mut s, &logits, 2000);
        assert_eq!(h[..7].iter().sum::<usize>(), 0, "{h:?}");
        assert!(h[7] > 0 && h[8] > 0 && h[9] > 0, "{h:?}");
        assert!(h[9] > h[8] && h[8] > h[7], "{h:?}");
    }

    #[test]
    fn top_p_keeps_smallest_prefix_reaching_p() {
        // probs = [0.5, 0.25, 0.125, 0.125] (logits ln(8), ln(4), ln(2), ln(2)).
        let logits = [8f32.ln(), 4f32.ln(), 2f32.ln(), 2f32.ln()];
        // p = 0.7: 0.5 < 0.7, 0.75 >= 0.7 -> keep {0, 1}.
        let mut s = Sampler::new(
            params(|p| {
                p.top_k = 0;
                p.top_p = 0.7;
                p.min_p = 0.0;
            }),
            4,
        );
        let h = histogram(&mut s, &logits, 2000);
        assert_eq!(h[2] + h[3], 0, "{h:?}");
        assert!(h[0] > h[1] && h[1] > 0, "{h:?}");
        // p = 0.75 exactly: cumulative 0.75 >= 0.75 -> still {0, 1} (inclusive cut like llama.cpp).
        let mut s = Sampler::new(
            params(|p| {
                p.top_k = 0;
                p.top_p = 0.75;
                p.min_p = 0.0;
            }),
            4,
        );
        let h = histogram(&mut s, &logits, 2000);
        assert_eq!(h[2] + h[3], 0, "{h:?}");
        // p = 0.76: need token 2 as well; token 3 ties with 2 but comes after -> excluded.
        let mut s = Sampler::new(
            params(|p| {
                p.top_k = 0;
                p.top_p = 0.76;
                p.min_p = 0.0;
            }),
            4,
        );
        let h = histogram(&mut s, &logits, 2000);
        assert!(h[2] > 0 && h[3] == 0, "{h:?}");
        // p = 1.0 disables: everything is reachable.
        let mut s = Sampler::new(plain(), 4);
        let h = histogram(&mut s, &logits, 4000);
        assert!(h.iter().all(|&c| c > 0), "{h:?}");
    }

    #[test]
    fn top_p_is_relative_to_top_k_survivors() {
        // After top_k = 2 the candidates are {0, 1} with renormalised probs 2/3, 1/3.
        // top_p = 0.6 < 2/3 -> only token 0 survives.
        let logits = [8f32.ln(), 4f32.ln(), 2f32.ln(), 2f32.ln()];
        let mut s = Sampler::new(
            params(|p| {
                p.top_k = 2;
                p.top_p = 0.6;
                p.min_p = 0.0;
            }),
            4,
        );
        let h = histogram(&mut s, &logits, 500);
        assert_eq!(h[0], 500, "{h:?}");
    }

    #[test]
    fn min_p_drops_tokens_below_fraction_of_max() {
        // probs proportional to 1.0, 0.3, 0.1, 0.05 (relative to the max).
        let logits = [0.0f32, 0.3f32.ln(), 0.1f32.ln(), 0.05f32.ln()];
        for (min_p, expect_reachable) in [(0.2f32, 2usize), (0.09, 3), (0.04, 4), (0.0, 4)] {
            // sorted path (top_k on)
            let mut s = Sampler::new(
                params(|p| {
                    p.top_k = 40;
                    p.top_p = 1.0;
                    p.min_p = min_p;
                }),
                4,
            );
            let h = histogram(&mut s, &logits, 3000);
            let reachable = h.iter().filter(|&&c| c > 0).count();
            assert_eq!(reachable, expect_reachable, "sorted min_p={min_p} {h:?}");
            assert!(h[expect_reachable..].iter().all(|&c| c == 0), "{h:?}");
            // unsorted path (no top_k, no top_p)
            let mut s = Sampler::new(
                params(|p| {
                    p.top_k = 0;
                    p.top_p = 1.0;
                    p.min_p = min_p;
                }),
                4,
            );
            let h = histogram(&mut s, &logits, 3000);
            let reachable = h.iter().filter(|&&c| c > 0).count();
            assert_eq!(reachable, expect_reachable, "unsorted min_p={min_p} {h:?}");
        }
    }

    #[test]
    fn temperature_sharpens_and_flattens() {
        let logits = [2.0f32, 1.0, 0.0, -1.0];
        // Truncation filters are off so only temperature shapes the draw (the defaults' min_p
        // would otherwise drop token 3 before temperature is applied, as in llama.cpp).
        let hot = {
            let mut p = plain();
            p.temperature = 5.0;
            let mut s = Sampler::new(p, 4).with_processor(Box::new(NoopProcessor));
            s.set_processor(None);
            histogram(&mut s, &logits, 4000)
        };
        let cold = {
            let mut p = plain();
            p.temperature = 0.2;
            let mut s = Sampler::new(p, 4);
            histogram(&mut s, &logits, 4000)
        };
        // Cold is far more concentrated on the argmax than hot.
        assert!(cold[0] > hot[0] + 500, "cold={cold:?} hot={hot:?}");
        assert!(hot[3] > cold[3], "cold={cold:?} hot={hot:?}");
    }

    #[test]
    fn multinomial_frequencies_track_probabilities() {
        // probs = 0.6, 0.3, 0.1
        let logits = [0.6f32.ln(), 0.3f32.ln(), 0.1f32.ln()];
        let mut s = Sampler::new(plain(), 3);
        let n = 20_000;
        let h = histogram(&mut s, &logits, n);
        let f: Vec<f64> = h.iter().map(|&c| c as f64 / n as f64).collect();
        assert!((f[0] - 0.6).abs() < 0.02, "{f:?}");
        assert!((f[1] - 0.3).abs() < 0.02, "{f:?}");
        assert!((f[2] - 0.1).abs() < 0.02, "{f:?}");
    }

    #[test]
    fn penalty_math_matches_hand_computation() {
        let mut s = Sampler::new(
            params(|p| {
                p.repeat_penalty = 2.0;
                p.frequency_penalty = 0.5;
                p.presence_penalty = 0.25;
                p.repeat_last_n = 8;
            }),
            4,
        );
        // Window: token 0 twice, token 1 once.
        s.accept(0);
        s.accept(1);
        s.accept(0);
        let mut logits = [2.0f32, -1.0, 0.5, 0.0];
        s.apply_penalties(&mut logits);
        // token 0: 2.0 / 2 = 1.0; minus (2 * 0.5 + 0.25) = -0.25
        // token 1: -1.0 * 2 = -2.0; minus (1 * 0.5 + 0.25) = -2.75
        assert!((logits[0] - (-0.25)).abs() < 1e-6, "{logits:?}");
        assert!((logits[1] - (-2.75)).abs() < 1e-6, "{logits:?}");
        assert_eq!(logits[2], 0.5);
        assert_eq!(logits[3], 0.0);
        // Greedy sampling sees the penalised logits: token 2 now wins.
        let mut g = Sampler::new(
            params(|p| {
                p.temperature = 0.0;
                p.repeat_penalty = 2.0;
                p.frequency_penalty = 0.5;
                p.presence_penalty = 0.25;
            }),
            4,
        );
        assert_eq!(g.sample(&[2.0, -1.0, 0.5, 0.0]), 0);
        g.accept(0);
        g.accept(1);
        g.accept(0);
        assert_eq!(g.sample(&[2.0, -1.0, 0.5, 0.0]), 2);
    }

    #[test]
    fn penalty_window_slides_and_resets() {
        let mut s = Sampler::new(
            params(|p| {
                p.temperature = 0.0;
                p.repeat_penalty = 10.0;
                p.repeat_last_n = 2;
            }),
            3,
        );
        let logits = [3.0f32, 2.0, 1.0];
        s.accept(0);
        assert_eq!(s.sample(&logits), 1, "0 penalised to 0.3");
        s.accept(1);
        assert_eq!(s.sample(&logits), 2, "0 -> 0.3, 1 -> 0.2");
        s.accept(2); // window is now [1, 2]; 0 fell out
        assert_eq!(s.sample(&logits), 0);
        assert_eq!(s.window().collect::<Vec<_>>(), vec![1, 2]);
        s.reset();
        assert_eq!(s.window().count(), 0);
        assert_eq!(s.sample(&logits), 0);
        // Out-of-range tokens are ignored rather than panicking.
        s.accept(999);
        assert_eq!(s.window().count(), 0);
    }

    #[test]
    fn repeat_last_n_zero_disables_penalties() {
        let mut s = Sampler::new(
            params(|p| {
                p.temperature = 0.0;
                p.repeat_penalty = 10.0;
                p.repeat_last_n = 0;
            }),
            2,
        );
        s.accept(0);
        s.accept(0);
        assert_eq!(s.sample(&[1.0, 0.5]), 0);
    }

    struct NoopProcessor;
    impl LogitProcessor for NoopProcessor {
        fn process(&mut self, _logits: &mut [f32]) {}
        fn accept(&mut self, _token: Token) {}
    }

    /// Forbids a fixed token and records accepted tokens.
    struct Mask {
        forbid: Token,
        accepted: std::sync::Arc<std::sync::Mutex<Vec<Token>>>,
    }
    impl LogitProcessor for Mask {
        fn process(&mut self, logits: &mut [f32]) {
            logits[self.forbid as usize] = f32::NEG_INFINITY;
        }
        fn accept(&mut self, token: Token) {
            self.accepted.lock().unwrap().push(token);
        }
    }

    #[test]
    fn processor_runs_before_penalties_and_sees_accepts() {
        let accepted = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut s =
            Sampler::new(params(|p| p.temperature = 0.0), 3).with_processor(Box::new(Mask {
                forbid: 0,
                accepted: accepted.clone(),
            }));
        let logits = [5.0f32, 1.0, 0.0];
        let t = s.sample(&logits);
        assert_eq!(t, 1, "token 0 is masked out");
        s.accept(t);
        assert_eq!(*accepted.lock().unwrap(), vec![1]);
        // Stochastic path never draws the masked token either.
        let mut s = Sampler::new(plain(), 3).with_processor(Box::new(Mask {
            forbid: 0,
            accepted: accepted.clone(),
        }));
        let h = histogram(&mut s, &logits, 500);
        assert_eq!(h[0], 0, "{h:?}");
        assert!(s.take_processor().is_some());
        assert!(s.take_processor().is_none());
    }

    #[test]
    fn fully_masked_distribution_does_not_panic() {
        struct All;
        impl LogitProcessor for All {
            fn process(&mut self, logits: &mut [f32]) {
                logits.fill(f32::NEG_INFINITY);
            }
            fn accept(&mut self, _: Token) {}
        }
        let mut s = Sampler::new(SamplingParams::default(), 4).with_processor(Box::new(All));
        let t = s.sample(&[1.0, 2.0, 3.0, 4.0]);
        assert!(t < 4);
        let mut s = Sampler::new(plain(), 4).with_processor(Box::new(All));
        let t = s.sample(&[1.0, 2.0, 3.0, 4.0]);
        assert!(t < 4);
    }

    #[test]
    fn top_logprobs_after_temperature() {
        let logits = [1.0f32, 2.0, 3.0, 0.0];
        let s = Sampler::new(params(|p| p.temperature = 0.5), 4);
        let lp = s.top_logprobs(&logits, 2);
        assert_eq!(lp.len(), 2);
        assert_eq!(lp[0].0, 2);
        assert_eq!(lp[1].0, 1);
        // Expected: log_softmax(logits / 0.5)
        let scaled: Vec<f32> = logits.iter().map(|l| l * 2.0).collect();
        let z: f32 = scaled.iter().map(|l| l.exp()).sum::<f32>().ln();
        assert!((lp[0].1 - (scaled[2] - z)).abs() < 1e-5);
        assert!((lp[1].1 - (scaled[1] - z)).abs() < 1e-5);
        assert!((s.token_logprob(&logits, 2) - lp[0].1).abs() < 1e-6);
        // Probabilities of the full top-n sum to 1.
        let all = s.top_logprobs(&logits, 10);
        assert_eq!(all.len(), 4);
        let total: f32 = all.iter().map(|(_, l)| l.exp()).sum();
        assert!((total - 1.0).abs() < 1e-5);
        assert!(s.top_logprobs(&logits, 0).is_empty());
        // Greedy: raw logits.
        let g = Sampler::new(params(|p| p.temperature = 0.0), 4);
        let z_raw: f32 = logits.iter().map(|l| l.exp()).sum::<f32>().ln();
        assert!((g.token_logprob(&logits, 2) - (3.0 - z_raw)).abs() < 1e-6);
    }

    #[test]
    fn heap_top_k_matches_partial_selection() {
        let logits: Vec<f32> = (0..5000)
            .map(|i| (((i as u64).wrapping_mul(2654435761) % 97) as f32) * 0.1)
            .collect();
        for k in [1usize, 2, 7, 40, 500, 1024] {
            let mut heap = Vec::new();
            top_k_heap(&logits, k, &mut heap);
            heap.sort_unstable_by(cand_desc);
            let mut full: Vec<Cand> = logits
                .iter()
                .enumerate()
                .map(|(i, &logit)| Cand {
                    logit,
                    id: i as Token,
                })
                .collect();
            full.select_nth_unstable_by(k - 1, cand_desc);
            full.truncate(k);
            full.sort_unstable_by(cand_desc);
            let h: Vec<(Token, f32)> = heap.iter().map(|c| (c.id, c.logit)).collect();
            let f: Vec<(Token, f32)> = full.iter().map(|c| (c.id, c.logit)).collect();
            assert_eq!(h, f, "k={k}");
        }
        // -inf logits (masked) never displace real candidates.
        let mut l = vec![f32::NEG_INFINITY; 100];
        l[42] = 1.0;
        let mut out = Vec::new();
        top_k_heap(&l, 3, &mut out);
        out.sort_unstable_by(cand_desc);
        assert_eq!(out[0].id, 42);
    }

    #[test]
    fn large_vocab_top_k_is_fast() {
        // Guards against accidental O(n log n) full sorts or O(n * k) scans on the hot path.
        // Debug builds are slow, so the bound is loose; the point is that 262,144 candidates with
        // top_k = 40 never approaches quadratic cost (which would take seconds).
        const N: usize = 262_144;
        let logits: Vec<f32> = (0..N)
            .map(|i| (((i as u64).wrapping_mul(2654435761) % 10007) as f32) * 1e-3)
            .collect();
        let mut s = Sampler::new(params(|p| p.top_k = 40), N);
        // Warm the scratch allocations once.
        let t = s.sample(&logits);
        s.accept(t);
        let iters = 20;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let t = s.sample(&logits);
            s.accept(t);
        }
        let per_call = start.elapsed() / iters;
        eprintln!("large_vocab_top_k_is_fast: {per_call:?} per call");
        assert!(
            per_call < std::time::Duration::from_millis(25),
            "sampling over {N} entries took {per_call:?} per call"
        );
    }
}
