//! Request scheduler (Architecture §9.1–§9.2).
//!
//! Slots share one KV pool. Each step schedules up to `token_budget` tokens: every running slot's
//! next decode token first, then prefill chunks of the oldest admitted prompts (vLLM V1's
//! token-budget rule with decode-first ordering). Admission reserves KV blocks for the prompt plus
//! the declared generation length (scaled by a conservativeness factor); when the pool runs dry a
//! higher-priority arrival retracts the lowest-priority running slot, which is recomputed later
//! from its prefix (never swapped).
//!
//! This crate holds the policy and bookkeeping only; the backend executes the [`Step`] it
//! produces. The single-slot server loop of M1 keeps working until the batched forward pass
//! lands (M3), at which point the server drives this scheduler instead.

use llmario_engine_kv::{BlockPool, CacheKey, Sequence};
use std::collections::VecDeque;

pub type SlotId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Batch = 0,
    Api = 1,
    Interactive = 2,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SlotState {
    /// Admitted; `prompt[done..]` still has to be prefilled.
    Prefilling { done: usize },
    /// Prompt complete; one token per step.
    Decoding,
}

#[derive(Clone, Debug)]
pub struct Slot {
    pub id: SlotId,
    pub priority: Priority,
    pub prompt: Vec<u32>,
    pub max_new: usize,
    pub generated: usize,
    pub state: SlotState,
    /// Arrival order, for fairness and age-based anti-starvation.
    pub seq_no: u64,
    pub sequence: Sequence,
}

/// What the backend runs this step.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Step {
    /// Slots taking one decode token each.
    pub decodes: Vec<SlotId>,
    /// `(slot, start, end)` prompt ranges to prefill.
    pub prefills: Vec<(SlotId, usize, usize)>,
    pub tokens: usize,
}

#[derive(Clone, Debug)]
pub struct Request {
    pub prompt: Vec<u32>,
    pub max_new: usize,
    pub priority: Priority,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Admission {
    Admitted {
        slot: SlotId,
        cached_tokens: usize,
        retracted: Vec<SlotId>,
    },
    /// Not enough KV even after retracting everything with lower priority.
    Queued,
}

pub struct Scheduler {
    pub token_budget: usize,
    pub max_slots: usize,
    /// Multiplier on `max_new` when reserving (clients over-declare `max_tokens`).
    pub conservativeness: f64,
    pub key: CacheKey,
    slots: Vec<Slot>,
    queue: VecDeque<(u64, Request)>,
    next_id: SlotId,
    next_seq: u64,
    pub retractions: u64,
}

impl Scheduler {
    pub fn new(token_budget: usize, max_slots: usize, key: CacheKey) -> Scheduler {
        Scheduler {
            token_budget: token_budget.max(1),
            max_slots: max_slots.max(1),
            conservativeness: 1.0,
            key,
            slots: Vec::new(),
            queue: VecDeque::new(),
            next_id: 1,
            next_seq: 1,
            retractions: 0,
        }
    }

    pub fn slots(&self) -> &[Slot] {
        &self.slots
    }
    pub fn slot(&self, id: SlotId) -> Option<&Slot> {
        self.slots.iter().find(|s| s.id == id)
    }
    pub fn queued(&self) -> usize {
        self.queue.len()
    }

    fn reserve_tokens(&self, r: &Request) -> usize {
        r.prompt.len() + (r.max_new as f64 * self.conservativeness).ceil() as usize
    }

    /// Try to admit `r` now. Retracts lower-priority running slots when the pool is short.
    pub fn admit(&mut self, pool: &mut BlockPool, r: Request) -> Admission {
        let mut retracted = Vec::new();
        loop {
            if self.slots.len() < self.max_slots {
                let reserve = self.reserve_tokens(&r);
                if let Some((sequence, acq)) =
                    Sequence::acquire(pool, &r.prompt, reserve, &self.key)
                {
                    let id = self.next_id;
                    self.next_id += 1;
                    let seq_no = self.next_seq;
                    self.next_seq += 1;
                    let done = acq.cached_tokens.min(r.prompt.len().saturating_sub(1));
                    self.slots.push(Slot {
                        id,
                        priority: r.priority,
                        prompt: r.prompt,
                        max_new: r.max_new,
                        generated: 0,
                        state: SlotState::Prefilling { done },
                        seq_no,
                        sequence,
                    });
                    return Admission::Admitted {
                        slot: id,
                        cached_tokens: done,
                        retracted,
                    };
                }
            }
            // Retract the lowest-priority, youngest slot below this request's priority.
            let victim = self
                .slots
                .iter()
                .filter(|s| s.priority < r.priority)
                .min_by_key(|s| (s.priority, std::cmp::Reverse(s.seq_no)))
                .map(|s| s.id);
            match victim {
                Some(v) => {
                    let slot = self.remove(v).unwrap();
                    slot.sequence.release(pool);
                    self.retractions += 1;
                    // The retracted request goes back to the queue head so it resumes first.
                    self.queue.push_front((
                        slot.seq_no,
                        Request {
                            prompt: {
                                let mut p = slot.prompt;
                                p.truncate(p.len());
                                p
                            },
                            max_new: slot.max_new.saturating_sub(slot.generated),
                            priority: slot.priority,
                        },
                    ));
                    retracted.push(v);
                }
                None => {
                    let seq_no = self.next_seq;
                    self.next_seq += 1;
                    self.queue.push_back((seq_no, r));
                    return Admission::Queued;
                }
            }
        }
    }

    /// Admit as many queued requests as fit, oldest first (no retraction from the queue).
    pub fn drain_queue(&mut self, pool: &mut BlockPool) -> Vec<SlotId> {
        let mut admitted = Vec::new();
        while let Some((_, r)) = self.queue.front().cloned() {
            if self.slots.len() >= self.max_slots {
                break;
            }
            let reserve = self.reserve_tokens(&r);
            match Sequence::acquire(pool, &r.prompt, reserve, &self.key) {
                Some((sequence, acq)) => {
                    self.queue.pop_front();
                    let id = self.next_id;
                    self.next_id += 1;
                    let seq_no = self.next_seq;
                    self.next_seq += 1;
                    let done = acq.cached_tokens.min(r.prompt.len().saturating_sub(1));
                    self.slots.push(Slot {
                        id,
                        priority: r.priority,
                        prompt: r.prompt,
                        max_new: r.max_new,
                        generated: 0,
                        state: SlotState::Prefilling { done },
                        seq_no,
                        sequence,
                    });
                    admitted.push(id);
                }
                None => break,
            }
        }
        admitted
    }

    /// Build the next step: decodes first, then prefill chunks under the token budget.
    pub fn next_step(&self) -> Step {
        let mut step = Step::default();
        let mut budget = self.token_budget;
        let mut order: Vec<&Slot> = self.slots.iter().collect();
        order.sort_by_key(|s| (std::cmp::Reverse(s.priority), s.seq_no));
        for s in &order {
            if budget == 0 {
                break;
            }
            if s.state == SlotState::Decoding {
                step.decodes.push(s.id);
                budget -= 1;
            }
        }
        for s in &order {
            if budget == 0 {
                break;
            }
            if let SlotState::Prefilling { done } = s.state {
                let remaining = s.prompt.len() - done;
                let take = remaining.min(budget);
                if take > 0 {
                    step.prefills.push((s.id, done, done + take));
                    budget -= take;
                }
            }
        }
        step.tokens = self.token_budget - budget;
        step
    }

    /// Record that the backend ran `step`: advance prefill cursors and move finished prefills to
    /// decoding; count one generated token per decode slot.
    pub fn complete(&mut self, step: &Step) {
        for (id, _, end) in &step.prefills {
            if let Some(s) = self.slots.iter_mut().find(|s| s.id == *id) {
                s.state = if *end >= s.prompt.len() {
                    SlotState::Decoding
                } else {
                    SlotState::Prefilling { done: *end }
                };
            }
        }
        for id in &step.decodes {
            if let Some(s) = self.slots.iter_mut().find(|s| s.id == *id) {
                s.generated += 1;
            }
        }
    }

    /// Remove a slot (finished, cancelled, or retracted) without releasing its blocks.
    pub fn remove(&mut self, id: SlotId) -> Option<Slot> {
        let i = self.slots.iter().position(|s| s.id == id)?;
        Some(self.slots.remove(i))
    }

    /// Finish a slot and return its blocks to the pool.
    pub fn finish(&mut self, pool: &mut BlockPool, id: SlotId) {
        if let Some(s) = self.remove(id) {
            s.sequence.release(pool);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> CacheKey {
        CacheKey {
            model_digest: vec![7],
            kv_types: "f16".into(),
            rope_params: String::new(),
            template_hash: String::new(),
            salt: vec![],
        }
    }

    fn req(n: usize, max_new: usize, p: Priority) -> Request {
        Request {
            prompt: (0..n as u32).collect(),
            max_new,
            priority: p,
        }
    }

    #[test]
    fn decode_first_then_prefill_under_budget() {
        let mut pool = BlockPool::new(64, 4, 16);
        let mut s = Scheduler::new(8, 4, key());
        let a = match s.admit(&mut pool, req(10, 4, Priority::Api)) {
            Admission::Admitted { slot, .. } => slot,
            _ => panic!(),
        };
        let b = match s.admit(&mut pool, req(3, 4, Priority::Api)) {
            Admission::Admitted { slot, .. } => slot,
            _ => panic!(),
        };
        // Step 1: nothing decoding; prefill a (8 of 10), budget exhausted.
        let st = s.next_step();
        assert_eq!(st.decodes, Vec::<SlotId>::new());
        assert_eq!(st.prefills, vec![(a, 0, 8)]);
        s.complete(&st);
        // Step 2: a has 2 left, b has 3: both fit.
        let st = s.next_step();
        assert_eq!(st.prefills, vec![(a, 8, 10), (b, 0, 3)]);
        s.complete(&st);
        assert_eq!(s.slot(a).unwrap().state, SlotState::Decoding);
        // Step 3: both decode.
        let st = s.next_step();
        assert_eq!(st.decodes, vec![a, b]);
        assert_eq!(st.tokens, 2);
    }

    #[test]
    fn admission_retracts_lower_priority_when_pool_is_short() {
        let mut pool = BlockPool::new(4, 4, 16); // 16 tokens total
        let mut s = Scheduler::new(16, 4, key());
        let low = match s.admit(&mut pool, req(8, 4, Priority::Batch)) {
            Admission::Admitted { slot, .. } => slot,
            _ => panic!(),
        };
        // Needs 12 tokens = 3 blocks; only 1 free → retract the batch slot.
        match s.admit(&mut pool, req(8, 4, Priority::Interactive)) {
            Admission::Admitted { retracted, .. } => assert_eq!(retracted, vec![low]),
            other => panic!("{other:?}"),
        }
        assert_eq!(s.retractions, 1);
        assert_eq!(
            s.queued(),
            1,
            "the retracted request waits at the queue head"
        );
        // Same priority never retracts: queued instead.
        assert_eq!(
            s.admit(&mut pool, req(8, 4, Priority::Interactive)),
            Admission::Queued
        );
    }

    #[test]
    fn finish_returns_blocks_and_queue_drains() {
        let mut pool = BlockPool::new(4, 4, 16);
        let mut s = Scheduler::new(16, 4, key());
        let a = match s.admit(&mut pool, req(8, 8, Priority::Api)) {
            Admission::Admitted { slot, .. } => slot,
            _ => panic!(),
        };
        assert_eq!(
            s.admit(&mut pool, req(8, 8, Priority::Api)),
            Admission::Queued
        );
        s.finish(&mut pool, a);
        assert_eq!(pool.free_blocks(), 4);
        assert_eq!(s.drain_queue(&mut pool).len(), 1);
        assert_eq!(s.queued(), 0);
    }
}
