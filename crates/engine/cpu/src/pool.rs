//! The engine's worker pool (Architecture §4.2 / §7.3).
//!
//! - Sized to the physical performance-class cores by default (never SMT siblings).
//! - Workers spin for a bounded time on a new job, then park on a condvar; nothing polls.
//! - One parallel region per fused op: [`ThreadPool::parallel_for`] splits an index range into
//!   contiguous chunks (static partition, so each worker's weight stream stays contiguous during
//!   decode) and blocks the caller until every chunk is done. The caller is worker 0.
//!
//! This is a from-scratch pool rather than rayon so the spin/park policy, the thread count and the
//! QoS hooks are the engine's own. `rayon` is not a dependency.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

type JobFn = dyn Fn(usize, usize) + Send + Sync;

/// One published parallel region.
struct Job {
    gen: u64,
    f: Arc<JobFn>,
    n_chunks: usize,
    chunk: usize,
    n: usize,
    /// Per-job counters: a worker that wakes late with a stale job only touches that job's
    /// counters, so it can neither run a finished closure nor disturb the next region.
    next_chunk: AtomicUsize,
    done: AtomicUsize,
    panicked: AtomicBool,
}

struct Shared {
    job: Mutex<Option<Arc<Job>>>,
    cv: Condvar,
    generation: AtomicU64,
    done_lock: Mutex<()>,
    done_cv: Condvar,
    stop: AtomicBool,
}

pub struct ThreadPool {
    shared: Arc<Shared>,
    n_threads: usize,
    handles: Vec<thread::JoinHandle<()>>,
}

/// Spin iterations before a worker parks (tens of microseconds on current CPUs).
const SPIN_ITERS: usize = 4_000;

impl ThreadPool {
    /// A pool with `n_threads` workers in total (the calling thread participates as worker 0, so
    /// `n_threads - 1` threads are spawned). `n_threads == 1` runs everything inline.
    pub fn new(n_threads: usize) -> ThreadPool {
        let n_threads = n_threads.max(1);
        let shared = Arc::new(Shared {
            job: Mutex::new(None),
            cv: Condvar::new(),
            generation: AtomicU64::new(0),
            done_lock: Mutex::new(()),
            done_cv: Condvar::new(),
            stop: AtomicBool::new(false),
        });
        let mut handles = Vec::new();
        for i in 1..n_threads {
            let s = shared.clone();
            handles.push(
                thread::Builder::new()
                    .name(format!("llmario-cpu-{i}"))
                    .spawn(move || worker_loop(s))
                    .expect("spawn worker"),
            );
        }
        ThreadPool {
            shared,
            n_threads,
            handles,
        }
    }

    /// Default size: `available_parallelism` (counts SMT siblings; callers with a hardware report
    /// should pass the physical performance-core count instead).
    pub fn with_default_threads() -> ThreadPool {
        ThreadPool::new(
            thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
        )
    }

    pub fn n_threads(&self) -> usize {
        self.n_threads
    }

    /// Run `f(start, end)` over `[0, n)` split into `chunks` contiguous chunks (default: one per
    /// thread). Blocks until all chunks finish. Panics in `f` are re-raised on the caller after
    /// every worker has left the region, so borrowed data is never used after it is gone.
    pub fn parallel_for<F>(&self, n: usize, chunks: Option<usize>, f: F)
    where
        F: Fn(usize, usize) + Send + Sync,
    {
        if n == 0 {
            return;
        }
        let n_chunks = chunks.unwrap_or(self.n_threads).clamp(1, n);
        if self.n_threads == 1 || n_chunks == 1 {
            f(0, n);
            return;
        }
        let chunk = n.div_ceil(n_chunks);
        // The closure borrows the caller's stack. We hand workers an `Arc<dyn Fn + 'static>` and
        // guarantee the borrow outlives every use by not returning until `done == n_chunks`
        // (even when a chunk panics). That is the only reason the transmute is sound.
        let boxed: Arc<dyn Fn(usize, usize) + Send + Sync + '_> = Arc::new(f);
        // SAFETY: see above; the pool never retains the job past this call (it is cleared below
        // before returning), and workers only run it between the generation bump and `done`.
        let f_static: Arc<JobFn> = unsafe { std::mem::transmute(boxed) };
        let gen = self.shared.generation.load(Ordering::Acquire) + 1;
        let job = Arc::new(Job {
            gen,
            f: f_static,
            n_chunks,
            chunk,
            n,
            next_chunk: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
        });
        {
            let mut slot = self.shared.job.lock().unwrap();
            *slot = Some(job.clone());
            self.shared.generation.store(gen, Ordering::Release);
        }
        self.shared.cv.notify_all();
        run_chunks(&self.shared, &job);
        let mut guard = self.shared.done_lock.lock().unwrap();
        while job.done.load(Ordering::Acquire) < n_chunks {
            guard = self.shared.done_cv.wait(guard).unwrap();
        }
        drop(guard);
        *self.shared.job.lock().unwrap() = None;
        if job.panicked.load(Ordering::Acquire) {
            panic!("a parallel_for chunk panicked");
        }
    }
}

fn run_chunks(s: &Shared, job: &Job) {
    loop {
        let c = job.next_chunk.fetch_add(1, Ordering::AcqRel);
        if c >= job.n_chunks {
            break;
        }
        let start = c * job.chunk;
        let end = ((c + 1) * job.chunk).min(job.n);
        if start < end {
            let f = &job.f;
            if catch_unwind(AssertUnwindSafe(|| f(start, end))).is_err() {
                job.panicked.store(true, Ordering::Release);
            }
        }
        let d = job.done.fetch_add(1, Ordering::AcqRel) + 1;
        if d == job.n_chunks {
            let _g = s.done_lock.lock().unwrap();
            s.done_cv.notify_all();
        }
    }
}

fn worker_loop(s: Arc<Shared>) {
    let mut seen_gen = 0u64;
    loop {
        if s.stop.load(Ordering::Acquire) {
            return;
        }
        // Bounded spin on the generation counter (no lock), then park.
        let mut job = None;
        for _ in 0..SPIN_ITERS {
            if s.generation.load(Ordering::Acquire) != seen_gen {
                job = s.job.lock().unwrap().clone();
                if job.as_ref().is_some_and(|j| j.gen != seen_gen) {
                    break;
                }
                job = None;
            }
            std::hint::spin_loop();
        }
        let job = match job {
            Some(j) => j,
            None => {
                let mut guard = s.job.lock().unwrap();
                loop {
                    if s.stop.load(Ordering::Acquire) {
                        return;
                    }
                    if let Some(j) = guard.as_ref() {
                        if j.gen != seen_gen {
                            break j.clone();
                        }
                    }
                    guard = s.cv.wait(guard).unwrap();
                }
            }
        };
        seen_gen = job.gen;
        run_chunks(&s, &job);
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // Hold the job lock while raising `stop`: a worker that checked `stop` under the lock
        // and is about to `wait` cannot miss the notification (lost wake-up).
        {
            let _guard = self.shared.job.lock().unwrap();
            self.shared.stop.store(true, Ordering::Release);
        }
        self.shared.cv.notify_all();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_for_covers_every_index_once() {
        let pool = ThreadPool::new(4);
        for n in [1usize, 5, 64, 1000, 4097] {
            let hits: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
            pool.parallel_for(n, None, |s, e| {
                for h in &hits[s..e] {
                    h.fetch_add(1, Ordering::Relaxed);
                }
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "n={n}");
        }
    }

    #[test]
    fn many_small_jobs_do_not_deadlock() {
        let pool = ThreadPool::new(3);
        let total = AtomicU64::new(0);
        for _ in 0..5000 {
            pool.parallel_for(10, Some(5), |s, e| {
                total.fetch_add((e - s) as u64, Ordering::Relaxed);
            });
        }
        assert_eq!(total.load(Ordering::Relaxed), 50_000);
    }

    #[test]
    fn chunk_panic_is_reraised_after_region_ends() {
        let pool = ThreadPool::new(2);
        let r = catch_unwind(AssertUnwindSafe(|| {
            pool.parallel_for(4, Some(4), |s, _| {
                if s == 2 {
                    panic!("boom");
                }
            });
        }));
        assert!(r.is_err());
        // The pool is still usable afterwards.
        let total = AtomicU64::new(0);
        pool.parallel_for(8, None, |s, e| {
            total.fetch_add((e - s) as u64, Ordering::Relaxed);
        });
        assert_eq!(total.load(Ordering::Relaxed), 8);
    }

    #[test]
    fn single_thread_runs_inline() {
        let pool = ThreadPool::new(1);
        let total = AtomicU64::new(0);
        pool.parallel_for(8, None, |s, e| {
            total.fetch_add((e - s) as u64, Ordering::Relaxed);
        });
        assert_eq!(total.load(Ordering::Relaxed), 8);
    }
}
