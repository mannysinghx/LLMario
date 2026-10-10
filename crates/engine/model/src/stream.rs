//! Routed experts streamed from the model file (Architecture §9.4, the plan's `ExpertsStreamed`
//! step): when the experts do not all fit in memory, the page cache keeps what it can and a token
//! that selects a cold expert reads it from disk. Left alone, those reads are page faults inside
//! the matrix-vector products, and on macOS faulting reaches about a third of what the SSD
//! delivers (one missing expert per layer on an M-series internal SSD: 1.3–1.4 GB/s faulting,
//! 3.8–4.4 GB/s with advisory reads; STATUS.md, Memory). [`Reader::fetch`] starts the reads of
//! every selected expert as soon as the router has chosen, without waiting for them:
//!
//! - macOS: `fcntl(F_RDADVISE)` on the file. (`madvise(MADV_WILLNEED)` there waits for the read
//!   to finish, so it cannot run ahead of the computation.)
//! - Other Unix: `madvise(MADV_WILLNEED)` on the mapping (asynchronous read-ahead on Linux).
//!
//! Which experts stay resident is left to the page cache. On recorded routing traces its
//! least-recently-used order came within a few percent of keeping the most-used experts, and
//! pinning experts profiled on a different prompt did worse than either.
//!
//! [`SimCache`] simulates a machine too small for the experts (`LLMARIO_SIM_EXPERT_RESIDENT`, a
//! share of the experts' bytes): a least-recently-used budget over experts that evicts with
//! `msync(MS_INVALIDATE)` at the end of each step.

use llmario_engine_formats::GgufFile;
use std::collections::BTreeSet;
use std::fs::File;
use std::path::Path;

/// What the MoE layers do about expert residency.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StreamOptions {
    /// Start reading the experts a step selected right after routing.
    pub prefetch: bool,
    /// Simulation: the share of the routed experts' bytes the page cache may hold.
    pub sim_resident: Option<f32>,
}

/// Starts reads of mapped model bytes without waiting for them.
pub(crate) struct Reader {
    regions: Vec<Region>,
}

struct Region {
    start: usize,
    len: usize,
    /// The part reopened for advisory reads (`None` if it could not be). Read only on macOS;
    /// elsewhere the advice goes to the mapping.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    file: Option<File>,
}

impl Reader {
    /// One region per mapped part of `file`.
    pub(crate) fn new(file: &GgufFile) -> Reader {
        Reader::from_parts(file.parts.iter().map(|p| (&p.mmap[..], p.path.as_path())))
    }

    fn from_parts<'p>(parts: impl Iterator<Item = (&'p [u8], &'p Path)>) -> Reader {
        let regions = parts
            .map(|(map, path)| Region {
                start: map.as_ptr() as usize,
                len: map.len(),
                file: File::open(path).ok(),
            })
            .collect();
        Reader { regions }
    }

    /// The region holding all of `bytes`, and their offset in its file.
    fn locate(&self, bytes: &[u8]) -> Option<(&Region, u64)> {
        let p = bytes.as_ptr() as usize;
        self.regions
            .iter()
            .find(|r| p >= r.start && p + bytes.len() <= r.start + r.len)
            .map(|r| (r, (p - r.start) as u64))
    }

    /// Start reading `bytes` (a slice of the mapped model) into the page cache unless they look
    /// resident already; returns at once.
    ///
    /// The probe (one page) costs a fraction of what the advice costs on resident bytes
    /// (measured per expert matrix: ~0.4 µs vs ~1–2 µs), and a matrix is read and dropped as a
    /// whole, so one of its pages stands for all of it.
    pub(crate) fn fetch(&self, bytes: &[u8]) {
        if bytes.is_empty() || probe_resident(bytes) {
            return;
        }
        let Some((region, offset)) = self.locate(bytes) else {
            return;
        };
        #[cfg(target_os = "macos")]
        if let Some(f) = &region.file {
            use std::os::fd::AsRawFd;
            let (mut offset, mut left) = (offset, bytes.len());
            while left > 0 {
                let n = left.min(1 << 30); // `ra_count` is a C int
                let ra = libc::radvisory {
                    ra_offset: offset as libc::off_t,
                    ra_count: n as libc::c_int,
                };
                // SAFETY: F_RDADVISE reads `ra` and only schedules reads into the page cache.
                unsafe { libc::fcntl(f.as_raw_fd(), libc::F_RDADVISE, &ra) };
                offset += n as u64;
                left -= n;
            }
        }
        #[cfg(all(unix, not(target_os = "macos")))]
        {
            let _ = (region, offset);
            let (start, len) = page_span(bytes);
            // SAFETY: the range covers whole pages of a live read-only file mapping; WILLNEED is
            // advisory and never changes the contents.
            unsafe { libc::madvise(start as *mut libc::c_void, len, libc::MADV_WILLNEED) };
        }
        #[cfg(not(unix))]
        let _ = (region, offset);
    }
}

#[cfg(unix)]
fn page_span(bytes: &[u8]) -> (usize, usize) {
    let page = page_size();
    let start = bytes.as_ptr() as usize & !(page - 1);
    let end = (bytes.as_ptr() as usize + bytes.len()).div_ceil(page) * page;
    (start, end - start)
}

/// Whether the last page wholly inside `bytes` (the last page they touch, if none is whole) is
/// in memory; `false` if unknown. A page shared with a neighbouring matrix would stay resident
/// while that neighbour is in use, so it is not probed when avoidable.
fn probe_resident(bytes: &[u8]) -> bool {
    #[cfg(unix)]
    {
        let page = page_size();
        let (a, b) = (
            bytes.as_ptr() as usize,
            bytes.as_ptr() as usize + bytes.len(),
        );
        let last_whole = (b & !(page - 1)).wrapping_sub(page);
        let probe = if (a..b).contains(&last_whole) {
            last_whole
        } else {
            (b - 1) & !(page - 1)
        };
        let mut v = [0u8; 1];
        // SAFETY: `probe` is one whole page of a live mapping and `v` holds its status byte.
        let r = unsafe { libc::mincore(probe as _, page, v.as_mut_ptr() as _) };
        r == 0 && v[0] & 1 != 0
    }
    #[cfg(not(unix))]
    {
        let _ = bytes;
        false
    }
}

#[cfg(unix)]
fn page_size() -> usize {
    static PAGE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    // SAFETY: sysconf is always safe to call.
    *PAGE.get_or_init(|| unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(4096) as usize)
}

/// Drop `bytes` from the page cache (simulation only; the next access reads the file again).
/// macOS drops the pages; Linux keeps clean cached pages (`MS_INVALIDATE` does not evict
/// there), so the simulation needs macOS.
pub(crate) fn evict(bytes: &[u8]) {
    #[cfg(unix)]
    if !bytes.is_empty() {
        let (start, len) = page_span(bytes);
        // SAFETY: the range covers whole pages of a live read-only shared file mapping, so
        // invalidating it only discards clean cached pages.
        unsafe { libc::msync(start as *mut libc::c_void, len, libc::MS_INVALIDATE) };
    }
    #[cfg(not(unix))]
    let _ = bytes;
}

/// Experts selected recently enough to be resident still, so not worth probing: a page cache
/// that drops the least recently used data first holds every expert of the last `window`
/// selections whenever it holds at least `window` experts. With the window at three single-row
/// steps' worth (`3 · MoE layers · n_expert_used`), no expert skipped this way was missing on the
/// recorded routing traces while the cache held at least a quarter of the experts, and it saved
/// 54–69% of the probes. Slot `layer · n_expert + expert`.
pub(crate) struct Recent {
    /// Per slot: the selection clock at its last use, plus one (0: never used).
    last: Vec<u64>,
    clock: u64,
    window: u64,
}

impl Recent {
    pub(crate) fn new(slots: usize, window: u64) -> Recent {
        Recent {
            last: vec![0; slots],
            clock: 0,
            window,
        }
    }

    /// Record a use of `slot`; returns whether it fell outside the window (worth fetching).
    pub(crate) fn use_slot(&mut self, slot: usize) -> bool {
        let stale = self.last[slot] == 0 || self.clock + 1 - self.last[slot] > self.window;
        self.last[slot] = self.clock + 1;
        stale
    }

    /// `selections` (row, expert) pairs were routed.
    pub(crate) fn advance(&mut self, selections: u64) {
        self.clock += selections;
    }
}

/// A least-recently-used byte budget over experts, slot `layer · n_expert + expert`: what a page
/// cache of `budget` bytes would keep.
pub(crate) struct SimCache {
    budget: u64,
    used: u64,
    /// Per slot: bytes and last-use stamp (0: not resident).
    size: Vec<u64>,
    stamp: Vec<u64>,
    lru: BTreeSet<(u64, u32)>,
    clock: u64,
    step_miss: u64,
    /// Bytes decode steps (one row) found missing, and their count.
    pub(crate) decode_miss: u64,
    pub(crate) decode_steps: u64,
    /// Seconds spent evicting (excluded from speed measurements).
    pub(crate) evict_secs: f64,
}

impl SimCache {
    pub(crate) fn new(slots: usize, budget: u64) -> SimCache {
        SimCache {
            budget,
            used: 0,
            size: vec![0; slots],
            stamp: vec![0; slots],
            lru: BTreeSet::new(),
            clock: 0,
            step_miss: 0,
            decode_miss: 0,
            decode_steps: 0,
            evict_secs: 0.0,
        }
    }

    /// Slot `slot` (of `bytes`) is used now.
    pub(crate) fn touch(&mut self, slot: usize, bytes: u64) {
        if self.stamp[slot] == 0 {
            self.used += bytes;
            self.step_miss += bytes;
            self.size[slot] = bytes;
        } else {
            self.lru.remove(&(self.stamp[slot], slot as u32));
        }
        self.clock += 1;
        self.stamp[slot] = self.clock;
        self.lru.insert((self.clock, slot as u32));
    }

    /// End of a step: evict least recently used slots until the budget holds (`evict(slot)`
    /// drops one from the page cache).
    pub(crate) fn end_step(&mut self, decode: bool, mut evict: impl FnMut(usize)) {
        let t0 = std::time::Instant::now();
        if decode {
            self.decode_miss += self.step_miss;
            self.decode_steps += 1;
        }
        self.step_miss = 0;
        while self.used > self.budget {
            let Some((_, slot)) = self.lru.pop_first() else {
                break;
            };
            let slot = slot as usize;
            self.stamp[slot] = 0;
            self.used -= self.size[slot];
            evict(slot);
        }
        self.evict_secs += t0.elapsed().as_secs_f64();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sim_cache_keeps_the_most_recent_within_budget() {
        let mut c = SimCache::new(4, 25);
        let mut gone = Vec::new();
        c.touch(0, 10);
        c.touch(1, 10);
        c.end_step(false, |s| gone.push(s));
        assert!(gone.is_empty());
        c.touch(2, 10); // 30 > 25: slot 0 is the least recent
        c.touch(1, 10); // a hit: no new bytes
        c.end_step(true, |s| gone.push(s));
        assert_eq!(gone, [0]);
        assert_eq!((c.decode_miss, c.decode_steps), (10, 1));
        c.touch(0, 10); // evicted before: a miss again; now 1 (oldest), 2, 0
        c.end_step(true, |s| gone.push(s));
        assert_eq!(gone, [0, 2]);
        assert_eq!((c.decode_miss, c.decode_steps), (20, 2));
    }

    #[test]
    fn recent_skips_only_inside_the_window() {
        let mut r = Recent::new(3, 16);
        assert!(r.use_slot(0), "never used");
        r.advance(8);
        assert!(!r.use_slot(0), "8 selections ago");
        assert!(r.use_slot(1));
        r.advance(16);
        assert!(!r.use_slot(0), "16 selections ago: still inside");
        r.advance(17);
        assert!(r.use_slot(0), "17 selections ago");
        assert!(r.use_slot(1));
        assert!(r.use_slot(2));
    }

    #[cfg(unix)]
    #[test]
    fn fetch_and_eviction_keep_the_bytes() {
        // A file mapping: advice and invalidation never change what a read returns.
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f");
        let data: Vec<u8> = (0..300_000u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(&p, &data).unwrap();
        let f = std::fs::File::open(&p).unwrap();
        // SAFETY: the file is not modified while mapped.
        let m = unsafe { memmap2::Mmap::map(&f).unwrap() };
        let r = Reader::from_parts(std::iter::once((&m[..], p.as_path())));
        assert!(r.regions[0].file.is_some());
        let (_, off) = r.locate(&m[1000..200_000]).unwrap();
        assert_eq!(off, 1000);
        assert!(r.locate(&data[..10]).is_none(), "not in the mapping");
        std::hint::black_box(m.iter().map(|&b| b as u64).sum::<u64>()); // fault every page in
        assert!(probe_resident(&m[1000..200_000]));
        evict(&m[..]);
        r.fetch(&m[1000..200_000]);
        r.fetch(&m[299_000..]);
        r.fetch(&data[..10]);
        evict(&m[5000..150_000]);
        assert_eq!(&m[..], &data[..]);
        assert!(probe_resident(&m[1000..200_000]));
    }
}
