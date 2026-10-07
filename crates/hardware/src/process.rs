/// Physical memory attributed to a process, in bytes.
///
/// - macOS: the larger of `ri_phys_footprint` and `ri_resident_size` from `proc_pid_rusage`.
///   The footprint (what Activity Monitor shows) includes Metal/MLX GPU allocations but leaves
///   out memory-mapped files; resident size includes them. llama.cpp maps its weight file, so
///   only the resident size counts its weights (gpt-oss-20b: footprint 0.48 GiB, resident
///   11.49 GiB, 11.28 GiB of weights); MLX copies weights into GPU buffers, so the footprint is
///   the larger there (Qwen3 1.7B: footprint 1.41 GiB, resident 1.21 GiB, 0.90 GiB of weights).
/// - Linux: `VmRSS` from `/proc/<pid>/status` (host memory only; CUDA VRAM is not included).
/// - Windows: the working set (host memory only; GPU memory is not included).
pub fn process_memory_bytes(pid: u32) -> Option<u64> {
    imp::footprint(pid)
}

#[cfg(target_os = "macos")]
mod imp {
    pub fn footprint(pid: u32) -> Option<u64> {
        // SAFETY: `rusage_info_v2` is plain data; the kernel fills it for the given flavor.
        unsafe {
            let mut info: libc::rusage_info_v2 = std::mem::zeroed();
            let rc = libc::proc_pid_rusage(
                pid as libc::c_int,
                libc::RUSAGE_INFO_V2,
                &mut info as *mut _ as *mut libc::rusage_info_t,
            );
            (rc == 0).then_some(info.ri_phys_footprint.max(info.ri_resident_size))
        }
    }
}

#[cfg(target_os = "linux")]
mod imp {
    pub fn footprint(pid: u32) -> Option<u64> {
        let s = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
        let line = s.lines().find(|l| l.starts_with("VmRSS:"))?;
        let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kb * 1024)
    }
}

#[cfg(windows)]
mod imp {
    pub fn footprint(pid: u32) -> Option<u64> {
        llmario_core::os::process_working_set_bytes(pid)
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
mod imp {
    pub fn footprint(_pid: u32) -> Option<u64> {
        None
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn own_process_has_memory() {
        let m = super::process_memory_bytes(std::process::id());
        if cfg!(any(target_os = "macos", target_os = "linux", windows)) {
            assert!(m.unwrap() > 0);
        }
    }

    /// llama.cpp memory-maps its weights; they must count (the footprint alone leaves them out).
    #[cfg(target_os = "macos")]
    #[test]
    fn memory_mapped_files_count() {
        use std::io::Write;
        const LEN: usize = 128 << 20;
        let path = std::env::temp_dir().join(format!("llmario-mmap-test-{}", std::process::id()));
        let mut f = std::fs::File::create(&path).unwrap();
        let chunk = vec![1u8; 1 << 20];
        for _ in 0..(LEN >> 20) {
            f.write_all(&chunk).unwrap();
        }
        drop(f);
        let f = std::fs::File::open(&path).unwrap();
        let pid = std::process::id();
        let before = super::process_memory_bytes(pid).unwrap();
        // SAFETY: read-only private mapping of a file we just wrote; unmapped below.
        let sum = unsafe {
            use std::os::fd::AsRawFd;
            let p = libc::mmap(
                std::ptr::null_mut(),
                LEN,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                f.as_raw_fd(),
                0,
            );
            assert_ne!(p, libc::MAP_FAILED);
            let bytes = std::slice::from_raw_parts(p as *const u8, LEN);
            let sum: u64 = bytes.iter().step_by(4096).map(|b| *b as u64).sum();
            let after = super::process_memory_bytes(pid).unwrap();
            libc::munmap(p, LEN);
            assert!(
                after >= before + (LEN as u64) * 3 / 4,
                "mapped {LEN} bytes: memory went from {before} to {after}"
            );
            sum
        };
        let _ = std::fs::remove_file(&path);
        assert_eq!(sum, (LEN / 4096) as u64);
    }
}
