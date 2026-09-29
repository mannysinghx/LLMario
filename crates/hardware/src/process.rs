/// Physical memory attributed to a process, in bytes.
///
/// - macOS: `ri_phys_footprint` from `proc_pid_rusage` — this is what Activity Monitor shows
///   and, unlike RSS, it includes Metal/MLX GPU allocations in unified memory.
/// - Linux: `VmRSS` from `/proc/<pid>/status` (host memory only; CUDA VRAM is not included).
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
            (rc == 0).then_some(info.ri_phys_footprint)
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

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
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
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            assert!(m.unwrap() > 0);
        }
    }
}
