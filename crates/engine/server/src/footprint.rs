//! Measured memory of this process, for the plan-versus-measured check (Architecture §5.3 step 8).
//!
//! - macOS: `proc_pid_rusage` — the larger of `ri_phys_footprint` (anonymous + wired, what
//!   Activity Monitor shows) and `ri_resident_size` (which also counts resident pages of the
//!   mapped weights). The plan charges mapped weights in full, so the resident size is the
//!   comparable number.
//! - Linux: `VmRSS` from `/proc/self/status`; `VmHWM` for the peak.
//! - Windows: the working set via `GetProcessMemoryInfo` (lands with the Windows validation).

#[cfg(target_os = "macos")]
pub fn current() -> Option<u64> {
    // SAFETY: `rusage_info_v2` is plain data; the kernel fills it for the given flavor.
    unsafe {
        let mut info: libc::rusage_info_v2 = std::mem::zeroed();
        let rc = libc::proc_pid_rusage(
            std::process::id() as libc::c_int,
            libc::RUSAGE_INFO_V2,
            &mut info as *mut _ as *mut libc::rusage_info_t,
        );
        (rc == 0).then_some(info.ri_phys_footprint.max(info.ri_resident_size))
    }
}

#[cfg(target_os = "linux")]
pub fn current() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = s.lines().find(|l| l.starts_with("VmRSS:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn current() -> Option<u64> {
    None
}
