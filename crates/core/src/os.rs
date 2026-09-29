//! Operating-system specifics, kept here so the other crates stay platform-neutral.
//!
//! Unix (macOS, Linux) and Windows differ in how child processes are hidden, ended and
//! tracked, and in how free disk space is read.

use std::ffi::OsStr;
use std::path::Path;

/// `CREATE_NO_WINDOW`: without it, every console program started from the desktop app (a GUI
/// process on Windows) opens a visible console window.
#[cfg(windows)]
pub const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// A command for a short helper program (engine `--version`, `nvidia-smi`). On Windows it runs
/// without a console window.
pub fn background_command(program: impl AsRef<OsStr>) -> std::process::Command {
    #[allow(unused_mut)]
    let mut cmd = std::process::Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

/// Whether a process with this id is running.
pub fn pid_alive(pid: u32) -> bool {
    imp::pid_alive(pid)
}

/// Bytes available to this user on the filesystem that holds `path` (an existing directory).
pub fn free_space_bytes(path: &Path) -> Option<u64> {
    imp::free_space_bytes(path)
}

#[cfg(windows)]
pub use imp::{kill_on_exit, process_working_set_bytes, terminate};

#[cfg(unix)]
mod imp {
    use std::path::Path;

    pub fn pid_alive(pid: u32) -> bool {
        // SAFETY: signal 0 only checks that the process exists.
        unsafe { libc::kill(pid as i32, 0) == 0 }
    }

    pub fn free_space_bytes(path: &Path) -> Option<u64> {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;
        let c = CString::new(path.as_os_str().as_bytes()).ok()?;
        // SAFETY: statvfs writes into the zeroed struct; we check the return code.
        unsafe {
            let mut s: libc::statvfs = std::mem::zeroed();
            if libc::statvfs(c.as_ptr(), &mut s) != 0 {
                return None;
            }
            #[allow(clippy::unnecessary_cast)]
            Some(s.f_bavail as u64 * s.f_frsize as u64)
        }
    }
}

#[cfg(windows)]
mod imp {
    use std::os::windows::ffi::OsStrExt;
    use std::os::windows::io::RawHandle;
    use std::path::Path;
    use std::sync::OnceLock;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, STILL_ACTIVE};
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, TerminateProcess, PROCESS_QUERY_LIMITED_INFORMATION,
        PROCESS_TERMINATE, PROCESS_VM_READ,
    };

    /// An open process handle, closed on drop.
    struct Process(HANDLE);

    impl Process {
        fn open(pid: u32, access: u32) -> Option<Self> {
            // SAFETY: plain Win32 call; a null handle means failure.
            let h = unsafe { OpenProcess(access, 0, pid) };
            (!h.is_null()).then_some(Self(h))
        }
    }

    impl Drop for Process {
        fn drop(&mut self) {
            // SAFETY: we own this handle.
            unsafe { CloseHandle(self.0) };
        }
    }

    pub fn pid_alive(pid: u32) -> bool {
        let Some(p) = Process::open(pid, PROCESS_QUERY_LIMITED_INFORMATION) else {
            return false;
        };
        let mut code = 0u32;
        // SAFETY: valid handle, valid out pointer.
        unsafe { GetExitCodeProcess(p.0, &mut code) != 0 && code == STILL_ACTIVE as u32 }
    }

    /// End a process immediately. Windows has no SIGTERM; the engines keep no state that needs
    /// a clean exit.
    pub fn terminate(pid: u32) -> bool {
        let Some(p) = Process::open(pid, PROCESS_TERMINATE) else {
            return false;
        };
        // SAFETY: valid handle with PROCESS_TERMINATE access.
        unsafe { TerminateProcess(p.0, 1) != 0 }
    }

    /// Working set (host memory) of a process. GPU memory is not included.
    pub fn process_working_set_bytes(pid: u32) -> Option<u64> {
        let p = Process::open(pid, PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ)?;
        // SAFETY: the struct is plain data; `cb` tells the API its size.
        unsafe {
            let mut c: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
            c.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
            (GetProcessMemoryInfo(p.0, &mut c, c.cb) != 0).then_some(c.WorkingSetSize as u64)
        }
    }

    /// Put a child process in a job object that Windows ends when this process exits, even
    /// if it crashes or is killed from Task Manager (like `PR_SET_PDEATHSIG` on Linux).
    ///
    /// One job holds every engine. Its handle is never closed: the OS closes it when this
    /// process exits, and `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` then ends the engines.
    pub fn kill_on_exit(child: RawHandle) -> std::io::Result<()> {
        static JOB: OnceLock<usize> = OnceLock::new();
        let job = *JOB.get_or_init(|| {
            // SAFETY: creates an unnamed job with default security; checked below.
            unsafe {
                let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
                if job.is_null() {
                    return 0;
                }
                let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                let ok = SetInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const std::ffi::c_void,
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                );
                if ok == 0 {
                    CloseHandle(job);
                    return 0;
                }
                job as usize
            }
        });
        if job == 0 {
            return Err(std::io::Error::other(
                "could not create the engine job object",
            ));
        }
        // SAFETY: both handles are valid; the job handle lives for the rest of the process.
        if unsafe { AssignProcessToJobObject(job as HANDLE, child as HANDLE) } == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }

    pub fn free_space_bytes(path: &Path) -> Option<u64> {
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
        let mut available = 0u64;
        // SAFETY: NUL-terminated path; the two unused out parameters may be null.
        let ok = unsafe {
            GetDiskFreeSpaceExW(
                wide.as_ptr(),
                &mut available,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        (ok != 0).then_some(available)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn own_process_is_alive() {
        assert!(pid_alive(std::process::id()));
    }

    #[test]
    fn free_space_of_temp_dir() {
        let free = free_space_bytes(&std::env::temp_dir()).expect("free space");
        assert!(free > 0);
    }

    #[test]
    fn background_command_runs() {
        let prog = if cfg!(windows) { "cmd" } else { "true" };
        let mut cmd = background_command(prog);
        if cfg!(windows) {
            cmd.args(["/C", "exit 0"]);
        }
        assert!(cmd.status().expect("spawn").success());
    }
}
