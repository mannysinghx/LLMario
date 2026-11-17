//! Memory pressure as the operating system reports it.
//!
//! The plan is made once, at load, against a ceiling. What the rest of the computer does
//! afterwards is not in it: a browser grows, another model is started, and a machine that had
//! room begins to compress and swap. The engine holds one thing it can give back without losing
//! work, the KV caches idle slots keep so a conversation's next turn does not recompute its
//! prompt. When the system reports pressure the inference thread frees them (saving each to the
//! KV disk tier first when one is on, so the next turn reads it back instead of recomputing), and
//! under critical pressure it frees them even when nothing can be saved.
//!
//! Sources:
//!
//! - macOS: `kern.memorystatus_vm_pressure_level` (1 normal, 2 warning, 4 critical), the level
//!   behind the system's own memory-pressure notifications.
//! - Linux: pressure stall information, `/proc/pressure/memory`. `some avg10` is the share of the
//!   last ten seconds in which at least one task waited for memory, `full avg10` the share in
//!   which every task did.
//! - `LLMARIO_SIM_PRESSURE=<file>`: the level is read from a file holding `normal`, `warning` or
//!   `critical` (tests and measurements).
//!
//! Elsewhere the level is unknown and nothing is freed.

use std::path::PathBuf;

/// How short of memory the system says it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Level {
    Normal,
    Warning,
    Critical,
}

impl Level {
    pub fn name(self) -> &'static str {
        match self {
            Level::Normal => "normal",
            Level::Warning => "warning",
            Level::Critical => "critical",
        }
    }

    fn parse(s: &str) -> Option<Level> {
        match s.trim().to_ascii_lowercase().as_str() {
            "normal" => Some(Level::Normal),
            "warning" | "warn" => Some(Level::Warning),
            "critical" => Some(Level::Critical),
            _ => None,
        }
    }
}

/// `kern.memorystatus_vm_pressure_level` values (xnu `kVMPressure*` as exported to user space).
pub fn from_macos(v: u32) -> Option<Level> {
    match v {
        1 => Some(Level::Normal),
        2 => Some(Level::Warning),
        4 => Some(Level::Critical),
        _ => None,
    }
}

/// Share of the last ten seconds some task stalled on memory at which pressure is a warning.
pub const PSI_SOME_WARNING: f32 = 10.0;
/// Share of the last ten seconds every task stalled on memory at which pressure is critical.
pub const PSI_FULL_CRITICAL: f32 = 10.0;

/// The contents of `/proc/pressure/memory`.
pub fn from_psi(text: &str) -> Option<Level> {
    let avg10 = |kind: &str| -> Option<f32> {
        text.lines()
            .find(|l| l.starts_with(kind))?
            .split_whitespace()
            .find_map(|f| f.strip_prefix("avg10="))?
            .parse()
            .ok()
    };
    let some = avg10("some")?;
    let full = avg10("full").unwrap_or(0.0);
    Some(if full >= PSI_FULL_CRITICAL {
        Level::Critical
    } else if some >= PSI_SOME_WARNING {
        Level::Warning
    } else {
        Level::Normal
    })
}

/// Reads the system's memory pressure.
#[derive(Clone, Debug, Default)]
pub struct Monitor {
    sim: Option<PathBuf>,
}

impl Monitor {
    /// The system's own source, or the file named by `LLMARIO_SIM_PRESSURE`.
    pub fn from_env() -> Monitor {
        Monitor {
            sim: std::env::var_os("LLMARIO_SIM_PRESSURE").map(PathBuf::from),
        }
    }

    /// A monitor that reads the level from `file` (tests).
    pub fn simulated(file: PathBuf) -> Monitor {
        Monitor { sim: Some(file) }
    }

    /// The current level; `None` when the system does not report one.
    pub fn level(&self) -> Option<Level> {
        if let Some(f) = &self.sim {
            return std::fs::read_to_string(f)
                .ok()
                .and_then(|s| Level::parse(&s));
        }
        system_level()
    }
}

#[cfg(target_os = "macos")]
fn system_level() -> Option<Level> {
    let mut v: u32 = 0;
    let mut len = std::mem::size_of::<u32>();
    // SAFETY: `v` and `len` describe a 4-byte buffer for an integer sysctl; the name is a
    // NUL-terminated literal.
    let r = unsafe {
        libc::sysctlbyname(
            c"kern.memorystatus_vm_pressure_level".as_ptr(),
            &mut v as *mut u32 as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (r == 0).then(|| from_macos(v)).flatten()
}

#[cfg(target_os = "linux")]
fn system_level() -> Option<Level> {
    from_psi(&std::fs::read_to_string("/proc/pressure/memory").ok()?)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn system_level() -> Option<Level> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn macos_levels() {
        assert_eq!(from_macos(1), Some(Level::Normal));
        assert_eq!(from_macos(2), Some(Level::Warning));
        assert_eq!(from_macos(4), Some(Level::Critical));
        assert_eq!(from_macos(0), None);
        assert_eq!(from_macos(3), None);
    }

    #[test]
    fn psi_levels() {
        let quiet = "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n\
                     full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n";
        assert_eq!(from_psi(quiet), Some(Level::Normal));
        let busy = "some avg10=23.50 avg60=4.00 avg300=1.00 total=9\n\
                    full avg10=2.10 avg60=0.50 avg300=0.10 total=3\n";
        assert_eq!(from_psi(busy), Some(Level::Warning));
        let stuck = "some avg10=80.00 avg60=40.00 avg300=9.00 total=9\n\
                     full avg10=41.00 avg60=20.00 avg300=5.00 total=3\n";
        assert_eq!(from_psi(stuck), Some(Level::Critical));
        // Kernels before 5.2 report no `full` line for some resources; `some` alone still counts.
        assert_eq!(
            from_psi("some avg10=12.00 avg60=0.00 avg300=0.00 total=1\n"),
            Some(Level::Warning)
        );
        assert_eq!(from_psi("garbage"), None);
    }

    #[test]
    fn simulated_file_wins_and_unknown_text_is_no_level() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("level");
        let m = Monitor::simulated(f.clone());
        assert_eq!(m.level(), None, "no file yet");
        std::fs::write(&f, "warning\n").unwrap();
        assert_eq!(m.level(), Some(Level::Warning));
        std::fs::write(&f, "CRITICAL").unwrap();
        assert_eq!(m.level(), Some(Level::Critical));
        std::fs::write(&f, "what").unwrap();
        assert_eq!(m.level(), None);
        assert!(Level::Normal < Level::Warning && Level::Warning < Level::Critical);
    }

    #[test]
    fn the_system_source_answers_or_declines_without_failing() {
        // Whatever this machine reports, reading it must not panic; on macOS and on Linux with
        // PSI it is a level.
        let l = Monitor::default().level();
        if cfg!(target_os = "macos") {
            assert!(l.is_some());
        }
    }
}
