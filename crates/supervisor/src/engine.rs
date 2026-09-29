//! One running engine child process: spawn, readiness, warm-up, crash detection, shutdown.

use crate::adapter::LaunchSpec;
use crate::memory::MemoryPlan;
use llmario_core::{BackendKind, ResolvedProfile, RuntimeError};
use llmario_registry::ModelEntry;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{oneshot, watch, Semaphore};

#[derive(Clone, Debug, Serialize)]
pub struct ExitInfo {
    pub code: Option<i32>,
    pub signal: Option<i32>,
    /// True when we asked it to stop (not a crash).
    pub requested: bool,
}

impl ExitInfo {
    pub fn describe(&self) -> String {
        match (self.code, self.signal) {
            (Some(c), _) => format!("exit code {c}"),
            (None, Some(s)) => format!("signal {s}"),
            _ => "unknown exit".into(),
        }
    }
}

/// Serializable snapshot for `/metrics`, `doctor`, and `ps`.
#[derive(Serialize, Clone, Debug)]
pub struct EngineInfo {
    pub model: String,
    pub backend: BackendKind,
    pub pid: u32,
    pub port: u16,
    pub profile: ResolvedProfile,
    pub active_requests: usize,
    pub ready_seconds: f64,
    pub idle_seconds: f64,
    pub estimated_bytes: u64,
    pub resident_bytes: Option<u64>,
    pub log: PathBuf,
}

pub struct Engine {
    pub model: ModelEntry,
    pub backend: BackendKind,
    pub base_url: String,
    pub port: u16,
    pub upstream_model: String,
    pub pid: u32,
    pub profile: ResolvedProfile,
    pub memory: MemoryPlan,
    pub log_path: PathBuf,
    pub notes: Vec<String>,
    pub(crate) permits: Arc<Semaphore>,
    pub(crate) active: AtomicUsize,
    pub(crate) last_used: Mutex<Instant>,
    ready_after: Mutex<Duration>,
    exit_rx: watch::Receiver<Option<ExitInfo>>,
    kill_tx: Mutex<Option<oneshot::Sender<()>>>,
}

#[derive(Serialize, serde::Deserialize)]
struct RunRecord {
    pid: u32,
    owner_pid: u32,
    program: String,
    model: String,
    started_at: String,
}

impl Engine {
    /// Spawn the process described by `spec`. Returns once the process exists (not ready).
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        spec: &LaunchSpec,
        model: ModelEntry,
        backend: BackendKind,
        port: u16,
        profile: ResolvedProfile,
        memory: MemoryPlan,
        log_dir: &Path,
        run_dir: &Path,
    ) -> Result<Arc<Self>, RuntimeError> {
        let log_path = log_dir.join(format!("{}-{}.log", backend, sanitize(&model.id)));
        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .map_err(|e| {
                RuntimeError::EngineStart(format!("opening log {}: {e}", log_path.display()))
            })?;
        let log2 = log.try_clone().map_err(|e| RuntimeError::Other(e.into()))?;
        {
            use std::io::Write;
            let mut l = &log;
            let _ = writeln!(
                l,
                "\n===== {} launching {} {} =====",
                chrono::Utc::now().to_rfc3339(),
                spec.program.display(),
                spec.args.join(" ")
            );
        }

        let mut cmd = tokio::process::Command::new(&spec.program);
        cmd.args(&spec.args)
            .envs(spec.env.iter().cloned())
            .stdin(Stdio::null())
            .stdout(log)
            .stderr(log2)
            .kill_on_drop(true);
        #[cfg(unix)]
        cmd.process_group(0);
        // Linux: the kernel kills the engine if llmario dies without cleanup. macOS has no
        // equivalent; there, leftovers are found via run records (`llmario doctor`).
        #[cfg(target_os = "linux")]
        unsafe {
            cmd.pre_exec(|| {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let mut child = cmd.spawn().map_err(|e| {
            RuntimeError::EngineStart(format!("cannot execute {}: {e}", spec.program.display()))
        })?;
        let pid = child
            .id()
            .ok_or_else(|| RuntimeError::EngineStart("process exited immediately".into()))?;

        let record_path = run_dir.join(format!("{pid}.json"));
        let record = RunRecord {
            pid,
            owner_pid: std::process::id(),
            program: spec.program.display().to_string(),
            model: model.id.clone(),
            started_at: chrono::Utc::now().to_rfc3339(),
        };
        let _ = std::fs::write(
            &record_path,
            serde_json::to_vec(&record).unwrap_or_default(),
        );

        let (exit_tx, exit_rx) = watch::channel(None);
        let (kill_tx, kill_rx) = oneshot::channel::<()>();
        let model_id = model.id.clone();
        tokio::spawn(async move {
            let info = tokio::select! {
                st = child.wait() => {
                    let st = st.ok();
                    ExitInfo { code: st.and_then(|s| s.code()), signal: st.and_then(signal_of), requested: false }
                }
                _ = kill_rx => {
                    terminate_group(pid);
                    let st = match tokio::time::timeout(Duration::from_secs(8), child.wait()).await {
                        Ok(st) => st.ok(),
                        Err(_) => {
                            tracing::warn!(pid, "engine ignored SIGTERM; sending SIGKILL");
                            kill_group(pid);
                            let _ = child.kill().await;
                            child.wait().await.ok()
                        }
                    };
                    ExitInfo { code: st.and_then(|s| s.code()), signal: st.and_then(signal_of), requested: true }
                }
            };
            if info.requested {
                tracing::info!(model = %model_id, pid, "engine stopped");
            } else {
                tracing::error!(model = %model_id, pid, exit = %info.describe(), "engine exited unexpectedly");
            }
            let _ = std::fs::remove_file(&record_path);
            let _ = exit_tx.send(Some(info));
        });

        Ok(Arc::new(Self {
            permits: Arc::new(Semaphore::new(profile.parallel.max(1) as usize)),
            model,
            backend,
            base_url: format!("http://127.0.0.1:{port}"),
            port,
            upstream_model: spec.upstream_model.clone(),
            pid,
            profile,
            memory,
            log_path,
            notes: spec.notes.clone(),
            active: AtomicUsize::new(0),
            last_used: Mutex::new(Instant::now()),
            ready_after: Mutex::new(Duration::ZERO),
            exit_rx,
            kill_tx: Mutex::new(Some(kill_tx)),
        }))
    }

    pub fn exit_info(&self) -> Option<ExitInfo> {
        self.exit_rx.borrow().clone()
    }

    pub fn is_alive(&self) -> bool {
        self.exit_rx.borrow().is_none()
    }

    pub fn active_requests(&self) -> usize {
        self.active.load(Ordering::SeqCst)
    }

    pub fn idle_for(&self) -> Duration {
        self.last_used.lock().unwrap().elapsed()
    }

    pub fn ready_after(&self) -> Duration {
        *self.ready_after.lock().unwrap()
    }

    /// Wait for `GET health_path` → 200, then run a 1-token warm-up generation so the first
    /// user request does not pay for lazy initialisation (shader compilation, allocation).
    pub async fn wait_ready(
        &self,
        http: &reqwest::Client,
        health_path: &str,
        timeout: Duration,
        started: Instant,
    ) -> Result<(), RuntimeError> {
        let deadline = started + timeout;
        let url = format!("{}{}", self.base_url, health_path);
        let mut exit_rx = self.exit_rx.clone();
        loop {
            if let Some(info) = self.exit_info() {
                return Err(RuntimeError::EngineStart(format!(
                    "{} {} during startup. Last log lines ({}):\n{}",
                    self.backend,
                    info.describe(),
                    self.log_path.display(),
                    tail(&self.log_path, 15)
                )));
            }
            if Instant::now() > deadline {
                return Err(RuntimeError::EngineStart(format!(
                    "{} did not become healthy within {}s; see {}",
                    self.backend,
                    timeout.as_secs(),
                    self.log_path.display()
                )));
            }
            if let Ok(r) = http.get(&url).timeout(Duration::from_secs(2)).send().await {
                if r.status().is_success() {
                    break;
                }
            }
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_millis(150)) => {}
                _ = exit_rx.changed() => {}
            }
        }

        let body = serde_json::json!({
            "model": self.upstream_model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "stream": false,
        });
        let remaining = deadline
            .saturating_duration_since(Instant::now())
            .max(Duration::from_secs(5));
        let resp = http
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(&body)
            .timeout(remaining)
            .send()
            .await
            .map_err(|e| {
                RuntimeError::EngineStart(format!(
                    "warm-up request failed: {e}; see {}",
                    self.log_path.display()
                ))
            })?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(RuntimeError::EngineStart(format!(
                "warm-up returned HTTP {status}: {}",
                text.chars().take(300).collect::<String>()
            )));
        }
        *self.ready_after.lock().unwrap() = started.elapsed();
        Ok(())
    }

    /// Ask the engine to stop and wait until it has exited.
    pub async fn stop(&self) {
        if let Some(tx) = self.kill_tx.lock().unwrap().take() {
            let _ = tx.send(());
        }
        let mut rx = self.exit_rx.clone();
        while rx.borrow().is_none() {
            if rx.changed().await.is_err() {
                break;
            }
        }
    }

    pub fn info(&self) -> EngineInfo {
        EngineInfo {
            model: self.model.id.clone(),
            backend: self.backend,
            pid: self.pid,
            port: self.port,
            profile: self.profile.clone(),
            active_requests: self.active_requests(),
            ready_seconds: self.ready_after().as_secs_f64(),
            idle_seconds: self.idle_for().as_secs_f64(),
            estimated_bytes: self.memory.total_bytes,
            resident_bytes: llmario_hardware::process_memory_bytes(self.pid),
            log: self.log_path.clone(),
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if let Some(tx) = self.kill_tx.lock().unwrap().take() {
            let _ = tx.send(());
        }
    }
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(unix)]
fn signal_of(s: std::process::ExitStatus) -> Option<i32> {
    use std::os::unix::process::ExitStatusExt;
    s.signal()
}
#[cfg(not(unix))]
fn signal_of(_: std::process::ExitStatus) -> Option<i32> {
    None
}

fn terminate_group(pid: u32) {
    // SAFETY: plain syscall; negative pid targets the process group we created.
    unsafe {
        libc::kill(-(pid as i32), libc::SIGTERM);
    }
}
fn kill_group(pid: u32) {
    // SAFETY: as above.
    unsafe {
        libc::kill(-(pid as i32), libc::SIGKILL);
    }
}

/// Last `n` lines of a log file (startup diagnostics only).
pub fn tail(path: &Path, n: usize) -> String {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

/// Pick a free loopback port. There is a small race between release and the engine's bind;
/// the supervisor retries the launch once if the engine exits during startup.
pub fn free_port() -> std::io::Result<u16> {
    let l = std::net::TcpListener::bind("127.0.0.1:0")?;
    Ok(l.local_addr()?.port())
}

/// Engines recorded under `run_dir` whose owning llmario process is gone but which are still
/// running. Reported by `doctor`; never killed automatically.
pub fn find_orphans(run_dir: &Path) -> Vec<(u32, String, String)> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(run_dir) else {
        return out;
    };
    for e in rd.flatten() {
        let Ok(text) = std::fs::read(e.path()) else {
            continue;
        };
        let Ok(r) = serde_json::from_slice::<RunRecord>(&text) else {
            continue;
        };
        let alive = |pid: u32| unsafe { libc::kill(pid as i32, 0) == 0 };
        if !alive(r.owner_pid) {
            if alive(r.pid) {
                out.push((r.pid, r.model, r.program));
            } else {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }
    out
}
