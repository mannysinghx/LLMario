//! macOS apps started from Finder/Dock inherit launchd's minimal PATH
//! (`/usr/bin:/bin:/usr/sbin:/sbin`). Engines installed with Homebrew or python.org live
//! elsewhere, so we adopt the user's login-shell PATH, as terminal `llmario` would see it.

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const MARKER: &str = "__LLMARIO_PATH__";
const FALLBACK: &[&str] = &["/opt/homebrew/bin", "/usr/local/bin"];

pub fn inherit_login_path() {
    let current = std::env::var("PATH").unwrap_or_default();
    let mut parts: Vec<String> = Vec::new();
    if let Some(login) = login_shell_path() {
        parts.extend(login.split(':').map(String::from));
    }
    parts.extend(current.split(':').map(String::from));
    parts.extend(FALLBACK.iter().map(|s| s.to_string()));
    let mut seen = std::collections::HashSet::new();
    parts.retain(|p| !p.is_empty() && seen.insert(p.clone()));
    // Called at the top of main, before Tauri or tokio start any threads.
    std::env::set_var("PATH", parts.join(":"));
}

fn login_shell_path() -> Option<String> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/zsh".into());
    let mut child = Command::new(shell)
        .args(["-ilc", &format!("printf '{MARKER}%s' \"$PATH\"")])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    // Never let a slow or interactive shell profile block app startup.
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(20)),
            _ => {
                let _ = child.kill();
                return None;
            }
        }
    }
    let mut out = String::new();
    child.stdout.take()?.read_to_string(&mut out).ok()?;
    out.rsplit_once(MARKER)
        .map(|(_, p)| p.trim().to_string())
        .filter(|p| !p.is_empty())
}
