//! Sandbox launch plans for stdio MCP servers and the fetcher subprocess.
//!
//! * macOS: `sandbox-exec -p <Seatbelt profile>`; filesystem read-only except the scratch dir,
//!   network denied except the configured egress-proxy socket (and optional loopback ports).
//! * Linux: `bwrap` with a read-only root, a writable scratch dir, a fresh network namespace
//!   (the proxy's Unix socket is bind-mounted in, and works across namespaces).
//! * Windows: `Unsupported` (AppContainer / restricted token lands with the server integration).
//!
//! The functions here only *render* plans; nothing is executed, so tests run without the
//! sandbox binaries. The exact command, untruncated, is available through [`display_command`]
//! for the approval dialog, as the MCP security guidance requires.

use std::path::{Path, PathBuf};

use crate::ToolsError;

/// What the sandboxed process may touch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxSpec {
    /// The only writable directory (created by the caller).
    pub scratch_dir: PathBuf,
    /// Unix socket of the local egress proxy; the only permitted network destination.
    pub proxy_socket: Option<PathBuf>,
    /// Additional writable directories (e.g. a server's own cache dir).
    pub extra_writable: Vec<PathBuf>,
    /// Loopback TCP ports the process may connect to (e.g. a local SearXNG). On Linux this
    /// keeps the host network namespace (bubblewrap cannot filter by port), so prefer the proxy.
    pub allow_loopback_tcp_ports: Vec<u16>,
}

impl SandboxSpec {
    /// Writable scratch dir only, no network.
    pub fn new(scratch_dir: impl Into<PathBuf>) -> Self {
        Self {
            scratch_dir: scratch_dir.into(),
            proxy_socket: None,
            extra_writable: Vec::new(),
            allow_loopback_tcp_ports: Vec::new(),
        }
    }

    /// Allow the egress proxy socket.
    pub fn with_proxy_socket(mut self, socket: impl Into<PathBuf>) -> Self {
        self.proxy_socket = Some(socket.into());
        self
    }

    /// Allow another writable directory.
    pub fn with_writable(mut self, dir: impl Into<PathBuf>) -> Self {
        self.extra_writable.push(dir.into());
        self
    }

    /// Allow a loopback TCP port.
    pub fn with_loopback_port(mut self, port: u16) -> Self {
        self.allow_loopback_tcp_ports.push(port);
        self
    }

    fn validate(&self) -> Result<(), ToolsError> {
        for p in std::iter::once(&self.scratch_dir)
            .chain(self.extra_writable.iter())
            .chain(self.proxy_socket.iter())
        {
            if !p.is_absolute() {
                return Err(ToolsError::Sandbox(format!(
                    "path must be absolute: {}",
                    p.display()
                )));
            }
        }
        Ok(())
    }
}

/// The program to launch (before sandboxing).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CommandSpec {
    /// Program path or name.
    pub program: String,
    /// Arguments.
    pub args: Vec<String>,
    /// The complete environment of the child (an allowlist resolved by the caller).
    pub env: Vec<(String, String)>,
    /// Working directory.
    pub cwd: Option<PathBuf>,
}

/// Target platform for [`plan_for`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetOs {
    /// Seatbelt via `sandbox-exec`.
    MacOs,
    /// bubblewrap.
    Linux,
    /// No sandbox available yet.
    Windows,
}

impl TargetOs {
    /// The platform this binary runs on.
    pub fn current() -> Self {
        if cfg!(target_os = "macos") {
            TargetOs::MacOs
        } else if cfg!(target_os = "linux") {
            TargetOs::Linux
        } else {
            TargetOs::Windows
        }
    }
}

/// A fully rendered launch: the wrapper program and all its arguments. The child environment
/// is applied by [`launch`] (`env_clear` + the spec's allowlist).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchPlan {
    /// Program to execute (`/usr/bin/sandbox-exec`, `bwrap`).
    pub program: String,
    /// Arguments, in order.
    pub args: Vec<String>,
    /// Environment for the child.
    pub env: Vec<(String, String)>,
    /// Working directory.
    pub cwd: Option<PathBuf>,
}

/// Render the Seatbelt (SBPL) profile for `spec`.
pub fn macos_profile(spec: &SandboxSpec) -> String {
    let mut p = String::new();
    p.push_str("(version 1)\n");
    p.push_str("(deny default)\n");
    p.push_str(";; process: may exec the server binary and fork helpers\n");
    p.push_str(
        "(allow process-exec*)\n(allow process-fork)\n(allow signal (target same-sandbox))\n",
    );
    p.push_str(";; read-only system access\n");
    p.push_str("(allow sysctl-read)\n(allow mach-lookup (global-name \"com.apple.system.logger\") (global-name \"com.apple.system.notification_center\"))\n");
    p.push_str("(allow file-read*)\n");
    p.push_str("(allow file-write-data (literal \"/dev/null\"))\n(allow file-ioctl (literal \"/dev/null\"))\n");
    p.push_str(";; writable scratch\n");
    for dir in std::iter::once(&spec.scratch_dir).chain(spec.extra_writable.iter()) {
        p.push_str(&format!(
            "(allow file-write* (subpath {}))\n",
            sbpl_string(dir)
        ));
    }
    p.push_str(
        ";; network: deny everything, then open the egress proxy socket and loopback ports\n",
    );
    p.push_str("(deny network*)\n");
    if let Some(sock) = &spec.proxy_socket {
        p.push_str(&format!(
            "(allow network-outbound (literal {}))\n",
            sbpl_string(sock)
        ));
    }
    for port in &spec.allow_loopback_tcp_ports {
        p.push_str(&format!(
            "(allow network-outbound (remote tcp \"localhost:{port}\"))\n"
        ));
    }
    p
}

fn sbpl_string(path: &Path) -> String {
    let raw = path.to_string_lossy();
    let mut s = String::with_capacity(raw.len() + 2);
    s.push('"');
    for c in raw.chars() {
        match c {
            '"' => s.push_str("\\\""),
            '\\' => s.push_str("\\\\"),
            c => s.push(c),
        }
    }
    s.push('"');
    s
}

/// Render the `bwrap` argument list (without the program name) for `spec` + `cmd`.
pub fn bwrap_args(spec: &SandboxSpec, cmd: &CommandSpec) -> Vec<String> {
    let mut a: Vec<String> = Vec::new();
    a.extend(
        [
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--tmpfs",
            "/tmp",
        ]
        .map(String::from),
    );
    for dir in std::iter::once(&spec.scratch_dir).chain(spec.extra_writable.iter()) {
        let d = dir.to_string_lossy().into_owned();
        a.extend(["--bind".to_string(), d.clone(), d]);
    }
    if let Some(sock) = &spec.proxy_socket {
        let s = sock.to_string_lossy().into_owned();
        a.extend(["--bind".to_string(), s.clone(), s]);
    }
    a.push("--unshare-all".into());
    if !spec.allow_loopback_tcp_ports.is_empty() {
        a.push("--share-net".into());
    }
    a.extend(["--die-with-parent", "--new-session", "--clearenv"].map(String::from));
    for (k, v) in &cmd.env {
        a.extend(["--setenv".to_string(), k.clone(), v.clone()]);
    }
    if let Some(cwd) = &cmd.cwd {
        a.extend(["--chdir".to_string(), cwd.to_string_lossy().into_owned()]);
    }
    a.push("--".into());
    a.push(cmd.program.clone());
    a.extend(cmd.args.iter().cloned());
    a
}

/// Build the launch plan for a platform.
pub fn plan_for(
    os: TargetOs,
    spec: &SandboxSpec,
    cmd: &CommandSpec,
) -> Result<LaunchPlan, ToolsError> {
    spec.validate()?;
    match os {
        TargetOs::MacOs => {
            let mut args = vec![
                "-p".to_string(),
                macos_profile(spec),
                "--".to_string(),
                cmd.program.clone(),
            ];
            args.extend(cmd.args.iter().cloned());
            Ok(LaunchPlan {
                program: "/usr/bin/sandbox-exec".into(),
                args,
                env: cmd.env.clone(),
                cwd: cmd.cwd.clone(),
            })
        }
        TargetOs::Linux => Ok(LaunchPlan {
            program: "bwrap".into(),
            args: bwrap_args(spec, cmd),
            // bwrap gets --clearenv/--setenv; its own environment is irrelevant to the child.
            env: cmd.env.clone(),
            cwd: cmd.cwd.clone(),
        }),
        TargetOs::Windows => Err(ToolsError::Sandbox(
            "Unsupported: no sandbox on Windows yet (AppContainer / restricted token planned)"
                .into(),
        )),
    }
}

/// Build a `std::process::Command` running `cmd` under the sandbox on the current platform.
/// The environment is cleared and replaced by `cmd.env`.
pub fn launch(spec: &SandboxSpec, cmd: &CommandSpec) -> Result<std::process::Command, ToolsError> {
    let plan = plan_for(TargetOs::current(), spec, cmd)?;
    Ok(command_from_plan(&plan))
}

/// Build a `std::process::Command` from a plan (no sandbox check; used by [`launch`] and tests).
pub fn command_from_plan(plan: &LaunchPlan) -> std::process::Command {
    let mut c = std::process::Command::new(&plan.program);
    c.args(&plan.args);
    c.env_clear();
    c.envs(plan.env.iter().map(|(k, v)| (k.as_str(), v.as_str())));
    if let Some(cwd) = &plan.cwd {
        c.current_dir(cwd);
    }
    c
}

/// The exact command line, untruncated and shell-quoted, for display before approval.
pub fn display_command(program: &str, args: &[String]) -> String {
    std::iter::once(program)
        .chain(args.iter().map(String::as_str))
        .map(shell_quote)
        .collect::<Vec<_>>()
        .join(" ")
}

fn shell_quote(s: &str) -> String {
    if !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_./:=@%+,".contains(&b))
    {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', "'\\''"))
    }
}

/// Patterns the approval dialog should highlight as dangerous in a server command.
pub fn dangerous_patterns(program: &str, args: &[String]) -> Vec<&'static str> {
    let joined = std::iter::once(program)
        .chain(args.iter().map(String::as_str))
        .collect::<Vec<_>>()
        .join(" ");
    let mut hits = Vec::new();
    for (needle, label) in [
        ("sudo", "sudo"),
        ("rm -rf", "rm -rf"),
        ("curl ", "network download (curl)"),
        ("wget ", "network download (wget)"),
        (".ssh", "SSH keys"),
        ("| sh", "pipe to shell"),
        ("| bash", "pipe to shell"),
        ("chmod ", "permission change"),
    ] {
        if joined.contains(needle) {
            hits.push(label);
        }
    }
    hits
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> SandboxSpec {
        SandboxSpec::new("/tmp/llmario/scratch")
            .with_proxy_socket("/tmp/llmario/egress.sock")
            .with_writable("/tmp/llmario/cache \"q\"")
            .with_loopback_port(8080)
    }

    fn cmd() -> CommandSpec {
        CommandSpec {
            program: "npx".into(),
            args: vec![
                "-y".into(),
                "@modelcontextprotocol/server-filesystem".into(),
                "/tmp/data dir".into(),
            ],
            env: vec![("PATH".into(), "/usr/bin".into())],
            cwd: Some("/tmp".into()),
        }
    }

    #[test]
    fn macos_profile_matches_documented_shape() {
        let p = macos_profile(&spec());
        assert!(p.starts_with("(version 1)\n(deny default)\n"));
        assert!(p.contains("(allow file-read*)"));
        assert!(p.contains("(allow file-write* (subpath \"/tmp/llmario/scratch\"))"));
        assert!(p.contains("(allow file-write* (subpath \"/tmp/llmario/cache \\\"q\\\"\"))"));
        assert!(p.contains("(deny network*)"));
        assert!(p.contains("(allow network-outbound (literal \"/tmp/llmario/egress.sock\"))"));
        assert!(p.contains("(allow network-outbound (remote tcp \"localhost:8080\"))"));
        // deny network* precedes the allows so the allows win (later rules take precedence).
        assert!(p.find("(deny network*)").unwrap() < p.find("(allow network-outbound").unwrap());
        // No proxy, no ports: nothing opened.
        let p = macos_profile(&SandboxSpec::new("/s"));
        assert!(!p.contains("network-outbound"));
    }

    #[test]
    fn bwrap_args_match_documented_shape() {
        let a = bwrap_args(&spec(), &cmd());
        let expect_prefix = [
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--tmpfs",
            "/tmp",
            "--bind",
            "/tmp/llmario/scratch",
            "/tmp/llmario/scratch",
            "--bind",
            "/tmp/llmario/cache \"q\"",
            "/tmp/llmario/cache \"q\"",
            "--bind",
            "/tmp/llmario/egress.sock",
            "/tmp/llmario/egress.sock",
            "--unshare-all",
            "--share-net",
            "--die-with-parent",
            "--new-session",
            "--clearenv",
            "--setenv",
            "PATH",
            "/usr/bin",
            "--chdir",
            "/tmp",
            "--",
            "npx",
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "/tmp/data dir",
        ];
        assert_eq!(
            a,
            expect_prefix
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        );
        let no_ports = SandboxSpec::new("/s");
        assert!(!bwrap_args(&no_ports, &cmd()).contains(&"--share-net".to_string()));
    }

    // Unix paths: the sandbox plans target macOS and Linux, and the engine is not built for Windows.
    #[cfg(unix)]
    #[test]
    fn plans_per_platform() {
        let mac = plan_for(TargetOs::MacOs, &spec(), &cmd()).unwrap();
        assert_eq!(mac.program, "/usr/bin/sandbox-exec");
        assert_eq!(mac.args[0], "-p");
        assert_eq!(mac.args[1], macos_profile(&spec()));
        assert_eq!(
            &mac.args[2..],
            &[
                "--",
                "npx",
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "/tmp/data dir"
            ]
        );
        assert_eq!(mac.env, cmd().env);
        let linux = plan_for(TargetOs::Linux, &spec(), &cmd()).unwrap();
        assert_eq!(linux.program, "bwrap");
        assert_eq!(linux.args, bwrap_args(&spec(), &cmd()));
        let win = plan_for(TargetOs::Windows, &spec(), &cmd());
        assert!(matches!(win, Err(ToolsError::Sandbox(m)) if m.starts_with("Unsupported")));
        let rel = plan_for(TargetOs::MacOs, &SandboxSpec::new("relative/dir"), &cmd());
        assert!(matches!(rel, Err(ToolsError::Sandbox(_))));
    }

    #[cfg(unix)]
    #[test]
    fn command_from_plan_clears_env_and_sets_cwd() {
        let plan = plan_for(TargetOs::MacOs, &spec(), &cmd()).unwrap();
        let c = command_from_plan(&plan);
        assert_eq!(c.get_program(), "/usr/bin/sandbox-exec");
        assert_eq!(c.get_current_dir(), Some(Path::new("/tmp")));
        let envs: Vec<_> = c.get_envs().collect();
        assert_eq!(envs.len(), 1);
        assert_eq!(envs[0].0, "PATH");
    }

    #[test]
    fn display_is_untruncated_and_quoted() {
        let c = cmd();
        let d = display_command(&c.program, &c.args);
        assert_eq!(
            d,
            "npx -y @modelcontextprotocol/server-filesystem '/tmp/data dir'"
        );
        assert_eq!(display_command("x", &["it's".to_string()]), "x 'it'\\''s'");
        assert_eq!(
            dangerous_patterns("sh", &["-c".into(), "curl http://x | sh".into()]),
            vec!["network download (curl)", "pipe to shell"]
        );
    }
}
