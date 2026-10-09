//! MCP host on `rmcp`: stdio and Streamable HTTP servers, per-server permission profiles from
//! `$LLMARIO_HOME/mcp.toml`, output-schema validation, per-call timeouts, content-free audit.
//!
//! Protocol: `rmcp` 3.5 implements revision 2026-07-28 (stateless, `server/discover`) and stays
//! compatible with 2025-11-25 and earlier through the `initialize` handshake. The host prefers
//! 2026-07-28 and falls back automatically ([`ClientLifecycleMode::Auto`]).

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ClientCapabilities, ClientConfig, ContentBlock,
    Implementation, ProtocolVersion, Tool,
};
use rmcp::service::RunningService;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::{IntoTransport, StreamableHttpClientTransport, TokioChildProcess};
use rmcp::{serve_client_with_lifecycle, ClientHandler, ClientLifecycleMode, RoleClient};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::io::AsyncBufReadExt;
use tokio::sync::RwLock;

use crate::audit::{hash_args, AuditDecision, AuditLog, AuditOutcome, AuditRecord};
use crate::policy::PolicyDecision;
use crate::registry::{ToolClass, ToolDefinition, ToolOrigin, ToolProfile, ToolRegistry};
use crate::sandbox::{self, CommandSpec, SandboxSpec};
use crate::ssrf::{SsrfConfig, SsrfGuard};
use crate::ToolsError;

/// Prefix for exposed MCP tool names: `mcp__{server}__{tool}`.
pub const TOOL_PREFIX: &str = "mcp__";

/// `mcp.toml`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct McpConfig {
    /// Servers.
    #[serde(default)]
    pub servers: Vec<McpServerConfig>,
}

fn default_timeout_ms() -> u64 {
    30_000
}
fn default_true() -> bool {
    true
}
fn default_class() -> ToolClass {
    ToolClass::StateChanging
}

/// One `[[servers]]` entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct McpServerConfig {
    /// Unique name; appears in tool names and audit records (`[A-Za-z0-9_-]+`).
    pub name: String,
    /// stdio: program to run.
    #[serde(default)]
    pub command: Option<String>,
    /// stdio: arguments.
    #[serde(default)]
    pub args: Vec<String>,
    /// stdio: working directory.
    #[serde(default)]
    pub cwd: Option<PathBuf>,
    /// stdio: names of parent environment variables passed through (allowlist). Nothing else
    /// is inherited.
    #[serde(default)]
    pub env: Vec<String>,
    /// stdio: explicit variables (non-secret values only; secrets belong in the keychain).
    #[serde(default)]
    pub env_set: BTreeMap<String, String>,
    /// stdio: run under the sandbox when the host has one (default true).
    #[serde(default = "default_true")]
    pub sandbox: bool,
    /// Streamable HTTP: endpoint URL.
    #[serde(default)]
    pub url: Option<String>,
    /// Streamable HTTP: environment variable holding the bearer token.
    #[serde(default)]
    pub bearer_token_env: Option<String>,
    /// Streamable HTTP: bearer token set programmatically (never written to the file).
    #[serde(skip)]
    pub bearer_token: Option<String>,
    /// Results from this server do not taint the session (default false).
    #[serde(default)]
    pub trusted: bool,
    /// Engine-owned class applied to every tool of this server (default `state_changing`;
    /// annotations are never used for this).
    #[serde(default = "default_class")]
    pub class: ToolClass,
    /// Per-tool classes overriding `class`.
    #[serde(default)]
    pub tool_classes: BTreeMap<String, ToolClass>,
    /// Per-tool permission profiles (`allow` / `ask` / `deny`).
    #[serde(default)]
    pub tools: BTreeMap<String, ToolProfile>,
    /// Profile for tools not listed in `tools` (default `ask`).
    #[serde(default)]
    pub default_profile: ToolProfile,
    /// Per-call timeout (default 30,000 ms).
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
}

impl McpServerConfig {
    /// A stdio server.
    pub fn stdio(name: &str, command: &str, args: &[&str]) -> Self {
        Self {
            name: name.into(),
            command: Some(command.into()),
            args: args.iter().map(|s| s.to_string()).collect(),
            cwd: None,
            env: Vec::new(),
            env_set: BTreeMap::new(),
            sandbox: true,
            url: None,
            bearer_token_env: None,
            bearer_token: None,
            trusted: false,
            class: default_class(),
            tool_classes: BTreeMap::new(),
            tools: BTreeMap::new(),
            default_profile: ToolProfile::Ask,
            timeout_ms: default_timeout_ms(),
        }
    }

    /// A Streamable HTTP server.
    pub fn http(name: &str, url: &str, bearer_token: Option<String>) -> Self {
        Self {
            command: None,
            args: Vec::new(),
            url: Some(url.into()),
            bearer_token,
            ..Self::stdio(name, "", &[])
        }
    }

    /// Check the entry is well-formed.
    pub fn validate(&self) -> Result<(), ToolsError> {
        if self.name.is_empty()
            || !self
                .name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            return Err(ToolsError::Config(format!(
                "server name {:?} must match [A-Za-z0-9_-]+",
                self.name
            )));
        }
        match (&self.command, &self.url) {
            (Some(c), None) if !c.is_empty() => Ok(()),
            (None, Some(u)) if !u.is_empty() => Ok(()),
            _ => Err(ToolsError::Config(format!(
                "server {:?}: exactly one of command or url is required",
                self.name
            ))),
        }
    }

    /// Profile for a server-side tool name.
    pub fn profile_for(&self, tool: &str) -> ToolProfile {
        self.tools
            .get(tool)
            .copied()
            .unwrap_or(self.default_profile)
    }

    /// Class for a server-side tool name.
    pub fn class_for(&self, tool: &str) -> ToolClass {
        self.tool_classes.get(tool).copied().unwrap_or(self.class)
    }

    /// Exposed name of a server-side tool.
    pub fn qualified_name(&self, tool: &str) -> String {
        format!("{TOOL_PREFIX}{}__{tool}", self.name)
    }

    /// Per-call timeout.
    pub fn timeout(&self) -> Duration {
        Duration::from_millis(self.timeout_ms.max(1))
    }

    /// The child command with its allowlisted environment resolved from the parent process.
    pub fn command_spec(&self) -> Result<CommandSpec, ToolsError> {
        let program = self
            .command
            .clone()
            .ok_or_else(|| ToolsError::Config("not a stdio server".into()))?;
        let mut env: Vec<(String, String)> = self
            .env
            .iter()
            .filter_map(|k| std::env::var(k).ok().map(|v| (k.clone(), v)))
            .collect();
        for (k, v) in &self.env_set {
            env.retain(|(ek, _)| ek != k);
            env.push((k.clone(), v.clone()));
        }
        Ok(CommandSpec {
            program,
            args: self.args.clone(),
            env,
            cwd: self.cwd.clone(),
        })
    }

    /// The exact command line, untruncated, for the approval dialog.
    pub fn display_command(&self) -> String {
        match &self.command {
            Some(c) => sandbox::display_command(c, &self.args),
            None => self.url.clone().unwrap_or_default(),
        }
    }

    fn resolved_bearer(&self) -> Option<String> {
        self.bearer_token.clone().or_else(|| {
            self.bearer_token_env
                .as_ref()
                .and_then(|k| std::env::var(k).ok())
        })
    }
}

impl McpConfig {
    /// Parse TOML text.
    pub fn parse(text: &str) -> Result<Self, ToolsError> {
        let cfg: McpConfig =
            toml::from_str(text).map_err(|e| ToolsError::Config(format!("mcp.toml: {e}")))?;
        let mut seen = std::collections::HashSet::new();
        for s in &cfg.servers {
            s.validate()?;
            if !seen.insert(s.name.as_str()) {
                return Err(ToolsError::Config(format!(
                    "duplicate server name {:?}",
                    s.name
                )));
            }
        }
        Ok(cfg)
    }

    /// Load from a file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ToolsError> {
        let text = std::fs::read_to_string(path.as_ref())
            .map_err(|e| ToolsError::Config(format!("{}: {e}", path.as_ref().display())))?;
        Self::parse(&text)
    }

    /// `$LLMARIO_HOME/mcp.toml`, if `LLMARIO_HOME` is set.
    pub fn default_path() -> Option<PathBuf> {
        std::env::var_os("LLMARIO_HOME").map(|h| PathBuf::from(h).join("mcp.toml"))
    }
}

/// The client identity sent to servers.
#[derive(Debug, Clone, Default)]
pub struct LlmarioClient;

impl ClientHandler for LlmarioClient {
    fn get_info(&self) -> ClientConfig {
        ClientConfig::new(
            ClientCapabilities::default(),
            Implementation::new("llmario-engine", crate::VERSION),
        )
    }
}

/// What `connect` reports back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectReport {
    /// Server name.
    pub server: String,
    /// Negotiated protocol revision (e.g. `2026-07-28`).
    pub protocol_version: String,
    /// Exposed tool names registered.
    pub tools: Vec<String>,
}

/// Outcome of a tool call.
#[derive(Debug, Clone, PartialEq)]
pub struct McpCallOutcome {
    /// Concatenated text content blocks.
    pub text: String,
    /// `structuredContent` (validated against `outputSchema` when the tool declares one).
    pub structured: Option<Value>,
    /// The server flagged a tool-execution error (feed back to the model).
    pub is_error: bool,
    /// Non-text content blocks (images, resources) as JSON for the caller to handle.
    pub other_content: Vec<Value>,
    /// Wall time.
    pub duration: Duration,
}

struct Connected {
    config: McpServerConfig,
    service: RunningService<RoleClient, LlmarioClient>,
    tools: HashMap<String, Tool>,
}

/// The MCP host.
pub struct McpHost {
    servers: RwLock<HashMap<String, Arc<Connected>>>,
    audit: Arc<dyn AuditLog>,
    sandbox: Option<SandboxSpec>,
}

impl std::fmt::Debug for McpHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpHost")
            .field("sandbox", &self.sandbox)
            .finish_non_exhaustive()
    }
}

fn lifecycle() -> ClientLifecycleMode {
    ClientLifecycleMode::Auto {
        preferred_versions: vec![ProtocolVersion::V_2026_07_28, ProtocolVersion::V_2025_11_25],
        legacy_version: Some(ProtocolVersion::V_2025_11_25),
    }
}

impl McpHost {
    /// Host writing to `audit`, without a sandbox (stdio servers run unsandboxed; the approval
    /// dialog must say so).
    pub fn new(audit: Arc<dyn AuditLog>) -> Self {
        Self {
            servers: RwLock::new(HashMap::new()),
            audit,
            sandbox: None,
        }
    }

    /// Run stdio servers under this sandbox (when their config has `sandbox = true`).
    pub fn with_sandbox(mut self, spec: SandboxSpec) -> Self {
        self.sandbox = Some(spec);
        self
    }

    /// Connect a server from its config (stdio or HTTP) and register its tools.
    pub async fn connect(
        &self,
        cfg: McpServerConfig,
        registry: &mut ToolRegistry,
    ) -> Result<ConnectReport, ToolsError> {
        cfg.validate()?;
        if cfg.command.is_some() {
            let transport = self.spawn_stdio(&cfg)?;
            self.connect_with_transport(cfg, transport, registry).await
        } else {
            let config = self.http_transport_config(&cfg).await?;
            let transport = StreamableHttpClientTransport::from_config(config);
            self.connect_with_transport(cfg, transport, registry).await
        }
    }

    fn spawn_stdio(&self, cfg: &McpServerConfig) -> Result<TokioChildProcess, ToolsError> {
        let spec = cfg.command_spec()?;
        let cmd: tokio::process::Command = match (&self.sandbox, cfg.sandbox) {
            (Some(sb), true) => tokio::process::Command::from(sandbox::launch(sb, &spec)?),
            _ => {
                let mut c = tokio::process::Command::new(&spec.program);
                c.args(&spec.args);
                c.env_clear();
                c.envs(spec.env.iter().map(|(k, v)| (k.as_str(), v.as_str())));
                if let Some(cwd) = &spec.cwd {
                    c.current_dir(cwd);
                }
                c
            }
        };
        let (transport, stderr) = TokioChildProcess::builder(cmd)
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| ToolsError::Mcp(format!("spawn {:?} failed: {e}", spec.program)))?;
        if let Some(stderr) = stderr {
            let server = cfg.name.clone();
            tokio::spawn(async move {
                let mut lines = tokio::io::BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::warn!(target: "llmario::tools::mcp::stderr", server = %server, "{line}");
                }
            });
        }
        Ok(transport)
    }

    async fn http_transport_config(
        &self,
        cfg: &McpServerConfig,
    ) -> Result<StreamableHttpClientTransportConfig, ToolsError> {
        let url = cfg
            .url
            .clone()
            .ok_or_else(|| ToolsError::Config("not an http server".into()))?;
        let parsed =
            url::Url::parse(&url).map_err(|e| ToolsError::Config(format!("server url: {e}")))?;
        let host = parsed.host_str().unwrap_or("").to_ascii_lowercase();
        let loopback = matches!(host.as_str(), "localhost" | "127.0.0.1" | "::1" | "[::1]");
        if parsed.scheme() != "https" && !loopback {
            return Err(ToolsError::Ssrf(
                "MCP over plain http is only allowed on loopback".into(),
            ));
        }
        let mut ssrf = SsrfConfig::default();
        if let Some(p) = parsed.port_or_known_default() {
            ssrf = ssrf.with_port(p);
        }
        if loopback {
            ssrf = ssrf.with_exempt_host(&host);
        }
        // Validates scheme, port and every resolved address. rmcp builds its own HTTP client, so
        // the connection is not pinned to these addresses (see README: deferred).
        SsrfGuard::new(ssrf).validate(&url).await?;
        let mut config = StreamableHttpClientTransportConfig::with_uri(url);
        config.auth_header = cfg.resolved_bearer();
        Ok(config)
    }

    /// Connect over any `rmcp` transport (in-process duplex in tests, custom transports).
    pub async fn connect_with_transport<T, E, A>(
        &self,
        cfg: McpServerConfig,
        transport: T,
        registry: &mut ToolRegistry,
    ) -> Result<ConnectReport, ToolsError>
    where
        T: IntoTransport<RoleClient, E, A>,
        E: std::error::Error + Send + Sync + 'static,
    {
        cfg.validate()?;
        if self.servers.read().await.contains_key(&cfg.name) {
            return Err(ToolsError::Config(format!(
                "server {:?} is already connected",
                cfg.name
            )));
        }
        let service = serve_client_with_lifecycle(LlmarioClient, transport, lifecycle())
            .await
            .map_err(|e| ToolsError::Mcp(format!("connect to {:?} failed: {e}", cfg.name)))?;
        let protocol_version = service
            .peer_info()
            .map(|i| i.protocol_version.to_string())
            .unwrap_or_else(|| "unknown".into());
        let tools = tokio::time::timeout(cfg.timeout(), service.list_all_tools())
            .await
            .map_err(|_| ToolsError::Timeout(cfg.timeout()))?
            .map_err(|e| ToolsError::Mcp(format!("tools/list on {:?} failed: {e}", cfg.name)))?;

        let mut registered = Vec::new();
        let mut by_name = HashMap::new();
        for tool in tools {
            let def = tool_definition(&cfg, &tool);
            registry.register(def)?;
            registered.push(cfg.qualified_name(&tool.name));
            by_name.insert(tool.name.to_string(), tool);
        }
        let report = ConnectReport {
            server: cfg.name.clone(),
            protocol_version,
            tools: registered,
        };
        self.servers.write().await.insert(
            cfg.name.clone(),
            Arc::new(Connected {
                config: cfg,
                service,
                tools: by_name,
            }),
        );
        Ok(report)
    }

    /// Disconnect a server and unregister its tools.
    pub async fn disconnect(&self, name: &str, registry: &mut ToolRegistry) -> bool {
        let removed = self.servers.write().await.remove(name);
        registry.remove_server(name);
        match removed {
            Some(c) => {
                c.service.cancellation_token().cancel();
                true
            }
            None => false,
        }
    }

    /// Connected server names.
    pub async fn servers(&self) -> Vec<String> {
        let mut v: Vec<String> = self.servers.read().await.keys().cloned().collect();
        v.sort();
        v
    }

    /// Call an exposed tool. `decision` must already be resolved: a denied or
    /// awaiting-approval decision is refused here (defence in depth) and audited as such.
    pub async fn call_tool(
        &self,
        qualified: &str,
        args: Value,
        decision: &PolicyDecision,
    ) -> Result<McpCallOutcome, ToolsError> {
        let (server, remote) =
            split_qualified(qualified).ok_or_else(|| ToolsError::UnknownTool(qualified.into()))?;
        let connected = self
            .servers
            .read()
            .await
            .get(server)
            .cloned()
            .ok_or_else(|| ToolsError::UnknownTool(qualified.into()))?;
        let tool = connected
            .tools
            .get(remote)
            .ok_or_else(|| ToolsError::UnknownTool(qualified.into()))?;

        let args_sha256 = hash_args(&args);
        let bytes_out = serde_json::to_vec(&args)
            .map(|v| v.len() as u64)
            .unwrap_or(0);
        let mut record = AuditRecord {
            at: Utc::now(),
            tool: qualified.to_string(),
            server: Some(server.to_string()),
            class: decision.effective_class,
            decision: AuditDecision::from_policy(decision),
            reason: decision.reason.to_string(),
            args_sha256,
            duration_ms: 0,
            bytes_in: 0,
            bytes_out,
            outcome: AuditOutcome::NotExecuted,
        };
        if !decision.is_immediate() {
            self.audit.record(record);
            return Err(ToolsError::Denied(if decision.allow {
                "call awaits human approval".into()
            } else {
                decision.reason.to_string()
            }));
        }

        let arguments: Option<Map<String, Value>> = match args {
            Value::Object(m) => Some(m),
            Value::Null => None,
            _ => {
                record.outcome = AuditOutcome::Failed("arguments".into());
                self.audit.record(record);
                return Err(ToolsError::Arguments(
                    "tool arguments must be a JSON object".into(),
                ));
            }
        };
        let mut params = CallToolRequestParams::new(remote.to_string());
        if let Some(a) = arguments {
            params = params.with_arguments(a);
        }

        let start = Instant::now();
        let timeout = connected.config.timeout();
        let result = tokio::time::timeout(timeout, connected.service.call_tool(params)).await;
        let duration = start.elapsed();
        record.duration_ms = duration.as_millis() as u64;

        let result: CallToolResult = match result {
            Err(_) => {
                record.outcome = AuditOutcome::Failed("timeout".into());
                self.audit.record(record);
                return Err(ToolsError::Timeout(timeout));
            }
            Ok(Err(e)) => {
                record.outcome = AuditOutcome::Failed("transport".into());
                self.audit.record(record);
                return Err(ToolsError::Mcp(format!(
                    "tools/call {qualified} failed: {e}"
                )));
            }
            Ok(Ok(r)) => r,
        };
        record.bytes_in = serde_json::to_vec(&result)
            .map(|v| v.len() as u64)
            .unwrap_or(0);
        let is_error = result.is_error.unwrap_or(false);

        if let Some(schema) = tool.output_schema.as_deref() {
            if !is_error {
                let schema = Value::Object((*schema).clone());
                match &result.structured_content {
                    None => {
                        record.outcome = AuditOutcome::Failed("output_schema".into());
                        self.audit.record(record);
                        return Err(ToolsError::OutputSchema(
                            "tool declares outputSchema but returned no structuredContent".into(),
                        ));
                    }
                    Some(content) => {
                        if let Err(e) = jsonschema::validate(&schema, content) {
                            record.outcome = AuditOutcome::Failed("output_schema".into());
                            self.audit.record(record);
                            return Err(ToolsError::OutputSchema(format!(
                                "{} at {}",
                                e,
                                e.instance_path()
                            )));
                        }
                    }
                }
            }
        }

        let mut text = String::new();
        let mut other = Vec::new();
        for block in &result.content {
            match block {
                ContentBlock::Text(t) => {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(&t.text);
                }
                other_block => other.push(serde_json::to_value(other_block).unwrap_or(Value::Null)),
            }
        }
        if text.is_empty() {
            if let Some(s) = &result.structured_content {
                text = serde_json::to_string(s).unwrap_or_default();
            }
        }
        record.outcome = if is_error {
            AuditOutcome::ToolError
        } else {
            AuditOutcome::Ok
        };
        self.audit.record(record);
        Ok(McpCallOutcome {
            text,
            structured: result.structured_content.clone(),
            is_error,
            other_content: other,
            duration,
        })
    }
}

/// Split `mcp__{server}__{tool}` into `(server, tool)`.
pub fn split_qualified(name: &str) -> Option<(&str, &str)> {
    let rest = name.strip_prefix(TOOL_PREFIX)?;
    let idx = rest.find("__")?;
    let (server, tool) = (&rest[..idx], &rest[idx + 2..]);
    if server.is_empty() || tool.is_empty() {
        return None;
    }
    Some((server, tool))
}

/// Build the registry entry for a server tool.
pub fn tool_definition(cfg: &McpServerConfig, tool: &Tool) -> ToolDefinition {
    ToolDefinition {
        name: cfg.qualified_name(&tool.name),
        description: tool.description.as_deref().unwrap_or("").to_string(),
        parameters: Value::Object((*tool.input_schema).clone()),
        class: cfg.class_for(&tool.name),
        origin: ToolOrigin::Mcp {
            server: cfg.name.clone(),
            remote_name: tool.name.to_string(),
        },
        profile: cfg.profile_for(&tool.name),
        trusted_source: cfg.trusted,
        exfiltration_when_tainted: false,
        annotations: tool
            .annotations
            .as_ref()
            .and_then(|a| serde_json::to_value(a).ok()),
        output_schema: tool
            .output_schema
            .as_deref()
            .map(|s| Value::Object(s.clone())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOML: &str = r#"
[[servers]]
name = "fs"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/docs"]
env = ["PATH", "HOME"]
env_set = { NODE_OPTIONS = "--max-old-space-size=256" }
trusted = false
class = "state_changing"
tool_classes = { read_file = "read_only_local" }
tools = { read_file = "allow", write_file = "ask", delete_file = "deny" }
default_profile = "ask"
timeout_ms = 10000

[[servers]]
name = "remote"
url = "https://mcp.example.com/mcp"
bearer_token_env = "REMOTE_MCP_TOKEN"
sandbox = false
"#;

    #[test]
    fn parses_documented_config() {
        let cfg = McpConfig::parse(TOML).unwrap();
        assert_eq!(cfg.servers.len(), 2);
        let fs = &cfg.servers[0];
        assert_eq!(fs.command.as_deref(), Some("npx"));
        assert_eq!(fs.env, vec!["PATH", "HOME"]);
        assert_eq!(
            fs.env_set.get("NODE_OPTIONS").unwrap(),
            "--max-old-space-size=256"
        );
        assert_eq!(fs.profile_for("read_file"), ToolProfile::Allow);
        assert_eq!(fs.profile_for("write_file"), ToolProfile::Ask);
        assert_eq!(fs.profile_for("delete_file"), ToolProfile::Deny);
        assert_eq!(fs.profile_for("other"), ToolProfile::Ask);
        assert_eq!(fs.class_for("read_file"), ToolClass::ReadOnlyLocal);
        assert_eq!(fs.class_for("write_file"), ToolClass::StateChanging);
        assert_eq!(fs.timeout(), Duration::from_millis(10_000));
        assert!(fs.sandbox);
        assert_eq!(fs.qualified_name("read_file"), "mcp__fs__read_file");
        assert_eq!(
            fs.display_command(),
            "npx -y @modelcontextprotocol/server-filesystem /Users/me/docs"
        );
        let remote = &cfg.servers[1];
        assert_eq!(remote.url.as_deref(), Some("https://mcp.example.com/mcp"));
        assert_eq!(remote.bearer_token_env.as_deref(), Some("REMOTE_MCP_TOKEN"));
        assert!(remote.bearer_token.is_none());
        assert!(!remote.sandbox);
        assert_eq!(remote.timeout_ms, 30_000);
        assert_eq!(remote.class, ToolClass::StateChanging);
    }

    #[test]
    fn rejects_bad_configs() {
        assert!(
            McpConfig::parse("[[servers]]\nname = \"x\"\n").is_err(),
            "no command/url"
        );
        assert!(
            McpConfig::parse("[[servers]]\nname = \"x\"\ncommand = \"a\"\nurl = \"https://b\"\n")
                .is_err(),
            "both"
        );
        assert!(McpConfig::parse("[[servers]]\nname = \"bad name\"\ncommand = \"a\"\n").is_err());
        assert!(McpConfig::parse("[[servers]]\nname = \"a\"\ncommand = \"a\"\n[[servers]]\nname = \"a\"\ncommand = \"b\"\n").is_err(), "dup");
        assert!(McpConfig::parse(
            "[[servers]]\nname = \"a\"\ncommand = \"a\"\ntools = { x = \"maybe\" }\n"
        )
        .is_err());
        assert_eq!(McpConfig::parse("").unwrap().servers.len(), 0);
    }

    #[test]
    fn command_spec_allowlists_env() {
        let mut cfg = McpServerConfig::stdio("s", "prog", &["--flag"]);
        cfg.env = vec!["PATH".into(), "LLMARIO_DEFINITELY_UNSET_VAR".into()];
        cfg.env_set.insert("X".into(), "1".into());
        let spec = cfg.command_spec().unwrap();
        assert_eq!(spec.program, "prog");
        assert_eq!(spec.args, vec!["--flag"]);
        assert!(spec.env.iter().any(|(k, _)| k == "PATH"));
        assert!(!spec
            .env
            .iter()
            .any(|(k, _)| k == "LLMARIO_DEFINITELY_UNSET_VAR"));
        assert!(spec.env.iter().any(|(k, v)| k == "X" && v == "1"));
        assert!(
            !spec.env.iter().any(|(k, _)| k == "HOME"),
            "HOME not allowlisted"
        );
    }

    #[test]
    fn qualified_names() {
        assert_eq!(
            split_qualified("mcp__fs__read_file"),
            Some(("fs", "read_file"))
        );
        assert_eq!(split_qualified("mcp__fs__a__b"), Some(("fs", "a__b")));
        assert_eq!(split_qualified("web_fetch"), None);
        assert_eq!(split_qualified("mcp____x"), None);
        assert_eq!(split_qualified("mcp__fs__"), None);
    }

    #[test]
    fn tool_definition_ignores_annotations_for_class() {
        let cfg = McpServerConfig::stdio("fs", "x", &[]);
        let tool = Tool::new("rm", "remove", serde_json::Map::new())
            .with_annotations(rmcp::model::ToolAnnotations::new().read_only(true));
        let def = tool_definition(&cfg, &tool);
        assert_eq!(def.name, "mcp__fs__rm");
        assert_eq!(
            def.class,
            ToolClass::StateChanging,
            "readOnlyHint is not trusted"
        );
        assert_eq!(def.profile, ToolProfile::Ask);
        assert!(def.annotations.unwrap()["readOnlyHint"].as_bool().unwrap());
        assert_eq!(def.parameters, serde_json::json!({}));
    }
}
