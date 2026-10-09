use std::time::Duration;

/// Errors from the tools crate. Messages never carry page content or tool results.
#[derive(Debug, thiserror::Error)]
pub enum ToolsError {
    /// The URL or a resolved address failed the SSRF guard.
    #[error("blocked by SSRF guard: {0}")]
    Ssrf(String),
    /// The URL does not parse or uses an unsupported form.
    #[error("invalid url: {0}")]
    Url(String),
    /// Transport-level fetch failure (connect, TLS, read).
    #[error("fetch failed: {0}")]
    Fetch(String),
    /// `robots.txt` disallows the engine's user agent for this URL.
    #[error("robots.txt disallows {0}")]
    RobotsDisallowed(String),
    /// The response is not a text content type.
    #[error("unsupported content type: {0}")]
    ContentType(String),
    /// The fetch deadline elapsed.
    #[error("deadline of {0:?} exceeded")]
    Deadline(Duration),
    /// More redirects than allowed.
    #[error("too many redirects (max {0})")]
    TooManyRedirects(usize),
    /// Main-content extraction failed.
    #[error("extraction failed: {0}")]
    Extract(String),
    /// A search provider failed or returned an unexpected payload.
    #[error("search provider error: {0}")]
    Search(String),
    /// MCP transport or protocol failure.
    #[error("mcp: {0}")]
    Mcp(String),
    /// A tool call exceeded its per-call timeout.
    #[error("tool call timed out after {0:?}")]
    Timeout(Duration),
    /// `structuredContent` did not validate against the tool's `outputSchema`.
    #[error("tool output failed schema validation: {0}")]
    OutputSchema(String),
    /// The policy gate (or a permission profile) refused the call.
    #[error("policy denied: {0}")]
    Denied(String),
    /// No tool with this name is registered.
    #[error("unknown tool: {0}")]
    UnknownTool(String),
    /// A tool with this name is already registered.
    #[error("duplicate tool name: {0}")]
    DuplicateTool(String),
    /// Invalid configuration (`mcp.toml`, provider config, ...).
    #[error("config error: {0}")]
    Config(String),
    /// Invalid tool arguments.
    #[error("invalid arguments: {0}")]
    Arguments(String),
    /// Sandbox launch plan could not be built on this platform.
    #[error("sandbox: {0}")]
    Sandbox(String),
}
