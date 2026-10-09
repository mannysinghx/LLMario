//! Native engine `tools` crate: everything a model may call that is not the model itself.
//!
//! * [`ToolRegistry`] — tool definitions in OpenAI function format with an engine-owned
//!   [`ToolClass`] per tool; built-in tools register themselves, MCP tools register per server.
//! * [`PolicyGate`] / [`RuleOfTwoGate`] — the Rule-of-Two gate over a [`SessionState`]: open-world
//!   reads taint the session; once tainted, state-changing or exfiltration-capable calls block on
//!   human approval unless a permission profile says `allow`.
//! * [`WebFetcher`] — SSRF-guarded fetcher (scheme/port allowlist, blocked address ranges, DNS
//!   pinning, hop-by-hop redirect re-validation, text-only, byte cap, deadline, `robots.txt`).
//! * [`extract`] — Readability-style main-content extraction to Markdown with hidden text and
//!   invisible Unicode stripped; [`page`] — the gpt-oss `simple_browser` page model (80-column
//!   wrap, `L{i}:` lines, token windows, `find`, numbered links, per-session LRU cache).
//! * [`SearchProvider`] — `web_search` behind a trait: [`SearxngProvider`] and
//!   [`GenericJsonEndpointProvider`].
//! * [`McpHost`] — `rmcp`-based MCP client for stdio and Streamable HTTP servers with per-server
//!   permission profiles (`$LLMARIO_HOME/mcp.toml`), output-schema validation and auditing.
//! * [`AuditLog`] — content-free audit records (tool, argument hash, class, decision, duration,
//!   bytes in/out).
//! * [`sandbox`] — Seatbelt / bubblewrap launch plans for stdio MCP servers and the fetcher.
//! * [`wrap_untrusted`] — provenance-tagged result blocks with a random nonce.
//!
//! The crate is a library; wiring into the agent loop and the HTTP server is the server crate's job.

#![forbid(unsafe_code)]

pub mod audit;
pub mod builtin;
mod error;
pub mod extract;
pub mod fetch;
mod http;
pub mod mcp;
pub mod page;
pub mod policy;
pub mod provenance;
pub mod registry;
pub mod sandbox;
pub mod search;
pub mod ssrf;

pub use audit::{AuditDecision, AuditLog, AuditOutcome, AuditRecord, MemoryAuditLog};
pub use error::ToolsError;
pub use fetch::{FetchConfig, FetchedDocument, WebFetcher};
pub use mcp::{McpConfig, McpHost, McpServerConfig};
pub use page::{Page, PageCache, PageView, TokenCounter};
pub use policy::{PolicyDecision, PolicyGate, RuleOfTwoGate, SessionState};
pub use provenance::{wrap_untrusted, ToolResult, ToolSource};
pub use registry::{ToolClass, ToolDefinition, ToolOrigin, ToolProfile, ToolRegistry};
pub use sandbox::SandboxSpec;
pub use search::{GenericJsonEndpointProvider, SearchProvider, SearchResult, SearxngProvider};
pub use ssrf::{SsrfConfig, SsrfGuard};

/// Crate version, also the product version in the fetcher's `User-Agent`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// `User-Agent` sent by the fetcher and the search providers.
pub fn user_agent() -> String {
    format!("llmario-engine/{VERSION}")
}
