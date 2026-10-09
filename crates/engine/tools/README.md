# llmario-engine-tools

Everything a model may call that is not the model itself: the tool registry, the Rule-of-Two
policy gate, the SSRF-guarded `web_fetch`, `web_search` providers, the MCP host, content-free
auditing and sandbox launch plans (Architecture §10.4–§10.7). It is a library; the agent loop in
the server crate wires it to generation.

| Type | Role |
|---|---|
| `ToolRegistry`, `ToolDefinition`, `ToolClass`, `ToolProfile` | Tools in OpenAI function format with an engine-owned class (`read_only_local`, `open_world_read`, `state_changing`, `exfiltration_capable`) and a permission profile (`allow`, `ask`, `deny`). MCP annotations are never used to classify. |
| `RuleOfTwoGate`, `SessionState`, `PolicyDecision` | Open-world reads taint the session; once tainted, state-changing or exfiltration-capable calls need human approval unless the tool's profile is `allow`. |
| `WebFetcher`, `FetchConfig`, `SsrfGuard`, `SsrfConfig` | `http`/`https` on ports 80/443 by default; every resolved address checked against blocked ranges (RFC 1918, loopback, link-local and cloud metadata, CGNAT, `::1`, `fc00::/7`, `fe80::/10`, IPv4-mapped forms); connections pinned to the validated address; redirects followed by hand and re-validated (at most 5); text media only; 600 KB cap; 20 s deadline; `robots.txt` honoured by default. |
| `extract`, `Page`, `PageCache`, `PageView` | Readability-style main content to Markdown with hidden elements and invisible Unicode removed; the gpt-oss `simple_browser` page model (80-column wrap, `L{i}:` lines, token windows, `find`, links as `【id†text†domain】`, per-session LRU cache). |
| `SearchProvider`, `SearxngProvider`, `GenericJsonEndpointProvider` | `web_search` over a self-hosted SearXNG instance (JSON format must be enabled on the instance) or any JSON endpoint described by a URL template and JSON paths. No paid or hosted search APIs. |
| `McpHost`, `McpConfig`, `McpServerConfig` | MCP client on `rmcp` 3.5.1, protocol revision **2026-07-28** with fallback to 2025-11-25; stdio servers (environment allowlist, optional sandbox) and Streamable HTTP servers (bearer token from an environment variable); tools exposed as `mcp__{server}__{tool}`; per-call timeouts; `structuredContent` validated against `outputSchema`; `isError` surfaced. |
| `AuditLog`, `AuditRecord` | Tool name, SHA-256 of the arguments, class, decision, duration, bytes in and out. Never arguments or results in clear. |
| `wrap_untrusted`, `ToolResult` | Results wrapped in `<tool_result … nonce=…>` blocks with a random nonce, so page content cannot close the block. |
| `SandboxSpec` | Seatbelt profile (macOS) or bubblewrap command line (Linux) for stdio MCP servers and the fetcher: read-only file system plus a scratch directory, network only through a configured proxy socket. Windows returns `Unsupported`. |

## `mcp.toml`

```toml
[[servers]]
name = "files"                 # [A-Za-z0-9_-]+; tools appear as mcp__files__<tool>
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/Documents"]
env = ["PATH"]                 # parent variables passed through; nothing else is inherited
sandbox = true                 # default true
class = "state_changing"       # default class for this server's tools
tool_classes = { read_file = "read_only_local", list_directory = "read_only_local" }
tools = { read_file = "allow", write_file = "ask", move_file = "deny" }
default_profile = "ask"
timeout_ms = 30000

[[servers]]
name = "issues"
url = "https://mcp.example.com/mcp"
bearer_token_env = "ISSUES_MCP_TOKEN"   # the token itself never goes in this file
trusted = false                          # results from untrusted servers taint the session
```

## Tests

`cargo test -p llmario-engine-tools` runs everything without internet access: 50 unit tests (SSRF
corpus with decimal, octal and IPv4-mapped encodings, extraction fixtures, page windows and
`find`, policy matrix, sandbox rendering, provenance), 6 web tests against a local mini HTTP server
(pinned DNS, redirect chains into private ranges, robots, byte cap, deadline, the `web_fetch` tool's
link following and cache, search providers) and 4 MCP tests against an in-process `rmcp` server
(`allow` and `ask` tools, audit contents, output-schema errors, timeouts, duplicate names).

## Not yet done

- The OS sandbox is generated and tested as a launch plan; enforcing it around the in-process
  fetcher and running the egress proxy land with the server's agent loop.
- The MCP host reads `$LLMARIO_HOME/mcp.toml`; the beta edition's home variable is
  `LLMARIO_BETA_HOME`, so the server passes the path explicitly when it wires the host in.
- Headless-browser fetching for script-rendered pages is deferred (Architecture §10.5).
