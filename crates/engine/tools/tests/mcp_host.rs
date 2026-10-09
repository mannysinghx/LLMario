//! `McpHost` against a tiny in-process `rmcp` server over a tokio duplex pipe: one `allow`
//! tool, one `ask` tool, output-schema validation, `isError`, timeouts, audit contents.

use std::sync::Arc;
use std::time::Duration;

use llmario_engine_tools::audit::hash_args;
use llmario_engine_tools::audit::{AuditDecision, AuditOutcome};
use llmario_engine_tools::{
    McpHost, McpServerConfig, MemoryAuditLog, PolicyGate, RuleOfTwoGate, SessionState, ToolClass,
    ToolProfile, ToolRegistry, ToolsError,
};
use rmcp::handler::server::ServerHandler;
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, ErrorData,
    InitializeResult, ListToolsResult, PaginatedRequestParams, ServerCapabilities, Tool,
    ToolAnnotations,
};
use rmcp::service::RequestContext;
use rmcp::{serve_server, RoleServer};
use serde_json::{json, Map, Value};

#[derive(Clone)]
struct TinyServer;

fn obj(v: Value) -> Map<String, Value> {
    v.as_object().cloned().unwrap()
}

impl ServerHandler for TinyServer {
    fn get_info(&self) -> InitializeResult {
        InitializeResult::new(ServerCapabilities::builder().enable_tools().build())
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        let echo = Tool::new(
            "echo",
            "Echo text",
            obj(json!({"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]})),
        )
        .with_raw_output_schema(Arc::new(obj(json!({"type": "object", "properties": {"echo": {"type": "string"}}, "required": ["echo"]}))))
        // A lying annotation: the host must not trust it.
        .with_annotations(ToolAnnotations::new().read_only(true).destructive(false));
        let write = Tool::new("write", "Write somewhere", obj(json!({"type": "object"})));
        let bad = Tool::new(
            "bad_schema",
            "Returns invalid structured content",
            obj(json!({"type": "object"})),
        )
        .with_raw_output_schema(Arc::new(obj(
            json!({"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}),
        )));
        let missing = Tool::new(
            "no_structured",
            "Declares a schema but returns text only",
            obj(json!({"type": "object"})),
        )
        .with_raw_output_schema(Arc::new(obj(json!({"type": "object"}))));
        let fail = Tool::new("fail", "Always errors", obj(json!({"type": "object"})));
        let slow = Tool::new("slow", "Sleeps", obj(json!({"type": "object"})));
        Ok(ListToolsResult::with_all_items(vec![
            echo, write, bad, missing, fail, slow,
        ]))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let args = request.arguments.unwrap_or_default();
        let result = match request.name.as_ref() {
            "echo" => {
                let text = args
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let mut r =
                    CallToolResult::success(vec![ContentBlock::text(format!("echo: {text}"))]);
                r.structured_content = Some(json!({"echo": text}));
                r
            }
            "write" => CallToolResult::success(vec![ContentBlock::text("wrote")]),
            "bad_schema" => CallToolResult::structured(json!({"n": "not an integer"})),
            "no_structured" => CallToolResult::success(vec![ContentBlock::text("text only")]),
            "fail" => CallToolResult::error(vec![ContentBlock::text("boom")]),
            "slow" => {
                tokio::time::sleep(Duration::from_secs(5)).await;
                CallToolResult::success(vec![ContentBlock::text("late")])
            }
            other => {
                return Err(ErrorData::invalid_params(
                    format!("unknown tool {other}"),
                    None,
                ))
            }
        };
        Ok(CallToolResponse::Complete(result))
    }
}

struct Harness {
    host: McpHost,
    audit: Arc<MemoryAuditLog>,
    registry: ToolRegistry,
}

async fn connect(cfg: McpServerConfig) -> Harness {
    let (client_side, server_side) = tokio::io::duplex(1 << 16);
    let (sr, sw) = tokio::io::split(server_side);
    tokio::spawn(async move {
        let running = serve_server(TinyServer, (sr, sw))
            .await
            .expect("server start");
        let _ = running.waiting().await;
    });
    let audit = Arc::new(MemoryAuditLog::new());
    let host = McpHost::new(audit.clone());
    let mut registry = ToolRegistry::new();
    let (cr, cw) = tokio::io::split(client_side);
    let report = host
        .connect_with_transport(cfg, (cr, cw), &mut registry)
        .await
        .expect("connect");
    assert_eq!(report.server, "tiny");
    assert_eq!(
        report.protocol_version, "2026-07-28",
        "rmcp negotiates the 2026-07-28 revision"
    );
    Harness {
        host,
        audit,
        registry,
    }
}

fn config() -> McpServerConfig {
    let mut cfg = McpServerConfig::stdio("tiny", "unused", &[]);
    cfg.tools.insert("echo".into(), ToolProfile::Allow);
    cfg.tool_classes
        .insert("echo".into(), ToolClass::ReadOnlyLocal);
    cfg.timeout_ms = 500;
    cfg
}

#[tokio::test]
async fn registers_tools_with_engine_owned_classes_and_profiles() {
    let h = connect(config()).await;
    assert_eq!(
        h.registry.names(),
        vec![
            "mcp__tiny__bad_schema",
            "mcp__tiny__echo",
            "mcp__tiny__fail",
            "mcp__tiny__no_structured",
            "mcp__tiny__slow",
            "mcp__tiny__write"
        ]
    );
    let echo = h.registry.get("mcp__tiny__echo").unwrap();
    assert_eq!(echo.profile, ToolProfile::Allow);
    assert_eq!(echo.class, ToolClass::ReadOnlyLocal);
    assert_eq!(echo.description, "Echo text");
    assert_eq!(echo.parameters["required"][0], "text");
    assert!(echo.output_schema.is_some());
    assert_eq!(echo.annotations.as_ref().unwrap()["readOnlyHint"], true);
    let write = h.registry.get("mcp__tiny__write").unwrap();
    assert_eq!(
        write.profile,
        ToolProfile::Ask,
        "default profile for MCP tools is ask"
    );
    assert_eq!(
        write.class,
        ToolClass::StateChanging,
        "default class is state_changing regardless of annotations"
    );
    assert!(!write.trusted_source);
    assert_eq!(h.host.servers().await, vec!["tiny"]);
    let f = h.registry.openai_tools();
    assert_eq!(f[1]["function"]["name"], "mcp__tiny__echo");
}

#[tokio::test]
async fn allow_tool_runs_ask_tool_waits_and_audit_is_content_free() {
    let mut h = connect(config()).await;
    let gate = RuleOfTwoGate::new();
    let mut session = SessionState::new();

    let echo = h.registry.get("mcp__tiny__echo").unwrap().clone();
    let args = json!({"text": "hello world"});
    let d = gate.decide(&session, &echo, &args);
    assert!(d.is_immediate());
    let out = h
        .host
        .call_tool("mcp__tiny__echo", args.clone(), &d)
        .await
        .unwrap();
    assert_eq!(out.text, "echo: hello world");
    assert_eq!(out.structured, Some(json!({"echo": "hello world"})));
    assert!(!out.is_error);
    session.record_result(&echo);
    assert!(
        session.tainted,
        "untrusted server output taints the session"
    );

    let write = h.registry.get("mcp__tiny__write").unwrap().clone();
    let d = gate.decide(&session, &write, &json!({}));
    assert!(d.allow && d.needs_approval);
    assert!(d.reason.starts_with("rule of two"));
    let r = h.host.call_tool("mcp__tiny__write", json!({}), &d).await;
    assert!(
        matches!(r, Err(ToolsError::Denied(ref m)) if m.contains("approval")),
        "{r:?}"
    );

    let approved = d.clone().approved();
    let out = h
        .host
        .call_tool("mcp__tiny__write", json!({}), &approved)
        .await
        .unwrap();
    assert_eq!(out.text, "wrote");

    let mut denied = write.clone();
    denied.profile = ToolProfile::Deny;
    let d = gate.decide(&session, &denied, &json!({}));
    assert!(d.is_denied());
    assert!(matches!(
        h.host.call_tool("mcp__tiny__write", json!({}), &d).await,
        Err(ToolsError::Denied(_))
    ));

    let records = h.audit.records();
    assert_eq!(records.len(), 4);
    assert_eq!(records[0].tool, "mcp__tiny__echo");
    assert_eq!(records[0].server.as_deref(), Some("tiny"));
    assert_eq!(records[0].class, ToolClass::ReadOnlyLocal);
    assert_eq!(records[0].decision, AuditDecision::Allowed);
    assert_eq!(records[0].args_sha256, hash_args(&args));
    assert_eq!(records[0].outcome, AuditOutcome::Ok);
    assert!(records[0].bytes_out > 0 && records[0].bytes_in > 0);
    assert_eq!(records[1].decision, AuditDecision::NeedsApproval);
    assert_eq!(records[1].outcome, AuditOutcome::NotExecuted);
    assert_eq!(records[2].decision, AuditDecision::Approved);
    assert_eq!(records[2].outcome, AuditOutcome::Ok);
    assert_eq!(records[3].decision, AuditDecision::Denied);
    let serialized = serde_json::to_string(&records).unwrap();
    assert!(
        !serialized.contains("hello world"),
        "arguments never appear in the audit log"
    );
    assert!(
        !serialized.contains("echo: "),
        "results never appear in the audit log"
    );

    assert!(h.host.disconnect("tiny", &mut h.registry).await);
    assert!(h.registry.is_empty());
    assert!(!h.host.disconnect("tiny", &mut h.registry).await);
    assert!(matches!(
        h.host
            .call_tool("mcp__tiny__echo", json!({}), &approved)
            .await,
        Err(ToolsError::UnknownTool(_))
    ));
}

#[tokio::test]
async fn output_schema_is_error_and_timeout() {
    let h = connect(config()).await;
    let allow = llmario_engine_tools::PolicyDecision {
        allow: true,
        needs_approval: false,
        reason: "approved by human",
        effective_class: ToolClass::StateChanging,
    };
    let r = h
        .host
        .call_tool("mcp__tiny__bad_schema", json!({}), &allow)
        .await;
    assert!(matches!(r, Err(ToolsError::OutputSchema(_))), "{r:?}");
    let r = h
        .host
        .call_tool("mcp__tiny__no_structured", json!({}), &allow)
        .await;
    assert!(
        matches!(r, Err(ToolsError::OutputSchema(ref m)) if m.contains("no structuredContent")),
        "{r:?}"
    );

    let out = h
        .host
        .call_tool("mcp__tiny__fail", json!({}), &allow)
        .await
        .unwrap();
    assert!(out.is_error);
    assert_eq!(out.text, "boom");

    let r = h.host.call_tool("mcp__tiny__slow", json!({}), &allow).await;
    assert!(
        matches!(r, Err(ToolsError::Timeout(d)) if d == Duration::from_millis(500)),
        "{r:?}"
    );

    let r = h
        .host
        .call_tool("mcp__tiny__echo", json!(["not", "an", "object"]), &allow)
        .await;
    assert!(matches!(r, Err(ToolsError::Arguments(_))), "{r:?}");
    assert!(matches!(
        h.host.call_tool("mcp__tiny__nope", json!({}), &allow).await,
        Err(ToolsError::UnknownTool(_))
    ));
    assert!(matches!(
        h.host.call_tool("web_fetch", json!({}), &allow).await,
        Err(ToolsError::UnknownTool(_))
    ));

    let records = h.audit.records();
    let outcomes: Vec<&AuditOutcome> = records.iter().map(|r| &r.outcome).collect();
    assert_eq!(
        outcomes,
        vec![
            &AuditOutcome::Failed("output_schema".into()),
            &AuditOutcome::Failed("output_schema".into()),
            &AuditOutcome::ToolError,
            &AuditOutcome::Failed("timeout".into()),
            &AuditOutcome::Failed("arguments".into()),
        ]
    );
    assert!(records[3].duration_ms >= 400, "{}", records[3].duration_ms);
}

#[tokio::test]
async fn duplicate_server_names_are_refused() {
    let h = connect(config()).await;
    let (client_side, _server_side) = tokio::io::duplex(1024);
    let (cr, cw) = tokio::io::split(client_side);
    let mut reg = ToolRegistry::new();
    let r = h
        .host
        .connect_with_transport(config(), (cr, cw), &mut reg)
        .await;
    assert!(
        matches!(r, Err(ToolsError::Config(ref m)) if m.contains("already connected")),
        "{r:?}"
    );
}
