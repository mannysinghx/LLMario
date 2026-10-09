//! Content-free audit log: tool name, argument hash, class, decision, duration, bytes in/out.
//! Arguments and results are never recorded in clear.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::policy::PolicyDecision;
use crate::registry::ToolClass;

/// What the policy gate decided for the audited call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditDecision {
    /// Ran without a human.
    Allowed,
    /// Ran after a human approved.
    Approved,
    /// Blocked pending approval (and not executed by this record).
    NeedsApproval,
    /// Refused.
    Denied,
}

impl AuditDecision {
    /// Map a gate decision to the audit value.
    pub fn from_policy(d: &PolicyDecision) -> Self {
        if !d.allow {
            AuditDecision::Denied
        } else if d.needs_approval {
            AuditDecision::NeedsApproval
        } else if d.reason == "approved by human" || d.reason == "approved for this session" {
            AuditDecision::Approved
        } else {
            AuditDecision::Allowed
        }
    }
}

/// How the call ended.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "detail")]
pub enum AuditOutcome {
    /// Completed.
    Ok,
    /// The tool reported `isError` (fed back to the model).
    ToolError,
    /// The engine failed the call (timeout, schema, transport, policy); detail is the error
    /// *kind*, never its payload.
    Failed(String),
    /// Not executed (denied or awaiting approval).
    NotExecuted,
}

/// One audited call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditRecord {
    /// Timestamp.
    pub at: DateTime<Utc>,
    /// Exposed tool name.
    pub tool: String,
    /// MCP server, if any.
    pub server: Option<String>,
    /// Effective class.
    pub class: ToolClass,
    /// Decision.
    pub decision: AuditDecision,
    /// Gate reason text.
    pub reason: String,
    /// SHA-256 (hex) of the canonical JSON of the arguments.
    pub args_sha256: String,
    /// Wall time of the call.
    pub duration_ms: u64,
    /// Bytes received from the tool (serialized result size).
    pub bytes_in: u64,
    /// Bytes sent to the tool (serialized argument size).
    pub bytes_out: u64,
    /// Outcome.
    pub outcome: AuditOutcome,
}

/// Sink for audit records.
pub trait AuditLog: Send + Sync {
    /// Record one call. Implementations must not block for long.
    fn record(&self, record: AuditRecord);
}

/// Keeps records in memory (tests, desktop UI).
#[derive(Debug, Default)]
pub struct MemoryAuditLog {
    records: Mutex<Vec<AuditRecord>>,
}

impl MemoryAuditLog {
    /// Empty log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot of all records.
    pub fn records(&self) -> Vec<AuditRecord> {
        self.records.lock().map(|r| r.clone()).unwrap_or_default()
    }
}

impl AuditLog for MemoryAuditLog {
    fn record(&self, record: AuditRecord) {
        if let Ok(mut r) = self.records.lock() {
            r.push(record);
        }
    }
}

/// Appends one JSON object per line to a file.
#[derive(Debug)]
pub struct JsonlAuditLog {
    path: PathBuf,
    file: Mutex<std::fs::File>,
}

impl JsonlAuditLog {
    /// Open (create/append) the file.
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        Ok(Self {
            path,
            file: Mutex::new(file),
        })
    }

    /// The log file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl AuditLog for JsonlAuditLog {
    fn record(&self, record: AuditRecord) {
        let Ok(line) = serde_json::to_string(&record) else {
            return;
        };
        if let Ok(mut f) = self.file.lock() {
            let _ = writeln!(f, "{line}");
        }
    }
}

/// Emits records as `tracing` info events (target `llmario::tools::audit`).
#[derive(Debug, Default)]
pub struct TracingAuditLog;

impl AuditLog for TracingAuditLog {
    fn record(&self, r: AuditRecord) {
        tracing::info!(
            target: "llmario::tools::audit",
            tool = %r.tool,
            server = r.server.as_deref().unwrap_or("-"),
            class = r.class.as_str(),
            decision = ?r.decision,
            args_sha256 = %r.args_sha256,
            duration_ms = r.duration_ms,
            bytes_in = r.bytes_in,
            bytes_out = r.bytes_out,
            outcome = ?r.outcome,
            "tool call"
        );
    }
}

/// SHA-256 (hex) of the arguments serialized with sorted keys, so the hash is independent of
/// key order.
pub fn hash_args(args: &Value) -> String {
    let canonical = canonical_json(args);
    let digest = Sha256::digest(canonical.as_bytes());
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn canonical_json(v: &Value) -> String {
    match v {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let inner: Vec<String> = keys
                .into_iter()
                .map(|k| {
                    format!(
                        "{}:{}",
                        serde_json::to_string(k).unwrap_or_default(),
                        canonical_json(&map[k])
                    )
                })
                .collect();
            format!("{{{}}}", inner.join(","))
        }
        Value::Array(items) => {
            let inner: Vec<String> = items.iter().map(canonical_json).collect();
            format!("[{}]", inner.join(","))
        }
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn hash_ignores_key_order_and_differs_on_content() {
        let a = hash_args(&json!({"b": 1, "a": [1, {"y": 2, "x": 3}]}));
        let b = hash_args(&json!({"a": [1, {"x": 3, "y": 2}], "b": 1}));
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
        assert_ne!(a, hash_args(&json!({"a": [1, {"x": 3, "y": 2}], "b": 2})));
    }

    #[test]
    fn decision_mapping() {
        let base = PolicyDecision {
            allow: true,
            needs_approval: false,
            reason: "allowed by permission profile",
            effective_class: ToolClass::OpenWorldRead,
        };
        assert_eq!(AuditDecision::from_policy(&base), AuditDecision::Allowed);
        assert_eq!(
            AuditDecision::from_policy(&PolicyDecision {
                needs_approval: true,
                ..base.clone()
            }),
            AuditDecision::NeedsApproval
        );
        assert_eq!(
            AuditDecision::from_policy(&PolicyDecision {
                allow: false,
                ..base.clone()
            }),
            AuditDecision::Denied
        );
        assert_eq!(
            AuditDecision::from_policy(&PolicyDecision {
                reason: "approved by human",
                ..base
            }),
            AuditDecision::Approved
        );
    }

    #[test]
    fn jsonl_log_writes_one_line_per_record_without_arguments() {
        let dir = std::env::temp_dir().join(format!("llmario-audit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("audit.jsonl");
        let log = JsonlAuditLog::open(&path).unwrap();
        let rec = AuditRecord {
            at: Utc::now(),
            tool: "mcp__fs__write".into(),
            server: Some("fs".into()),
            class: ToolClass::StateChanging,
            decision: AuditDecision::Approved,
            reason: "approved by human".into(),
            args_sha256: hash_args(&json!({"path": "/secret"})),
            duration_ms: 12,
            bytes_in: 10,
            bytes_out: 20,
            outcome: AuditOutcome::Ok,
        };
        log.record(rec.clone());
        log.record(rec);
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.lines().count(), 2);
        assert!(!text.contains("/secret"));
        let parsed: AuditRecord = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        assert_eq!(parsed.tool, "mcp__fs__write");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
