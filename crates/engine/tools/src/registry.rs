//! Tool definitions (OpenAI function format) with engine-owned classification.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::ToolsError;

/// Engine-owned classification of what a tool can do. Never derived from server annotations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolClass {
    /// Reads only local, user-owned data (e.g. `retrieve` over the user's index).
    ReadOnlyLocal,
    /// Reads from the open world (search, fetch). Results taint the session.
    OpenWorldRead,
    /// Writes anywhere (files, databases, remote APIs).
    StateChanging,
    /// Can send data out of the machine (mail, chat, HTTP with context-derived payloads).
    ExfiltrationCapable,
}

impl ToolClass {
    /// True for the classes the Rule-of-Two gate treats as "can change state or communicate".
    pub fn has_side_effects(self) -> bool {
        matches!(
            self,
            ToolClass::StateChanging | ToolClass::ExfiltrationCapable
        )
    }

    /// Stable snake_case name (used in audit records and config files).
    pub fn as_str(self) -> &'static str {
        match self {
            ToolClass::ReadOnlyLocal => "read_only_local",
            ToolClass::OpenWorldRead => "open_world_read",
            ToolClass::StateChanging => "state_changing",
            ToolClass::ExfiltrationCapable => "exfiltration_capable",
        }
    }
}

/// Per-tool permission profile. Default `Ask` for MCP tools, `Allow` for built-in read-only tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolProfile {
    /// Execute without asking (still subject to `Deny`-free policy; see [`crate::RuleOfTwoGate`]).
    Allow,
    /// Ask the human before every call (unless approved for the session).
    #[default]
    Ask,
    /// Never execute.
    Deny,
}

impl ToolProfile {
    /// Stable lowercase name as written in `mcp.toml`.
    pub fn as_str(self) -> &'static str {
        match self {
            ToolProfile::Allow => "allow",
            ToolProfile::Ask => "ask",
            ToolProfile::Deny => "deny",
        }
    }
}

/// Where a tool comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolOrigin {
    /// Implemented in this crate (`web_search`, `web_fetch`, `retrieve`).
    Builtin,
    /// Served by an MCP server; `remote_name` is the server-side tool name.
    Mcp { server: String, remote_name: String },
}

impl ToolOrigin {
    /// The MCP server name, if any.
    pub fn server(&self) -> Option<&str> {
        match self {
            ToolOrigin::Builtin => None,
            ToolOrigin::Mcp { server, .. } => Some(server),
        }
    }
}

/// One registered tool.
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    /// Name exposed to the model (`^[A-Za-z0-9_.-]{1,128}$`). MCP tools are `mcp__{server}__{tool}`.
    pub name: String,
    /// Description shown to the model.
    pub description: String,
    /// JSON Schema for the arguments object.
    pub parameters: Value,
    /// Engine-owned classification.
    pub class: ToolClass,
    /// Built-in or MCP.
    pub origin: ToolOrigin,
    /// Permission profile.
    pub profile: ToolProfile,
    /// Results from a trusted source do not taint the session. Built-in open-world tools are
    /// never trusted; MCP servers are trusted only when `mcp.toml` says `trusted = true`.
    pub trusted_source: bool,
    /// When the session is tainted, treat this open-world tool as exfiltration-capable if its
    /// arguments carry a URL with a query string or userinfo (data can leave in the request).
    pub exfiltration_when_tainted: bool,
    /// Server-provided annotations (`readOnlyHint`, ...). Read for display only; never trusted.
    pub annotations: Option<Value>,
    /// Server-provided `outputSchema`, validated against `structuredContent` on every call.
    pub output_schema: Option<Value>,
}

impl ToolDefinition {
    /// A built-in tool. Read-only classes default to `Allow`, others to `Ask`.
    pub fn builtin(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        class: ToolClass,
    ) -> Self {
        let profile = if class.has_side_effects() {
            ToolProfile::Ask
        } else {
            ToolProfile::Allow
        };
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            class,
            origin: ToolOrigin::Builtin,
            profile,
            trusted_source: false,
            exfiltration_when_tainted: false,
            annotations: None,
            output_schema: None,
        }
    }

    /// Override the permission profile.
    pub fn with_profile(mut self, profile: ToolProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Mark the tool as exfiltration-capable once the session is tainted (see field docs).
    pub fn with_exfiltration_when_tainted(mut self, flag: bool) -> Self {
        self.exfiltration_when_tainted = flag;
        self
    }

    /// The OpenAI function-tool JSON (`{"type":"function","function":{...}}`).
    pub fn openai_function(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        })
    }
}

/// Validate a tool name against the MCP/OpenAI-compatible charset.
pub fn is_valid_tool_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'.' || b == b'-')
}

/// All tools the model may call, keyed by exposed name. Iteration order is deterministic
/// (sorted by name) so prompt caches hit.
#[derive(Debug, Default, Clone)]
pub struct ToolRegistry {
    tools: BTreeMap<String, ToolDefinition>,
}

impl ToolRegistry {
    /// Empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool; fails on a duplicate or invalid name.
    pub fn register(&mut self, def: ToolDefinition) -> Result<(), ToolsError> {
        if !is_valid_tool_name(&def.name) {
            return Err(ToolsError::Config(format!(
                "invalid tool name {:?} (allowed: [A-Za-z0-9_.-]{{1,128}})",
                def.name
            )));
        }
        if self.tools.contains_key(&def.name) {
            return Err(ToolsError::DuplicateTool(def.name));
        }
        self.tools.insert(def.name.clone(), def);
        Ok(())
    }

    /// Register or replace.
    pub fn upsert(&mut self, def: ToolDefinition) {
        self.tools.insert(def.name.clone(), def);
    }

    /// Look up by exposed name.
    pub fn get(&self, name: &str) -> Option<&ToolDefinition> {
        self.tools.get(name)
    }

    /// Mutable lookup (e.g. to change a profile after a session approval).
    pub fn get_mut(&mut self, name: &str) -> Option<&mut ToolDefinition> {
        self.tools.get_mut(name)
    }

    /// Remove one tool.
    pub fn remove(&mut self, name: &str) -> Option<ToolDefinition> {
        self.tools.remove(name)
    }

    /// Remove every tool served by `server`; returns how many were removed.
    pub fn remove_server(&mut self, server: &str) -> usize {
        let before = self.tools.len();
        self.tools
            .retain(|_, def| def.origin.server() != Some(server));
        before - self.tools.len()
    }

    /// All definitions, sorted by name.
    pub fn iter(&self) -> impl Iterator<Item = &ToolDefinition> {
        self.tools.values()
    }

    /// Exposed names, sorted.
    pub fn names(&self) -> Vec<&str> {
        self.tools.keys().map(String::as_str).collect()
    }

    /// Number of tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// True when nothing is registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// The `tools` array for an OpenAI-compatible request.
    pub fn openai_tools(&self) -> Vec<Value> {
        self.tools
            .values()
            .map(ToolDefinition::openai_function)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn def(name: &str, class: ToolClass) -> ToolDefinition {
        ToolDefinition::builtin(name, "d", json!({"type": "object"}), class)
    }

    #[test]
    fn builtin_profile_defaults_follow_class() {
        assert_eq!(
            def("a", ToolClass::ReadOnlyLocal).profile,
            ToolProfile::Allow
        );
        assert_eq!(
            def("b", ToolClass::OpenWorldRead).profile,
            ToolProfile::Allow
        );
        assert_eq!(def("c", ToolClass::StateChanging).profile, ToolProfile::Ask);
        assert_eq!(
            def("d", ToolClass::ExfiltrationCapable).profile,
            ToolProfile::Ask
        );
    }

    #[test]
    fn register_rejects_duplicates_and_bad_names() {
        let mut r = ToolRegistry::new();
        r.register(def("web_fetch", ToolClass::OpenWorldRead))
            .unwrap();
        assert!(matches!(
            r.register(def("web_fetch", ToolClass::OpenWorldRead)),
            Err(ToolsError::DuplicateTool(_))
        ));
        assert!(matches!(
            r.register(def("bad name", ToolClass::OpenWorldRead)),
            Err(ToolsError::Config(_))
        ));
        assert!(matches!(
            r.register(def("", ToolClass::OpenWorldRead)),
            Err(ToolsError::Config(_))
        ));
    }

    #[test]
    fn openai_format_and_ordering() {
        let mut r = ToolRegistry::new();
        r.register(def("zeta", ToolClass::OpenWorldRead)).unwrap();
        r.register(def("alpha", ToolClass::OpenWorldRead)).unwrap();
        let tools = r.openai_tools();
        assert_eq!(tools[0]["function"]["name"], "alpha");
        assert_eq!(tools[1]["function"]["name"], "zeta");
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn remove_server_only_touches_that_server() {
        let mut r = ToolRegistry::new();
        let mut a = def("mcp__a__x", ToolClass::StateChanging);
        a.origin = ToolOrigin::Mcp {
            server: "a".into(),
            remote_name: "x".into(),
        };
        let mut b = def("mcp__b__x", ToolClass::StateChanging);
        b.origin = ToolOrigin::Mcp {
            server: "b".into(),
            remote_name: "x".into(),
        };
        r.register(a).unwrap();
        r.register(b).unwrap();
        r.register(def("web_fetch", ToolClass::OpenWorldRead))
            .unwrap();
        assert_eq!(r.remove_server("a"), 1);
        assert_eq!(r.names(), vec!["mcp__b__x", "web_fetch"]);
    }
}
