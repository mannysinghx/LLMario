//! The Rule-of-Two policy gate.
//!
//! A session may satisfy at most two of {processes untrusted input, has access to sensitive
//! data or systems, can change state or communicate externally} without a human. The engine
//! tracks the first two in [`SessionState`]; the third is the tool's [`ToolClass`]. Once the
//! session is tainted, every state-changing or exfiltration-capable call needs approval unless
//! the tool's permission profile says `allow` (or the gate runs in strict mode).

use std::collections::HashSet;

use serde_json::Value;

use crate::registry::{ToolClass, ToolDefinition, ToolOrigin, ToolProfile};

/// Per-session security state.
#[derive(Debug, Clone, Default)]
pub struct SessionState {
    /// Untrusted content (web page, search result, output of an untrusted MCP server) has
    /// entered the context.
    pub tainted: bool,
    /// The session has access to sensitive data or systems (user-marked, e.g. a filesystem or
    /// database MCP server is connected).
    pub has_sensitive_access: bool,
    /// Tools the human approved "for this session"; calls to them no longer ask.
    pub approvals: HashSet<String>,
}

impl SessionState {
    /// Fresh, untainted session.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark untrusted content as present.
    pub fn mark_tainted(&mut self) {
        self.tainted = true;
    }

    /// Approve a tool for the rest of the session.
    pub fn approve_for_session(&mut self, tool: &str) {
        self.approvals.insert(tool.to_string());
    }

    /// Whether a tool has a session-wide approval.
    pub fn is_approved(&self, tool: &str) -> bool {
        self.approvals.contains(tool)
    }

    /// Update the state after a tool's result entered the context: open-world reads and outputs
    /// of untrusted MCP servers taint the session.
    pub fn record_result(&mut self, tool: &ToolDefinition) {
        let untrusted_mcp = matches!(tool.origin, ToolOrigin::Mcp { .. }) && !tool.trusted_source;
        if tool.class == ToolClass::OpenWorldRead || untrusted_mcp {
            self.tainted = true;
        }
    }
}

/// Outcome of the gate for one call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyDecision {
    /// The call may run (possibly after approval).
    pub allow: bool,
    /// A human must approve before the call runs.
    pub needs_approval: bool,
    /// Short reason, stable text for the audit log.
    pub reason: &'static str,
    /// The class the gate evaluated (may be stricter than the registered class, see
    /// [`ToolDefinition::exfiltration_when_tainted`]).
    pub effective_class: ToolClass,
}

impl PolicyDecision {
    /// True when the call must not run at all.
    pub fn is_denied(&self) -> bool {
        !self.allow
    }

    /// True when the call can run right now without a human.
    pub fn is_immediate(&self) -> bool {
        self.allow && !self.needs_approval
    }

    /// The decision after the human approved an `needs_approval` call.
    pub fn approved(mut self) -> Self {
        self.needs_approval = false;
        self.reason = "approved by human";
        self
    }
}

/// Decides whether a tool call may run.
pub trait PolicyGate: Send + Sync {
    /// Evaluate one call. Does not mutate the session: callers apply
    /// [`SessionState::record_result`] after execution.
    fn decide(&self, session: &SessionState, tool: &ToolDefinition, args: &Value)
        -> PolicyDecision;
}

/// The default gate.
#[derive(Debug, Clone, Default)]
pub struct RuleOfTwoGate {
    /// When true, an `allow` profile does not bypass the taint rule: tainted sessions always
    /// ask before state-changing or exfiltration-capable calls.
    pub strict: bool,
}

impl RuleOfTwoGate {
    /// Default (profile `allow` bypasses the taint rule, as the architecture specifies).
    pub fn new() -> Self {
        Self::default()
    }

    /// Strict variant (see [`RuleOfTwoGate::strict`]).
    pub fn strict() -> Self {
        Self { strict: true }
    }

    /// The class the gate evaluates for this call.
    pub fn effective_class(
        session: &SessionState,
        tool: &ToolDefinition,
        args: &Value,
    ) -> ToolClass {
        if tool.class == ToolClass::OpenWorldRead
            && tool.exfiltration_when_tainted
            && session.tainted
            && args_carry_payload(args)
        {
            ToolClass::ExfiltrationCapable
        } else {
            tool.class
        }
    }
}

/// True when any string argument is a URL carrying a query string, fragment or userinfo —
/// a channel through which context-derived data can leave the machine.
pub fn args_carry_payload(args: &Value) -> bool {
    fn walk(v: &Value) -> bool {
        match v {
            Value::String(s) => looks_like_url_with_payload(s),
            Value::Array(a) => a.iter().any(walk),
            Value::Object(o) => o.values().any(walk),
            _ => false,
        }
    }
    walk(args)
}

fn looks_like_url_with_payload(s: &str) -> bool {
    let Ok(url) = url::Url::parse(s.trim()) else {
        return false;
    };
    if !matches!(url.scheme(), "http" | "https") {
        return false;
    }
    url.query().is_some_and(|q| !q.is_empty())
        || url.fragment().is_some_and(|f| !f.is_empty())
        || !url.username().is_empty()
        || url.password().is_some()
}

impl PolicyGate for RuleOfTwoGate {
    fn decide(
        &self,
        session: &SessionState,
        tool: &ToolDefinition,
        args: &Value,
    ) -> PolicyDecision {
        let effective_class = Self::effective_class(session, tool, args);
        let decision = |allow, needs_approval, reason| PolicyDecision {
            allow,
            needs_approval,
            reason,
            effective_class,
        };

        if tool.profile == ToolProfile::Deny {
            return decision(false, false, "denied by permission profile");
        }
        if session.is_approved(&tool.name) {
            return decision(true, false, "approved for this session");
        }
        if session.tainted && effective_class.has_side_effects() {
            if tool.profile == ToolProfile::Allow && !self.strict {
                return decision(true, false, "allowed by permission profile despite taint");
            }
            return decision(
                true,
                true,
                "rule of two: session holds untrusted content and the tool can change state or communicate externally",
            );
        }
        match tool.profile {
            ToolProfile::Allow => decision(true, false, "allowed by permission profile"),
            ToolProfile::Ask => decision(true, true, "permission profile asks for approval"),
            ToolProfile::Deny => unreachable!("handled above"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool(name: &str, class: ToolClass, profile: ToolProfile) -> ToolDefinition {
        ToolDefinition::builtin(name, "d", json!({"type": "object"}), class).with_profile(profile)
    }

    fn mcp_tool(
        name: &str,
        class: ToolClass,
        profile: ToolProfile,
        trusted: bool,
    ) -> ToolDefinition {
        let mut t = tool(name, class, profile);
        t.origin = ToolOrigin::Mcp {
            server: "s".into(),
            remote_name: name.into(),
        };
        t.trusted_source = trusted;
        t
    }

    const CLASSES: [ToolClass; 4] = [
        ToolClass::ReadOnlyLocal,
        ToolClass::OpenWorldRead,
        ToolClass::StateChanging,
        ToolClass::ExfiltrationCapable,
    ];
    const PROFILES: [ToolProfile; 3] = [ToolProfile::Allow, ToolProfile::Ask, ToolProfile::Deny];

    #[test]
    fn matrix_untainted() {
        let gate = RuleOfTwoGate::new();
        let session = SessionState::new();
        for class in CLASSES {
            for profile in PROFILES {
                let d = gate.decide(&session, &tool("t", class, profile), &json!({}));
                match profile {
                    ToolProfile::Deny => assert!(d.is_denied(), "{class:?}/{profile:?}"),
                    ToolProfile::Allow => assert!(d.is_immediate(), "{class:?}/{profile:?}"),
                    ToolProfile::Ask => {
                        assert!(d.allow && d.needs_approval, "{class:?}/{profile:?}")
                    }
                }
                assert_eq!(d.effective_class, class);
            }
        }
    }

    #[test]
    fn matrix_tainted() {
        let gate = RuleOfTwoGate::new();
        let mut session = SessionState::new();
        session.mark_tainted();
        for class in CLASSES {
            for profile in PROFILES {
                let d = gate.decide(&session, &tool("t", class, profile), &json!({}));
                match (profile, class.has_side_effects()) {
                    (ToolProfile::Deny, _) => assert!(d.is_denied()),
                    (ToolProfile::Allow, _) => assert!(d.is_immediate(), "{class:?}"),
                    (ToolProfile::Ask, true) => {
                        assert!(d.allow && d.needs_approval);
                        assert!(d.reason.starts_with("rule of two"), "{}", d.reason);
                    }
                    (ToolProfile::Ask, false) => {
                        assert!(d.allow && d.needs_approval);
                        assert!(d.reason.starts_with("permission profile"), "{}", d.reason);
                    }
                }
            }
        }
    }

    #[test]
    fn strict_mode_ignores_allow_once_tainted() {
        let gate = RuleOfTwoGate::strict();
        let mut session = SessionState::new();
        session.mark_tainted();
        let d = gate.decide(
            &session,
            &tool("w", ToolClass::StateChanging, ToolProfile::Allow),
            &json!({}),
        );
        assert!(d.allow && d.needs_approval);
        let d = gate.decide(
            &session,
            &tool("r", ToolClass::ReadOnlyLocal, ToolProfile::Allow),
            &json!({}),
        );
        assert!(d.is_immediate());
    }

    #[test]
    fn session_approval_short_circuits_ask_but_not_deny() {
        let gate = RuleOfTwoGate::new();
        let mut session = SessionState::new();
        session.mark_tainted();
        session.approve_for_session("w");
        let d = gate.decide(
            &session,
            &tool("w", ToolClass::StateChanging, ToolProfile::Ask),
            &json!({}),
        );
        assert!(d.is_immediate());
        assert_eq!(d.reason, "approved for this session");
        let d = gate.decide(
            &session,
            &tool("w", ToolClass::StateChanging, ToolProfile::Deny),
            &json!({}),
        );
        assert!(d.is_denied());
    }

    #[test]
    fn fetch_with_query_becomes_exfiltration_when_tainted() {
        let gate = RuleOfTwoGate::new();
        let fetch = tool("web_fetch", ToolClass::OpenWorldRead, ToolProfile::Allow)
            .with_exfiltration_when_tainted(true);
        let mut session = SessionState::new();
        let plain = json!({"url": "https://example.com/page"});
        let query = json!({"url": "https://example.com/p?secret=abc"});
        let userinfo = json!({"url": "https://user:pw@example.com/"});
        // Untainted: everything is an open-world read.
        assert_eq!(
            gate.decide(&session, &fetch, &query).effective_class,
            ToolClass::OpenWorldRead
        );
        session.mark_tainted();
        assert_eq!(
            gate.decide(&session, &fetch, &plain).effective_class,
            ToolClass::OpenWorldRead
        );
        assert!(gate.decide(&session, &fetch, &plain).is_immediate());
        let d = gate.decide(&session, &fetch, &query);
        assert_eq!(d.effective_class, ToolClass::ExfiltrationCapable);
        // Profile is Allow, so the non-strict gate still lets it through; strict asks.
        assert!(d.is_immediate());
        let d = RuleOfTwoGate::strict().decide(&session, &fetch, &query);
        assert!(d.needs_approval);
        let d = RuleOfTwoGate::strict().decide(&session, &fetch, &userinfo);
        assert!(d.needs_approval);
        // A tool without the flag is never upgraded.
        let search = tool("web_search", ToolClass::OpenWorldRead, ToolProfile::Allow);
        assert_eq!(
            gate.decide(&session, &search, &query).effective_class,
            ToolClass::OpenWorldRead
        );
    }

    #[test]
    fn record_result_taints_on_open_world_and_untrusted_mcp() {
        let mut s = SessionState::new();
        s.record_result(&tool(
            "retrieve",
            ToolClass::ReadOnlyLocal,
            ToolProfile::Allow,
        ));
        assert!(!s.tainted);
        s.record_result(&mcp_tool(
            "x",
            ToolClass::StateChanging,
            ToolProfile::Ask,
            true,
        ));
        assert!(!s.tainted, "trusted server output does not taint");
        s.record_result(&mcp_tool(
            "x",
            ToolClass::ReadOnlyLocal,
            ToolProfile::Ask,
            false,
        ));
        assert!(s.tainted, "untrusted server output taints");
        let mut s = SessionState::new();
        s.record_result(&tool(
            "web_search",
            ToolClass::OpenWorldRead,
            ToolProfile::Allow,
        ));
        assert!(s.tainted);
    }

    #[test]
    fn approved_helper_clears_needs_approval() {
        let d = PolicyDecision {
            allow: true,
            needs_approval: true,
            reason: "x",
            effective_class: ToolClass::StateChanging,
        };
        let a = d.approved();
        assert!(a.is_immediate());
        assert_eq!(a.reason, "approved by human");
    }
}
