//! What LLMario's own engine says about its memory while it runs.
//!
//! The engine reports two things on `/engine/stats` that the plan made at load cannot know: the
//! pressure level the operating system reports and what the engine did about it (idle caches
//! freed), and whether the weights no longer fit in memory, so every token reads them from
//! storage again. Only the built-in engine serves these; llama.cpp and MLX report neither.

use std::time::Duration;

use serde::Serialize;
use serde_json::Value;

/// The engine's memory state, as shown to people.
#[derive(Serialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct EngineMemory {
    /// `normal`, `warning`, `critical`, or `unknown` (the system does not say).
    pub pressure_level: String,
    /// Whether the engine frees idle conversations when the system runs short.
    pub pressure_relief: bool,
    /// How many times it has done so, and how much memory that gave back.
    pub pressure_sheds: u64,
    pub pressure_released_bytes: u64,
    /// The weights are being read from storage for every token: the model does not fit memory.
    pub paging: bool,
    /// Bytes read from storage for the last reply, per generated token.
    pub paged_bytes_per_token: u64,
}

impl EngineMemory {
    /// Whether anything here is worth telling the person about.
    pub fn needs_attention(&self) -> bool {
        self.paging || self.pressure_level == "warning" || self.pressure_level == "critical"
    }
}

/// Reads the memory fields out of an `/engine/stats` body; `None` when they are absent (an
/// older engine).
pub fn parse(stats: &Value) -> Option<EngineMemory> {
    let p = stats.get("memory_pressure")?;
    let g = stats.get("paging")?;
    let n = |v: &Value, k: &str| v.get(k).and_then(Value::as_u64).unwrap_or(0);
    Some(EngineMemory {
        pressure_level: p
            .get("level")
            .and_then(Value::as_str)
            .filter(|l| matches!(*l, "normal" | "warning" | "critical"))
            .unwrap_or("unknown")
            .to_string(),
        pressure_relief: p.get("relief").and_then(Value::as_bool).unwrap_or(false),
        pressure_sheds: n(p, "sheds"),
        pressure_released_bytes: n(p, "released_bytes"),
        paging: g
            .get("model_does_not_fit_memory")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        paged_bytes_per_token: n(g, "bytes_per_token_last"),
    })
}

/// Asks the engine listening on `port` (loopback) for its memory state. `None` on any failure:
/// the engine is busy loading, gone, or too old to report it.
pub async fn fetch(http: &reqwest::Client, port: u16) -> Option<EngineMemory> {
    let body: Value = http
        .get(format!("http://127.0.0.1:{port}/engine/stats"))
        .timeout(Duration::from_secs(2))
        .send()
        .await
        .ok()?
        .error_for_status()
        .ok()?
        .json()
        .await
        .ok()?;
    parse(&body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn reads_the_engine_s_own_field_names() {
        let m = parse(&json!({
            "memory_pressure": {"relief": true, "level": "warning", "sheds": 2, "released_bytes": 123},
            "paging": {"bytes": 9, "tokens": 3, "bytes_per_token_last": 777, "model_does_not_fit_memory": true},
        }))
        .unwrap();
        assert_eq!(m.pressure_level, "warning");
        assert!(m.pressure_relief && m.paging && m.needs_attention());
        assert_eq!((m.pressure_sheds, m.pressure_released_bytes), (2, 123));
        assert_eq!(m.paged_bytes_per_token, 777);
    }

    #[test]
    fn a_calm_engine_needs_no_attention_and_unknown_levels_stay_unknown() {
        let calm = parse(&json!({
            "memory_pressure": {"relief": true, "level": "normal", "sheds": 0, "released_bytes": 0},
            "paging": {"model_does_not_fit_memory": false},
        }))
        .unwrap();
        assert!(!calm.needs_attention());
        let odd = parse(&json!({
            "memory_pressure": {"level": "<script>", "relief": false},
            "paging": {},
        }))
        .unwrap();
        assert_eq!(odd.pressure_level, "unknown");
        assert!(!odd.needs_attention());
    }

    #[test]
    fn an_engine_without_the_fields_gives_none() {
        assert_eq!(parse(&json!({"cancelled": 0})), None);
        assert_eq!(parse(&json!({"memory_pressure": {}})), None);
    }
}
