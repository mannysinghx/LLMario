//! Typed configuration. Precedence (lowest → highest):
//! built-in defaults < `$LLMARIO_HOME/config.toml` < `LLMARIO_*` env vars < CLI flags.
//! CLI flags are applied by the `cli` crate after [`Config::load`].

use crate::types::{BackendKind, ProfileKind};
use crate::Paths;
use serde::{Deserialize, Serialize};
use std::net::IpAddr;
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub server: ServerConfig,
    pub runtime: RuntimeConfig,
    pub backends: BackendsConfig,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    /// Bind address. Loopback by default; anything else requires `allow_remote` + `api_key`.
    pub host: String,
    pub port: u16,
    /// Bearer token required on every request when set. Prefer the `LLMARIO_API_KEY` env var.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Explicit opt-in to binding a non-loopback interface.
    pub allow_remote: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 11500,
            api_key: None,
            allow_remote: false,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct RuntimeConfig {
    pub profile: ProfileKind,
    /// Per-request context tokens. `None` = the profile's value.
    pub context: Option<u32>,
    /// Maximum models resident at once. Laptops: keep at 1.
    pub max_loaded_models: usize,
    /// Memory kept free for the OS and other apps. `None` = max(2 GiB, 10% of RAM).
    pub memory_headroom_gb: Option<f64>,
    /// Hard cap on memory llmario may plan to use. `None` = the detected accelerator budget.
    pub memory_limit_gb: Option<f64>,
    /// How long a request waits for a free slot or a model swap before 503.
    pub queue_timeout_secs: u64,
    /// Upper bound on a single request, including streaming.
    pub request_timeout_secs: u64,
    /// Time allowed for an engine to load weights and answer a warm-up request.
    pub engine_start_timeout_secs: u64,
    /// Unload a model after this many idle seconds. 0 = never.
    pub idle_unload_secs: u64,
    /// Relaunches allowed after engine crashes within 10 minutes before giving up.
    pub max_restarts: u32,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            profile: ProfileKind::Latency,
            context: None,
            max_loaded_models: 1,
            memory_headroom_gb: None,
            memory_limit_gb: None,
            queue_timeout_secs: 120,
            request_timeout_secs: 900,
            engine_start_timeout_secs: 300,
            idle_unload_secs: 0,
            max_restarts: 3,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct BackendsConfig {
    /// Preferred backend when a model is available in several formats.
    pub prefer: Option<BackendKind>,
    pub llamacpp: LlamaCppConfig,
    pub mlx: MlxConfig,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct LlamaCppConfig {
    /// Path to `llama-server`. `None` = search `PATH`.
    pub server_path: Option<PathBuf>,
    /// Layers to offload to the GPU. `None` = computed from the memory plan.
    pub gpu_layers: Option<i32>,
    /// Extra flags appended verbatim (advanced; bypasses profile validation).
    pub extra_args: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct MlxConfig {
    /// Python interpreter with `mlx-lm` installed. `None` = `$LLMARIO_HOME/venvs/mlx`, then the
    /// interpreter behind `mlx_lm.server` on `PATH`.
    pub python: Option<PathBuf>,
    pub extra_args: Vec<String>,
}

impl Config {
    /// Load defaults, then the config file (if present), then environment overrides.
    pub fn load(paths: &Paths) -> anyhow::Result<Self> {
        let file = paths.config_file();
        let mut cfg = if file.exists() {
            let text = std::fs::read_to_string(&file)?;
            Self::from_toml(&text).map_err(|e| anyhow::anyhow!("{}: {e}", file.display()))?
        } else {
            Self::default()
        };
        cfg.apply_env(|k| std::env::var(k).ok())?;
        Ok(cfg)
    }

    pub fn from_toml(text: &str) -> anyhow::Result<Self> {
        Ok(toml::from_str(text)?)
    }

    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).unwrap_or_default()
    }

    /// Apply `LLMARIO_*` overrides. Takes a lookup fn so tests do not touch process env.
    pub fn apply_env(&mut self, get: impl Fn(&str) -> Option<String>) -> anyhow::Result<()> {
        let p = crate::APP_NAME.to_ascii_uppercase();
        let var = |name: &str| get(&format!("{p}_{name}")).filter(|v| !v.is_empty());
        if let Some(v) = var("HOST") {
            self.server.host = v;
        }
        if let Some(v) = var("PORT") {
            self.server.port = v
                .parse()
                .map_err(|_| anyhow::anyhow!("{p}_PORT: not a port: {v}"))?;
        }
        if let Some(v) = var("API_KEY") {
            self.server.api_key = Some(v);
        }
        if let Some(v) = var("PROFILE") {
            self.runtime.profile = v.parse().map_err(anyhow::Error::msg)?;
        }
        if let Some(v) = var("CONTEXT") {
            self.runtime.context = Some(
                v.parse()
                    .map_err(|_| anyhow::anyhow!("{p}_CONTEXT: not a number: {v}"))?,
            );
        }
        if let Some(v) = var("LLAMA_SERVER") {
            self.backends.llamacpp.server_path = Some(v.into());
        }
        if let Some(v) = var("MLX_PYTHON") {
            self.backends.mlx.python = Some(v.into());
        }
        Ok(())
    }

    /// Enforce the security defaults: remote binding needs explicit opt-in and a key.
    pub fn validate(&self) -> anyhow::Result<()> {
        if !is_loopback_host(&self.server.host) {
            if !self.server.allow_remote {
                anyhow::bail!(
                    "refusing to bind non-loopback address '{}' without --allow-remote",
                    self.server.host
                );
            }
            if self.server.api_key.as_deref().is_none_or(str::is_empty) {
                anyhow::bail!(
                    "remote binding requires an API key (--api-key or {}_API_KEY)",
                    crate::APP_NAME.to_ascii_uppercase()
                );
            }
        }
        if self.runtime.max_loaded_models == 0 {
            anyhow::bail!("runtime.max_loaded_models must be at least 1");
        }
        if let Some(h) = self.runtime.memory_headroom_gb {
            if !(0.0..=1024.0).contains(&h) {
                anyhow::bail!("runtime.memory_headroom_gb out of range: {h}");
            }
        }
        Ok(())
    }
}

pub fn is_loopback_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    host.trim_matches(|c| c == '[' || c == ']')
        .parse::<IpAddr>()
        .map(|ip| ip.is_loopback())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn defaults_are_safe() {
        let c = Config::default();
        assert_eq!(c.server.host, "127.0.0.1");
        assert!(!c.server.allow_remote);
        assert_eq!(c.runtime.max_loaded_models, 1);
        c.validate().unwrap();
    }

    #[test]
    fn file_then_env_precedence() {
        let mut c = Config::from_toml(
            r#"
            [server]
            port = 9000
            [runtime]
            profile = "balanced"
            context = 4096
            "#,
        )
        .unwrap();
        assert_eq!(c.server.port, 9000);
        assert_eq!(c.runtime.profile, ProfileKind::Balanced);
        let env: HashMap<&str, &str> =
            [("LLMARIO_PORT", "9100"), ("LLMARIO_PROFILE", "throughput")].into();
        c.apply_env(|k| env.get(k).map(|s| s.to_string())).unwrap();
        assert_eq!(c.server.port, 9100);
        assert_eq!(c.runtime.profile, ProfileKind::Throughput);
        assert_eq!(
            c.runtime.context,
            Some(4096),
            "untouched file values survive"
        );
    }

    #[test]
    fn unknown_keys_rejected() {
        assert!(Config::from_toml("[server]\nhots = \"0.0.0.0\"").is_err());
    }

    #[test]
    fn bad_env_value_is_an_error() {
        let mut c = Config::default();
        assert!(c
            .apply_env(|k| (k == "LLMARIO_PORT").then(|| "abc".into()))
            .is_err());
    }

    #[test]
    fn remote_requires_opt_in_and_key() {
        let mut c = Config::default();
        c.server.host = "0.0.0.0".into();
        assert!(c.validate().is_err());
        c.server.allow_remote = true;
        assert!(c.validate().is_err(), "remote without key must fail");
        c.server.api_key = Some("s3cret".into());
        c.validate().unwrap();
    }

    #[test]
    fn loopback_detection() {
        for h in ["127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.2"] {
            assert!(is_loopback_host(h), "{h}");
        }
        for h in ["0.0.0.0", "192.168.1.5", "example.com", "::"] {
            assert!(!is_loopback_host(h), "{h}");
        }
    }
}
