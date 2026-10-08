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
            port: crate::DEFAULT_PORT,
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
    /// How the memory planner sizes the KV cache.
    pub kv_accounting: KvAccounting,
    /// Size of llmario's fixed memory reserves (prompt caches, buffer cache).
    pub memory_profile: MemoryProfile,
}

/// Size of llmario's fixed memory reserves: llama.cpp's host prompt cache (`--cache-ram`), and
/// MLX's prompt-cache entries and buffer cache.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "kebab-case")]
pub enum MemoryProfile {
    /// llama.cpp 1 GiB host prompt cache; MLX the profile's prompt-cache entries, 1 GiB buffer
    /// cache.
    #[default]
    Standard,
    /// For machines with 16 GB or less: llama.cpp 256 MiB host prompt cache; MLX 1 prompt-cache
    /// entry, 512 MiB buffer cache. Less memory, less prompt-prefix reuse.
    Small,
    /// `small` on machines with 16 GB of memory or less, otherwise `standard`.
    Auto,
}

/// How the memory planner sizes the KV cache.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "kebab-case")]
pub enum KvAccounting {
    /// Count only caches that grow, when the model's layout is known: full-attention layers for
    /// the whole context, sliding-window layers up to their window, and linear-attention layers
    /// as fixed state. Models with an unknown layout fall back to `conservative`.
    #[default]
    PerLayer,
    /// Every layer as full attention (the planner's formula before LLMario Beta Phase 2).
    Conservative,
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
            kv_accounting: KvAccounting::PerLayer,
            memory_profile: MemoryProfile::Standard,
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
    /// KV cache element type (`-ctk`/`-ctv`).
    pub kv_cache_type: KvCacheType,
    /// Speculative decoding (`--spec-type`).
    pub speculative: Speculative,
    /// Registered GGUF model used as the draft for `speculative = "draft"`; it must share the
    /// main model's tokenizer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub draft_model: Option<String>,
    /// Tokens drafted per step (`--spec-draft-n-max`). `None` = 1 for `mtp` (measured best on
    /// Qwen3.5 9B: +22% on a code edit, +1% on prose; 3 tokens slowed prose by 41%) and the
    /// engine's default (3) for `draft`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub draft_tokens: Option<u32>,
    /// Extra flags appended verbatim (advanced; bypasses profile validation).
    pub extra_args: Vec<String>,
}

/// llama.cpp speculative decoding: the model checks several guessed tokens per pass over its
/// weights. Output is unchanged; only speed differs, and it can be slower when guesses miss.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "kebab-case")]
pub enum Speculative {
    #[default]
    Off,
    /// Guess from repeated text in the conversation (`ngram-simple`). No extra memory; helps
    /// when the answer repeats the input (code edits, quoting), slightly slower otherwise.
    Ngram,
    /// The model's own multi-token-prediction layers (`draft-mtp`), for builds that ship them
    /// (catalog ids ending in `-mtp`). Other models run without speculation.
    Mtp,
    /// A separate small model (`draft-simple` with `draft_model`).
    Draft,
}

/// llama.cpp KV cache element type.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum KvCacheType {
    /// 16-bit (llama.cpp's default).
    #[default]
    #[serde(rename = "f16")]
    F16,
    /// 8-bit: 34 bytes per 32 values, so the KV cache takes 17/32 of its f16 size. llama.cpp
    /// turns flash attention on for it (required for a quantized V cache). Recurrent state is
    /// unaffected.
    #[serde(rename = "q8_0")]
    Q8_0,
}

impl KvCacheType {
    /// The value for `-ctk`/`-ctv`.
    pub fn as_arg(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct MlxConfig {
    /// Python interpreter with `mlx-lm` installed. `None` = `$LLMARIO_HOME/venvs/mlx`, then the
    /// interpreter behind `mlx_lm.server` on `PATH`.
    pub python: Option<PathBuf>,
    /// Registered MLX model used as a draft model (`--draft-model`); it must share the main
    /// model's tokenizer. Used only with one request at a time (`latency` profile): MLX-LM turns
    /// off request batching when a draft model is set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub draft_model: Option<String>,
    /// Tokens drafted per step (`--num-draft-tokens`). `None` = MLX-LM's default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub draft_tokens: Option<u32>,
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
        let p = crate::ENV_PREFIX;
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
                    crate::ENV_PREFIX
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
        let p = crate::ENV_PREFIX;
        let env: HashMap<String, &str> = [
            (format!("{p}_PORT"), "9100"),
            (format!("{p}_PROFILE"), "throughput"),
        ]
        .into();
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
    fn kv_accounting_defaults_and_parses() {
        assert_eq!(
            Config::default().runtime.kv_accounting,
            KvAccounting::PerLayer
        );
        let c = Config::from_toml("[runtime]\nkv_accounting = \"conservative\"").unwrap();
        assert_eq!(c.runtime.kv_accounting, KvAccounting::Conservative);
        assert!(Config::from_toml("[runtime]\nkv_accounting = \"guess\"").is_err());
    }

    #[test]
    fn memory_profile_and_kv_cache_type_parse() {
        let c = Config::default();
        assert_eq!(c.runtime.memory_profile, MemoryProfile::Standard);
        assert_eq!(c.backends.llamacpp.kv_cache_type, KvCacheType::F16);
        let c = Config::from_toml(
            "[runtime]\nmemory_profile = \"auto\"\n[backends.llamacpp]\nkv_cache_type = \"q8_0\"",
        )
        .unwrap();
        assert_eq!(c.runtime.memory_profile, MemoryProfile::Auto);
        assert_eq!(c.backends.llamacpp.kv_cache_type, KvCacheType::Q8_0);
        assert_eq!(KvCacheType::Q8_0.as_arg(), "q8_0");
        assert!(Config::from_toml("[runtime]\nmemory_profile = \"tiny\"").is_err());
        assert!(Config::from_toml("[backends.llamacpp]\nkv_cache_type = \"q4_0\"").is_err());
    }

    #[test]
    fn speculative_settings_parse() {
        let c = Config::default();
        assert_eq!(c.backends.llamacpp.speculative, Speculative::Off);
        assert!(c.backends.llamacpp.draft_model.is_none() && c.backends.mlx.draft_model.is_none());
        let c = Config::from_toml(
            "[backends.llamacpp]\nspeculative = \"draft\"\ndraft_model = \"small\"\ndraft_tokens = 2\n[backends.mlx]\ndraft_model = \"tiny\"",
        )
        .unwrap();
        assert_eq!(c.backends.llamacpp.speculative, Speculative::Draft);
        assert_eq!(c.backends.llamacpp.draft_model.as_deref(), Some("small"));
        assert_eq!(c.backends.llamacpp.draft_tokens, Some(2));
        assert_eq!(c.backends.mlx.draft_model.as_deref(), Some("tiny"));
        for m in ["off", "ngram", "mtp"] {
            Config::from_toml(&format!("[backends.llamacpp]\nspeculative = \"{m}\"")).unwrap();
        }
        assert!(Config::from_toml("[backends.llamacpp]\nspeculative = \"eagle\"").is_err());
    }

    #[test]
    fn unknown_keys_rejected() {
        assert!(Config::from_toml("[server]\nhots = \"0.0.0.0\"").is_err());
    }

    #[test]
    fn bad_env_value_is_an_error() {
        let mut c = Config::default();
        assert!(c
            .apply_env(|k| (k == format!("{}_PORT", crate::ENV_PREFIX)).then(|| "abc".into()))
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
