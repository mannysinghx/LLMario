//! Shared types for every llmario crate: configuration, filesystem layout, error
//! taxonomy, and the enums that name formats, backends and workload profiles.

pub mod config;
pub mod error;
pub mod os;
pub mod paths;
pub mod types;

pub use config::Config;
pub use error::RuntimeError;
pub use paths::Paths;
pub use types::{BackendKind, ModelFormat, ProfileKind, ResolvedProfile};

// LLMario Beta identity (Phase 0): a separate edition that installs and runs alongside
// production (`llmario`, `~/.llmario`, `LLMARIO_*`, port 11500) without touching it.

/// Project name: the home directory (`~/.<APP_NAME>`), the command-line binary named in hints,
/// and the user agent. A fork or a separate edition renames the project by changing these
/// constants and the `[[bin]]` names.
pub const APP_NAME: &str = "llmario-beta";
/// Prefix of the environment variables (`<ENV_PREFIX>_HOME`, `<ENV_PREFIX>_PORT`, …).
pub const ENV_PREFIX: &str = "LLMARIO_BETA";
/// Default port of the local API (`serve`).
pub const DEFAULT_PORT: u16 = 11501;
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn user_agent() -> String {
    format!("{APP_NAME}/{VERSION}")
}
