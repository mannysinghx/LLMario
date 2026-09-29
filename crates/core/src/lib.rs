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

/// Project name. Used for the home directory, env-var prefix and user agent, so a fork can
/// rename the project by changing this constant and the `[[bin]]` name.
pub const APP_NAME: &str = "llmario";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn user_agent() -> String {
    format!("{APP_NAME}/{VERSION}")
}
