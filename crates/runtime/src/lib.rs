//! Shared assembly for llmario front-ends (the CLI and the desktop app): the standard set of
//! engine adapters, supervisor construction, and a streaming chat client with cancellation.

pub mod chat;
pub mod library;

use llmario_core::{Config, Paths};
use llmario_hardware::HardwareReport;
use llmario_supervisor::{EngineAdapter, Supervisor};
use std::path::PathBuf;
use std::sync::Arc;

/// Engine adapters every front-end ships. `mock_engine` is the executable that implements the
/// hidden `mock-engine` subcommand (the CLI binary); front-ends without it pass `None`.
pub fn standard_adapters(
    paths: &Paths,
    mock_engine: Option<PathBuf>,
) -> Vec<Arc<dyn EngineAdapter>> {
    let mut v: Vec<Arc<dyn EngineAdapter>> = vec![
        Arc::new(llmario_adapter_llamacpp::LlamaCppAdapter),
        Arc::new(llmario_adapter_mlx::MlxAdapter::new(&paths.home)),
    ];
    if let Some(program) = mock_engine {
        v.push(Arc::new(llmario_adapter_mock::MockAdapter {
            program,
            args_prefix: vec!["mock-engine".into()],
        }));
    }
    v
}

pub async fn detect_hardware() -> HardwareReport {
    tokio::task::spawn_blocking(HardwareReport::detect)
        .await
        .expect("hardware detection panicked")
}

/// Detect hardware, probe backends and build a supervisor (blocking work off the async runtime).
pub async fn build_supervisor(
    cfg: Config,
    paths: Paths,
    mock_engine: Option<PathBuf>,
) -> anyhow::Result<Arc<Supervisor>> {
    let hw = detect_hardware().await;
    let adapters = standard_adapters(&paths, mock_engine);
    tokio::task::spawn_blocking(move || Supervisor::new(cfg, paths, hw, adapters)).await?
}
