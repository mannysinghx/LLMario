//! LLMario desktop app: a native window (Tauri + system webview) around the llmario runtime.
//! Inference runs in engine child processes supervised in-process; the UI talks to them through
//! the same gateway code path as `llmario serve`, on a private loopback port.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod shell_env;
mod state;

use tauri::Manager;

fn main() {
    // Before any threads start: apps launched from Finder get a minimal PATH and would not find
    // llama-server or the MLX Python install.
    shell_env::inherit_login_path();

    let app = tauri::Builder::default()
        .manage(state::AppState::default())
        .invoke_handler(tauri::generate_handler![
            commands::start,
            commands::overview,
            commands::apply_settings,
            commands::list_models,
            commands::list_catalog,
            commands::pull_model,
            commands::remove_model,
            commands::load_model,
            commands::unload_model,
            commands::chat,
            commands::cancel_chat,
        ])
        .build(tauri::generate_context!())
        .expect("failed to build the LLMario app");

    app.run(|handle, event| {
        if let tauri::RunEvent::Exit = event {
            // Stop every engine process before the app exits (no orphans).
            let st = handle.state::<state::AppState>();
            tauri::async_runtime::block_on(st.shutdown());
        }
    });
}
