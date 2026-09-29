use crate::RuntimeArgs;
use llmario_core::{BackendKind, Config, Paths};
use llmario_hardware::HardwareReport;
use llmario_supervisor::memory::fmt_bytes;
use llmario_supervisor::{EngineAdapter, Supervisor};
use std::sync::Arc;

pub fn load_config(rt: &RuntimeArgs) -> anyhow::Result<(Paths, Config)> {
    let paths = Paths::from_env()?;
    paths.ensure()?;
    let mut cfg = Config::load(&paths)?;
    if let Some(p) = rt.profile {
        cfg.runtime.profile = p;
    }
    if let Some(c) = rt.context {
        cfg.runtime.context = Some(c);
    }
    if let Some(m) = rt.memory_limit_gb {
        cfg.runtime.memory_limit_gb = Some(m);
    }
    if let Some(b) = rt.backend {
        cfg.backends.prefer = Some(b);
    }
    cfg.validate()?;
    Ok((paths, cfg))
}

pub fn adapters(paths: &Paths) -> Vec<Arc<dyn EngineAdapter>> {
    // The CLI binary implements the hidden `mock-engine` subcommand used by tests.
    llmario_runtime::standard_adapters(paths, std::env::current_exe().ok())
}

pub async fn detect_hardware() -> HardwareReport {
    llmario_runtime::detect_hardware().await
}

pub async fn supervisor(cfg: Config, paths: Paths) -> anyhow::Result<Arc<Supervisor>> {
    llmario_runtime::build_supervisor(cfg, paths, std::env::current_exe().ok()).await
}

pub fn print_backends(sup: &Supervisor) {
    for kind in [BackendKind::LlamaCpp, BackendKind::Mlx] {
        if let Some(s) = sup.statuses().get(&kind) {
            let mark = if s.available { "✓" } else { "✗" };
            eprintln!(
                "{mark} {:<9} {}  {}",
                kind.to_string(),
                s.version.as_deref().unwrap_or(""),
                s.detail
            );
        }
    }
}

/// Print the selection and memory estimate, load the model, and report readiness.
pub async fn load_with_report(sup: &Arc<Supervisor>, name: &str) -> anyhow::Result<()> {
    let sel = sup.select(name)?;
    let plan = sup.plan_memory(&sel, 0);
    eprintln!("model:   {}", sel.reason);
    eprintln!(
        "profile: {} — {} slot(s) × {} tokens",
        sel.profile.kind, sel.profile.parallel, sel.profile.ctx_per_slot
    );
    eprintln!("memory:  {}", plan.explain());
    for n in &plan.notes {
        eprintln!("         note: {n}");
    }
    if !plan.fits {
        anyhow::bail!(plan.refusal(&sel.model.id, &sel.profile));
    }
    eprint!("loading {} via {}… ", sel.model.id, sel.backend);
    let lease = sup
        .acquire(name)
        .await
        .inspect_err(|_| eprintln!("failed"))?;
    let e = &lease.engine;
    let resident = llmario_hardware::process_memory_bytes(e.pid)
        .map(fmt_bytes)
        .unwrap_or_else(|| "?".into());
    eprintln!(
        "ready in {:.2}s (pid {}, footprint {resident})",
        e.ready_after().as_secs_f64(),
        e.pid
    );
    for n in &e.notes {
        eprintln!("         note: {n}");
    }
    Ok(())
}

pub fn power_note() -> String {
    if cfg!(target_os = "macos") {
        let batt = std::process::Command::new("pmset")
            .args(["-g", "batt"])
            .output()
            .ok();
        let src = batt
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
            .and_then(|t| {
                t.lines().next().map(|l| {
                    l.replace("Now drawing from", "")
                        .trim()
                        .trim_matches('\'')
                        .to_string()
                })
            })
            .unwrap_or_else(|| "unknown".into());
        let low = std::process::Command::new("pmset")
            .args(["-g"])
            .output()
            .ok()
            .map(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .lines()
                    .any(|l| l.contains("lowpowermode") && l.trim().ends_with('1'))
            })
            .unwrap_or(false);
        format!("{src}{}", if low { ", Low Power Mode ON" } else { "" })
    } else {
        "not recorded on this OS".into()
    }
}
