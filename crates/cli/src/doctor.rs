use crate::util;
use llmario_core::{BackendKind, ModelFormat, ResolvedProfile};
use llmario_supervisor::memory::{self, fmt_bytes};
use llmario_supervisor::planner;
use serde_json::json;

pub async fn run(as_json: bool) -> anyhow::Result<()> {
    let (paths, cfg) = util::load_config(&Default::default())?;
    let sup = util::supervisor(cfg.clone(), paths.clone()).await?;
    let hw = &sup.hw;
    let reg = sup.registry();
    let orphans = llmario_supervisor::engine::find_orphans(&paths.run_dir());
    let other_app = llmario_supervisor::sibling::sibling_usage(&paths);

    let backends: Vec<_> = [BackendKind::LlamaCpp, BackendKind::Mlx]
        .iter()
        .filter_map(|k| sup.statuses().get(k).cloned())
        .collect();
    let models: Vec<_> = reg
        .models
        .iter()
        .filter(|m| m.format != ModelFormat::Mock)
        .map(|m| {
            let backend = planner::backend_for(m.format);
            let profile = planner::effective_profile(cfg.runtime.profile, cfg.runtime.context, m);
            let sel = planner::Selection {
                model: m.clone(),
                backend,
                profile: profile.clone(),
                reason: String::new(),
            };
            let plan = sup.plan_memory(&sel, 0);
            let backend_ok = sup.statuses().get(&backend).is_some_and(|s| s.available);
            (m, backend, profile, plan, backend_ok)
        })
        .collect();

    if as_json {
        let v = json!({
            "version": llmario_core::VERSION,
            "hardware": hw,
            "hardware_fingerprint": hw.fingerprint(),
            "backends": backends,
            "profile": ResolvedProfile::resolve(cfg.runtime.profile, cfg.runtime.context),
            "memory_profile": if memory::small_machine(hw, &cfg) { "small" } else { "standard" },
            "kv_cache_type": cfg.backends.llamacpp.kv_cache_type,
            "comfortable_bytes": memory::comfortable_bytes(hw),
            "speculative": {
                "llamacpp": cfg.backends.llamacpp.speculative,
                "llamacpp_draft_model": cfg.backends.llamacpp.draft_model,
                "mlx_draft_model": cfg.backends.mlx.draft_model,
            },
            "models": models.iter().map(|(m, b, p, plan, ok)| json!({
                "id": m.id, "format": m.format, "backend": b, "backend_available": ok,
                "size_bytes": m.size_bytes, "quantization": m.quantization,
                "profile": p, "memory": plan,
            })).collect::<Vec<_>>(),
            "orphaned_engines": orphans.iter().map(|(pid, model, prog)| json!({"pid": pid, "model": model, "program": prog})).collect::<Vec<_>>(),
            "other_app_engines": other_app,
            "paths": {"home": paths.home, "config": paths.config_file(), "registry": paths.registry_file()},
            "server": {"host": cfg.server.host, "port": cfg.server.port, "api_key_set": cfg.server.api_key.is_some()},
        });
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }

    println!(
        "{} {} — doctor\n",
        llmario_core::APP_NAME,
        llmario_core::VERSION
    );
    println!("Hardware");
    let cores = match (hw.performance_cores, hw.efficiency_cores) {
        (Some(p), Some(e)) => format!("{p}P + {e}E cores, {} threads", hw.logical_cores),
        _ => format!("{} cores, {} threads", hw.physical_cores, hw.logical_cores),
    };
    println!(
        "  CPU      {} ({cores}) [{}]",
        hw.cpu_brand,
        hw.cpu_features.join(" ")
    );
    println!(
        "  Memory   {} total, {} available{}",
        fmt_bytes(hw.total_memory_bytes),
        fmt_bytes(hw.available_memory_bytes),
        if hw.unified_memory {
            " (unified with GPU)"
        } else {
            ""
        }
    );
    for g in &hw.gpus {
        println!(
            "  GPU      {} ({:?}{}) memory {}{}",
            g.name,
            g.api,
            g.cores.map(|c| format!(", {c} cores")).unwrap_or_default(),
            g.memory_total_bytes
                .map(fmt_bytes)
                .unwrap_or_else(|| "unknown".into()),
            if g.api == llmario_hardware::GpuApi::Metal {
                " usable working set"
            } else {
                ""
            }
        );
    }
    println!("  OS       {}", hw.os_version);
    println!("  ID       {}", hw.fingerprint());
    for n in &hw.notes {
        println!("  note     {n}");
    }

    println!("\nBackends");
    for b in &backends {
        println!(
            "  {} {:<9} {:<32} {}",
            if b.available { "✓" } else { "✗" },
            b.kind.to_string(),
            b.version.as_deref().unwrap_or("-"),
            b.detail
        );
        if let Some(p) = &b.path {
            println!(
                "    {:<9} {}  (tested: {})",
                "",
                p.display(),
                b.tested_version
            );
        }
    }

    let p = ResolvedProfile::resolve(cfg.runtime.profile, cfg.runtime.context);
    println!(
        "\nProfile  {} — {} slot(s) × {} tokens (change with --profile / --context)",
        p.kind, p.parallel, p.ctx_per_slot
    );
    println!(
        "Headroom {} kept free for the OS and apps",
        fmt_bytes(memory::headroom_bytes(hw, &cfg))
    );
    println!(
        "Memory   {} reserves (runtime.memory_profile = \"{}\"); llama.cpp KV cache {}{}",
        if memory::small_machine(hw, &cfg) {
            "small"
        } else {
            "standard"
        },
        format!("{:?}", cfg.runtime.memory_profile).to_lowercase(),
        cfg.backends.llamacpp.kv_cache_type.as_arg(),
        memory::comfortable_bytes(hw)
            .map(|c| format!("; comfortable up to {}", fmt_bytes(c)))
            .unwrap_or_default()
    );
    let l = &cfg.backends.llamacpp;
    println!(
        "Speed    speculative decoding: llama.cpp {}{}; MLX draft model {}",
        format!("{:?}", l.speculative).to_lowercase(),
        l.draft_model
            .as_deref()
            .map(|d| format!(" (draft {d})"))
            .unwrap_or_default(),
        cfg.backends.mlx.draft_model.as_deref().unwrap_or("none")
    );

    println!("\nModels ({} installed)", models.len());
    if models.is_empty() {
        let cli = llmario_core::APP_NAME;
        println!("  none — try `{cli} model catalog` then `{cli} model pull qwen3-1.7b`");
    }
    for (m, b, _p, plan, ok) in &models {
        let fit = if !ok {
            format!("✗ {b} backend unavailable")
        } else if plan.fits {
            format!(
                "✓ fits{}: ~{} of {}",
                if plan.tight { ", tight" } else { "" },
                fmt_bytes(plan.total_bytes),
                fmt_bytes(plan.budget_bytes)
            )
        } else {
            format!(
                "✗ too big: ~{} of {}",
                fmt_bytes(plan.total_bytes),
                fmt_bytes(plan.budget_bytes)
            )
        };
        println!(
            "  {:<26} {:<5} {:<18} {:>10}  {fit}",
            m.id,
            m.format.to_string(),
            m.quantization.as_deref().unwrap_or("-"),
            fmt_bytes(m.size_bytes)
        );
    }

    println!(
        "\nServer   http://{}:{} (api key {})",
        cfg.server.host,
        cfg.server.port,
        if cfg.server.api_key.is_some() {
            "set"
        } else {
            "not set"
        }
    );
    println!("Home     {}", paths.home.display());
    if orphans.is_empty() {
        println!("Engines  no orphaned engine processes");
    } else {
        println!("Engines  ⚠ engine processes left running by an llmario that exited (not killed automatically):");
        for (pid, model, prog) in &orphans {
            println!(
                "           pid {pid}  {model}  {prog}   → `kill {pid}` if you do not need it"
            );
        }
    }
    if let Some(u) = &other_app {
        println!("Also     ⚠ {}", u.warning());
    }
    Ok(())
}
