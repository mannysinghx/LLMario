use crate::util;
use clap::Subcommand;
use llmario_core::{BackendKind, ModelFormat};
use llmario_registry::download::{self, HubClient, PullOptions};
use llmario_registry::{Catalog, Registry};
use llmario_supervisor::memory::fmt_bytes;
use llmario_supervisor::planner;
use std::path::PathBuf;

#[derive(Subcommand)]
pub enum ModelCmd {
    /// List installed models.
    List {
        #[arg(long)]
        json: bool,
    },
    /// List models available to pull.
    Catalog,
    /// Download a catalog model (id or family name) with checksum verification.
    Pull {
        name: String,
        /// Re-download even if installed.
        #[arg(long)]
        force: bool,
        /// Do not reuse matching blobs from the local Hugging Face cache.
        #[arg(long)]
        no_hf_cache: bool,
    },
    /// Register a model already on disk (a .gguf file or an MLX model directory). Files are
    /// hashed, not copied, and never deleted by `remove`.
    Add {
        path: PathBuf,
        /// Local id (default: derived from the path).
        #[arg(long)]
        name: Option<String>,
        /// Family name used for backend auto-selection.
        #[arg(long)]
        family: Option<String>,
    },
    /// Remove a model (deletes files only for models downloaded by `pull`).
    Remove { id: String },
    /// Show everything recorded about a model.
    Info { id: String },
    /// Re-hash an installed model's files against the recorded checksums.
    Verify { id: String },
    /// Estimate memory for a model with a profile/context, without loading it.
    Fit {
        name: String,
        #[command(flatten)]
        rt: crate::RuntimeArgs,
    },
}

pub async fn run(cmd: ModelCmd) -> anyhow::Result<()> {
    let (paths, _cfg) = util::load_config(&Default::default())?;
    let mut reg = Registry::load(&paths.registry_file())?;
    match cmd {
        ModelCmd::List { json } => {
            let models: Vec<_> = reg
                .models
                .iter()
                .filter(|m| m.format != ModelFormat::Mock)
                .collect();
            if json {
                println!("{}", serde_json::to_string_pretty(&models)?);
                return Ok(());
            }
            if models.is_empty() {
                println!("No models installed. See `llmario model catalog`.");
            }
            println!(
                "{:<26} {:<16} {:<5} {:<18} {:>10}  SOURCE",
                "ID", "FAMILY", "FMT", "QUANT", "SIZE"
            );
            for m in models {
                println!(
                    "{:<26} {:<16} {:<5} {:<18} {:>10}  {}",
                    m.id,
                    m.family.as_deref().unwrap_or("-"),
                    m.format.to_string(),
                    m.quantization.as_deref().unwrap_or("-"),
                    fmt_bytes(m.size_bytes),
                    m.source
                        .as_ref()
                        .map(|s| format!("{}@{}", s.repo, &s.revision[..s.revision.len().min(8)]))
                        .unwrap_or_else(|| paths.redact(&m.path))
                );
            }
        }
        ModelCmd::Catalog => {
            let cat = Catalog::builtin();
            println!(
                "{:<24} {:<14} {:<5} {:<28} DESCRIPTION",
                "ID", "FAMILY", "FMT", "LICENSE"
            );
            for m in &cat.models {
                let installed = if reg.get(&m.id).is_some() {
                    " [installed]"
                } else {
                    ""
                };
                println!(
                    "{:<24} {:<14} {:<5} {:<28} {}{installed}",
                    m.id,
                    m.family,
                    m.format.to_string(),
                    m.license,
                    m.description
                );
            }
            println!("\nPull by id, or by family to get the best variant for this machine: `llmario model pull qwen3-1.7b`");
        }
        ModelCmd::Pull {
            name,
            force,
            no_hf_cache,
        } => {
            let cat = Catalog::builtin();
            let entry = match cat.get(&name) {
                Some(e) => e.clone(),
                None => {
                    let variants = cat.family(&name);
                    if variants.is_empty() {
                        anyhow::bail!(
                            "'{name}' is not in the catalog (see `llmario model catalog`)"
                        );
                    }
                    // Pick the variant this machine would serve best.
                    let (_, cfg) = util::load_config(&Default::default())?;
                    let hw = util::detect_hardware().await;
                    let statuses: std::collections::HashMap<_, _> = util::adapters(&paths)
                        .iter()
                        .map(|a| (a.kind(), a.probe(&hw, &cfg)))
                        .collect();
                    let mut ranked: Vec<_> = variants
                        .into_iter()
                        .filter(|v| {
                            statuses
                                .get(&planner::backend_for(v.format))
                                .is_some_and(|s| s.available)
                        })
                        .collect();
                    ranked.sort_by_key(|v| {
                        let b = planner::backend_for(v.format);
                        if cfg.backends.prefer == Some(b) {
                            0
                        } else if hw.apple_silicon && b == BackendKind::Mlx {
                            1
                        } else {
                            2
                        }
                    });
                    let chosen = ranked.first().ok_or_else(|| anyhow::anyhow!("no installed backend can run any variant of '{name}'; run `llmario doctor`"))?;
                    eprintln!(
                        "family '{name}' → {} ({} via {})",
                        chosen.id,
                        chosen.format,
                        planner::backend_for(chosen.format)
                    );
                    (*chosen).clone()
                }
            };
            eprintln!(
                "pulling {} from huggingface.co/{} ({}) — license: {}",
                entry.id, entry.repo, entry.revision, entry.license
            );
            let hub = HubClient::from_env()?;
            let out = download::pull(
                &hub,
                &entry,
                &paths,
                &mut reg,
                &PullOptions {
                    force,
                    show_progress: true,
                    use_hf_cache: !no_hf_cache,
                    on_progress: None,
                },
            )
            .await?;
            let rev = out
                .entry
                .source
                .as_ref()
                .map(|s| s.revision.clone())
                .unwrap_or_default();
            println!(
                "✓ {} installed ({}; downloaded {}, reused {} from local HF cache; commit {})",
                out.entry.id,
                fmt_bytes(out.entry.size_bytes),
                fmt_bytes(out.downloaded_bytes),
                fmt_bytes(out.reused_bytes),
                &rev[..rev.len().min(12)]
            );
            println!(
                "  license: {} — review the model card before use.",
                entry.license
            );
            println!("  try: llmario run {} \"hello\"", out.entry.id);
        }
        ModelCmd::Add { path, name, family } => {
            eprintln!("hashing files…");
            let mut e = download::add_local(&path, name.as_deref(), &mut reg)?;
            if let Some(f) = family {
                reg.models
                    .iter_mut()
                    .filter(|m| m.id == e.id)
                    .for_each(|m| m.family = Some(f.clone()));
                reg.save()?;
                e.family = Some(f);
            }
            println!(
                "✓ registered {} ({} {}, {}) — files stay at {} and will not be deleted by `remove`",
                e.id,
                e.format,
                e.quantization.as_deref().unwrap_or(""),
                fmt_bytes(e.size_bytes),
                e.path.display()
            );
        }
        ModelCmd::Remove { id } => {
            let e = download::remove(&id, &paths, &mut reg)?;
            if e.managed {
                println!(
                    "✓ removed {id} and deleted {} of downloaded files",
                    fmt_bytes(e.size_bytes)
                );
            } else {
                println!(
                    "✓ unregistered {id}; files left in place at {}",
                    e.path.display()
                );
            }
        }
        ModelCmd::Info { id } => {
            let m = reg
                .get(&id)
                .ok_or_else(|| anyhow::anyhow!("model '{id}' is not registered"))?;
            println!("{}", toml::to_string_pretty(m)?);
            println!(
                "# content hash: {}",
                m.content_hash()
                    .unwrap_or_else(|| "(missing checksums)".into())
            );
        }
        ModelCmd::Verify { id } => {
            let m = reg
                .get(&id)
                .ok_or_else(|| anyhow::anyhow!("model '{id}' is not registered"))?;
            eprintln!("verifying {} file(s)…", m.files.len());
            let problems = download::verify(m)?;
            if problems.is_empty() {
                println!("✓ all files match their recorded SHA-256");
            } else {
                for p in &problems {
                    println!("✗ {p}");
                }
                anyhow::bail!("{} problem(s) found", problems.len());
            }
        }
        ModelCmd::Fit { name, rt } => {
            let (paths, cfg) = util::load_config(&rt)?;
            let sup = util::supervisor(cfg, paths).await?;
            let sel = sup.select(&name)?;
            let plan = sup.plan_memory(&sel, 0);
            println!("{}", sel.reason);
            println!(
                "profile {}: {} slot(s) × {} tokens",
                sel.profile.kind, sel.profile.parallel, sel.profile.ctx_per_slot
            );
            println!("weights   {}", fmt_bytes(plan.weights_bytes));
            println!(
                "KV cache  {} ({} per token × {} tokens)",
                fmt_bytes(plan.kv_cache_bytes),
                fmt_bytes(plan.kv_bytes_per_token),
                plan.ctx_total
            );
            println!("overhead  {}", fmt_bytes(plan.overhead_bytes));
            println!("total     {}", fmt_bytes(plan.total_bytes));
            println!(
                "budget    {} ({})",
                fmt_bytes(plan.budget_bytes),
                plan.budget_source
            );
            if let Some(l) = plan.gpu_layers {
                println!("gpu layers {l}");
            }
            for n in &plan.notes {
                println!("note      {n}");
            }
            if plan.fits {
                println!(
                    "✓ fits (largest per-request context that would fit: {} tokens)",
                    plan.max_ctx_per_slot_that_fits.unwrap_or(0)
                );
            } else {
                println!("✗ {}", plan.refusal(&sel.model.id, &sel.profile));
                std::process::exit(2);
            }
        }
    }
    Ok(())
}
