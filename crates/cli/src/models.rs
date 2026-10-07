use crate::util;
use clap::Subcommand;
use llmario_core::ModelFormat;
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
    /// The model library: every downloadable model with what it is for, its size, whether it
    /// fits this computer and whether your installed engines can run it.
    Catalog {
        /// Filter by text (name, publisher, task, repo).
        filter: Option<String>,
        #[arg(long)]
        json: bool,
    },
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
                println!(
                    "No models installed. See `{} model catalog`.",
                    llmario_core::APP_NAME
                );
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
        ModelCmd::Catalog { filter, json } => {
            let (paths, cfg) = util::load_config(&Default::default())?;
            let sup = util::supervisor(cfg, paths).await?;
            let mut views = llmario_runtime::library::catalog_views(&sup);
            if let Some(f) = filter.map(|f| f.to_lowercase()) {
                views.retain(|v| {
                    [&v.id, &v.name, &v.repo, &v.description]
                        .iter()
                        .any(|s| s.to_lowercase().contains(&f))
                        || v.publisher
                            .as_deref()
                            .unwrap_or("")
                            .to_lowercase()
                            .contains(&f)
                        || v.tasks.iter().any(|t| t.contains(&f))
                });
            }
            if json {
                println!("{}", serde_json::to_string_pretty(&views)?);
                return Ok(());
            }
            let mut family = String::new();
            for v in &views {
                if v.family != family {
                    family = v.family.clone();
                    println!(
                        "\n{} — {} · {} · {} · {}\n  {}",
                        v.name,
                        v.publisher.as_deref().unwrap_or("?"),
                        v.params.as_deref().unwrap_or("?"),
                        v.license,
                        v.tasks.join(", "),
                        v.description
                    );
                }
                let status = if v.installed {
                    "installed".to_string()
                } else if !v.backend_available {
                    format!("{} not installed", v.backend)
                } else if v.supported == Some(false) {
                    format!(
                        "needs newer {} (arch {})",
                        v.backend,
                        v.architecture.as_deref().unwrap_or("?")
                    )
                } else if !v.fits {
                    format!("too large (needs ~{})", fmt_bytes(v.needs_bytes))
                } else {
                    format!("fits (needs ~{})", fmt_bytes(v.needs_bytes))
                };
                println!(
                    "  {} {:<40} {:<9} {:>9}  {}",
                    if v.recommended { "★" } else { " " },
                    v.id,
                    v.backend.to_string(),
                    v.approx_bytes.map(fmt_bytes).unwrap_or_default(),
                    status
                );
            }
            println!("\n★ = recommended for this computer. Pull by id, or by family for the best variant: `{} model pull qwen3.5-9b`", llmario_core::APP_NAME);
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
                            "'{name}' is not in the catalog (see `{} model catalog`)",
                            llmario_core::APP_NAME
                        );
                    }
                    // Use the library's recommendation: engine installed and able to load the
                    // architecture, fits in memory, preferred engine for this hardware.
                    let (paths2, cfg) = util::load_config(&Default::default())?;
                    let sup = util::supervisor(cfg, paths2).await?;
                    let views = llmario_runtime::library::catalog_views(&sup);
                    let pick = views.iter().find(|v| v.family == name && v.recommended);
                    let Some(pick) = pick else {
                        let why: Vec<String> = views
                            .iter()
                            .filter(|v| v.family == name)
                            .map(|v| {
                                let reason = if !v.backend_available {
                                    format!("{} not installed", v.backend)
                                } else if v.supported == Some(false) {
                                    format!("needs a newer {}", v.backend)
                                } else {
                                    format!("too large (needs ~{})", fmt_bytes(v.needs_bytes))
                                };
                                format!("{}: {reason}", v.id)
                            })
                            .collect();
                        anyhow::bail!("no variant of '{name}' can run here: {}", why.join("; "));
                    };
                    let chosen = variants
                        .into_iter()
                        .find(|v| v.id == pick.id)
                        .expect("view comes from the same catalog");
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
