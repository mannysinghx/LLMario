use std::path::{Path, PathBuf};

/// Filesystem layout under the llmario home (default `~/.llmario`, override `LLMARIO_HOME`).
#[derive(Clone, Debug)]
pub struct Paths {
    pub home: PathBuf,
}

impl Paths {
    pub fn from_env() -> anyhow::Result<Self> {
        let var = format!("{}_HOME", crate::ENV_PREFIX);
        if let Some(h) = std::env::var_os(&var).filter(|v| !v.is_empty()) {
            return Ok(Self::at(PathBuf::from(h)));
        }
        let home = dirs::home_dir()
            .ok_or_else(|| anyhow::anyhow!("cannot find home directory; set {var}"))?;
        Ok(Self::at(home.join(format!(".{}", crate::APP_NAME))))
    }

    pub fn at(home: impl Into<PathBuf>) -> Self {
        Self { home: home.into() }
    }

    pub fn config_file(&self) -> PathBuf {
        self.home.join("config.toml")
    }
    pub fn registry_file(&self) -> PathBuf {
        self.home.join("registry.toml")
    }
    /// Managed model storage (downloaded by `model pull`).
    pub fn models_dir(&self) -> PathBuf {
        self.home.join("models")
    }
    pub fn logs_dir(&self) -> PathBuf {
        self.home.join("logs")
    }
    /// Records of engine processes we launched (for orphan detection).
    pub fn run_dir(&self) -> PathBuf {
        self.home.join("run")
    }
    pub fn bench_dir(&self) -> PathBuf {
        self.home.join("bench")
    }
    pub fn tmp_dir(&self) -> PathBuf {
        self.home.join("tmp")
    }

    pub fn ensure(&self) -> anyhow::Result<()> {
        for d in [
            self.home.clone(),
            self.models_dir(),
            self.logs_dir(),
            self.run_dir(),
            self.bench_dir(),
            self.tmp_dir(),
        ] {
            std::fs::create_dir_all(&d)
                .map_err(|e| anyhow::anyhow!("creating {}: {e}", d.display()))?;
        }
        Ok(())
    }

    /// Replace the home prefix with `~` style placeholder for routine logs (privacy).
    pub fn redact(&self, p: &Path) -> String {
        match p.strip_prefix(&self.home) {
            Ok(rest) => format!("${}_HOME/{}", crate::ENV_PREFIX, rest.display()),
            Err(_) => match dirs::home_dir()
                .and_then(|h| p.strip_prefix(h).ok().map(|r| r.to_path_buf()))
            {
                Some(rest) => format!("~/{}", rest.display()),
                None => p.display().to_string(),
            },
        }
    }
}
