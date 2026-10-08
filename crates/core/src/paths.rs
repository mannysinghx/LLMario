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

    /// Home of the other edition ([`crate::SIBLING`]) when it exists on this computer and is
    /// not this edition's own home. For reading only.
    pub fn sibling(&self) -> Option<Paths> {
        let s = crate::SIBLING.as_ref()?;
        let home = sibling_home(s, |k| std::env::var_os(k), dirs::home_dir())?;
        let same = |a: &Path, b: &Path| match (a.canonicalize(), b.canonicalize()) {
            (Ok(a), Ok(b)) => a == b,
            _ => a == b,
        };
        (home.is_dir() && !same(&home, &self.home)).then(|| Self::at(home))
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
    /// Measured speeds on this computer (written by `bench`, read by the speed planner).
    pub fn autotune_file(&self) -> PathBuf {
        self.home.join("autotune.json")
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

/// Where a sibling edition keeps its data: its `<prefix>_HOME` if set, else `~/<home_dir>`.
fn sibling_home(
    s: &crate::Sibling,
    env: impl Fn(&str) -> Option<std::ffi::OsString>,
    user_home: Option<PathBuf>,
) -> Option<PathBuf> {
    env(&format!("{}_HOME", s.env_prefix))
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| user_home.map(|h| h.join(s.home_dir)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: crate::Sibling = crate::Sibling {
        name: "Other",
        home_dir: ".other",
        env_prefix: "OTHER",
    };

    #[test]
    fn sibling_home_prefers_its_env_var() {
        let env = |k: &str| (k == "OTHER_HOME").then(|| "/data/other".into());
        assert_eq!(
            sibling_home(&S, env, Some("/u".into())),
            Some(PathBuf::from("/data/other"))
        );
        let empty = |k: &str| (k == "OTHER_HOME").then(|| "".into());
        assert_eq!(
            sibling_home(&S, empty, Some("/u".into())),
            Some(PathBuf::from("/u/.other"))
        );
        assert_eq!(sibling_home(&S, |_| None, None), None);
    }

    #[test]
    fn sibling_is_never_this_editions_own_home() {
        let d = tempfile::tempdir().unwrap();
        // Whatever the real sibling is, a Paths at the sibling's own home has no sibling.
        if let Some(s) = Paths::at(d.path()).sibling() {
            assert!(s.sibling().is_none(), "{}", s.home.display());
        }
    }
}
