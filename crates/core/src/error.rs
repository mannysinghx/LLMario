/// Error taxonomy shared by the supervisor, API and CLI. Every variant maps to a stable
/// machine-readable code and an HTTP status, and carries an actionable message.
#[derive(thiserror::Error, Debug)]
pub enum RuntimeError {
    #[error(
        "model '{0}' is not installed. See `llmario model list`, or `llmario model pull <name>`."
    )]
    ModelNotFound(String),
    #[error("{0}")]
    InsufficientMemory(String),
    #[error("{0}")]
    BackendUnavailable(String),
    #[error("{0}")]
    Unsupported(String),
    #[error("{0}")]
    InvalidRequest(String),
    #[error("{0}")]
    ContextOverflow(String),
    #[error("engine failed to start: {0}")]
    EngineStart(String),
    #[error("engine crashed: {0}")]
    EngineCrashed(String),
    #[error("{0}")]
    Busy(String),
    #[error("{0}")]
    Timeout(String),
    #[error("{0}")]
    Unauthorized(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl RuntimeError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::ModelNotFound(_) => "model_not_found",
            Self::InsufficientMemory(_) => "insufficient_memory",
            Self::BackendUnavailable(_) => "backend_unavailable",
            Self::Unsupported(_) => "unsupported_feature",
            Self::InvalidRequest(_) => "invalid_request",
            Self::ContextOverflow(_) => "context_length_exceeded",
            Self::EngineStart(_) => "engine_start_failed",
            Self::EngineCrashed(_) => "engine_crashed",
            Self::Busy(_) => "server_busy",
            Self::Timeout(_) => "timeout",
            Self::Unauthorized(_) => "invalid_api_key",
            Self::Other(_) => "internal_error",
        }
    }

    /// OpenAI-style error `type` field.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::ModelNotFound(_)
            | Self::InvalidRequest(_)
            | Self::ContextOverflow(_)
            | Self::Unsupported(_) => "invalid_request_error",
            Self::Unauthorized(_) => "authentication_error",
            _ => "server_error",
        }
    }

    pub fn http_status(&self) -> u16 {
        match self {
            Self::ModelNotFound(_) => 404,
            Self::InvalidRequest(_) | Self::ContextOverflow(_) | Self::Unsupported(_) => 400,
            Self::Unauthorized(_) => 401,
            Self::InsufficientMemory(_) => 507,
            Self::BackendUnavailable(_) | Self::Busy(_) => 503,
            Self::EngineStart(_) | Self::EngineCrashed(_) => 502,
            Self::Timeout(_) => 504,
            Self::Other(_) => 500,
        }
    }
}
