//! Runtime supervisor: owns engine child processes, plans memory, selects the backend for a
//! model, and admits requests. Inference never runs in this process.

pub mod adapter;
pub mod engine;
pub mod memory;
pub mod planner;
pub mod supervisor;

pub use adapter::{BackendStatus, EngineAdapter, LaunchContext, LaunchSpec};
pub use engine::{Engine, EngineInfo};
pub use memory::MemoryPlan;
pub use supervisor::{Lease, Supervisor};
