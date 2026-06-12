//! Backend dispatch — minimal `enum` selection between CUDA and CPU engines.
//!
//! `Backend` is a tag + a CUDA handle.  Callers match on it and forward to the
//! corresponding engine.  The heavy lifting lives in `cudarc_engine` (GPU) and
//! `cpu_engine` (CPU); this module just owns the dispatch.

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use crate::cudarc_engine::CudaState;

/// The chosen compute backend.  Construct via [`Backend::best`] unless the
/// caller wants to force a specific device.
#[cfg(feature = "cuda")]
pub enum Backend {
    Cuda(Arc<CudaState>),
    Cpu,
}

#[cfg(not(feature = "cuda"))]
pub enum Backend {
    Cpu,
}

impl Backend {
    /// Pick the "best" available backend.  With `cuda` feature, prefers CUDA
    /// (GPU 0); otherwise falls back to CPU.
    #[cfg(feature = "cuda")]
    pub fn best() -> anyhow::Result<Self> {
        match CudaState::new(0) {
            Ok(state) => Ok(Backend::Cuda(Arc::new(state))),
            Err(e) => {
                log::warn!("CUDA init failed ({}); falling back to CPU", e);
                Ok(Backend::Cpu)
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn best() -> anyhow::Result<Self> {
        Ok(Backend::Cpu)
    }

    /// Short human label — useful for logs.
    pub fn tag(&self) -> &'static str {
        match self {
            #[cfg(feature = "cuda")]
            Backend::Cuda(_) => "cuda:0",
            Backend::Cpu => "cpu",
        }
    }
}
