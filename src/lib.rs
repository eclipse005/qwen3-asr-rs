mod backend;
mod config;
mod error;
#[cfg(feature = "hub")]
pub(crate) mod hub;
mod inference;
mod mel;
pub(crate) mod raw_tensor;
#[cfg(feature = "cuda")]
mod cudarc_engine;
#[cfg(feature = "cuda")]
mod gpu_audio_encoder;
#[cfg(feature = "cpu")]
mod cpu_engine;

pub use backend::Backend;
pub use error::{AsrError, Result};
pub use inference::{AsrInference, TranscribeOptions, TranscribeResult};
pub use mel::load_audio_wav;
