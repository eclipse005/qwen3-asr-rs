mod backend;
mod config;
mod error;
#[cfg(feature = "hub")]
pub(crate) mod hub;
mod inference;
mod mel;
pub(crate) mod raw_tensor;
mod mrope;
mod prompt;
mod weights;
#[cfg(feature = "cuda")]
mod cudarc_engine;
#[cfg(feature = "cuda")]
mod gpu_audio_encoder;
mod cpu_engine;
mod cpu_audio_encoder;

pub use backend::Backend;
pub use error::{AsrError, Result};
pub use inference::{AsrInference, TranscribeOptions, TranscribeResult, StreamToken};
pub use mel::load_audio_wav;
