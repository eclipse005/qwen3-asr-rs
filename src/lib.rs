mod config;
mod decoder;
mod encoder;
mod error;
#[cfg(feature = "hub")]
pub(crate) mod hub;
mod inference;
mod mel;
mod streaming;
#[cfg(feature = "cuda")]
mod cudarc_engine;
#[cfg(feature = "cuda")]
mod gpu_audio_encoder;
#[cfg(feature = "cpu")]
mod cpu_engine;

pub use error::{AsrError, Result};
pub use inference::{AsrInference, TranscribeOptions, TranscribeResult};
pub use mel::load_audio_wav;
pub use streaming::{StreamingOptions, StreamingState};

// ─── Backend / Device cfg 分支 ─────────────────────────────────────
// 一套业务代码，编译时选择 GPU 后端。

#[cfg(feature = "cuda")]
pub type Backend = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime, half::f16, i32, u8>;
#[cfg(feature = "cuda")]
pub type Device = burn::backend::cuda::CudaDevice;
#[cfg(feature = "cuda")]
pub fn best_device() -> Device { Device::new(0) }

#[cfg(feature = "rocm")]
pub type Backend = burn_cubecl::CubeBackend<cubecl::hip::HipRuntime, half::f16, i32, u8>;
#[cfg(feature = "rocm")]
pub type Device = burn::backend::rocm::RocmDevice;
#[cfg(feature = "rocm")]
pub fn best_device() -> Device { Device::new(0) }

#[cfg(feature = "metal")]
pub type Backend = burn_cubecl::CubeBackend<cubecl::metal::MetalRuntime, half::f16, i32, u8>;
#[cfg(feature = "metal")]
pub type Device = burn::backend::metal::MetalDevice;
#[cfg(feature = "metal")]
pub fn best_device() -> Device { Device::default() }

#[cfg(feature = "vulkan")]
pub type Backend = burn::backend::wgpu::Wgpu<half::f16, i32>;
#[cfg(feature = "vulkan")]
pub type Device = burn::backend::wgpu::WgpuDevice;
#[cfg(feature = "vulkan")]
pub fn best_device() -> Device { Device::default() }

#[cfg(feature = "cpu")]
pub type Backend = burn::backend::Flex;
#[cfg(feature = "cpu")]
pub type Device = burn::backend::flex::FlexDevice;
#[cfg(feature = "cpu")]
pub fn best_device() -> Device { Device::default() }
