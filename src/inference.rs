//! Public ASR API and backend dispatch.
//!
//! Heavy encode/generate logic lives in [`crate::pipeline`] (CPU / CUDA
//! backends).  This module owns load, mel, streaming hooks, and a single
//! generate path that calls into the active [`AsrPipeline`].

use anyhow::Context;
use log::{debug, info};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use crate::backend::{Backend, ResolvedBackend};
use crate::config::AsrConfig;
use crate::error::AsrError;
use crate::mel::{load_audio_wav, MelExtractor, HOP_LENGTH, MEL_SAMPLE_RATE, N_FFT};
use crate::pipeline::{AsrPipeline, CpuPipeline};
use crate::prompt;
use crate::raw_tensor::RawTensor;

#[cfg(feature = "cuda")]
use crate::pipeline::CudaPipeline;

// ── Public API types ──────────────────────────────────────────────

#[non_exhaustive]
pub struct TranscribeOptions {
    pub language: Option<String>,
    pub max_new_tokens: usize,
}

impl Default for TranscribeOptions {
    /// Default `max_new_tokens` matches common product / transformers use (2048).
    /// Generation must stop on EOS like HF `generate`; do not rely on decode-time
    /// n-gram early-stop or lowered token ceilings as the primary control.
    fn default() -> Self {
        Self {
            language: None,
            max_new_tokens: 2048,
        }
    }
}

impl TranscribeOptions {
    pub fn with_max_new_tokens(mut self, n: usize) -> Self {
        self.max_new_tokens = n;
        self
    }
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }
}

#[non_exhaustive]
pub struct TranscribeResult {
    pub text: String,
    pub language: String,
    pub raw_output: String,
}

/// Emitted once per generated token during streaming transcription.
#[non_exhaustive]
pub struct StreamToken {
    /// The token ID just produced by the decoder.
    pub token_id: u32,
    /// Raw decoded text of all generated tokens so far (incremental).
    pub text_so_far: String,
}

// ── Engine enum (CPU / CUDA are equal backends) ───────────────────

/// Active compute backend.  Each variant owns a full pipeline (encoder + decoder).
pub(crate) enum Engine {
    Cpu(CpuPipeline),
    #[cfg(feature = "cuda")]
    Cuda(CudaPipeline),
}

impl Engine {
    fn pipeline(&self) -> &dyn AsrPipeline {
        match self {
            Engine::Cpu(p) => p,
            #[cfg(feature = "cuda")]
            Engine::Cuda(p) => p,
        }
    }
}

pub(crate) struct AsrInferenceInner {
    pub(crate) engine: Engine,
    pub(crate) mel_extractor: MelExtractor,
    pub(crate) tokenizer: tokenizers::Tokenizer,
    pub(crate) config: AsrConfig,
}

unsafe impl Send for AsrInferenceInner {}

pub struct AsrInference {
    pub(crate) inner: Mutex<AsrInferenceInner>,
}

// ── Public entry points ───────────────────────────────────────────

impl AsrInference {
    pub fn load(model_dir: &Path, backend: Backend) -> crate::Result<Self> {
        let t0 = std::time::Instant::now();
        info!("Loading config...");
        let config = AsrConfig::from_file(&model_dir.join("config.json"))
            .context("load config")
            .map_err(AsrError::ModelLoad)?;

        info!("Loading weights...");
        let t_weights = std::time::Instant::now();
        let weight_data = crate::weights::load_weights(model_dir)
            .context("load weights")
            .map_err(AsrError::ModelLoad)?;
        info!(
            "Loaded {} weight tensors in {:.1}ms",
            weight_data.len(),
            t_weights.elapsed().as_secs_f64() * 1000.0
        );

        info!("Loading tokenizer...");
        let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer load failed: {}", e))
            .map_err(AsrError::ModelLoad)?;

        info!(
            "Total load+build: {:.1}ms",
            t0.elapsed().as_secs_f64() * 1000.0
        );
        Self::build_engine(config, weight_data, tokenizer, backend).map_err(AsrError::ModelLoad)
    }

    pub fn new(model_dir: &Path) -> crate::Result<Self> {
        Self::load(model_dir, Backend::Auto)
    }

    #[cfg(feature = "hub")]
    pub fn from_pretrained(
        model_id: &str,
        cache_dir: &Path,
        backend: Backend,
    ) -> crate::Result<Self> {
        let model_dir =
            crate::hub::ensure_model_cached(model_id, cache_dir).map_err(AsrError::ModelLoad)?;
        Self::load(&model_dir, backend)
    }

    fn build_engine(
        config: AsrConfig,
        weights: HashMap<String, RawTensor>,
        tokenizer: tokenizers::Tokenizer,
        backend: Backend,
    ) -> anyhow::Result<Self> {
        let mel_extractor = MelExtractor::new(
            N_FFT,
            HOP_LENGTH,
            config.thinker_config.audio_config.num_mel_bins,
            MEL_SAMPLE_RATE,
        );
        let resolved = backend.resolve()?;

        let engine = match resolved {
            ResolvedBackend::Cpu => {
                info!("Loading text decoder (CPU gemm+rayon engine)...");
                let t1 = std::time::Instant::now();
                let decoder = crate::cpu_engine::CpuTextDecoder::load(
                    &weights,
                    "thinker.model",
                    &config.thinker_config.text_config,
                )
                .context("load CPU text decoder")?;
                info!(
                    "CPU decoder loaded in {:.1}ms",
                    t1.elapsed().as_secs_f64() * 1000.0
                );
                info!("Loading audio encoder (CPU f32 engine)...");
                let t2 = std::time::Instant::now();
                let audio_encoder = crate::cpu_audio_encoder::CpuAudioEncoder::load(
                    &weights,
                    "thinker.audio_tower",
                    &config.thinker_config.audio_config,
                )
                .context("load CPU audio encoder")?;
                info!(
                    "CPU audio encoder loaded in {:.1}ms",
                    t2.elapsed().as_secs_f64() * 1000.0
                );
                Engine::Cpu(CpuPipeline {
                    decoder,
                    audio_encoder,
                })
            }
            #[cfg(feature = "cuda")]
            ResolvedBackend::Cuda(cuda) => {
                info!("Loading text decoder (GPU-resident cuBLAS+kernels)...");
                let t1 = std::time::Instant::now();
                let decoder = crate::cudarc_engine::GpuTextDecoder::load_with(
                    cuda.clone(),
                    &weights,
                    "thinker.model",
                    &config.thinker_config.text_config,
                )
                .context("load GPU text decoder")?;
                info!(
                    "GPU decoder loaded in {:.1}ms",
                    t1.elapsed().as_secs_f64() * 1000.0
                );
                info!("Loading audio encoder transformer (cuBLAS+kernels)...");
                let t2 = std::time::Instant::now();
                let audio_encoder = crate::gpu_audio_encoder::GpuAudioEncoder::load(
                    cuda.clone(),
                    &weights,
                    "thinker.audio_tower",
                    &config.thinker_config.audio_config,
                )
                .context("load GPU audio encoder")?;
                info!(
                    "GPU audio encoder loaded in {:.1}ms",
                    t2.elapsed().as_secs_f64() * 1000.0
                );
                Engine::Cuda(CudaPipeline {
                    cuda,
                    decoder,
                    audio_encoder,
                })
            }
        };

        info!("Active backend: {}", engine.pipeline().tag());

        Ok(AsrInference {
            inner: Mutex::new(AsrInferenceInner {
                engine,
                mel_extractor,
                tokenizer,
                config,
            }),
        })
    }

    // ── Non-streaming API ─────────────────────────────────────────

    pub fn transcribe(
        &self,
        audio_path: &str,
        options: TranscribeOptions,
    ) -> crate::Result<TranscribeResult> {
        info!("Loading audio: {}", audio_path);
        let samples = load_audio_wav(audio_path, MEL_SAMPLE_RATE)?;
        info!("Audio: {} samples @ {}Hz", samples.len(), MEL_SAMPLE_RATE);
        let inner = self
            .inner
            .lock()
            .map_err(|_| AsrError::Inference(anyhow::anyhow!("mutex poisoned")))?;
        inner
            .run_inference(&samples, &options)
            .map_err(AsrError::Inference)
    }

    pub fn transcribe_samples(
        &self,
        samples: &[f32],
        options: TranscribeOptions,
    ) -> crate::Result<TranscribeResult> {
        let inner = self
            .inner
            .lock()
            .map_err(|_| AsrError::Inference(anyhow::anyhow!("mutex poisoned")))?;
        inner
            .run_inference(samples, &options)
            .map_err(AsrError::Inference)
    }

    // ── Streaming API ──────────────────────────────────────────────

    /// Streaming variant of `transcribe`. `on_token` is called for each
    /// generated token with the incremental decoded text so far.
    /// Returns the final `TranscribeResult` when done.
    pub fn transcribe_streaming<F>(
        &self,
        audio_path: &str,
        options: TranscribeOptions,
        on_token: F,
    ) -> crate::Result<TranscribeResult>
    where
        F: FnMut(StreamToken),
    {
        info!("Loading audio: {}", audio_path);
        let samples = load_audio_wav(audio_path, MEL_SAMPLE_RATE)?;
        info!("Audio: {} samples @ {}Hz", samples.len(), MEL_SAMPLE_RATE);
        self.transcribe_samples_streaming(&samples, options, on_token)
    }

    /// Streaming variant of `transcribe_samples`.
    pub fn transcribe_samples_streaming<F>(
        &self,
        samples: &[f32],
        options: TranscribeOptions,
        mut on_token: F,
    ) -> crate::Result<TranscribeResult>
    where
        F: FnMut(StreamToken),
    {
        let inner = self
            .inner
            .lock()
            .map_err(|_| AsrError::Inference(anyhow::anyhow!("mutex poisoned")))?;
        inner
            .run_inference_streaming(samples, &options, &mut on_token)
            .map_err(AsrError::Inference)
    }

    /// Create a streaming session that accepts audio incrementally.
    /// Audio is encoded chunk-by-chunk during `push_samples()`.
    /// Call `flush()` or `flush_streaming()` to finalize and get text.
    pub fn create_streaming_session(
        &self,
        options: TranscribeOptions,
    ) -> crate::Result<crate::streaming::AsrStreamingSession<'_>> {
        let inner = self
            .inner
            .lock()
            .map_err(|_| AsrError::Inference(anyhow::anyhow!("mutex poisoned")))?;
        Ok(crate::streaming::AsrStreamingSession::new(inner, options))
    }
}

// ── Internal dispatch ─────────────────────────────────────────────

impl AsrInferenceInner {
    fn run_inference(
        &self,
        samples: &[f32],
        options: &TranscribeOptions,
    ) -> anyhow::Result<TranscribeResult> {
        let audio_embeds = self.encode_audio(samples)?;
        let generated_ids = self.generate(
            &audio_embeds,
            options.language.as_deref(),
            None,
            options.max_new_tokens,
        )?;
        prompt::decode_result(
            &self.tokenizer,
            &generated_ids,
            options.language.as_deref(),
        )
    }

    fn run_inference_streaming<F>(
        &self,
        samples: &[f32],
        options: &TranscribeOptions,
        on_token: &mut F,
    ) -> anyhow::Result<TranscribeResult>
    where
        F: FnMut(StreamToken),
    {
        let audio_embeds = self.encode_audio(samples)?;
        let tokenizer = &self.tokenizer;
        let mut all_ids: Vec<u32> = Vec::new();
        let mut streaming_cb = |token_id: u32| {
            all_ids.push(token_id);
            let text = tokenizer.decode(&all_ids, true).unwrap_or_default();
            on_token(StreamToken {
                token_id,
                text_so_far: text,
            });
        };
        let final_ids = self.generate_with_callback(
            &audio_embeds,
            options.language.as_deref(),
            None,
            options.max_new_tokens,
            &mut streaming_cb,
        )?;
        prompt::decode_result(&self.tokenizer, &final_ids, options.language.as_deref())
    }

    pub(crate) fn encode_audio(&self, samples: &[f32]) -> anyhow::Result<Vec<f32>> {
        let t_mel = std::time::Instant::now();
        let (mel_data, n_mels, n_frames) = self.mel_extractor.extract(samples)?;
        debug!("Mel: {}×{} frames", n_mels, n_frames);

        let n_window = self.config.thinker_config.audio_config.n_window;
        let pipeline = self.engine.pipeline();

        let t_enc = std::time::Instant::now();
        #[cfg(feature = "cuda")]
        let log_cuda_timing = matches!(&self.engine, Engine::Cuda(_));
        #[cfg(not(feature = "cuda"))]
        let log_cuda_timing = false;

        if log_cuda_timing {
            info!(
                "CUDA mel: {:.2}ms ({}x{} frames)",
                t_mel.elapsed().as_secs_f64() * 1000.0,
                n_mels,
                n_frames
            );
        }

        let out = pipeline.encode_from_mel(&mel_data, n_mels, n_frames, n_window)?;

        if log_cuda_timing {
            // encode_from_mel ends with a D2H → host timing is accurate.
            info!(
                "CUDA audio_enc: {:.2}ms",
                t_enc.elapsed().as_secs_f64() * 1000.0
            );
        }

        let output_dim = self.config.thinker_config.audio_config.output_dim;
        let n_tokens = out.len() / output_dim;
        info!("Audio tokens: {}", n_tokens);
        Ok(out)
    }

    /// Non-streaming generate (no-op callback).
    pub(crate) fn generate(
        &self,
        audio_embeds: &[f32],
        language: Option<&str>,
        prefix_text: Option<&str>,
        max_new_tokens: usize,
    ) -> anyhow::Result<Vec<u32>> {
        self.generate_with_callback(
            audio_embeds,
            language,
            prefix_text,
            max_new_tokens,
            &mut |_| {},
        )
    }

    /// Core generate with per-token callback. Single entry for all backends.
    pub(crate) fn generate_with_callback(
        &self,
        audio_embeds: &[f32],
        language: Option<&str>,
        prefix_text: Option<&str>,
        max_new_tokens: usize,
        on_token: &mut dyn FnMut(u32),
    ) -> anyhow::Result<Vec<u32>> {
        self.engine.pipeline().generate(
            &self.tokenizer,
            &self.config,
            audio_embeds,
            language,
            prefix_text,
            max_new_tokens,
            on_token,
        )
    }
}
