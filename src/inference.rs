use anyhow::Context;
use log::{debug, info};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use crate::backend::{Backend, ResolvedBackend};
use crate::config::AsrConfig;
use crate::cpu_engine::CpuTextDecoder;
use crate::cpu_audio_encoder::CpuAudioEncoder;
use crate::error::AsrError;
use crate::mel::{load_audio_wav, MelExtractor, MEL_SAMPLE_RATE, N_FFT, HOP_LENGTH};
use crate::raw_tensor::RawTensor;
use crate::prompt;

#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use crate::cudarc_engine::{GpuTextDecoder, CudaState,
    compute_mrope_cos_sin as cublas_compute_mrope_cos_sin};
#[cfg(feature = "cuda")]
use crate::gpu_audio_encoder::GpuAudioEncoder;

// ── Public API types ──────────────────────────────────────────────

#[non_exhaustive]
pub struct TranscribeOptions {
    pub language: Option<String>,
    pub max_new_tokens: usize,
}

impl Default for TranscribeOptions {
    fn default() -> Self { Self { language: None, max_new_tokens: 512 } }
}

impl TranscribeOptions {
    pub fn with_max_new_tokens(mut self, n: usize) -> Self { self.max_new_tokens = n; self }
    pub fn with_language(mut self, lang: impl Into<String>) -> Self { self.language = Some(lang.into()); self }
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

// ── Engine enum ───────────────────────────────────────────────────

pub(crate) enum Engine {
    Cpu {
        decoder: CpuTextDecoder,
        audio_encoder: CpuAudioEncoder,
    },
    #[cfg(feature = "cuda")]
    Cuda {
        cuda: Arc<CudaState>,
        decoder: GpuTextDecoder,
        audio_encoder: GpuAudioEncoder,
    },
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
            .context("load config").map_err(AsrError::ModelLoad)?;

        info!("Loading weights...");
        let t_weights = std::time::Instant::now();
        let weight_data = crate::weights::load_weights(model_dir)
            .context("load weights").map_err(AsrError::ModelLoad)?;
        info!("Loaded {} weight tensors in {:.1}ms", weight_data.len(), t_weights.elapsed().as_secs_f64() * 1000.0);

        info!("Loading tokenizer...");
        let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer load failed: {}", e))
            .map_err(AsrError::ModelLoad)?;

        info!("Total load+build: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
        Self::build_engine(config, weight_data, tokenizer, backend).map_err(AsrError::ModelLoad)
    }

    pub fn new(model_dir: &Path) -> crate::Result<Self> {
        Self::load(model_dir, Backend::Auto)
    }

    #[cfg(feature = "hub")]
    pub fn from_pretrained(
        model_id: &str, cache_dir: &Path, backend: Backend,
    ) -> crate::Result<Self> {
        let model_dir = crate::hub::ensure_model_cached(model_id, cache_dir)
            .map_err(AsrError::ModelLoad)?;
        Self::load(&model_dir, backend)
    }

    fn build_engine(
        config: AsrConfig, weights: HashMap<String, RawTensor>,
        tokenizer: tokenizers::Tokenizer, backend: Backend,
    ) -> anyhow::Result<Self> {
        let mel_extractor = MelExtractor::new(
            N_FFT, HOP_LENGTH,
            config.thinker_config.audio_config.num_mel_bins,
            MEL_SAMPLE_RATE,
        );
        let resolved = backend.resolve()?;

        let engine = match resolved {
            ResolvedBackend::Cpu => {
                info!("Loading text decoder (CPU gemm+rayon engine)...");
                let t1 = std::time::Instant::now();
                let decoder = CpuTextDecoder::load(
                    &weights, "thinker.model", &config.thinker_config.text_config,
                ).context("load CPU text decoder")?;
                info!("CPU decoder loaded in {:.1}ms", t1.elapsed().as_secs_f64() * 1000.0);
                info!("Loading audio encoder (CPU f32 engine)...");
                let t2 = std::time::Instant::now();
                let audio_encoder = CpuAudioEncoder::load(
                    &weights, "thinker.audio_tower", &config.thinker_config.audio_config,
                ).context("load CPU audio encoder")?;
                info!("CPU audio encoder loaded in {:.1}ms", t2.elapsed().as_secs_f64() * 1000.0);
                Engine::Cpu { decoder, audio_encoder }
            }
            #[cfg(feature = "cuda")]
            ResolvedBackend::Cuda(cuda) => {
                info!("Loading text decoder (GPU-resident cuBLAS+kernels)...");
                let t1 = std::time::Instant::now();
                let decoder = GpuTextDecoder::load_with(
                    cuda.clone(), &weights, "thinker.model", &config.thinker_config.text_config,
                ).context("load GPU text decoder")?;
                info!("GPU decoder loaded in {:.1}ms", t1.elapsed().as_secs_f64() * 1000.0);
                info!("Loading audio encoder transformer (cuBLAS+kernels)...");
                let t2 = std::time::Instant::now();
                let audio_encoder = GpuAudioEncoder::load(
                    cuda.clone(), &weights, "thinker.audio_tower", &config.thinker_config.audio_config,
                ).context("load GPU audio encoder")?;
                info!("GPU audio encoder loaded in {:.1}ms", t2.elapsed().as_secs_f64() * 1000.0);
                Engine::Cuda { cuda, decoder, audio_encoder }
            }
        };

        Ok(AsrInference {
            inner: Mutex::new(AsrInferenceInner {
                engine, mel_extractor, tokenizer, config,
            }),
        })
    }

    // ── Non-streaming API ─────────────────────────────────────────

    pub fn transcribe(&self, audio_path: &str, options: TranscribeOptions) -> crate::Result<TranscribeResult> {
        info!("Loading audio: {}", audio_path);
        let samples = load_audio_wav(audio_path, MEL_SAMPLE_RATE)?;
        info!("Audio: {} samples @ {}Hz", samples.len(), MEL_SAMPLE_RATE);
        let inner = self.inner.lock().map_err(|_| AsrError::Inference(anyhow::anyhow!("mutex poisoned")))?;
        inner.run_inference(&samples, &options).map_err(AsrError::Inference)
    }

    pub fn transcribe_samples(&self, samples: &[f32], options: TranscribeOptions) -> crate::Result<TranscribeResult> {
        let inner = self.inner.lock().map_err(|_| AsrError::Inference(anyhow::anyhow!("mutex poisoned")))?;
        inner.run_inference(samples, &options).map_err(AsrError::Inference)
    }

    // ── Streaming API ──────────────────────────────────────────────

    /// Streaming variant of `transcribe`. `on_token` is called for each
    /// generated token with the incremental decoded text so far.
    /// Returns the final `TranscribeResult` when done.
    pub fn transcribe_streaming<F>(
        &self, audio_path: &str, options: TranscribeOptions, on_token: F,
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
        &self, samples: &[f32], options: TranscribeOptions, mut on_token: F,
    ) -> crate::Result<TranscribeResult>
    where
        F: FnMut(StreamToken),
    {
        let inner = self.inner.lock().map_err(|_| AsrError::Inference(anyhow::anyhow!("mutex poisoned")))?;
        inner.run_inference_streaming(samples, &options, &mut on_token).map_err(AsrError::Inference)
    }

    /// Create a streaming session that accepts audio incrementally.
    /// Audio is encoded chunk-by-chunk during `push_samples()`.
    /// Call `flush()` or `flush_streaming()` to finalize and get text.
    pub fn create_streaming_session(
        &self, options: TranscribeOptions,
    ) -> crate::Result<crate::streaming::AsrStreamingSession<'_>> {
        let inner = self.inner.lock().map_err(|_| AsrError::Inference(anyhow::anyhow!("mutex poisoned")))?;
        Ok(crate::streaming::AsrStreamingSession::new(inner, options))
    }
}

// ── Internal dispatch ─────────────────────────────────────────────

impl AsrInferenceInner {
    fn run_inference(&self, samples: &[f32], options: &TranscribeOptions) -> anyhow::Result<TranscribeResult> {
        let audio_embeds = self.encode_audio(samples)?;
        let generated_ids = self.generate(&audio_embeds, options.language.as_deref(), None, options.max_new_tokens)?;
        prompt::decode_result(&self.tokenizer, &generated_ids, options.language.as_deref())
    }

    fn run_inference_streaming<F>(
        &self, samples: &[f32], options: &TranscribeOptions, on_token: &mut F,
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
            on_token(StreamToken { token_id, text_so_far: text });
        };
        let final_ids = self.generate_with_callback(
            &audio_embeds, options.language.as_deref(), None, options.max_new_tokens,
            &mut streaming_cb,
        )?;
        prompt::decode_result(&self.tokenizer, &final_ids, options.language.as_deref())
    }

    pub(crate) fn encode_audio(&self, samples: &[f32]) -> anyhow::Result<Vec<f32>> {
        match &self.engine {
            Engine::Cpu { audio_encoder, .. } => {
                let (mel_data, n_mels, n_frames) = self.mel_extractor.extract(samples)?;
                debug!("Mel: {}×{} frames", n_mels, n_frames);
                let out = audio_encoder.forward(&mel_data, n_mels, n_frames)?;
                let output_dim = self.config.thinker_config.audio_config.output_dim;
                let n_tokens = out.len() / output_dim;
                info!("Audio tokens: {}", n_tokens);
                Ok(out)
            }
            #[cfg(feature = "cuda")]
            Engine::Cuda { audio_encoder, .. } => {
                let t_mel = std::time::Instant::now();
                let (mel_data, n_mels, n_frames) = self.mel_extractor.extract(samples)?;
                let t_enc = std::time::Instant::now();
                info!("CUDA mel: {:.2}ms ({}x{} frames)", t_mel.elapsed().as_secs_f64() * 1000.0, n_mels, n_frames);
                let n_window = self.config.thinker_config.audio_config.n_window;
                let out = audio_encoder.encode_from_mel(&mel_data, n_mels, n_frames, n_window)?;
                // encode_from_mel ends with a D2H → host timing is accurate (no extra sync needed).
                info!("CUDA audio_enc: {:.2}ms", t_enc.elapsed().as_secs_f64() * 1000.0);
                let output_dim = self.config.thinker_config.audio_config.output_dim;
                let n_tokens = out.len() / output_dim;
                info!("Audio tokens: {}", n_tokens);
                Ok(out)
            }
        }
    }

    /// Non-streaming generate (no-op callback).
    pub(crate) fn generate(
        &self, audio_embeds: &[f32], language: Option<&str>,
        prefix_text: Option<&str>, max_new_tokens: usize,
    ) -> anyhow::Result<Vec<u32>> {
        self.generate_with_callback(audio_embeds, language, prefix_text, max_new_tokens, &mut |_| {})
    }

    /// Core generate with per-token callback. Used by both streaming and non-streaming paths.
    pub(crate) fn generate_with_callback(
        &self, audio_embeds: &[f32], language: Option<&str>,
        prefix_text: Option<&str>, max_new_tokens: usize,
        on_token: &mut dyn FnMut(u32),
    ) -> anyhow::Result<Vec<u32>> {
        match &self.engine {
            Engine::Cpu { decoder, .. } => {
                self.generate_cpu(decoder, audio_embeds, language, prefix_text, max_new_tokens, on_token)
            }
            #[cfg(feature = "cuda")]
            Engine::Cuda { cuda, decoder, .. } => {
                self.generate_cuda(cuda, decoder, audio_embeds, language, prefix_text, max_new_tokens, on_token)
            }
        }
    }

    fn generate_cpu(
        &self, cpu: &CpuTextDecoder,
        audio_embeds: &[f32], language: Option<&str>, prefix_text: Option<&str>,
        max_new_tokens: usize, on_token: &mut dyn FnMut(u32),
    ) -> anyhow::Result<Vec<u32>> {
        use crate::cpu_engine::{CpuTensor, CpuKvCache, compute_mrope_cos_sin as cpu_mrope};
        use crate::prompt::{IM_END_TOKEN_ID, ENDOFTEXT_TOKEN_ID};

        let text_cfg = &self.config.thinker_config.text_config;
        let hidden_size = text_cfg.hidden_size;
        let nat = audio_embeds.len() / hidden_size;
        let (input_ids, audio_start_pos) = prompt::build_prompt(
            &self.tokenizer,
            self.config.thinker_config.audio_start_token_id,
            self.config.thinker_config.audio_token_id,
            self.config.thinker_config.audio_end_token_id,
            nat, language, prefix_text,
        )?;
        let seq_len = input_ids.len();

        let before_ids: Vec<i64> = input_ids[..audio_start_pos].to_vec();
        let after_ids: Vec<i64> = input_ids[audio_start_pos + nat..].to_vec();
        let before_emb = cpu.embed_ids(&before_ids);
        let after_emb = cpu.embed_ids(&after_ids);

        let mut hs_data = Vec::with_capacity(seq_len * hidden_size);
        hs_data.extend_from_slice(&before_emb.data);
        hs_data.extend_from_slice(audio_embeds);
        hs_data.extend_from_slice(&after_emb.data);
        let hidden_states = CpuTensor::new(hs_data, vec![1, seq_len, hidden_size]);

        let total_positions = seq_len + max_new_tokens;
        let all_pos: Vec<i64> = (0..total_positions as i64).collect();
        let full_ids: [Vec<i64>; 3] = [all_pos.clone(), all_pos.clone(), all_pos.clone()];
        let (cos_table, sin_table) = cpu_mrope(
            &full_ids, text_cfg.head_dim, text_cfg.rope_theta,
            &text_cfg.mrope_section(), text_cfg.mrope_interleaved(),
        );

        let mut kv_cache = CpuKvCache::new(
            text_cfg.num_hidden_layers, 1,
            text_cfg.num_key_value_heads, total_positions, text_cfg.head_dim,
        );

        let t_prefill = std::time::Instant::now();
        let logits = cpu.forward(
            hidden_states, &cos_table, &sin_table, &mut kv_cache, 0, true, true,
        );
        let mut current_pos = seq_len;

        let mut generated_ids: Vec<u32> = Vec::new();
        let eos_ids: &[i64] = &[ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID];
        let mut next_token = crate::cpu_engine::argmax(&logits.data) as i64;
        info!("Prefill: {:.2}ms", t_prefill.elapsed().as_secs_f64() * 1000.0);

        let t_decode = std::time::Instant::now();
        for _step in 0..max_new_tokens {
            if eos_ids.contains(&next_token) { break; }
            generated_ids.push(next_token as u32);
            on_token(next_token as u32);

            let ne = cpu.embed_ids(&[next_token]).reshape(vec![1, 1, hidden_size]);
            let sl = cpu.forward(ne, &cos_table, &sin_table, &mut kv_cache, current_pos, false, true);
            next_token = crate::cpu_engine::argmax(&sl.data) as i64;
            current_pos += 1;
        }
        let n_gen = generated_ids.len().max(1);
        info!("Decode: {:.2}ms total ({} tokens, {:.2}ms/tok)",
              t_decode.elapsed().as_secs_f64() * 1000.0, generated_ids.len(),
              t_decode.elapsed().as_secs_f64() * 1000.0 / n_gen as f64);

        info!("Generated {} tokens", generated_ids.len());
        Ok(generated_ids)
    }

    #[cfg(feature = "cuda")]
    fn generate_cuda(
        &self, cuda: &Arc<CudaState>, decoder: &GpuTextDecoder,
        audio_embeds: &[f32], language: Option<&str>, prefix_text: Option<&str>,
        max_new_tokens: usize, on_token: &mut dyn FnMut(u32),
    ) -> anyhow::Result<Vec<u32>> {
        use half::f16;
        use crate::cudarc_engine::{CpuTensor, GpuKvCache, DecodeScratch};
        use crate::prompt::{IM_END_TOKEN_ID, ENDOFTEXT_TOKEN_ID};

        let nat = audio_embeds.len() / self.config.thinker_config.text_config.hidden_size;
        let (input_ids, audio_start_pos) = prompt::build_prompt(
            &self.tokenizer,
            self.config.thinker_config.audio_start_token_id,
            self.config.thinker_config.audio_token_id,
            self.config.thinker_config.audio_end_token_id,
            nat, language, prefix_text,
        )?;
        let seq_len = input_ids.len();
        let hidden_size = self.config.thinker_config.text_config.hidden_size;
        let text_cfg = &self.config.thinker_config.text_config;

        let ae_f16: Vec<f16> = audio_embeds.iter().map(|&v| f16::from_f32(v)).collect();

        let before_ids: Vec<i64> = input_ids[..audio_start_pos].to_vec();
        let after_ids: Vec<i64> = input_ids[audio_start_pos + nat..].to_vec();
        let before_emb = decoder.embed_ids(&before_ids)?;
        let after_emb = decoder.embed_ids(&after_ids)?;
        let before_cpu = cuda.download_tensor(&before_emb)?;
        let after_cpu = cuda.download_tensor(&after_emb)?;

        let mut hs_data = Vec::with_capacity(seq_len * hidden_size);
        hs_data.extend_from_slice(&before_cpu.data);
        hs_data.extend_from_slice(&ae_f16);
        hs_data.extend_from_slice(&after_cpu.data);
        let hidden_cpu = CpuTensor::new(hs_data, vec![1, seq_len, hidden_size]);
        let hidden_states = cuda.upload_tensor(&hidden_cpu)?;

        let total_positions = seq_len + max_new_tokens;
        let all_pos: Vec<i64> = (0..total_positions as i64).collect();
        let full_ids: [Vec<i64>; 3] = [all_pos.clone(), all_pos.clone(), all_pos.clone()];
        let (cos_table_cpu, sin_table_cpu) = cublas_compute_mrope_cos_sin(
            &full_ids, text_cfg.head_dim, text_cfg.rope_theta,
            &text_cfg.mrope_section(), text_cfg.mrope_interleaved(),
        );
        let cos_table = cuda.upload_f16(&cos_table_cpu.data)?;
        let sin_table = cuda.upload_f16(&sin_table_cpu.data)?;

        let mut kv_cache = GpuKvCache::new(
            cuda, text_cfg.num_hidden_layers, 1,
            text_cfg.num_key_value_heads, total_positions, text_cfg.head_dim,
        )?;

        let t_prefill = std::time::Instant::now();
        let logits = decoder.forward(hidden_states, &cos_table, &sin_table, &mut kv_cache, 0, true, true)?;
        let mut current_pos = seq_len;

        let mut generated_ids: Vec<u32> = Vec::new();
        let eos_ids: &[i64] = &[ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID];

        let mut token_buf = cuda.alloc_uninit_i32(1)?;
        cuda.argmax_into(&logits, &mut token_buf, 0)?;

        let mut scratch = DecodeScratch::new(cuda, total_positions, text_cfg)?;
        let mut h_buf = scratch.embed_out.clone();

        // D2H of first token into pinned host memory — the copy syncs, so t_prefill is accurate.
        // Safety: as_ptr() calls event.synchronize() ensuring the copy completes before returning.
        cuda.download_i32_into_pinned(&token_buf, &mut scratch.pinned_token)?;
        let mut next_token = unsafe { *scratch.pinned_token.as_ptr()? } as i64;
        info!("Prefill: {:.2}ms", t_prefill.elapsed().as_secs_f64() * 1000.0);

        let t_decode = std::time::Instant::now();
        loop {
            if eos_ids.contains(&next_token) { break; }
            generated_ids.push(next_token as u32);
            on_token(next_token as u32);
            if generated_ids.len() >= max_new_tokens { break; }

            cuda.embed_id_from_gpu_slot_into(&decoder.embed_table, &token_buf, 0, &mut h_buf)?;
            decoder.forward_decode_scratch(
                &mut h_buf, &cos_table, &sin_table, &mut kv_cache, current_pos,
                &mut token_buf, &mut scratch,
            )?;
            // D2H into pinned memory — avoids implicit full-stream sync of pageable D2H.
            // Safety: as_ptr() calls event.synchronize() ensuring the copy completes before returning.
            cuda.download_i32_into_pinned(&token_buf, &mut scratch.pinned_token)?;
            next_token = unsafe { *scratch.pinned_token.as_ptr()? } as i64;
            current_pos += 1;
        }
        cuda.synchronize()?;
        let n_gen = generated_ids.len().max(1);
        info!("Decode: {:.2}ms total ({} tokens, {:.2}ms/tok)",
              t_decode.elapsed().as_secs_f64() * 1000.0, n_gen,
              t_decode.elapsed().as_secs_f64() * 1000.0 / n_gen as f64);

        info!("Generated {} tokens", generated_ids.len());
        Ok(generated_ids)
    }
}
