use anyhow::Context;
#[cfg(not(feature = "cuda"))]
use burn::tensor::{Int, Tensor, TensorData};
#[cfg(feature = "cuda")]
use burn::tensor::{Tensor, TensorData};
#[cfg(feature = "cuda")]
use half::f16;
use log::{debug, info};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use crate::config::AsrConfig;
use crate::encoder::AudioEncoder;
use crate::error::AsrError;
use crate::mel::{load_audio_wav, MelExtractor};
use crate::{Backend, Device};

#[cfg(not(feature = "cuda"))]
use crate::decoder::{compute_mrope_cos_sin, KvCache, TextDecoder};

#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use crate::cudarc_engine::{GpuTextDecoder, GpuKvCache, CpuTensor, CudaState,
    compute_mrope_cos_sin as cublas_compute_mrope_cos_sin};
#[cfg(feature = "cuda")]
use crate::gpu_audio_encoder::GpuAudioEncoder;

pub(crate) const IM_END_TOKEN_ID: i64 = 151645;
pub(crate) const ENDOFTEXT_TOKEN_ID: i64 = 151643;
pub(crate) const ASR_TEXT_SEP_TOKEN_ID: u32 = 151704;
pub(crate) const MEL_SAMPLE_RATE: u32 = 16000;
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;

pub(crate) const TOK_IM_START: i64 = 151644;
pub(crate) const TOK_SYSTEM: i64 = 8948;
pub(crate) const TOK_NEWLINE: i64 = 198;
pub(crate) const TOK_IM_END: i64 = IM_END_TOKEN_ID;
pub(crate) const TOK_USER: i64 = 872;
pub(crate) const TOK_ASSISTANT: i64 = 77091;

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

pub(crate) struct AsrInferenceInner {
    pub(crate) audio_encoder: AudioEncoder<Backend>,
    #[cfg(not(feature = "cuda"))]
    pub(crate) text_decoder: TextDecoder<Backend>,
    #[cfg(feature = "cuda")]
    pub(crate) gpu_decoder: GpuTextDecoder,
    #[cfg(feature = "cuda")]
    pub(crate) gpu_audio_encoder: GpuAudioEncoder,
    #[cfg(feature = "cpu")]
    pub(crate) cpu_decoder: crate::cpu_engine::CpuTextDecoder,
    pub(crate) mel_extractor: MelExtractor,
    pub(crate) tokenizer: tokenizers::Tokenizer,
    pub(crate) config: AsrConfig,
    pub(crate) device: Device,
}

unsafe impl Send for AsrInferenceInner {}

pub struct AsrInference {
    pub(crate) inner: Mutex<AsrInferenceInner>,
}

impl AsrInference {
    pub fn load(model_dir: &Path, device: Device) -> crate::Result<Self> {
        info!("Loading config...");
        let config = AsrConfig::from_file(&model_dir.join("config.json"))
            .context("load config").map_err(AsrError::ModelLoad)?;

        info!("Loading weights...");
        let weight_data = load_weights(model_dir)
            .context("load weights").map_err(AsrError::ModelLoad)?;
        info!("Loaded {} weight tensors", weight_data.len());

        info!("Loading tokenizer...");
        let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer load failed: {}", e))
            .map_err(AsrError::ModelLoad)?;

        info!("Model loaded successfully.");
        Self::build_engine(config, weight_data, tokenizer, device).map_err(AsrError::ModelLoad)
    }

    #[cfg(feature = "hub")]
    pub fn from_pretrained(
        model_id: &str, cache_dir: &Path, device: Device,
    ) -> crate::Result<Self> {
        let model_dir = crate::hub::ensure_model_cached(model_id, cache_dir)
            .map_err(AsrError::ModelLoad)?;
        Self::load(&model_dir, device)
    }

    fn build_engine(config: AsrConfig, weights: HashMap<String, TensorData>, tokenizer: tokenizers::Tokenizer, device: Device) -> anyhow::Result<Self> {
        info!("Loading audio encoder...");
        let audio_encoder = AudioEncoder::load(&weights, "thinker.audio_tower", &config.thinker_config.audio_config, &device)
            .context("load audio encoder")?;

        #[cfg(not(feature = "cuda"))]
        {
            info!("Loading text decoder (burn)...");
            let text_decoder = TextDecoder::load(&weights, "thinker.model", &config.thinker_config.text_config, &device)
                .context("load text decoder")?;
            let mel_extractor = MelExtractor::new(N_FFT, HOP_LENGTH, config.thinker_config.audio_config.num_mel_bins, MEL_SAMPLE_RATE);
            #[cfg(feature = "cpu")]
            let cpu_decoder = {
                info!("Loading text decoder (CPU gemm+rayon engine)...");
                crate::cpu_engine::CpuTextDecoder::load(&weights, "thinker.model", &config.thinker_config.text_config)
                    .context("load CPU text decoder")?
            };
            let inner = AsrInferenceInner {
                audio_encoder, text_decoder, mel_extractor, tokenizer, config, device,
                #[cfg(feature = "cpu")] cpu_decoder,
            };
            Ok(AsrInference { inner: Mutex::new(inner) })
        }

        #[cfg(feature = "cuda")]
        {
            info!("Building shared CUDA state...");
            let cuda = Arc::new(CudaState::new(0).context("init CUDA")?);
            info!("Loading text decoder (GPU-resident cuBLAS+kernels)...");
            let gpu_decoder = GpuTextDecoder::load_with(cuda.clone(), &weights, "thinker.model", &config.thinker_config.text_config)
                .context("load GPU text decoder")?;
            info!("Loading audio encoder transformer (cuBLAS+kernels)...");
            let gpu_audio_encoder = GpuAudioEncoder::load(cuda.clone(), &weights, "thinker.audio_tower", &config.thinker_config.audio_config)
                .context("load GPU audio encoder")?;
            let mel_extractor = MelExtractor::new(N_FFT, HOP_LENGTH, config.thinker_config.audio_config.num_mel_bins, MEL_SAMPLE_RATE);
            let inner = AsrInferenceInner { audio_encoder, gpu_decoder, gpu_audio_encoder, mel_extractor, tokenizer, config, device };
            Ok(AsrInference { inner: Mutex::new(inner) })
        }
    }

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
}

impl AsrInferenceInner {
    fn run_inference(&self, samples: &[f32], options: &TranscribeOptions) -> anyhow::Result<TranscribeResult> {
        let audio_embeds = self.encode_audio(samples)?;
        let generated_ids = self.generate(&audio_embeds, options.language.as_deref(), None, options.max_new_tokens)?;
        self.decode_result(&generated_ids, options.language.as_deref())
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn encode_audio(&self, samples: &[f32]) -> anyhow::Result<Tensor<Backend, 2>> {
        let (mel_data, n_mels, n_frames) = self.mel_extractor.extract(samples)?;
        debug!("Mel: {}×{} frames", n_mels, n_frames);
        let mel = Tensor::<Backend, 2>::from_data(TensorData::new(mel_data, [n_mels, n_frames]), &self.device);
        let audio_embeds = self.audio_encoder.forward(&mel)?;
        info!("Audio tokens: {}", audio_embeds.dims()[0]);
        Ok(audio_embeds)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn encode_audio(&self, samples: &[f32]) -> anyhow::Result<Tensor<Backend, 2>> {
        // Mel extraction on CPU (cheap).
        let (mel_data, n_mels, n_frames) = self.mel_extractor.extract(samples)?;
        debug!("Mel: {}×{} frames", n_mels, n_frames);

        // Chunk the mel into [n_chunks, 1, n_mels, cs], zero-padding the tail chunk.
        let cs = self.config.thinker_config.audio_config.n_window * 2;
        let tpc = feo_inf(cs);
        let nfull = n_frames / cs;
        let tail = n_frames % cs;
        let n_chunks = nfull + if tail > 0 { 1 } else { 0 };

        // Build chunked f16 buffer in (chunk-major, mel-major, frame-minor) layout.
        let mut chunked = vec![f16::ZERO; n_chunks * n_mels * cs];
        let mut chunk_tokens: Vec<usize> = Vec::with_capacity(n_chunks);
        for i in 0..nfull {
            let s = i * cs;
            for m in 0..n_mels {
                let dst_base = (i * n_mels + m) * cs;
                let src_base = m * n_frames + s;
                for j in 0..cs {
                    chunked[dst_base + j] = f16::from_f32(mel_data[src_base + j]);
                }
            }
            chunk_tokens.push(tpc);
        }
        if tail > 0 {
            let s = nfull * cs;
            for m in 0..n_mels {
                let dst_base = (nfull * n_mels + m) * cs;
                let src_base = m * n_frames + s;
                for j in 0..tail {
                    chunked[dst_base + j] = f16::from_f32(mel_data[src_base + j]);
                }
                // Rest is already zero-padded.
            }
            chunk_tokens.push(feo_inf(tail));
        }

        let (out_f16, out_dim) = self.gpu_audio_encoder.run(&chunked, n_chunks, n_mels, cs, &chunk_tokens)?;

        let n_tokens_out = out_f16.len() / out_dim;
        let out_f32: Vec<f32> = out_f16.iter().map(|&v| f32::from(v)).collect();
        let audio_embeds = Tensor::<Backend, 2>::from_data(
            TensorData::new(out_f32, [n_tokens_out, out_dim]), &self.device);
        info!("Audio tokens: {}", n_tokens_out);
        Ok(audio_embeds)
    }

    #[cfg(all(not(feature = "cuda"), not(feature = "cpu")))]
    pub(crate) fn generate(&self, audio_embeds: &Tensor<Backend, 2>, language: Option<&str>, prefix_text: Option<&str>, max_new_tokens: usize) -> anyhow::Result<Vec<u32>> {
        let nat = audio_embeds.dims()[0];
        let (input_ids, audio_start_pos) = self.build_prompt(nat, language, prefix_text)?;
        let seq_len = input_ids.len();

        let before_ids: Vec<i64> = input_ids[..audio_start_pos].to_vec();
        let after_ids: Vec<i64> = input_ids[audio_start_pos + nat..].to_vec();

        let before_t = Tensor::<Backend, 1, Int>::from_data(TensorData::new(before_ids, [audio_start_pos]), &self.device);
        let after_t = Tensor::<Backend, 1, Int>::from_data(TensorData::new(after_ids, [input_ids.len() - audio_start_pos - nat]), &self.device);

        let before_emb = self.text_decoder.embed(&before_t);
        let after_emb = self.text_decoder.embed(&after_t);
        let hidden_states = Tensor::cat(vec![before_emb, audio_embeds.clone(), after_emb], 0).unsqueeze_dim::<3>(0);

        let text_cfg = &self.config.thinker_config.text_config;
        let total_positions = seq_len + max_new_tokens;
        let all_pos: Vec<i64> = (0..total_positions as i64).collect();
        let full_ids: [Vec<i64>; 3] = [all_pos.clone(), all_pos.clone(), all_pos.clone()];
        let (cos_table, sin_table) = compute_mrope_cos_sin(&full_ids, text_cfg.head_dim, text_cfg.rope_theta, &text_cfg.mrope_section(), text_cfg.mrope_interleaved(), &self.device);

        let cos = cos_table.clone().slice([0..seq_len]);
        let sin = sin_table.clone().slice([0..seq_len]);
        let mut kv_cache = KvCache::new(text_cfg.num_hidden_layers, total_positions, text_cfg.num_key_value_heads, text_cfg.head_dim, &self.device);

        let logits = self.text_decoder.forward(&hidden_states, &cos, &sin, &mut kv_cache, true, true);

        let mut generated_ids: Vec<u32> = Vec::new();
        let eos_ids: &[i64] = &[ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID];
        let vocab_size = logits.dims()[2];
        let mut next_logits = logits.reshape([1, vocab_size]);
        let mut current_pos = seq_len;

        for _step_idx in 0..max_new_tokens {
            let next_token = next_logits.argmax(1).into_scalar() as i64;

            if eos_ids.contains(&next_token) { break; }
            generated_ids.push(next_token as u32);

            let nid = Tensor::<_, 1, Int>::from_data(TensorData::new(vec![next_token as i32], [1]), &self.device);
            let ne = self.text_decoder.embed(&nid).unsqueeze_dim::<3>(0);

            let nc = cos_table.clone().slice([current_pos..current_pos + 1]);
            let ns = sin_table.clone().slice([current_pos..current_pos + 1]);

            let sl = self.text_decoder.forward(&ne, &nc, &ns, &mut kv_cache, false, true);
            let vs = sl.dims()[2];
            next_logits = sl.reshape([1, vs]);
            current_pos += 1;
        }

        info!("Generated {} tokens", generated_ids.len());
        Ok(generated_ids)
    }

    /// CPU-feature generate(): runs the entire text decoder through the hand-written
    /// gemm + rayon engine in `cpu_engine.rs`.  The audio encoder still uses burn-flex
    /// (audio is ~5% of total time; the win is the decode loop, where every cuBLAS-equivalent
    /// matmul runs with rayon all-cores instead of the m=1 single-threaded GEMV that
    /// burn-flex's threshold leaves behind).
    #[cfg(feature = "cpu")]
    pub(crate) fn generate(&self, audio_embeds: &Tensor<Backend, 2>, language: Option<&str>, prefix_text: Option<&str>, max_new_tokens: usize) -> anyhow::Result<Vec<u32>> {
        use crate::cpu_engine::{CpuTensor, CpuKvCache, compute_mrope_cos_sin as cpu_mrope};

        let nat = audio_embeds.dims()[0];
        let (input_ids, audio_start_pos) = self.build_prompt(nat, language, prefix_text)?;
        let seq_len = input_ids.len();
        let text_cfg = &self.config.thinker_config.text_config;
        let hidden_size = text_cfg.hidden_size;

        // Build prompt hidden states on CPU: embed prompt tokens, splice in audio embeds.
        let before_ids: Vec<i64> = input_ids[..audio_start_pos].to_vec();
        let after_ids: Vec<i64> = input_ids[audio_start_pos + nat..].to_vec();
        let before_emb = self.cpu_decoder.embed_ids(&before_ids);
        let after_emb = self.cpu_decoder.embed_ids(&after_ids);
        let ae_data = audio_embeds.clone().into_data();
        let ae_f32: Vec<f32> = ae_data.to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("audio_embeds to_vec failed: {:?}", e))?;

        let mut hs_data = Vec::with_capacity(seq_len * hidden_size);
        hs_data.extend_from_slice(&before_emb.data);
        hs_data.extend_from_slice(&ae_f32);
        hs_data.extend_from_slice(&after_emb.data);
        let hidden_states = CpuTensor::new(hs_data, vec![1, seq_len, hidden_size]);

        // MRoPE tables for the full conversation.
        let total_positions = seq_len + max_new_tokens;
        let all_pos: Vec<i64> = (0..total_positions as i64).collect();
        let full_ids: [Vec<i64>; 3] = [all_pos.clone(), all_pos.clone(), all_pos.clone()];
        let (cos_table, sin_table) = cpu_mrope(
            &full_ids, text_cfg.head_dim, text_cfg.rope_theta,
            &text_cfg.mrope_section(), text_cfg.mrope_interleaved(),
        );

        // Pre-allocate KV cache for the full max length.
        let mut kv_cache = CpuKvCache::new(
            text_cfg.num_hidden_layers, 1,
            text_cfg.num_key_value_heads, total_positions, text_cfg.head_dim,
        );

        // Prefill.
        let logits = self.cpu_decoder.forward(
            hidden_states, &cos_table, &sin_table, &mut kv_cache, 0, true, true,
        );
        let mut current_pos = seq_len;

        let mut generated_ids: Vec<u32> = Vec::new();
        let eos_ids: &[i64] = &[ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID];
        let mut next_token = crate::cpu_engine::argmax(&logits.data) as i64;

        // Decode loop.
        for _step in 0..max_new_tokens {
            if eos_ids.contains(&next_token) { break; }
            generated_ids.push(next_token as u32);

            let ne = self.cpu_decoder.embed_ids(&[next_token])
                .reshape(vec![1, 1, hidden_size]);
            let sl = self.cpu_decoder.forward(
                ne, &cos_table, &sin_table, &mut kv_cache, current_pos, false, true,
            );
            next_token = crate::cpu_engine::argmax(&sl.data) as i64;
            current_pos += 1;
        }

        info!("Generated {} tokens", generated_ids.len());
        Ok(generated_ids)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn generate(&self, audio_embeds: &Tensor<Backend, 2>, language: Option<&str>, prefix_text: Option<&str>, max_new_tokens: usize) -> anyhow::Result<Vec<u32>> {
        let nat = audio_embeds.dims()[0];
        let (input_ids, audio_start_pos) = self.build_prompt(nat, language, prefix_text)?;
        let seq_len = input_ids.len();
        let hidden_size = self.config.thinker_config.text_config.hidden_size;
        let text_cfg = &self.config.thinker_config.text_config;
        let cuda = &self.gpu_decoder.cuda;

        // Audio embeds → f16 vec (already on GPU as burn Tensor; download once to host)
        let ae_data = audio_embeds.clone().into_data();
        let ae_f16: Vec<f16> = ae_data.to_vec::<f16>()
            .unwrap_or_else(|_| {
                let f32_data: Vec<f32> = ae_data.to_vec::<f32>().unwrap();
                f32_data.into_iter().map(f16::from_f32).collect()
            });

        // Build hidden_states on CPU once, then upload to GPU.
        let before_ids: Vec<i64> = input_ids[..audio_start_pos].to_vec();
        let after_ids: Vec<i64> = input_ids[audio_start_pos + nat..].to_vec();
        let before_emb = self.gpu_decoder.embed_ids(&before_ids)?;
        let after_emb = self.gpu_decoder.embed_ids(&after_ids)?;
        let before_cpu = cuda.download_tensor(&before_emb)?;
        let after_cpu = cuda.download_tensor(&after_emb)?;

        let mut hs_data = Vec::with_capacity(seq_len * hidden_size);
        hs_data.extend_from_slice(&before_cpu.data);
        hs_data.extend_from_slice(&ae_f16);
        hs_data.extend_from_slice(&after_cpu.data);
        let hidden_cpu = CpuTensor::new(hs_data, vec![1, seq_len, hidden_size]);
        let hidden_states = cuda.upload_tensor(&hidden_cpu)?;

        // MRoPE tables — full positions, upload entire table to GPU once.
        let total_positions = seq_len + max_new_tokens;
        let all_pos: Vec<i64> = (0..total_positions as i64).collect();
        let full_ids: [Vec<i64>; 3] = [all_pos.clone(), all_pos.clone(), all_pos.clone()];
        let (cos_table_cpu, sin_table_cpu) = cublas_compute_mrope_cos_sin(
            &full_ids, text_cfg.head_dim, text_cfg.rope_theta,
            &text_cfg.mrope_section(), text_cfg.mrope_interleaved(),
        );
        let cos_table = cuda.upload_f16(&cos_table_cpu.data)?;
        let sin_table = cuda.upload_f16(&sin_table_cpu.data)?;

        // KV cache pre-allocated for the full sequence (b=1).
        let mut kv_cache = GpuKvCache::new(
            cuda, text_cfg.num_hidden_layers, 1,
            text_cfg.num_key_value_heads, total_positions, text_cfg.head_dim
        )?;

        // ── Prefill ──
        let logits = self.gpu_decoder.forward(hidden_states, &cos_table, &sin_table, &mut kv_cache, 0, true, true)?;
        let mut current_pos = seq_len;

        let mut generated_ids: Vec<u32> = Vec::new();
        let eos_ids: &[i64] = &[ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID];

        // GPU-resident next-token slot (slot 0) so we can chain argmax→embed without an htod roundtrip.
        let mut token_buf = cuda.alloc_uninit_i32(1)?;

        // First token from prefill (slice last row already done by llo).  Write into token_buf[0]
        // and download to seed the EOS check.
        cuda.argmax_into(&logits, &mut token_buf, 0)?;
        let dl = cuda.download_i32(&token_buf)?;
        let mut next_token = dl[0] as i64;

        // ── Decode loop ──
        let t_decode = std::time::Instant::now();
        for _step in 0..max_new_tokens {
            if eos_ids.contains(&next_token) { break; }
            generated_ids.push(next_token as u32);

            let ne = self.gpu_decoder.embed_id_from_gpu_slot(&token_buf, 0)?
                .reshape(vec![1, 1, hidden_size]);
            let sl = self.gpu_decoder.forward(ne, &cos_table, &sin_table, &mut kv_cache, current_pos, false, true)?;
            cuda.argmax_into(&sl, &mut token_buf, 0)?;
            let dl = cuda.download_i32(&token_buf)?;
            next_token = dl[0] as i64;
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

    pub(crate) fn decode_result(&self, generated_ids: &[u32], language: Option<&str>) -> anyhow::Result<TranscribeResult> {
        let raw_text = self.tokenizer.decode(generated_ids, true).map_err(|e| anyhow::anyhow!("decode: {}", e))?;
        let (lang, text) = if language.is_some() {
            ("forced".to_string(), raw_text.trim().to_string())
        } else if let Some(sep_pos) = generated_ids.iter().position(|&id| id == ASR_TEXT_SEP_TOKEN_ID) {
            let lang_ids: Vec<u32> = generated_ids[..sep_pos].to_vec();
            let text_ids: Vec<u32> = generated_ids[sep_pos + 1..].to_vec();
            let lang_raw = self.tokenizer.decode(&lang_ids, true).map_err(|e| anyhow::anyhow!("decode lang: {}", e))?;
            let text_raw = self.tokenizer.decode(&text_ids, true).map_err(|e| anyhow::anyhow!("decode text: {}", e))?;
            (lang_raw.strip_prefix("language ").unwrap_or(&lang_raw).trim().to_string(), text_raw.trim().to_string())
        } else { parse_asr_output(&raw_text, false) };
        Ok(TranscribeResult { text, language: lang, raw_output: raw_text })
    }

    pub(crate) fn tokenizer_decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.tokenizer.decode(ids, true).map_err(|e| anyhow::anyhow!("decode: {}", e))
    }

    pub(crate) fn build_prompt(&self, nat: usize, language: Option<&str>, prefix_text: Option<&str>) -> anyhow::Result<(Vec<i64>, usize)> {
        let cfg = &self.config.thinker_config;
        let mut tokens: Vec<i64> = vec![TOK_IM_START, TOK_SYSTEM, TOK_NEWLINE, TOK_IM_END, TOK_NEWLINE, TOK_IM_START, TOK_USER, TOK_NEWLINE, cfg.audio_start_token_id];
        let asp = tokens.len();
        tokens.extend(std::iter::repeat_n(cfg.audio_token_id, nat));
        tokens.extend_from_slice(&[cfg.audio_end_token_id, TOK_IM_END, TOK_NEWLINE, TOK_IM_START]);
        if let Some(lang) = language {
            tokens.push(TOK_ASSISTANT); tokens.push(TOK_NEWLINE);
            let lang_str = format!("language {}", capitalize_first(lang));
            let enc = self.tokenizer.encode(lang_str.as_str(), false).map_err(|e| anyhow::anyhow!("encode: {}", e))?;
            tokens.extend(enc.get_ids().iter().map(|&id| id as i64));
        } else { tokens.push(TOK_ASSISTANT); tokens.push(TOK_NEWLINE); }
        if let Some(prefix) = prefix_text { if !prefix.is_empty() {
            let enc = self.tokenizer.encode(prefix, false).map_err(|e| anyhow::anyhow!("encode prefix: {}", e))?;
            tokens.extend(enc.get_ids().iter().map(|&id| id as i64));
        }}
        Ok((tokens, asp))
    }
}

fn parse_asr_output(raw: &str, forced: bool) -> (String, String) {
    if forced { return ("forced".to_string(), raw.trim().to_string()); }
    let raw = raw.trim();
    if let Some(rest) = raw.strip_prefix("language ") {
        if let Some(pos) = rest.find("<asr_text>") { return (rest[..pos].trim().to_string(), rest[pos + "<asr_text>".len()..].trim().to_string()); }
        let mut le = rest.len();
        for (i, c) in rest.char_indices() { if c.is_whitespace() || !c.is_alphabetic() { le = i; break; } }
        if le > 0 && le < rest.len() { return (rest[..le].to_string(), rest[le..].trim().to_string()); }
    }
    ("unknown".to_string(), raw.to_string())
}

fn capitalize_first(s: &str) -> String {
    let mut c = s.chars(); match c.next() { None => String::new(), Some(f) => f.to_uppercase().collect::<String>() + c.as_str() }
}

/// Conv-stem token-count formula: 3 stride-2 convs reduce sequence length to ((((l-1)/2+1)-1)/2+1)/2+1.
#[cfg(feature = "cuda")]
fn feo_inf(l: usize) -> usize {
    let f = |x: usize| (x - 1) / 2 + 1;
    f(f(f(l)))
}

// ─── Weight loading ──────────────────────────────────────────────

fn load_weights(model_dir: &Path) -> anyhow::Result<HashMap<String, TensorData>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;
        let wm = idx["weight_map"].as_object().ok_or_else(|| anyhow::anyhow!("invalid index.json"))?;
        let mut sf: std::collections::HashSet<String> = std::collections::HashSet::new();
        for v in wm.values() { if let Some(s) = v.as_str() { sf.insert(s.to_string()); } }
        let mut all = HashMap::new();
        for s in sf { all.extend(load_safetensors_file(&model_dir.join(&s))?); }
        return Ok(all);
    }
    load_safetensors_file(&model_dir.join("model.safetensors"))
}

fn load_safetensors_file(path: &Path) -> anyhow::Result<HashMap<String, TensorData>> {
    let buf = std::fs::read(path)?;
    let st = safetensors::SafeTensors::deserialize(&buf).map_err(|e| anyhow::anyhow!("safetensors: {}", e))?;
    let names = st.names();
    let tensors = st.tensors();
    let mut weights = HashMap::new();
    for i in 0..names.len() {
        let name = names[i];
        let view = &tensors[i];
        let raw = view.1.data();
        let shape: Vec<usize> = view.1.shape().to_vec();
        let f32_data: Vec<f32> = match view.1.dtype() {
            safetensors::Dtype::F32 => raw.chunks_exact(4).map(|c| {
                f32::from_ne_bytes([c[0], c[1], c[2], c[3]])
            }).collect(),
            safetensors::Dtype::BF16 => raw.chunks_exact(2).map(|c| {
                let b = u16::from_ne_bytes([c[0], c[1]]);
                f32::from_bits((b as u32) << 16)
            }).collect(),
            safetensors::Dtype::F16 => raw.chunks_exact(2).map(|c| {
                half::f16::from_ne_bytes([c[0], c[1]]).to_f32()
            }).collect(),
            other => anyhow::bail!("unsupported dtype: {:?} in {}", other, name),
        };
        weights.insert(name.to_string(), TensorData::new(f32_data, shape));
    }
    Ok(weights)
}
