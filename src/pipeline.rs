//! Backend pipeline abstraction.
//!
//! CPU and CUDA (and future HIP/Metal/…) are equal backends behind
//! [`AsrPipeline`].  Shared code owns load/API/generate *dispatch*; each
//! backend owns memory, kernels, and the concrete encode/generate path.
//!
//! Intentionally stage-level (encode + generate), not per-op — so CUDA can
//! keep fused decode, async D2H, and zero-alloc scratch without the trait
//! forcing host round-trips.

use anyhow::Result;
use tokenizers::Tokenizer;

use crate::config::AsrConfig;

/// One ASR compute backend (CPU, CUDA, …).
///
/// Implementors hold decoder + audio encoder (+ device state).  Methods must
/// preserve existing numerical behaviour of the pre-refactor engines.
pub(crate) trait AsrPipeline: Send {
    /// Short tag for logs (`"cpu"`, `"cuda"`, …).
    fn tag(&self) -> &'static str;

    /// Mel spectrogram → audio token embeddings as host `f32` `[nat * dim]`.
    ///
    /// `n_window` is the audio encoder config field (used by the CUDA path;
    /// CPU may ignore it).
    fn encode_from_mel(
        &self,
        mel: &[f32],
        n_mels: usize,
        n_frames: usize,
        n_window: usize,
    ) -> Result<Vec<f32>>;

    /// Prefill + autoregressive decode.  Invokes `on_token` for each emitted
    /// non-EOS token (same contract as the pre-refactor generate loops).
    fn generate(
        &self,
        tokenizer: &Tokenizer,
        config: &AsrConfig,
        audio_embeds: &[f32],
        language: Option<&str>,
        prefix_text: Option<&str>,
        max_new_tokens: usize,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<Vec<u32>>;
}

// ── CPU backend ───────────────────────────────────────────────────

pub(crate) struct CpuPipeline {
    pub decoder: crate::cpu_engine::CpuTextDecoder,
    pub audio_encoder: crate::cpu_audio_encoder::CpuAudioEncoder,
}

impl AsrPipeline for CpuPipeline {
    fn tag(&self) -> &'static str {
        "cpu"
    }

    fn encode_from_mel(
        &self,
        mel: &[f32],
        n_mels: usize,
        n_frames: usize,
        _n_window: usize,
    ) -> Result<Vec<f32>> {
        self.audio_encoder.forward(mel, n_mels, n_frames)
    }

    fn generate(
        &self,
        tokenizer: &Tokenizer,
        config: &AsrConfig,
        audio_embeds: &[f32],
        language: Option<&str>,
        prefix_text: Option<&str>,
        max_new_tokens: usize,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<Vec<u32>> {
        use crate::cpu_engine::{compute_mrope_cos_sin as cpu_mrope, CpuKvCache, CpuTensor};
        use crate::prompt::{self, ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID};
        use log::info;

        let cpu = &self.decoder;
        let text_cfg = &config.thinker_config.text_config;
        let hidden_size = text_cfg.hidden_size;
        let nat = audio_embeds.len() / hidden_size;
        let (input_ids, audio_start_pos) = prompt::build_prompt(
            tokenizer,
            config.thinker_config.audio_start_token_id,
            config.thinker_config.audio_token_id,
            config.thinker_config.audio_end_token_id,
            nat,
            language,
            prefix_text,
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
            &full_ids,
            text_cfg.head_dim,
            text_cfg.rope_theta,
            &text_cfg.mrope_section(),
            text_cfg.mrope_interleaved(),
        );

        let mut kv_cache = CpuKvCache::new(
            text_cfg.num_hidden_layers,
            1,
            text_cfg.num_key_value_heads,
            total_positions,
            text_cfg.head_dim,
        );

        let t_prefill = std::time::Instant::now();
        let logits = cpu.forward(
            hidden_states,
            &cos_table,
            &sin_table,
            &mut kv_cache,
            0,
            true,
            true,
        );
        let mut current_pos = seq_len;

        let mut generated_ids: Vec<u32> = Vec::new();
        let eos_ids: &[i64] = &[ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID];
        let mut next_token = crate::cpu_engine::argmax(&logits.data) as i64;
        info!(
            "Prefill: {:.2}ms",
            t_prefill.elapsed().as_secs_f64() * 1000.0
        );

        let t_decode = std::time::Instant::now();
        for _step in 0..max_new_tokens {
            if eos_ids.contains(&next_token) {
                break;
            }
            generated_ids.push(next_token as u32);
            on_token(next_token as u32);

            let ne = cpu.embed_ids(&[next_token]).reshape(vec![1, 1, hidden_size]);
            let sl = cpu.forward(
                ne,
                &cos_table,
                &sin_table,
                &mut kv_cache,
                current_pos,
                false,
                true,
            );
            next_token = crate::cpu_engine::argmax(&sl.data) as i64;
            current_pos += 1;
        }
        let n_gen = generated_ids.len().max(1);
        info!(
            "Decode: {:.2}ms total ({} tokens, {:.2}ms/tok)",
            t_decode.elapsed().as_secs_f64() * 1000.0,
            generated_ids.len(),
            t_decode.elapsed().as_secs_f64() * 1000.0 / n_gen as f64
        );

        info!("Generated {} tokens", generated_ids.len());
        Ok(generated_ids)
    }
}

// ── CUDA backend ──────────────────────────────────────────────────

#[cfg(feature = "cuda")]
pub(crate) struct CudaPipeline {
    pub cuda: std::sync::Arc<crate::cudarc_engine::CudaState>,
    pub decoder: crate::cudarc_engine::GpuTextDecoder,
    pub audio_encoder: crate::gpu_audio_encoder::GpuAudioEncoder,
}

#[cfg(feature = "cuda")]
impl AsrPipeline for CudaPipeline {
    fn tag(&self) -> &'static str {
        "cuda"
    }

    fn encode_from_mel(
        &self,
        mel: &[f32],
        n_mels: usize,
        n_frames: usize,
        n_window: usize,
    ) -> Result<Vec<f32>> {
        self.audio_encoder
            .encode_from_mel(mel, n_mels, n_frames, n_window)
    }

    fn generate(
        &self,
        tokenizer: &Tokenizer,
        config: &AsrConfig,
        audio_embeds: &[f32],
        language: Option<&str>,
        prefix_text: Option<&str>,
        max_new_tokens: usize,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<Vec<u32>> {
        use crate::cudarc_engine::{
            compute_mrope_cos_sin as cublas_compute_mrope_cos_sin, CpuTensor, DecodeScratch,
            GpuKvCache,
        };
        use crate::prompt::{self, ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID};
        use half::f16;
        use log::info;

        let cuda = &self.cuda;
        let decoder = &self.decoder;

        let nat = audio_embeds.len() / config.thinker_config.text_config.hidden_size;
        let (input_ids, audio_start_pos) = prompt::build_prompt(
            tokenizer,
            config.thinker_config.audio_start_token_id,
            config.thinker_config.audio_token_id,
            config.thinker_config.audio_end_token_id,
            nat,
            language,
            prefix_text,
        )?;
        let seq_len = input_ids.len();
        let hidden_size = config.thinker_config.text_config.hidden_size;
        let text_cfg = &config.thinker_config.text_config;

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
            &full_ids,
            text_cfg.head_dim,
            text_cfg.rope_theta,
            &text_cfg.mrope_section(),
            text_cfg.mrope_interleaved(),
        );
        let cos_table = cuda.upload_f16(&cos_table_cpu.data)?;
        let sin_table = cuda.upload_f16(&sin_table_cpu.data)?;

        let mut kv_cache = GpuKvCache::new(
            cuda,
            text_cfg.num_hidden_layers,
            1,
            text_cfg.num_key_value_heads,
            total_positions,
            text_cfg.head_dim,
        )?;

        let t_prefill = std::time::Instant::now();
        let logits =
            decoder.forward(hidden_states, &cos_table, &sin_table, &mut kv_cache, 0, true, true)?;
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
        info!(
            "Prefill: {:.2}ms",
            t_prefill.elapsed().as_secs_f64() * 1000.0
        );

        let t_decode = std::time::Instant::now();
        loop {
            if eos_ids.contains(&next_token) {
                break;
            }
            generated_ids.push(next_token as u32);
            on_token(next_token as u32);
            if generated_ids.len() >= max_new_tokens {
                break;
            }

            cuda.embed_id_from_gpu_slot_into(&decoder.embed_table, &token_buf, 0, &mut h_buf)?;
            decoder.forward_decode_scratch(
                &mut h_buf,
                &cos_table,
                &sin_table,
                &mut kv_cache,
                current_pos,
                &mut token_buf,
                &mut scratch,
            )?;
            // D2H into pinned memory — avoids implicit full-stream sync of pageable D2H.
            // Safety: as_ptr() calls event.synchronize() ensuring the copy completes before returning.
            cuda.download_i32_into_pinned(&token_buf, &mut scratch.pinned_token)?;
            next_token = unsafe { *scratch.pinned_token.as_ptr()? } as i64;
            current_pos += 1;
        }
        cuda.synchronize()?;
        let n_gen = generated_ids.len().max(1);
        info!(
            "Decode: {:.2}ms total ({} tokens, {:.2}ms/tok)",
            t_decode.elapsed().as_secs_f64() * 1000.0,
            n_gen,
            t_decode.elapsed().as_secs_f64() * 1000.0 / n_gen as f64
        );

        info!("Generated {} tokens", generated_ids.len());
        Ok(generated_ids)
    }
}
