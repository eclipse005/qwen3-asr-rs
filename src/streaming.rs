use anyhow::Result;
use burn::tensor::{Tensor, TensorData};
use log::{debug, info};

use crate::encoder::EncoderCache;
use crate::error::AsrError;
use crate::inference::{AsrInference, AsrInferenceInner, TranscribeResult, MEL_SAMPLE_RATE};
use crate::Backend;

#[non_exhaustive]
pub struct StreamingOptions {
    pub language: Option<String>,
    pub chunk_size_sec: f32,
    pub unfixed_chunk_num: usize,
    pub unfixed_token_num: usize,
    pub max_new_tokens_streaming: usize,
    pub max_new_tokens_final: usize,
    pub initial_text: Option<String>,
}

impl Default for StreamingOptions {
    fn default() -> Self {
        Self {
            language: None,
            chunk_size_sec: 2.0,
            unfixed_chunk_num: 2,
            unfixed_token_num: 5,
            max_new_tokens_streaming: 32,
            max_new_tokens_final: 512,
            initial_text: None,
        }
    }
}

impl StreamingOptions {
    pub fn with_chunk_size_sec(mut self, sec: f32) -> Self { self.chunk_size_sec = sec; self }
    pub fn with_unfixed_chunk_num(mut self, n: usize) -> Self { self.unfixed_chunk_num = n; self }
    pub fn with_unfixed_token_num(mut self, n: usize) -> Self { self.unfixed_token_num = n; self }
    pub fn with_max_new_tokens_streaming(mut self, n: usize) -> Self { self.max_new_tokens_streaming = n; self }
    pub fn with_max_new_tokens_final(mut self, n: usize) -> Self { self.max_new_tokens_final = n; self }
    pub fn with_language(mut self, lang: impl Into<String>) -> Self { self.language = Some(lang.into()); self }
    pub fn with_initial_text(mut self, text: impl Into<String>) -> Self {
        let t = text.into();
        self.initial_text = if t.is_empty() { None } else { Some(t) };
        self
    }
}

pub struct StreamingState {
    buffer: Vec<f32>,
    audio_accum: Vec<f32>,
    chunk_size_samples: usize,
    chunk_id: usize,
    raw_token_ids: Vec<u32>,
    options: StreamingOptions,
    language: String,
    text: String,
    encoder_cache: EncoderCache<Backend>,
}

impl AsrInference {
    pub fn init_streaming(&self, options: StreamingOptions) -> StreamingState {
        let chunk_size_samples = (options.chunk_size_sec * MEL_SAMPLE_RATE as f32) as usize;
        StreamingState {
            buffer: Vec::new(),
            audio_accum: Vec::new(),
            chunk_size_samples,
            chunk_id: 0,
            raw_token_ids: Vec::new(),
            options,
            language: String::new(),
            text: String::new(),
            encoder_cache: EncoderCache::new(),
        }
    }

    pub fn feed_audio(
        &self,
        state: &mut StreamingState,
        samples: &[f32],
    ) -> crate::Result<Option<TranscribeResult>> {
        state.buffer.extend_from_slice(samples);
        if !try_drain_chunk(state) { return Ok(None); }

        let inner = self.inner.lock()
            .map_err(|_| AsrError::Inference(anyhow::anyhow!("mutex poisoned")))?;
        let result = run_streaming_step(&inner, state).map_err(AsrError::Inference)?;
        Ok(Some(result))
    }

    pub fn finish_streaming(&self, state: &mut StreamingState) -> crate::Result<TranscribeResult> {
        if !flush_remaining_buffer(state) {
            return Ok(TranscribeResult {
                text: String::new(), language: String::new(), raw_output: String::new(),
            });
        }

        let inner = self.inner.lock()
            .map_err(|_| AsrError::Inference(anyhow::anyhow!("mutex poisoned")))?;

        let audio_embeds = encode_audio_incremental(&inner, state).map_err(AsrError::Inference)?;
        let prefix = build_prefix(&inner, state);

        let generated_ids = inner.generate(
            &audio_embeds, state.options.language.as_deref(),
            prefix.as_deref(), state.options.max_new_tokens_final,
        ).map_err(AsrError::Inference)?;

        let full_ids = combine_prefix_and_generated(state, &prefix, &generated_ids);
        let result = inner.decode_result(&full_ids, state.options.language.as_deref())
            .map_err(AsrError::Inference)?;

        state.text = result.text.clone();
        state.language = result.language.clone();
        state.raw_token_ids = full_ids;

        Ok(result)
    }
}

fn encode_audio_incremental(
    inner: &AsrInferenceInner,
    state: &mut StreamingState,
) -> Result<Tensor<Backend, 2>> {
    let (mel_data, n_mels, n_frames) = inner.mel_extractor.extract(&state.audio_accum)?;
    debug!("Mel: {}×{} frames (incremental)", n_mels, n_frames);
    let mel = Tensor::<Backend, 2>::from_data(TensorData::new(mel_data, [n_mels, n_frames]), &inner.device);
    let audio_embeds = inner.audio_encoder.forward_incremental(&mel, &mut state.encoder_cache)?;
    info!(
        "Audio tokens (incremental): {} (cached: {})",
        audio_embeds.dims()[0], state.encoder_cache.cached_tokens()
    );
    Ok(audio_embeds)
}

fn run_streaming_step(
    inner: &AsrInferenceInner,
    state: &mut StreamingState,
) -> Result<TranscribeResult> {
    let audio_embeds = encode_audio_incremental(inner, state)?;
    let prefix = build_prefix(inner, state);

    info!(
        "Streaming step: chunk_id={}, accum_samples={}, prefix={:?}",
        state.chunk_id, state.audio_accum.len(),
        prefix.as_deref().unwrap_or("(none)"),
    );

    let generated_ids = inner.generate(
        &audio_embeds, state.options.language.as_deref(),
        prefix.as_deref(), state.options.max_new_tokens_streaming,
    )?;

    let full_ids = combine_prefix_and_generated(state, &prefix, &generated_ids);
    let result = inner.decode_result(&full_ids, state.options.language.as_deref())?;

    state.raw_token_ids = full_ids;
    state.text = result.text.clone();
    state.language = result.language.clone();

    Ok(result)
}

pub(crate) fn compute_prefix_ids(state: &StreamingState) -> Option<&[u32]> {
    if state.chunk_id <= state.options.unfixed_chunk_num { return None; }
    if state.raw_token_ids.is_empty() { return None; }
    let keep = state.raw_token_ids.len().saturating_sub(state.options.unfixed_token_num);
    if keep == 0 { return None; }
    Some(&state.raw_token_ids[..keep])
}

pub(crate) fn build_prefix(inner: &AsrInferenceInner, state: &StreamingState) -> Option<String> {
    if state.chunk_id <= state.options.unfixed_chunk_num {
        return state.options.initial_text.clone();
    }
    let prefix_ids = compute_prefix_ids(state)?;
    let prefix_text = inner.tokenizer_decode(prefix_ids).ok()?;
    if prefix_text.is_empty() { None } else { Some(prefix_text) }
}

fn try_drain_chunk(state: &mut StreamingState) -> bool {
    if state.buffer.len() < state.chunk_size_samples { return false; }
    let chunk: Vec<f32> = state.buffer.drain(..state.chunk_size_samples).collect();
    state.audio_accum.extend_from_slice(&chunk);
    state.chunk_id += 1;
    true
}

fn flush_remaining_buffer(state: &mut StreamingState) -> bool {
    if !state.buffer.is_empty() {
        let remaining: Vec<f32> = state.buffer.drain(..).collect();
        state.audio_accum.extend_from_slice(&remaining);
        state.chunk_id += 1;
    }
    !state.audio_accum.is_empty()
}

pub(crate) fn combine_prefix_and_generated(
    state: &StreamingState,
    prefix: &Option<String>,
    generated_ids: &[u32],
) -> Vec<u32> {
    if prefix.is_none() || state.raw_token_ids.is_empty() {
        return generated_ids.to_vec();
    }
    let keep = state.raw_token_ids.len().saturating_sub(state.options.unfixed_token_num);
    if keep == 0 { return generated_ids.to_vec(); }
    let mut full = Vec::with_capacity(keep + generated_ids.len());
    full.extend_from_slice(&state.raw_token_ids[..keep]);
    full.extend_from_slice(generated_ids);
    full
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(css: f32, ucn: usize, utn: usize) -> StreamingState {
        let opts = StreamingOptions {
            language: None, chunk_size_sec: css, unfixed_chunk_num: ucn, unfixed_token_num: utn,
            max_new_tokens_streaming: 32, max_new_tokens_final: 512, initial_text: None,
        };
        StreamingState {
            buffer: Vec::new(), audio_accum: Vec::new(),
            chunk_size_samples: (css * 16000.0) as usize, chunk_id: 0,
            raw_token_ids: Vec::new(), options: opts, language: String::new(), text: String::new(),
            encoder_cache: EncoderCache::new(),
        }
    }

    #[test] fn test_defaults() {
        let opts = StreamingOptions::default();
        assert_eq!(opts.chunk_size_sec, 2.0);
        assert_eq!(opts.unfixed_chunk_num, 2);
        assert_eq!(opts.unfixed_token_num, 5);
    }

    #[test] fn test_chunk_size_samples() {
        assert_eq!(make_state(2.0, 2, 5).chunk_size_samples, 32000);
        assert_eq!(make_state(1.0, 2, 5).chunk_size_samples, 16000);
    }

    #[test] fn test_drain_not_enough() {
        let mut s = make_state(2.0, 2, 5);
        s.buffer = vec![0.0; 16000];
        assert!(!try_drain_chunk(&mut s));
    }

    #[test] fn test_drain_exact() {
        let mut s = make_state(2.0, 2, 5);
        s.buffer = vec![0.1; 32000];
        assert!(try_drain_chunk(&mut s));
        assert_eq!(s.chunk_id, 1);
    }

    #[test] fn test_combine_no_prefix() {
        let s = make_state(2.0, 2, 5);
        assert_eq!(combine_prefix_and_generated(&s, &None, &[100, 200]), vec![100, 200]);
    }

    #[test] fn test_combine_with_prefix() {
        let mut s = make_state(2.0, 2, 5);
        s.raw_token_ids = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(combine_prefix_and_generated(&s, &Some("p".into()), &[100, 200]), vec![1, 2, 3, 4, 5, 100, 200]);
    }

    #[test] fn test_prefix_cold_start() {
        let mut s = make_state(2.0, 2, 5);
        s.chunk_id = 1;
        assert_eq!(compute_prefix_ids(&s), None);
    }

    #[test] fn test_prefix_eligible() {
        let mut s = make_state(2.0, 2, 5);
        s.chunk_id = 3;
        s.raw_token_ids = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(compute_prefix_ids(&s), Some(&[1, 2, 3, 4, 5][..]));
    }

    #[test] fn test_flush_empty() {
        let mut s = make_state(2.0, 2, 5);
        assert!(!flush_remaining_buffer(&mut s));
    }

    #[test] fn test_flush_partial() {
        let mut s = make_state(2.0, 2, 5);
        s.buffer = vec![0.1; 5000];
        assert!(flush_remaining_buffer(&mut s));
        assert_eq!(s.audio_accum.len(), 5000);
    }
}
