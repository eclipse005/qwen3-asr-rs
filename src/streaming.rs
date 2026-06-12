//! Chunked streaming ASR session — accepts audio incrementally, encodes in chunks,
//! and produces streaming text on flush.

use log::{debug, info};
use std::sync::MutexGuard;

use crate::config::AsrConfig;
use crate::cpu_audio_encoder::{feo, CpuAudioEncoder};
use crate::error::AsrError;
use crate::inference::{
    AsrInferenceInner, Engine, StreamToken, TranscribeOptions, TranscribeResult,
};
use crate::mel::MelExtractor;
use crate::prompt;

#[cfg(feature = "cuda")]
use crate::gpu_audio_encoder::{feo as gpu_feo, GpuAudioEncoder};
#[cfg(feature = "cuda")]
use half::f16;

// ── Public session ────────────────────────────────────────────────

/// A streaming ASR session that accepts audio incrementally.
///
/// Created via [`AsrInference::create_streaming_session()`].
/// Audio is encoded chunk-by-chunk during [`push_samples()`](Self::push_samples),
/// so by the time [`flush()`](Self::flush) is called, most encoding work is done.
///
/// Holds the inference lock for its lifetime — callers must not use
/// the parent `AsrInference` while a session is active.
pub struct AsrStreamingSession<'a> {
    inner: MutexGuard<'a, AsrInferenceInner>,
    options: TranscribeOptions,
    state: StreamingEncoderState,
}

/// Backend-agnostic encoder state for chunked processing.
enum StreamingEncoderState {
    Cpu(CpuEncoderState),
    #[cfg(feature = "cuda")]
    Cuda(CudaEncoderState),
}

// ── CPU encoder state ─────────────────────────────────────────────

struct CpuEncoderState {
    encoder: &'static CpuAudioEncoder,
    mel_extractor: &'static MelExtractor,
    config: &'static AsrConfig,

    // Audio accumulation
    sample_buffer: Vec<f32>,
    processed_mel_frames: usize, // mel frames already encoded

    // Conv-stem token accumulation [n_tokens, d_model]
    token_buffer: Vec<f32>,
    n_conv_tokens: usize,

    // Transformer output [n_total, output_dim]
    embeddings: Vec<f32>,
    n_embed_tokens: usize,

    // Chunking constants
    cs: usize,   // n_window * 2 mel frames per conv chunk
    tpc: usize,  // tokens per conv chunk = feo(cs)
    ws: usize,   // transformer window size in tokens
}

// ── CUDA encoder state ────────────────────────────────────────────

#[cfg(feature = "cuda")]
struct CudaEncoderState {
    encoder: &'static GpuAudioEncoder,
    mel_extractor: &'static MelExtractor,
    audio_config: crate::config::AudioEncoderConfig,

    sample_buffer: Vec<f32>,
    processed_mel_frames: usize,
    token_buffer: Vec<f16>,
    n_conv_tokens: usize,
    embeddings: Vec<f32>,
    n_embed_tokens: usize,
    cs: usize,
    tpc: usize,
    ws: usize,
}

// ── Session implementation ────────────────────────────────────────

impl<'a> AsrStreamingSession<'a> {
    pub(crate) fn new(
        inner: MutexGuard<'a, AsrInferenceInner>,
        options: TranscribeOptions,
    ) -> Self {
        let state = match &inner.engine {
            Engine::Cpu { audio_encoder, .. } => {
                let cfg = &audio_encoder.config();
                let cs = cfg.n_window * 2;
                let tpc = feo(cs);
                let cpw = cfg.n_window_infer / cs;
                let ws = tpc * cpw;
                // SAFETY: The MutexGuard ensures the inner struct outlives the session.
                // We transmute references to 'static because the session lifetime is
                // bounded by the MutexGuard — the data won't be dropped while held.
                StreamingEncoderState::Cpu(CpuEncoderState {
                    encoder: unsafe { std::mem::transmute(audio_encoder) },
                    mel_extractor: unsafe { std::mem::transmute(&inner.mel_extractor) },
                    config: unsafe { std::mem::transmute(&inner.config) },
                    sample_buffer: Vec::new(),
                    processed_mel_frames: 0,
                    token_buffer: Vec::new(),
                    n_conv_tokens: 0,
                    embeddings: Vec::new(),
                    n_embed_tokens: 0,
                    cs, tpc, ws,
                })
            }
            #[cfg(feature = "cuda")]
            Engine::Cuda { audio_encoder, .. } => {
                let cfg = audio_encoder.config();
                let cs = cfg.n_window * 2;
                let tpc = gpu_feo(cs);
                let cpw = cfg.n_window_infer / cs;
                let ws = tpc * cpw;
                StreamingEncoderState::Cuda(CudaEncoderState {
                    encoder: unsafe { std::mem::transmute(audio_encoder) },
                    mel_extractor: unsafe { std::mem::transmute(&inner.mel_extractor) },
                    audio_config: cfg.clone(),
                    sample_buffer: Vec::new(),
                    processed_mel_frames: 0,
                    token_buffer: Vec::new(),
                    n_conv_tokens: 0,
                    embeddings: Vec::new(),
                    n_embed_tokens: 0,
                    cs, tpc, ws,
                })
            }
        };
        Self { inner, options, state }
    }

    /// Feed more audio samples (16kHz mono f32).
    pub fn push_samples(&mut self, samples: &[f32]) -> crate::Result<()> {
        match &mut self.state {
            StreamingEncoderState::Cpu(s) => s.push(samples),
            #[cfg(feature = "cuda")]
            StreamingEncoderState::Cuda(s) => s.push(samples),
        }
    }

    /// Finalize without streaming callback.
    pub fn flush(&mut self) -> crate::Result<TranscribeResult> {
        self.flush_streaming(|_| {})
    }

    /// Finalize: encode remaining audio, then decode text with streaming callback.
    pub fn flush_streaming<F>(&mut self, on_token: F) -> crate::Result<TranscribeResult>
    where
        F: FnMut(StreamToken),
    {
        let embeddings = match &mut self.state {
            StreamingEncoderState::Cpu(s) => s.flush()?,
            #[cfg(feature = "cuda")]
            StreamingEncoderState::Cuda(s) => s.flush()?,
        };

        if embeddings.is_empty() {
            return Err(AsrError::Inference(anyhow::anyhow!("no audio data provided")));
        }

        let nat = embeddings.len() / self.inner.config.thinker_config.text_config.hidden_size;
        info!("Flush: {} audio tokens, starting text decode", nat);

        // Run generate with a wrapper that does incremental text decode
        let tokenizer = &self.inner.tokenizer;
        let mut all_ids: Vec<u32> = Vec::new();
        let mut on_token = on_token;
        let mut raw_cb = |token_id: u32| {
            all_ids.push(token_id);
            let text = tokenizer.decode(&all_ids, true).unwrap_or_default();
            on_token(StreamToken { token_id, text_so_far: text });
        };

        let final_ids = self.inner.generate_with_callback(
            &embeddings,
            self.options.language.as_deref(),
            None,
            self.options.max_new_tokens,
            &mut raw_cb,
        ).map_err(AsrError::Inference)?;

        prompt::decode_result(&self.inner.tokenizer, &final_ids, self.options.language.as_deref())
            .map_err(AsrError::Inference)
    }

    /// Number of audio samples accumulated so far.
    pub fn sample_count(&self) -> usize {
        match &self.state {
            StreamingEncoderState::Cpu(s) => s.sample_buffer.len(),
            #[cfg(feature = "cuda")]
            StreamingEncoderState::Cuda(s) => s.sample_buffer.len(),
        }
    }
}

// ── CPU encoder state methods ─────────────────────────────────────

impl CpuEncoderState {
    fn push(&mut self, samples: &[f32]) -> crate::Result<()> {
        self.sample_buffer.extend_from_slice(samples);
        self.try_encode_new_chunks()?;
        Ok(())
    }

    fn flush(&mut self) -> crate::Result<Vec<f32>> {
        // Process any remaining mel frames through conv-stem
        self.encode_all_remaining_mel()?;

        // Process any remaining tokens through transformer
        self.flush_transformer()?;

        info!("Encoder flush: {} tokens ({} floats)", self.n_embed_tokens, self.embeddings.len());
        Ok(self.embeddings.clone())
    }

    fn try_encode_new_chunks(&mut self) -> crate::Result<()> {
        let n_mels = self.config.thinker_config.audio_config.num_mel_bins;
        let cs = self.cs;

        // Need enough samples for at least one new mel frame beyond what we've processed
        loop {
            // Re-extract mel from full buffer
            let (mel_data, _n_mels, n_frames) = self.mel_extractor.extract(&self.sample_buffer)
                .map_err(AsrError::Inference)?;

            let new_frames = n_frames.saturating_sub(self.processed_mel_frames);
            let new_full_chunks = new_frames / cs;
            if new_full_chunks == 0 {
                break;
            }

            // Process new conv-stem chunks
            for ci in 0..new_full_chunks {
                let frame_start = self.processed_mel_frames + ci * cs;
                // Extract mel chunk [n_mels, cs] from the full mel
                let mut mel_chunk = vec![0.0f32; n_mels * cs];
                for m in 0..n_mels {
                    let src_base = m * n_frames + frame_start;
                    let dst_base = m * cs;
                    for j in 0..cs {
                        mel_chunk[dst_base + j] = mel_data[src_base + j];
                    }
                }

                let tokens = self.encoder.run_conv_stem(&mel_chunk, n_mels, cs)
                    .map_err(AsrError::Inference)?;
                self.token_buffer.extend_from_slice(&tokens);
                self.n_conv_tokens += self.tpc;
            }
            self.processed_mel_frames += new_full_chunks * cs;

            // Run transformer on complete windows
            self.process_complete_windows()?;
        }
        Ok(())
    }

    fn encode_all_remaining_mel(&mut self) -> crate::Result<()> {
        let n_mels = self.config.thinker_config.audio_config.num_mel_bins;
        let cs = self.cs;

        let (mel_data, _n_mels, n_frames) = self.mel_extractor.extract(&self.sample_buffer)
            .map_err(AsrError::Inference)?;

        let remaining = n_frames - self.processed_mel_frames;

        // Process full chunks
        let full_chunks = remaining / cs;
        for ci in 0..full_chunks {
            let frame_start = self.processed_mel_frames + ci * cs;
            let mut mel_chunk = vec![0.0f32; n_mels * cs];
            for m in 0..n_mels {
                let src_base = m * n_frames + frame_start;
                let dst_base = m * cs;
                for j in 0..cs {
                    mel_chunk[dst_base + j] = mel_data[src_base + j];
                }
            }
            let tokens = self.encoder.run_conv_stem(&mel_chunk, n_mels, cs)
                .map_err(AsrError::Inference)?;
            self.token_buffer.extend_from_slice(&tokens);
            self.n_conv_tokens += self.tpc;
        }
        self.processed_mel_frames += full_chunks * cs;

        // Process tail chunk
        let tail_frames = remaining % cs;
        if tail_frames > 0 {
            let frame_start = self.processed_mel_frames;
            let mut mel_chunk = vec![0.0f32; n_mels * tail_frames];
            for m in 0..n_mels {
                let src_base = m * n_frames + frame_start;
                let dst_base = m * tail_frames;
                for j in 0..tail_frames {
                    mel_chunk[dst_base + j] = mel_data[src_base + j];
                }
            }
            let tpc_tail = feo(tail_frames);
            let tokens = self.encoder.run_conv_stem_tail(&mel_chunk, n_mels, tail_frames, cs)
                .map_err(AsrError::Inference)?;
            self.token_buffer.extend_from_slice(&tokens);
            self.n_conv_tokens += tpc_tail;
        }

        // Process complete windows
        self.process_complete_windows()?;
        Ok(())
    }

    fn process_complete_windows(&mut self) -> crate::Result<()> {
        let dm = self.config.thinker_config.audio_config.d_model;

        while self.n_conv_tokens >= self.ws {
            // Extract one window worth of tokens
            let window_len = self.ws * dm;
            let window_tokens: Vec<f32> = self.token_buffer.drain(..window_len).collect();
            self.n_conv_tokens -= self.ws;

            // Run transformer
            let out = self.encoder.run_transformer(&window_tokens, self.ws)
                .map_err(AsrError::Inference)?;

            self.embeddings.extend_from_slice(&out);
            self.n_embed_tokens += self.ws;
            debug!("Processed transformer window: {} total embed tokens", self.n_embed_tokens);
        }
        Ok(())
    }

    fn flush_transformer(&mut self) -> crate::Result<()> {
        if self.n_conv_tokens == 0 {
            return Ok(());
        }

        let tokens: Vec<f32> = self.token_buffer.drain(..).collect();
        let n = self.n_conv_tokens;
        self.n_conv_tokens = 0;

        let out = self.encoder.run_transformer(&tokens, n)
            .map_err(AsrError::Inference)?;

        self.embeddings.extend_from_slice(&out);
        self.n_embed_tokens += n;
        Ok(())
    }
}

// ── CUDA encoder state methods ────────────────────────────────────

#[cfg(feature = "cuda")]
impl CudaEncoderState {
    fn push(&mut self, samples: &[f32]) -> crate::Result<()> {
        self.sample_buffer.extend_from_slice(samples);
        self.try_encode_new_chunks()?;
        Ok(())
    }

    fn flush(&mut self) -> crate::Result<Vec<f32>> {
        self.encode_all_remaining_mel()?;
        self.flush_transformer()?;
        Ok(self.embeddings.clone())
    }

    fn try_encode_new_chunks(&mut self) -> crate::Result<()> {
        let n_mels = self.audio_config.num_mel_bins;
        let cs = self.cs;

        loop {
            let (mel_data, _n_mels, n_frames) = self.mel_extractor.extract(&self.sample_buffer)
                .map_err(AsrError::Inference)?;

            let new_frames = n_frames.saturating_sub(self.processed_mel_frames);
            let new_full_chunks = new_frames / cs;
            if new_full_chunks == 0 {
                break;
            }

            for ci in 0..new_full_chunks {
                let frame_start = self.processed_mel_frames + ci * cs;
                let mut mel_chunk = vec![0.0f32; n_mels * cs];
                for m in 0..n_mels {
                    let src_base = m * n_frames + frame_start;
                    let dst_base = m * cs;
                    for j in 0..cs {
                        mel_chunk[dst_base + j] = mel_data[src_base + j];
                    }
                }
                let tokens_f16 = self.encoder.run_conv_stem_single(&mel_chunk, n_mels, cs)
                    .map_err(AsrError::Inference)?;
                self.token_buffer.extend_from_slice(&tokens_f16);
                self.n_conv_tokens += self.tpc;
            }
            self.processed_mel_frames += new_full_chunks * cs;
            self.process_complete_windows()?;
        }
        Ok(())
    }

    fn encode_all_remaining_mel(&mut self) -> crate::Result<()> {
        let n_mels = self.audio_config.num_mel_bins;
        let cs = self.cs;

        let (mel_data, _n_mels, n_frames) = self.mel_extractor.extract(&self.sample_buffer)
            .map_err(AsrError::Inference)?;

        let remaining = n_frames - self.processed_mel_frames;
        let full_chunks = remaining / cs;

        for ci in 0..full_chunks {
            let frame_start = self.processed_mel_frames + ci * cs;
            let mut mel_chunk = vec![0.0f32; n_mels * cs];
            for m in 0..n_mels {
                let src_base = m * n_frames + frame_start;
                let dst_base = m * cs;
                for j in 0..cs {
                    mel_chunk[dst_base + j] = mel_data[src_base + j];
                }
            }
            let tokens_f16 = self.encoder.run_conv_stem_single(&mel_chunk, n_mels, cs)
                .map_err(AsrError::Inference)?;
            self.token_buffer.extend_from_slice(&tokens_f16);
            self.n_conv_tokens += self.tpc;
        }
        self.processed_mel_frames += full_chunks * cs;

        let tail_frames = remaining % cs;
        if tail_frames > 0 {
            let frame_start = self.processed_mel_frames;
            let mut mel_chunk = vec![0.0f32; n_mels * tail_frames];
            for m in 0..n_mels {
                let src_base = m * n_frames + frame_start;
                let dst_base = m * tail_frames;
                for j in 0..tail_frames {
                    mel_chunk[dst_base + j] = mel_data[src_base + j];
                }
            }
            let tpc_tail = gpu_feo(tail_frames);
            let tokens_f16 = self.encoder.run_conv_stem_tail(&mel_chunk, n_mels, tail_frames, cs)
                .map_err(AsrError::Inference)?;
            self.token_buffer.extend_from_slice(&tokens_f16);
            self.n_conv_tokens += tpc_tail;
        }

        self.process_complete_windows()?;
        Ok(())
    }

    fn process_complete_windows(&mut self) -> crate::Result<()> {
        let dm = self.audio_config.d_model;

        while self.n_conv_tokens >= self.ws {
            let window_len = self.ws * dm;
            let window_tokens: Vec<f16> = self.token_buffer.drain(..window_len).collect();
            self.n_conv_tokens -= self.ws;

            let (out_f16, _out_dim) = self.encoder.run_transformer(&window_tokens, self.ws)
                .map_err(AsrError::Inference)?;
            let out_f32: Vec<f32> = out_f16.iter().map(|&v| f32::from(v)).collect();
            self.embeddings.extend_from_slice(&out_f32);
            self.n_embed_tokens += self.ws;
        }
        Ok(())
    }

    fn flush_transformer(&mut self) -> crate::Result<()> {
        if self.n_conv_tokens == 0 {
            return Ok(());
        }

        let tokens: Vec<f16> = self.token_buffer.drain(..).collect();
        let n = self.n_conv_tokens;
        self.n_conv_tokens = 0;

        let (out_f16, _out_dim) = self.encoder.run_transformer(&tokens, n)
            .map_err(AsrError::Inference)?;
        let out_f32: Vec<f32> = out_f16.iter().map(|&v| f32::from(v)).collect();
        self.embeddings.extend_from_slice(&out_f32);
        self.n_embed_tokens += n;
        Ok(())
    }
}
