//! CPU audio encoder for Qwen3-ASR — f32 throughout.
//!
//! Mirrors `src/gpu_audio_encoder.rs` 1:1 but uses `gemm` + hand-written f32
//! loops (no cudarc, no half SIMD, no f16 internal storage). Weights are
//! upcast from f16 safetensors to `Vec<f32>` at load time.
//!
//! Architecture:
//!   mel [T_mel, 128]  →  conv2d stem (3 × {im2col + gemm + bias + GELU})
//!   → conv_out (Linear + bias)  →  + sinusoidal PE
//!   → 18 × { LN + windowed attn + LN + FFN(GELU) }
//!   → ln_post + proj1 + GELU + proj2
//!   → [n_total, output_dim]
//!
//! This module is `#[cfg(feature = "cpu")]` only.

use anyhow::Result;
use gemm::{gemm, Parallelism};
use std::collections::HashMap;

use crate::config::AudioEncoderConfig;
use crate::cpu_engine::{linear, CpuTensor, CpuWeight};
use crate::raw_tensor::RawTensor;

// ─── Linear + LayerNorm primitives ─────────────────────────────────

pub(crate) struct CpuAudioLinear {
    pub w: CpuWeight,
    pub bias: Option<Vec<f32>>,
}

impl CpuAudioLinear {
    pub(crate) fn load(
        weights: &HashMap<String, RawTensor>,
        prefix: &str,
    ) -> Result<Self> {
        let (data, shape) = weights
            .get(&format!("{}.weight", prefix))
            .ok_or_else(|| anyhow::anyhow!("weight not found: {}.weight", prefix))?
            .as_f32()?;
        let rows = shape[0];
        let cols = shape[1];
        let bias = if weights.contains_key(&format!("{}.bias", prefix)) {
            let (b, _) = weights
                .get(&format!("{}.bias", prefix))
                .unwrap()
                .as_f32()?;
            Some(b)
        } else {
            None
        };
        Ok(Self { w: CpuWeight { data, rows, cols }, bias })
    }

    /// x: [..., in_features] → [..., out_features]  (bias added if present)
    pub(crate) fn forward(&self, x: &CpuTensor) -> Result<CpuTensor> {
        let mut y = linear(x, &self.w);
        if let Some(b) = &self.bias {
            let nd = y.shape.len();
            let last = y.shape[nd - 1];
            let outer: usize = y.shape[..nd - 1].iter().product();
            for i in 0..outer {
                for j in 0..last {
                    y.data[i * last + j] += b[j];
                }
            }
        }
        Ok(y)
    }
}

pub(crate) struct CpuAudioLayerNorm {
    pub w: Vec<f32>,
    pub bias: Vec<f32>,
    pub eps: f32,
}

impl CpuAudioLayerNorm {
    pub(crate) fn load(
        weights: &HashMap<String, RawTensor>,
        prefix: &str,
        eps: f32,
    ) -> Result<Self> {
        let (w, _) = weights
            .get(&format!("{}.weight", prefix))
            .ok_or_else(|| anyhow::anyhow!("ln weight not found: {}.weight", prefix))?
            .as_f32()?;
        let (bias, _) = weights
            .get(&format!("{}.bias", prefix))
            .ok_or_else(|| anyhow::anyhow!("ln bias not found: {}.bias", prefix))?
            .as_f32()?;
        Ok(Self { w, bias, eps })
    }

    /// x: [outer, d] → [outer, d]  (LayerNorm over last dim)
    pub(crate) fn forward(&self, x: &CpuTensor) -> CpuTensor {
        let nd = x.shape.len();
        let d = x.shape[nd - 1];
        let outer: usize = x.shape[..nd - 1].iter().product();
        let mut out = vec![0.0f32; outer * d];
        for i in 0..outer {
            let row = &x.data[i * d..(i + 1) * d];
            let mut mean = 0.0f32;
            for &v in row { mean += v; }
            mean /= d as f32;
            let mut var = 0.0f32;
            for &v in row { let d_ = v - mean; var += d_ * d_; }
            var /= d as f32;
            let inv_std = 1.0 / (var + self.eps).sqrt();
            for j in 0..d {
                out[i * d + j] = (row[j] - mean) * inv_std * self.w[j] + self.bias[j];
            }
        }
        CpuTensor::new(out, x.shape.clone())
    }
}

/// In-place f32 GELU (tanh approximation, matching `cudarc::nn::gelu` default).
pub(crate) fn gelu_inplace(x: &mut CpuTensor) {
    for v in x.data.iter_mut() {
        // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let xf = *v;
        let inner = 0.7978845608_f32 * (xf + 0.044715_f32 * xf * xf * xf);
        *v = 0.5 * xf * (1.0 + inner.tanh());
    }
}

/// im2col for a single conv2d layer with kernel=3, stride=2, pad=1.
/// Input x: [b, c_in, h, w]   (h, w are 1 and T_mel initially).
/// Output: [col_count, 3*3*c_in] row-major, where col_count = b * c_in * h_out * w_out
///   with h_out = (h + 2 - 3) / 2 + 1, w_out = (w + 2 - 3) / 2 + 1.
pub(crate) fn im2col_3x3_s2p1(x: &[f32], b: usize, c_in: usize, h: usize, w: usize) -> (Vec<f32>, usize, usize) {
    let h_out = (h + 2 - 3) / 2 + 1;
    let w_out = (w + 2 - 3) / 2 + 1;
    let col_count = b * c_in * h_out * w_out;
    let mut cols = vec![0.0f32; col_count * 9 * c_in];
    for ib in 0..b {
        for ic in 0..c_in {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let col_idx = ((ib * c_in + ic) * h_out + ho) * w_out + wo;
                    let mut k = 0;
                    for kh in 0..3 {
                        for kw in 0..3 {
                            let ih = (ho * 2 + kh) as isize - 1;
                            let iw = (wo * 2 + kw) as isize - 1;
                            let v = if ih < 0 || ih >= h as isize || iw < 0 || iw >= w as isize {
                                0.0
                            } else {
                                x[((ib * c_in + ic) * h + ih as usize) * w + iw as usize]
                            };
                            cols[col_idx * 9 * c_in + k * c_in + ic] = v;
                            k += 1;
                        }
                    }
                }
            }
        }
    }
    (cols, h_out, w_out)
}

pub(crate) struct CpuConvStem {
    c1_w: CpuWeight, c1_b: Vec<f32>,
    c2_w: CpuWeight, c2_b: Vec<f32>,
    c3_w: CpuWeight, c3_b: Vec<f32>,
    co: CpuAudioLinear,
    pe: Vec<f32>,        // [max_pos, d_model] f32
    d_model: usize,
    max_pos: usize,
}

impl CpuConvStem {
    pub(crate) fn load(
        weights: &HashMap<String, RawTensor>,
        prefix: &str,
        config: &AudioEncoderConfig,
    ) -> Result<Self> {
        let c1_w = load_conv_weight(weights, &format!("{}.conv2d1.weight", prefix))?;
        let c1_b = load_bias(weights, &format!("{}.conv2d1.bias", prefix))?;
        let c2_w = load_conv_weight(weights, &format!("{}.conv2d2.weight", prefix))?;
        let c2_b = load_bias(weights, &format!("{}.conv2d2.bias", prefix))?;
        let c3_w = load_conv_weight(weights, &format!("{}.conv2d3.weight", prefix))?;
        let c3_b = load_bias(weights, &format!("{}.conv2d3.bias", prefix))?;
        let co = CpuAudioLinear::load(weights, &format!("{}.conv_out", prefix))?;

        // Sinusoidal PE — formula identical to `gpu_audio_encoder.rs:220-228`.
        let dm = config.d_model;
        let max_pos = config.max_source_positions;
        let half = dm / 2;
        let lt = (10000.0f64).ln() / (half as f64 - 1.0);
        let mut pe = vec![0.0f32; max_pos * dm];
        for p in 0..max_pos {
            for i in 0..half {
                let a = p as f64 * (-(i as f64) * lt).exp();
                pe[p * dm + i] = a.sin() as f32;
                pe[p * dm + half + i] = a.cos() as f32;
            }
        }
        Ok(Self { c1_w, c1_b, c2_w, c2_b, c3_w, c3_b, co, pe, d_model: dm, max_pos })
    }

    /// mel: [n_mels * T_mel] flat. Returns [n_total, d_model] flat.
    pub(crate) fn forward(
        &self,
        mel: &[f32],
        n_mels: usize,
        mel_len: usize,
    ) -> Result<(Vec<f32>, usize)> {
        // 1. Pack mel into [1, n_mels, 1, T_mel] (4D, batch=1, h=1).
        let h_in = 1usize;
        let w_in = mel_len;
        let c_in = n_mels;
        let x4d: Vec<f32> = mel.to_vec();

        // 2. Three conv2d (kernel=3, stride=2, pad=1) with bias + GELU.
        let (x1, h1, w1) = self.conv_block(&x4d, 1, c_in, h_in, w_in, &self.c1_w, &self.c1_b)?;
        let (x2, h2, w2) = self.conv_block(&x1, 1, c_in, h1, w1, &self.c2_w, &self.c2_b)?;
        let (x3, h3, w3) = self.conv_block(&x2, 1, c_in, h2, w2, &self.c3_w, &self.c3_b)?;
        // x3: [1, c_in(=c_out=128), h3=1, w3 = T_mel/8]
        let t2 = w3;

        // 3. Permute [b, c, f, t] → [b, t, c, f] then reshape [b, t, c*f].
        //    Source layout: [1, 128, 1, t2]  →  [1, t2, 128*1] = [1, t2, 128]
        let c2 = c_in;       // 128
        let f2 = h3;         // 1
        let mut perm = vec![0.0f32; t2 * c2 * f2];
        for it in 0..t2 {
            for ic in 0..c2 {
                for f in 0..f2 {
                    let src = (ic * f2 + f) * t2 + it;
                    let dst = (it * c2 + ic) * f2 + f;
                    perm[dst] = x3[src];
                }
            }
        }

        // 4. ConvOut (Linear) + bias → [1, t2, d_model]
        let perm_t = CpuTensor::new(perm, vec![1, t2, c2 * f2]);
        let co_out = self.co.forward(&perm_t)?;
        // co_out: [1, t2, d_model]

        // 5. Add PE (broadcast over batch). PE is indexed by t.
        let mut out = co_out.data.clone();
        let dm = self.d_model;
        for it in 0..t2 {
            for j in 0..dm {
                out[it * dm + j] += self.pe[it * dm + j];
            }
        }
        Ok((out, t2))
    }

    /// Single conv2d (3×3, stride 2, pad 1) + bias + GELU.
    /// Returns (flat out [b, c_out, h_out, w_out], h_out, w_out).
    fn conv_block(
        &self,
        x: &[f32],
        b: usize, c_in: usize, h: usize, w: usize,
        w_w: &CpuWeight, w_b: &[f32],
    ) -> Result<(Vec<f32>, usize, usize)> {
        let c_out = w_w.rows;
        let (cols, h_out, w_out) = im2col_3x3_s2p1(x, b, c_in, h, w);
        let col_count = b * c_in * h_out * w_out;
        let k = 9 * c_in;
        let mut out = vec![0.0f32; c_out * col_count];
        unsafe {
            gemm(
                c_out, col_count, k,
                out.as_mut_ptr(), 1, col_count as isize,
                false,
                w_w.data.as_ptr(), 1, k as isize,
                cols.as_ptr(), 1, k as isize,
                0.0, 1.0, false, false, false,
                Parallelism::Rayon(0),
            );
        }

        // Add bias and GELU. out is [c_out, col_count] (one per-channel conv output per col,
        // since im2col only fills the ic_in slot matching the col's channel — no implicit sum
        // over input channels). We must sum over ic_in to get the full conv2d output, then add
        // bias once per (ib, oc, ho, wo).
        let mut out4d = vec![0.0f32; b * c_out * h_out * w_out];
        for ib in 0..b {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    for oc in 0..c_out {
                        let mut v = w_b[oc];
                        for ic_in in 0..c_in {
                            let col = ((ib * c_in + ic_in) * h_out + ho) * w_out + wo;
                            v += out[oc * col_count + col];
                        }
                        out4d[((ib * c_out + oc) * h_out + ho) * w_out + wo] = v;
                    }
                }
            }
        }
        let mut out_t = CpuTensor::new(out4d, vec![b, c_out, h_out, w_out]);
        gelu_inplace(&mut out_t);
        Ok((out_t.data, h_out, w_out))
    }
}

fn load_conv_weight(weights: &HashMap<String, RawTensor>, name: &str) -> Result<CpuWeight> {
    let (data, shape) = weights.get(name).ok_or_else(|| anyhow::anyhow!("weight not found: {}", name))?.as_f32()?;
    let c_out = shape[0];
    let k = shape[1..].iter().product::<usize>();
    Ok(CpuWeight { data, rows: c_out, cols: k })
}

fn load_bias(weights: &HashMap<String, RawTensor>, name: &str) -> Result<Vec<f32>> {
    let (data, _) = weights.get(name).ok_or_else(|| anyhow::anyhow!("bias not found: {}", name))?.as_f32()?;
    Ok(data)
}

pub(crate) struct CpuAudioAttention {
    q_proj: CpuAudioLinear,
    k_proj: CpuAudioLinear,
    v_proj: CpuAudioLinear,
    out_proj: CpuAudioLinear,
    num_heads: usize,
    head_dim: usize,
}

impl CpuAudioAttention {
    pub(crate) fn load(
        weights: &HashMap<String, RawTensor>,
        prefix: &str,
        nh: usize,
        d_model: usize,
    ) -> Result<Self> {
        let hd = d_model / nh;
        Ok(Self {
            q_proj: CpuAudioLinear::load(weights, &format!("{}.q_proj", prefix))?,
            k_proj: CpuAudioLinear::load(weights, &format!("{}.k_proj", prefix))?,
            v_proj: CpuAudioLinear::load(weights, &format!("{}.v_proj", prefix))?,
            out_proj: CpuAudioLinear::load(weights, &format!("{}.out_proj", prefix))?,
            num_heads: nh,
            head_dim: hd,
        })
    }

    /// x [b, s, d_model] → [b, s, d_model]. ws: window size for attention.
    pub(crate) fn forward(
        &self,
        x: &CpuTensor,
        ws: Option<usize>,
    ) -> Result<CpuTensor> {
        let b = x.shape[0];
        let s = x.shape[1];
        let dm = x.shape[2];
        let nh = self.num_heads;
        let hd = self.head_dim;

        // Project Q, K, V.
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let scale = 1.0f32 / (hd as f32).sqrt();
        let window = ws.filter(|&w| w > 0 && w < s);

        let attn_out = if let Some(w) = window {
            // Windowed: process chunks of `w` tokens, each chunk attends only to itself.
            let mut out = vec![0.0f32; b * nh * s * hd];
            for st in (0..s).step_by(w) {
                let len = w.min(s - st);
                let q_chunk = slice_dim1(&q, st, len);
                let k_chunk = slice_dim1(&k, st, len);
                let v_chunk = slice_dim1(&v, st, len);
                let o = attention_window(&q_chunk, &k_chunk, &v_chunk, b, nh, len, hd, scale);
                scatter_dim2(&mut out, &o, b, nh, s, hd, st, len);
            }
            out
        } else {
            // Full attention (rare in this model — fallback).
            attention_window(&q, &k, &v, b, nh, s, hd, scale)
        };

        // Reshape [b, nh, s, hd] → [b, s, nh, hd] (swap dims 1 and 2) → [b, s, nh*hd].
        let mut flat = vec![0.0f32; b * s * nh * hd];
        for ib in 0..b {
            for ih in 0..nh {
                for is_ in 0..s {
                    for jd in 0..hd {
                        let src = ((ib * nh + ih) * s + is_) * hd + jd;
                        let dst = ((ib * s + is_) * nh + ih) * hd + jd;
                        flat[dst] = attn_out[src];
                    }
                }
            }
        }
        let attn_flat = CpuTensor::new(flat, vec![b, s, nh * hd]);
        self.out_proj.forward(&attn_flat)
    }
}

/// x: [b, s, dm] → out_chunk: [b, len, dm] for s in [st, st+len).
fn slice_dim1(x: &CpuTensor, st: usize, len: usize) -> CpuTensor {
    let b = x.shape[0];
    let dm = x.shape[2];
    let mut data = vec![0.0f32; b * len * dm];
    for ib in 0..b {
        for i in 0..len {
            for j in 0..dm {
                data[(ib * len + i) * dm + j] = x.data[((ib * x.shape[1] + st + i) * dm + j)];
            }
        }
    }
    CpuTensor::new(data, vec![b, len, dm])
}

/// Scatter `src [b, nh, len, hd]` into `dst [b, nh, s, hd]` at `dst[..., st:st+len, ...]`.
fn scatter_dim2(dst: &mut [f32], src: &[f32], b: usize, nh: usize, s: usize, hd: usize, st: usize, len: usize) {
    for ib in 0..b {
        for ih in 0..nh {
            for i in 0..len {
                for jd in 0..hd {
                    let sidx = ((ib * nh + ih) * len + i) * hd + jd;
                    let didx = ((ib * nh + ih) * s + (st + i)) * hd + jd;
                    dst[didx] = src[sidx];
                }
            }
        }
    }
}

/// q,k,v: [b, s, dm] (treated as [b, s, nh*hd]). Returns [b, nh, s, hd].
/// Single-call attention (windowed uses this with s = chunk_len).
fn attention_window(
    q: &CpuTensor, k: &CpuTensor, v: &CpuTensor,
    b: usize, nh: usize, s: usize, hd: usize, scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; b * nh * s * hd];
    for ib in 0..b {
        for ih in 0..nh {
            for is_ in 0..s {
                let mut scores = vec![0.0f32; s];
                let mut max_s = f32::NEG_INFINITY;
                for it in 0..s {
                    let mut dot = 0.0f32;
                    for jd in 0..hd {
                        let qv = q.data[((ib * s + is_) * nh * hd + ih * hd) + jd];
                        let kv = k.data[((ib * s + it) * nh * hd + ih * hd) + jd];
                        dot += qv * kv;
                    }
                    let s_ = dot * scale;
                    scores[it] = s_;
                    if s_ > max_s { max_s = s_; }
                }
                let mut sum = 0.0f32;
                for t in 0..s {
                    scores[t] = (scores[t] - max_s).exp();
                    sum += scores[t];
                }
                let inv = 1.0 / sum;
                for it in 0..s {
                    let w = scores[it] * inv;
                    for jd in 0..hd {
                        let vv = v.data[((ib * s + it) * nh * hd + ih * hd) + jd];
                        out[((ib * nh + ih) * s + is_) * hd + jd] += w * vv;
                    }
                }
            }
        }
    }
    out
}

pub(crate) struct CpuAudioFfn {
    fc1: CpuAudioLinear,
    fc2: CpuAudioLinear,
}

impl CpuAudioFfn {
    pub(crate) fn load(weights: &HashMap<String, RawTensor>, prefix: &str) -> Result<Self> {
        Ok(Self {
            fc1: CpuAudioLinear::load(weights, &format!("{}.fc1", prefix))?,
            fc2: CpuAudioLinear::load(weights, &format!("{}.fc2", prefix))?,
        })
    }

    pub(crate) fn forward(&self, x: &CpuTensor) -> Result<CpuTensor> {
        let mut h = self.fc1.forward(x)?;
        gelu_inplace(&mut h);
        self.fc2.forward(&h)
    }
}

pub(crate) struct CpuAudioLayer {
    sln: CpuAudioLayerNorm,
    attn: CpuAudioAttention,
    fln: CpuAudioLayerNorm,
    ffn: CpuAudioFfn,
}

impl CpuAudioLayer {
    pub(crate) fn load(
        weights: &HashMap<String, RawTensor>,
        prefix: &str,
        nh: usize,
        d_model: usize,
    ) -> Result<Self> {
        Ok(Self {
            sln: CpuAudioLayerNorm::load(weights, &format!("{}.self_attn_layer_norm", prefix), 1e-5)?,
            attn: CpuAudioAttention::load(weights, &format!("{}.self_attn", prefix), nh, d_model)?,
            fln: CpuAudioLayerNorm::load(weights, &format!("{}.final_layer_norm", prefix), 1e-5)?,
            ffn: CpuAudioFfn::load(weights, prefix)?,
        })
    }

    /// x: [b, s, d_model] consumed; returns post-residual h of same shape.
    pub(crate) fn forward(&self, x: CpuTensor, ws: Option<usize>) -> Result<CpuTensor> {
        let normed = self.sln.forward(&x);
        let attn_out = self.attn.forward(&normed, ws)?;
        let mut x1_data = x.data.clone();
        for (a, b) in x1_data.iter_mut().zip(attn_out.data.iter()) { *a += *b; }
        let x1 = CpuTensor::new(x1_data, x.shape.clone());
        let normed2 = self.fln.forward(&x1);
        let ffn_out = self.ffn.forward(&normed2)?;
        let mut x2_data = x1.data;
        for (a, b) in x2_data.iter_mut().zip(ffn_out.data.iter()) { *a += *b; }
        Ok(CpuTensor::new(x2_data, x1.shape))
    }
}

/// Replicate of `gpu_audio_encoder.rs::feo` (line 389-392).
fn feo(ifr: usize) -> usize {
    let f = |l: usize| -> usize { (l - 1) / 2 + 1 };
    f(f(f(ifr)))
}

pub(crate) struct CpuAudioEncoder {
    conv_stem: CpuConvStem,
    layers: Vec<CpuAudioLayer>,
    ln_post: CpuAudioLayerNorm,
    proj1: CpuAudioLinear,
    proj2: CpuAudioLinear,
    config: AudioEncoderConfig,
}

impl CpuAudioEncoder {
    pub(crate) fn load(
        weights: &HashMap<String, RawTensor>,
        prefix: &str,
        config: &AudioEncoderConfig,
    ) -> Result<Self> {
        let dm = config.d_model;
        let nh = config.encoder_attention_heads;
        let mut layers = Vec::with_capacity(config.encoder_layers);
        for i in 0..config.encoder_layers {
            layers.push(CpuAudioLayer::load(weights, &format!("{}.layers.{}", prefix, i), nh, dm)?);
        }
        let ln_post = CpuAudioLayerNorm::load(weights, &format!("{}.ln_post", prefix), 1e-5)?;
        let proj1 = CpuAudioLinear::load(weights, &format!("{}.proj1", prefix))?;
        let proj2 = CpuAudioLinear::load(weights, &format!("{}.proj2", prefix))?;
        let conv_stem = CpuConvStem::load(weights, prefix, config)?;
        Ok(Self { conv_stem, layers, ln_post, proj1, proj2, config: config.clone() })
    }

    /// mel: [n_mels * T_mel] flat. Returns [n_total, output_dim] flat.
    pub(crate) fn forward(&self, mel: &[f32], n_mels: usize, mel_len: usize) -> Result<Vec<f32>> {
        let (conv_data, n_total) = self.conv_stem.forward(mel, n_mels, mel_len)?;
        let dm = self.config.d_model;

        let cs2 = self.config.n_window * 2;
        let tpc = feo(cs2);
        let cpw = self.config.n_window_infer / cs2;
        let ws = tpc * cpw;
        let mut h = CpuTensor::new(conv_data, vec![1, n_total, dm]);
        for layer in &self.layers {
            h = layer.forward(h, Some(ws))?;
        }

        let h = self.ln_post.forward(&h);
        let mut h = self.proj1.forward(&h)?;
        gelu_inplace(&mut h);
        let h = self.proj2.forward(&h)?;
        Ok(h.data)
    }
}
