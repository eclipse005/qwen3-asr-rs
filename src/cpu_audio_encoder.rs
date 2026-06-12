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
use crate::cpu_engine::{CpuTensor, CpuWeight};
use crate::raw_tensor::RawTensor;

// ─── Linear + LayerNorm primitives ─────────────────────────────────

/// Local copy of `cpu_engine::linear` (which is private). Mirrors its body
/// exactly: x [m, k] @ W^T [n, k] → [m, n], with m=1 hand-written GEMV path.
fn linear(x: &CpuTensor, w: &CpuWeight) -> CpuTensor {
    let nd = x.shape.len();
    let m: usize = x.shape[..nd - 1].iter().product();
    let k = x.shape[nd - 1];
    let n = w.rows;
    debug_assert_eq!(k, w.cols, "linear K mismatch: x last={} vs W cols={}", k, w.cols);
    let mut out_shape = x.shape.clone();
    out_shape[nd - 1] = n;
    let mut out = vec![0.0f32; m * n];
    if m == 1 {
        // Hand-written GEMV for the m=1 case (lifts burn-flex's m=1 perf cliff).
        for j in 0..n {
            let w_row = &w.data[j * k..(j + 1) * k];
            let mut acc = 0.0f32;
            for i in 0..k { acc += x.data[i] * w_row[i]; }
            out[j] = acc;
        }
        return CpuTensor::new(out, out_shape);
    }
    gemm_row_major(&mut out, &x.data, w, m, 0.0);
    CpuTensor::new(out, out_shape)
}

fn gemm_row_major(out: &mut [f32], x: &[f32], w: &CpuWeight, m: usize, beta: f32) {
    let n = w.rows;
    let k = w.cols;
    unsafe {
        gemm(
            m, n, k,
            out.as_mut_ptr(),
            1,
            n as isize,
            beta != 0.0,
            x.as_ptr(),
            1,
            k as isize,
            w.data.as_ptr(),
            k as isize,
            1,
            beta,
            1.0,
            false, false, false,
            Parallelism::Rayon(0),
        );
    }
}

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

        // Add bias and GELU. out is [c_out, col_count] but we want [b, c_out, h_out, w_out].
        let mut out4d = vec![0.0f32; b * c_out * h_out * w_out];
        for ib in 0..b {
            for ic_in in 0..c_in {
                for ho in 0..h_out {
                    for wo in 0..w_out {
                        let col = ((ib * c_in + ic_in) * h_out + ho) * w_out + wo;
                        for oc in 0..c_out {
                            let v = out[oc * col_count + col] + w_b[oc];
                            out4d[((ib * c_out + oc) * h_out + ho) * w_out + wo] = v;
                        }
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
