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
