//! CPU-resident text decoder for Qwen3-ASR.
//!
//! Mirrors the structure of `cudarc_engine.rs` but runs everything on the host CPU:
//! - `gemm` crate handles every matmul, **with `Parallelism::Rayon(0)` forced even on
//!   the m=1 GEMVs** that burn-flex leaves single-threaded (its threshold is m*n*k >= 7M
//!   and decode never reaches that).  This is the single biggest win — without it a
//!   20-core machine spends decode on one core.
//! - `rayon` parallelises every hand-written elementwise/reduction op (rms_norm,
//!   silu_mul, attention) across heads or rows.
//! - All tensors are f32 (Vec<f32>) — x86 has no native f16 SIMD outside
//!   Sapphire/Granite Rapids AVX-512-FP16, so f32 ends up faster than f16 with upcast.
//! - KV cache is pre-allocated per layer (Vec<f32> sized [b, nkvh, max_seq, d]).
//! - Weights are loaded once into `CpuWeight` (Vec<f32>); the embed table is reused
//!   as the lm_head weight (Qwen3 ties them).
//!
//! This module is `#[cfg(feature = "cpu")]` only — the CUDA path is untouched.

use anyhow::Result;
use burn::tensor::TensorData;
use gemm::{gemm, Parallelism};
use rayon::prelude::*;
use std::collections::HashMap;

use crate::config::TextDecoderConfig;

// ═══════════════════════════════════════════════════════════════════════
//  Tensors (host-side, f32)
// ═══════════════════════════════════════════════════════════════════════

pub(crate) struct CpuTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl CpuTensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(data.len(), expected, "CpuTensor len mismatch (shape {:?})", shape);
        Self { data, shape }
    }
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self { data: vec![0.0; n], shape }
    }
    pub fn shape(&self) -> &[usize] { &self.shape }
    pub fn numel(&self) -> usize { self.data.len() }
    pub fn reshape(mut self, shape: Vec<usize>) -> Self {
        assert_eq!(self.numel(), shape.iter().product::<usize>());
        self.shape = shape; self
    }
}

pub(crate) struct CpuWeight {
    pub data: Vec<f32>,
    pub rows: usize,   // = out_features (N)
    pub cols: usize,   // = in_features  (K)
}

// ═══════════════════════════════════════════════════════════════════════
//  Matmul: y = x @ W^T  with forced rayon parallelism
// ═══════════════════════════════════════════════════════════════════════
//
// Layout: x [m, k] row-major, W [n, k] row-major (PyTorch nn.Linear convention),
// y [m, n] row-major.  Computing y = x @ W^T means each y[i, j] = sum_k x[i, k] * W[j, k].
//
// gemm's stride API: dst_cs = stride for j+1, dst_rs = stride for i+1.  For row-major C[m,n]
// element (i, j) is at i*n + j, so cs=1, rs=n.  Same for x.  W^T conceptually is [k, n] with
// element (i, j) = W[j, i]; since W is row-major [n, k] (W[j, i] at j*k + i), B's cs = k, rs = 1.

fn linear(x: &CpuTensor, w: &CpuWeight) -> CpuTensor {
    let nd = x.shape.len();
    let m: usize = x.shape[..nd - 1].iter().product();
    let k = x.shape[nd - 1];
    let n = w.rows;
    assert_eq!(k, w.cols, "linear K mismatch: x last={} vs W cols={}", k, w.cols);
    let mut out = vec![0.0f32; m * n];
    gemm_row_major(&mut out, &x.data, w, m, 0.0);
    let mut out_shape = x.shape.clone();
    out_shape[nd - 1] = n;
    CpuTensor::new(out, out_shape)
}

fn linear_accum(out: &mut CpuTensor, x: &CpuTensor, w: &CpuWeight) {
    let nd = x.shape.len();
    let m: usize = x.shape[..nd - 1].iter().product();
    let k = x.shape[nd - 1];
    let n = w.rows;
    assert_eq!(k, w.cols);
    assert_eq!(out.numel(), m * n);
    gemm_row_major(&mut out.data, &x.data, w, m, 1.0);
}

// y[i, j] = sum_k x[i, k] * W[j, k] + beta * y[i, j].  Row-major throughout.
// Forces gemm to use `Parallelism::Rayon(0)` (all cores) even for m=1 GEMVs, which is
// the whole point of this engine — burn-flex's internal threshold leaves decode
// single-threaded otherwise.
fn gemm_row_major(out: &mut [f32], x: &[f32], w: &CpuWeight, m: usize, beta: f32) {
    let n = w.rows;
    let k = w.cols;
    unsafe {
        gemm(
            m, n, k,
            out.as_mut_ptr(),
            1,                  // dst_cs (column stride for j+1, since C is row-major)
            n as isize,         // dst_rs (row stride for i+1)
            beta != 0.0,
            x.as_ptr(),
            1,                  // lhs_cs
            k as isize,         // lhs_rs
            w.data.as_ptr(),
            k as isize,         // rhs_cs (B is W^T; j+1 advances by k in W's row-major layout)
            1,                  // rhs_rs
            beta,
            1.0,
            false, false, false,
            Parallelism::Rayon(0),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Elementwise / reduction ops (rayon-parallel by row)
// ═══════════════════════════════════════════════════════════════════════

/// out[i, j] = x[i, j] * w[j] / sqrt(mean(x[i, :]^2) + eps)
pub fn rms_norm(x: &CpuTensor, w: &[f32], eps: f32) -> CpuTensor {
    let nd = x.shape.len();
    let last = x.shape[nd - 1];
    let outer: usize = x.shape[..nd - 1].iter().product();
    assert_eq!(w.len(), last);
    let mut out = vec![0.0f32; outer * last];
    out.par_chunks_mut(last)
        .zip(x.data.par_chunks(last))
        .for_each(|(o, xrow)| {
            let mut ss = 0.0f64;
            for &v in xrow { ss += (v as f64) * (v as f64); }
            let inv_rms = 1.0 / ((ss / last as f64 + eps as f64).sqrt() as f32);
            for j in 0..last {
                o[j] = xrow[j] * inv_rms * w[j];
            }
        });
    CpuTensor::new(out, x.shape.clone())
}

/// residual += add_in (in place); returns rms_norm(residual, w).  Saves a pass over memory.
pub fn add_residual_rms_norm(residual: &mut CpuTensor, add_in: &CpuTensor, w: &[f32], eps: f32) -> CpuTensor {
    let nd = residual.shape.len();
    let last = residual.shape[nd - 1];
    let outer: usize = residual.shape[..nd - 1].iter().product();
    let mut out = vec![0.0f32; outer * last];
    residual.data
        .par_chunks_mut(last)
        .zip(add_in.data.par_chunks(last))
        .zip(out.par_chunks_mut(last))
        .for_each(|((r, a), o)| {
            let mut ss = 0.0f64;
            for j in 0..last {
                let v = r[j] + a[j];
                r[j] = v;
                ss += (v as f64) * (v as f64);
            }
            let inv_rms = 1.0 / ((ss / last as f64 + eps as f64).sqrt() as f32);
            for j in 0..last {
                o[j] = r[j] * inv_rms * w[j];
            }
        });
    CpuTensor::new(out, residual.shape.clone())
}

/// out = silu(gate) * up where gu = [gate | up] along last dim.  gu: [outer, 2*inter] → out: [outer, inter].
pub fn silu_mul_split(gu: &CpuTensor) -> CpuTensor {
    let nd = gu.shape.len();
    let two_inter = gu.shape[nd - 1];
    let inter = two_inter / 2;
    let outer: usize = gu.shape[..nd - 1].iter().product();
    let mut out = vec![0.0f32; outer * inter];
    out.par_chunks_mut(inter)
        .zip(gu.data.par_chunks(two_inter))
        .for_each(|(o, row)| {
            let (gate, up) = row.split_at(inter);
            for j in 0..inter {
                let g = gate[j];
                let sig = 1.0 / (1.0 + (-g).exp());
                o[j] = g * sig * up[j];
            }
        });
    let mut shape = gu.shape.clone();
    shape[nd - 1] = inter;
    CpuTensor::new(out, shape)
}

/// Embedding lookup.  ids: [n], table: [vocab, d] → out [n, d].
pub fn embed_lookup(table: &CpuWeight, ids: &[i64]) -> CpuTensor {
    let n = ids.len();
    let d = table.cols;
    let mut out = vec![0.0f32; n * d];
    out.par_chunks_mut(d)
        .zip(ids.par_iter())
        .for_each(|(o, &id)| {
            let src = &table.data[(id as usize) * d..(id as usize + 1) * d];
            o.copy_from_slice(src);
        });
    CpuTensor::new(out, vec![n, d])
}

/// Argmax of a flat slice.
pub fn argmax(x: &[f32]) -> i32 {
    // Rayon reduce: find max + its index.  For vocab~152k this is cheap.
    let (idx, _) = x
        .par_iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    idx as i32
}

// ═══════════════════════════════════════════════════════════════════════
//  Rotary embedding (in-place on Q or K head row)
// ═══════════════════════════════════════════════════════════════════════
//
// For a head row of length d, with `cos[i]` and `sin[i]` (i in [0, d)), the standard
// "half-rotation" Qwen formulation is:
//   for i < d/2:        out[i] = x[i]*cos[i] - x[i+d/2]*sin[i]
//   for i in [d/2, d):  out[i] = x[i]*cos[i] + x[i-d/2]*sin[i]

#[inline]
fn apply_rotary_row(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let d = x.len();
    let half = d / 2;
    // Cache halves before overwrite.
    let mut tmp = [0.0f32; 256]; // head_dim <= 256 covers Qwen3-ASR (hd=128)
    debug_assert!(d <= tmp.len());
    tmp[..d].copy_from_slice(x);
    for i in 0..half {
        x[i]        = tmp[i]        * cos[i]        - tmp[i + half] * sin[i];
        x[i + half] = tmp[i + half] * cos[i + half] + tmp[i]        * sin[i + half];
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Fused QKV extract + (Q,K) norm + rotary + KV cache write
// ═══════════════════════════════════════════════════════════════════════
//
// qkv: [b, s, total_cols] where total_cols = q_dim + 2*kv_dim
//   q_dim = nqh * hd,  kv_dim = nkvh * hd
// Outputs:
//   q_out: [b, nqh, s, hd]   (norm+rotary applied)
//   k_cache, v_cache: [b, nkvh, max_seq, hd]  at rows [start..start+s)
//     - K: norm+rotary; V: raw copy
pub fn qkv_extract_qkv_norm_rotary_cache(
    qkv: &CpuTensor,
    qn_w: &[f32], kn_w: &[f32],
    cos_table: &[f32], sin_table: &[f32],
    k_cache: &mut [f32], v_cache: &mut [f32],
    b: usize, nqh: usize, nkvh: usize, hd: usize,
    q_dim: usize, kv_dim: usize,
    max_seq: usize, start: usize, pos_offset: usize, eps: f32,
) -> CpuTensor {
    let s = qkv.shape[1];
    let total_cols = qkv.shape[2];
    let mut q_out = vec![0.0f32; b * nqh * s * hd];

    // Parallelise over (b, s) — each token is independent.  Within a token we sequentially
    // handle all heads (nqh + nkvh ≤ 24 for Qwen3-ASR, not worth nested rayon).
    //
    // We need disjoint &mut access to the per-(b,s) slices of q_out, k_cache, v_cache.  Use
    // par_chunks_mut on q_out and index k_cache/v_cache by computed offset (the writes for
    // different (b, s) never overlap because of the (start + is) offset into max_seq).
    let q_per_token = nqh * hd;
    // Pre-split for parallel iteration.
    q_out.par_chunks_mut(q_per_token).enumerate().for_each(|(token_idx, q_dst)| {
        let ib = token_idx / s;
        let is = token_idx % s;
        let cs_row = pos_offset + is;
        let cos_row = &cos_table[cs_row * hd..(cs_row + 1) * hd];
        let sin_row = &sin_table[cs_row * hd..(cs_row + 1) * hd];

        let row_base = (ib * s + is) * total_cols;
        let qkv_row = &qkv.data[row_base..row_base + total_cols];
        let q_src = &qkv_row[..q_dim];
        let k_src = &qkv_row[q_dim..q_dim + kv_dim];
        let v_src = &qkv_row[q_dim + kv_dim..];

        // Q: per head, RMSNorm(qn_w) → rotary → write to q_dst
        for h in 0..nqh {
            let head_in = &q_src[h * hd..(h + 1) * hd];
            let head_out = &mut q_dst[h * hd..(h + 1) * hd];
            let mut ss = 0.0f64;
            for &v in head_in { ss += (v as f64) * (v as f64); }
            let inv_rms = 1.0 / ((ss / hd as f64 + eps as f64).sqrt() as f32);
            for j in 0..hd { head_out[j] = head_in[j] * inv_rms * qn_w[j]; }
            apply_rotary_row(head_out, cos_row, sin_row);
        }

        // K: per kv-head, RMSNorm(kn_w) → rotary → write to k_cache[ib, h, start+is, :]
        // V: raw copy to v_cache.
        // SAFETY: k_cache/v_cache are passed in as &mut [f32].  Different (ib, is) write
        // disjoint slots in max_seq, and different (ib, h) write disjoint heads.  Per-token
        // parallel iteration above is the only outer parallelism.
        let k_cache_ptr = k_cache.as_ptr() as *mut f32;
        let v_cache_ptr = v_cache.as_ptr() as *mut f32;
        for h in 0..nkvh {
            let k_in = &k_src[h * hd..(h + 1) * hd];
            let v_in = &v_src[h * hd..(h + 1) * hd];
            let cache_idx = ((ib * nkvh + h) * max_seq + (start + is)) * hd;
            unsafe {
                let k_dst = std::slice::from_raw_parts_mut(k_cache_ptr.add(cache_idx), hd);
                let v_dst = std::slice::from_raw_parts_mut(v_cache_ptr.add(cache_idx), hd);
                let mut ss = 0.0f64;
                for &v in k_in { ss += (v as f64) * (v as f64); }
                let inv_rms = 1.0 / ((ss / hd as f64 + eps as f64).sqrt() as f32);
                for j in 0..hd { k_dst[j] = k_in[j] * inv_rms * kn_w[j]; }
                apply_rotary_row(k_dst, cos_row, sin_row);
                v_dst.copy_from_slice(v_in);
            }
        }
    });

    CpuTensor::new(q_out, vec![b, nqh, s, hd])
}

// ═══════════════════════════════════════════════════════════════════════
//  Fused GQA attention (decode-only, s_q = 1)
// ═══════════════════════════════════════════════════════════════════════
//
// q: [b, nqh, 1, hd]
// k_cache, v_cache: [b, nkvh, max_seq, hd]  (valid rows [0..cur_len))
// out: [b, nqh, 1, hd]
//
// One (b, qh) per rayon job: compute scores = q · K^T, softmax, out = attn · V.
// Done in f32 throughout; cheap vs prefill because cur_len is small (decode-only).
pub fn fused_gqa_decode(
    q: &CpuTensor,
    k_cache: &[f32], v_cache: &[f32],
    b: usize, nqh: usize, nkvh: usize, max_seq: usize, hd: usize, cur_len: usize,
    scale: f32,
) -> CpuTensor {
    let n_rep = nqh / nkvh;
    let mut out = vec![0.0f32; b * nqh * hd];
    out.par_chunks_mut(hd).enumerate().for_each(|(idx, out_head)| {
        let ib = idx / nqh;
        let qh = idx % nqh;
        let kh = qh / n_rep;
        let q_row = &q.data[(ib * nqh + qh) * hd..(ib * nqh + qh + 1) * hd];
        let k_base = (ib * nkvh + kh) * max_seq * hd;
        let v_base = (ib * nkvh + kh) * max_seq * hd;

        // Compute scores[0..cur_len] in a scratch buffer.
        let mut scores = vec![0.0f32; cur_len];
        let mut max_score = f32::NEG_INFINITY;
        for t in 0..cur_len {
            let k_row = &k_cache[k_base + t * hd..k_base + (t + 1) * hd];
            let mut dot = 0.0f32;
            for j in 0..hd { dot += q_row[j] * k_row[j]; }
            let s = dot * scale;
            scores[t] = s;
            if s > max_score { max_score = s; }
        }
        // softmax
        let mut sum_exp = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_score).exp();
            sum_exp += *s;
        }
        let inv_sum = 1.0 / sum_exp;
        // out_head[j] = sum_t scores[t] * V[t, j] * inv_sum
        for j in 0..hd { out_head[j] = 0.0; }
        for t in 0..cur_len {
            let v_row = &v_cache[v_base + t * hd..v_base + (t + 1) * hd];
            let w = scores[t] * inv_sum;
            for j in 0..hd { out_head[j] += w * v_row[j]; }
        }
    });
    CpuTensor::new(out, vec![b, nqh, 1, hd])
}

// ═══════════════════════════════════════════════════════════════════════
//  Decoder layer
// ═══════════════════════════════════════════════════════════════════════

pub(crate) struct CpuDecoderLayer {
    pub iln_w: Vec<f32>,
    pub pln_w: Vec<f32>,
    pub qn_w: Vec<f32>,
    pub kn_w: Vec<f32>,
    pub qkv_w: CpuWeight,
    pub o_w: CpuWeight,
    pub gu_w: CpuWeight,   // [2*inter, hidden]
    pub dp_w: CpuWeight,   // [hidden, inter]
    pub nqh: usize, pub nkvh: usize, pub hd: usize, pub eps: f32,
}

impl CpuDecoderLayer {
    pub fn load(
        weights: &HashMap<String, TensorData>,
        prefix: &str,
        cfg: &TextDecoderConfig,
    ) -> Result<Self> {
        Ok(Self {
            iln_w: load_vec_f32(weights, &format!("{}.input_layernorm.weight", prefix))?,
            pln_w: load_vec_f32(weights, &format!("{}.post_attention_layernorm.weight", prefix))?,
            qn_w: load_vec_f32(weights, &format!("{}.self_attn.q_norm.weight", prefix))?,
            kn_w: load_vec_f32(weights, &format!("{}.self_attn.k_norm.weight", prefix))?,
            qkv_w: load_fused_qkv(weights, &format!("{}.self_attn", prefix))?,
            o_w: load_weight(weights, &format!("{}.self_attn.o_proj.weight", prefix))?,
            gu_w: load_fused_gate_up(weights, &format!("{}.mlp", prefix))?,
            dp_w: load_weight(weights, &format!("{}.mlp.down_proj.weight", prefix))?,
            nqh: cfg.num_attention_heads,
            nkvh: cfg.num_key_value_heads,
            hd: cfg.head_dim,
            eps: cfg.rms_norm_eps as f32,
        })
    }

    /// x: [b, s, hidden] consumed; returns h (post all residuals).
    pub fn forward(
        &self,
        x: CpuTensor,
        cos_table: &[f32], sin_table: &[f32],
        kv: &mut CpuKvCache, layer_idx: usize,
        kv_start: usize, use_causal: bool,
    ) -> CpuTensor {
        let b = x.shape[0]; let s = x.shape[1];

        // 1. Input RMSNorm
        let normed = rms_norm(&x, &self.iln_w, self.eps);

        // 2. Fused QKV linear
        let qkv = linear(&normed, &self.qkv_w);
        drop(normed);
        let q_dim = self.nqh * self.hd;
        let kv_dim = self.nkvh * self.hd;

        // 3. Extract Q+K+V, norm+rotary Q&K, write K/V to cache.
        let q = qkv_extract_qkv_norm_rotary_cache(
            &qkv, &self.qn_w, &self.kn_w, cos_table, sin_table,
            &mut kv.k[layer_idx], &mut kv.v[layer_idx],
            b, self.nqh, self.nkvh, self.hd, q_dim, kv_dim,
            kv.max_seq, kv_start, kv_start, self.eps,
        );
        drop(qkv);
        let cur_len = kv_start + s;

        // 4. Attention.  Decode-only path (s == 1) uses fused_gqa_decode; prefill (s > 1)
        //    materialises full attention scores via two matmuls.
        let attn_out = if s == 1 {
            let scale = 1.0f32 / (self.hd as f32).sqrt();
            fused_gqa_decode(&q, &kv.k[layer_idx], &kv.v[layer_idx],
                b, self.nqh, self.nkvh, kv.max_seq, self.hd, cur_len, scale)
        } else {
            prefill_attention(&q, &kv.k[layer_idx], &kv.v[layer_idx],
                b, self.nqh, self.nkvh, kv.max_seq, self.hd, cur_len, use_causal)
        };

        // 5. O projection with residual add.  attn_out is laid out as [b, s, nqh, hd]
        //    regardless of whether decode (fused_gqa_decode) or prefill (prefill_attention)
        //    produced it, so we can reshape directly to [b, s, nqh*hd] — no swap needed.
        let attn_flat = attn_out.reshape(vec![b, s, self.nqh * self.hd]);
        let mut h = x;
        linear_accum(&mut h, &attn_flat, &self.o_w);
        drop(attn_flat);

        // 6. Post-attn RMSNorm
        let normed2 = rms_norm(&h, &self.pln_w, self.eps);

        // 7. Gate-up linear → SiLU·up
        let gu = linear(&normed2, &self.gu_w);
        drop(normed2);
        let activated = silu_mul_split(&gu);
        drop(gu);

        // 8. Down projection with residual add
        linear_accum(&mut h, &activated, &self.dp_w);
        h
    }
}

// Prefill-path attention: materialise scores [b, nqh, s, cur_len], softmax, out = attn · V.
// q is laid out as [b, s, nqh, hd] (the natural per-token output of qkv_extract).
// Returns out in the same [b, s, nqh, hd] layout so the caller can reshape directly to
// [b, s, nqh*hd] for the O projection without a swap.
// One rayon job per (b, qh).
fn prefill_attention(
    q: &CpuTensor,
    k_cache: &[f32], v_cache: &[f32],
    b: usize, nqh: usize, nkvh: usize, max_seq: usize, hd: usize, cur_len: usize,
    causal: bool,
) -> CpuTensor {
    // q.shape is logically [b, nqh, s, hd] but the bytes are actually [b, s, nqh, hd].
    let s = q.shape[2];
    let n_rep = nqh / nkvh;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let mut out = vec![0.0f32; b * s * nqh * hd];

    // Each rayon job handles one (b, qh) — it owns a strided view of the output (length s*hd).
    // We compute scores [s, cur_len], softmax, then matmul with V to produce out[ib, :, qh, :].
    (0..b * nqh).into_par_iter().for_each(|idx| {
        let ib = idx / nqh;
        let qh = idx % nqh;
        let kh = qh / n_rep;
        let k_base = (ib * nkvh + kh) * max_seq * hd;
        let v_base = (ib * nkvh + kh) * max_seq * hd;

        // Pre-extract Q for this (ib, qh) across all s tokens into a small contiguous buffer.
        let mut q_qh = vec![0.0f32; s * hd];
        for i in 0..s {
            let src = ((ib * s + i) * nqh + qh) * hd;
            q_qh[i * hd..(i + 1) * hd].copy_from_slice(&q.data[src..src + hd]);
        }

        let mut scores = vec![0.0f32; s * cur_len];
        // scores[i, t] = q[i, :] · K[t, :] * scale, with causal mask: positions > i + (cur_len - s) masked.
        for i in 0..s {
            let qi = &q_qh[i * hd..(i + 1) * hd];
            let limit = if causal { i + (cur_len - s) + 1 } else { cur_len };
            for t in 0..cur_len {
                if t >= limit {
                    scores[i * cur_len + t] = f32::NEG_INFINITY;
                } else {
                    let kt = &k_cache[k_base + t * hd..k_base + (t + 1) * hd];
                    let mut dot = 0.0f32;
                    for j in 0..hd { dot += qi[j] * kt[j]; }
                    scores[i * cur_len + t] = dot * scale;
                }
            }
        }
        // softmax per row
        for i in 0..s {
            let row = &mut scores[i * cur_len..(i + 1) * cur_len];
            let mut mx = f32::NEG_INFINITY;
            for &v in row.iter() { if v > mx { mx = v; } }
            let mut sum = 0.0f32;
            for v in row.iter_mut() { *v = (*v - mx).exp(); sum += *v; }
            let inv = 1.0 / sum;
            for v in row.iter_mut() { *v *= inv; }
        }
        // out[ib, i, qh, j] = sum_t scores[i, t] * V[t, j]
        let out_ptr = out.as_ptr() as *mut f32;
        for i in 0..s {
            let dst_off = ((ib * s + i) * nqh + qh) * hd;
            // SAFETY: each (ib, qh) writes a disjoint stride of slots; no overlap with other (ib', qh').
            unsafe {
                let out_i = std::slice::from_raw_parts_mut(out_ptr.add(dst_off), hd);
                for j in 0..hd { out_i[j] = 0.0; }
                let row = &scores[i * cur_len..(i + 1) * cur_len];
                for t in 0..cur_len {
                    let w = row[t];
                    if w == 0.0 { continue; }
                    let vt = &v_cache[v_base + t * hd..v_base + (t + 1) * hd];
                    for j in 0..hd { out_i[j] += w * vt[j]; }
                }
            }
        }
    });

    // Logical shape [b, nqh, s, hd] but actual bytes are [b, s, nqh, hd] — the caller knows
    // to reshape to [b, s, nqh*hd] without swap_dims_12.
    CpuTensor::new(out, vec![b, nqh, s, hd])
}

/// Swap dims 1 and 2 of a 4D tensor: [d0, d1, d2, d3] → [d0, d2, d1, d3].
pub fn swap_dims_12(x: &CpuTensor) -> CpuTensor {
    assert_eq!(x.shape.len(), 4);
    let (d0, d1, d2, d3) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
    let mut out = vec![0.0f32; x.numel()];
    for i0 in 0..d0 {
        for i2 in 0..d2 {
            for i1 in 0..d1 {
                let src_off = ((i0 * d1 + i1) * d2 + i2) * d3;
                let dst_off = ((i0 * d2 + i2) * d1 + i1) * d3;
                out[dst_off..dst_off + d3].copy_from_slice(&x.data[src_off..src_off + d3]);
            }
        }
    }
    CpuTensor::new(out, vec![d0, d2, d1, d3])
}

// ═══════════════════════════════════════════════════════════════════════
//  KV cache
// ═══════════════════════════════════════════════════════════════════════

pub(crate) struct CpuKvCache {
    pub k: Vec<Vec<f32>>,  // per layer: [b, nkvh, max_seq, hd]
    pub v: Vec<Vec<f32>>,
    pub cur_len: usize,
    pub max_seq: usize,
    pub b: usize,
    pub nkvh: usize,
    pub hd: usize,
}

impl CpuKvCache {
    pub fn new(num_layers: usize, b: usize, nkvh: usize, max_seq: usize, hd: usize) -> Self {
        let cap = b * nkvh * max_seq * hd;
        let mut k = Vec::with_capacity(num_layers);
        let mut v = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k.push(vec![0.0; cap]);
            v.push(vec![0.0; cap]);
        }
        Self { k, v, cur_len: 0, max_seq, b, nkvh, hd }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Text decoder
// ═══════════════════════════════════════════════════════════════════════

pub(crate) struct CpuTextDecoder {
    pub embed_table: CpuWeight,   // [vocab, hidden] (reused as lm_head)
    pub layers: Vec<CpuDecoderLayer>,
    pub norm_w: Vec<f32>,         // [hidden]
    pub eps: f32,
    pub config: TextDecoderConfig,
}

impl CpuTextDecoder {
    pub fn load(weights: &HashMap<String, TensorData>, prefix: &str, config: &TextDecoderConfig) -> Result<Self> {
        let embed_table = load_weight(weights, &format!("{}.embed_tokens.weight", prefix))?;
        let norm_w = load_vec_f32(weights, &format!("{}.norm.weight", prefix))?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(CpuDecoderLayer::load(weights, &format!("{}.layers.{}", prefix, i), config)?);
        }
        Ok(Self { embed_table, layers, norm_w, eps: config.rms_norm_eps as f32, config: config.clone() })
    }

    /// Embed ids into [n, hidden] tensor.
    pub fn embed_ids(&self, ids: &[i64]) -> CpuTensor {
        embed_lookup(&self.embed_table, ids)
    }

    /// Forward pass.
    /// hs: [1, sl, hidden].  cos/sin_table: full [total_positions, hd] tables.
    /// kv_start: how many positions already in cache.
    /// Returns logits as [1, out_sl, vocab] (out_sl = 1 if llo).
    pub fn forward(
        &self,
        hs: CpuTensor,
        cos_table: &[f32], sin_table: &[f32],
        kv: &mut CpuKvCache, kv_start: usize, use_causal: bool, llo: bool,
    ) -> CpuTensor {
        let sl = hs.shape[1];
        let mut h = hs;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(h, cos_table, sin_table, kv, i, kv_start, use_causal);
        }
        kv.cur_len = kv_start + sl;

        // Final RMSNorm
        let h = rms_norm(&h, &self.norm_w, self.eps);

        // Low-latency: keep only last token for prefill
        let h = if llo && sl > 1 {
            let hidden = h.shape[2];
            let last_off = (sl - 1) * hidden;
            let last_data = h.data[last_off..last_off + hidden].to_vec();
            CpuTensor::new(last_data, vec![1, 1, hidden])
        } else {
            h
        };

        // lm_head (shared with embed_table)
        linear(&h, &self.embed_table)
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  MRoPE cos/sin precompute (same shape as cudarc_engine version, but f32 only)
// ═══════════════════════════════════════════════════════════════════════

pub(crate) fn compute_mrope_cos_sin(
    pos: &[Vec<i64>; 3], hd: usize, rt: f64, ms: &[usize], il: bool,
) -> (Vec<f32>, Vec<f32>) {
    let hh = hd / 2;
    let sl = pos[0].len();
    let inv: Vec<f64> = (0..hh).map(|i| 1.0 / rt.powf(2.0 * i as f64 / hd as f64)).collect();
    let dm = if il { build_interleaved_dim_map(ms, hh) } else { build_contiguous_dim_map(ms, hh) };
    let mut cv = vec![0.0f32; sl * hd];
    let mut sv = vec![0.0f32; sl * hd];
    for t in 0..sl {
        for j in 0..hh {
            let a = pos[dm[j]][t] as f64 * inv[j];
            cv[t * hd + j] = a.cos() as f32;
            sv[t * hd + j] = a.sin() as f32;
            cv[t * hd + j + hh] = a.cos() as f32;
            sv[t * hd + j + hh] = a.sin() as f32;
        }
    }
    (cv, sv)
}

fn build_contiguous_dim_map(s: &[usize], t: usize) -> Vec<usize> {
    let mut m = Vec::with_capacity(t);
    for (d, &sz) in s.iter().enumerate() { for _ in 0..sz { if m.len() >= t { break; } m.push(d); } }
    while m.len() < t { m.push(s.len() - 1); } m
}

fn build_interleaved_dim_map(s: &[usize], t: usize) -> Vec<usize> {
    let nd = s.len(); let mut m = Vec::with_capacity(t); let mut c = vec![0usize; nd];
    while m.len() < t {
        let pv = m.len();
        for d in 0..nd {
            if m.len() >= t { break; }
            if c[d] < s[d] { m.push(d); c[d] += 1; }
        }
        if m.len() == pv { break; }
    } m
}

// ═══════════════════════════════════════════════════════════════════════
//  Weight loading helpers
// ═══════════════════════════════════════════════════════════════════════

fn load_f32_vec(weights: &HashMap<String, TensorData>, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let td = weights.get(name).ok_or_else(|| anyhow::anyhow!("weight not found: {}", name))?;
    let shape = td.shape.to_vec();
    let data: Vec<f32> = match td.dtype {
        burn::tensor::DType::F32 => td.to_vec::<f32>().map_err(|e| anyhow::anyhow!("dtype mismatch for {}: {:?}", name, e))?,
        burn::tensor::DType::F16 => td.to_vec::<half::f16>()
            .map_err(|e| anyhow::anyhow!("dtype mismatch for {}: {:?}", name, e))?
            .into_iter().map(|v| v.to_f32()).collect(),
        _ => anyhow::bail!("unsupported dtype {:?} for {}", td.dtype, name),
    };
    Ok((data, shape))
}

fn load_vec_f32(weights: &HashMap<String, TensorData>, name: &str) -> Result<Vec<f32>> {
    let (data, _) = load_f32_vec(weights, name)?;
    Ok(data)
}

fn load_weight(weights: &HashMap<String, TensorData>, name: &str) -> Result<CpuWeight> {
    let (data, shape) = load_f32_vec(weights, name)?;
    assert_eq!(shape.len(), 2, "weight {} should be 2D", name);
    Ok(CpuWeight { data, rows: shape[0], cols: shape[1] })
}

/// Load Q+K+V projections and concatenate into a single [q_dim + 2*kv_dim, hidden] matrix.
fn load_fused_qkv(weights: &HashMap<String, TensorData>, prefix: &str) -> Result<CpuWeight> {
    let (qw, qs) = load_f32_vec(weights, &format!("{}.q_proj.weight", prefix))?;
    let (kw, ks) = load_f32_vec(weights, &format!("{}.k_proj.weight", prefix))?;
    let (vw, vs) = load_f32_vec(weights, &format!("{}.v_proj.weight", prefix))?;
    let q_dim = qs[0]; let kv_dim = ks[0]; let hidden = qs[1];
    assert_eq!(ks[1], hidden); assert_eq!(vs[1], hidden);
    let mut fused = Vec::with_capacity((q_dim + 2 * kv_dim) * hidden);
    fused.extend_from_slice(&qw);
    fused.extend_from_slice(&kw);
    fused.extend_from_slice(&vw);
    Ok(CpuWeight { data: fused, rows: q_dim + 2 * kv_dim, cols: hidden })
}

/// Load gate_proj and up_proj, concatenate into [2*inter, hidden] matrix.
fn load_fused_gate_up(weights: &HashMap<String, TensorData>, prefix: &str) -> Result<CpuWeight> {
    let (gw, gs) = load_f32_vec(weights, &format!("{}.gate_proj.weight", prefix))?;
    let (uw, us) = load_f32_vec(weights, &format!("{}.up_proj.weight", prefix))?;
    let inter = gs[0]; let hidden = gs[1];
    assert_eq!(us[0], inter); assert_eq!(us[1], hidden);
    let mut fused = Vec::with_capacity(2 * inter * hidden);
    fused.extend_from_slice(&gw);
    fused.extend_from_slice(&uw);
    Ok(CpuWeight { data: fused, rows: 2 * inter, cols: hidden })
}
