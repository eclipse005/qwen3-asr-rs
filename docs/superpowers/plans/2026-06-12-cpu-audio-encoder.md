# CPU Audio Encoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the CPU-side audio encoder so that `Backend::Cpu` can run end-to-end `transcribe()` and produce a `TranscribeResult` whose `text` field is byte-identical to what `Backend::Cuda` produces on the same fixture.

**Architecture:** New module `src/cpu_audio_encoder.rs` mirrors `src/gpu_audio_encoder.rs` 1:1. f32 throughout (no f16 SIMD). Mel is computed on CPU (existing `mel.rs`). Conv2d uses im2col + `gemm` crate. LayerNorm / GELU / windowed-attention / FFN are hand-written f32 loops parallelised by rayon. Weights upcast from f16 safetensors to f32 at load time. The encoder output is a `Vec<f32>` of shape `[n_total, output_dim]` and feeds the existing `CpuTextDecoder` unchanged.

**Tech Stack:** Rust 2021, `gemm` crate (existing, used by `cpu_engine.rs`), `rayon` (existing), `half` (f16 → f32 upcast only, no SIMD), existing `cpu_engine::CpuTensor` / `CpuWeight` types.

**Spec:** `docs/superpowers/specs/2026-06-12-cpu-audio-encoder-design.md` (commit 91f85ba)

**Key invariants:**
- f32 throughout; do not introduce f16 SIMD intrinsics.
- All f32 ops parallelised with rayon (same idiom as `cpu_engine.rs`).
- `Linear` for `m > 1` uses the `gemm` crate via `cpu_engine::linear()`; for `m == 1` it uses `cpu_engine::linear_gemv()` (existing).
- `LayerNorm` eps is `1e-5` (per `gpu_audio_encoder.rs:171`, not the text decoder's `1e-6`).
- `ws` for windowed attention is `feo(cs2) * (n_window_infer / cs2)` where `cs2 = n_window * 2` (per `gpu_audio_encoder.rs:341-344`).
- Sinusoidal PE: `pe[p, i] = sin(p * exp(-i * ln(10000) / (half - 1)))`, `pe[p, half+i] = cos(...)` for `half = d_model / 2` (per `gpu_audio_encoder.rs:220-228`).

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/cpu_audio_encoder.rs` | **Create** | `CpuConvStem`, `CpuAudioAttention`, `CpuAudioFfn`, `CpuAudioLayer`, `CpuAudioEncoder`, f32 LN, f32 GELU, im2col, f32 PE, f32 windowed attention. |
| `src/lib.rs` | Modify | Add `#[cfg(feature = "cpu")] mod cpu_audio_encoder;` |
| `src/inference.rs` | Modify | Replace `encode_audio_cpu` body with real impl; add `cpu_audio_encoder: Option<CpuAudioEncoder>` field; load it in `AsrInferenceInner::new`. |
| `tests/cpu_transcribe.rs` | **Create** | `test_cpu_q06_sample1` (#[ignore]) comparing CPU transcribe text to fixture. |
| `tests/fixtures/expected/gpu_sample1.txt` | **Create** (manual step) | CUDA ground truth text for `sample1.wav`. |
| `ROADMAP.md` | Modify | §3.2 update, §4.1 P0 add CPU test, §1.1 tree add node, §0 wording update. |

---

## Task 1: Verify environment and grab CUDA ground truth

**Files:**
- Modify (none — discovery only)
- Create: `tests/fixtures/expected/gpu_sample1.txt` (manual, hand-paste)

- [ ] **Step 1: Verify GPU is reachable and CUDA path works**

Run:
```bash
cargo test --release --features cuda --test transcribe -- --ignored --nocapture --test-threads=1 test_q06_sample1 2>&1 | tail -50
```

Expected: prints something like
```
Language : <LANG>
Text: <TRANSCRIPT>
```
plus timing lines.

- [ ] **Step 2: Capture the printed `Text:` line for `sample1.wav`**

From the output above, copy the **transcript text only** (the line after `Text:`). The text may contain Unicode (e.g. Chinese) — copy raw, do not normalize.

- [ ] **Step 3: Write it to the expected fixture**

```bash
mkdir -p tests/fixtures/expected
# Paste the captured text exactly. Use a single line, trailing newline allowed.
echo '<CAPTURED TEXT>' > tests/fixtures/expected/gpu_sample1.txt
```

If the transcript is multi-line, preserve newlines as they appeared in the test output.

Verify:
```bash
cat tests/fixtures/expected/gpu_sample1.txt
wc -c tests/fixtures/expected/gpu_sample1.txt
```

- [ ] **Step 4: Commit the fixture**

```bash
git add tests/fixtures/expected/gpu_sample1.txt
git commit -m "test(fixture): CUDA ground-truth transcript for sample1.wav"
```

---

## Task 2: Create `cpu_audio_encoder.rs` skeleton with `Linear` + `LayerNorm` + `GELU`

**Files:**
- Create: `src/cpu_audio_encoder.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Add the module declaration**

In `src/lib.rs`, after line 14 (`mod cpu_engine;`) add:
```rust
#[cfg(feature = "cpu")]
mod cpu_audio_encoder;
```

- [ ] **Step 2: Write the skeleton with `CpuAudioLinear`, `CpuAudioLayerNorm`, `gelu`**

Create `src/cpu_audio_encoder.rs` with the content below.

```rust
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
```

- [ ] **Step 3: Verify it compiles (cpu feature only)**

```bash
cargo check --no-default-features --features cpu 2>&1 | tail -20
```

Expected: `Finished` line, no errors. (There may be unused-code warnings for the types we just defined — that's expected for now.)

- [ ] **Step 4: Verify CUDA path still compiles**

```bash
cargo check --features cuda 2>&1 | tail -5
```

Expected: `Finished` line, no errors.

- [ ] **Step 5: Commit**

```bash
git add src/cpu_audio_encoder.rs src/lib.rs
git commit -m "feat(cpu-audio): skeleton with CpuAudioLinear / LayerNorm / GELU"
```

---

## Task 3: Add `im2col` + `CpuConvStem` (3 × conv2d + permute + conv_out + PE)

**Files:**
- Modify: `src/cpu_audio_encoder.rs`

- [ ] **Step 1: Append `im2col_3x3_s2p1` to `src/cpu_audio_encoder.rs`**

Add at the bottom of the file:

```rust
/// im2col for a single conv2d layer with kernel=3, stride=2, pad=1.
/// Input x: [b, c_in, h, w]   (h, w are 1 and T_mel initially).
/// Output: [col_count, 3*3*c_in] row-major, where col_count = b * c_out * h_out * w_out
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
```

- [ ] **Step 2: Append `CpuConvStem` to `src/cpu_audio_encoder.rs`**

```rust
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
        let x4d: Vec<f32> = mel.iter().enumerate().map(|(i, &v)| {
            // mel layout: [n_mels, T_mel] row-major. 4D: [b, c, h, w] = [1, n_mels, 1, T_mel]
            // Same byte order — no copy needed.
            v
        }).collect();

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
        // GEMM: w_w [c_out, 9*c_in] @ cols.T [9*c_in, b*c_in*h_out*w_out]
        //   We need output [b, c_out, h_out, w_out] flat. cols is [b*c_in*h_out*w_out, 9*c_in].
        //   Transpose cols to [9*c_in, b*c_in*h_out*w_out] then gemm w_w * cols_T.
        // Simpler: view cols as [col_count, k] with col_count=b*c_in*h_out*w_out, k=9*c_in.
        // w_w is [n=c_out, k=9*c_in]. We want out [c_out, col_count].
        //   y = w_w @ cols.T   →  out[n, col_count]
        // gemm layout: y[n, col_count] = sum_k w_w[n, k] * cols[col_count, k]
        // gemm's API: gemm(m, n, k, c, cs_c, rs_c, ..., a, cs_a, rs_a, b, cs_b, rs_b, ...)
        //   Here m = c_out, n = col_count, k = 9*c_in.
        //   a = w_w: shape [c_out, 9*c_in] row-major → cs_a=1, rs_a=9*c_in
        //   b = cols: shape [col_count, 9*c_in] row-major. We want b^T effectively:
        //     b[k, col] = cols[col, k]. gemm's B operand stride: cs_b = 1 (k+1), rs_b = 9*c_in (col+1).
        let col_count = b * c_in * h_out * w_out;
        let k = 9 * c_in;
        let mut out = vec![0.0f32; c_out * col_count];
        gemm::gemm(
            c_out, col_count, k,
            out.as_mut_ptr(), 1, col_count as isize,
            false,
            w_w.data.as_ptr(), 1, k as isize,
            cols.as_ptr(), 1, k as isize,
            0.0, 1.0, false, false, false,
            gemm::Parallelism::Rayon(0),
        );

        // Add bias and GELU. out is [c_out, col_count] but we want [b, c_out, h_out, w_out].
        // col_count = b * h_out * w_out * c_in. The mapping of column index to (ib, ic_in, ho, wo)
        // is the same as im2col: ((ib * c_in + ic) * h_out + ho) * w_out + wo.
        // We need to scatter into the right [b, c_out, h_out, w_out] slot.
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
    // Conv2D weight is [c_out, c_in, kH, kW]. Flatten to [c_out, c_in*kH*kW].
    let c_out = shape[0];
    let k = shape[1..].iter().product::<usize>();
    Ok(CpuWeight { data, rows: c_out, cols: k })
}

fn load_bias(weights: &HashMap<String, RawTensor>, name: &str) -> Result<Vec<f32>> {
    let (data, _) = weights.get(name).ok_or_else(|| anyhow::anyhow!("bias not found: {}", name))?.as_f32()?;
    Ok(data)
}
```

- [ ] **Step 3: Verify it compiles**

```bash
cargo check --no-default-features --features cpu 2>&1 | tail -20
```

Expected: `Finished` line. (Unused warnings for `C1_W`, `C2_W` etc. on fields is expected — they're used in `conv_block`. The real warnings will be on `Self` fields if methods are unused; we add usage in Task 4.)

If you see `error[E0599] no function or associated item named 'gemm' found` — check the import: `use gemm::gemm;` is missing. Add it next to the existing `use crate::cpu_engine::...` block.

- [ ] **Step 4: Commit**

```bash
git add src/cpu_audio_encoder.rs
git commit -m "feat(cpu-audio): CpuConvStem with im2col + 3 conv2d + permute + PE"
```

---

## Task 4: Add `CpuAudioAttention` (Q/K/V/O projections + windowed attention)

**Files:**
- Modify: `src/cpu_audio_encoder.rs`

- [ ] **Step 1: Append `CpuAudioAttention` to `src/cpu_audio_encoder.rs`**

```rust
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
        // Each is [b, s, dm] = [b, s, nh*hd]. View as [b, s, nh, hd] then permute to [b, nh, s, hd].

        // Compute attention output [b, nh, s, hd] then permute back to [b, s, nh*hd].
        let scale = 1.0f32 / (hd as f32).sqrt();
        let window = ws.filter(|&w| w > 0 && w < s);

        let attn_out = if let Some(w) = window {
            // Windowed: process chunks of `w` tokens, each chunk attends only to itself.
            let mut out = vec![0.0f32; b * nh * s * hd];
            for st in (0..s).step_by(w) {
                let len = w.min(s - st);
                // q_chunk, k_chunk, v_chunk are [b, len, dm]. Reshape to [b, nh, len, hd].
                let q_chunk = slice_dim1(&q, st, len);
                let k_chunk = slice_dim1(&k, st, len);
                let v_chunk = slice_dim1(&v, st, len);
                let o = attention_window(&q_chunk, &k_chunk, &v_chunk, b, nh, len, hd, scale);
                // o: [b, nh, len, hd]. Scatter into out[*, *, st..st+len, *].
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
            // For each query position is_ in [0, s):
            for is_ in 0..s {
                // scores[t] = q[is_] · k[t] for t in [0, s)
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
                // softmax
                let mut sum = 0.0f32;
                for t in 0..s {
                    scores[t] = (scores[t] - max_s).exp();
                    sum += scores[t];
                }
                let inv = 1.0 / sum;
                // weighted sum of V
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
```

- [ ] **Step 2: Verify it compiles**

```bash
cargo check --no-default-features --features cpu 2>&1 | tail -20
```

Expected: `Finished` line.

- [ ] **Step 3: Commit**

```bash
git add src/cpu_audio_encoder.rs
git commit -m "feat(cpu-audio): CpuAudioAttention with windowed Q·K·V·O"
```

---

## Task 5: Add `CpuAudioFfn` + `CpuAudioLayer` + `CpuAudioEncoder`

**Files:**
- Modify: `src/cpu_audio_encoder.rs`

- [ ] **Step 1: Append `CpuAudioFfn` and `CpuAudioLayer`**

```rust
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
        // x1 = x + attn_out
        let mut x1_data = x.data.clone();
        for (a, b) in x1_data.iter_mut().zip(attn_out.data.iter()) { *a += *b; }
        let x1 = CpuTensor::new(x1_data, x.shape.clone());
        let normed2 = self.fln.forward(&x1);
        let ffn_out = self.ffn.forward(&normed2)?;
        // x2 = x1 + ffn_out
        let mut x2_data = x1.data;
        for (a, b) in x2_data.iter_mut().zip(ffn_out.data.iter()) { *a += *b; }
        Ok(CpuTensor::new(x2_data, x1.shape))
    }
}
```

- [ ] **Step 2: Append `CpuAudioEncoder` and the `feo` helper**

```rust
/// Replicate of `gpu_audio_encoder.rs::feo` (line 389-392).
/// f(l) = ceil((l - 1) / 2) + 1, applied three times.
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
        // 1. Conv stem.
        let (conv_data, n_total) = self.conv_stem.forward(mel, n_mels, mel_len)?;
        let dm = self.config.d_model;

        // 2. Transformer layers (windowed). ws from feo formula.
        let cs2 = self.config.n_window * 2;
        let tpc = feo(cs2);
        let cpw = self.config.n_window_infer / cs2;
        let ws = tpc * cpw;
        let mut h = CpuTensor::new(conv_data, vec![1, n_total, dm]);
        for layer in &self.layers {
            h = layer.forward(h, Some(ws))?;
        }

        // 3. ln_post + proj1 + GELU + proj2.
        let h = self.ln_post.forward(&h);
        let mut h = self.proj1.forward(&h)?;
        gelu_inplace(&mut h);
        let h = self.proj2.forward(&h)?;
        Ok(h.data)
    }
}
```

- [ ] **Step 3: Verify it compiles**

```bash
cargo check --no-default-features --features cpu 2>&1 | tail -20
```

Expected: `Finished` line. Warnings about unused fields/methods on `CpuAudioEncoder` and the lower types are fine — they will be used in Task 6.

- [ ] **Step 4: Commit**

```bash
git add src/cpu_audio_encoder.rs
git commit -m "feat(cpu-audio): CpuAudioFfn + CpuAudioLayer + CpuAudioEncoder"
```

---

## Task 6: Wire `CpuAudioEncoder` into `inference.rs`

**Files:**
- Modify: `src/inference.rs`

- [ ] **Step 1: Read the current `encode_audio_cpu` and `AsrInferenceInner::new`**

Confirm:
- `AsrInferenceInner` lives around `src/inference.rs:50-80` (after the type defs).
- `encode_audio_cpu` is around line 242-251.
- `AsrInferenceInner::new` is around line 100-150.

- [ ] **Step 2: Add `cpu_audio_encoder` field**

In `AsrInferenceInner`, after the existing field
```rust
    #[cfg(feature = "cpu")]
    pub(crate) cpu_decoder: Option<CpuTextDecoder>,
```
add:
```rust
    #[cfg(feature = "cpu")]
    cpu_audio_encoder: Option<CpuAudioEncoder>,
```

- [ ] **Step 3: Initialize the field in `new`**

In the `#[cfg(feature = "cpu")]` block inside `AsrInferenceInner::new`, after the line that loads `cpu_decoder`, add:
```rust
                    let cpu_audio_encoder = CpuAudioEncoder::load(
                        &weights, "thinker.audio_tower", &config.thinker_config.audio_config,
                    )?;
                    cpu_audio_encoder: Some(cpu_audio_encoder),
```

Also add the import at the top of the file inside the existing `#[cfg(feature = "cpu")] use crate::cpu_engine::CpuTextDecoder;` block:
```rust
    #[cfg(feature = "cpu")]
    use crate::cpu_audio_encoder::CpuAudioEncoder;
```

- [ ] **Step 4: Replace `encode_audio_cpu` body**

Replace the entire function body of `encode_audio_cpu` (currently line 242-251) with:

```rust
    #[cfg(feature = "cpu")]
    fn encode_audio_cpu(&self, samples: &[f32]) -> anyhow::Result<Vec<f32>> {
        let audio_cfg = &self.config.thinker_config.audio_config;
        let n_mels = audio_cfg.num_mel_bins;
        let (mel_data, _, n_frames) = self.mel_extractor.extract(samples)?;
        debug!("Mel: {}×{} frames", n_mels, n_frames);
        let enc = self.cpu_audio_encoder.as_ref().expect("CPU audio encoder not built");
        let out = enc.forward(&mel_data, n_mels, n_frames)?;
        info!("Audio tokens: {}", out.len() / audio_cfg.output_dim);
        Ok(out)
    }
```

- [ ] **Step 5: Verify it compiles (cpu feature only)**

```bash
cargo check --no-default-features --features cpu 2>&1 | tail -20
```

Expected: `Finished` line, no errors. Warnings about `let _ = samples;` style if any — should be gone now.

- [ ] **Step 6: Verify CUDA path still compiles**

```bash
cargo check --features cuda 2>&1 | tail -5
```

Expected: `Finished` line, no errors.

- [ ] **Step 7: Commit**

```bash
git add src/inference.rs
git commit -m "feat(cpu): wire CpuAudioEncoder into encode_audio_cpu"
```

---

## Task 7: Add CPU transcribe integration test

**Files:**
- Create: `tests/cpu_transcribe.rs`

- [ ] **Step 1: Create the test file**

```rust
//! CPU transcribe integration test — byte-equality with the CUDA fixture.
//!
//! Run:
//!   cargo test --release --features cpu --test cpu_transcribe -- --ignored --nocapture --test-threads=1
//!
//! Reads the ground-truth text captured by Task 1 (one-shot CUDA run) and
//! asserts the CPU `Backend::Cpu` transcribe() output is byte-identical.

use std::path::PathBuf;
use std::time::Instant;

use qwen3_asr::AsrInference;
use qwen3_asr::Backend;
use qwen3_asr::TranscribeOptions;

fn model_dir_06() -> String {
    // Mirror the helper used in tests/transcribe.rs.
    std::env::var("QWEN3_ASR_MODEL_06").unwrap_or_else(|_| "../models/qwen3-asr-0.6b".to_string())
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(format!("tests/fixtures/{}", name))
}

fn expected_path() -> PathBuf {
    PathBuf::from("tests/fixtures/expected/gpu_sample1.txt")
}

#[test]
#[ignore]
fn test_cpu_q06_sample1() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .is_test(true).try_init();

    let backend = Backend::Cpu;
    let engine = AsrInference::load(
        std::path::Path::new(&model_dir_06()), backend,
    ).expect("load 0.6B on CPU");

    let t0 = Instant::now();
    let result = engine.transcribe(
        &fixture("sample1.wav"),
        TranscribeOptions::default(),
    ).expect("cpu transcribe");
    let elapsed = t0.elapsed().as_secs_f32();

    println!("CPU-0.6B-sample1 | {:.3}s elapsed", elapsed);
    println!("Language : {}", result.language);
    println!("Text     : {}", result.text);

    let expected = std::fs::read_to_string(expected_path())
        .expect("read expected fixture (run Task 1 first)");
    let expected_trim = expected.trim_end_matches(['\n', '\r']);
    assert_eq!(
        result.text, expected_trim,
        "CPU transcribe text differs from CUDA fixture.\nGPU: {:?}\nCPU: {:?}",
        expected_trim, result.text
    );
}
```

- [ ] **Step 2: Verify it compiles**

```bash
cargo check --tests --no-default-features --features cpu 2>&1 | tail -20
```

Expected: `Finished` line. If `Backend` is not exported, check `src/lib.rs` — it re-exports `Backend` from `backend::Backend`. It is.

- [ ] **Step 3: Run the test**

```bash
cargo test --release --no-default-features --features cpu --test cpu_transcribe -- --ignored --nocapture --test-threads=1
```

Expected: passes. If it fails, the most likely causes (in order of likelihood) are:
1. **Conv permute layout** — re-check `perm[dst] = x3[src];` indexing in `CpuConvStem::forward` (Task 3).
2. **im2col column ordering** — the `(ib * c_in + ic) * h_out + ho` mapping in `conv_block` must match im2col's `(ib * c_in + ic) * h_out + ho`.
3. **PE add** — confirm broadcast over batch dim is `it` not `ib * t + it`.
4. **LayerNorm** — confirm mean/var are over the last dim only and that `bias` is added.
5. **Windowed attention ws** — must equal `feo(cs2) * (n_window_infer / cs2)`.

When debugging, run the existing CUDA test on the same fixture and add temporary `println!` in `CpuConvStem::forward` to compare intermediate values against the GPU path. (Re-running the CUDA test is non-destructive — it just prints to stdout.)

- [ ] **Step 4: Commit**

```bash
git add tests/cpu_transcribe.rs
git commit -m "test(cpu): integration test comparing CPU transcribe to CUDA fixture"
```

---

## Task 8: Update ROADMAP.md

**Files:**
- Modify: `ROADMAP.md`

- [ ] **Step 1: Update §0 wording about CPU path**

In §0 (line 7), change:
```
... CPU 路径只实现了**文本解码器**（gemm + rayon），CPU 音频编码器**未实现**——CPU 端 `transcribe()` 在运行时直接报错。
```
to:
```
... CPU 路径实现到**端到端可用**（文本解码器 + 音频编码器，f32 路径）。RTFx 显著低于 CUDA（量级：CPU 15s ~1-2× vs CUDA 15s ~24×）。
```

- [ ] **Step 2: Update §1.1 tree**

In the tree (line 16-30), after `cpu_engine.rs` line, add:
```
├── cpu_audio_encoder.rs # ★ 手写 CPU 音频编码器（f32 im2col + gemm + 手写 LN/attn/FFN）
```

- [ ] **Step 3: Update §3.2**

In §3.2 (line 110-112), change the section header and content:
```
### 3.2 CPU 路径：已端到端可用

`encode_audio_cpu` 已实现（`src/cpu_audio_encoder.rs`，f32 im2col + gemm + 手写 LN/attn/FFN）。CPU 端 `transcribe()` 现在能跑出结果，输出与 CUDA 路径字节对齐（参考 `tests/cpu_transcribe.rs`）。RTFx 优化未做。
```

- [ ] **Step 4: Add §4.1 P0 entry for CPU test**

In §4.1 (around line 140), after the existing P0 entries, add:
```
- [ ] **跑 CPU 集成测试**（`cargo test --release --features cpu --test cpu_transcribe -- --ignored --nocapture --test-threads=1`），确认 CPU 输出与 CUDA fixture 字节对齐
```

- [ ] **Step 5: Verify ROADMAP still renders correctly**

```bash
cargo check --features cuda --no-default-features --features cpu 2>&1 | tail -3
```

Expected: `Finished` line. (The check doesn't read ROADMAP, but it confirms we didn't break anything.)

```bash
git diff --stat ROADMAP.md
```

Expected: ROADMAP.md has small changes (4 hunks).

- [ ] **Step 6: Commit**

```bash
git add ROADMAP.md
git commit -m "docs(roadmap): CPU path end-to-end, f32 audio encoder shipped"
```

---

## Task 9: End-to-end verification

**Files:**
- (no file changes — verification only)

- [ ] **Step 1: Run all CPU checks**

```bash
cargo check --no-default-features --features cpu 2>&1 | tail -3
cargo check --tests --no-default-features --features cpu 2>&1 | tail -3
```

Expected: both `Finished` with no errors.

- [ ] **Step 2: Run the CPU integration test**

```bash
cargo test --release --no-default-features --features cpu --test cpu_transcribe -- --ignored --nocapture --test-threads=1
```

Expected: passes. Wall time: probably 5-30s depending on hardware (we're targeting Ultra 7 265K; expect ~1-2× RTFx so 15s audio takes 8-15s).

- [ ] **Step 3: Confirm CUDA tests still pass**

```bash
cargo check --features cuda 2>&1 | tail -3
cargo check --tests --features cuda 2>&1 | tail -3
```

Expected: both `Finished` with no errors. (Don't actually run the 13 CUDA tests — they need GPU + model — but the build must succeed.)

- [ ] **Step 4: Final summary commit (only if changes were needed)**

If step 1-3 all pass, no commit. If any tweaks were needed, commit them with `chore: address review comments on CPU audio encoder`.

---

## Notes for the implementer

- **Weight dtype**: All Qwen3-ASR weights in `safetensors` are f16. We call `RawTensor::as_f32()` to upcast at load time. This is the same idiom as `cpu_engine.rs::load_vec_f32`.
- **`AsrInferenceInner::new`**: there are TWO `new` functions (one for the `#[cfg(feature = "cuda")]` path, one for the cpu path). Make sure you're editing the cpu one. The cpu one initializes `backend: Backend::Cpu` and `gpu_audio_encoder: None`.
- **`gemm` crate signature**: `gemm(m, n, k, c_ptr, c_cs, c_rs, beta_is_nonzero, a_ptr, a_cs, a_rs, b_ptr, b_cs, b_rs, beta, alpha, a_trans, b_trans, c_trans, parallelism)`. We use `a_trans=false, b_trans=false, c_trans=false, beta=0.0, alpha=1.0`.
- **The first test run may be slow** as the OS pages in the ~3GB of weights. Subsequent runs (within the same OS session) are faster.
- **Debugging byte-mismatch**: in `CpuConvStem::forward`, add a `println!` that compares `x3[t]` and the equivalent from the GPU path. The GPU path's `GpuConvStem::forward` (line 252-264 of `gpu_audio_encoder.rs`) does `cuda.download_tensor(&x)?` — the f16 values are the gold standard. Upcast them to f32 and diff.
