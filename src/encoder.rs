use anyhow::Result;
use burn::tensor::{activation, Bool, Tensor, TensorData};
use burn::tensor::backend::Backend;
use burn::tensor::module::{attention, conv2d};
use burn::tensor::ops::{AttentionModuleOptions, ConvOptions};
use std::collections::HashMap;

use crate::config::AudioEncoderConfig;

// ─── Matmul alignment ──────────────────────────────────────────────
// cubecl CUDA GEMV kernel requires K (reduction dim) % 32 == 0.
// For decode (m=1) attention:
//   Q @ K^T: reduction dim = head_dim (always 128, 128%32==0) → safe
//   attn @ V: reduction dim = n (KV seq len) → may need padding
// Strategy:
//   • m > 1 (prefill): use burn's built-in attention() — no alignment issue
//   • m == 1 (decode) + n % 32 == 0: use attention() directly
//   • m == 1 (decode) + n % 32 != 0: manual attention, only pad V for final matmul

const K_ALIGN: usize = 32;

pub(crate) fn safe_attention<B: Backend>(
    q: Tensor<B, 4>,  // [b, h, m, d]
    k: Tensor<B, 4>,  // [b, h, n, d]
    v: Tensor<B, 4>,  // [b, h, n, d]
    is_causal: bool,
) -> Tensor<B, 4> {
    let n = k.dims()[2];
    let m = q.dims()[2];

    // Prefill (m > 1): GEMM kernel, no alignment issue
    // Decode (m == 1) with aligned n: also fine
    if m > 1 || n % K_ALIGN == 0 {
        let opts = AttentionModuleOptions { scale: None, softcap: None, is_causal };
        return attention(q, k, v, None::<Tensor<B, 4, Bool>>, None, opts);
    }

    // Decode (m == 1) with unaligned n:
    // Q @ K^T is safe (reduction dim = head_dim = 128, aligned)
    // Only attn @ V needs padding (reduction dim = n, unaligned)
    let [b, h, _, d] = q.dims();
    let device = q.device();
    let scale = 1.0 / (d as f64).sqrt();

    // Step 1: scores = Q @ K^T / sqrt(d) — no padding needed for K
    let scores = q.matmul(k.swap_dims(2, 3)) * scale; // [b, h, 1, n]

    // Step 2: softmax on valid positions
    let attn = activation::softmax(scores, 3); // [b, h, 1, n]

    // Step 3: pad attn and V for the final matmul (reduction dim = n_padded)
    let n_padded = ((n + K_ALIGN - 1) / K_ALIGN) * K_ALIGN;
    let pad = n_padded - n;
    let attn_padded = Tensor::cat(vec![attn, Tensor::zeros([b, h, 1, pad], &device)], 3);
    let v_padded = Tensor::cat(vec![v, Tensor::zeros([b, h, pad, d], &device)], 2);

    attn_padded.matmul(v_padded) // [b, h, 1, d]
}

// ─── Weight helpers ────────────────────────────────────────────────

fn get_w<B: Backend, const D: usize>(
    weights: &HashMap<String, TensorData>,
    name: &str,
    device: &B::Device,
) -> Result<Tensor<B, D>> {
    weights
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("weight not found: {}", name))
        .map(|d| Tensor::from_data(d.clone(), device))
}

fn load_linear<B: Backend>(
    weights: &HashMap<String, TensorData>,
    prefix: &str,
    device: &B::Device,
) -> Result<LinearW<B>> {
    let weight = get_w(weights, &format!("{}.weight", prefix), device)?;
    let bias = weights
        .get(&format!("{}.bias", prefix))
        .map(|d| Tensor::<B, 1>::from_data(d.clone(), device));
    Ok(LinearW::new(weight, bias))
}

// ─── Linear (generic D-dim matmul + broadcast) ─────────────────────

pub(crate) struct LinearW<B: Backend> {
    weight_t: Tensor<B, 2>, // pre-transposed: [out_features, in_features]
    bias: Option<Tensor<B, 1>>,
}

impl<B: Backend> LinearW<B> {
    pub fn new(weight: Tensor<B, 2>, bias: Option<Tensor<B, 1>>) -> Self {
        Self { weight_t: weight.transpose(), bias }
    }

    pub fn forward<const D: usize>(&self, x: &Tensor<B, D>) -> Tensor<B, D> {
        let wd = self.weight_t.dims();
        let mut ws = [1; D];
        ws[D - 2] = wd[0];
        ws[D - 1] = wd[1];
        let out = x.clone().matmul(self.weight_t.clone().reshape(ws));
        match &self.bias {
            Some(b) => {
                let bd = b.dims();
                let mut bs = [1; D];
                bs[D - 1] = bd[0];
                out + b.clone().reshape(bs)
            }
            None => out,
        }
    }
}

// ─── Conv2d (native backend) ──────────────────────────────────────

fn conv2d_forward<B: Backend>(
    input: &Tensor<B, 4>,
    weight: &Tensor<B, 4>,
    bias: Option<&Tensor<B, 1>>,
    stride: usize,
    padding: usize,
) -> Tensor<B, 4> {
    let opts = ConvOptions::new([stride, stride], [padding, padding], [1, 1], 1);
    conv2d(input.clone(), weight.clone(), bias.cloned(), opts)
}

// ─── Manual LayerNorm ──────────────────────────────────────────────

struct ManualLayerNorm<B: Backend> {
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
    eps: f64,
    size: usize,
}

impl<B: Backend> ManualLayerNorm<B> {
    fn load(weights: &HashMap<String, TensorData>, prefix: &str, size: usize, eps: f64, device: &B::Device) -> Result<Self> {
        Ok(Self {
            weight: get_w(weights, &format!("{}.weight", prefix), device)?,
            bias: get_w(weights, &format!("{}.bias", prefix), device)?,
            eps, size,
        })
    }

    fn forward<const D: usize>(&self, x: &Tensor<B, D>) -> Tensor<B, D> {
        let last = D - 1;
        let mean = x.clone().mean_dim(last);
        let var = x.clone().var_bias(last);
        let xn = (x.clone() - mean) / (var + self.eps).sqrt();
        let mut ws = [1; D];
        ws[D - 1] = self.size;
        xn * self.weight.clone().reshape(ws) + self.bias.clone().reshape(ws)
    }
}

// ─── Audio Self-Attention ──────────────────────────────────────────

struct AudioAttention<B: Backend> {
    q_proj: LinearW<B>, k_proj: LinearW<B>, v_proj: LinearW<B>, out_proj: LinearW<B>,
    num_heads: usize, head_dim: usize,
}

impl<B: Backend> AudioAttention<B> {
    fn load(weights: &HashMap<String, TensorData>, prefix: &str, nh: usize, dm: usize, device: &B::Device) -> Result<Self> {
        Ok(Self {
            q_proj: load_linear(weights, &format!("{}.q_proj", prefix), device)?,
            k_proj: load_linear(weights, &format!("{}.k_proj", prefix), device)?,
            v_proj: load_linear(weights, &format!("{}.v_proj", prefix), device)?,
            out_proj: load_linear(weights, &format!("{}.out_proj", prefix), device)?,
            num_heads: nh, head_dim: dm / nh,
        })
    }

    fn forward(&self, x: &Tensor<B, 3>, ws: Option<usize>) -> Tensor<B, 3> {
        let [b, s, _] = x.dims();
        let nh = self.num_heads; let hd = self.head_dim;
        let q = self.q_proj.forward(x).reshape([b, s, nh, hd]).swap_dims(1, 2);
        let k = self.k_proj.forward(x).reshape([b, s, nh, hd]).swap_dims(1, 2);
        let v = self.v_proj.forward(x).reshape([b, s, nh, hd]).swap_dims(1, 2);
        if let Some(w) = ws.filter(|&w| w > 0 && w < s) {
            return self.windowed(q, k, v, b, w);
        }
        let out = safe_attention(q, k, v, false);
        self.out_proj.forward(&out.swap_dims(1, 2).reshape([b, s, nh * hd]))
    }

    fn windowed(&self, q: Tensor<B, 4>, k: Tensor<B, 4>, v: Tensor<B, 4>, b: usize, ws: usize) -> Tensor<B, 3> {
        let sl = q.dims()[2];
        let mut ch = Vec::new();
        for st in (0..sl).step_by(ws) {
            let ln = ws.min(sl - st);
            let qw = q.clone().slice([0..b, 0..self.num_heads, st..st + ln]);
            let kw = k.clone().slice([0..b, 0..self.num_heads, st..st + ln]);
            let vw = v.clone().slice([0..b, 0..self.num_heads, st..st + ln]);
            ch.push(safe_attention(qw, kw, vw, false));
        }
        let out = Tensor::cat(ch, 2);
        self.out_proj.forward(&out.swap_dims(1, 2).reshape([b, sl, self.num_heads * self.head_dim]))
    }
}

// ─── Audio FFN ─────────────────────────────────────────────────────

struct AudioFfn<B: Backend> { fc1: LinearW<B>, fc2: LinearW<B> }

impl<B: Backend> AudioFfn<B> {
    fn load(weights: &HashMap<String, TensorData>, prefix: &str, device: &B::Device) -> Result<Self> {
        Ok(Self { fc1: load_linear(weights, &format!("{}.fc1", prefix), device)?, fc2: load_linear(weights, &format!("{}.fc2", prefix), device)? })
    }
    fn forward(&self, x: &Tensor<B, 3>) -> Tensor<B, 3> {
        self.fc2.forward(&activation::gelu(self.fc1.forward(x)))
    }
}

// ─── Audio Encoder Layer ───────────────────────────────────────────

struct AudioEncoderLayer<B: Backend> {
    sln: ManualLayerNorm<B>, attn: AudioAttention<B>, fln: ManualLayerNorm<B>, ffn: AudioFfn<B>,
}

impl<B: Backend> AudioEncoderLayer<B> {
    fn load(weights: &HashMap<String, TensorData>, prefix: &str, nh: usize, dm: usize, device: &B::Device) -> Result<Self> {
        Ok(Self {
            sln: ManualLayerNorm::load(weights, &format!("{}.self_attn_layer_norm", prefix), dm, 1e-5, device)?,
            attn: AudioAttention::load(weights, &format!("{}.self_attn", prefix), nh, dm, device)?,
            fln: ManualLayerNorm::load(weights, &format!("{}.final_layer_norm", prefix), dm, 1e-5, device)?,
            ffn: AudioFfn::load(weights, prefix, device)?,
        })
    }
    fn forward(&self, x: &Tensor<B, 3>, ws: Option<usize>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(&self.sln.forward(x), ws);
        x.clone() + self.ffn.forward(&self.fln.forward(&x))
    }
}

// ─── Sinusoidal PE ─────────────────────────────────────────────────

fn create_sinusoidal_embedding<B: Backend>(max_len: usize, dim: usize, device: &B::Device) -> Tensor<B, 2> {
    let half = dim / 2;
    let lt = (10000.0f64).ln() / (half as f64 - 1.0);
    let mut e = vec![0.0f32; max_len * dim];
    for p in 0..max_len {
        for i in 0..half {
            let a = p as f64 * (-(i as f64) * lt).exp();
            e[p * dim + i] = a.sin() as f32;
            e[p * dim + half + i] = a.cos() as f32;
        }
    }
    Tensor::from_data(TensorData::new(e, [max_len, dim]), device)
}

// ─── Encoder Cache ─────────────────────────────────────────────────

pub struct EncoderCache<B: Backend> {
    pub completed_windows: Vec<Tensor<B, 2>>,
    pub committed_chunks: usize,
}

impl<B: Backend> EncoderCache<B> {
    pub fn new() -> Self { Self { completed_windows: Vec::new(), committed_chunks: 0 } }
    pub fn cached_tokens(&self) -> usize { self.completed_windows.iter().map(|t| t.dims()[0]).sum() }
}

impl<B: Backend> Default for EncoderCache<B> { fn default() -> Self { Self::new() } }

// ─── Audio Encoder ─────────────────────────────────────────────────

pub(crate) struct AudioEncoder<B: Backend> {
    c1w: Tensor<B, 4>, c1b: Tensor<B, 1>,
    c2w: Tensor<B, 4>, c2b: Tensor<B, 1>,
    c3w: Tensor<B, 4>, c3b: Tensor<B, 1>,
    co: LinearW<B>, pe: Tensor<B, 2>,
    layers: Vec<AudioEncoderLayer<B>>,
    lnp: ManualLayerNorm<B>, p1: LinearW<B>, p2: LinearW<B>,
    config: AudioEncoderConfig,
}

impl<B: Backend> AudioEncoder<B> {
    pub(crate) fn load(weights: &HashMap<String, TensorData>, prefix: &str, config: &AudioEncoderConfig, device: &B::Device) -> Result<Self> {
        let dm = config.d_model;
        let c1w = get_w(weights, &format!("{}.conv2d1.weight", prefix), device)?;
        let c1b = get_w(weights, &format!("{}.conv2d1.bias", prefix), device)?;
        let c2w = get_w(weights, &format!("{}.conv2d2.weight", prefix), device)?;
        let c2b = get_w(weights, &format!("{}.conv2d2.bias", prefix), device)?;
        let c3w = get_w(weights, &format!("{}.conv2d3.weight", prefix), device)?;
        let c3b = get_w(weights, &format!("{}.conv2d3.bias", prefix), device)?;
        let co = load_linear(weights, &format!("{}.conv_out", prefix), device)?;
        let mut layers = Vec::new();
        for i in 0..config.encoder_layers {
            layers.push(AudioEncoderLayer::load(weights, &format!("{}.layers.{}", prefix, i), config.encoder_attention_heads, dm, device)?);
        }
        let lnp = ManualLayerNorm::load(weights, &format!("{}.ln_post", prefix), dm, 1e-5, device)?;
        let p1 = load_linear(weights, &format!("{}.proj1", prefix), device)?;
        let p2 = load_linear(weights, &format!("{}.proj2", prefix), device)?;
        let pe = create_sinusoidal_embedding(config.max_source_positions, dm, device);
        Ok(Self { c1w, c1b, c2w, c2b, c3w, c3b, co, pe, layers, lnp, p1, p2, config: config.clone() })
    }

    fn run_conv_stem(&self, mel: &Tensor<B, 2>, cs: usize) -> Vec<Tensor<B, 2>> {
        let nf = mel.dims()[1]; let tpc = Self::feo(cs);
        let nfull = nf / cs; let tail = nf % cs;
        let mut cm = Vec::new(); let mut cv = Vec::new();
        for i in 0..nfull {
            let s = i * cs;
            cm.push(mel.clone().slice([0..mel.dims()[0], s..s + cs]).unsqueeze_dim::<3>(0));
            cv.push(tpc);
        }
        if tail > 0 {
            let s = nfull * cs;
            let tm = mel.clone().slice([0..mel.dims()[0], s..s + tail]);
            let pad = Tensor::zeros([mel.dims()[0], cs - tail], &mel.device());
            cm.push(Tensor::cat(vec![tm, pad], 1).unsqueeze_dim::<3>(0));
            cv.push(Self::feo(tail));
        }
        let b = Tensor::cat(cm, 0).unsqueeze_dim::<4>(1);
        let x = activation::gelu(conv2d_forward(&b, &self.c1w, Some(&self.c1b), 2, 1));
        let x = activation::gelu(conv2d_forward(&x, &self.c2w, Some(&self.c2b), 2, 1));
        let x = activation::gelu(conv2d_forward(&x, &self.c3w, Some(&self.c3b), 2, 1));
        let [b2, c2, f2, t2] = x.dims();
        let r = x.permute([0, 3, 1, 2]).reshape([b2, t2, c2 * f2]);
        let co = self.co.forward(&r);
        let pe = self.pe.clone().slice([0..t2]).unsqueeze_dim::<3>(0);
        let co = co + pe;
        let mut av: Vec<Tensor<B, 2>> = Vec::new();
        for (idx, &v) in cv.iter().enumerate() {
            av.push(co.clone().slice([idx..idx + 1, 0..v]).squeeze_dim(0));
        }
        av
    }

    /// Run only the conv stem + sinusoidal PE.  Returns the [n_tokens, d_model] tensor
    /// (kept for the legacy GPU-burn audio path; the cuda feature uses GpuAudioEncoder
    /// which runs the entire conv stem + transformer on cuBLAS/custom kernels).
    pub(crate) fn forward_conv_only(&self, mel: &Tensor<B, 2>) -> Tensor<B, 2> {
        let cs = self.config.n_window * 2;
        let nf = mel.dims()[1]; let tpc = Self::feo(cs);
        let nfull = nf / cs; let tail = nf % cs;
        let mut cm = Vec::new(); let mut cv = Vec::new();
        for i in 0..nfull {
            let s = i * cs;
            cm.push(mel.clone().slice([0..mel.dims()[0], s..s + cs]).unsqueeze_dim::<3>(0));
            cv.push(tpc);
        }
        if tail > 0 {
            let s = nfull * cs;
            let tm = mel.clone().slice([0..mel.dims()[0], s..s + tail]);
            let pad = Tensor::zeros([mel.dims()[0], cs - tail], &mel.device());
            cm.push(Tensor::cat(vec![tm, pad], 1).unsqueeze_dim::<3>(0));
            cv.push(Self::feo(tail));
        }
        let b = Tensor::cat(cm, 0).unsqueeze_dim::<4>(1);
        let x = activation::gelu(conv2d_forward(&b, &self.c1w, Some(&self.c1b), 2, 1));
        let x = activation::gelu(conv2d_forward(&x, &self.c2w, Some(&self.c2b), 2, 1));
        let x = activation::gelu(conv2d_forward(&x, &self.c3w, Some(&self.c3b), 2, 1));
        let [b2, c2, f2, t2] = x.dims();
        let r = x.permute([0, 3, 1, 2]).reshape([b2, t2, c2 * f2]);
        let co = self.co.forward(&r);
        let pe = self.pe.clone().slice([0..t2]).unsqueeze_dim::<3>(0);
        let co = co + pe;
        // Download and slice on host: avoids burn's per-slice kernel-launch overhead.
        let dm = co.dims()[2];
        let co_data = co.into_data();
        let co_vals: Vec<f32> = co_data.to_vec::<f32>().unwrap_or_else(|_| {
            co_data.to_vec::<half::f16>().expect("conv_out dtype").into_iter().map(|v| v.to_f32()).collect()
        });
        let n_total: usize = cv.iter().sum();
        let mut out = Vec::with_capacity(n_total * dm);
        for (idx, &v) in cv.iter().enumerate() {
            let base = idx * t2 * dm;
            out.extend_from_slice(&co_vals[base..base + v * dm]);
        }
        Tensor::<B, 2>::from_data(TensorData::new(out, [n_total, dm]), &mel.device())
    }

    pub(crate) fn forward(&self, mel: &Tensor<B, 2>) -> Result<Tensor<B, 2>> {
        let cs = self.config.n_window * 2;
        let tpc = Self::feo(cs);
        let cpw = self.config.n_window_infer / cs;
        let ws = tpc * cpw;
        let av = self.run_conv_stem(mel, cs);
        let mut h = Tensor::cat(av, 0).unsqueeze_dim::<3>(0);
        if log::log_enabled!(log::Level::Debug) {
            let hd = h.clone().into_data();
            let hv: Vec<f32> = hd.to_vec().unwrap_or_default();
            let first5: Vec<f32> = hv.iter().take(5).copied().collect();
            log::debug!("  conv_stem_out first5: {:?}  dims: {:?}", first5, h.dims());
        }
        for (_i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, Some(ws));
        }
        let h = self.lnp.forward(&h);
        let h = self.p2.forward(&activation::gelu(self.p1.forward(&h)));
        Ok(h.squeeze_dim(0))
    }

    pub(crate) fn forward_incremental(&self, mel: &Tensor<B, 2>, cache: &mut EncoderCache<B>) -> Result<Tensor<B, 2>> {
        let nf = mel.dims()[1]; let cs = self.config.n_window * 2;
        let tpc = Self::feo(cs); let cpw = self.config.n_window_infer / cs;
        let ws = tpc * cpw; let nfull = nf / cs; let tail = nf % cs;
        let nc = nfull + if tail > 0 { 1 } else { 0 };
        let ncw = nfull / cpw; let cw = cache.completed_windows.len();
        for wi in cw..ncw { let sc = wi * cpw; cache.completed_windows.push(self.ew(mel, cs, sc, cpw, ws)?); }
        cache.committed_chunks = ncw * cpw;
        let psc = ncw * cpw; let pnc = nc - psc;
        if pnc == 0 && !cache.completed_windows.is_empty() {
            return Ok(Tensor::cat(cache.completed_windows.iter().cloned().collect(), 0));
        }
        let partial = if pnc > 0 {
            let mut cv = Vec::new(); let mut cm = Vec::new();
            for i in 0..pnc {
                let ci = psc + i;
                if ci < nfull { let s = ci * cs; cm.push(mel.clone().slice([0..mel.dims()[0], s..s + cs]).unsqueeze_dim::<3>(0)); cv.push(tpc); }
                else if tail > 0 { let s = nfull * cs; let tm = mel.clone().slice([0..mel.dims()[0], s..s + tail]); let pad = Tensor::zeros([mel.dims()[0], cs - tail], &mel.device()); cm.push(Tensor::cat(vec![tm, pad], 1).unsqueeze_dim::<3>(0)); cv.push(Self::feo(tail)); }
            }
            if !cm.is_empty() {
                let b = Tensor::cat(cm, 0).unsqueeze_dim::<4>(1);
                let x = activation::gelu(conv2d_forward(&b, &self.c1w, Some(&self.c1b), 2, 1));
                let x = activation::gelu(conv2d_forward(&x, &self.c2w, Some(&self.c2b), 2, 1));
                let x = activation::gelu(conv2d_forward(&x, &self.c3w, Some(&self.c3b), 2, 1));
                let [b2, c2, f2, t2] = x.dims();
                let r = x.permute([0, 3, 1, 2]).reshape([b2, t2, c2 * f2]);
                let co = self.co.forward(&r);
                let pe = self.pe.clone().slice([0..t2]).unsqueeze_dim::<3>(0);
                let co = co + pe;
                let mut av: Vec<Tensor<B, 2>> = Vec::new();
                for (idx, &v) in cv.iter().enumerate() { av.push(co.clone().slice([idx..idx + 1, 0..v]).squeeze_dim(0)); }
                let mut h = Tensor::cat(av, 0).unsqueeze_dim::<3>(0);
                for layer in &self.layers { h = layer.forward(&h, Some(ws)); }
                let h = self.lnp.forward(&h);
                let h = self.p2.forward(&activation::gelu(self.p1.forward(&h)));
                Some(h.squeeze_dim(0))
            } else { None }
        } else { None };
        let mut ap: Vec<Tensor<B, 2>> = cache.completed_windows.clone();
        if let Some(p) = partial { ap.push(p); }
        if ap.is_empty() { anyhow::bail!("no audio tokens"); }
        Ok(Tensor::cat(ap, 0))
    }

    fn ew(&self, mel: &Tensor<B, 2>, cs: usize, sc: usize, nc: usize, ws: usize) -> Result<Tensor<B, 2>> {
        let tpc = Self::feo(cs);
        let mut cm = Vec::new();
        for i in 0..nc { let s = (sc + i) * cs; cm.push(mel.clone().slice([0..mel.dims()[0], s..s + cs]).unsqueeze_dim::<3>(0)); }
        let b = Tensor::cat(cm, 0).unsqueeze_dim::<4>(1);
        let x = activation::gelu(conv2d_forward(&b, &self.c1w, Some(&self.c1b), 2, 1));
        let x = activation::gelu(conv2d_forward(&x, &self.c2w, Some(&self.c2b), 2, 1));
        let x = activation::gelu(conv2d_forward(&x, &self.c3w, Some(&self.c3b), 2, 1));
        let [b2, c2, f2, t2] = x.dims();
        let r = x.permute([0, 3, 1, 2]).reshape([b2, t2, c2 * f2]);
        let co = self.co.forward(&r);
        let pe = self.pe.clone().slice([0..t2]).unsqueeze_dim::<3>(0);
        let co = co + pe;
        let mut av: Vec<Tensor<B, 2>> = Vec::new();
        for idx in 0..nc { av.push(co.clone().slice([idx..idx + 1, 0..tpc]).squeeze_dim(0)); }
        let mut h = Tensor::cat(av, 0).unsqueeze_dim::<3>(0);
        for layer in &self.layers { h = layer.forward(&h, Some(ws)); }
        let h = self.lnp.forward(&h);
        let h = self.p2.forward(&activation::gelu(self.p1.forward(&h)));
        Ok(h.squeeze_dim(0))
    }

    pub(crate) fn feo(ifr: usize) -> usize { let f = |l: usize| -> usize { (l - 1) / 2 + 1 }; f(f(f(ifr))) }
}
