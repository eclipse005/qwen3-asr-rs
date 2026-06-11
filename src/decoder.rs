use anyhow::Result;
use burn::tensor::{activation, DType, Int, Tensor, TensorData};
use burn::tensor::backend::Backend;
use std::collections::HashMap;

use crate::config::TextDecoderConfig;
use crate::encoder::{safe_attention, LinearW};

// ─── Weight helpers ────────────────────────────────────────────────

fn get_w<B: Backend, const D: usize>(weights: &HashMap<String, TensorData>, name: &str, device: &B::Device) -> Result<Tensor<B, D>> {
    weights.get(name).ok_or_else(|| anyhow::anyhow!("weight not found: {}", name))
        .map(|d| Tensor::from_data(d.clone(), device))
}

fn load_linear<B: Backend>(weights: &HashMap<String, TensorData>, prefix: &str, device: &B::Device) -> Result<LinearW<B>> {
    Ok(LinearW::new(get_w(weights, &format!("{}.weight", prefix), device)?, None))
}

// ─── Manual RmsNorm (f32-precision rms computation) ────────────────

pub(crate) struct ManualRmsNorm<B: Backend> {
    weight: Tensor<B, 1>, eps: f64, size: usize,
}

impl<B: Backend> ManualRmsNorm<B> {
    pub fn load(weights: &HashMap<String, TensorData>, prefix: &str, size: usize, eps: f64, device: &B::Device) -> Result<Self> {
        Ok(Self { weight: get_w(weights, &format!("{}.weight", prefix), device)?, eps, size })
    }
    pub fn forward<const D: usize>(&self, x: &Tensor<B, D>) -> Tensor<B, D> {
        let dtype = x.dtype();
        let last = D - 1;
        // Full f32 cast for rms computation — required for numerical stability on f16 CUDA
        let rms = (x.clone().cast(DType::F32).square().mean_dim(last) + self.eps).sqrt();
        let mut ws = [1; D]; ws[D - 1] = self.size;
        (x.clone() / rms.cast(dtype)) * self.weight.clone().reshape(ws)
    }
}

// ─── MRoPE ─────────────────────────────────────────────────────────

pub(crate) fn compute_mrope_cos_sin<B: Backend>(
    pos: &[Vec<i64>; 3], hd: usize, rt: f64, ms: &[usize], il: bool, device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let hh = hd / 2; let sl = pos[0].len();
    let inv: Vec<f64> = (0..hh).map(|i| 1.0 / rt.powf(2.0 * i as f64 / hd as f64)).collect();
    let dm = if il { build_interleaved_dim_map(ms, hh) } else { build_contiguous_dim_map(ms, hh) };
    let mut cv = vec![0.0f32; sl * hd]; let mut sv = vec![0.0f32; sl * hd];
    for t in 0..sl { for j in 0..hh {
        let a = pos[dm[j]][t] as f64 * inv[j];
        cv[t * hd + j] = a.cos() as f32; sv[t * hd + j] = a.sin() as f32;
        cv[t * hd + j + hh] = a.cos() as f32; sv[t * hd + j + hh] = a.sin() as f32;
    }}
    (Tensor::from_data(TensorData::new(cv, [sl, hd]), device), Tensor::from_data(TensorData::new(sv, [sl, hd]), device))
}

fn build_contiguous_dim_map(s: &[usize], t: usize) -> Vec<usize> {
    let mut m = Vec::with_capacity(t);
    for (d, &sz) in s.iter().enumerate() { for _ in 0..sz { if m.len() >= t { break; } m.push(d); } }
    while m.len() < t { m.push(s.len() - 1); } m
}

fn build_interleaved_dim_map(s: &[usize], t: usize) -> Vec<usize> {
    let nd = s.len(); let mut m = Vec::with_capacity(t); let mut c = vec![0usize; nd];
    while m.len() < t { let pv = m.len(); for d in 0..nd { if m.len() >= t { break; } if c[d] < s[d] { m.push(d); c[d] += 1; } } if m.len() == pv { break; } } m
}

fn apply_rotary_emb<B: Backend>(x: &Tensor<B, 4>, cos: &Tensor<B, 4>, sin: &Tensor<B, 4>) -> Tensor<B, 4> {
    let xr = rotate_half(x); x.clone() * cos.clone() + xr * sin.clone()
}

fn rotate_half<B: Backend>(x: &Tensor<B, 4>) -> Tensor<B, 4> {
    let [b0, b1, b2, l] = x.dims(); let h = l / 2;
    let x1 = x.clone().slice([0..b0, 0..b1, 0..b2, 0..h]);
    let x2 = x.clone().slice([0..b0, 0..b1, 0..b2, h..l]);
    Tensor::cat(vec![x2 * (-1.0f64), x1], 3)
}

fn repeat_kv<B: Backend>(x: Tensor<B, 4>, n: usize) -> Tensor<B, 4> {
    if n == 1 { return x; }
    let [b, nkv, s, hd] = x.dims();
    x.unsqueeze_dim::<5>(2).expand([b, nkv, n, s, hd]).reshape([b, nkv * n, s, hd])
}

// ─── KV Cache ──────────────────────────────────────────────────────

pub(crate) struct KvCache<B: Backend> {
    pub k: Vec<Tensor<B, 4>>,
    pub v: Vec<Tensor<B, 4>>,
}

impl<B: Backend> KvCache<B> {
    pub(crate) fn new(_n: usize, _max_len: usize, _nkvh: usize, _hd: usize, _device: &B::Device) -> Self {
        Self { k: vec![], v: vec![] }
    }
}

// ─── Fused QKV Linear ─────────────────────────────────────────────
// Combines Q, K, V projections into a single matmul to reduce kernel launches.

struct FusedQkv<B: Backend> {
    weight_t: Tensor<B, 3>,  // [1, hidden_size, q_dim + 2*kv_dim]
    q_dim: usize, kv_dim: usize,
}

impl<B: Backend> FusedQkv<B> {
    fn load(w: &HashMap<String, TensorData>, p: &str, nqh: usize, nkvh: usize, hd: usize, device: &B::Device) -> Result<Self> {
        let qw: Tensor<B, 2> = get_w(w, &format!("{}.q_proj.weight", p), device)?;
        let kw: Tensor<B, 2> = get_w(w, &format!("{}.k_proj.weight", p), device)?;
        let vw: Tensor<B, 2> = get_w(w, &format!("{}.v_proj.weight", p), device)?;
        // Concatenate along output dim: [q_dim+2*kv_dim, hidden_size]
        let fused = Tensor::cat(vec![qw, kw, vw], 0);
        let [out_dim, inp_dim] = fused.dims();
        Ok(Self {
            weight_t: fused.transpose().reshape([1, inp_dim, out_dim]),
            q_dim: nqh * hd, kv_dim: nkvh * hd,
        })
    }

    /// Returns (q, k, v) each as [b, s, dim]
    fn forward(&self, x: &Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let [b, s, _] = x.dims();
        let mut ws = [1; 3]; ws[1] = self.weight_t.dims()[1]; ws[2] = self.weight_t.dims()[2];
        let qkv = x.clone().matmul(self.weight_t.clone().reshape(ws)); // [b, s, q_dim+2*kv_dim]
        let q = qkv.clone().slice([0..b, 0..s, 0..self.q_dim]);
        let k = qkv.clone().slice([0..b, 0..s, self.q_dim..self.q_dim + self.kv_dim]);
        let v = qkv.clone().slice([0..b, 0..s, self.q_dim + self.kv_dim..self.q_dim + 2 * self.kv_dim]);
        (q, k, v)
    }
}

// ─── Fused Gate-Up Linear ─────────────────────────────────────────
// Combines gate_proj and up_proj into a single matmul.

struct FusedGateUp<B: Backend> {
    weight_t: Tensor<B, 3>,  // [1, hidden_size, 2*intermediate_size]
    intermediate_size: usize,
}

impl<B: Backend> FusedGateUp<B> {
    fn load(w: &HashMap<String, TensorData>, p: &str, device: &B::Device) -> Result<Self> {
        let gw: Tensor<B, 2> = get_w(w, &format!("{}.gate_proj.weight", p), device)?;
        let uw: Tensor<B, 2> = get_w(w, &format!("{}.up_proj.weight", p), device)?;
        let fused = Tensor::cat(vec![gw, uw], 0);
        let [out_dim, inp_dim] = fused.dims();
        Ok(Self {
            weight_t: fused.transpose().reshape([1, inp_dim, out_dim]),
            intermediate_size: out_dim / 2,
        })
    }

    /// Returns (gate, up) each as [b, s, intermediate_size]
    fn forward(&self, x: &Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [b, s, _] = x.dims();
        let mut ws = [1; 3]; ws[1] = self.weight_t.dims()[1]; ws[2] = self.weight_t.dims()[2];
        let gu = x.clone().matmul(self.weight_t.clone().reshape(ws)); // [b, s, 2*is]
        let gate = gu.clone().slice([0..b, 0..s, 0..self.intermediate_size]);
        let up = gu.clone().slice([0..b, 0..s, self.intermediate_size..2 * self.intermediate_size]);
        (gate, up)
    }
}

// ─── Text Attention ────────────────────────────────────────────────

struct TextAttention<B: Backend> {
    qkv: FusedQkv<B>, op: LinearW<B>,
    qn: ManualRmsNorm<B>, kn: ManualRmsNorm<B>,
    nqh: usize, nkvh: usize, hd: usize,
}

impl<B: Backend> TextAttention<B> {
    fn load(w: &HashMap<String, TensorData>, p: &str, nqh: usize, nkvh: usize, hd: usize, eps: f64, d: &B::Device) -> Result<Self> {
        Ok(Self {
            qkv: FusedQkv::load(w, p, nqh, nkvh, hd, d)?,
            op: load_linear(w, &format!("{}.o_proj", p), d)?,
            qn: ManualRmsNorm::load(w, &format!("{}.q_norm", p), hd, eps, d)?,
            kn: ManualRmsNorm::load(w, &format!("{}.k_norm", p), hd, eps, d)?,
            nqh, nkvh, hd,
        })
    }

    fn forward(&self, x: &Tensor<B, 3>, cos: &Tensor<B, 4>, sin: &Tensor<B, 4>, kvc: &mut KvCache<B>, layer_idx: usize, use_causal: bool) -> Tensor<B, 3> {
        let [b, s, _] = x.dims();
        // Fused QKV projection — single matmul instead of 3
        let (q, k, v) = self.qkv.forward(x);
        let q = q.reshape([b, s, self.nqh, self.hd]).swap_dims(1, 2);
        let k = k.reshape([b, s, self.nkvh, self.hd]).swap_dims(1, 2);
        let v = v.reshape([b, s, self.nkvh, self.hd]).swap_dims(1, 2);
        let q = apply_rotary_emb(&self.qn.forward(&q), cos, sin);
        let k = apply_rotary_emb(&self.kn.forward(&k), cos, sin);

        // Update KV cache
        let (k_full, v_full) = if layer_idx < kvc.k.len() {
            let k_cat = Tensor::cat(vec![kvc.k[layer_idx].clone(), k], 2);
            let v_cat = Tensor::cat(vec![kvc.v[layer_idx].clone(), v], 2);
            (k_cat, v_cat)
        } else {
            (k, v)
        };
        if layer_idx >= kvc.k.len() {
            kvc.k.push(k_full.clone());
            kvc.v.push(v_full.clone());
        } else {
            kvc.k[layer_idx] = k_full.clone();
            kvc.v[layer_idx] = v_full.clone();
        }
        let nr = self.nqh / self.nkvh;
        let k_rep = repeat_kv(k_full, nr);
        let v_rep = repeat_kv(v_full, nr);

        let out = safe_attention(q, k_rep, v_rep, use_causal && s > 1);
        self.op.forward(&out.swap_dims(1, 2).reshape([b, s, self.nqh * self.hd]))
    }
}

// ─── SwiGLU MLP ────────────────────────────────────────────────────

struct TextMlp<B: Backend> {
    gate_up: FusedGateUp<B>,
    dp: LinearW<B>,
}

impl<B: Backend> TextMlp<B> {
    fn load(w: &HashMap<String, TensorData>, p: &str, d: &B::Device) -> Result<Self> {
        Ok(Self {
            gate_up: FusedGateUp::load(w, p, d)?,
            dp: load_linear(w, &format!("{}.down_proj", p), d)?,
        })
    }
    fn forward(&self, x: &Tensor<B, 3>) -> Tensor<B, 3> {
        let (gate, up) = self.gate_up.forward(x);
        self.dp.forward(&(activation::silu(gate) * up))
    }
}

// ─── Text Decoder Layer ────────────────────────────────────────────

struct TextDecoderLayer<B: Backend> {
    iln: ManualRmsNorm<B>, attn: TextAttention<B>, pln: ManualRmsNorm<B>, mlp: TextMlp<B>,
}

impl<B: Backend> TextDecoderLayer<B> {
    fn load(w: &HashMap<String, TensorData>, p: &str, nqh: usize, nkvh: usize, hd: usize, hs: usize, eps: f64, d: &B::Device) -> Result<Self> {
        Ok(Self {
            iln: ManualRmsNorm::load(w, &format!("{}.input_layernorm", p), hs, eps, d)?,
            attn: TextAttention::load(w, &format!("{}.self_attn", p), nqh, nkvh, hd, eps, d)?,
            pln: ManualRmsNorm::load(w, &format!("{}.post_attention_layernorm", p), hs, eps, d)?,
            mlp: TextMlp::load(w, &format!("{}.mlp", p), d)?,
        })
    }
    fn forward(&self, x: &Tensor<B, 3>, cos: &Tensor<B, 4>, sin: &Tensor<B, 4>, kvc: &mut KvCache<B>, layer_idx: usize, use_causal: bool) -> Tensor<B, 3> {
        let normed = self.iln.forward(x);
        let h = self.attn.forward(&normed, cos, sin, kvc, layer_idx, use_causal);
        let x = x.clone() + h;
        x.clone() + self.mlp.forward(&self.pln.forward(&x))
    }
}

// ─── Text Decoder ─────────────────────────────────────────────────

pub(crate) struct TextDecoder<B: Backend> {
    pub embed_tokens: Tensor<B, 2>,
    layers: Vec<TextDecoderLayer<B>>,
    norm: ManualRmsNorm<B>,
    pub lm_head: LinearW<B>,
}

impl<B: Backend> TextDecoder<B> {
    pub(crate) fn load(weights: &HashMap<String, TensorData>, prefix: &str, config: &TextDecoderConfig, device: &B::Device) -> Result<Self> {
        let et: Tensor<B, 2> = get_w(weights, &format!("{}.embed_tokens.weight", prefix), device)?;
        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            layers.push(TextDecoderLayer::load(weights, &format!("{}.layers.{}", prefix, i), config.num_attention_heads, config.num_key_value_heads, config.head_dim, config.hidden_size, config.rms_norm_eps, device)?);
        }
        Ok(Self { embed_tokens: et.clone(), layers, norm: ManualRmsNorm::load(weights, &format!("{}.norm", prefix), config.hidden_size, config.rms_norm_eps, device)?, lm_head: LinearW::new(et, None) })
    }

    pub(crate) fn embed(&self, input_ids: &Tensor<B, 1, Int>) -> Tensor<B, 2> {
        self.embed_tokens.clone().select(0, input_ids.clone())
    }

    pub(crate) fn forward(&self, hs: &Tensor<B, 3>, cos: &Tensor<B, 2>, sin: &Tensor<B, 2>, kvc: &mut KvCache<B>, use_causal: bool, llo: bool) -> Tensor<B, 3> {
        let sl = hs.dims()[1]; let mut h = hs.clone();
        let cos4 = cos.clone().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);
        let sin4 = sin.clone().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &cos4, &sin4, kvc, i, use_causal);
        }
        h = self.norm.forward(&h);
        if llo && sl > 1 { h = h.clone().slice([0..1, sl - 1..sl]); }
        self.lm_head.forward(&h)
    }
}
