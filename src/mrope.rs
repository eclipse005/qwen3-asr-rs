//! Multi-dimensional Rotary Position Embedding (MRoPE) computation.
//!
//! Shared by both CPU and GPU text decoders. Returns `Vec<f32>` tables;
//! the GPU engine wraps the result in an f16 staging buffer.

/// Build MRoPE cosine/sine tables — returns `(cos, sin)` each of shape `[seq_len, head_dim]`.
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

pub(crate) fn build_contiguous_dim_map(s: &[usize], t: usize) -> Vec<usize> {
    let mut m = Vec::with_capacity(t);
    for (d, &sz) in s.iter().enumerate() { for _ in 0..sz { if m.len() >= t { break; } m.push(d); } }
    while m.len() < t { m.push(s.len() - 1); } m
}

pub(crate) fn build_interleaved_dim_map(s: &[usize], t: usize) -> Vec<usize> {
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
