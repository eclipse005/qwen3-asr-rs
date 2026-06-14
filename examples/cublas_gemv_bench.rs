//! cuBLAS GEMV bandwidth benchmark on P104.
//!
//! Measures the actual achieved bandwidth of cuBLAS f16 GEMV (n=1, decode-shaped)
//! for the 5 GEMV sizes used in one decode step. This determines whether decode's
//! 2× gap to the bandwidth ceiling (ROADMAP §1.4: 9-12ms/tok vs 3.75ms/tok floor)
//! is due to cuBLAS GEMV inefficiency or launch overhead.
//!
//! P104-100 theoretical bandwidth: ~200 GB/s (GDDR5X).
//!
//! If cuBLAS GEMV achieves >150 GB/s → cuBLAS is efficient, optimize elsewhere.
//! If cuBLAS GEMV achieves <100 GB/s → hand-written GEMV could be a big win.
//!
//! Usage: cargo run --release --example cublas_gemv_bench -- models\Qwen3-ASR-0.6B

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use cudarc::cublas::safe::{CudaBlas, Gemm, GemmConfig};
use cudarc::cublas::sys;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DriverError};
use half::f16;
use safetensors::SafeTensors;

const HS: usize = 1024;
const Q_DIM: usize = 16 * 128;      // 2048
const KV_DIM: usize = 8 * 128;       // 1024
const INTER: usize = 3072;
const FUSED_QKV_COLS: usize = Q_DIM + 2 * KV_DIM;  // 4096
const FUSED_GU_COLS: usize = 2 * INTER;             // 6144
const VOCAB: usize = 151936;

struct BenchGEMV {
    name: &'static str,
    rows: usize,  // N (output dim)
    cols: usize,  // K (input dim = HS)
    weight_bytes: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir: PathBuf = env::args()
        .nth(1)
        .unwrap_or_else(|| "models/Qwen3-ASR-0.6B".into())
        .into();

    let benches = [
        BenchGEMV { name: "qkv",       rows: FUSED_QKV_COLS, cols: HS, weight_bytes: FUSED_QKV_COLS * HS * 2 },
        BenchGEMV { name: "o_proj",    rows: HS,             cols: Q_DIM, weight_bytes: HS * Q_DIM * 2 },
        BenchGEMV { name: "gate_up",   rows: FUSED_GU_COLS,  cols: HS, weight_bytes: FUSED_GU_COLS * HS * 2 },
        BenchGEMV { name: "down_proj", rows: HS,             cols: INTER, weight_bytes: HS * INTER * 2 },
        BenchGEMV { name: "lm_head",   rows: VOCAB,          cols: HS, weight_bytes: VOCAB * HS * 2 },
    ];

    println!("=== cuBLAS GEMV bandwidth benchmark ===");
    println!("P104-100 theoretical bandwidth: ~200 GB/s\n");

    // === CUDA setup ===
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone())?;
    unsafe {
        sys::cublasSetMathMode(*blas.handle(), sys::cublasMath_t::CUBLAS_TENSOR_OP_MATH);
    }

    // Load real weights (BF16→F16).
    let buf = std::fs::read(model_dir.join("model.safetensors"))?;
    let st = SafeTensors::deserialize(&buf)?;
    let weight_names = [
        "thinker.model.layers.0.self_attn.q_proj.weight",
        "thinker.model.layers.0.self_attn.o_proj.weight",
        "thinker.model.layers.0.mlp.gate_proj.weight",
        "thinker.model.layers.0.mlp.down_proj.weight",
        "thinker.model.embed_tokens.weight",
    ];

    // For gate_up we need fused gate+up; for qkv we need fused q+k+v.
    // For simplicity, load the raw weight and pad/repeat to the bench size.
    fn load_f16(st: &SafeTensors, name: &str) -> Vec<f16> {
        let view = st.tensor(name).unwrap();
        view.data().chunks_exact(2).map(|c| {
            let b = u16::from_le_bytes([c[0], c[1]]);
            f16::from_f32(f32::from_bits((b as u32) << 16))
        }).collect()
    }

    // Build weight vectors matching bench sizes (pad with zeros if needed).
    let mut weights: Vec<Vec<f16>> = Vec::new();
    for (i, b) in benches.iter().enumerate() {
        if i == 0 {
            // qkv: fuse q+k+v
            let q = load_f16(&st, "thinker.model.layers.0.self_attn.q_proj.weight");
            let k = load_f16(&st, "thinker.model.layers.0.self_attn.k_proj.weight");
            let v = load_f16(&st, "thinker.model.layers.0.self_attn.v_proj.weight");
            let mut fused = Vec::with_capacity(b.rows * b.cols);
            fused.extend_from_slice(&q); fused.extend_from_slice(&k); fused.extend_from_slice(&v);
            assert_eq!(fused.len(), b.rows * b.cols, "qkv size mismatch");
            weights.push(fused);
        } else if i == 2 {
            // gate_up: fuse gate+up
            let g = load_f16(&st, "thinker.model.layers.0.mlp.gate_proj.weight");
            let u = load_f16(&st, "thinker.model.layers.0.mlp.up_proj.weight");
            let mut fused = Vec::with_capacity(b.rows * b.cols);
            fused.extend_from_slice(&g); fused.extend_from_slice(&u);
            assert_eq!(fused.len(), b.rows * b.cols, "gate_up size mismatch");
            weights.push(fused);
        } else {
            weights.push(load_f16(&st, weight_names[i]));
            assert_eq!(weights[i].len(), b.rows * b.cols, "{} size mismatch", b.name);
        }
    }

    println!("{:<12} {:>8} {:>12} {:>12} {:>10}", "GEMV", "rows", "weight(MB)", "time(ms)", "GB/s");
    println!("{}", "-".repeat(58));

    let mut total_ms = 0.0f64;
    let mut total_bytes = 0usize;

    for (i, b) in benches.iter().enumerate() {
        // Upload weight [rows, cols] row-major, input x [cols] = ones, output y [rows].
        let w_dev: CudaSlice<f16> = stream.clone_htod(&weights[i])?;
        let x_host = vec![f16::ONE; b.cols];
        let x_dev: CudaSlice<f16> = stream.clone_htod(&x_host)?;
        let mut y_dev: CudaSlice<f16> = stream.alloc_zeros(b.rows)?;

        // Warmup
        for _ in 0..3 {
            unsafe {
                blas.gemm(
                    GemmConfig {
                        transa: sys::cublasOperation_t::CUBLAS_OP_T,
                        transb: sys::cublasOperation_t::CUBLAS_OP_N,
                        m: b.rows as i32, n: 1, k: b.cols as i32,
                        alpha: f16::from_f32(1.0),
                        lda: b.cols as i32, ldb: b.cols as i32,
                        beta: f16::from_f32(0.0), ldc: b.rows as i32,
                    },
                    &w_dev, &x_dev, &mut y_dev,
                )?;
            }
        }
        stream.synchronize()?;

        // Benchmark: 50 iterations
        let iters = 50;
        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe {
                blas.gemm(
                    GemmConfig {
                        transa: sys::cublasOperation_t::CUBLAS_OP_T,
                        transb: sys::cublasOperation_t::CUBLAS_OP_N,
                        m: b.rows as i32, n: 1, k: b.cols as i32,
                        alpha: f16::from_f32(1.0),
                        lda: b.cols as i32, ldb: b.cols as i32,
                        beta: f16::from_f32(0.0), ldc: b.rows as i32,
                    },
                    &w_dev, &x_dev, &mut y_dev,
                )?;
            }
        }
        stream.synchronize()?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let gb = b.weight_bytes as f64 / 1e9;
        let gbps = gb / (ms / 1000.0);
        let pct = gbps / 200.0 * 100.0;

        println!("{:<12} {:>8} {:>12.1} {:>12.3} {:>10.1}  ({:.0}% of 200GB/s)",
            b.name, b.rows, b.weight_bytes as f64 / (1024.0*1024.0), ms, gbps, pct);

        total_ms += ms;
        total_bytes += b.weight_bytes;
    }

    println!("{}", "-".repeat(58));
    let total_gb = total_bytes as f64 / 1e9;
    println!("\n5 GEMVs total (one decode step, 1 layer worth of GEMVs):");
    println!("  total time:     {:.3}ms", total_ms);
    println!("  total weight:   {:.1}MB", total_bytes as f64 / (1024.0*1024.0));
    println!("  avg bandwidth:  {:.1} GB/s", total_gb / (total_ms / 1000.0));
    println!("\n  ×28 layers (qkv+o+gate_up+down only, no lm_head):");
    let per_layer_ms = total_ms - benches[4].weight_bytes as f64 / 1e9 / (total_gb / (total_ms/1000.0)) * 1000.0;
    let layer4_ms = (benches[0].weight_bytes + benches[1].weight_bytes + benches[2].weight_bytes + benches[3].weight_bytes) as f64 / 1e9;
    let bw_avg = total_gb / (total_ms / 1000.0);
    let decode_28_ms = layer4_ms / (total_gb / (total_ms/1000.0)) * 1000.0 * 28.0 + benches[4].weight_bytes as f64 / 1e9 / bw_avg * 1000.0;
    println!("  28-layer GEMV time (bandwidth floor): {:.2}ms", decode_28_ms);
    println!("  vs ROADMAP decode actual: 9-12ms/tok");
    println!("  → cuBLAS efficiency: {:.0}%", decode_28_ms / 10.5 * 100.0);

    let _ = per_layer_ms;
    Ok(())
}
