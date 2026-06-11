# qwen3-asr-burn

High-performance Qwen3-ASR inference in pure Rust on the [Burn](https://burn.dev) framework, with a hand-tuned CUDA path (cuBLAS + custom NVRTC kernels) that beats `candle-transformers` by ~40% on a single P104-100.

| Audio (0.6B model, P104-100) | RTFx |
|--------|------|
| 15 s English | **12.5×** |
| 30 s Chinese | **10.2×** |
| 90 s English | **10.8×** |
| 180 s English | **10.7×** |
| 180 s Chinese | **9.8×** |

(candle baseline on the same GPU: ~7.5×.)

The Burn frontend lets a single inference codebase target CUDA, ROCm, Metal, Vulkan and CPU.  The CUDA path delegates the hot loops to a hand-written engine in [`src/cudarc_engine.rs`](src/cudarc_engine.rs) and [`src/gpu_audio_encoder.rs`](src/gpu_audio_encoder.rs); other backends fall back to pure-Burn implementations in `src/decoder.rs` and `src/encoder.rs`.

## Prerequisites

- **Rust** ≥ 1.79 (`rustup default stable`)
- **CUDA Toolkit** 12.x with NVRTC and cuBLAS (kernels are compiled at runtime from `src/kernels/kernels.cu`)
  - `CUDA_PATH` env var must point at the toolkit root (the build picks up its `include/` for `cuda_fp16.h`)
  - On Windows: typically `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8` (auto-set by the installer)
- An NVIDIA GPU of compute capability ≥ 6.1 (Pascal or newer)

## Models

The CUDA path expects a Qwen3-ASR checkpoint laid out exactly like the HuggingFace `Qwen/Qwen3-ASR-*` repos (`config.json`, `model.safetensors[.index.json]`, `tokenizer.json`).

By default the tests look in `models/Qwen3-ASR-0.6B` and `models/Qwen3-ASR-1.7B` next to this README.  Override with env vars:

```bash
export QWEN3_ASR_MODEL_06_DIR=/path/to/Qwen3-ASR-0.6B
export QWEN3_ASR_MODEL_17_DIR=/path/to/Qwen3-ASR-1.7B
```

`models/` is `.gitignore`-d; download the checkpoints separately, e.g.:

```bash
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir models/Qwen3-ASR-0.6B
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir models/Qwen3-ASR-1.7B
```

## Build

```bash
cargo build --release --features cuda     # default
# or, for other backends:
# cargo build --release --no-default-features --features rocm
# cargo build --release --no-default-features --features metal
# cargo build --release --no-default-features --features vulkan
# cargo build --release --no-default-features --features cpu
```

## Benchmarks

Tests are `#[ignore]` by default so they don't run unsupervised.  Run a single audio:

```bash
cargo test --release --features cuda --test transcribe_burn test_q06_15s -- --ignored --nocapture
```

Run them all serially (avoid the `cargo test` parallel-test GPU contention):

```bash
cargo test --release --features cuda --test transcribe_burn -- \
  --ignored --nocapture --test-threads=1 \
  test_q06_15s test_q06_30s test_q06_90s test_q06_180s test_q06_180s_en
```

> Test names that prefix-match (e.g. `test_q06_180s` matches both `test_q06_180s` and `test_q06_180s_en`) will run in parallel and fight over the GPU — always pass `--test-threads=1` or list every test name explicitly.

The fixtures (`tests/fixtures/*.wav`) ship with the repo so you can reproduce the numbers above out of the box.

## Library use

```rust
use qwen3_asr_burn::{AsrInference, TranscribeOptions};

let device = qwen3_asr_burn::best_device();
let engine = AsrInference::load(std::path::Path::new("models/Qwen3-ASR-0.6B"), device)?;
let result = engine.transcribe("path/to/audio.wav", TranscribeOptions::default())?;
println!("[{}] {}", result.language, result.text);
```

## What's inside

- **`src/cudarc_engine.rs`** — GPU-resident text decoder.  cuBLAS for matmul, hand-written NVRTC kernels for RMSNorm/SiLU·up/softmax/rotary/embed/argmax/GQA-attention.  Flash-attention-style split-K kernel kicks in for long contexts.  All weights, KV cache and MRoPE tables live on the device.
- **`src/gpu_audio_encoder.rs`** — GPU audio encoder: im2col + cuBLAS conv2d for the stem, then 18 transformer layers reusing the same kernels as the text decoder.
- **`src/kernels/kernels.cu`** — every custom CUDA kernel in one file.  Compiled once at startup via NVRTC.
- **`src/encoder.rs`, `src/decoder.rs`** — pure-Burn implementations used by the non-CUDA backends (and by the streaming path).
- **`src/inference.rs`** — orchestration (mel extraction, prompt build, prefill + decode loop).
- **`HANDOFF.md`** — full architecture + optimization history (read this before optimizing further).

## Status / roadmap

- ✅ 0.6B and 1.7B correctness verified against reference outputs on the included fixtures.
- ✅ All CUDA optimization wins documented in `HANDOFF.md`; see §5 ("接手者：下一步优化方向") for what's still on the table (CUDA graphs, fused lm-head GEMV, conv-stem permute kernel, streaming.rs adaptation).
- 🚧 ROCm / Metal / Vulkan paths build but use the slower pure-Burn engine; they need their own equivalents of `cudarc_engine` to match the CUDA numbers.

## License

MIT — see `Cargo.toml`.
