# qwen3-asr

> Qwen3-ASR（0.6B / 1.7B）的纯 Rust 推理库。**手写 CUDA + CPU 双后端**，零深度学习框架依赖。

## 为什么不用 Candle / Burn？

原版 [`candle`](https://github.com/huggingface/candle) 是通用深度学习框架，对 ASR 推理场景有几个痛点：

- **冷启动慢**：candle 的图形调度和动态分发对单模型推理是纯开销
- **GEMV 单线程**：m=1 的 lm_head GEMM 在 candle 下走单线程，长文本解码严重受限于这块
- **内存占用高**：candle 的 Tensor 抽象层有额外引用计数和元数据开销
- **CUDA kernel 无法精细控制**：想加 FlashAttention 风格的 tiled online softmax、fused RMSNorm+Rotary 等优化，在 candle 里要绕很多路

本项目直接用 `cudarc`（CUDA driver binding）+ `cuBLAS` + `NVRTC` 手写所有 kernel，CPU 路径用 `gemm` + `rayon` 直接驱动。**热路径上没有任何深度学习框架**。

## 性能（vs Candle 原版）

| 指标 | Candle 原版 | 本项目 | 提升 |
|------|------------|--------|------|
| 180s 音频 CPU 推理 | ~50s | ~21.6s | **2.3x** |
| 15s 音频 CPU 推理 | ~2.3s | ~1.0s | **2.3x** |
| 180s 音频 GPU 推理 (P104-100) | ~12s | ~5s | **2.4x** |
| 峰值内存 (180s) | ~3GB | ~1.2GB | **2.5x** |
| 模型加载时间 | ~40s | ~5s | **8x**（mmap + 零拷贝） |

> 测试硬件：CPU = Intel Core Ultra 7 265K (Arrow Lake, 20c, AVX2)；GPU = P104-100 (Pascal sm_61, 8GB, 无 tensor core)。

## 特性

- **双后端，单一二进制**：CUDA + CPU 同时编译进同一个库，运行时通过 `Backend::Cuda` / `Backend::Cpu` 切换
- **CUDA 路径**：cuBLAS HGEMM + NVRTC 运行时编译的手写 kernel（fused RMSNorm、Rotary Embedding、FlashAttention 风格 tiled online softmax）
- **CPU 路径**：INT8 weight-only 量化（per-channel + AVX2 GEMV）+ gemm + rayon 并行
- **零拷贝权重加载**：mmap safetensors + `Bytes::from_owner`，加载 1.7B 模型 < 5s
- **流式识别**：支持 chunk-by-chunk 流式推理
- **确定性输出**：同输入多次运行逐 token 一致

## 快速开始

```rust
use qwen3_asr::{Backend, AsrInference, TranscribeOptions};

let model_dir = "path/to/Qwen3-ASR-0.6B";
let infer = AsrInference::load(model_dir, Backend::best())?;
let result = infer.transcribe("audio.wav", TranscribeOptions::default())?;
println!("{}", result.text);
```

## Features

```toml
default = ["cuda"]      # 端到端可用（CUDA + CPU 都编译进来）
cuda = ["dep:cudarc"]   # CUDA 后端
cpu  = []               # CPU 后端（总是可用）
hub  = ["dep:reqwest"]  # HuggingFace Hub 下载
```

CPU-only 构建：`cargo build --no-default-features --features cpu`

## 模型下载

从 HuggingFace 下载：
- [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)

转换为 safetensors 格式后放到 `models/` 目录。

## 项目结构

```
src/
├── backend.rs            # Backend 枚举（Cuda | Cpu）+ best() 调度
├── inference.rs          # 主推理循环：mel → embed → decode
├── cpu_audio_encoder.rs  # 手写 CPU 音频编码器（im2col + gemm + rayon）
├── cpu_engine.rs         # 手写 CPU 文本解码器（INT8 量化 + AVX2 GEMV）
├── cudarc_engine.rs      # 手写 GPU 文本解码器（cuBLAS + NVRTC kernel）
├── gpu_audio_encoder.rs  # 手写 GPU 音频编码器（cuBLAS + 自定义 conv2d）
├── kernels/kernels.cu    # 所有 CUDA kernel（NVRTC 运行时编译）
└── weights.rs            # safetensors mmap 零拷贝权重加载
```

## License

MIT
