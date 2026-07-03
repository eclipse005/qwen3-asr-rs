# qwen3-asr-rs

[Qwen3-ASR](https://github.com/QwenLM/Qwen3) 的纯 Rust 推理库，手写 CUDA + CPU 双后端，零深度学习框架依赖。

## 特性

- **双后端**：CUDA（cuBLAS + NVRTC 手写 kernel）和 CPU（gemm + rayon），运行时切换
- **零拷贝权重加载**：mmap safetensors
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
default = ["cuda"]      # CUDA + CPU 都编译进来
cuda = ["dep:cudarc"]   # CUDA 后端
cpu  = []               # CPU 后端（总是可用）
hub  = ["dep:reqwest"]  # HuggingFace Hub 下载
```

CPU-only 构建：`cargo build --no-default-features --features cpu`

## 模型下载

从 HuggingFace 下载：
- [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)

## License

MIT
