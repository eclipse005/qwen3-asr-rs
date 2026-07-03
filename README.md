# qwen3-asr-rs

[Qwen3-ASR](https://github.com/QwenLM/Qwen3) 的 Rust 推理库。支持 CUDA 和 CPU 双后端，零深度学习框架依赖。

## 安装

```toml
[dependencies]
qwen3-asr = { git = "https://github.com/eclipse005/qwen3-asr-rs.git" }
```

CPU-only 构建：

```toml
qwen3-asr = { git = "https://github.com/eclipse005/qwen3-asr-rs.git", default-features = false }
```

## 使用

```rust
use qwen3_asr::{Backend, AsrInference, TranscribeOptions};

let model_dir = "path/to/Qwen3-ASR-0.6B";
let infer = AsrInference::load(model_dir, Backend::best())?;
let result = infer.transcribe("audio.wav", TranscribeOptions::default())?;
println!("{}", result.text);
```

## 模型下载

从 HuggingFace 下载 safetensors 格式的模型：

- [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)

## Features

| Feature | 说明 |
|---------|------|
| `cuda`（默认） | CUDA 后端，需要 CUDA 12.8+ |
| `cpu` | CPU 后端，始终可用 |
| `hub` | HuggingFace Hub 自动下载 |

## License

MIT
