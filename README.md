# qwen3-asr-rs

[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 的 Rust 推理库。支持 CUDA 和 CPU 双后端，零深度学习框架依赖。

Qwen3-ASR 是阿里通义千问开源的语音识别模型，支持 52 种语言和方言，提供 0.6B 和 1.7B 两种规格。

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

从 HuggingFace 下载 safetensors 格式的模型（权重版权归原作者）：

- [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)

官方项目与文档：

- 代码与说明：[QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- 模型集合：[Qwen3-ASR on Hugging Face](https://huggingface.co/collections/Qwen/qwen3-asr)

## Features

| Feature | 说明 |
|---------|------|
| `cuda`（默认） | CUDA 后端，需要 CUDA 12.8+ |
| `cpu` | CPU 后端，始终可用 |
| `hub` | HuggingFace Hub 自动下载 |

## Benchmark

在 NVIDIA P104-100（8 GiB）上，使用 `tests/fixtures` 中的音频对比 Rust 实现与 Python 原版（`qwen-asr` transformers backend）的 CUDA 性能。

| 模型 | 音频 | Rust 耗时 | Rust RTFx | Rust 显存峰值 | Python 耗时 | Python RTFx | Python 显存峰值 | Rust 加速比 |
|------|------|-----------|-----------|---------------|-------------|-------------|-----------------|-------------|
| 0.6B | 15s.wav | 0.63s | 23.90x | 1913 MiB | 3.61s | 4.15x | 1943 MiB | 5.75x |
| 0.6B | 30s.wav | 1.82s | 16.52x | 2105 MiB | 6.37s | 4.71x | 1969 MiB | 3.51x |
| 0.6B | 90s.wav | 5.49s | 16.39x | 2841 MiB | 17.89s | 5.03x | 2515 MiB | 3.26x |
| 0.6B | ja_89s.wav | 5.05s | 17.63x | 2841 MiB | 15.01s | 5.93x | 2513 MiB | 2.97x |
| 0.6B | 180s.wav | 12.51s | 14.39x | 3993 MiB | 40.58s | 4.44x | 3645 MiB | 3.24x |
| 0.6B | 180s_en.wav | 11.64s | 15.46x | 3961 MiB | 36.37s | 4.95x | 3615 MiB | 3.12x |
| 1.7B | 15s.wav | 1.65s | 9.07x | 4313 MiB | 4.12s | 3.64x | 4631 MiB | 2.49x |
| 1.7B | 30s.wav | 3.77s | 7.97x | 4505 MiB | 7.09s | 4.23x | 4663 MiB | 1.88x |
| 1.7B | 90s.wav | 10.43s | 8.63x | 5241 MiB | 23.16s | 3.89x | 4937 MiB | 2.22x |
| 1.7B | ja_89s.wav | 9.04s | 9.85x | 5241 MiB | 20.71s | 4.30x | 5031 MiB | 2.29x |
| 1.7B | 180s.wav | 22.55s | 7.98x | 6393 MiB | 51.21s | 3.52x | 6233 MiB | 2.27x |
| 1.7B | 180s_en.wav | 21.07s | 8.54x | 6329 MiB | 47.83s | 3.76x | 6197 MiB | 2.27x |

结论：Rust 实现全面快于 Python 原版，0.6B 约 3–6 倍，1.7B 约 2–2.5 倍；显存占用两者相当。

运行方式：

```bash
# Rust
cargo test --release --test bench_cuda -- --ignored --test-threads=1

# Python（需要 conda activate asr）
conda activate asr
python scripts/bench_original.py

# 生成对比报告
python scripts/summarize_bench.py target/rust_bench_cuda_v2.tsv target/python_bench_original_v2.tsv
```

详细日志和 TSV 数据保存在 `target/` 目录下。

## 致谢 / 原版出处

本仓库是 **独立的 Rust 推理实现**，用于加载并运行官方发布的 Qwen3-ASR 权重；**不是** Alibaba / Qwen 官方发行版，与原作者无隶属关系。

| 组件 | 原版 | 链接 | 协议（以官方页面为准） |
|------|------|------|------------------------|
| 模型权重 | Qwen3-ASR 0.6B / 1.7B | [HF 0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) · [HF 1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | Apache-2.0 |
| 官方推理与文档 | Qwen3-ASR | [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) | Apache-2.0 |

使用模型权重时请遵守原作者许可证；本仓库的 Rust 推理代码以本仓库 License 为准。

## License

MIT
