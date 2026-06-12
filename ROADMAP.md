# ROADMAP — qwen3-asr

> 写给下一个接手会话的 AI。读这一篇就能在 5 分钟内进入状态。

## 0. 项目是什么

Qwen3-ASR（0.6B / 1.7B）的 Rust 推理库。**双后端**：CUDA（cuBLAS + NVRTC 手写 kernel）和 CPU（gemm + rayon），均端到端可用。权重 f16 存储 + f32 计算，兼顾内存和精度。

## 1. 当前在哪

### 1.1 树

```
src/
├── backend.rs            # Backend 枚举（Cuda | Cpu）+ best() 调度
├── config.rs             # AsrConfig serde
├── error.rs              # AsrError / Result
├── hub.rs                # (可选) hub 模式下载
├── mel.rs                # mel spectrogram（CPU, f32）
├── inference.rs          # AsrInference 装配 + mel→embed→generate 主循环
├── raw_tensor.rs         # safetensors 原始字节 view（weight loading 用）
├── cpu_audio_encoder.rs  # ★ 手写 CPU 音频编码器（im2col + gemm + rayon + f16 权重）
├── cpu_engine.rs         # ★ 手写 CPU 文本解码器（gemm + rayon + f16 权重）
├── cudarc_engine.rs      # ★ 手写 GPU 文本解码器（cuBLAS + NVRTC kernel）+ DecodeScratch 复用
├── gpu_audio_encoder.rs  # ★ 手写 GPU 音频编码器（cuBLAS + 自定义 conv2d/im2col）
└── kernels/kernels.cu    # 所有 CUDA kernel（运行时 NVRTC 编译）
tests/
├── transcribe.rs          # 13 个 #[ignore] CUDA 集成测试（0.6B/1.7B × 各种时长）
├── cpu_transcribe.rs      # 7 个 CPU 集成测试（0.6B × 全 fixture）
├── cpu_90s.rs             # 90s 单独 CPU 测试（独立进程跑，可单独 benchmark）
└── fixtures/              # 7 个 wav: sample1 / 15s / 30s / 90s / 89s_ja / 180s / 180s_en
```

### 1.2 公共 API

`lib.rs` 导出 `Backend`、`AsrError` / `Result`、`AsrInference` / `TranscribeOptions` / `TranscribeResult`、`load_audio_wav`。**没有** `StreamingOptions` / `StreamingState` / `best_device()`。

### 1.3 Features

```toml
default = ["cuda"]      # 端到端可用
cuda = ["dep:cudarc"]
cpu  = ["dep:gemm", "dep:rayon"]  # 端到端可用（f16 权重 + f32 计算）
hub  = ["dep:reqwest"]
```

### 1.4 性能（基准线，P104-100）

#### CPU 0.6B（Ryzen/Intel，f16 权重存储）

| 音频 | RTFx | 耗时 |
|---|---|---|
| 15s 英文 | 4.16× | 3.6s |
| 30s 中文 | 3.72× | 8.1s |
| 90s 英文 | 3.87× | 23.2s |
| 89s 日文 | 4.22× | 21.1s |
| 180s 中文 | 4.18× | 43.0s |
| 180s 英文 | 4.15× | 43.3s |

#### CUDA 0.6B（P104-100，Pascal sm_61）

| 音频 | RTFx |
|---|---|
| 15s 英文 | 24.23× |
| 30s 中文 | 16.00× |
| 90s 英文 | 16.48× |
| 89s 日文 | 17.51× |
| 180s 中文 | 14.20× |
| 180s 英文 | 15.19× |

#### CUDA 1.7B（P104-100）

| 音频 | RTFx |
|---|---|
| 15s 英文 | 8.93× |
| 30s 中文 | 7.74× |
| 90s 英文 | 8.56× |
| 89s 日文 | 9.74× |
| 180s 中文 | 7.94× |
| 180s 英文 | 8.45× |

数字来源：
- CUDA: `cargo test --release --test transcribe -- --ignored --nocapture --test-threads=1`
- CPU: `cargo test --release --no-default-features --features cpu --test cpu_transcribe -- --nocapture --test-threads=1`

### 1.5 内存

90s 音频 0.6B CPU 峰值工作集约 **3,482 MB**（f16 权重），比纯 f32 的 4,657 MB 节省 ~25%。

## 2. 关键架构事实

### 2.1 权重存储：f16 + f32 计算

两个后端统一模式：
- **存储**：`CpuWeightF16 { data: Vec<half::f16>, rows, cols }`（音频编码器 + 文本解码器共用类型名，分别定义在各自的文件里）
- **CPU 解码** (m=1)：`linear_gemv_f16` 直接读 f16 权重，寄存器内转 f32 累加 → 省一半内存带宽，比 f32 GEMV 快 5-9%
- **CPU prefill** (m>1)：rayon `par_iter` 批量 f16→f32 转换后走 gemm crate 的 f32 GEMM
- **CPU 音频编码器**：顺序 f16→f32 转换（权重小，~100 次/forward，rayon 反而慢）
- **GPU**：cuBLAS f16 GEMM 直接消费 f16 权重

### 2.2 CPU 音频编码器（`cpu_audio_encoder.rs`）

```
ConvStem: 3× conv2d (im2col + gemm) + GELU + permute + positional encoding
18× CpuAudioLayer:
  LayerNorm → windowed self-attention (scalar + rayon across heads) → residual
  LayerNorm → FFN (gate + up + SiLU + down) → residual
```

- im2col 生成列矩阵后 gemm 做卷积
- Attention: 标量循环 + rayon 跨 (batch, head) 对并行（GEMM per-head 在小窗口上反而慢）
- f16 权重在 `CpuAudioLinear::forward` 和 `conv_block` 里转 f32 后计算

### 2.3 CPU 文本解码器（`cpu_engine.rs`）

```
28 层 decoder，每层:
  RMSNorm → QKV fused linear → MRoPE → windowed attention → O-proj → residual
  RMSNorm → gate_up fused linear → SiLU → down → residual
+ final RMSNorm → embed_table linear → argmax
```

- decode (m=1): `linear_gemv_f16` 直接读 f16，寄存器内转 f32
- prefill (m>1): rayon 批量 f16→f32，然后 gemm crate f32 GEMM
- `linear_accum_f16`: 带 residual 累加的线性层（cuBLAS beta=1 思路的 CPU 版）

### 2.4 GPU 解码器主路径（`inference.rs::generate_cuda`）

```
prefill（一次）：
  build hidden_states (CPU splice audio embeds) → upload
  cos/sin MRoPE 表预计算（CPU）+ upload
  GpuKvCache 预分配
  decoder.forward(hs, cos, sin, kv, 0, true, true)  → logits [1, 1, vocab]
  argmax → token_buf[0]
decode loop（每步）：
  embed_id_from_gpu_slot_into(embed_table, token_buf, 0, h_buf)
  forward_decode_scratch(h_buf, cos, sin, kv, current_pos, token_buf, scratch)
    = 28 layers × (rms_norm + linear QKV + qkv_extract + attn + O + rms + gate_up + silu + down)
    + final_norm + linear(embed_table) + argmax
  current_pos += 1
  loop until EOS
```

**关键不变量**：
- KV cache 一次性预分配到 `total_positions = seq_len + max_new_tokens`，decode 步不重新分配
- scratch buffer（`DecodeScratch`）也一次性预分配，decode 步 alloc 数为 0
- `token_buf[0]` 是唯一跨步传递的 i32，不做 htod
- decode 步 28×1 步用 `fused_gqa_decode_split`（chunk=256 或 512），prefill 走 `fused_gqa_decode_split`（当 cur_len>1024）否则走非 split

### 2.5 GPU 优化历史

数字对应 0.6B 15s 英文 cold-start RTFx，从低到高：

1. baseline burn-cubecl → 0.25x
2. 手写 GpuTensor + cuBLAS：0.55x
3. NVRTC element-wise + GPU KV cache：1.4x
4. GPU conv stem（im2col + cuBLAS）：6.4x
5. fused_gqa_decode 融合：7.6x
6. fused QKV extract + RMSNorm + rotary + cache 写：9.8x
7. linear_gpu_accum（cuBLAS beta=1）：10.6x
8. fused_gqa_decode_split（flash-attn 2-kernel）：12.5x
9. **alloc_uninit_f16（Pascal driver enqueue 限速 → memset 占 80% launch time）**：23× — 最大单步收益 ← `6251120`
10. skip clone_tensor at O-proj：23.5× ← `50d2524`
11. GPU-resident next-token：23.7× ← `4675c74`
12. fused QKV extract kernel：24× ← `f1df993`

## 3. 卡点 / 已知问题

### 3.1 [已删] CUDA Graph — 硬件不支持（sm_61），代码已清理

### 3.2 Conv stem 走 CPU detour

`gpu_audio_encoder.rs::GpuConvStem::forward` 末尾：GPU→CPU download→permute `[b,c,f,t]→[b,t,c,f]`→重新 upload。**单次推理 ~5-10ms**。修法：写一个 4D permute kernel 替换。

### 3.3 已知 dead code（rustc 警告）

- `gpu_audio_encoder.rs::GpuAudioEncoder::run_transformer` — 保留，未来 streaming 入口
- `inference.rs::AsrInferenceInner::tokenizer_decode` — 保留，debug/streaming 复用
- `inference.rs::AsrInferenceInner::decode_result` — **不是 dead code**，rustc 误报
- `raw_tensor.rs::to_f32_vec` / `as_f32` — CPU 路径改用 `as_f16()`，GPU 不用 f32，但保留作为工具方法

### 3.4 streaming API

旧的 `StreamingState` 已删。需要重新设计 chunked 音频 → 文本解码器的 prefix + overlap 策略。

### 3.5 测试 fixture

7 个 wav 文件在 `tests/fixtures/`。无自动下载脚本。

### 3.6 lm_head GEMV + argmax 融合

kernel 存在（`lm_head_gemv_argmax_f16`）但注释说"目前输给 cuBLAS"，未用。值得重测。

## 4. 已完成

- [x] CPU 音频编码器：conv stem + 18 层 transformer（im2col + gemm + rayon）
- [x] CPU 文本解码器：28 层 decoder（gemm + rayon）
- [x] CPU 端到端 transcribe 全部 fixture 通过
- [x] f16 权重存储（音频编码器 + 文本解码器），内存节省 ~25%
- [x] GPU 全部 13 个集成测试通过
- [x] CUDA Graph 死代码清理
- [x] 脱离 burn 框架

## 5. 下一步规划

### 5.1 P1：CUDA 小优化（明确收益）

- [ ] **Conv stem GPU permute kernel**（§3.2）— 消除 CPU detour ~5-10ms
- [ ] **lm_head 融合重测**（§3.6）— 可能已比纯 cuBLAS 快
- [ ] dead code warning 清理（`#[allow(dead_code)]` 或删）

### 5.2 P2：CPU 优化

- [ ] **AVX2 手写 GEMV** — 当前 decode 用标量循环读 f16→f32 累加，SIMD 可加速 2-4×
- [ ] **im2col SIMD** — 卷积 im2col 生成目前是标量赋值
- [ ] **注意力微优化** — 目前标量 + rayon 跨 head，可尝试分块矩阵乘
- [ ] 更激进的 rayon 策略（跨层并行？prefill 阶段流水线？）

### 5.3 P3：功能

- [ ] streaming API 重建（§3.4）
- [ ] fixture 自动下载脚本（§3.5）

### 5.4 P4：CUDA 极限

- [ ] PinnedHostSlice 异步 argmax（`cudaStreamAddCallback`，省 ~50µs/step）
- [ ] KV cache 预 fetch（prefill 后第一个 decode 步不等 cuBLAS）
- [ ] cuBLASLt 替换（Pascal f16 GEMM ~10% 提升空间）

### 5.5 不做

- **CUDA Graph** — sm_61 不支持
- **ROCm / Metal / Vulkan** — 已明确放弃，cudarc 专属
- **回到 burn 框架** — 这次重构的目的就是脱离它

## 6. 一些不能忘的事实

- **NVRTC 编译需要 CUDA_PATH**（Windows: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`）
- **cudarc 0.19**（`cuda-12080` feature）
- **`__launch_bounds__` 在所有 kernel 上**（Pascal sm_61 register pressure 优化）
- **f16 + f32 累加**是统一模式，不要降级
- **Pascal driver 有 enqueue 限速**（alloc_zeros → memset 触发），alloc_uninit 收益巨大
- **测试多线程跑会抢 GPU** — 必须 `--test-threads=1`
- **`h_buf` 跨步复用**：decode 步 28 层改 `h` in-place，final_norm 写到 `scratch.final_norm`，不修改 `h`
- **CPU 音频编码器**：attention 用标量循环 + rayon 跨 head 并行，不要换 GEMM per-head（小窗口 overhead 更大）
- **CPU f16→f32 转换**：音频编码器顺序转（100+ 次小权重，rayon 开销大于收益）；文本解码器 prefill 用 rayon 转（大权重 ~1.2GB，rayon 明显更快）
- **`half` crate v2** 是无条件依赖（不是 optional），GPU 和 CPU 路径都用

## 7. 给接手 AI 的具体操作

```
1. 读 §0-§2（本文件）
2. 确认想改的方向（CPU / CUDA），看 §5 对应优先级
3. 改前跑对应基线：
   CUDA: cargo test --release --test transcribe -- --ignored --nocapture --test-threads=1 test_q06_15s
   CPU:  cargo test --release --no-default-features --features cpu --test cpu_transcribe -- --nocapture --test-threads=1 test_cpu_15s
4. 改完跑全量测试验证不退化
5. 更新 ROADMAP 里的 RTFx 数字
```
