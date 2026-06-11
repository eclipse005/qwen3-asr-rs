# Burn Qwen3-ASR 优化 — 交接文档（接手请先读）

> **项目目标**：用 `burn 0.21 + cubecl 0.10` 重写 Qwen3-ASR 推理，**跨后端统一编译**（CUDA / ROCm / Metal / Vulkan），目标硬件 NVIDIA P104-100（sm_61, 8GB）。
> **当前状态（2026-06-10）**：CUDA f16 路径已**全 GPU 化**，0.6B 模型 15s 英文 RTFx **7.57x（已超 candle 7.5x）**，30s 中文 6.10x，90s 6.09x，转录正确。

---

## 0. 30 秒读懂项目

| 维度 | 当前状态 |
|------|---------|
| **后端** | `burn::backend::cuda::CubeBackend<CudaRuntime, half::f16, i32, u8>` — 仅用于 burn Tensor 接口（音频编码器入口 mel→Tensor、conv stem fallback for streaming）|
| **CUDA 引擎** | `cudarc 0.19`（cuBLAS + NVRTC）+ 手写 CUDA kernels；文本解码器与音频编码器主路径**完全 GPU 常驻**（CudaSlice<f16>）|
| **精度** | **f16** 端到端 |
| **设备** | `burn::backend::cuda::CudaDevice`（GPU 0） |
| **依赖** | `burn = "0.21" + cuda`、`burn-cubecl = "0.21"`、`cubecl = "0.10"`、`half = "2"`、`cudarc = "0.19" + cublas/f16` |
| **CUDA 工具链** | 需要 `CUDA_PATH` 环境变量指向 CUDA Toolkit（NVRTC 要 include `cuda_fp16.h`） |
| **模型** | Qwen3-ASR-0.6B ✅ 已验证（15s 英文 + 30s 中文 + 90s 英文）；Qwen3-ASR-1.7B ⏳ 未测 |

### 性能（0.6B，P104-100，cold-start 单次运行）

| 音频 | 起点 | **现在 RTFx** | candle CUDA |
|------|-----|-----|-----|
| 15s 英文 | 0.44x | **23.50x** ✅ | ~7.5x |
| 30s 中文 | 0.25x | **20.32x** ✅ | ~7.5x |
| 90s 英文 | — | **18.00x** ✅ | — |
| 90s 日文 | — | **19.53x** ✅ | — |
| 180s 英文 | — | **16.92x** ✅ | — |
| 180s 中文 | — | **15.62x** | — |
| 1.7B-15s | — | 11.37x | — |
| 1.7B-30s | — | 10.05x | — |

> **关键拐点**：从 0.25x → 20.32x 是约 **80x 提速**，0.6B 全部测试 ≥ 15x。

---

## 1. 项目结构

```
D:\qwen3-asr-burn\
├── Cargo.toml          # burn+cubecl+half+cudarc 依赖，default = ["cuda"]
├── cubecl.toml         # autotune 配置（level = "full"）
├── HANDOFF.md          # 本文件
├── src\
│   ├── lib.rs                 # Backend/Device cfg 分支
│   ├── config.rs              # AsrConfig serde 反序列化
│   ├── error.rs               # AsrError/Result
│   ├── hub.rs                 # (可选) hub 模式下载
│   ├── mel.rs                 # mel spectrogram (CPU, f32)
│   ├── encoder.rs             # Burn 版 AudioEncoder（streaming.rs forward_incremental 使用；cuda 路径仅用其 weight loading 入口）
│   ├── decoder.rs             # Burn 版 TextDecoder（#[cfg(not(feature = "cuda"))] 才编译）
│   ├── cudarc_engine.rs       # ★ GPU-resident 文本解码器 + GPU element-wise/cuBLAS 封装 + 共享 CudaState/CudaKernels
│   ├── gpu_audio_encoder.rs   # ★ GPU-resident 音频编码器（cuBLAS + 自定义 conv2d/im2col + 自定义 element-wise）
│   ├── inference.rs           # AsrInference 装配；条件编译两套 encode_audio + generate
│   ├── streaming.rs           # 流式（forward_incremental，仍用 burn 路径）
│   └── kernels\
│       └── kernels.cu         # ★ 全部 CUDA C kernel 源码（NVRTC 编译，运行时缓存）
└── tests\
    └── transcribe_burn.rs     # RTFx benchmark：0.6B/1.7B × sample1/15s/30s/90s
```

### 关键 Backend 别名（在 `lib.rs`）

```rust
#[cfg(feature = "cuda")]
pub type Backend = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime, half::f16, i32, u8>;
#[cfg(feature = "cuda")]
pub type Device = burn::backend::cuda::CudaDevice;
```

### `Cargo.toml` 关键行

```toml
[dependencies]
burn = { version = "0.21", default-features = false, features = ["std"] }
burn-cubecl = { version = "0.21", optional = true }
cubecl = { version = "0.10", optional = true }
half = { version = "2", optional = true }
cudarc = { version = "0.19", features = ["cublas", "f16"], optional = true }

[features]
default = ["cuda"]
cuda  = ["burn/cuda", "dep:burn-cubecl", "dep:cubecl", "dep:half", "dep:cudarc", "cubecl/cuda"]
```

---

## 2. 架构：全 GPU CUDA 引擎

### 2.1 共享 CudaState

`CudaState`（在 `cudarc_engine.rs`）封装：
- `Arc<CudaStream>` — 默认 stream（同一个 stream 上 cuBLAS 和 kernel 自然顺序执行，无需额外同步）
- `CudaBlas` — cuBLAS handle
- `CudaKernels` — NVRTC 编译并缓存的所有 kernel function handle

通过 `Arc<CudaState>` 在文本解码器和音频编码器之间共享，避免多次 NVRTC 编译。

### 2.2 CUDA Kernels（`src/kernels/kernels.cu`，NVRTC 运行时编译）

每个 kernel 都用 f16 存储 + f32 累加。关键 kernel：

| Kernel | 用途 |
|--------|------|
| `rms_norm_f16` | RMSNorm（一个 block 一行，shared-mem 归约） |
| `add_residual_rms_norm_f16` | 残差 + RMSNorm 融合（一次启动） |
| `silu_mul_split_f16` | SwiGLU（gate-up fused 输出 → SiLU(gate)*up） |
| `softmax_scaled_causal_f16` | scale + 可选 causal mask + softmax（一次启动） |
| `rotary_emb_f16` | 旋转位置编码（接受全表 + pos_offset） |
| `rms_norm_rotary_f16` | RMSNorm + 旋转编码融合（Q/K 路径） |
| `kv_cache_write_f16` / `kv_cache_write_pair_f16` | KV cache 写入；pair 版本一次启动写 K 和 V |
| `repeat_kv_from_cache_f16` | GQA: KV cache → 重复扩展（仅 prefill 用） |
| `fused_gqa_decode_f16` | ★ **decode 路径专用**：Q·K^T + softmax + attn·V 全融合 + 直接读 KV cache（绕过 repeat_kv，省 4 次 launch + 2 次大 alloc） |
| `embed_lookup_f16` | embedding 查表 |
| `argmax_f16` | logits argmax（单 block） |
| `swap_dims_12_f16` | 4D 张量 dim 1/2 交换（attention 的 [b,s,h,d]↔[b,h,s,d]） |
| `qkv_split_f16` | 融合 QKV 投影 → 拆 + reshape + swap 一次完成 |
| `slice_dim2_f16` / `concat_dim2_write_f16` | 窗口注意力切片/写回 |
| `im2col_3x3_s2p1_f16` | conv2d im2col（仅 3×3 stride=2 pad=1） |
| `conv_postprocess_f16` | conv 后处理：[b*h*w, c_out] → [b, c_out, h, w] + bias + GELU |
| `layer_norm_f16` | LayerNorm（音频编码器用） |
| `gelu_inplace_f16` / `gelu_f16` | GELU（精确实现，用 erff） |
| `add_f16` / `add_inplace_f16` | element-wise 加 |
| `add_bias_inplace_f16` | Linear 后的 bias 广播加 |

### 2.3 文本解码器（GpuTextDecoder）

- **embed_table**: GPU 上 [vocab, hidden]，**同时复用为 lm_head**（权重共享）
- **每层** `GpuDecoderLayer` 把所有权重放 GPU（fused QKV + fused gate-up）
- **KV cache** `GpuKvCache`：每层预分配 `[1, nkvh, max_seq, hd]`，slice-write 写入，无重新分配
- **每步 forward**（s=1, decode）：约 13 次 kernel/cuBLAS launch
  1. `rms_norm` (input norm)
  2. `linear_gpu` (fused QKV)
  3. `qkv_split` × 3
  4. `rms_norm_rotary` Q
  5. `rms_norm_rotary` K
  6. `kv_cache_write_pair` (K + V 一次)
  7. **`fused_gqa_decode`**（一次完成 Q·K + softmax + ·V）
  8. `swap_dims_12` + reshape
  9. `linear_gpu` (O proj)
  10. `add` (residual)
  11. `rms_norm` (post-attn)
  12. `linear_gpu` (gate-up)
  13. `silu_mul_split`
  14. `linear_gpu` (down)
  15. `add_inplace` (residual)

  + 最终 `rms_norm` + `slice_last_token` + `linear_gpu` (lm_head) + `argmax` + 1 次 dtoh sync。
  约 **15 × 28 + 5 = 425 launches/token** in decode loop.

### 2.4 音频编码器（GpuAudioEncoder）

- **conv stem**: 3 层 3×3 stride=2 pad=1 conv2d，全 GPU（im2col + cuBLAS GEMM + fused bias+GELU）
- **conv_out** Linear + sinusoidal PE（小幅 CPU detour 用于跨 chunk 拼接，可优化）
- **18 层** transformer（LayerNorm + windowed self-attention + LayerNorm + FFN），与文本解码器共享 CudaState
- **窗口注意力**：每窗口走一次 `attention_qk` + `softmax` + `attention_av`，via `slice_dim2` 和 `concat_dim2_write` 在 GPU 上切片/拼接

### 2.5 编译时配置

NVRTC 在 `CudaState::new` 时编译 `kernels.cu`，需要 `cuda_fp16.h`。代码从环境变量 `CUDA_PATH` 取 include 路径：

```rust
let cuda_include = std::env::var("CUDA_PATH").map(|p| format!("{}/include", p))
    .unwrap_or_else(|_| "/usr/local/cuda/include".to_string());
```

Windows 默认安装位置：`C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8`，`CUDA_PATH` 通常已设置。

---

## 3. 已解决的关键问题

### 3.1 burn-cubecl conv2d/slice 在小 shape 下极慢

`run_conv_stem` 末尾的 31 次 `slice(...)` 调用每次触发 burn-cubecl autotune+kernel launch，**单独耗时 6-8 秒**。改成下载一次 → CPU 切片 → 重新上传修复。最终的 GPU conv stem 用自定义 im2col + cuBLAS GEMM，**总耗时 ~150ms**。

### 3.2 cuBLAS GEMM 列主序映射

行主序 `y = x @ W^T`（x [m,K], W [N,K]）→ cuBLAS `y^T = W @ x^T`：transa=T(W), transb=N(x^T), m=N, n=M, k=K, lda=K, ldb=K, ldc=N。

### 3.3 NVRTC kernel 参数生命周期

`bb.arg(&(x as i32))` 会引用临时值，rustc 报 E0716。必须先 `let x_i = x as i32;` 再 `bb.arg(&x_i);`。

### 3.4 cudarc 0.19 API 差异

- `clone_htod/clone_dtoh` 替代旧的 `htod_copy/dtoh_copy`
- `CudaContext::new(0)?` → `ctx.default_stream() -> Arc<CudaStream>`
- `stream.launch_builder(&function).arg(...).launch(cfg)?`
- 必须 `use cudarc::driver::PushKernelArg;` 才能 `.arg()`
- `compile_ptx_with_opts` 接受 `CompileOptions { arch, include_paths, .. Default::default() }`

### 3.5 GQA + repeat_kv 是 decode 的瓶颈

每步两次 `repeat_kv_from_cache` 分配 `b*nqh*cur_len*hd` f16 缓冲 + 一次 kernel launch，外加 `attention_qk` + `softmax` + `attention_av` 三次 launch。

→ 写 `fused_gqa_decode_f16`：grid = (b*nqh)，一个 block 负责一个 query head，shared mem 存 scores[cur_len]，直接从 cache 索引 KV（GQA 通过 `kh = qh / n_rep` 映射）。**4 次 launch + 2 次 alloc → 1 次 launch + 1 次 alloc**，35-40% decode 加速。

### 3.6 多线程 cargo test 串行化 GPU

`cargo test` 默认多线程跑 ignored test，多个测试同时持有 model + GPU → 性能崩塌。运行用 `--test-threads=1`：

```bash
cargo test --release --features cuda --test transcribe_burn -- --ignored --nocapture --test-threads=1 test_q06
```

---

## 4. 测试与基准

### 4.1 命令速查

```bash
cd D:\qwen3-asr-burn

# 单测（推荐方式，避免多线程 GPU 抢占）
cargo test --release --features cuda --test transcribe_burn test_q06_15s -- --ignored --nocapture
cargo test --release --features cuda --test transcribe_burn test_q06_30s -- --ignored --nocapture
cargo test --release --features cuda --test transcribe_burn test_q06_90s -- --ignored --nocapture

# 全跑（必须 --test-threads=1）
cargo test --release --features cuda --test transcribe_burn -- --ignored --nocapture --test-threads=1 test_q06

# 清空 autotune cache（修改 kernel 后建议）
Remove-Item D:\qwen3-asr-burn\target\autotune -Recurse -Force
```

### 4.2 验证输出

**0.6B-15s 英文**（candle 参考）：
> All right, viewers. Welcome everyone into this crash course about our new 50-minute time frame entry models and strategy. If you are here, it means you have already completed the first swing trading strategy. So congratulations on reaching this.

**0.6B-30s 中文**：
> 我现在在天海酒吧，限你十分钟内到我面前。十分钟，二十公里呢。自己想办法。怎么说啊，舒心？...

---

## 5. 接手者：下一步优化方向

| 优化 | 预期收益 | 工作量 | 备注 |
|------|---------|--------|------|
| **CUDA Graph 捕获 decode 步骤** | decode → 接近 0 launch overhead，30s/90s 上 6x → 9-10x | 中 | 每个 cur_len 需要独立 graph；考虑用 graph update 复用 |
| **PinnedHostSlice 异步 argmax 下载** | 每步省 ~50µs sync = 6ms / 125 步 | 小 | cudarc 支持，需要重写 argmax 路径 |
| **lm_head GEMV + argmax 融合** | 省 1 大 launch + 1 sync + 减少带宽 | 中 | 写一个自定义 kernel：M×N matmul 同时维护 max |
| **conv stem permute 走 GPU**（当前 CPU detour） | 节省 ~5-10ms / 推理（一次性） | 小 | 写一个 permute 4D kernel |
| **1.7B 模型验证** | — | 小 | 转录正确性 |
| **streaming.rs 适配新引擎** | streaming 用例可达同等性能 | 中 | 当前 streaming 仍走 burn audio encoder |
| **ROCm/Metal/Vulkan 后端** | 跨平台目标 | 大 | cudarc_engine 是 CUDA 专属；其它后端需要类似的 rocBLAS/MPS 封装 |

---

## 6. 关键 commit / 优化历程

| 阶段 | RTFx (15s) | RTFx (30s) | RTFx (180s_en) | 关键改动 |
|------|------------|------------|------------|---------|
| baseline (candle path 起点) | 0.44x | 0.25x | — | burn-cubecl + safe_attention |
| cuBLAS 文本解码器 (CPU↔GPU) | 0.48x | 0.25x | — | linear/QK/AV 走 cuBLAS |
| GPU-first GpuTensor | 0.55x | 0.25x | — | matmul 链不离 GPU |
| GPU element-wise kernels + KV cache | 1.4x | 1.26x | — | NVRTC kernels, KV 常驻 GPU |
| GPU 音频编码器（cuBLAS transformer）| 2.44x | 2.24x | — | encoder 18 层走 cuBLAS+kernels |
| GPU conv2d 替换 burn-cubecl conv | 6.39x | 4.87x | — | im2col + cuBLAS 取代 burn conv stem |
| fused_gqa_decode kernel | 7.57x | 6.10x | — | decode 路径 Q·K + softmax + ·V 融合 + 绕过 repeat_kv |
| qkv_extract_*_norm_rotary fused kernels | 9.79x | 7.19x | — | QKV split + RMSNorm + rotary + cache write 6 launches → 2 |
| linear_gpu_accum (cuBLAS beta=1 残差融合) | 10.62x | 7.72x | 7.23x | O-proj 和 down-proj 用 cuBLAS 直接累加到残差 |
| skip swap_dims_12 on s=1 (no-op) | 11.39x | 8.40x | 7.32x | s=1 时 swap [b,h,1,d]→[b,1,h,d] 是 no-op |
| **fused_gqa_decode_split (flash-attn 2-kernel)** | **12.55x** | **10.19x** | **10.69x** | 长 context decode 跨 block 并行 + online-softmax 合并 |
| **alloc_uninit_f16 (skip memset_d8_async)** | **23.03x** | **19.93x** | **16.76x** | 用 `stream.alloc::<f16>` 替代 `alloc_zeros`，每个 cuBLAS/kernel output 省一次 memset launch；Pascal+Windows 下 driver enqueue 限速 → 80% 节省 |
| **skip clone_tensor at O-proj** | **23.50x** | **20.32x** | **16.92x** | layer.forward 改成按值消费 x，直接复用 x 当 residual buffer，省 28 × N_step 个 memcpy_dtod launch |

---

## 7. 跨平台编译（未来目标）

`cudarc_engine.rs` 和 `gpu_audio_encoder.rs` 仅在 `#[cfg(feature = "cuda")]` 下编译。其它后端走 `decoder.rs` + `encoder.rs`（burn 原生路径）。

要在 ROCm/Metal 上达到类似性能，需要为每个后端实现一个并行的"cudarc_engine"：
- **ROCm**：rocBLAS + HIP kernel via cubecl/hip 或自定义 hipcc 预编译
- **Metal**：Metal Performance Shaders + Metal compute kernels
- **Vulkan**：可能可以走 wgpu compute（已有 burn::backend::wgpu）

---

## 8. 给接手 AI 的具体第一步建议

```
1. 读 §0 性能表 + §2 架构 — 理解当前结构
2. 检查 1.7B 模型转录是否正确（test_q17_15s / test_q17_30s）
3. 如果还要继续提速：实现 CUDA Graph 捕获（§5 第 1 项）
4. 适配 streaming.rs 用新引擎
5. 跨平台后端（ROCm/Metal 等）按 §7 计划展开
```
