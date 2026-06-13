# ROADMAP — qwen3-asr

> 写给下一个接手会话的 AI。读这一篇就能在 5 分钟内进入状态。

## 0. 项目是什么

Qwen3-ASR（0.6B / 1.7B）的 Rust 推理库。**双后端**：CUDA（cuBLAS + NVRTC 手写 kernel）和 CPU（gemm + rayon），均端到端可用。CPU 解码器 body 走 INT8 weight-only 量化（per-channel + AVX2 GEMV）；其余权重 f16 存储 + f32 计算。

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
├── cpu_engine.rs         # ★ 手写 CPU 文本解码器（gemm + rayon + INT8 权重，AVX2 GEMV）
├── cudarc_engine.rs      # ★ 手写 GPU 文本解码器（cuBLAS + NVRTC kernel）+ DecodeScratch 复用
├── gpu_audio_encoder.rs  # ★ 手写 GPU 音频编码器（cuBLAS + 自定义 conv2d/im2col）
└── kernels/kernels.cu    # 所有 CUDA kernel（运行时 NVRTC 编译）
tests/
├── transcribe.rs          # #[ignore] CUDA 集成测试（0.6B/1.7B × 各种时长 + streaming demo）
├── cpu_transcribe.rs      # CPU 集成测试（0.6B + 1.7B × 全 fixture + streaming；RTFx + 峰值 RSS）
└── fixtures/              # 7 个 wav: sample1 / 15s / 30s / 90s / 89s_ja / 180s / 180s_en
```

### 1.2 公共 API

`lib.rs` 导出 `Backend`、`AsrError` / `Result`、`AsrInference` / `TranscribeOptions` / `TranscribeResult`、`load_audio_wav`。**没有** `StreamingOptions` / `StreamingState` / `best_device()`。

### 1.3 Features

```toml
default = ["cuda"]      # 端到端可用
cuda = ["dep:cudarc"]
cpu  = ["dep:gemm", "dep:rayon"]  # 端到端可用（解码器 body INT8 + 其余 f16，f32 计算）
hub  = ["dep:reqwest"]
```

### 1.4 性能（基准线，P104-100）

#### CPU（Intel Core Ultra 7 265K，Arrow Lake，20c，AVX2；31.5GB RAM；INT8 weight-only 量化，默认）

解码器 4 个 linear（qkv/o/gate_up/down）走 INT8（per-channel 对称量化 + AVX2 GEMV，运行时 `is_x86_feature_detected!` 分发 + 标量 fallback）；lm_head/embed_table/音频编码器仍 f16。解码器 body 永久 INT8，无开关、无 f16 回退分支。RTFx + 峰值 RSS 来自每 fixture 独立进程（Win32 `GetProcessMemoryInfo` PeakWorkingSet，在 `cpu_transcribe.rs::run_cpu_with` 里抓）。

**0.6B**

| 音频 | RTFx | 耗时 | 峰值 RSS |
|---|---|---|---|
| 15s 英文 | 4.91× | 3.0s | 3584 MB |
| 30s 中文 | 5.55× | 5.4s | ~3584 MB |
| 90s 英文 | 5.75× | 15.6s | 3584 MB |
| 89s 日文 | 6.21× | 14.3s | ~3584 MB |
| 180s 中文 | 5.34× | 33.7s | 5134 MB |
| 180s 英文 | 5.50× | 32.7s | 5134 MB |

**1.7B**（首次 CPU 基线）

| 音频 | RTFx | 耗时 | 峰值 RSS |
|---|---|---|---|
| 15s 英文 | 2.23× | 6.7s | 8056 MB |
| 30s 中文 | 2.58× | 11.6s | 8056 MB |
| 90s 英文 | 3.21× | 28.1s | 8056 MB |
| 89s 日文 | 3.59× | 24.8s | 8056 MB |
| 180s 中文 | 2.92× | 61.7s | 8512 MB |
| 180s 英文 | 3.01× | 59.8s | 8512 MB |

观察：1.7B RTFx 在 89s_ja 见顶（3.59×）后 180s 回落（2.92×）——长音频 prefill 的 O(s²) attention 曾拖低 RTFx，**已于 2026-06-13 修复**（见下）。峰值 RSS：15-90s 权重主导（0.6B ~3.5GB、1.7B ~8GB，与音频长度无关），180s 因 prefill attention scores scratch 抬高（0.6B +1.5GB → 5134MB；1.7B +0.5GB，其 seq_len 较短故增幅小）。

**2026-06-13 优化：prefill attention SSE2-标量 → gemm crate（AVX2-FMA）。** 原 `prefill_attention` 两个矩阵乘（scores=q@Kᵀ、out=scores@V）是手写标量循环，release 默认 baseline 只编到 SSE2（4 float/周期），而同文件 GEMM 走 gemm crate 的 AVX2-FMA（8/周期）—— attention 占 prefill ~60%，故成 prefill 瓶颈。改用 gemm crate 调用（stride 参数直接读写 q/out 交错布局，免 gather/scatter；外层 rayon 跨 (b,qh)，内层 `Parallelism::None` 避免嵌套并行超订）。**实测（进程内直接计时，不受热噪声影响）：attn 182→91ms/层（2×），prefill 总 8845→5791ms @0.6B-180s（−34%）。** 180s isolated RTFx 5.34→5.79×（+8%，且本次实测在热降频下仍成立 → 真实收益 ≥8%）。短音频按 prefill 占比递减（15s 的 seq_len ~375，gemm 仍适用，无大小分发必要）。零精度风险（数学等价，仅 SSE2→AVX2 求和重排）。注：本 VM（CPUID 撒谎 + 持续负载热降频）cross-run total RTFx 噪声大，**prefill 的进程内计时是可靠信号**。单测 `prefill_attention_matches_reference` 锁正确性。

精度：0.6B 全 7 fixture 归一化 CER ~1%（英文/中文短音频全过；180s 中文 ~3.6%、89s 日文 ~1.9% raw，但绝大多数是标点/合法改写，真实存疑字 <1%，集中在同音决策边界）。

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

**2026-06-13 CUDA 阶段分解（0.6B，进程内 info! 计时，D2H 自动 sync）**：decode 占 75-80%（8.78ms/tok@15s → 11.70ms/tok@90s），prefill 10-11%，audio_enc 7-10%，mel+setup ~4%。decode vs 带宽下限（decoder+lm_head ~750MB f16 / P104 ~200GB/s ≈ 3.75ms/tok）有 ~2× 余量，疑为 launch 开销（§2.5 #9 Pascal enqueue 限速）+ 非-TC kernel。

**跨架构可移植（强卡更强，已具备）**：① cuBLAS GEMM 设了 `CUBLAS_TENSOR_OP_MATH`（cudarc_engine.rs:133）→ 在 tensor-core 卡（Volta+，如 RTX 3080 sm_86）自动用 TC，Pascal 走非-TC。② NVRTC kernel 运行时按当前卡 `sm_{major}{minor}` 编译（cudarc_engine.rs:141）→ 不写死 sm_61。所以**无需为 3080 改代码**，直接打包即可，RTFx 会显著高于 P104。唯一 P104 经验旋钮：attention chunk 256/512（cudarc_engine.rs:1381），3080 上 SM 更多可能要重调。

### 1.5 内存

CPU 峰值 RSS 随音频长度分两段（INT8 默认；每 fixture 独立进程实测，Win32 PeakWorkingSet）：
- **15-90s**：权重主导、基本恒定。0.6B ~3,584 MB、1.7B ~8,056 MB（与 15s 持平）。INT8 只减解码器 body 权重存储（f16→i8，0.6B 省 ~205MB）；prefill 把权重反量化成 f32 跑 gemm 的临时 f32 权重（每层最大 ~25MB，用完即释）不主导峰值。
- **180s**：prefill attention 的 O(s²) scores scratch（20 线程并发 × s×cur_len×4B ≈ 1.5GB）把峰值抬高。0.6B → 5,134 MB（+1.5GB），1.7B → 8,512 MB（+0.5GB，1.7B seq_len 较短故增幅小）。

f16 权重相对纯 f32 仍省 ~25%（0.6B 3.5GB vs 4.66GB）。

## 2. 关键架构事实

### 2.1 权重存储：f16 + f32 计算

两个后端统一模式（CPU 解码器 body 例外，已 INT8 量化）：
- **存储**：音频编码器 / lm_head / embed_table 用 `CpuWeightF16 { data: Vec<half::f16>, rows, cols }`；解码器 body 4 个 linear 用 `CpuWeightI8 { data: Vec<i8>, scale: Vec<f32>, rows, cols }`（load 时从 f16 量化，cols 补齐到 32 倍数）
- **CPU 解码** (m=1)：`linear_gemv_i8` 走 AVX2 INT8 GEMV（per-channel 权重 scale + 单尺度激活 scale，widen→madd），比 f16 GEMV 快 1.2-1.6×（标量 fallback 在无 AVX2 时）
- **CPU prefill** (m>1)：`dequant_to_f32` 把 INT8 权重反量化成 f32，走 gemm crate 的 f32 GEMM（prefill 不量化激活）
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

- decode (m=1): `linear_gemv_i8` AVX2 INT8 GEMV（per-channel 权重 scale + 单尺度激活 scale，无 AVX2 走标量 fallback）
- prefill (m>1): `dequant_to_f32` 反量化 INT8→f32，然后 gemm crate f32 GEMM
- `linear_accum_i8`: 带 residual 累加的 INT8 线性层（cuBLAS beta=1 思路的 CPU 版）

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

### 3.2 [已修] Conv stem permute — 已是 GPU kernel

`gpu_audio_encoder.rs::GpuConvStem::forward` 的 `[b,c,f,t]→[b,t,c,f]` permute 现在走 GPU kernel `permute_bcft_to_btcf_f16`（kernels.cu:1275），无 CPU 绕路。曾是 CPU detour（~5-10ms），已消除。

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
- [x] CPU 文本解码器 INT8 weight-only 量化（per-channel 对称 + AVX2 GEMV，默认开启；RTFx 1.2-1.6×，归一化 CER ~1%；内存持平）
- [x] CPU prefill attention：手写 SSE2-标量 → gemm crate（AVX2-FMA）。attn 182→91ms/层（2×），prefill −34% @180s，180s RTFx +8%。零精度风险。单测 `prefill_attention_matches_reference`（见 §1.4）
- [x] GPU conv stem permute：已是 GPU kernel `permute_bcft_to_btcf_f16`（kernels.cu，无 CPU detour）。此前文件头/ROADMAP 注释过时，已更正（§3.2）。
- [x] GPU 全部 13 个集成测试通过
- [x] CUDA Graph 死代码清理
- [x] 脱离 burn 框架

## 5. 下一步规划

### 5.1 P1：CUDA 小优化（明确收益）

- [x] **Conv stem GPU permute kernel**（§3.2）— 已是 GPU kernel `permute_bcft_to_btcf_f16`，CPU detour 已消除
- [ ] **lm_head 融合重测**（§3.6）— **2026-06-13 分析后判定不值得**：现有融合 kernel 是单 block（grid=1，只用 1 SM，这才是"输给 cuBLAS"的真因）；且 lm_head GEMV 读 311MB/token 是带宽受限，多 block 融合顶多追平 cuBLAS + 省个 argmax pass（~10µs/token ≈ 0.1%）。cuBLAS f16 GEMV 已在带宽下限。**结论：~0.1%，跳过。**
- [ ] dead code warning 清理（`#[allow(dead_code)]` 或删）

### 5.2 P2：CPU 优化

- [x] **INT8 weight-only 量化 + AVX2 GEMV** — 已完成并收口（解码器 body 永久 INT8，无 f16 回退分支）。decode 路径 RTFx 1.2-1.6×（见 §1.4）。
- [ ] **prefill 也走 INT8 GEMM**（当前 prefill 的 4 个 linear 仍反量化到 f32 跑 gemm；是峰值 RSS 不降的原因，也是 prefill 剩余大头。注：prefill attention 已改 gemm crate，见 §4）
- [ ] **im2col SIMD** — 卷积 im2col 生成目前是标量赋值
- [x] **prefill attention** — 已从 SSE2-标量改 gemm crate（AVX2），2×（见 §4）。decode attention（`fused_gqa_decode`）仍是标量，但 KV 流读已近 DRAM 峰值（~72/80 GB/s），bandwidth-bound，SIMD 收益小；若要进一步压 decode，考虑 KV cache f16（半带宽，有精度权衡）
- [ ] 更激进的 rayon 策略（跨层并行？prefill 阶段流水线？）

### 5.3 P3：功能

- [ ] streaming API 重建（§3.4）
- [ ] fixture 自动下载脚本（§3.5）

### 5.4 P4：CUDA 极限

- [ ] PinnedHostSlice 异步 argmax（`cudaStreamAddCallback`，省 ~50µs/step）
- [ ] KV cache 预 fetch（prefill 后第一个 decode 步不等 cuBLAS）
- [ ] **cuBLASLt 替换** — **2026-06-13 评估后降级**：① Pascal 无 tensor core，cuBLAS 和 cuBLASLt 跑同类 f16 kernel，~10% 不保证；② cuBLAS 已设 `CUBLAS_TENSOR_OP_MATH`，在 Ampere+ 自动用 TC，**cuBLASLt 非启用 TC 的必要条件**；③ cudarc safe API 每次 matmul 重建 desc+跑 heuristic（decode 33k+ GEMM 会反超），需手写 plan 缓存（高工作量）；④ decode 疑 launch-bound（非 kernel-bound），cuBLASLt 动不到主因。**仅当上 Ampere 实测 cuBLAS 不够、且能本地验证时再考虑。**

### 5.5 不做

- **CUDA Graph** — sm_61 不支持
- **ROCm / Metal / Vulkan** — 已明确放弃，cudarc 专属
- **回到 burn 框架** — 这次重构的目的就是脱离它

## 6. 一些不能忘的事实

- **NVRTC 编译需要 CUDA_PATH**（Windows: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`）
- **cudarc 0.19**（`cuda-12080` feature）
- **`__launch_bounds__` 在所有 kernel 上**（Pascal sm_61 register pressure 优化）
- **f16 + f32 累加**是音频编码器 / lm_head / embed_table 的统一模式，不要降级；解码器 body 已是 INT8（AVX2 GEMV），见 §2.1 / §2.3
- **Pascal driver 有 enqueue 限速**（alloc_zeros → memset 触发），alloc_uninit 收益巨大
- **测试多线程跑会抢 GPU** — 必须 `--test-threads=1`
- **`h_buf` 跨步复用**：decode 步 28 层改 `h` in-place，final_norm 写到 `scratch.final_norm`，不修改 `h`
- **CPU 音频编码器**：attention 用标量循环 + rayon 跨 head 并行，不要换 GEMM per-head（小窗口 overhead 更大）
- **CPU f16→f32 转换**：音频编码器顺序转（100+ 次小权重，rayon 开销大于收益）；文本解码器 prefill 用 rayon 转（大权重 ~1.2GB，rayon 明显更快）
- **`half` crate v2** 是无条件依赖（不是 optional），GPU 和 CPU 路径都用
- **CPU 解码器 body 永久 INT8**（per-channel 对称 + AVX2 GEMV，`is_x86_feature_detected!` 分发）。4 个 decoder linear 在 `CpuDecoderLayer::load` 时从 f16 量化成 `CpuWeightI8`；lm_head/embed_table/音频编码器仍 f16（`linear_gemv_f16`）。无开关、无 f16 回退分支。INT8 是**提速**优化，**不减峰值内存**（prefill 仍反量化 f32）。
- **AVX-VNNI 在本开发 VM 是 CPUID 假阳性**（`is_x86_feature_detected!("avxvnni")` 返回 true，但执行 `_mm256_dpbusd_epi32` 直接 `STATUS_ILLEGAL_INSTRUCTION` —— hypervisor 透传了 VNNI 标志位但不支持执行）。所以 CPU INT8 GEMV 只用 AVX2 `madd` 路径，**不要尝试 VNNI**（曾试过，崩；代码已还原）。真实 Arrow Lake 硬件上 VNNI 理论可用但本环境无法验证。decode GEMV 的进一步提速需等真实 VNNI 硬件或换思路（如 KV cache f16 / 投机解码）。
- **本 VM 持续负载热降频严重**：cross-run 的 total RTFx 噪声大（连续跑会越跑越慢）。要测 total RTFx 须凉机单跑；**进程内直接计时（如 generate_cpu 的 `info!("Prefill/Decode ms")`、cpu_engine 的采样 pmark）不受热噪声影响，是可靠信号**。
- **CER 对照工具**：`examples/cer_compare.rs`（解析两次 transcribe stdout 算 raw + 归一化 CER）。峰值 RSS 已折进 `cpu_transcribe.rs::run_cpu_with`（in-process Win32 `GetProcessMemoryInfo`，每 fixture 独立进程）；`examples/mem_probe.ps1` 是旧的 100ms 轮询版（其 `QASR_CPU_INT8` 开关是死代码——INT8 永久，无开关），已被取代。

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
