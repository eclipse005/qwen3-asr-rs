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

权重加载（非 feature，常驻依赖）：`memmap2` + `bytes`（>=1.11，`Bytes::from_owner` + 零拷贝 slice；见 §2.1）。

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
| 15s 英文 | 25.4× |
| 30s 中文 | 21.3× |
| 90s 英文 | 16.6× |
| 89s 日文 | 21.1× |
| 180s 中文 | 7.9× |
| 180s 英文 | 8.2× |

> 注：180s 等长音频 RTFx 跨 run 波动大（VM 热降频，§6），单次运行不可靠。15-90s 短音频数字较稳定。2026-06-14 更新：decode attention 从"总是 split（2 kernel）"改为"短 context 用单 kernel"（cur_len ≤ 1024），15s RTFx 24.23→25.43（+5%），属 launch 开销减少的真实收益。长音频数字差异以热噪声为主。
>
> **2026-06-15 修正：180s RTFx 之前记录的 27.50×/23.37× 是错误数字**——那是 `728d372` 引入的 `fused_gqa_decode_split_p1_f16` 上 `__launch_bounds__(256,4)` bug 的"假快"：长 context（cur_len > 1024，走 split path）算出的 attention score 被破坏 → 解码器陷入重复循环、生成 token 数从 640 崩到 ~465 就停 → decode 阶段变短 → RTFx 反而虚高，但转录文本是垃圾。`2ef95b6` 删掉该 `__launch_bounds__` 后，180s 文本恢复正常（640/582 token，和 candle 版 / 历史 `517c3e9` 基准逐字一致），真实 RTFx 是 ~8×。**短音频（15s/30s/90s）不受影响**——它们走单 kernel path（cur_len ≤ 1024），数字一直正确。

#### CUDA 1.7B（P104-100）

| 音频 | RTFx |
|---|---|
| 15s 英文 | 9.12× |
| 30s 中文 | 7.96× |
| 90s 英文 | 8.56×（旧；连续跑热噪声大）|
| 89s 日文 | 9.76×（单独凉机跑）|
| 180s 中文 | ~8×（噪声大，单次不可靠）|
| 180s 英文 | ~8.5×（噪声大）|

> 注：1.7B 计算量大，连续跑多 fixture 极易触发热降频（§6），cross-run RTFx 噪声远大于 0.6B。15s/30s 短音频 + 89s_ja 单独凉机跑的数字可靠。90s/180s 连续跑出的数字（如 2.98×）是热噪声假象，不代表真实性能。

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

**加载路径（`weights.rs`，2026-06-15 重写）**：safetensors 不再 `std::fs::read` 整文件入内存 + 每 tensor `to_vec()` 拷贝；改为单次 `Mmap::map` + `Bytes::from_owner` 持有 mmap，每个 tensor 用 `buf.slice(off..off+len)` O(1) refcount 切片（零拷贝）。`RawTensor.data` 由 `Vec<u8>` 改为 `bytes::Bytes`（`Deref<Target=[u8]>`，转换逻辑不变 → bit-exact）。`load_weights`：1111ms → 0.8ms；peak host mem ~3.6GB → ~1.8GB。`bytes` 锁 >=1.11（`from_owner` 1.9 引入，1.11 修了其 `to_vec` 内存泄漏 #773）。

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
13. **decode attention 短 context 用单 kernel（cur_len ≤ 1024 不 split）**：25.4× ← 2026-06-14。原"总是 split"是为已删的 CUDA Graph 稳定性（§3.1），改回条件分支后省 28 launches/step，15s RTFx +5%。"短用单 / 长用 split"和 prefill 路径（§2.4 forward）逻辑一致。

**正确性回退 + 修复（2026-06-15）**：`728d372`（"vectorized kernels"）给 split path 的 `fused_gqa_decode_split_p1_f16` 加了 `__launch_bounds__(256,4)`，本意提 occupancy，实际破坏长 context（cur_len > 1024）的 attention score → 解码器重复循环。**只有长生成触发**（短音频走单 kernel path 不受影响），且表现为"RTFx 虚高 + 文本崩坏"，极具迷惑性。bisect 定位后 `2ef95b6` 删除该 `__launch_bounds__`，180s 恢复 640/582 正常 token。教训：`__launch_bounds` 的 min-blocks-per-SM 约束会压寄存器，对 shared-mem 密集型 attention kernel 可能改语义，须长生成回归测试覆盖。

## 3. 卡点 / 已知问题

### 3.1 [已删] CUDA Graph — 硬件不支持（sm_61），代码已清理

### 3.2 [已修] Conv stem permute — 已是 GPU kernel

`gpu_audio_encoder.rs::GpuConvStem::forward` 的 `[b,c,f,t]→[b,t,c,f]` permute 现在走 GPU kernel `permute_bcft_to_btcf_f16`（kernels.cu:1275），无 CPU 绕路。曾是 CPU detour（~5-10ms），已消除。

### 3.3 死代码清理状态（2026-06-13 全树复核）

原先标注的"死代码"经核查大多其实在用：
- `gpu_audio_encoder.rs::GpuAudioEncoder::run_transformer` — **在用**（streaming.rs 4 处调用），非死代码。
- `prompt::decode_result` — **在用**（inference.rs / streaming.rs 调用）。原条目标"inference.rs::AsrInferenceInner::decode_result"路径过时，实际在 prompt.rs。
- `raw_tensor.rs::to_f32_vec` / `as_f32` — **在用**（CPU 路径 `load_vec_f32 → as_f32` 读 f32 的 layernorm/bias 权重）。原"CPU 改用 as_f16"说法不准，as_f32 仍用于 f32 权重。
- `inference.rs::tokenizer_decode` — **已不存在**（早被删，原条目过时）。
- **已删（2026-06-13）**：`lm_head_argmax`（cudarc_engine.rs）+ 其 kernel `lm_head_gemv_argmax_f16`（kernels.cu）+ CudaKernels 字段/load —— 0 调用方，融合收益 ~0.1%（见 §5.1 分析），移除。
- **已删（2026-06-13）**：`mem_probe.ps1` 的 `QASR_CPU_INT8`/f16 死分支（INT8 永久无开关），脚本简化为纯 RSS 轮询。

其余 `#[allow(dead_code)]`（~11 处，在 CpuKvCache/CudaKernels/DecodeScratch/GpuConvStem 等结构体上）是结构体个别未用字段（加载/scratch 完整保留），非整项死代码，保留。

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
- [x] **decode attention 短 context 用单 kernel**（cur_len ≤ 1024 不走 split，省 28 launches/step）。原"总是 split"是为已删的 CUDA Graph，2026-06-14 修正。15s RTFx 24.23→25.43（+5%）。cuBLAS GEMV 带宽 benchmark 确认 GEMV 已跑满 208GB/s（峰值），decode 余量在 launch 开销不在 GEMM 效率。
- [x] CUDA Graph 死代码清理
- [x] 脱离 burn 框架
- [x] **修复 split-path attention 长上下文重复 bug**（2026-06-15，`2ef95b6`）：删除 `728d372` 误加的 `fused_gqa_decode_split_p1_f16` 上的 `__launch_bounds__(256,4)`。bisect 定位（`517c3e9` 好 → `728d372` 坏）；180s 文本恢复 640/582 token（崩前 465/282）。详见 §2.5。
- [x] **mmap safetensors 加载**（2026-06-15，`c5a37cc`）：`weights.rs` 改 mmap + `Bytes::from_owner` 零拷贝 slice；`load_weights` 1111ms → 0.8ms，peak host mem ~3.6GB → ~1.8GB。bit-exact（15s/30s/180s 转录逐字一致）。详见 §2.1。

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
- **回到 burn 框架** — 这次重构的目的就是脱离它
- **wgpu / Vulkan 后端** — Intel iGPU Vulkan driver 对 f16 storage buffer 连续读取有 bug（device lost），详见 §5.6
- **DirectML 后端** — GEMV 带宽只有 CUDA 的 1/4（P104 50.9 vs 200 GB/s），比 CPU INT8 还慢（Intel 核显）。技术上可行但性能不值得，详见 §5.6

### 5.6 新后端：DirectML（Intel 核显 / 跨厂商 GPU 通用）— ❌ 性能不值得，已归档

> **2026-06-14 最终结论（benchmark 实证）**：DirectML 技术上可行（f16 GEMM bit-exact 正确），但 **GEMM 性能不值得**。经 dispatch benchmark + 真实 lm_head 规模 GEMV 测量，DirectML 的 GEMM 带宽只有 CUDA 的 1/4（P104），比 CPU INT8 还慢（Intel 核显）。**不做正式后端**。保留 spike + benchmark 代码作为调研存档，避免未来重复探索。
>
> **关键 benchmark 数据（2026-06-14 实测，原 `dml_dispatch_bench.rs` 已随归档删除，数据留存此处）**：
>
> | 指标 | Intel 核显 (DirectML) | P104-100 (DirectML) | P104-100 (CUDA 现有) |
> |---|---|---|---|
> | dispatch overhead (batched) | 6.2µs | 11.2µs | ~5µs |
> | lm_head GEMV (297MB) | **9.14ms** | **5.70ms** | ~3.75ms |
> | GEMV 带宽 | **31.7 GB/s** | **50.9 GB/s** | ~200 GB/s |
> | decode 预估 ms/tok | ~25ms | ~17ms | ~10ms |
>
> 根因：**DirectML 的 GEMM shader 远不如 cuBLAS 优化**。在 Pascal（老卡）上 DML 可能没用上最优 GEMM 实现。这不是 dispatch overhead 能解释的（overhead 只 2.8ms，compute 15ms+）。Intel 核显更糟 —— unified memory 带宽 ~80GB/s 但 DML 只跑到 31.7GB/s。
>
> **什么情况下值得重新评估**：(1) 有 Intel Arc 独显或 NPU（Intel AI Boost），DML 有专用加速；(2) DML GEMM 实现更新（目前 FL 5.0）；(3) 只需 CPU 卸载不在乎速度。

---

**探索过程存档（2026-06-14）**：用户有 Intel 核显设备（128MB ded + 18GB shared unified memory），计划未来支持 A 卡 + Metal。原 §5.5 "ROCm/Metal/Vulkan 已明确放弃" 已作废 —— 有第 3 个真实硬件目标，抽象有了依据。经过 wgpu → DirectML → benchmark 三轮实证，**三条路全部否决**（wgpu=driver bug，DirectML=性能不值得）。

#### 探索过的三条路（实证结论）

| 路 | GEMM bit-exact | 复杂算子 | 结论 |
|---|---|---|---|
| **wgpu / Vulkan** | ✅ Step 2 通过 | ❌ **qkv_extract device lost** | **否决**：Intel Vulkan driver 对 f16 storage buffer 连续读取有 bug，单 work-item 在循环里读多个连续 f16 元素即触发 device lost（30+ 轮二分定位，与 shared memory/reduction/rotary/uniform 布局全部无关，纯连续读取触发）|
| **OpenCL** | 未 spike | — | **否决**：ocl crate 维护停滞；只覆盖 Intel 核显，A 卡仍需 ROCm 另写，未达成"一次写多后端"目标 |
| **DirectML (DX12)** | ✅ **0/8192 mismatch, bit-exact** | ✅ GEMM operator 含同等复杂度 | **选定**：DML Feature Level 5.0 + FLOAT16 tensor 支持，算子级 API 绕过 driver bug |

#### 选定 DirectML 的理由

1. **f16 完整支持，无 driver bug**：DirectML 走 DX12 不走 Vulkan，避开了 Intel Vulkan 的 f16 连续读取 bug。`dml_probe` 确认 DML Feature Level 5.0（最高，远超 f16 需要的 4.0）+ `FLOAT16 tensor` 支持 = YES。
2. **算子级 API 绕过 shader 手写**：你调 `DML_OPERATOR_GEMM`，DirectML 内部生成优化的 DX12 shader（Intel MetaCommand 加速），不碰 f16 读取细节。这和 cuBLAS 的哲学一致 —— GEMM 由厂商优化，你不写 kernel。
3. **跨厂商**：DirectML 支持 Intel / AMD / NVIDIA（通过 DX12）。adapter 选择明确（DXGI vendor id 0x8086=Intel / 0x10de=NVIDIA / 0x1002=AMD），可运行时指定。
4. **对称 cudarc**：`IDMLDevice` 对应 `CudaContext`，`DML_OPERATOR_GEMM` 对应 cuBLAS GEMM，`ID3D12Resource` 对应 `CudaSlice`，`IDMLCommandRecorder::RecordDispatch` 对应 kernel launch。
5. **Rust 生态**：通过官方 `windows` crate（`Win32_AI_MachineLearning_DirectML` feature）访问，无额外依赖（wgpu 已经依赖 windows crate）。

#### 已知约束

- **不支持自定义融合 kernel**：DirectML 是算子级 API，现有的 `fused_gqa_decode`（attention 融合）、`qkv_extract`（norm+rotary+cache write 融合）必须**拆成独立算子调用**。这是架构性差异 —— DirectML 后端是"算子图调度"，不是"手写 kernel"。代价是更多 launch 开销 + 中间 buffer；收益是算子由厂商各自优化（Intel MetaCommand）。
- **Windows only**：DirectML 只在 Windows 上可用（Mac/Linux 不支持）。但 CUDA 后端也是平台专属的，这不是新问题。
- **GEMM 调用需 3 个输入**：`DML_GEMM_OPERATOR_DESC` 要求 bind A/B/C（即使 beta=0，CTensor residual 也必须 bind）。少了 C input 会输出全 0（已踩坑）。
- **uniform buffer size**：DML 无此问题（wgpu/Vulkan 的坑，Intel 要求 ≥256 字节）。
- **BF16 on disk → F16 in memory**：Qwen3-ASR 权重原生 BF16 存储，加载时转 F16（和现有 CPU/CUDA 路径一致，见 §2.1）。

#### 对称映射表（实施对照）

| 现有 CUDA | DirectML 后端 |
|---|---|
| `cudarc` crate | `windows` crate（`Win32_AI_MachineLearning_DirectML` feature）|
| `CudaContext` / `CudaStream` | `ID3D12Device` / `ID3D12CommandQueue` |
| `CudaSlice<f16>` | `ID3D12Resource`（Default/Upload/Readback heap）|
| cuBLAS GEMM | `DML_OPERATOR_GEMM` |
| 手写 CUDA kernel | `DML_OPERATOR_*`（算子库：ELEMENT_WISE / REDUCTION / ACTIVATION 等）|
| `CudaFunction` + `LaunchConfig` | `IDMLCompiledOperator` + `IDMLCommandRecorder::RecordDispatch` |
| `cudarc_engine.rs` | `directml_engine.rs`（新）|
| `gpu_audio_encoder.rs` | `directml_audio_encoder.rs`（新）|
| `kernels/kernels.cu` | **无**（用 DML 算子组合，不手写 shader）|
| `feature = "cuda"` | `feature = "directml"` |
| `Backend::Cuda` | `Backend::DirectML { adapter_index }` |
| `Engine::Cuda {...}` | `Engine::DirectML {...}` |

#### 算子映射（decode 路径）

decode 每步的算子在 DirectML 里的对应（拆融合后的独立算子）：

| decode 算子 | DirectML 实现 |
|---|---|
| GEMV (qkv/o/gate_up/down/lm_head) | `DML_OPERATOR_GEMM`（已验证 bit-exact ✅）|
| RMSNorm | `DML_OPERATOR_REDUCE`（sum of squares）+ `ELEMENT_WISE_SQRT` + `ELEMENT_WISE_MULTIPLY` 组合；或 `DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION`（若支持 RMS 模式）|
| RoPE (rotary embedding) | `ELEMENT_WISE` 组合（cos/sin 表 + rotate half），无单算子 |
| SiLU·up | `DML_OPERATOR_ACTIVATION_SIGMOID` + `ELEMENT_WISE_MULTIPLY`，或 `ACTIVATION_SILU` |
| softmax (attention) | `DML_OPERATOR_ACTIVATION_SOFTMAX` |
| argmax | `DML_OPERATOR_ARGMAX` |
| embed lookup | `DML_OPERATOR_GATHER`（按 token id 取行）|
| KV cache write | `DML_OPERATOR_COPY` 或直接 D3D12 `CopyBufferRegion` |

#### 已验证的 spike 代码（保留在 examples/）

- `examples/dml_probe.rs` — device 创建 + feature level 查询（DML FL 5.0 + f16）
- `examples/dml_gemm_smoke.rs` — f16 GEMM 完整执行链路（0/8192 mismatch, 0.87ms）

#### 实施顺序（正式后端）

1. 加 `feature = "directml"`，把 `windows` crate 从 dev-dep 提为正式 dep（features: `Win32_Graphics_Dxgi` + `Win32_Graphics_Direct3D12` + `Win32_AI_MachineLearning_DirectML`）
2. 写 `directml_engine.rs`：封装 D3D12 device/queue/fence + DML device/command recorder/binding table 的样板（dml_gemm_smoke.rs 的 setup 部分可提取）
3. 移植 decode 算子链：先 GEMM（已验证）→ RMSNorm → SiLU → softmax → argmax → embed → RoPE → KV cache
4. 端到端 decode N 步（CPU prefill 起点，与 CPU 参考 token 对比）→ 测 ms/token
5. go/no-go：对比 CPU INT8（~3-6ms/tok）和 CUDA（~10ms/tok）基线
6. 若 go：接入 `Engine::DirectML` arm + `Backend::DirectML`，移植音频编码器

**关键风险**：拆融合 kernel 后的 launch 开销。DirectML 每个算子是一次 dispatch（RecordDispatch），decode 每步 28 层 × ~9 算子 = ~252 dispatches/step。如果每 dispatch 有固定开销（~10µs），仅 overhead 就 ~2.5ms/tok。需要实测。这是 DirectML 相比手写 CUDA kernel 的最大不确定性。

#### 正式后端实施规划（2026-06-14）

**接入点**（对称现有 CUDA 后端的 cfg 模式）：

```toml
# Cargo.toml
[features]
default = ["cuda"]
cuda = ["dep:cudarc"]
directml = ["dep:windows"]  # 新增
```
```rust
// src/backend.rs
pub enum Backend {
    Auto,
    Cpu,
    #[cfg(feature = "cuda")] Cuda,
    #[cfg(feature = "directml")] DirectML { adapter_vendor_id: Option<u32> }, // None=Auto, Some(0x8086)=Intel
}
pub(crate) enum ResolvedBackend {
    Cpu,
    #[cfg(feature = "cuda")] Cuda(Arc<CudaState>),
    #[cfg(feature = "directml")] DirectML(Arc<DmlState>),  // 新增
}
// src/inference.rs
pub(crate) enum Engine {
    Cpu { decoder, audio_encoder },
    #[cfg(feature = "cuda")] Cuda { cuda, decoder, audio_encoder },
    #[cfg(feature = "directml")] DirectML { dml: Arc<DmlState>, decoder: DmlTextDecoder, audio_encoder: DmlAudioEncoder },
}
```

**`windows` crate 从 dev-dep 提为正式 dep**（只在 `directml` feature 下）。需要的 features（实测）：
```
Win32_Graphics_Dxgi + Win32_Graphics_Dxgi_Common + Win32_Graphics_Direct3D +
Win32_Graphics_Direct3D12 + Win32_System_Threading + Win32_Security +
Win32_AI_MachineLearning_DirectML
```
（`Win32_Security` 是 `CreateEventW` 的 gate，`Win32_System_Threading` 是 fence event 的，`Win32_Graphics_Direct3D` 是 `D3D_FEATURE_LEVEL` 的。）

**新增文件**：
- `src/directml_engine.rs` — DML 设备封装（D3D12 device/queue/fence + DML device/command recorder/descriptor heap/binding table 的样板，可从 `dml_gemm_smoke.rs` 提取）+ DML 算子调度器（封装 create+compile+bind+dispatch 序列）
- `src/directml_audio_encoder.rs` — 音频编码器（后期，conv stem + transformer 用 DML 算子组合）

**decode 算子实现的优先级 + 难度**（已按依赖排序）：

| 算子 | DML 实现 | 难度 | 备注 |
|---|---|---|---|
| GEMM | `DML_OPERATOR_GEMM` | ✅ 已验证 | decode 用 GEMV-shaped（m=1），prefill 用真 GEMM |
| RMSNorm | `DML_OPERATOR_REDUCE`(L2) + `ELEMENT_WISE_*` 组合 | 中 | DML 无原生 RMSNorm，需 3-4 算子组合；或试 `MEAN_VARIANCE_NORMALIZATION` |
| SiLU·up | `DML_OPERATOR_ACTIVATION_SIGMOID` + `ELEMENT_WISE_MULTIPLY` | 低 | 或 `ACTIVATION_SILU` 直接有 |
| softmax | `DML_OPERATOR_ACTIVATION_SOFTMAX` | 低 | 原生支持 |
| argmax | `DML_OPERATOR_ARGMAX` | 低 | 原生支持 |
| embed lookup | `DML_OPERATOR_GATHER` | 低 | 按 token id 取行 |
| RoPE | `ELEMENT_WISE` 组合（cos/sin 表 + rotate half） | 中-高 | 无单算子，需 ~5 个 element-wise 组合 |
| KV cache write | `DML_OPERATOR_COPY` 或 D3D12 `CopyBufferRegion` | 低 | 不用 DML，直接 D3D12 拷贝 |

**最大风险点（必须早期验证）**：
- **launch 开销**：252 dispatches/step 是否可接受？建议**先只移植 GEMM + 一个 element-wise**，跑 decode 几步测 dispatch 开销，再决定是否继续
- **RMSNorm 组合**：DML 无原生 RMSNorm，组合多个算子可能比手写慢。若 RMSNorm 太慢，考虑保留它在 CPU（混合后端）
- **RoPE**：5 个 element-wise 算子组合，可能成为瓶颈

**建议的实施顺序（风险驱动）**：
1. 先把 `dml_gemm_smoke.rs` 的 D3D12/DML setup 提取成 `directml_engine.rs` 的 `DmlState`（reusable context）
2. 实现 decode 算子里**最简单的 3 个**（GEMM + SiLU + softmax），跑一个"只含这 3 个"的假 decode 循环，测 **dispatch 开销基线**
3. 如果 dispatch 开销可接受（< 1ms/tok），继续移植 RMSNorm + RoPE + argmax + embed
4. 跑端到端 decode，对比 CPU/CUDA
5. go/no-go 后再决定是否接 audio encoder

**不做**（明确排除，避免范围蔓延）：
- ❌ 不做 DML 算子图融合（DML graph API 复杂，收益不确定）
- ❌ 不做 INT8 量化（DML 路径 f16，和 CPU INT8 是不同优化轴）
- ❌ 不做 prefill（先用 CPU prefill 当起点，DML 只跑 decode；prefill 后期再加）
- ❌ 不做 streaming（先验证核心 decode）

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
- **CER 对照工具**：`examples/cer_compare.rs`（解析两次 transcribe stdout 算 raw + 归一化 CER）。峰值 RSS 已折进 `cpu_transcribe.rs::run_cpu_with`（in-process Win32 `GetProcessMemoryInfo`，每 fixture 独立进程，优先用这个）；`examples/mem_probe.ps1` 是 100ms 外部轮询的 fallback（死掉的 `QASR_CPU_INT8` 开关于 2026-06-13 移除，INT8 永久）。

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
