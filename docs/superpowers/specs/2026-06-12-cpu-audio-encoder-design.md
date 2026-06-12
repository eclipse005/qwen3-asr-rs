# Design: CPU 端 transcribe() 端到端跑通

**日期**：2026-06-12
**作者**：brainstorm session
**状态**：已设计，待用户 review

## 1. 目标

**在 CPU 路径上端到端跑通 `transcribe()`**：
- 输入：wav
- 输出：`TranscribeResult`（text + language + raw_output）
- 验收标准：`Backend::Cpu` 跑出来的 `result.text` 跟 `Backend::Cuda` 跑出来的同 fixture 字节一致

**非目标**（明确不做）：
- 不引入 f16 SIMD（AVX-512-FP16 / std::simd）
- 不做 RTFx 优化（能跑通就好）
- 不删除 cudarc / cuda feature
- 不动 `cudarc_engine.rs` / `gpu_audio_encoder.rs` / 现有 13 个 CUDA 测试
- 不实现 streaming API（ROADMAP §3.5 / §4.3 P2）
- 不重写 conv stem permute GPU kernel（ROADMAP §3.3）

## 2. 范围（包含什么）

1. **新增** `src/cpu_audio_encoder.rs`：CPU 端手写 f32 音频编码器（mel → conv stem → transformer → proj）
2. **修改** `src/inference.rs::encode_audio_cpu`：从占位符 Err 替换成真实实现
3. **新增** `tests/cpu_transcribe.rs`：1 个 `#[ignore]` 集成测试，对比 CPU vs GPU fixture
4. **新增** `tests/fixtures/expected/gpu_sample1.txt`（或类似文件）：CUDA 跑出来的 ground truth
5. **更新** `ROADMAP.md`：§3.2 标"已实现"、新增 P0 条目、tree 节点加 `cpu_audio_encoder.rs`

**不修改**：`cpu_engine.rs`（文本解码器已 f32 实现）、`mel.rs`（CPU f32 已实现）、`config.rs`（架构不需要新参数）。

## 3. 架构

### 3.1 主路径

```
transcribe(wav) — Backend::Cpu
  ├─ mel.rs::compute_mel()           — 已有，CPU f32
  ├─ CpuAudioEncoder::forward()       ← NEW
  │    ├─ ConvStem.forward()         ← NEW: 3 × {im2col + gemm + bias + GELU}
  │    ├─ ConvOut.forward()          ← NEW: Linear + bias
  │    ├─ SinusoidalPE.add()         ← NEW: PE 加法（half sin + half cos 拼成 d_model 维）
  │    ├─ 18 × TransformerLayer      ← NEW: {LN + windowed_attn + LN + FFN}
  │    └─ LnPost + Proj1 + GELU + Proj2  ← NEW
  │    → Vec<f32> shape [n_total, hidden]  (与 GPU run() 输出同 shape)
  ├─ CpuTextDecoder::forward()       — 已有，CPU f32，接口不变
  ├─ argmax()                        — 已有
  └─ tokenizer.decode() → TranscribeResult
```

### 3.2 类型统一

- **所有算子**用 `cpu_engine.rs` 已经定义的 `CpuTensor` / `CpuWeight`（`pub(crate)`）
- **不**重新定义 `CpuAudioTensor` 之类 —— 跨模块零摩擦
- `cpu_audio_encoder.rs` 也加 `#[cfg(feature = "cpu")]`

### 3.3 与 GPU 路径 1:1 对应

架构上每个组件 `GpuXxx` 都有 `CpuXxx` 对应：
- `GpuLinear` → `CpuLinear`（`linear()` 已在 `cpu_engine.rs` 里有，等价物）
- `GpuLayerNorm` → `CpuLayerNorm`（手写 f32 LN，仿 `add_residual_rms_norm` 但 LN eps 1e-5 不是 rms_norm 的 1e-6）
- `GpuAudioAttention` → `CpuAudioAttention`（Q/K/V/O 投影 + windowed attention）
- `GpuAudioFfn` → `CpuAudioFfn`（fc1 + GELU + fc2）
- `GpuAudioLayer` → `CpuAudioLayer`（pre-LN residual）
- `GpuConvStem` → `CpuConvStem`（3 × conv2d + conv_out + PE）
- `GpuAudioEncoder` → `CpuAudioEncoder`（完整 forward）

**价值**：算子出错时，diff `gpu_audio_encoder.rs` 同一函数能 1:1 对照。

### 3.4 接口形状（重要）

- `CpuAudioEncoder::forward(&self, mel: &[f32], n_mels: usize, mel_len: usize) -> Result<Vec<f32>>`
  - `mel`：`[n_mels * mel_len]` 一段展平 f32 mel（从 `mel.rs::compute_mel` 出来）
  - `n_mels`：固定 128（来自 `mel.rs` 内部约定，不参数化）
  - `mel_len`：T_mel
  - 返回：`Vec<f32>` shape `[n_total, d_model_out]`，其中 `n_total = mel_len / 8`（conv 三次 stride 2 后）
- 内部全部用 `CpuTensor`，但**入口/出口**用 `Vec<f32>`，跟 `inference.rs::encode_audio_cpu` 的现有签名对齐（它当前是 `-> Vec<f32>`，shape `[nat, hidden_size]`）

## 4. 关键设计细节

### 4.1 Conv stem

- **输入**：mel `[T_mel, n_mels=128]`，视为 `[1, 1, n_mels, T_mel]` 4D 张量
  - 注意：GPU 路径是 **chunked**（`mel_chunks: [b_chunks, 1, 128, cs]`），CPU 路径是**整段**（没有显存限制）
  - b_chunks 在 CPU 路径 = 1
- **3 × conv2d**：每层 `Conv2D(n_in, n_out, kernel=3, stride=2, pad=1) + bias + GELU`
  - layer 0: 128→128，f 1→1，t T_mel→T_mel/2
  - layer 1: 128→128，stride 2×2
  - layer 2: 128→128，stride 2×2
  - 输出 shape `[1, 128, 1, T_mel/8]`
- **实现**：im2col（手写，f32） + `gemm` crate + 手写 f32 bias_add + 手写 f32 GELU
  - im2col 输出 `[col_count, 3*3*n_in]`，`col_count = c_out * t_out`
  - gemm: `w [c_out, 3*3*n_in] @ im2col [3*3*n_in, col_count] = out [c_out, col_count]`
- **permute `[b, c, f, t] → [b, t, c, f]`**：CPU 路径**直接 f32 内存拷贝**（4 重循环，按 cache line 顺序读+写）
  - 这里跟 GPU 路径的"下载→CPU permute→上传"思路一致，但**全程 CPU**，无序列化成本
  - 对应 GPU 路径的 ROADMAP §3.3 TODO 不适用本任务
- **ConvOut**：单层 Linear `[c_out, d_model] + bias`，用 `gemm`

### 4.2 Sinusoidal PE

- **公式**（来自 `gpu_audio_encoder.rs::GpuConvStem::load`，line 220-228）：
  ```
  for p in 0..max_pos:
    for i in 0..half:  // half = d_model / 2
      a = p * exp(-i * ln(10000) / (half - 1))
      pe[p, i]       = sin(a)
      pe[p, half+i]  = cos(a)
  ```
- **存储**：`Vec<f32>` shape `[max_pos, d_model]`
- **应用**：ConvOut 输出 `[b, t, d_model]` 上加 `pe[:t, :]`（广播到 b）
- **half = 64** for `d_model = 128`，`max_source_positions` 从 `AudioEncoderConfig` 读

### 4.3 18-layer Transformer

每层 = **pre-LN residual**：
```
h1 = x + windowed_attn(layer_norm(x))
h2 = h1 + ffn(layer_norm(h1))
```

#### 4.3.1 Windowed attention 参数

来自 `gpu_audio_encoder.rs::GpuAudioEncoder::run`，line 341-344：
```
cs2  = n_window * 2
tpc  = feo(cs2)        // 三次 ceil((l-1)/2 + 1) 嵌套
cpw  = n_window_infer / cs2
ws   = tpc * cpw
```

CPU 路径**必须用同一个 `ws`** —— 任何偏差会导致 attention 视野不同，结果和 GPU 不字节对齐。

**关键不变量**：windowed attention 把 seq 分成大小为 `ws` 的 chunk，每个 chunk 内的 token 互相关注；不同 chunk 之间不直接关联。CPU 实现要**严格按这个**做。

#### 4.3.2 Attention 内部

- Q/K/V 用 `linear()` (m=1 GEMV 在 decode，m=batch 在 prefill —— 但音频编码器 seq 比较长，**总是 m=batch** 走 gemm crate)
- softmax + scale + multiply V 走手写 f32 循环
- 不需要 causal mask（音频编码器不是自回归）

### 4.4 LnPost + Proj1 + GELU + Proj2

- 18-layer 输出 → LN → Proj1 (linear) → GELU → Proj2 (linear)
- 全部 f32
- 输出 `Vec<f32>` shape `[n_total, d_model_out]`

### 4.5 Packing（chunked mel → single seq）

GPU 路径做这件事（`GpuAudioEncoder::run` line 332-338）—— CPU 路径下 b_chunks = 1，所以**没有 tail chunk 的问题**，n_total = t2 = T_mel/8。**直接展平**。

**简化**：CPU 路径**没有 chunking 需求**（CPU 内存够），但 conv 输出 `t2` 是确定的；n_total = t2 一行就完事。

## 5. 验证策略

### 5.1 Ground truth 来源

- **现状**：`tests/fixtures/` 已有 wav（`sample1.wav`, `15s.wav`, ...）
- **缺**：没有任何 expected 文本
- **第一步**：在 P104-100 上跑 `cargo test --release --features cuda --test transcribe -- --ignored --nocapture test_q06_sample1`，把打印的 `result.text` 复制到 `tests/fixtures/expected/gpu_sample1.txt`
- **一次**就够（CPU 实现要保证字节对齐一次，以后所有测试都基于此 ground truth）

### 5.2 集成测试

**新增** `tests/cpu_transcribe.rs`：
```rust
#[test]
#[ignore]  // 需要 --features cpu，手动跑
fn test_cpu_q06_sample1() {
    let backend = qwen3_asr::Backend::Cpu;  // 显式 CPU
    let engine = qwen3_asr::AsrInference::load(
        std::path::Path::new(&model_dir_06()), backend,
    ).expect("load 0.6B on CPU");
    let result = engine.transcribe(
        &fixture("sample1.wav"),
        qwen3_asr::TranscribeOptions::default(),
    ).expect("cpu transcribe");
    let expected = std::fs::read_to_string("tests/fixtures/expected/gpu_sample1.txt")
        .expect("read expected fixture");
    assert_eq!(result.text, expected.trim());
}
```

**运行**：`cargo test --release --features cpu --test cpu_transcribe -- --ignored --nocapture --test-threads=1`

### 5.3 增量验证（开发期）

- **第一步**：CPU 音频编码器 + mel → conv_out 输出，shape 对、值域合理（f32 不爆 NaN）
- **第二步**：加上 18-layer transformer，**对比 GPU `run_transformer`** 路径 —— 这是个干净的对照点（输入是 `conv_out`，输出是 `proj2` 输出）
- **第三步**：拼到 transcribe 入口，跑集成测试

## 6. 实现里程碑（粗粒度）

1. **M0**：spec 落地，开始写代码
2. **M1**：`cpu_audio_encoder.rs` 骨架 + `CpuConvStem`（conv2d 用 im2col + gemm）—— 单元级能跑
3. **M2**：`CpuConvStem` 加 PE —— 跟 GPU ConvStem 输出对齐
4. **M3**：18 × `CpuAudioLayer`（LN + attn + FFN）—— 跟 GPU `run_transformer` 输出对齐
5. **M4**：`LnPost + Proj1 + GELU + Proj2` —— 跟 GPU `run` 输出对齐
6. **M5**：`inference.rs::encode_audio_cpu` 接入 + `tests/cpu_transcribe.rs` 跑通

每个 M 的退出标准是 **CUDA 路径输出在数值上可对照**（不需要完全 byte-identical，f32 累加序可能差 1 ulp；但 text 输出应该一致）。

## 7. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| **数值对齐失败** | CPU/GPU 输出 text 不同 | §6 里程碑 M1~M4 增量验证逐段对照；先验 f32 值域再验 byte-identical |
| **windowed attention ws 算错** | attention 视野不一致 | 4.3.1 强调**严格沿用** `feo()` 公式；写一个单测 `test_feo_matches_gpu` |
| **conv2d stride/pad 错** | t_out 跟 GPU 不同 | 单测：f32 跑 1 层 conv 后 t_out 等于预期值；再跟 `gpu_audio_encoder.rs` 对照 |
| **PE 公式错** | conv_out + PE 错位 | 直接把 PE 计算抄到 `cpu_audio_encoder.rs`（不抽 helper），跟 `gpu_audio_encoder.rs::GpuConvStem::load` line 220-228 byte-identical |
| **chunked vs unchunked 误判** | CPU 路径要走 chunked 才行 | §4.5 明确：CPU 路径 b_chunks=1，n_total = t2；不引入 tail chunk 逻辑 |
| **f32 数值 vs f16 数值** | 模型权重在 safetensors 里是 f16（Qwen3 标准），CPU 路径 upcast 到 f32 | 跟现有文本解码器一样，权重加载时 upcast（`raw_tensor.rs::as_f32` 已经有了） |

## 8. 不在本 spec 范围内（后续 PR 候选）

- **f16 路径**（用户提到"优先测试 f16"—— 留作下一个 PR，**本次不做**）
- **RTFx 优化**（手写 m=1 GEMV 已经做过；其他手写算子跑通后再考虑 f16 权重存储）
- **CPU streaming**（ROADMAP §3.5 / §4.3 P2）
- **Conv stem GPU permute kernel**（ROADMAP §3.3 — CPU 路径不需要）

## 9. ROADMAP 更新草案

- §3.2 CPU 路径不完整 → 改成"CPU 音频编码器（已实现，f32 路径）"
- §4.1 P0 加一条："跑 CPU 集成测试（`cargo test --release --features cpu --test cpu_transcribe -- --ignored --nocapture`），确认 text 字节对齐"
- §1.1 tree 加：`src/cpu_audio_encoder.rs # ★ 手写 CPU 音频编码器（f32 im2col + gemm + 手写 LN/attn/FFN）`
- §2.2 加一段：CPU forward 主路径（镜像 §2.1 CUDA forward）
- §0 加一句：CPU 路径端到端可用，但 RTFx 显著低于 CUDA（量级说明：CUDA 15s ~24× vs CPU 15s ~1-2× 是预期）
