# ROADMAP — qwen3-asr

> 写给下一个接手会话的 AI。读这一篇就能在 5 分钟内进入状态。

## 0. 项目是什么

Qwen3-ASR（0.6B / 1.7B）的 Rust 推理库。在 NVIDIA P104-100（Pascal sm_61）上做过极限优化，单卡 RTFx 在 0.6B 15s 上 ~24×、30s ~21×、90s ~19×。当前唯一能端到端跑通的是 **CUDA 路径**（cuBLAS + NVRTC 手写 kernel）。CPU 路径只实现了**文本解码器**（gemm + rayon），CPU 音频编码器**未实现**——CPU 端 `transcribe()` 在运行时直接报错。

仓库里没有 burn 框架代码——这次重构把它彻底移除了，引擎全是手写的。`burn_cubecl` / `cubecl` / `burn` 都不在依赖里。

## 1. 当前在哪

### 1.1 树

```
src/
├── backend.rs           # Backend 枚举（Cuda | Cpu）+ best() 调度
├── config.rs            # AsrConfig serde
├── error.rs             # AsrError / Result
├── hub.rs               # (可选) hub 模式下载
├── mel.rs               # mel spectrogram（CPU, f32）
├── inference.rs         # AsrInference 装配 + mel→embed→generate 主循环
├── raw_tensor.rs        # safetensors 原始字节 view（weight loading 用）
├── cpu_engine.rs        # 手写 CPU 文本解码器（gemm + rayon）
├── cudarc_engine.rs     # ★ 手写 GPU 文本解码器（cuBLAS + NVRTC kernel）+ DecodeScratch 复用
├── gpu_audio_encoder.rs # ★ 手写 GPU 音频编码器（cuBLAS + 自定义 conv2d/im2col）
└── kernels/kernels.cu   # 所有 CUDA kernel（运行时 NVRTC 编译）
tests/transcribe.rs       # 13 个 #[ignore] 集成测试（0.6B/1.7B × 各种时长）
scripts/bench.ps1        # 每个 test 独立 `cargo test` 进程跑（避免 cuBLAS/cache 跨测串味）
```

### 1.2 公共 API

`lib.rs` 只导出 `Backend`、`AsrError` / `Result`、`AsrInference` / `TranscribeOptions` / `TranscribeResult`、`load_audio_wav`。**没有** `StreamingOptions` / `StreamingState` / `best_device()`（这次重构删了 streaming.rs）。

### 1.3 Features

```toml
default = ["cuda"]      # 端到端可用
cuda = ["dep:cudarc"]
cpu  = ["dep:gemm", "dep:rayon"]  # 只文本；transcribe 跑不通
hub  = ["dep:reqwest"]
```

### 1.4 性能（基准线，0.6B 模型，P104-100）

| 音频 | RTFx |
|---|---|
| 15s 英文 | ~24× |
| 30s 中文 | ~21× |
| 90s 英文 | ~19× |
| 180s 英文 | ~17× |
| 1.7B-15s | ~11× |

数字来源：`tests/transcribe.rs`（必须 `--features cuda` + `--ignored --nocapture --test-threads=1`）。

## 2. 关键架构事实

### 2.1 GPU 解码器的主路径（`inference.rs::generate_cuda`）

```
prefill（一次）：
  build hidden_states (CPU splice audio embeds) → upload
  cos/sin MRoPE 表预计算（CPU）+ upload
  GpuKvCache 预分配
  decoder.forward(hs, cos, sin, kv, 0, true, true)  → logits [1, 1, vocab]
  argmax → token_buf[0]   # 第一次用
decode loop（每步）：
  embed_id_from_gpu_slot_into(embed_table, token_buf, 0, h_buf)  # 不 htod
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
- 注意：decode 步 28×1 步用 `fused_gqa_decode_split`（chunk=256 或 512），prefill 走 `fused_gqa_decode_split`（当 cur_len>1024）否则走非 split——两边在数学上等价

### 2.2 优化历史（看代码看到满地黑科技时参考）

数字从旧 HANDOFF 摘来，对应 0.6B 模型 15s 英文 cold-start RTFx。**前 8 步在父仓库里做**（本仓 `250ca4d` 是 extract commit），本仓只看到后面 4 步的 commit。历史脉络（从低到高）：
1. baseline burn-cubecl → 0.25x
2. 手写 GpuTensor + cuBLAS：0.55x
3. NVRTC element-wise + GPU KV cache：1.4x
4. GPU conv stem（im2col + cuBLAS）：6.4x
5. fused_gqa_decode 融合（Q·K + softmax + ·V 一次 launch）：7.6x
6. fused QKV extract + RMSNorm + rotary + cache 写一次：9.8x
7. linear_gpu_accum（cuBLAS beta=1 省残差 launch）：10.6x
8. fused_gqa_decode_split（flash-attn 2-kernel，跨 block 并行）：12.5x
9. **alloc_uninit_f16（Pascal driver enqueue 限速 → memset 占 80% 的 launch time）**：23× — **最大的单步收益** ← 仓内 commit `6251120`
10. skip clone_tensor at O-proj：23.5× ← 仓内 `50d2524`
11. GPU-resident next-token（argmax_into_slot + embed_lookup_single_i32）：23.7× ← 仓内 `4675c74`
12. fused QKV extract kernel：24× ← 仓内 `f1df993`

**未完成 / 失败 / 不可行**：
- **CUDA Graph 捕获 decode 步** — 已删（详见 §3.1）。P104 是 sm_61，CUDA Graph 需要 sm_70+。
- **lm_head GEMV + argmax 融合** — kernel 存在（`lm_head_gemv_argmax_f16`）但注释里说"目前输给 cuBLAS"，未用。
- **conv stem permute 走 GPU** — `gpu_audio_encoder.rs::GpuConvStem::forward` 末尾还有一段 download→CPU permute→upload，是已知遗留。

## 3. 卡点 / 已知问题

### 3.1 [已删] CUDA Graph 死代码

重构前留了一段 `DecodeGraph`（~140 行）+ `set_cublas_workspace`（~40 行），注释说"capture 返回 STREAM_CAPTURE_UNSUPPORTED"。整个推理路径不用。**已经删除**。如果你看到类似东西进来，删掉并更新 ROADMAP。

### 3.2 CPU 路径不完整

`encode_audio_cpu` 直接返回 Err，CPU 端只能跑文本解码器。如果有人想跑 `--no-default-features --features cpu`，需要先实现 CPU 音频编码器（建议参考 `gpu_audio_encoder.rs` 的 GpuConvStem + transformer，照搬手写 matmul + LayerNorm + windowed attention）。

### 3.3 Conv stem 走 CPU detour

`gpu_audio_encoder.rs::GpuConvStem::forward` 第 252-264 行：从 GPU 下载 3-conv 后的 tensor，按 `[b, c, f, t] → [b, t, c, f]` permute，重新上传。**单次推理总成本里 ~5-10ms**。修法：写一个 4D permute kernel 替换这段。

### 3.4 已知 dead code（rustc 警告）

- `gpu_audio_encoder.rs::GpuAudioEncoder::run_transformer` — 无人调用，原本注释"legacy path，burn conv stem 上游用的"已过时（burn 删了）。**保留**：未来 streaming 重做时可能用，标注是 "caller uploads conv_out 跳过 conv stem" 的入口。
- `inference.rs::AsrInferenceInner::tokenizer_decode` — 未用，**保留**：可能在 streaming 或 debug API 里复用。
- `inference.rs::AsrInferenceInner::decode_result` — **不是 dead code**，被 `run_inference` 调（line 178）。rustc 误报可以 ignore。

### 3.5 streaming.rs 整个删了

旧的 `StreamingState` 公共 API 没了。如果用户提"流式转写"需求，需要：
1. 在 `gpu_audio_encoder.rs` 复用 conv stem 一次性算全 mel
2. chunked 喂给文本解码器（音频 embeds 按窗口切）
3. 设计 prefix + overlap 拼接策略

### 3.6 测试 fixture 缺失

`tests/fixtures/` 目录存在但具体 wav 文件状态未知。运行测试前先 `ls tests/fixtures/`，缺什么就 huggingface 找或自己录。**没有**对应的 fixture 自动下载脚本。

## 4. 下一步规划（按优先级）

### 4.1 P0：稳定 + 验证

- [ ] **跑一遍 `cargo check --features cuda`** 确认重构后编译通过（rustc 已经报了 dead-code 警告，已知 §3.4）
- [ ] **跑 13 个集成测试**（`cargo test --release --features cuda --test transcribe -- --ignored --nocapture --test-threads=1`），逐个确认 RTFx 数字没有回退
- [ ] 写一个最小 smoke test：输入 5s wav，断言输出非空、语言识别对——纳入 CI

### 4.2 P1：小优化（风险低、明确收益）

- [ ] **修 conv stem permute kernel**（§3.3）— 写 `permute_bctf_to_btcf_f16` kernel，替换 252-264 行
- [ ] **lm_head 融合用回** — 重测 `lm_head_gemv_argmax_f16` 在当前硬件下是否还输 cuBLAS，可能更新了
- [ ] **删 run_transformer + tokenizer_decode** 如果确定未来用不到（§3.4）— 但用户没要求，先观察
- [ ] 减 warning：把 `#[allow(dead_code)]` 加到 §3.4 里的两个保留 API 上，干净

### 4.3 P2：跨平台

- [ ] **CPU 音频编码器**（§3.2）— 这是大工程。先做最小可用版：纯 f32 跑通正确性，再考虑 rayon 并行 / 手写 GEMV
- [ ] **去掉 `Backend::Cpu` 这条死路或完成它** — 现在 CPU 既不能 transcribe 又会给人错觉。两条路：
  - 删 cpu feature（推荐，等真做出来了再加）
  - 或者在 `encode_audio_cpu` 改成"feature detected"——只 `cfg(feature = "cpu")` 才编译，主张"cpu 路径不接受 transcribe，只跑纯文本 forward"
- [ ] streaming API 重建（§3.5）

### 4.4 P3：真正的提速

- [ ] **PinnedHostSlice 异步 argmax**（`cudaStreamAddCallback`）省 50µs/step
- [ ] **kv cache 预 fetch**：prefill 后第一个 decode 步不等 cuBLAS
- [ ] **1.7B 完整 benchmark**（10s/15s/30s/90s/180s 全跑）
- [ ] 看 cuBLASLt 替换 cudarc::cublas（f16 GEMM 在 Pascal 上有 ~10% 提升空间）

### 4.5 不做

- **CUDA Graph** — 硬件不支持
- **ROCm / Metal / Vulkan 路径** — 上一任已经决定不做（cudarc 专属），明确放弃
- **把架构改回多后端**（burn 框架）— 这次重构的目的就是脱离 burn

## 5. 给接手 AI 的具体操作

```
1. 读 §0-§2（本文件）
2. cargo check --features cuda  — 看新警告
3. cargo test --release --features cuda --test transcribe -- --ignored --nocapture --test-threads=1 test_q06_15s
   — 验证主路径不退化
4. 想要新优化前先看 §3 卡点，别重蹈覆辙
5. 改完跑 §4.1 三个验证步骤
```

## 6. 一些不能忘的事实

- **NVRTC 编译时需要 CUDA_PATH**（Windows 通常 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`，Linux 通常 `/usr/local/cuda`）
- **cudarc 0.19**（不是 0.17），`cuda-12080` 特性
- **`__launch_bounds__` 在所有 kernel 上**（Pascal sm_61 register pressure 优化，重写时记得加）
- **f16 + f32 累加**是统一模式，不要降级
- **Pascal driver 有 enqueue 限速**（alloc_zeros → memset 触发），这正是 alloc_uninit 收益巨大的原因
- **测试多线程跑会抢 GPU**——必须 `--test-threads=1`
- **`h_buf` 跨步复用**：decode 步 28 层都改 `h` in-place，最后 `forward_decode_scratch` 里的 final_norm 是 read-only 写到 `scratch.final_norm`，**不修改 h**——下个 iter 的 `embed_id_from_gpu_slot_into` 才覆盖它。这点别重构坏
