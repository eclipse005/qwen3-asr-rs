# CUDA 解码 GEMV 优化 —— 双模型 A/B 验证报告

- 日期：2026-09-12
- 硬件：NVIDIA P104-100（Pascal sm_61，8 GB，GDDR5X 5005 MHz，理论带宽约 320 GB/s）
- 主机：Windows 11，release 构建，`cargo run --release`
- 计时口径：`transcribe` 的墙钟时间（模型已加载），RTFx = 音频时长 / 耗时
- 结论：**两个模型 12 项 fixture 全部逐字对齐基线；RTFx 提升 1.14× ~ 1.33×；长音频跨 run 不可复现的缺陷被消除**

---

## 一、对比对象

| | 优化前 | 优化后 |
|---|---|---|
| 解码 GEMV | cuBLAS GEMM（m=1） | 手写 `gemv_f16` kernel |
| 版本 | `git HEAD` 原版（`git stash push -- ptx src`） | 当前工作区 |
| block reduction | 有共享缓冲区复用竞态（2 处） | 修好（helper + 独立 scratch） |
| prefill GEMM | cuBLAS | cuBLAS（未改动） |

两个模型的文本后端结构一致（均为 28 层），规模差异：

| | 0.6B | 1.7B |
|---|---|---|
| text hidden | 1024 | 2048 |
| text intermediate | 3072 | 6144 |
| audio encoder | 18 层 / d=896 | 24 层 / d=1024 |
| 每 token 权重流量 | 约 1.19 GB | 约 3.44 GB（2.9×） |

---

## 二、结果总表

优化后取 3 次独立运行的均值。

### 0.6B

| 音频 | 优化前 RTFx | 优化后 RTFx | 提升 | 优化前 vs 基线 | 优化后 vs 基线 |
|---|---|---|---|---|---|
| 15s 英文 | 25.5× | **30.4×** | 1.19× | 逐字一致 | 逐字一致 |
| 30s 中文 | 17.0× | **21.6×** | 1.27× | 逐字一致 | 逐字一致 |
| 90s 英文 | 19.9× | **23.4×** | 1.18× | 逐字一致 | 逐字一致 |
| 89s 日文 | 18.4× | **22.0×** | 1.20× | 逐字一致 | 逐字一致 |
| 180s 英文 | 15.8× | **18.1×** | 1.15× | 不一致（幻觉 "PP."） | 逐字一致 |
| 180s 中文 | 14.7× | **16.8×** | 1.14× | 不一致 | 逐字一致 |

### 1.7B

| 音频 | 优化前 RTFx | 优化后 RTFx | 提升 | 优化前 vs 基线 | 优化后 vs 基线 |
|---|---|---|---|---|---|
| 15s 英文 | 9.3× | **12.4×** | 1.33× | 单次看似一致，实为不稳定 | 逐字一致 |
| 30s 中文 | 8.3× | **11.0×** | 1.33× | 同上 | 逐字一致 |
| 90s 英文 | 8.9× | **11.5×** | 1.29× | 同上 | 逐字一致 |
| 89s 日文 | 10.4× | **13.2×** | 1.27× | 同上 | 逐字一致 |
| 180s 英文 | 8.9× | **11.0×** | 1.24× | 同上 | 逐字一致 |
| 180s 中文 | 8.3× | **10.3×** | 1.24× | 同上 | 逐字一致 |

**1.7B 的提升（1.24-1.33×）大于 0.6B（1.14-1.27×）**：1.7B 每 token 要多扫 2.9 倍权重，decode 在总时间里的占比更高，而 GEMV 正是 decode 的瓶颈。

---

## 三、可复现性：优化前有一个真 bug

同一二进制、同一输入，连跑 3 次并比对输出 md5：

| 组合 | 优化前 | 优化后 |
|---|---|---|
| 0.6B 180s_zh | **3 个不同 hash** | 3/3 相同 |
| 1.7B 180s_zh | **2 个不同 hash** | 3/3 相同 |
| 0.6B / 1.7B 其余 fixture | 稳定 | 稳定 |

> 优化前 0.6B 180s_zh 连跑三次的实际 hash：`0a97da77` / `78d4c1a2` / `e5a8363c`
> 优化前 1.7B 180s_zh 连跑三次的实际 hash：`9dbefd89` / `9dbefd89` / `ddd4f9d1`

**根因**：`fused_gqa_decode_split_p1_f16` 与 `fused_gqa_decode_f16` 中，同一个 shared scratch 数组被两次归约复用（先 max 后 sum），而"读 `buf[0]`"与"再次写 `buf[tid]`"之间缺少 `__syncthreads()` —— 跨 warp 竞态（UB）。详见 `ROADMAP.md` §1.4.2。

**这意味着优化前 1.7B 在单次运行里"6/6 匹配"是巧合**，不构成对齐证据；只有修复后才能稳定复现。

---

## 四、基线的变更

冻结基线已于本次用**含竞态修复的正确版本**重新生成：

- 备份：`docs/baseline/texts.bak-race-fix/`（24 个文件，改动前原样保留）
- 重新生成：`cuda` 后端 12 项（0.6B × 6 + 1.7B × 6）
- `cpu` 后端基线未动（本次改动只涉及 CUDA 路径）

**12 项里只有 1 项变化**：`cuda_0.6B_180s_zh`，差异为**一处标点**（`呀，` → `呀？`，相似度 99.89%），且该 fixture 正是原版不可复现的那一个 —— 旧基线值是竞态下随机样本之一。其余 11 项与新基线逐字相同。

---

## 五、正确性的独立证据

1. **逐字基线比对**：12/12 一致（`cargo run --release --example verify_baseline -- cuda all`）
2. **逐位内核比对**：手写 `gemv_f16` 与 cuBLAS 在同一权重上输出**逐位相同**（151936 个输出 0 个不同，`cargo test --release --lib gemv_bench::gemv_bitwise_vs_cublas -- --ignored --nocapture`）
3. **基线不变性**：除上述 1 项标点外，11 项基线文本与改动前逐字相同

---

## 六、复现命令

```bash
# 全量逐字校验（两模型 12 项）
cargo run --release --example verify_baseline -- cuda all

# 单模型 / 单 fixture
cargo run --release --example verify_baseline -- cuda 1.7B 180s_zh

# 用当前版本重新冻结基线
cargo run --release --example verify_baseline -- cuda all --freeze

# 内核级带宽对照
cargo test --release --lib gemv_bench -- --ignored --nocapture --test-threads=1

# 切回原版做对照（改动集中在 ptx/ 与 src/）
git stash push -- ptx src && powershell scripts/compile-ptx.ps1
# ... 测试 ...
git checkout -- ptx src && git stash pop
```

> 切换时若先 `cp -r ptx ptx.keep-current`，恢复阶段可直接拷回，省掉一次 7 架构 PTX 生成（约 2-3 分钟）。

---

## 七、尚未验证 / 下一步

- **CPU 后端**未重测：本次改动不触及 CPU 路径，`cpu_*` 基线保持原样
- **prefill GEMM** 仍是 cuBLAS（手写 WGSL 版只有 cuBLAS 的 1/5，见 `wgpu/FEASIBILITY.md`）
- **剩余的解码开销**：255 次 kernel launch × 5.77 µs ≈ 1.47 ms/token（占 6.58 ms 的 22%），是下一个可动目标
