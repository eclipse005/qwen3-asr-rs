# CUDA 性能重测计划：Rust 旧版 vs Python 旧后端 vs Transformers-native

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在干净的 GPU 状态下重新测试 `D:\qwen3-asr-rs\models` 中的旧版 0.6B/1.7B 模型，对比 Rust 实现、`D:\asr` 中的 Python `qwen-asr` 旧后端，以及 `D:\asr\models` 中的 HF-native `transformers` 新版；记录显存占用；最终更新 `README.md` 并只推送它。

**Architecture:** 复用现有脚本 `scripts/bench_original.py`、`scripts/bench_hf_native.py`、Rust 测试 `tests/bench_cuda.rs` 和汇总脚本 `scripts/summarize_all.py`，重新生成 TSV 和汇总报告。

**Tech Stack:** Rust (cargo), Python 3.10 (conda `asr`), PyTorch, transformers 5.13+, nvidia-smi.

## Global Constraints

- 只修改并提交 `README.md`；`scripts/` 和 `tests/bench_cuda.rs` 不进入 git。
- 使用 `D:\qwen3-asr-rs\models` 中的旧版模型（`Qwen3-ASR-0.6B`、`Qwen3-ASR-1.7B`）。
- 使用 `D:\asr\models` 中的 HF-native 新版模型（`Qwen3-ASR-0.6B-hf`、`Qwen3-ASR-1.7B-hf`）。
- 音频固定为 `D:\qwen3-asr-rs\tests\fixtures` 中的 6 条样本。
- 记录峰值显存（`nvidia-smi` 后台轮询）。
- 设备为 NVIDIA P104-100（8 GiB，Compute Capability 6.1），`torch.compile` 预期因 Triton 不支持 CC 6.1 而失败，需在 README 中说明。

---

### Task 1: 确认 GPU 空闲并清理环境

**Files:**
- None

**Interfaces:**
- Consumes: 当前系统 GPU 状态
- Produces: 无其它占用、GPU 显存 < 200 MiB 的干净环境

- [ ] **Step 1: 检查 nvidia-smi 进程占用**

Run: `nvidia-smi`
Expected: 无其他显存占用进程，已用显存 < 200 MiB。

- [ ] **Step 2: 清理 PyTorch cache**

Run:
```bash
conda activate asr
python -c "import torch; torch.cuda.empty_cache(); torch.cuda.synchronize()"
```
Expected: 命令成功返回。

---

### Task 2: 重新运行 Rust CUDA benchmark

**Files:**
- Modify: `target/rust_bench_cuda_v2.tsv`（覆盖写入）

**Interfaces:**
- Consumes: Rust 源码、旧版模型、`tests/fixtures`
- Produces: `target/rust_bench_cuda_v2.tsv`

- [ ] **Step 1: 运行 Rust bench**

Run:
```bash
cargo test --release --test bench_cuda -- --ignored --test-threads=1
```
Expected: 12 条记录全部完成，TSV 写入 `target/rust_bench_cuda_v2.tsv`。

- [ ] **Step 2: 检查 TSV 行数**

Run:
```bash
wc -l target/rust_bench_cuda_v2.tsv
```
Expected: 至少 13 行（1 行表头 + 12 条数据）。

---

### Task 3: 重新运行 Python 旧后端 benchmark

**Files:**
- Modify: `target/python_bench_original_v2.tsv`（覆盖写入）

**Interfaces:**
- Consumes: `scripts/bench_original.py`、conda `asr`、旧版模型
- Produces: `target/python_bench_original_v2.tsv`

- [ ] **Step 1: 运行 Python 旧后端**

Run:
```bash
conda activate asr
python scripts/bench_original.py > target/python_bench_original_v2.tsv
```
Expected: 12 条记录全部成功，文件为 UTF-8。

- [ ] **Step 2: 检查 TSV 行数**

Run:
```bash
wc -l target/python_bench_original_v2.tsv
```
Expected: 至少 13 行。

---

### Task 4: 运行 HF-native benchmark（无 torch.compile）

**Files:**
- Modify: `target/hf_native_bench.tsv`（覆盖写入）

**Interfaces:**
- Consumes: `scripts/bench_hf_native.py`、conda `asr`、HF-native 模型
- Produces: `target/hf_native_bench.tsv`

- [ ] **Step 1: 运行 HF-native**

Run:
```bash
conda activate asr
python scripts/bench_hf_native.py > target/hf_native_bench.tsv
```
Expected: 12 条记录全部成功，文件为 UTF-8。

- [ ] **Step 2: 检查 TSV 行数**

Run:
```bash
wc -l target/hf_native_bench.tsv
```
Expected: 至少 13 行。

---

### Task 5: 可选验证 torch.compile（预期失败）

**Files:**
- Modify: `target/hf_native_compile_bench.tsv`（覆盖写入）

**Interfaces:**
- Consumes: `scripts/bench_hf_native.py compile`
- Produces: 失败日志或错误记录

- [ ] **Step 1: 尝试 torch.compile**

Run:
```bash
conda activate asr
python scripts/bench_hf_native.py compile > target/hf_native_compile_bench.tsv
```
Expected: 报错包含 `Triton` / `compute capability` / `6.1` 等关键字，或所有记录为 ERROR。

---

### Task 6: 生成三合一汇总报告

**Files:**
- Create/Modify: `target/benchmark_all_summary.md`

**Interfaces:**
- Consumes: `target/rust_bench_cuda_v2.tsv`、`target/python_bench_original_v2.tsv`、`target/hf_native_bench.tsv`
- Produces: 干净 UTF-8 的 `target/benchmark_all_summary.md`

- [ ] **Step 1: 以 UTF-8 重新生成汇总**

Run:
```bash
conda activate asr
python scripts/summarize_all.py target/rust_bench_cuda_v2.tsv target/python_bench_original_v2.tsv target/hf_native_bench.tsv > target/benchmark_all_summary.md
```
Expected: `file target/benchmark_all_summary.md` 显示 `UTF-8`。

- [ ] **Step 2: 验证可读性**

Run:
```bash
file target/benchmark_all_summary.md
```
Expected: 输出包含 `UTF-8`，不含 `ISO-8859`。

---

### Task 7: 更新 README.md Benchmark 章节

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: `target/benchmark_all_summary.md` 中的数据
- Produces: 新的 Benchmark 表格和结论

- [ ] **Step 1: 用汇总数据替换 Benchmark 章节**

将 `README.md` 中 `## Benchmark` 到下一个 `##` 之间的内容替换为：
- 新的三合一对比表格（Rust / Python 旧后端 / HF-native）。
- 新的结论文字。
- 保留运行方式，补充 `scripts/bench_hf_native.py` 的用法。
- 加入 P104-100 不支持 `torch.compile` 的说明。

- [ ] **Step 2: 预览 README 无乱码**

Run:
```bash
file README.md
```
Expected: `UTF-8`。

---

### Task 8: 只提交并推送 README.md

**Files:**
- Modify: git index / remote

**Interfaces:**
- Consumes: 已修改的 `README.md`
- Produces: 远程仓库更新

- [ ] **Step 1: 添加并提交**

Run:
```bash
git add README.md
git commit -m "docs: update benchmark with Rust / old Python / HF-native comparison"
```
Expected: 提交成功，只包含 `README.md`。

- [ ] **Step 2: 推送**

Run:
```bash
git push
```
Expected: 推送成功。

---

## Self-Review

1. **Spec coverage:**
   - Rust 旧版重测：Task 2。
   - Python 旧后端重测：Task 3。
   - HF-native 测试：Task 4。
   - 显存记录：所有 bench 脚本已内置。
   - torch.compile 验证：Task 5。
   - README 更新并只推送：Task 7、Task 8。
2. **Placeholder scan：** 无 TBD/TODO，所有命令可执行。
3. **Type consistency：** TSV 列格式与现有脚本保持一致。
