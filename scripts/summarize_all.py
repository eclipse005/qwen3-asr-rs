#!/usr/bin/env python3
"""Combine Rust / qwen-asr Python / HF-native Python benchmark TSVs into one report.

Usage:
    python scripts/summarize_all.py \
        target/rust_bench_cuda_v2.tsv \
        target/python_bench_original_v2.tsv \
        target/hf_native_bench.tsv \
        > target/benchmark_all_summary.md
"""

import sys
from pathlib import Path


def parse_line(line: str) -> dict | None:
    line = line.strip()
    if not line or line.startswith("backend\t"):
        return None
    parts = line.split("\t")
    if len(parts) < 11:
        return None
    backend, model, wav, elapsed_s, rtfx, peak_field, baseline_field, after_load_field, after_transcribe_field, lang, text = parts[:11]

    def _extract_miB(field: str) -> float | None:
        if field == "n/a":
            return None
        if "=" in field:
            field = field.split("=", 1)[1]
        field = field.replace("MiB", "").strip()
        try:
            return float(field)
        except ValueError:
            return None

    def _extract_sec(elapsed: str) -> float | None:
        elapsed = elapsed.replace("s", "").strip()
        try:
            return float(elapsed)
        except ValueError:
            return None

    return {
        "backend": backend,
        "model": model,
        "wav": wav,
        "elapsed_s": _extract_sec(elapsed_s),
        "rtfx": float(rtfx.replace("x", "")) if "x" in rtfx else None,
        "peak_mib": _extract_miB(peak_field),
    }


def load_results(path: Path) -> dict:
    results = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = parse_line(line)
            if r is None:
                continue
            results[(r["model"], r["wav"])] = r
    return results


def fmt(val, unit=""):
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.2f}{unit}"
    return f"{val}{unit}"


def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <rust.tsv> <python_qwen_asr.tsv> <hf_native.tsv>", file=sys.stderr)
        sys.exit(1)

    rust = load_results(Path(sys.argv[1]))
    py_old = load_results(Path(sys.argv[2]))
    py_hf = load_results(Path(sys.argv[3]))

    keys = sorted(set(rust.keys()) | set(py_old.keys()) | set(py_hf.keys()), key=lambda k: (k[0], k[1]))

    print("# Qwen3-ASR CUDA 性能对比（Rust / Python 旧后端 / Transformers-native）")
    print()
    print("- 设备：NVIDIA P104-100（8 GiB，Compute Capability 6.1）")
    print("- Rust：`cargo test --release --test bench_cuda -- --ignored --test-threads=1`")
    print("- Python 旧后端：`conda activate asr` + `qwen-asr` transformers backend")
    print("- Python HF-native：`transformers.AutoModelForMultimodalLM`（无 torch.compile，P104-100 不支持 Triton）")
    print()
    print("| 模型 | 音频 | Rust 耗时/RTFx/显存 | Python 旧后端 | HF-native | Rust vs 旧后端 | Rust vs HF-native | HF-native vs 旧后端 |")
    print("|------|------|---------------------|---------------|-----------|----------------|-------------------|---------------------|")

    for key in keys:
        model, wav = key
        r = rust.get(key, {})
        po = py_old.get(key, {})
        ph = py_hf.get(key, {})

        r_elapsed = r.get("elapsed_s")
        po_elapsed = po.get("elapsed_s")
        ph_elapsed = ph.get("elapsed_s")

        speedup_old = po_elapsed / r_elapsed if r_elapsed and po_elapsed else None
        speedup_hf = ph_elapsed / r_elapsed if r_elapsed and ph_elapsed else None
        hf_vs_old = po_elapsed / ph_elapsed if po_elapsed and ph_elapsed else None

        def cell(res):
            return f"{fmt(res.get('elapsed_s'), 's')} / {fmt(res.get('rtfx'), 'x')} / {fmt(res.get('peak_mib'), ' MiB')}"

        print(
            f"| {model} | {wav} | {cell(r)} | {cell(po)} | {cell(ph)} | "
            f"{fmt(speedup_old, 'x')} | {fmt(speedup_hf, 'x')} | {fmt(hf_vs_old, 'x')} |"
        )

    print()
    print("说明：")
    print("- 加速比 > 1 表示行首方案更快。")
    print("- `Rust vs 旧后端`：Python 旧后端耗时 / Rust 耗时。")
    print("- `Rust vs HF-native`：HF-native 耗时 / Rust 耗时。")
    print("- `HF-native vs 旧后端`：Python 旧后端耗时 / HF-native 耗时。")
    print("- torch.compile 在 P104-100 上因 Triton 不支持 CC 6.1 而无法运行。")


if __name__ == "__main__":
    main()
