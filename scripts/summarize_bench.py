#!/usr/bin/env python3
"""Summarize Rust vs original Python Qwen3-ASR CUDA benchmarks.

Usage:
    python scripts/summarize_bench.py target/rust_bench_cuda_v2.tsv target/python_bench_original_v2.tsv
"""

import sys
from pathlib import Path


def parse_line(line: str) -> dict | None:
    """Parse a TSV result line. Returns None for header or malformed lines."""
    line = line.strip()
    if not line or line.startswith("backend\t"):
        return None
    parts = line.split("\t")
    if len(parts) < 11:
        return None
    backend, model, wav, elapsed_s, rtfx, peak_used_field, baseline_field, after_load_field, after_transcribe_field, lang, text = parts[:11]

    def _extract_miB(field: str) -> float | None:
        if field == "n/a":
            return None
        # e.g. "peak_used=1913.0MiB" or "1913.0MiB"
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
        "peak_used_mib": _extract_miB(peak_used_field),
        "baseline_mib": _extract_miB(baseline_field),
        "after_load_mib": _extract_miB(after_load_field),
        "after_transcribe_mib": _extract_miB(after_transcribe_field),
        "lang": lang,
    }


def load_results(path: Path) -> dict:
    results = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = parse_line(line)
            if r is None:
                continue
            key = (r["model"], r["wav"])
            results[key] = r
    return results


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <rust.tsv> <python.tsv>", file=sys.stderr)
        sys.exit(1)

    rust_path = Path(sys.argv[1])
    py_path = Path(sys.argv[2])

    rust = load_results(rust_path)
    py = load_results(py_path)

    keys = sorted(set(rust.keys()) | set(py.keys()), key=lambda k: (k[0], k[1]))

    print("# Qwen3-ASR CUDA Benchmark Summary")
    print()
    print("- Device: NVIDIA P104-100 (8 GiB)")
    print("- Rust: `cargo test --release --test bench_cuda -- --ignored --test-threads=1`")
    print("- Python: `conda activate asr`, qwen-asr transformers backend")
    print()
    print("| Model | Fixture | Rust elapsed | Rust RTFx | Rust peak VRAM | Python elapsed | Python RTFx | Python peak VRAM | Speedup (Py/Rust) |")
    print("|-------|---------|--------------|-----------|----------------|----------------|-------------|------------------|-------------------|")

    for key in keys:
        model, wav = key
        r = rust.get(key, {})
        p = py.get(key, {})

        r_elapsed = r.get("elapsed_s")
        p_elapsed = p.get("elapsed_s")
        speedup = p_elapsed / r_elapsed if r_elapsed and p_elapsed else None

        def fmt(val, unit=""):
            if val is None:
                return "—"
            if isinstance(val, float):
                return f"{val:.2f}{unit}"
            return f"{val}{unit}"

        print(
            f"| {model} | {wav} | "
            f"{fmt(r_elapsed, 's')} | {fmt(r.get('rtfx'), 'x')} | {fmt(r.get('peak_used_mib'), ' MiB')} | "
            f"{fmt(p_elapsed, 's')} | {fmt(p.get('rtfx'), 'x')} | {fmt(p.get('peak_used_mib'), ' MiB')} | "
            f"{fmt(speedup, 'x')} |"
        )

    print()
    print("Notes:")
    print("- 'peak VRAM' is the maximum observed GPU memory usage during load + inference.")
    print("- 'Speedup' > 1 means Rust is faster; < 1 means Python is faster.")
    print("- Python peak VRAM excludes any memory held by a previous test because each config runs in a fresh subprocess.")


if __name__ == "__main__":
    main()
