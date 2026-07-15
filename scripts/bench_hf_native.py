#!/usr/bin/env python3
"""Benchmark Qwen3-ASR Transformers-native checkpoints (HF format) with GPU memory tracking.

Run inside the `asr` conda environment with a recent transformers (>=5.13) installed:
    conda activate asr
    python scripts/bench_hf_native.py [compile]

If the first positional argument is "compile", the model forward pass is wrapped with
`torch.compile` and warmed up before timing, so we can verify the official torch.compile
speed-up claim.

Uses local checkpoints under D:\asr\models and the same fixtures as the Rust project.
"""

import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parent.parent
ASR_MODEL_ROOT = Path("D:/asr/models")
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"

COMPILE = len(sys.argv) > 1 and sys.argv[1].lower() == "compile"
BACKEND_TAG = "HF_NATIVE_COMPILE" if COMPILE else "HF_NATIVE"


@dataclass
class BenchConfig:
    model_key: str
    hf_name: str
    wav: str
    duration_s: float
    max_new_tokens: int


CONFIGS = [
    BenchConfig("0.6B", "Qwen3-ASR-0.6B-hf", "15s.wav", 15.0, 512),
    BenchConfig("0.6B", "Qwen3-ASR-0.6B-hf", "30s.wav", 30.0, 512),
    BenchConfig("0.6B", "Qwen3-ASR-0.6B-hf", "90s.wav", 90.0, 1024),
    BenchConfig("0.6B", "Qwen3-ASR-0.6B-hf", "ja_89s.wav", 89.0, 1024),
    BenchConfig("0.6B", "Qwen3-ASR-0.6B-hf", "180s.wav", 180.0, 1024),
    BenchConfig("0.6B", "Qwen3-ASR-0.6B-hf", "180s_en.wav", 180.0, 1024),
    BenchConfig("1.7B", "Qwen3-ASR-1.7B-hf", "15s.wav", 15.0, 512),
    BenchConfig("1.7B", "Qwen3-ASR-1.7B-hf", "30s.wav", 30.0, 512),
    BenchConfig("1.7B", "Qwen3-ASR-1.7B-hf", "90s.wav", 90.0, 1024),
    BenchConfig("1.7B", "Qwen3-ASR-1.7B-hf", "ja_89s.wav", 89.0, 1024),
    BenchConfig("1.7B", "Qwen3-ASR-1.7B-hf", "180s.wav", 180.0, 1024),
    BenchConfig("1.7B", "Qwen3-ASR-1.7B-hf", "180s_en.wav", 180.0, 1024),
]


def get_gpu_memory_used_mib() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return None


def get_gpu_memory_total_mib() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return None


class GpuMemSampler:
    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_used_mib = 0.0

    def start(self):
        self._stop.clear()
        self.peak_used_mib = 0.0
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self):
        while not self._stop.is_set():
            used = get_gpu_memory_used_mib()
            if used is not None and used > self.peak_used_mib:
                self.peak_used_mib = used
            time.sleep(self.interval_s)
        used = get_gpu_memory_used_mib()
        if used is not None and used > self.peak_used_mib:
            self.peak_used_mib = used

    def stop(self) -> float:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return self.peak_used_mib


def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_bench(cfg: BenchConfig):
    model_path = str(ASR_MODEL_ROOT / cfg.hf_name)
    wav_path = str(FIXTURE_DIR / cfg.wav)

    if not Path(model_path).exists():
        print(f"{BACKEND_TAG}\t{cfg.model_key}\t{cfg.wav}\tSKIP\tmodel_missing={model_path}")
        return
    if not Path(wav_path).exists():
        print(f"{BACKEND_TAG}\t{cfg.model_key}\t{cfg.wav}\tSKIP\twav_missing={wav_path}")
        return

    baseline_used = get_gpu_memory_used_mib()
    total_mib = get_gpu_memory_total_mib()

    print(f"# Loading {cfg.model_key} from {model_path} ...", file=sys.stderr)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    if COMPILE:
        print(f"# Compiling {cfg.model_key} ...", file=sys.stderr)
        model.forward = torch.compile(model.forward)

    clear_gpu_cache()
    after_load_used = get_gpu_memory_used_mib()

    # Prepare inputs once; timing covers only generate().
    inputs = processor.apply_transcription_request(audio=wav_path).to(model.device, model.dtype)

    # Optional torch.compile warmup.
    if COMPILE:
        print(f"# Warming up {cfg.model_key} ...", file=sys.stderr)
        with torch.inference_mode():
            for _ in range(3):
                _ = model.generate(**inputs, max_new_tokens=cfg.max_new_tokens, do_sample=False)
        clear_gpu_cache()

    sampler = GpuMemSampler(interval_s=0.05)
    sampler.start()

    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=cfg.max_new_tokens, do_sample=False)
    elapsed = time.perf_counter() - t0

    peak_used = sampler.stop()
    clear_gpu_cache()
    after_transcribe_used = get_gpu_memory_used_mib()

    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    raw = processor.decode(generated_ids)[0]
    parsed = processor.decode(generated_ids, return_format="parsed")[0]
    text = parsed.get("transcription", raw)
    lang = parsed.get("language", "?")

    rtfx = cfg.duration_s / elapsed if cfg.duration_s > 0 else 0.0

    print(
        f"{BACKEND_TAG}\t{cfg.model_key}\t{cfg.wav}\t{elapsed:.3f}s\t{rtfx:.2f}x\t"
        f"peak_used={peak_used:.1f}MiB\t"
        f"baseline={baseline_used if baseline_used is not None else 'n/a'}MiB"
        f"/{total_mib if total_mib is not None else 'n/a'}MiB\t"
        f"after_load={after_load_used if after_load_used is not None else 'n/a'}MiB\t"
        f"after_transcribe={after_transcribe_used if after_transcribe_used is not None else 'n/a'}MiB\t"
        f"lang={lang}\ttext={text.replace(chr(9), ' ').replace(chr(10), ' ')}"
    )

    del model
    del processor
    clear_gpu_cache()


def main():
    # If invoked with a single-config args, run it directly. Otherwise spawn a
    # fresh subprocess per config so each measurement starts with a clean GPU.
    compile_flag = "compile" if COMPILE else ""
    if len(sys.argv) == 6 + (1 if COMPILE else 0):
        cfg = BenchConfig(
            model_key=sys.argv[1 + (1 if COMPILE else 0)],
            hf_name=sys.argv[2 + (1 if COMPILE else 0)],
            wav=sys.argv[3 + (1 if COMPILE else 0)],
            duration_s=float(sys.argv[4 + (1 if COMPILE else 0)]),
            max_new_tokens=int(sys.argv[5 + (1 if COMPILE else 0)]),
        )
        try:
            run_bench(cfg)
        except Exception as e:
            print(f"{BACKEND_TAG}\t{cfg.model_key}\t{cfg.wav}\tERROR\t{str(e).replace(chr(9), ' ')}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        return

    print(f"# fixture_dir={FIXTURE_DIR}", file=sys.stderr)
    print(f"# model_dir={ASR_MODEL_ROOT}", file=sys.stderr)
    print(f"# torch={torch.__version__} cuda_available={torch.cuda.is_available()}", file=sys.stderr)
    print(f"# compile={COMPILE}", file=sys.stderr)
    if torch.cuda.is_available():
        print(f"# device={torch.cuda.get_device_name(0)}", file=sys.stderr)

    header = "backend\tmodel\twav\telapsed\tRTFx\tpeak_used\tbaseline\tafter_load\tafter_transcribe\tlang\ttext"
    print(header)

    python = sys.executable
    script = Path(__file__).resolve()
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    for cfg in CONFIGS:
        cmd = [python, str(script)]
        if COMPILE:
            cmd.append("compile")
        cmd.extend([
            cfg.model_key, cfg.hf_name, cfg.wav,
            str(cfg.duration_s), str(cfg.max_new_tokens),
        ])
        try:
            subprocess.run(cmd, check=False, env=env)
        except Exception as e:
            print(f"{BACKEND_TAG}\t{cfg.model_key}\t{cfg.wav}\tERROR\t{str(e).replace(chr(9), ' ')}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
