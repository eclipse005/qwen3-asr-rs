#!/usr/bin/env python3
"""Benchmark original Qwen3-ASR (Python / transformers backend) with GPU memory tracking.

Run inside the `asr` conda environment:
    conda activate asr
    python scripts/bench_original.py

Uses the same Qwen3-ASR checkpoints as the Rust implementation
(D:\qwen3-asr-rs\models) and the same fixtures under D:\qwen3-asr-rs\tests\fixtures.
GPU memory is polled via nvidia-smi in a background thread so we capture the true
peak used by the process.
"""

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from qwen_asr import Qwen3ASRModel
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
ASR_MODEL_ROOT = Path("D:/qwen3-asr-rs/models")
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"

# Make sure qwen_asr can be imported when run from this directory.
sys.path.insert(0, str(REPO_ROOT))


@dataclass
class BenchConfig:
    model_key: str          # "0.6B" or "1.7B"
    hf_name: str            # local HF checkpoint directory name
    wav: str
    duration_s: float
    max_new_tokens: int


CONFIGS = [
    BenchConfig("0.6B", "Qwen3-ASR-0.6B", "15s.wav", 15.0, 512),
    BenchConfig("0.6B", "Qwen3-ASR-0.6B", "30s.wav", 30.0, 512),
    BenchConfig("0.6B", "Qwen3-ASR-0.6B", "90s.wav", 90.0, 1024),
    BenchConfig("0.6B", "Qwen3-ASR-0.6B", "ja_89s.wav", 89.0, 1024),
    BenchConfig("0.6B", "Qwen3-ASR-0.6B", "180s.wav", 180.0, 1024),
    BenchConfig("0.6B", "Qwen3-ASR-0.6B", "180s_en.wav", 180.0, 1024),
    BenchConfig("1.7B", "Qwen3-ASR-1.7B", "15s.wav", 15.0, 512),
    BenchConfig("1.7B", "Qwen3-ASR-1.7B", "30s.wav", 30.0, 512),
    BenchConfig("1.7B", "Qwen3-ASR-1.7B", "90s.wav", 90.0, 1024),
    BenchConfig("1.7B", "Qwen3-ASR-1.7B", "ja_89s.wav", 89.0, 1024),
    BenchConfig("1.7B", "Qwen3-ASR-1.7B", "180s.wav", 180.0, 1024),
    BenchConfig("1.7B", "Qwen3-ASR-1.7B", "180s_en.wav", 180.0, 1024),
]


def get_gpu_memory_used_mib() -> float | None:
    """Return total GPU memory used (MiB) via nvidia-smi."""
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
        # Final sample to catch peaks right at workload end.
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
        print(f"PYTHON_ORIGINAL\t{cfg.model_key}\t{cfg.wav}\tSKIP\tmodel_missing={model_path}")
        return
    if not Path(wav_path).exists():
        print(f"PYTHON_ORIGINAL\t{cfg.model_key}\t{cfg.wav}\tSKIP\twav_missing={wav_path}")
        return

    baseline_used = get_gpu_memory_used_mib()
    total_mib = get_gpu_memory_total_mib()

    print(f"# Loading {cfg.model_key} from {model_path} ...", file=sys.stderr)

    # Load the qwen-asr config object in memory so we never modify the model files.
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    model_config = Qwen3ASRConfig.from_dict(config_dict)

    model = Qwen3ASRModel.from_pretrained(
        model_path,
        config=model_config,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=1,
        max_new_tokens=cfg.max_new_tokens,
    )
    clear_gpu_cache()
    after_load_used = get_gpu_memory_used_mib()

    sampler = GpuMemSampler(interval_s=0.05)
    sampler.start()

    t0 = time.perf_counter()
    results = model.transcribe(
        audio=wav_path,
        language=None,
        return_time_stamps=False,
    )
    elapsed = time.perf_counter() - t0

    peak_used = sampler.stop()
    clear_gpu_cache()
    after_transcribe_used = get_gpu_memory_used_mib()

    result = results[0] if isinstance(results, list) and results else results
    text = result.text if hasattr(result, "text") else str(result)
    lang = result.language if hasattr(result, "language") else "?"

    rtfx = cfg.duration_s / elapsed if cfg.duration_s > 0 else 0.0

    print(
        f"PYTHON_ORIGINAL\t{cfg.model_key}\t{cfg.wav}\t{elapsed:.3f}s\t{rtfx:.2f}x\t"
        f"peak_used={peak_used:.1f}MiB\t"
        f"baseline={baseline_used if baseline_used is not None else 'n/a'}MiB"
        f"/{total_mib if total_mib is not None else 'n/a'}MiB\t"
        f"after_load={after_load_used if after_load_used is not None else 'n/a'}MiB\t"
        f"after_transcribe={after_transcribe_used if after_transcribe_used is not None else 'n/a'}MiB\t"
        f"lang={lang}\ttext={text.replace(chr(9), ' ').replace(chr(10), ' ')}"
    )

    # Free model before next bench to avoid OOM on 8GB cards.
    del model
    clear_gpu_cache()


def main():
    # If invoked with a single-config args, run it directly. Otherwise spawn a
    # fresh subprocess per config so each measurement starts with a clean GPU.
    if len(sys.argv) == 6:
        cfg = BenchConfig(
            model_key=sys.argv[1],
            hf_name=sys.argv[2],
            wav=sys.argv[3],
            duration_s=float(sys.argv[4]),
            max_new_tokens=int(sys.argv[5]),
        )
        try:
            run_bench(cfg)
        except Exception as e:
            print(f"PYTHON_ORIGINAL\t{cfg.model_key}\t{cfg.wav}\tERROR\t{str(e).replace(chr(9), ' ')}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        return

    print(f"# fixture_dir={FIXTURE_DIR}", file=sys.stderr)
    print(f"# model_dir={ASR_MODEL_ROOT}", file=sys.stderr)
    print(f"# torch={torch.__version__} cuda_available={torch.cuda.is_available()}", file=sys.stderr)
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
        cmd = [
            python, str(script),
            cfg.model_key, cfg.hf_name, cfg.wav,
            str(cfg.duration_s), str(cfg.max_new_tokens),
        ]
        try:
            subprocess.run(cmd, check=False, env=env)
        except Exception as e:
            print(f"PYTHON_ORIGINAL\t{cfg.model_key}\t{cfg.wav}\tERROR\t{str(e).replace(chr(9), ' ')}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
