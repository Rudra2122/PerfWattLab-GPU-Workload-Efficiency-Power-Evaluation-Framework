"""
env.py — Record the hardware/software context of a run.

Every results directory gets an env.json so numbers are never separated from
the GPU, driver, clocks and library versions that produced them.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path


def _try(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def collect_env() -> dict:
    env = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_runtime"] = torch.version.cuda
            env["gpu_name"] = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            env["compute_capability"] = f"sm_{cap[0]}{cap[1]}"
            env["gpu_total_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
            env["bf16_supported"] = _try(torch.cuda.is_bf16_supported, False)
    except ImportError:
        pass
    for mod in ("transformers", "triton", "vllm", "pynvml"):
        env[mod] = _try(lambda m=mod: __import__(m).__version__, None)

    # Driver + clocks via NVML (optional)
    try:
        import pynvml as nv
        nv.nvmlInit()
        h = nv.nvmlDeviceGetHandleByIndex(0)
        env["driver"] = _try(lambda: _s(nv.nvmlSystemGetDriverVersion()))
        env["power_limit_w"] = _try(lambda: nv.nvmlDeviceGetPowerManagementLimit(h) / 1000.0)
        env["sm_clock_mhz"] = _try(lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM))
        env["mem_clock_mhz"] = _try(lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM))
        env["max_sm_clock_mhz"] = _try(lambda: nv.nvmlDeviceGetMaxClockInfo(h, nv.NVML_CLOCK_SM))
        env["temperature_c"] = _try(lambda: nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU))
        env["energy_counter_supported"] = _try(
            lambda: nv.nvmlDeviceGetTotalEnergyConsumption(h) is not None, False)
        nv.nvmlShutdown()
    except Exception:
        pass

    env["nvidia_smi"] = _try(lambda: subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm,clocks.mem,power.limit",
         "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip())
    return env


def _s(x):
    return x.decode() if isinstance(x, bytes) else x


def write_env(out_dir: Path, extra: dict | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = collect_env()
    if extra:
        env["run_config"] = extra
    (out_dir / "env.json").write_text(json.dumps(env, indent=2, default=str))
    return env
