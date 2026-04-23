"""
power.py — GPU board power sampling via NVML and energy-per-query computation.

Key insight from this project:
Two configurations with nearly identical p50 latency had meaningfully different
energy profiles. Latency alone does not tell the full story of serving efficiency.
The Pareto curve (latency vs energy) makes this tradeoff visible.

Limitations (documented honestly):
  - NVML reports GPU board power, not wall power. Does not include CPU or DRAM.
  - Sampling at 5 Hz can miss short spikes — energy values are approximations.
  - In shared environments (Colab, cloud VMs), DVFS and background load affect readings.
  - Timing alignment is host-timestamp-based, so there is some jitter.
"""

import threading
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# NVML availability check
# ---------------------------------------------------------------------------

try:
    from pynvml import (
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetName,
        nvmlDeviceGetPowerUsage,
        nvmlDeviceGetUtilizationRates,
        nvmlInit,
        nvmlShutdown,
    )
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("WARNING: pynvml not installed. Power measurement will be skipped.")
    print("Install with: pip install nvidia-ml-py3")


# ---------------------------------------------------------------------------
# Power sampler
# ---------------------------------------------------------------------------

class PowerSampler:
    """
    Background thread that samples GPU board power via NVML at a fixed rate.

    Usage:
        sampler = PowerSampler(hz=5.0)
        sampler.start()
        # ... run inference ...
        sampler.stop()
        df = sampler.to_df()
    """

    def __init__(self, hz: float = 5.0, device_index: int = 0):
        self.hz = hz
        self.device_index = device_index
        self.samples: List[dict] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.t0: Optional[float] = None

    def start(self):
        if not NVML_AVAILABLE:
            return
        nvmlInit()
        self._handle = nvmlDeviceGetHandleByIndex(self.device_index)
        self.gpu_name = nvmlDeviceGetName(self._handle)
        self.samples = []
        self._stop.clear()
        self.t0 = time.perf_counter()

        def _loop():
            period = 1.0 / self.hz
            while not self._stop.is_set():
                t_rel = time.perf_counter() - self.t0
                power_mw = nvmlDeviceGetPowerUsage(self._handle)
                util = nvmlDeviceGetUtilizationRates(self._handle)
                mem = nvmlDeviceGetMemoryInfo(self._handle)
                self.samples.append({
                    "t_rel_s":      round(t_rel, 4),
                    "power_w":      power_mw / 1000.0,
                    "gpu_util_pct": int(util.gpu),
                    "mem_util_pct": int(util.memory),
                    "mem_used_mb":  round(mem.used / (1024 ** 2), 1),
                })
                time.sleep(period)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if NVML_AVAILABLE:
            nvmlShutdown()

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.samples)


# ---------------------------------------------------------------------------
# Energy integration
# ---------------------------------------------------------------------------

def integrate_energy_j(power_df: pd.DataFrame, t_start: float, t_end: float) -> float:
    """
    Integrate power samples over [t_start, t_end] using the trapezoidal rule.
    t_start and t_end are relative to the sampler's t0.
    """
    if t_end <= t_start or power_df.empty:
        return 0.0

    df = power_df[
        (power_df["t_rel_s"] >= t_start - 1.0) &
        (power_df["t_rel_s"] <= t_end + 1.0)
    ].sort_values("t_rel_s").reset_index(drop=True)

    if len(df) < 2:
        return 0.0

    t = df["t_rel_s"].to_numpy()
    p = df["power_w"].to_numpy()

    energy = 0.0
    for i in range(len(df) - 1):
        seg_t0 = max(t[i], t_start)
        seg_t1 = min(t[i + 1], t_end)
        if seg_t1 > seg_t0:
            energy += float(p[i]) * (seg_t1 - seg_t0)
    return round(energy, 4)


# ---------------------------------------------------------------------------
# Instrumented run
# ---------------------------------------------------------------------------

def run_with_power(rag_fn, queries: list, sampler: PowerSampler,
                   config_name: str = "config") -> pd.DataFrame:
    """
    Run rag_fn(query) for each query while power is sampled in the background.
    Annotates each result row with energy_j and avg_power_w for that query.
    """
    rows = []

    for q in queries:
        t_start = time.perf_counter() - sampler.t0
        row = rag_fn(q)
        t_end = time.perf_counter() - sampler.t0

        power_df = sampler.to_df()
        energy_j = integrate_energy_j(power_df, t_start, t_end)
        duration_s = max(t_end - t_start, 1e-9)

        row["config"] = config_name
        row["t_start_rel_s"] = round(t_start, 4)
        row["t_end_rel_s"] = round(t_end, 4)
        row["energy_j"] = energy_j
        row["avg_power_w"] = round(energy_j / duration_s, 2)
        row["energy_j_per_token"] = round(energy_j / max(row.get("gen_tokens", 1), 1), 4)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary and Pareto
# ---------------------------------------------------------------------------

def summarize_power_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Compute p50/p95 latency and energy per query per config."""
    summaries = []
    for cfg, df in runs_df.groupby("config"):
        summaries.append({
            "config":               cfg,
            "n_queries":            len(df),
            "p50_total_ms":         round(float(np.percentile(df["total_ms"], 50)), 2),
            "p95_total_ms":         round(float(np.percentile(df["total_ms"], 95)), 2),
            "p50_toks_per_sec":     round(float(np.percentile(df["toks_per_sec"], 50)), 2),
            "mean_power_w":         round(float(df["avg_power_w"].mean()), 2),
            "p50_energy_j":         round(float(np.percentile(df["energy_j"], 50)), 2),
            "p95_energy_j":         round(float(np.percentile(df["energy_j"], 95)), 2),
            "p50_energy_j_per_tok": round(float(np.percentile(df["energy_j_per_token"], 50)), 4),
        })
    return pd.DataFrame(summaries)


def plot_pareto(summary_df: pd.DataFrame, out_path: Path):
    """Latency vs energy per query Pareto curve."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping Pareto plot")
        return

    x = summary_df["p50_total_ms"].to_numpy()
    y = summary_df["p50_energy_j"].to_numpy()
    labels = summary_df["config"].tolist()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=80, zorder=5)
    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=9)

    ax.set_xlabel("p50 latency (ms)")
    ax.set_ylabel("p50 energy per query (J)")
    ax.set_title("Pareto: Latency vs Energy per Query")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved Pareto plot: {out_path}")
    plt.close(fig)
