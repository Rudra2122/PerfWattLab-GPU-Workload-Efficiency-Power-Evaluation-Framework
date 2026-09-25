"""
power.py — NVML board-power sampling and sample-based energy integration.

What this measures, precisely
-----------------------------
* Board-level GPU power (the whole card: SMs, memory, VRM losses, fan),
  polled from NVML by a background thread.
* At the default 5 Hz, one sample every 200 ms. That is adequate for coarse
  energy over windows of several seconds. It CANNOT attribute energy to
  kernels, prefill vs decode, or anything shorter than a few samples.
* NVML's reported power may itself be internally averaged depending on GPU
  and driver, so the effective resolution can be coarser than the poll rate.

For energy, prefer perfwattlab.energy.EnergyMeter, which reads NVML's
cumulative hardware energy counter (Volta and newer, including T4) and does
not depend on sampling. The sampler here is kept for power/utilization
timelines and as a fallback.

NVML "gpu_util_pct" is the fraction of time at least one kernel was running
during the last sample period — not SM occupancy and not achieved FLOPs.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import pynvml as nv  # provided by the `nvidia-ml-py` package
    NVML_AVAILABLE = True
except ImportError:
    nv = None
    NVML_AVAILABLE = False


class PowerSampler:
    """
    Background thread sampling board power, utilization, memory, SM clock
    and temperature at a fixed rate.

        sampler = PowerSampler(hz=5.0)
        sampler.start()
        ...
        sampler.stop()
        df = sampler.to_df()
    """

    def __init__(self, hz: float = 5.0, device_index: int = 0):
        self.hz = hz
        self.device_index = device_index
        self.samples: List[dict] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.t0: Optional[float] = None
        self.enabled = NVML_AVAILABLE

    def start(self):
        self.t0 = time.perf_counter()
        self.samples = []
        if not self.enabled:
            return
        try:
            nv.nvmlInit()
            self._handle = nv.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception as e:  # no GPU / no driver
            print(f"[PowerSampler] NVML unavailable ({e}); sampling disabled")
            self.enabled = False
            return
        self._stop.clear()

        def _loop():
            period = 1.0 / self.hz
            h = self._handle
            while not self._stop.is_set():
                t_rel = time.perf_counter() - self.t0
                try:
                    util = nv.nvmlDeviceGetUtilizationRates(h)
                    row = {
                        "t_rel_s": round(t_rel, 4),
                        "power_w": nv.nvmlDeviceGetPowerUsage(h) / 1000.0,
                        "gpu_util_pct": int(util.gpu),
                        "mem_util_pct": int(util.memory),
                        "mem_used_mb": round(nv.nvmlDeviceGetMemoryInfo(h).used / 2**20, 1),
                        "sm_clock_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
                        "temp_c": nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
                    }
                    with self._lock:
                        self.samples.append(row)
                except Exception:
                    pass
                time.sleep(period)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def now(self) -> float:
        """Current time relative to the sampler's t0."""
        return time.perf_counter() - self.t0

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self.enabled:
            try:
                nv.nvmlShutdown()
            except Exception:
                pass

    def to_df(self) -> pd.DataFrame:
        with self._lock:
            return pd.DataFrame(list(self.samples))

    def window_stats(self, t_start: float, t_end: float) -> dict:
        """Mean power/util/clock over samples inside [t_start, t_end]."""
        df = self.to_df()
        if df.empty:
            return {}
        w = df[(df.t_rel_s >= t_start) & (df.t_rel_s <= t_end)]
        if w.empty:
            return {}
        return {
            "mean_power_w": round(float(w.power_w.mean()), 2),
            "mean_gpu_util_pct": round(float(w.gpu_util_pct.mean()), 1),
            "mean_mem_util_pct": round(float(w.mem_util_pct.mean()), 1),
            "mean_sm_clock_mhz": round(float(w.sm_clock_mhz.mean()), 0),
            "max_temp_c": int(w.temp_c.max()),
            "n_power_samples": int(len(w)),
        }


def integrate_energy_j(power_df: pd.DataFrame, t_start: float, t_end: float,
                       method: str = "trapezoid") -> float:
    """
    Integrate sampled power over [t_start, t_end] (seconds, relative to the
    sampler's t0).

    method="trapezoid": linear interpolation between samples, with the window
        edges interpolated too.
    method="zoh": zero-order hold (each sample held until the next one).
        This is what v1 actually computed, although its docstring said
        "trapezoidal".

    With ~200 ms between samples, a 2 s window has ~10 samples, so the
    quantization error is material. Use EnergyMeter where possible.
    """
    if t_end <= t_start or power_df is None or power_df.empty:
        return 0.0
    df = power_df.sort_values("t_rel_s")
    t = df["t_rel_s"].to_numpy(dtype=float)
    p = df["power_w"].to_numpy(dtype=float)
    if len(t) < 2:
        return 0.0

    if method == "zoh":
        e = 0.0
        for i in range(len(t) - 1):
            a, b = max(t[i], t_start), min(t[i + 1], t_end)
            if b > a:
                e += p[i] * (b - a)
        return round(float(e), 4)

    # trapezoid with interpolated edges
    grid = np.concatenate(([t_start], t[(t > t_start) & (t < t_end)], [t_end]))
    pg = np.interp(grid, t, p)
    return round(float(np.sum((pg[1:] + pg[:-1]) * 0.5 * np.diff(grid))), 4)


def plot_pareto(summary_df: pd.DataFrame, out_path: Path, x: str = "p50_total_ms",
                y: str = "p50_energy_j", label: str = "config",
                xerr: tuple = None, yerr: tuple = None):
    """Latency vs energy scatter; optional (lo_col, hi_col) error bars."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    xs, ys = summary_df[x].to_numpy(), summary_df[y].to_numpy()
    xe = ye = None
    if xerr:
        xe = [xs - summary_df[xerr[0]].to_numpy(), summary_df[xerr[1]].to_numpy() - xs]
    if yerr:
        ye = [ys - summary_df[yerr[0]].to_numpy(), summary_df[yerr[1]].to_numpy() - ys]
    ax.errorbar(xs, ys, xerr=xe, yerr=ye, fmt="o", capsize=4)
    for xi, yi, lab in zip(xs, ys, summary_df[label]):
        ax.annotate(str(lab), (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title("Latency vs energy (95% CI)" if (xerr or yerr) else "Latency vs energy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")
