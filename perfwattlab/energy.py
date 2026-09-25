"""
energy.py — Energy measurement v2 (README §10.3).

EnergyMeter reads NVML's cumulative hardware energy counter
(nvmlDeviceGetTotalEnergyConsumption, millijoules since driver load,
Volta and newer — T4 is supported). Energy for a window is simply
counter(end) - counter(start): no sampling-rate quantization.

If the counter is unavailable, it falls back to integrating a PowerSampler
(trapezoid), and every result records which method was used.

Idle subtraction: measure_idle_power() records the board's idle draw before
a run. Results report BOTH total energy and active energy
(total - idle_power * duration), clearly labeled. Active energy is an
estimate: idle power is not constant while the GPU is busy (clocks rise),
so treat it as "energy above the idle floor", not "energy of the model".

Counter caveat: the counter's update granularity is driver-dependent. Short
windows (well under ~1 s) should be aggregated — measure steady-state
windows or whole batches of requests, then divide by tokens.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .power import NVML_AVAILABLE, PowerSampler, integrate_energy_j, nv


@dataclass
class EnergyReading:
    duration_s: float
    energy_j: float
    method: str                        # "counter" | "sampler" | "none"
    idle_power_w: Optional[float] = None
    extra: dict = field(default_factory=dict)

    @property
    def avg_power_w(self) -> float:
        return self.energy_j / self.duration_s if self.duration_s > 0 else float("nan")

    @property
    def active_energy_j(self) -> Optional[float]:
        if self.idle_power_w is None:
            return None
        return self.energy_j - self.idle_power_w * self.duration_s

    def as_dict(self, prefix: str = "") -> dict:
        d = {
            f"{prefix}duration_s": round(self.duration_s, 6),
            f"{prefix}energy_j": round(self.energy_j, 4),
            f"{prefix}avg_power_w": round(self.avg_power_w, 3),
            f"{prefix}energy_method": self.method,
        }
        if self.idle_power_w is not None:
            d[f"{prefix}idle_power_w"] = round(self.idle_power_w, 3)
            d[f"{prefix}active_energy_j"] = round(self.active_energy_j, 4)
        d.update({f"{prefix}{k}": v for k, v in self.extra.items()})
        return d


class EnergyMeter:
    """
        meter = EnergyMeter()          # starts NVML + background sampler
        meter.measure_idle_power(5)    # optional
        tok = meter.begin()
        ... workload ...
        reading = meter.end(tok)       # EnergyReading
        meter.close()
    """

    def __init__(self, device_index: int = 0, sampler_hz: float = 10.0,
                 use_sampler: bool = True):
        self.device_index = device_index
        self.idle_power_w: Optional[float] = None
        self.counter_ok = False
        self.handle = None
        if NVML_AVAILABLE:
            try:
                nv.nvmlInit()
                self.handle = nv.nvmlDeviceGetHandleByIndex(device_index)
                nv.nvmlDeviceGetTotalEnergyConsumption(self.handle)
                self.counter_ok = True
            except Exception:
                self.counter_ok = False
        self.sampler = PowerSampler(hz=sampler_hz, device_index=device_index) if use_sampler else None
        if self.sampler:
            self.sampler.start()

    @property
    def method(self) -> str:
        if self.counter_ok:
            return "counter"
        if self.sampler and self.sampler.enabled:
            return "sampler"
        return "none"

    def _counter_mj(self) -> Optional[int]:
        if not self.counter_ok:
            return None
        return nv.nvmlDeviceGetTotalEnergyConsumption(self.handle)

    def begin(self) -> dict:
        return {"t": time.perf_counter(),
                "t_rel": self.sampler.now() if self.sampler else None,
                "mj": self._counter_mj()}

    def end(self, tok: dict) -> EnergyReading:
        t1 = time.perf_counter()
        mj1 = self._counter_mj()
        dur = t1 - tok["t"]
        extra = {}
        t_rel1 = self.sampler.now() if self.sampler else None
        if self.sampler and self.sampler.enabled:
            extra.update(self.sampler.window_stats(tok["t_rel"], t_rel1))
        if mj1 is not None and tok["mj"] is not None:
            e = (mj1 - tok["mj"]) / 1000.0
            method = "counter"
        elif self.sampler and self.sampler.enabled:
            e = integrate_energy_j(self.sampler.to_df(), tok["t_rel"], t_rel1)
            method = "sampler"
        else:
            e, method = float("nan"), "none"
        return EnergyReading(dur, e, method, self.idle_power_w, extra)

    def measure_idle_power(self, seconds: float = 5.0) -> Optional[float]:
        """Board power with no work queued. Call after torch.cuda.synchronize()."""
        tok = self.begin()
        time.sleep(seconds)
        r = self.end(tok)
        if r.method == "none":
            return None
        self.idle_power_w = r.avg_power_w
        print(f"[EnergyMeter] idle power ≈ {self.idle_power_w:.2f} W over {seconds:.0f}s ({r.method})")
        return self.idle_power_w

    def close(self):
        if self.sampler:
            self.sampler.stop()
        if NVML_AVAILABLE and self.handle is not None:
            try:
                nv.nvmlShutdown()
            except Exception:
                pass
