"""
stats.py — Statistics used for every headline number in PerfWattLab.

Rules (see README §3):
  * Report medians with bootstrap 95% confidence intervals.
  * Compare configs as a ratio (B / A). A ratio CI that contains 1.0 is
    reported as "no significant change".
  * When configs ran on the same prompts, use the paired per-prompt
    geometric-mean ratio — it removes prompt-mix effects (the bug that
    inflated v1's pooled-p50 result).
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def percentile(xs: Iterable[float], p: float) -> float:
    arr = np.asarray(list(xs), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, p))


def bootstrap_ci(xs: Iterable[float], stat=np.median, n_boot: int = 10_000,
                 alpha: float = 0.05, seed: int = 0) -> tuple:
    """(point, lo, hi) bootstrap CI of `stat` over xs."""
    arr = np.asarray(list(xs), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boots = stat(arr[idx], axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(arr)), float(lo), float(hi)


def ratio_ci(a: Iterable[float], b: Iterable[float], stat=np.median,
             n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple:
    """
    Bootstrap CI of stat(b) / stat(a) with independent resampling.
    Returns (ratio, lo, hi). ratio < 1 means b is smaller than a.
    """
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    # ratios need strictly positive values (e.g. an energy reading of 0 J on a
    # run shorter than the counter's update interval)
    if a.size == 0 or b.size == 0 or (a <= 0).any() or (b <= 0).any():
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, a.size, size=(n_boot, a.size))
    ib = rng.integers(0, b.size, size=(n_boot, b.size))
    boots = stat(b[ib], axis=1) / stat(a[ia], axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(b) / stat(a)), float(lo), float(hi)


def paired_geomean_ratio(df: pd.DataFrame, metric: str, config_col: str,
                         base: str, other: str, pair_col: str = "prompt_id",
                         n_boot: int = 10_000, seed: int = 0) -> tuple:
    """
    Per-pair (e.g. per-prompt) median of `metric` for each config, then the
    geometric mean of other/base across pairs, with a bootstrap CI over pairs.
    Returns (geomean_ratio, lo, hi, per_pair_ratios: dict).
    """
    med = (df[df[config_col].isin([base, other])]
           .groupby([pair_col, config_col])[metric].median().unstack())
    med = med.dropna(subset=[base, other])
    med = med[(med[base] > 0) & (med[other] > 0)]
    if med.empty:
        return float("nan"), float("nan"), float("nan"), {}
    logr = np.log(med[other].to_numpy() / med[base].to_numpy())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, logr.size, size=(n_boot, logr.size))
    boots = np.exp(logr[idx].mean(axis=1))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    per_pair = dict(zip(med.index.tolist(), np.exp(logr).round(4).tolist()))
    return float(np.exp(logr.mean())), float(lo), float(hi), per_pair


def verdict(lo: float, hi: float) -> str:
    """Human-readable significance verdict for a ratio CI."""
    if any(math.isnan(x) for x in (lo, hi)):
        return "insufficient data"
    if hi < 1.0:
        return "significant decrease"
    if lo > 1.0:
        return "significant increase"
    return "no significant change (CI contains 1.0)"


def fmt_ratio(r: float, lo: float, hi: float) -> str:
    return f"{(r - 1) * 100:+.1f}%  (ratio {r:.3f}, 95% CI {lo:.3f}–{hi:.3f}) → {verdict(lo, hi)}"


def compare_configs(df: pd.DataFrame, metrics: Dict[str, str], config_col: str,
                    base: str, other: str, pair_col: Optional[str] = "prompt_id") -> pd.DataFrame:
    """
    Build a comparison table for several metrics.
    metrics: {column_name: pretty_label}
    """
    rows = []
    a_df = df[df[config_col] == base]
    b_df = df[df[config_col] == other]
    for col, label in metrics.items():
        if col not in df.columns:
            continue
        r, lo, hi = ratio_ci(a_df[col], b_df[col])
        row = {
            "metric": label,
            f"median_{base}": round(float(a_df[col].median()), 4),
            f"median_{other}": round(float(b_df[col].median()), 4),
            "pooled_ratio": round(r, 4), "pooled_lo": round(lo, 4), "pooled_hi": round(hi, 4),
            "pooled_verdict": verdict(lo, hi),
        }
        if pair_col and pair_col in df.columns:
            g, glo, ghi, _ = paired_geomean_ratio(df, col, config_col, base, other, pair_col)
            row.update({"paired_ratio": round(g, 4), "paired_lo": round(glo, 4),
                        "paired_hi": round(ghi, 4), "paired_verdict": verdict(glo, ghi)})
        rows.append(row)
    return pd.DataFrame(rows)
