"""
loadgen.py — Open-loop workload generation (README §7.2).

Arrivals are a Poisson process at a fixed rate and are generated up front, so
the arrival schedule never depends on how fast the server responds (that
dependency is what closed-loop / semaphore-gated clients get wrong).

Every policy is run on an identical, deep-copied request list.
"""

from __future__ import annotations

import copy
from typing import List

import numpy as np

from .scheduler import Request

LENGTH_MIXES = {
    # name: (prompt_low, prompt_high, out_low, out_high)  — uniform
    "short":  (64, 256, 32, 128),
    "long":   (512, 1536, 128, 256),
    # heavy-tailed: lognormal prompt lengths, mostly short with a long tail
    "heavy_tail": None,
}


def make_workload(n: int, rate_rps: float, mix: str, vocab_size: int, max_positions: int,
                  bos_id: int = None, seed: int = 0) -> List[Request]:
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(1.0 / rate_rps, size=n)
    arrivals = np.cumsum(gaps) - gaps[0]          # first request at t=0
    reqs = []
    for i in range(n):
        if mix == "heavy_tail":
            p = int(np.clip(rng.lognormal(mean=5.3, sigma=0.8), 32, 1800))
            o = int(np.clip(rng.lognormal(mean=4.5, sigma=0.6), 16, 256))
        else:
            pl, ph, ol, oh = LENGTH_MIXES[mix]
            p, o = int(rng.integers(pl, ph + 1)), int(rng.integers(ol, oh + 1))
        if p + o > max_positions:
            p = max(8, max_positions - o)
        ids = rng.integers(100, vocab_size, size=p).tolist()
        if bos_id is not None:
            ids[0] = bos_id
        reqs.append(Request(rid=i, prompt_ids=ids, max_new_tokens=o, arrival_s=float(arrivals[i])))
    return reqs


def fresh(requests: List[Request]) -> List[Request]:
    """Deep copy with all runtime state cleared."""
    out = []
    for r in requests:
        c = copy.copy(r)
        c.prompt_ids = list(r.prompt_ids)
        c.output_ids, c.token_times = [], []
        c.admit_s = c.finish_s = None
        c.preemptions = 0
        c.prefilled = 0
        out.append(c)
    return out
