"""
profiler.py — torch.profiler capture plus quantitative trace analysis.

v1 of this file claimed that aten::copy_ / cudaDeviceSynchronize "dominate"
the pipeline path. That was never verified. v2 doesn't assert a cause; it
extracts numbers from the trace that can support or refute one:

  * GPU busy fraction — union of CUDA kernel intervals / profiled window.
    Low busy fraction at batch 1 means the GPU waits on the host.
  * Idle gaps between consecutive kernels (count, p50, p99, total).
  * CUDA kernels launched per generated token.
  * CPU-side op counts (aten::* ops, cudaLaunchKernel, cudaDeviceSynchronize,
    cudaStreamSynchronize, cudaMemcpy*) per generated token.
  * Top kernels by device time.

Compare these across generation paths (Experiment 0 H1–H3) and across batch
sizes (Experiment 1) instead of eyeballing timelines.

For a system-level view (host threads, NVTX ranges "prefill"/"decode_step",
CUDA API), also run Nsight Systems — see README §14.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

SYNC_APIS = ("cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize")


def analyze_chrome_trace(trace_path: str, n_tokens: int) -> dict:
    """Quantify GPU busy time, gaps, kernel and API counts from a Chrome trace."""
    with open(trace_path) as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) else data
    kernels, cpu_ops, runtime = [], Counter(), Counter()
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = (e.get("cat") or "").lower()
        name = e.get("name", "")
        if cat == "kernel":
            kernels.append((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0)), name))
        elif cat in ("cuda_runtime", "cuda_driver"):
            runtime[name] += 1
        elif cat == "cpu_op":
            cpu_ops[name] += 1
    if not kernels:
        return {"n_kernels": 0, "note": "no CUDA kernels in trace (CPU run?)"}

    kernels.sort()
    start, end = kernels[0][0], max(k[1] for k in kernels)
    busy, gaps, cur_s, cur_e = 0.0, [], kernels[0][0], kernels[0][1]
    for s, e, _ in kernels[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            busy += cur_e - cur_s
            gaps.append(s - cur_e)
            cur_s, cur_e = s, e
    busy += cur_e - cur_s
    window = max(end - start, 1e-9)
    gaps = np.asarray(gaps) if gaps else np.zeros(1)
    by_kernel = Counter()
    for s, e, n in kernels:
        by_kernel[n] += e - s
    top = [{"kernel": n[:90], "total_us": round(t, 1), "share": round(t / max(busy, 1e-9), 4)}
           for n, t in by_kernel.most_common(10)]
    n_tok = max(n_tokens, 1)
    return {
        "window_ms": round(window / 1000.0, 3),
        "gpu_busy_ms": round(busy / 1000.0, 3),
        "gpu_busy_fraction": round(busy / window, 4),
        "n_kernels": len(kernels),
        "kernels_per_token": round(len(kernels) / n_tok, 2),
        "n_gaps": int((gaps > 0).sum()),
        "gap_p50_us": round(float(np.percentile(gaps, 50)), 2),
        "gap_p99_us": round(float(np.percentile(gaps, 99)), 2),
        "gap_total_ms": round(float(gaps.sum()) / 1000.0, 3),
        "cpu_ops_per_token": round(sum(cpu_ops.values()) / n_tok, 1),
        "aten_copy_per_token": round(cpu_ops.get("aten::copy_", 0) / n_tok, 2),
        "launches_per_token": round(runtime.get("cudaLaunchKernel", 0) / n_tok, 2),
        "sync_calls": {k: runtime.get(k, 0) for k in SYNC_APIS},
        "memcpy_calls": {k: v for k, v in runtime.items() if "Memcpy" in k},
        "top_kernels": top,
    }


def profile_callable(fn: Callable[[], int], trace_dir: Path, label: str,
                     warmup: int = 2, record_shapes: bool = False) -> dict:
    """
    Profile one call of fn() after `warmup` unprofiled calls.
    fn must return the number of generated tokens.

    profile_memory / with_stack are off: they add host overhead that would
    distort exactly the host-side cost we are trying to measure.
    """
    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    with profile(activities=acts, record_shapes=record_shapes) as prof:
        t0 = time.perf_counter()
        n_tokens = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000.0
    trace_path = str(trace_dir / f"{label}.json")
    prof.export_chrome_trace(trace_path)
    sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_key, row_limit=20)
    (trace_dir / f"{label}_key_averages.txt").write_text(table)
    stats = analyze_chrome_trace(trace_path, n_tokens)
    stats.update({"label": label, "wall_ms_profiled": round(wall_ms, 2), "n_tokens": n_tokens,
                  "trace_path": trace_path})
    return stats


def compare_paths(fns: Dict[str, Callable[[], int]], trace_dir: Path, warmup: int = 2) -> dict:
    """Profile several generation paths with the same prompt and token count."""
    out = {}
    for label, fn in fns.items():
        print(f"Profiling {label} ...")
        out[label] = profile_callable(fn, trace_dir, label, warmup=warmup)
        s = out[label]
        print(f"  wall {s['wall_ms_profiled']:.0f} ms | GPU busy {s.get('gpu_busy_fraction', 'n/a')} | "
              f"kernels/token {s.get('kernels_per_token', 'n/a')} | "
              f"cpu ops/token {s.get('cpu_ops_per_token', 'n/a')}")
    (Path(trace_dir) / "trace_comparison.json").write_text(json.dumps(out, indent=2))
    return out
