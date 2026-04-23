"""
How to read the output:
  - Look at cuda_time_total in the key_averages table.
  - If you see aten::copy_ or cudaDeviceSynchronize dominating, that's the
    CPU-GPU transfer overhead introduced by the pipeline wrapper.
  - After switching to generate_direct(), those entries shrink significantly
    and the GPU execution lane in the trace becomes more continuous.
"""

import time
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)


def profile_generation(
    generate_fn: Callable,
    prompt: str,
    trace_dir: Path,
    label: str = "trace",
    max_new_tokens: int = 96,
    warmup_steps: int = 1,
    active_steps: int = 2,
) -> dict:
    """
    Profile generate_fn under torch.profiler and export a Chrome trace.

    Args:
        generate_fn: callable(prompt, max_new_tokens) -> (text, latency_ms, tokens, tps)
        prompt:       input prompt string
        trace_dir:    directory to write trace files
        label:        filename prefix for the trace
        max_new_tokens: tokens to generate per call
        warmup_steps: profiler warmup (not recorded)
        active_steps: profiler active recording steps

    Returns:
        dict with p50 latency, top kernel table string, and trace path
    """
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = str(trace_dir / f"{label}.json")

    # Warmup — stabilize caches, compilation, memory allocation
    print(f"Warming up for {warmup_steps} step(s)...")
    for _ in range(warmup_steps):
        generate_fn(prompt, max_new_tokens=32)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    latencies = []
    total_steps = warmup_steps + active_steps

    prof_schedule = schedule(wait=0, warmup=warmup_steps, active=active_steps)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for step in range(total_steps):
            t0 = time.perf_counter()
            generate_fn(prompt, max_new_tokens=max_new_tokens)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            prof.step()

    prof.export_chrome_trace(trace_path)
    print(f"Chrome trace saved to: {trace_path}")
    print(f"Open in chrome://tracing or https://ui.perfetto.dev\n")

    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)
    print(table)

    return {
        "label": label,
        "trace_path": trace_path,
        "p50_latency_ms": round(sorted(latencies)[len(latencies) // 2], 2),
        "kernel_table": table,
    }


def compare_profiles(
    before_fn: Callable,
    after_fn: Callable,
    prompt: str,
    trace_dir: Path,
    max_new_tokens: int = 96,
) -> dict:
    """
    Profile two generation functions and print a side-by-side comparison.
    Used to validate that the pipeline -> direct switch actually helped.
    """
    print("=" * 60)
    print("Profiling BEFORE (pipeline path)")
    print("=" * 60)
    before = profile_generation(
        before_fn, prompt, trace_dir,
        label="before_pipeline",
        max_new_tokens=max_new_tokens,
    )

    print("\n" + "=" * 60)
    print("Profiling AFTER (direct path)")
    print("=" * 60)
    after = profile_generation(
        after_fn, prompt, trace_dir,
        label="after_direct",
        max_new_tokens=max_new_tokens,
    )

    delta_ms = before["p50_latency_ms"] - after["p50_latency_ms"]
    pct = delta_ms / before["p50_latency_ms"] * 100 if before["p50_latency_ms"] > 0 else 0

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  Before p50 latency:  {before['p50_latency_ms']} ms")
    print(f"  After  p50 latency:  {after['p50_latency_ms']} ms")
    print(f"  Improvement:         {delta_ms:.1f} ms  ({pct:.1f}%)")
    print(f"\n  Before trace: {before['trace_path']}")
    print(f"  After  trace: {after['trace_path']}")

    return {"before": before, "after": after, "delta_ms": round(delta_ms, 2), "pct_improvement": round(pct, 2)}
