"""
sweep.py — Config sweep and benchmark runner.

Runs p50/p95 latency, throughput and tokens/sec across token lengths and
sampling modes (single request at a time). The fixed-rate function at the
bottom is a serialized-server queueing test, not a batching benchmark.
"""

import csv
import json
import math
import time
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------

def percentile(xs: list, p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


# ---------------------------------------------------------------------------
# Single-threaded benchmark
# ---------------------------------------------------------------------------

def run_benchmark(rag_fn, queries: list, label: str = "benchmark") -> dict:
    """
    Run rag_fn(query) for each query and return p50/p95 summary.
    rag_fn must return a dict with keys: total_ms, generation_ms, toks_per_sec,
    retrieval_ms, rerank_ms.
    """
    rows = [rag_fn(q) for q in queries]

    total_ms   = [r["total_ms"] for r in rows]
    gen_ms     = [r["generation_ms"] for r in rows]
    tps        = [r["toks_per_sec"] for r in rows]
    retr_ms    = [r["retrieval_ms"] for r in rows]
    rer_ms     = [r["rerank_ms"] for r in rows]

    return {
        "config": label,
        "n_queries": len(rows),
        "p50_total_ms":    round(percentile(total_ms, 50), 2),
        "p95_total_ms":    round(percentile(total_ms, 95), 2),
        "p50_gen_ms":      round(percentile(gen_ms, 50), 2),
        "p95_gen_ms":      round(percentile(gen_ms, 95), 2),
        "p50_toks_per_sec":round(percentile(tps, 50), 2),
        "mean_retrieval_ms": round(float(np.mean(retr_ms)), 2),
        "mean_rerank_ms":    round(float(np.mean(rer_ms)), 2),
    }


# ---------------------------------------------------------------------------
# Config sweep
# ---------------------------------------------------------------------------

DEFAULT_SWEEP = [
    {"name": "det_96",   "max_new_tokens": 96,  "do_sample": False, "temperature": 0.0, "top_p": 1.0},
    {"name": "det_160",  "max_new_tokens": 160, "do_sample": False, "temperature": 0.0, "top_p": 1.0},
    {"name": "det_256",  "max_new_tokens": 256, "do_sample": False, "temperature": 0.0, "top_p": 1.0},
    {"name": "samp_160", "max_new_tokens": 160, "do_sample": True,  "temperature": 0.7, "top_p": 0.9},
]

DEFAULT_QUERIES = [
    "What is CUDA and why is it useful?",
    "What is Triton Inference Server used for?",
    "Why do people use FAISS in RAG systems?",
    "What are Prometheus and Grafana used for?",
    "Explain dynamic batching in simple terms.",
] * 6  # 30 runs


def run_sweep(make_rag_fn, out_dir: Path,
              sweep: list = DEFAULT_SWEEP,
              queries: list = DEFAULT_QUERIES) -> list:
    """
    Run each config in sweep, write CSV + JSON to out_dir.
    make_rag_fn(cfg) returns a callable rag_fn(query) -> dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for cfg in sweep:
        print(f"\nRunning config: {cfg['name']}")
        rag_fn = make_rag_fn(cfg)
        summary = run_benchmark(rag_fn, queries, label=cfg["name"])
        summary.update({
            "max_new_tokens": cfg["max_new_tokens"],
            "do_sample": cfg["do_sample"],
        })
        all_rows.append(summary)
        print(f"  p50={summary['p50_total_ms']}ms  p95={summary['p95_total_ms']}ms  "
              f"toks/s={summary['p50_toks_per_sec']}")

    # Write outputs
    out_json = out_dir / "sweep_results.json"
    out_csv  = out_dir / "sweep_results.csv"

    with open(out_json, "w") as f:
        json.dump(all_rows, f, indent=2)

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_csv}")
    return all_rows


# ---------------------------------------------------------------------------
# Fixed-rate load against a SERIALIZED server (v1 experiment, corrected)
# ---------------------------------------------------------------------------

def run_fixed_rate(rag_fn, queries: list, concurrency: int,
                   target_rps: float, gpu_lock: Optional[threading.Lock] = None) -> pd.DataFrame:
    """
    Drive queries at a fixed request rate with bounded client concurrency.

    WHAT THIS IS NOT: a batching server. Every request runs inside
    `gpu_lock`, so the GPU executes exactly one request at a time at batch
    size 1. It measures queueing in front of a single-request server — nothing
    more. It is not comparable to NVIDIA Triton Inference Server or any
    dynamic/continuous batching system. For real batching see
    perfwattlab/engine/scheduler.py and run_serving.py (README §7).

    Latency accounting (fixed in v2): latency is measured from each
    request's SCHEDULED arrival time, so time spent waiting for a client
    slot (semaphore) or for the GPU lock is included. v1 started the clock
    after the semaphore, which hid queueing delay (coordinated omission).

    Returned columns:
      latency_from_arrival_ms — what a client would observe
      queue_ms               — scheduled arrival → start of service
      service_ms             — time holding the GPU
    """
    start = time.perf_counter() + 0.05
    arrivals = [start + i / target_rps for i in range(len(queries))]
    results = []
    res_lock = threading.Lock()
    sem = threading.Semaphore(concurrency)
    lock = gpu_lock or threading.Lock()

    def worker(i: int):
        now = time.perf_counter()
        if arrivals[i] > now:
            time.sleep(arrivals[i] - now)
        sem.acquire()
        try:
            with lock:
                t_service = time.perf_counter()
                row = rag_fn(queries[i])
                t_done = time.perf_counter()
            row["queue_ms"] = round((t_service - arrivals[i]) * 1000.0, 2)
            row["service_ms"] = round((t_done - t_service) * 1000.0, 2)
            row["latency_from_arrival_ms"] = round((t_done - arrivals[i]) * 1000.0, 2)
            row["query_index"] = i
            row["concurrency"] = concurrency
            row["target_rps"] = target_rps
            with res_lock:
                results.append(row)
        finally:
            sem.release()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(queries))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    df = pd.DataFrame(results).sort_values("query_index").reset_index(drop=True)
    service_rps = 1000.0 / max(df["service_ms"].median(), 1e-9)
    if target_rps > service_rps:
        print(f"  NOTE: offered load {target_rps} req/s exceeds serialized capacity "
              f"≈{service_rps:.2f} req/s — the queue grows for the whole run (overloaded).")
    return df
