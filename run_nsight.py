"""
run_nsight.py — Hardware-counter evidence with Nsight Compute and Nsight Systems
(README §5.5). Replaces "estimated GB/s" with MEASURED DRAM bytes.

What it does
  0. Calibrates ACHIEVABLE DRAM bandwidth with a large device-to-device copy
     (the realistic ceiling, below the datasheet peak).
  1. Nsight Compute (ncu): for each target, profiles exactly one iteration and
     reads per-kernel hardware counters:
        dram__bytes_read.sum, dram__bytes_write.sum     measured DRAM traffic
        gpu__time_duration.sum                          kernel execution time
        dram__throughput.avg.pct_of_peak_sustained_elapsed
        sm__throughput.avg.pct_of_peak_sustained_elapsed
     and compares measured bytes with the analytic model used elsewhere
     (weights + KV for decode; 2·N·H·2 bytes for RMSNorm).
     Targets: decode steps (TinyLlama b1/b32, Qwen-7B b1/b32), a prefill,
     RMSNorm eager / torch.compile / Triton.
  2. Nsight Systems (nsys): a timeline of the explicit decode loop with NVTX
     ranges; reports GPU kernel time vs NVTX prefill/decode time (a
     system-level cross-check of the GPU-busy numbers from torch.profiler).

Each target is run twice: once plainly (to record undisturbed wall time) and
once under ncu (which serializes and replays kernels, so its timings are not
wall time).

    python run_nsight.py                         # everything available
    python run_nsight.py --skip-qwen --skip-nsys
"""

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

METRICS = ["dram__bytes_read.sum", "dram__bytes_write.sum", "gpu__time_duration.sum",
           "dram__throughput.avg.pct_of_peak_sustained_elapsed",
           "sm__throughput.avg.pct_of_peak_sustained_elapsed"]


def find_tool(name):
    p = shutil.which(name)
    if p:
        return p
    pats = [f"/usr/local/cuda*/bin/{name}", f"/opt/nvidia/nsight-compute/*/{name}",
            f"/opt/nvidia/nsight-systems/*/bin/{name}", f"/usr/local/cuda*/nsight-compute*/{name}",
            f"/usr/local/cuda*/nsight-systems*/bin/{name}"]
    for pat in pats:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


# ---------------------------------------------------------------------------
# ncu CSV parsing
# ---------------------------------------------------------------------------

def _num(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return float("nan")


def parse_ncu_csv(path) -> pd.DataFrame:
    """Parse `ncu --csv --page raw --print-units base` output (one row per kernel)."""
    lines = Path(path).read_text(errors="ignore").splitlines()
    start = next((i for i, l in enumerate(lines) if '"Kernel Name"' in l or l.startswith('"ID"')), None)
    if start is None:
        raise ValueError(f"no ncu CSV header in {path}:\n" + "\n".join(lines[:20]))
    rows = list(csv.DictReader(lines[start:]))
    out = []
    for r in rows:
        if not str(r.get("ID", "")).strip().isdigit():      # units row / junk
            continue
        d = {"kernel": r.get("Kernel Name", "")}
        for m in METRICS:
            d[m] = _num(r.get(m, "nan"))
        out.append(d)
    return pd.DataFrame(out)


def summarize_kernels(df: pd.DataFrame) -> dict:
    rd, wr = df["dram__bytes_read.sum"].sum(), df["dram__bytes_write.sum"].sum()
    t_ns = df["gpu__time_duration.sum"].sum()
    w = df["gpu__time_duration.sum"] / max(t_ns, 1e-9)
    return {
        "kernels": int(len(df)),
        "dram_read_mb": rd / 1e6, "dram_write_mb": wr / 1e6, "dram_total_mb": (rd + wr) / 1e6,
        "kernel_time_ms": t_ns / 1e6,
        "dram_gbps_over_kernel_time": (rd + wr) / max(t_ns, 1e-9),       # bytes/ns == GB/s
        "dram_pct_peak_time_weighted": float((df["dram__throughput.avg.pct_of_peak_sustained_elapsed"] * w).sum()),
        "sm_pct_peak_time_weighted": float((df["sm__throughput.avg.pct_of_peak_sustained_elapsed"] * w).sum()),
    }


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------

def calibrate_copy_bandwidth(gib=2.0):
    import torch
    n = int(gib * 2**30 // 2)
    x = torch.empty(n, dtype=torch.float16, device="cuda")
    y = torch.empty_like(x)
    for _ in range(3):
        y.copy_(x)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(10):
        y.copy_(x)
    e.record()
    torch.cuda.synchronize()
    t = s.elapsed_time(e) / 10 / 1000
    del x, y
    torch.cuda.empty_cache()
    return 2 * n * 2 / t / 1e9          # read + write


def run_target(ncu, name, target_args, out_dir: Path, env):
    meta_plain = out_dir / f"{name}_meta.json"
    meta_ncu = out_dir / f"{name}_meta_ncu.json"
    base = [sys.executable, "-m", "perfwattlab.nsight_targets"] + target_args
    r = subprocess.run(base + ["--meta", str(meta_plain)], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [{name}] plain run failed:\n{r.stderr[-2000:]}")
        return None
    log = out_dir / f"{name}_ncu.csv"
    cmd = [ncu, "--profile-from-start", "off", "--target-processes", "all", "--clock-control", "none",
           "--metrics", ",".join(METRICS), "--csv", "--page", "raw", "--print-units", "base",
           "--log-file", str(log)] + base + ["--meta", str(meta_ncu)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    text = (r.stdout + r.stderr + (log.read_text(errors="ignore") if log.exists() else ""))
    if "ERR_NVGPUCTRPERM" in text:
        raise SystemExit("ncu: no permission to read GPU performance counters (ERR_NVGPUCTRPERM). "
                         "This environment blocks profiling counters; see README §5.5 for alternatives.")
    if r.returncode != 0 or not log.exists():
        print(f"  [{name}] ncu failed (exit {r.returncode}):\n{text[-2000:]}")
        return None
    df = parse_ncu_csv(log)
    df.to_csv(out_dir / f"{name}_kernels.csv", index=False)
    meta = json.loads(meta_plain.read_text())
    s = {"target": name, **summarize_kernels(df), "wall_ms": meta["wall_ms"]}
    s["gpu_busy_fraction"] = s["kernel_time_ms"] / meta["wall_ms"] if meta["wall_ms"] else float("nan")
    s["dram_gbps_over_wall_time"] = s["dram_total_mb"] / 1e3 / (meta["wall_ms"] / 1e3)
    model_bytes = meta.get("model_bytes_total") or meta.get("model_bytes_min")
    if model_bytes:
        s["model_estimate_mb"] = model_bytes / 1e6
        s["measured_over_model"] = s["dram_total_mb"] * 1e6 / model_bytes
    for k in ("model_flops",):
        if k in meta:
            s["achieved_tflops_wall"] = meta[k] / (meta["wall_ms"] / 1e3) / 1e12
    top = df.sort_values("dram__bytes_read.sum", ascending=False).head(5)
    s["top_kernels_by_bytes"] = "; ".join(f"{k[:50]} ({b / 1e6:.0f} MB)" for k, b in
                                         zip(top.kernel, top["dram__bytes_read.sum"] + top["dram__bytes_write.sum"]))
    print(f"  [{name}] {s['kernels']} kernels | DRAM {s['dram_total_mb']:.0f} MB "
          f"(model {s.get('model_estimate_mb', float('nan')):.0f} MB) | "
          f"{s['dram_gbps_over_kernel_time']:.0f} GB/s over kernel time, "
          f"{s['dram_gbps_over_wall_time']:.0f} GB/s over wall | GPU busy {s['gpu_busy_fraction']:.0%}")
    return s


def run_nsys(nsys, out_dir: Path, env, model):
    rep = out_dir / "nsys_decode"
    cmd = [nsys, "profile", "-t", "cuda,nvtx", "--force-overwrite", "true", "-o", str(rep),
           sys.executable, "run_prefill_decode.py", "--model", model, "--prompt-lens", "512",
           "--out-lens", "64", "--batches", "1,8", "--repeats", "1", "--nvtx",
           "--out-dir", str(out_dir / "nsys_run")]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  nsys profile failed:\n{(r.stdout + r.stderr)[-2000:]}")
        return None
    rep_file = next(iter(glob.glob(str(rep) + ".nsys-rep")), None) or next(iter(glob.glob(str(rep) + ".qdrep")), None)
    res = {}
    for reports in (["nvtx_sum", "cuda_gpu_kern_sum"], ["nvtxsum", "gpukernsum"]):
        r = subprocess.run([nsys, "stats", "--report", ",".join(reports), "--format", "csv",
                            "--output", str(out_dir / "nsys"), "--force-export", "true", rep_file],
                           capture_output=True, text=True)
        files = sorted(glob.glob(str(out_dir / "nsys_*.csv")))
        if r.returncode == 0 and files:
            break
    for f in sorted(glob.glob(str(out_dir / "nsys_*.csv"))):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        col = next((c for c in d.columns if c.startswith("Total Time")), None)
        if col is None:
            continue
        if "nvtx" in f:
            rng = next((c for c in d.columns if c.lower() in ("range", "name")), None)
            for _, row in d.iterrows():
                name = str(row[rng]).lstrip(":")
                if name in ("prefill", "decode_step"):
                    res[f"nvtx_{name}_total_ms"] = row[col] / 1e6
                    res[f"nvtx_{name}_count"] = int(row.get("Instances", row.get("Count", 0)))
        elif "kern" in f:
            res["gpu_kernel_total_ms"] = d[col].sum() / 1e6
            res["distinct_kernels"] = len(d)
    t = res.get("nvtx_prefill_total_ms", 0) + res.get("nvtx_decode_step_total_ms", 0)
    if t and "gpu_kernel_total_ms" in res:
        res["gpu_busy_fraction_in_nvtx_ranges"] = res["gpu_kernel_total_ms"] / t
    print(f"  nsys: {res}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--large-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--skip-qwen", action="store_true")
    ap.add_argument("--skip-nsys", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated target names to run (default: all)")
    ap.add_argument("--out-dir", default="results/v2/nsight")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TRANSFORMERS_VERBOSITY="error", TOKENIZERS_PARALLELISM="false")

    from perfwattlab.env import write_env
    write_env(out, vars(args))
    ncu, nsys = find_tool("ncu"), find_tool("nsys")
    print(f"ncu: {ncu}\nnsys: {nsys}")

    copy_gbps = calibrate_copy_bandwidth()
    print(f"Achievable DRAM bandwidth (2 GiB device copy): {copy_gbps:.0f} GB/s")

    summaries = []
    if ncu:
        targets = [
            ("decode_tinyllama_b1_ctx1024", ["decode", "--model", args.small_model, "--batch", "1", "--context", "1024"]),
            ("decode_tinyllama_b32_ctx1024", ["decode", "--model", args.small_model, "--batch", "32", "--context", "1024"]),
            ("decode_tinyllama_b32_ctx1024_static", ["decode", "--model", args.small_model, "--batch", "32", "--context", "1024", "--static"]),
            ("decode_tinyllama_b1_ctx1024_static", ["decode", "--model", args.small_model, "--batch", "1", "--context", "1024", "--static"]),
            ("prefill_tinyllama_b8_p512", ["prefill", "--model", args.small_model, "--batch", "8", "--prompt", "512"]),
            ("rmsnorm_eager_16384x4096", ["rmsnorm", "--impl", "eager"]),
            ("rmsnorm_compile_16384x4096", ["rmsnorm", "--impl", "compile"]),
            ("rmsnorm_triton_16384x4096", ["rmsnorm", "--impl", "triton"]),
            ("rmsnorm_triton_1x2048", ["rmsnorm", "--impl", "triton", "--rows", "1", "--hidden", "2048"]),
        ]
        if not args.skip_qwen:
            targets += [
                ("decode_qwen7b_b1_ctx2048", ["decode", "--model", args.large_model, "--batch", "1", "--context", "2048"]),
                ("decode_qwen7b_b32_ctx2048", ["decode", "--model", args.large_model, "--batch", "32", "--context", "2048"]),
            ]
        if args.only:
            keep = set(args.only.split(","))
            targets = [t for t in targets if t[0] in keep]
        for name, targs in targets:
            print(f"ncu → {name}")
            s = run_target(ncu, name, targs, out, env)
            if s:
                s["achievable_copy_gbps"] = copy_gbps
                s["pct_of_achievable_over_kernel_time"] = 100 * s["dram_gbps_over_kernel_time"] / copy_gbps
                s["pct_of_achievable_over_wall"] = 100 * s["dram_gbps_over_wall_time"] / copy_gbps
                summaries.append(s)
    else:
        print("ncu not found — see README §5.5 (scripts/install_nsight.sh)")

    nsys_res = run_nsys(nsys, out, env, args.small_model) if (nsys and not args.skip_nsys) else None

    md = ["# Nsight evidence — measured DRAM traffic", "",
          f"Achievable DRAM bandwidth (2 GiB device-to-device copy): **{copy_gbps:.0f} GB/s**.", ""]
    if summaries:
        S = pd.DataFrame(summaries)
        S.to_csv(out / "summary.csv", index=False)
        cols = ["target", "kernels", "wall_ms", "kernel_time_ms", "gpu_busy_fraction", "dram_total_mb",
                "model_estimate_mb", "measured_over_model", "dram_gbps_over_kernel_time",
                "dram_gbps_over_wall_time", "pct_of_achievable_over_wall", "dram_pct_peak_time_weighted",
                "sm_pct_peak_time_weighted", "achieved_tflops_wall"]
        md += [S[[c for c in cols if c in S.columns]].round(3).to_markdown(index=False), "",
               "`kernel_time_ms` is the sum of ncu-measured kernel durations (kernels serialized by ncu); "
               "`wall_ms` is the same iteration timed without the profiler. "
               "`gpu_busy_fraction` = kernel time / wall time. `dram_*_pct` are ncu's own "
               "percent-of-peak counters, weighted by kernel time.", "",
               "Top kernels by DRAM bytes:", ""]
        md += [f"- **{r.target}**: {r.top_kernels_by_bytes}" for r in S.itertuples()]
        md.append("")
    if nsys_res:
        md += ["## Nsight Systems (explicit decode loop, TinyLlama, prompt 512, batch 1 and 8)", "",
               pd.DataFrame([nsys_res]).T.rename(columns={0: "value"}).round(3).to_markdown(), ""]
    (out / "report.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
