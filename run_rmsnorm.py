"""
Experiment 4 — Custom Triton RMSNorm vs eager vs torch.compile (README §8).

Part 1 (microbenchmark), for hidden ∈ {1024, 2048, 4096} × rows N ∈ {1 … 16K}:
  latency (median of CUDA-event-timed iterations), effective GB/s from the
  minimum-bytes model, % of peak bandwidth, speedup, kernels launched per
  call (from torch.profiler), and max abs error vs an fp32 reference.
  num_warps is tuned per shape for the Triton kernel.

Part 2 (end-to-end, --e2e): TinyLlama decode ITL and TTFT with
  eager RMSNorm | Triton RMSNorm | identity (RMSNorm removed).
  The identity run is not a valid model — it only gives the UPPER BOUND on
  what any RMSNorm optimization can save (Amdahl), measured rather than guessed.

FP16 only on T4: Turing has no native BF16. On Ampere+ pass --dtypes fp16,bf16.

    python run_rmsnorm.py
    python run_rmsnorm.py --e2e
    TRITON_INTERPRET=1 python run_rmsnorm.py --check-only     # CPU correctness check
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from perfwattlab.env import write_env
import os

from perfwattlab.kernels.rmsnorm_triton import (min_bytes, patch_model_rmsnorm, require_triton_gpu,
                                                rmsnorm_eager, rmsnorm_reference_fp32, rmsnorm_triton,
                                                triton_gpu_supported, unpatch_model_rmsnorm)

DT = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def time_ms(fn, iters=200, warmup=25):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.median(ts))


def kernels_per_call(fn):
    from torch.profiler import ProfilerActivity, profile
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn(); torch.cuda.synchronize()
    return sum(1 for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA)


def check(dtype=torch.float32, dev="cpu"):
    rows = []
    for H in (100, 1024, 2048, 4096):
        x = torch.randn(33, H, device=dev, dtype=dtype)
        w = torch.randn(H, device=dev, dtype=dtype)
        ref = rmsnorm_reference_fp32(x, w)
        rows.append({"hidden": H,
                     "triton_max_abs_err": (rmsnorm_triton(x, w).float() - ref).abs().max().item(),
                     "eager_max_abs_err": (rmsnorm_eager(x, w).float() - ref).abs().max().item()})
    return pd.DataFrame(rows)


def microbench(hiddens, rows_list, dtypes, bw):
    compiled = torch.compile(rmsnorm_eager)
    out = []
    for dname in dtypes:
        dt = DT[dname]
        for H in hiddens:
            w = torch.randn(H, device="cuda", dtype=dt)
            for N in rows_list:
                x = torch.randn(N, H, device="cuda", dtype=dt)
                ref = rmsnorm_reference_fp32(x, w)
                t_eager = time_ms(lambda: rmsnorm_eager(x, w))
                t_comp = time_ms(lambda: compiled(x, w))
                best = (float("inf"), None)
                for nw in (1, 2, 4, 8, 16):
                    try:
                        t = time_ms(lambda: rmsnorm_triton(x, w, num_warps=nw), iters=50)
                        best = min(best, (t, nw))
                    except Exception:
                        pass
                t_tri = time_ms(lambda: rmsnorm_triton(x, w, num_warps=best[1]))
                nb = min_bytes(N, H, x.element_size())
                row = {
                    "dtype": dname, "hidden": H, "rows": N,
                    "eager_us": t_eager * 1e3, "compile_us": t_comp * 1e3, "triton_us": t_tri * 1e3,
                    "triton_num_warps": best[1],
                    "min_bytes": nb,
                    "eager_gbps": nb / (t_eager / 1e3) / 1e9,
                    "compile_gbps": nb / (t_comp / 1e3) / 1e9,
                    "triton_gbps": nb / (t_tri / 1e3) / 1e9,
                    "speedup_vs_eager": t_eager / t_tri, "speedup_vs_compile": t_comp / t_tri,
                    "eager_kernels": kernels_per_call(lambda: rmsnorm_eager(x, w)),
                    "compile_kernels": kernels_per_call(lambda: compiled(x, w)),
                    "triton_kernels": kernels_per_call(lambda: rmsnorm_triton(x, w, num_warps=best[1])),
                    "triton_max_abs_err": (rmsnorm_triton(x, w).float() - ref).abs().max().item(),
                    "eager_max_abs_err": (rmsnorm_eager(x, w).float() - ref).abs().max().item(),
                }
                if bw:
                    row["triton_pct_peak"] = 100 * row["triton_gbps"] / bw
                out.append(row)
                print(f"{dname} H={H:5d} N={N:6d}  eager {row['eager_us']:8.1f}us  compile {row['compile_us']:8.1f}us  "
                      f"triton {row['triton_us']:8.1f}us  ({row['triton_gbps']:6.1f} GB/s, "
                      f"{row['speedup_vs_eager']:.2f}× vs eager, kernels {row['eager_kernels']}→{row['triton_kernels']})")
    return pd.DataFrame(out)


def e2e(model_name, prompt_len, out_len, batches, repeats):
    from perfwattlab.engine.generate_loop import generate
    from perfwattlab.engine.model_utils import load_causal_lm, synthetic_prompt_ids
    model, tok = load_causal_lm(model_name)
    rows = []

    def identity_patch(m):
        n = 0
        for mod in m.modules():
            if type(mod).__name__.endswith("RMSNorm"):
                mod._pwl_orig_forward = mod.forward
                mod.forward = lambda h: h
                n += 1
        return n

    for mode in ("eager", "triton", "identity_upper_bound"):
        if mode == "triton":
            patch_model_rmsnorm(model)
        elif mode == "identity_upper_bound":
            unpatch_model_rmsnorm(model)
            identity_patch(model)
        for B in batches:
            ids = synthetic_prompt_ids(tok, prompt_len, B)
            generate(model, ids, 8, pad_id=tok.pad_token_id)
            for r in range(repeats):
                m = generate(model, ids, out_len, pad_id=tok.pad_token_id).batch_metrics()
                rows.append({"mode": mode, "batch": B, "rep": r, **m})
        print(f"{mode} done")
    unpatch_model_rmsnorm(model)
    df = pd.DataFrame(rows)
    s = df.groupby(["batch", "mode"])[["ttft_ms", "itl_p50_ms", "tpot_ms", "decode_tok_s"]].median().unstack()
    res = []
    for B in batches:
        e, t, i = (s.loc[B, ("tpot_ms", k)] for k in ("eager", "triton", "identity_upper_bound"))
        res.append({"batch": B, "tpot_eager_ms": e, "tpot_triton_ms": t, "tpot_no_rmsnorm_ms": i,
                    "rmsnorm_share_upper_bound": (e - i) / e,
                    "triton_saving": (e - t) / e,
                    "fraction_of_possible_saving_captured": (e - t) / (e - i) if e > i else np.nan,
                    "ttft_eager_ms": s.loc[B, ("ttft_ms", "eager")],
                    "ttft_triton_ms": s.loc[B, ("ttft_ms", "triton")]})
    return df, pd.DataFrame(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hiddens", default="1024,2048,4096")
    ap.add_argument("--rows", default="1,8,64,512,4096,16384")
    ap.add_argument("--dtypes", default="fp16")
    ap.add_argument("--e2e", action="store_true")
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--out-len", type=int, default=128)
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--out-dir", default="results/v2/exp4_rmsnorm")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_env(out, vars(args))

    if args.check_only or not torch.cuda.is_available():
        # Triton's interpreter runs on CPU tensors; on sm_60 GPUs it is the only option.
        use_gpu = triton_gpu_supported() and os.environ.get("TRITON_INTERPRET") != "1"
        dev = "cuda" if use_gpu else "cpu"
        if not use_gpu and not os.environ.get("TRITON_INTERPRET"):
            raise SystemExit("No Triton-capable GPU: rerun with TRITON_INTERPRET=1 for a CPU correctness check")
        c = check(torch.float16 if dev == "cuda" else torch.float32, dev)
        print(c.to_string(index=False))
        c.to_csv(out / "correctness.csv", index=False)
        return

    require_triton_gpu()
    if "bf16" in args.dtypes and not torch.cuda.is_bf16_supported():
        print("BF16 not supported on this GPU (e.g. T4/Turing) — dropping bf16")
        args.dtypes = ",".join(d for d in args.dtypes.split(",") if d != "bf16")

    from perfwattlab.engine.model_utils import peak_bw_gbps
    bw = peak_bw_gbps()
    md = ["# Experiment 4 — RMSNorm: eager vs torch.compile vs Triton", ""]
    mb = microbench([int(x) for x in args.hiddens.split(",")], [int(x) for x in args.rows.split(",")],
                    args.dtypes.split(","), bw)
    mb.to_csv(out / "microbench.csv", index=False)
    cols = ["dtype", "hidden", "rows", "eager_us", "compile_us", "triton_us", "triton_gbps",
            "triton_pct_peak", "speedup_vs_eager", "speedup_vs_compile", "eager_kernels",
            "compile_kernels", "triton_kernels", "triton_max_abs_err"]
    md += [mb[[c for c in cols if c in mb.columns]].round(3).to_markdown(index=False), ""]
    plot(mb, out, bw)

    if args.e2e:
        runs, res = e2e(args.model, args.prompt_len, args.out_len,
                        [int(x) for x in args.batches.split(",")], args.repeats)
        runs.to_csv(out / "e2e_runs.csv", index=False)
        res.to_csv(out / "e2e_summary.csv", index=False)
        md += ["## End-to-end (decode TPOT, TTFT)", "", res.round(4).to_markdown(index=False), "",
               "`rmsnorm_share_upper_bound` = TPOT saved by deleting RMSNorm entirely (Amdahl bound).", ""]
    (out / "report.md").write_text("\n".join(md))
    print(f"\nWrote {out}")


def plot(mb, out, bw):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for H, g in mb.groupby("hidden"):
        ax.plot(g.rows, g.triton_gbps, "o-", label=f"Triton H={H}")
        ax.plot(g.rows, g.eager_gbps, "x--", label=f"eager H={H}", alpha=.6)
    if bw:
        ax.axhline(bw, color="k", ls=":", label=f"peak {bw:.0f} GB/s")
    ax.set(xscale="log", xlabel="rows (batch × seq)", ylabel="effective GB/s (min-bytes model)",
           title="RMSNorm effective bandwidth")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "rmsnorm_bandwidth.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
