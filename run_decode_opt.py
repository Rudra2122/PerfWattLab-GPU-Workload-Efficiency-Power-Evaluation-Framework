"""
Experiment 6 — Removing decode overhead: in-place KV cache and CUDA graphs
(README §5.6). Fixed-batch decode, three modes on identical prompts:

  dynamic  HF DynamicCache (torch.cat append) — what every earlier experiment used
  static   HF StaticCache, KV written in place, eager kernel launches
  graph    StaticCache + one decode step captured as a CUDA graph, replayed per token

Reports per batch size: ITL/TPOT, decode tok/s, TTFT, energy per token, peak
memory, graph capture time and graph memory, and token agreement:
  graph vs static  should be identical (same kernels, replayed)
  static vs dynamic can differ in rare FP16 near-ties (different attention
                    kernels: the static cache always passes a mask)

Bucket test (--buckets 17:24,17:32): a graph is captured for one batch size, so a
batch of 17 would run in the 24 or 32 graph with padded dummy rows. Measures
what that padding costs vs a graph captured for exactly 17.

    python run_decode_opt.py
    python run_decode_opt.py --model Qwen/Qwen2.5-7B-Instruct --batches 1,4,16,32
    python run_decode_opt.py --model tiny-random --batches 1,2 --out-len 16 --repeats 1   # CPU smoke
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from perfwattlab.energy import EnergyMeter
from perfwattlab.engine.generate_loop import generate
from perfwattlab.engine.model_utils import DEFAULT_MODEL, load_causal_lm, synthetic_prompt_ids
from perfwattlab.engine.static_decode import generate_static
from perfwattlab.env import write_env


def run_mode(mode, model, ids, O, pad):
    if mode == "dynamic":
        return generate(model, ids, O, pad_id=pad)
    if mode == "static":
        return generate_static(model, ids, O, pad_id=pad, graph_mode="none")
    if mode == "graph":
        return generate_static(model, ids, O, pad_id=pad, graph_mode="manual")
    raise ValueError(mode)


def agreement(a, b):
    tot = sum(len(x) for x in a)
    same = sum(sum(1 for p, q in zip(x, y) if p == q) for x, y in zip(a, b))
    return same / tot if tot else float("nan")


def measure(meter, mode, model, ids, O, pad):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t = meter.begin()
    res = run_mode(mode, model, ids, O, pad)
    er = meter.end(t)
    m = res.batch_metrics()
    # steady-state ITL: skip the warmup/capture steps of the graph path
    steady = np.asarray(res.step_ms[5:] if len(res.step_ms) > 10 else res.step_ms)
    row = {"mode": mode, **m,
           "itl_steady_p50_ms": float(np.median(steady)),
           "energy_j_per_out_token": er.energy_j / m["output_tokens_total"],
           "graph_mode_used": getattr(res, "graph_mode_used", "none"),
           "capture_ms": getattr(res, "capture_ms", 0.0),
           "graph_mem_mb": getattr(res, "graph_mem_mb", 0.0),
           "graph_error": getattr(res, "graph_error", None)}
    if torch.cuda.is_available():
        row["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return row, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--out-len", type=int, default=128)
    ap.add_argument("--batches", default="1,2,4,8,16,32")
    ap.add_argument("--modes", default="dynamic,static,graph")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--buckets", default="17:24,17:32")
    ap.add_argument("--out-dir", default="results/v2/exp6_decode_opt")
    args = ap.parse_args()

    model, tok = load_causal_lm(args.model)
    out = Path(args.out_dir) / args.model.split("/")[-1]
    write_env(out, vars(args))
    P, O, pad = args.prompt_len, args.out_len, tok.pad_token_id
    modes = args.modes.split(",")
    generate(model, synthetic_prompt_ids(tok, 32, 1), 4, pad_id=pad)          # warmup
    meter = EnergyMeter()
    meter.measure_idle_power(5)

    rows = []
    for B in [int(x) for x in args.batches.split(",")]:
        for rep in range(args.repeats):
            ids = synthetic_prompt_ids(tok, P, B, seed=rep)
            got = {}
            for mode in modes:
                try:
                    row, res = measure(meter, mode, model, ids, O, pad)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    rows.append({"batch": B, "rep": rep, "mode": mode, "status": "OOM"})
                    continue
                got[mode] = res.output_ids
                row.update({"batch": B, "rep": rep, "status": "ok"})
                if "dynamic" in got and mode != "dynamic":
                    row["agree_vs_dynamic"] = agreement(got["dynamic"], res.output_ids)
                if mode == "graph" and "static" in got:
                    row["agree_vs_static"] = agreement(got["static"], res.output_ids)
                rows.append(row)
            r = [x for x in rows if x.get("batch") == B and x.get("rep") == rep and x.get("status") == "ok"]
            print(f"B={B:3d} rep {rep}: " + " | ".join(
                f"{x['mode']} {x['itl_steady_p50_ms']:.2f} ms" + (f" ({x['graph_mode_used']})" if x['mode'] == 'graph' else "")
                for x in r))
            errs = [x["graph_error"] for x in r if x.get("graph_error")]
            if errs and rep == 0:
                print(f"   manual capture failed → fell back: {errs[0]}")

    # bucket padding test (graph mode)
    bucket_rows = []
    for spec in [s for s in args.buckets.split(",") if s]:
        real, bucket = (int(x) for x in spec.split(":"))
        ids_real = synthetic_prompt_ids(tok, P, real, seed=100)
        ids_bucket = ids_real + synthetic_prompt_ids(tok, P, bucket - real, seed=200)   # dummy rows
        for label, ids, n_useful in (("exact", ids_real, real), ("bucket", ids_bucket, real)):
            try:
                row, _ = measure(meter, "graph", model, ids, O, pad)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            useful_tok_s = n_useful * (O - 1) / ((row["e2e_ms"] - row["ttft_ms"]) / 1000.0)
            bucket_rows.append({"real_batch": real, "graph_batch": len(ids), "kind": label,
                                "itl_steady_p50_ms": row["itl_steady_p50_ms"],
                                "useful_decode_tok_s": useful_tok_s,
                                "wasted_rows_pct": 100 * (len(ids) - real) / len(ids),
                                "energy_j_per_useful_token": row["energy_j_per_out_token"] * len(ids) / real})
    meter.close()

    df = pd.DataFrame(rows)
    df.to_csv(out / "runs.csv", index=False)
    ok = df[df.status == "ok"]
    S = ok.groupby(["batch", "mode"]).agg(
        itl_ms=("itl_steady_p50_ms", "median"), ttft_ms=("ttft_ms", "median"),
        decode_tok_s=("decode_tok_s", "median"), j_per_tok=("energy_j_per_out_token", "median"),
        peak_mem_gb=("peak_mem_gb", "median") if "peak_mem_gb" in ok else ("itl_steady_p50_ms", "size"),
        capture_ms=("capture_ms", "median"), graph_mem_mb=("graph_mem_mb", "median"),
        agree_vs_dynamic=("agree_vs_dynamic", "min") if "agree_vs_dynamic" in ok else ("batch", "size"),
    ).reset_index()
    S.to_csv(out / "summary.csv", index=False)
    piv = S.pivot(index="batch", columns="mode", values="itl_ms")
    for m in ("static", "graph"):
        if m in piv and "dynamic" in piv:
            piv[f"{m}_speedup"] = piv["dynamic"] / piv[m]
    piv.to_csv(out / "speedup.csv")
    used = ok[ok["mode"] == "graph"]["graph_mode_used"].unique().tolist() if "graph" in modes else []
    errs = ok["graph_error"].dropna().unique().tolist() if "graph_error" in ok else []

    md = [f"# Experiment 6 — in-place KV cache and CUDA graphs — {args.model}", "",
          f"Prompt {P}, {O} output tokens, median of {args.repeats}. Steady-state ITL excludes warmup/capture steps.", "",
          "## Decode step time (ms) and speedup vs dynamic cache", "", piv.round(3).to_markdown(), "",
          "## All metrics", "", S.round(4).to_markdown(index=False), "",
          f"Graph mode used: {used}." + (f" Manual capture error: `{errs[0]}`" if errs else ""), ""]
    if "agree_vs_static" in ok:
        a = ok["agree_vs_static"].dropna()
        md += [f"Graph replay vs eager static: token agreement min {a.min():.4f} (1.0 = identical).", ""]
    if bucket_rows:
        Bk = pd.DataFrame(bucket_rows)
        Bk.to_csv(out / "buckets.csv", index=False)
        md += ["## Bucket padding (graph mode)", "", Bk.round(3).to_markdown(index=False), ""]
    (out / "report.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
