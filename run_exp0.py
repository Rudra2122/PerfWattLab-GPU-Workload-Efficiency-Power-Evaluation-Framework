"""
Experiment 0 — Baseline characterization: pipeline() vs direct model.generate()
(README §4). Replaces the v1 run_power.py methodology.

What's different from v1
  * Fixed output length (min_new_tokens = max_new_tokens) → no token-count artifact.
  * 20 prompts × R repeats, variants INTERLEAVED in random order per prompt (H4).
  * Energy from NVML's hardware energy counter, idle power subtracted (§10.3).
  * Paired per-prompt comparisons with bootstrap 95% CIs.
  * Ablations:
      H1  direct under inference_mode vs direct under no_grad
      H2  pipeline preprocess / forward / postprocess timed separately
      H3  kwargs each path passes to model.generate(), diffed
      H4  optional blocked-order run (v1's AAA…BBB) vs interleaved
  * Optional torch.profiler comparison with quantitative trace stats.

    python run_exp0.py                       # ~15–25 min on a T4
    python run_exp0.py --repeats 5 --blocked --profile
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from perfwattlab.energy import EnergyMeter
from perfwattlab.engine.model_utils import model_spec, peak_bw_gbps
from perfwattlab.env import write_env
from perfwattlab.pipeline import (GenerateRecorder, build_prompt, ensure_index, generate_direct,
                                  generate_pipeline, generate_pipeline_staged, load_models, rerank,
                                  retrieve)
from perfwattlab.stats import compare_configs, fmt_ratio, paired_geomean_ratio, ratio_ci

QUERIES = [
    "What is CUDA and why is it useful?",
    "What is Triton Inference Server used for?",
    "Why do people use FAISS in RAG systems?",
    "What are Prometheus and Grafana used for?",
    "Explain dynamic batching in simple terms.",
    "What does the KV cache store during generation?",
    "How does vLLM manage KV cache memory?",
    "Why is GPU throughput higher with batching?",
    "What is a vector index used for?",
    "How do dashboards help with observability?",
    "What problem does PagedAttention solve?",
    "Why would you avoid recomputing previous tokens?",
    "What is retrieval augmented generation?",
    "How can you monitor a GPU inference service?",
    "What is the difference between latency and throughput?",
    "How does similarity search work on dense vectors?",
    "What frameworks can an inference server support?",
    "When does batching hurt latency?",
    "What is a time series database?",
    "Why run many sequences concurrently on one GPU?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="pipeline,direct,direct_no_grad")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-prompts", type=int, default=len(QUERIES))
    ap.add_argument("--idle-seconds", type=float, default=10.0)
    ap.add_argument("--blocked", action="store_true", help="H4: also run v1-style blocked order")
    ap.add_argument("--profile", action="store_true", help="torch.profiler comparison of variants")
    ap.add_argument("--h2-runs", type=int, default=10)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--index-dir", default="index")
    ap.add_argument("--out-dir", default="results/v2/exp0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    variants = args.variants.split(",")
    N = args.max_new_tokens
    write_env(out, vars(args))

    print("Loading models ...")
    embedder, reranker, tok, model, gen_pipe = load_models()
    index, chunks = ensure_index(Path(args.data_dir), Path(args.index_dir), embedder)

    # Prompts are built once: retrieval + rerank are <1% of latency and are
    # not what this experiment compares. Their timing is still recorded.
    prompts, rag_rows = [], []
    for i, q in enumerate(QUERIES[: args.n_prompts]):
        ret, t_r = retrieve(q, index, chunks, embedder)
        rr, t_rr = rerank(q, ret, reranker)
        p = build_prompt(q, rr)
        prompts.append(p)
        rag_rows.append({"prompt_id": i, "query": q, "retrieval_ms": t_r, "rerank_ms": t_rr,
                         "prompt_tokens": len(tok(p)["input_ids"])})
    pd.DataFrame(rag_rows).to_csv(out / "prompts.csv", index=False)

    def call(variant, prompt):
        if variant == "pipeline":
            return generate_pipeline(prompt, gen_pipe, tok, max_new_tokens=N, min_new_tokens=N)
        if variant == "direct":
            return generate_direct(prompt, model, tok, max_new_tokens=N, min_new_tokens=N,
                                   grad_ctx="inference_mode")
        if variant == "direct_no_grad":
            return generate_direct(prompt, model, tok, max_new_tokens=N, min_new_tokens=N,
                                   grad_ctx="no_grad")
        raise ValueError(variant)

    print("Warmup ...")
    for v in variants:
        for p in prompts[:2]:
            call(v, p)

    meter = EnergyMeter()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    meter.measure_idle_power(args.idle_seconds)
    print(f"Energy method: {meter.method}")

    rows = []
    rng = random.Random(args.seed)

    def run_one(v, pid, rep, pos, mode):
        t = meter.begin()
        _, gen_ms, n_tok, tps = call(v, prompts[pid])
        r = meter.end(t)
        row = {"variant": v, "order_mode": mode, "rep": rep, "prompt_id": pid, "position": pos,
               "generation_ms": gen_ms, "gen_tokens": n_tok, "ms_per_token": gen_ms / n_tok,
               "toks_per_sec": tps}
        row.update(r.as_dict())
        row["energy_j_per_token"] = row["energy_j"] / n_tok
        if row.get("active_energy_j") is not None:
            row["active_energy_j_per_token"] = row["active_energy_j"] / n_tok
        rows.append(row)

    total = args.repeats * len(prompts)
    print(f"Interleaved runs: {total} rounds × {len(variants)} variants")
    for rep in range(args.repeats):
        for pid in range(len(prompts)):
            order = variants[:]
            rng.shuffle(order)
            for pos, v in enumerate(order):
                run_one(v, pid, rep, pos, "interleaved")
        print(f"  repeat {rep + 1}/{args.repeats} done")

    if args.blocked:
        print("H4: blocked-order runs (v1 methodology) ...")
        for pos, v in enumerate(variants):
            for pid in range(len(prompts)):
                run_one(v, pid, 0, pos, "blocked")

    df = pd.DataFrame(rows)
    df.to_csv(out / "runs.csv", index=False)
    meter.close()

    # ------------------------------------------------------------------ H2
    h2 = []
    if "pipeline" in variants:
        print("H2: pipeline stage timing ...")
        for i in range(args.h2_runs):
            h2.append(generate_pipeline_staged(prompts[i % len(prompts)], gen_pipe, N, N))
        pd.DataFrame(h2).to_csv(out / "h2_pipeline_stages.csv", index=False)

    # ------------------------------------------------------------------ H3
    print("H3: recording generate() kwargs per path ...")
    h3 = {}
    for v in variants:
        with GenerateRecorder(model) as rec:
            call(v, prompts[0])
        h3[v] = rec.calls[-1] if rec.calls else {}
    keys = sorted(set().union(*[set(c) for c in h3.values()]))
    diff = {k: {v: h3[v].get(k, "<absent>") for v in variants} for k in keys
            if len({json.dumps(h3[v].get(k, "<absent>"), default=str) for v in variants}) > 1}
    (out / "h3_generate_kwargs.json").write_text(json.dumps({"calls": h3, "differences": diff},
                                                            indent=2, default=str))

    # ------------------------------------------------------------ profiler
    prof = {}
    if args.profile:
        from perfwattlab.profiler import compare_paths
        fns = {v: (lambda v=v: call(v, prompts[0])[2]) for v in variants}
        prof = compare_paths(fns, out / "traces")

    # --------------------------------------------------------------- report
    write_report(out, df, variants, h2, diff, prof, model, meter.method, N)


def write_report(out, df, variants, h2, diff, prof, model, energy_method, N):
    inter = df[df.order_mode == "interleaved"]
    metrics = {"generation_ms": "generation latency (ms)", "ms_per_token": "ms / output token",
               "energy_j": "energy / query (J)", "energy_j_per_token": "energy / token (J)",
               "active_energy_j_per_token": "active (above-idle) energy / token (J)",
               "avg_power_w": "average board power (W)",
               "mean_gpu_util_pct": "NVML GPU utilization (%)"}
    metrics = {k: v for k, v in metrics.items() if k in inter.columns and inter[k].notna().any()}
    pairs = [("pipeline", "direct"), ("direct_no_grad", "direct"), ("pipeline", "direct_no_grad")]
    lines = ["# Experiment 0 — pipeline vs direct generate (v2 methodology)", "",
             f"Fixed output length: {N} tokens. Energy method: `{energy_method}`. "
             f"Runs per variant: {len(inter) // max(len(variants), 1)} (interleaved).", ""]

    tables = []
    for a, b in pairs:
        if a in variants and b in variants:
            t = compare_configs(inter, metrics, "variant", a, b)
            t.insert(0, "comparison", f"{a} → {b}")
            tables.append(t)
            lines += [f"## {a} → {b}", ""]
            for _, r in t.iterrows():
                lines.append(f"- **{r['metric']}**: pooled {fmt_ratio(r.pooled_ratio, r.pooled_lo, r.pooled_hi)}; "
                             f"paired per-prompt {fmt_ratio(r.paired_ratio, r.paired_lo, r.paired_hi)}")
            lines.append("")
    if tables:
        pd.concat(tables).to_csv(out / "comparison.csv", index=False)

    # steady-state energy per token (sum over all queries)
    if energy_method == "none":
        lines += ["No NVML energy source available — energy metrics skipped.", ""]
    agg = inter.groupby("variant").agg(energy_j=("energy_j", "sum"), tokens=("gen_tokens", "sum"),
                                       seconds=("duration_s", "sum"))
    agg["j_per_token"] = agg.energy_j / agg.tokens
    agg["avg_power_w"] = agg.energy_j / agg.seconds
    agg.to_csv(out / "energy_aggregate.csv")
    if energy_method != "none":
        lines += ["## Aggregate energy (all interleaved queries)", "", agg.round(4).to_markdown(), ""]

    # roofline
    spec = model_spec(model)
    bw = peak_bw_gbps()
    if bw:
        floor = spec.weight_bytes / (bw * 1e9) * 1000.0
        lines += ["## Roofline check (batch-1 decode)", "",
                  f"Weights ≈ {spec.weight_bytes / 1e9:.2f} GB; peak BW ≈ {bw:.0f} GB/s → "
                  f"bandwidth floor ≈ **{floor:.1f} ms/token**.", ""]
        for v in variants:
            m = inter[inter.variant == v].ms_per_token.median()
            lines.append(f"- {v}: median {m:.1f} ms/token = {m / floor:.1f}× the floor")
        lines += ["", "ms/token here includes prefill amortized over output tokens; "
                  "Experiment 1 measures decode ITL directly.", ""]

    # H1
    if {"direct", "direct_no_grad"} <= set(variants):
        g, lo, hi, _ = paired_geomean_ratio(inter, "ms_per_token", "variant", "direct_no_grad", "direct")
        lines += ["## H1 — inference_mode vs no_grad (same direct path)", "",
                  f"direct(inference_mode) vs direct(no_grad), ms/token: {fmt_ratio(g, lo, hi)}", ""]

    # H2
    if h2:
        h = pd.DataFrame(h2)
        lines += ["## H2 — pipeline stage breakdown (median ms)", "",
                  h[["preprocess_ms", "forward_ms", "postprocess_ms", "total_ms"]].median().round(2).to_markdown(),
                  "", f"Pipeline inference context: `{h.inference_context.iloc[0]}`", ""]

    # H3
    lines += ["## H3 — generate() kwargs that differ between paths", ""]
    lines += ([f"- `{k}`: {json.dumps(v, default=str)}" for k, v in diff.items()] or ["- none"])
    lines.append("")

    # H4
    blk = df[df.order_mode == "blocked"]
    if not blk.empty and {"pipeline", "direct"} <= set(variants):
        ri = ratio_ci(inter[inter.variant == "pipeline"].ms_per_token, inter[inter.variant == "direct"].ms_per_token)
        rb = ratio_ci(blk[blk.variant == "pipeline"].ms_per_token, blk[blk.variant == "direct"].ms_per_token)
        lines += ["## H4 — run order", "",
                  f"- interleaved pipeline→direct ms/token: {fmt_ratio(*ri)}",
                  f"- blocked (v1-style) pipeline→direct ms/token: {fmt_ratio(*rb)}", ""]
    pos = inter.groupby(["variant", "position"]).ms_per_token.median().unstack()
    lines += ["Median ms/token by position within each interleaved round:", "", pos.round(2).to_markdown(), ""]

    if prof:
        keep = ["wall_ms_profiled", "gpu_busy_fraction", "kernels_per_token", "gap_p50_us",
                "gap_total_ms", "cpu_ops_per_token", "launches_per_token", "aten_copy_per_token"]
        pt = pd.DataFrame({k: {c: v.get(c) for c in keep} for k, v in prof.items()})
        lines += ["## Profiler evidence (one prompt, profiled run)", "", pt.to_markdown(), "",
                  "Sync API counts: " + json.dumps({k: v.get("sync_calls") for k, v in prof.items()}), ""]

    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {out}/report.md")


if __name__ == "__main__":
    main()
