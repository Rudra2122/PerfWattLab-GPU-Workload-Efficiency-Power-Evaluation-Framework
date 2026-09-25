"""
Experiment 3 — Continuous batching vs static batching vs the serialized server
(README §7).

For each arrival rate, the SAME open-loop Poisson workload (identical prompts,
output lengths and arrival times) is replayed against each policy.

Outputs (results/v2/exp3_serving/<model>_<mix>/):
  summary.csv                   one row per (policy, rate)
  requests_<policy>_<rate>.csv  per-request TTFT / TPOT / ITL / queue / slowdown
  iterations_<policy>_<rate>.csv  per-iteration batch size, padding waste,
                                KV blocks, preemptions, gather time
  report.md, serving_curves.png, batch_timeline.png

    python run_serving.py                         # short mix, 5 rates, 3 policies
    python run_serving.py --mix heavy_tail --rates 1,2,4,8
    python run_serving.py --kv-budget-gb 0.5      # force KV pressure → preemption
    python run_serving.py --model tiny-random --n-requests 12 --rates 50   # CPU smoke
    python run_serving.py --mix long --policies continuous,continuous_chunked --prefill-chunk 256
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from perfwattlab.energy import EnergyMeter
from perfwattlab.engine.generate_loop import generate
from perfwattlab.engine.kv_cache import PagedKVCache
from perfwattlab.engine.loadgen import fresh, make_workload
from perfwattlab.engine.model_utils import (DEFAULT_MODEL, load_causal_lm, model_spec,
                                            synthetic_prompt_ids)
from perfwattlab.engine.scheduler import (ContinuousBatchingEngine, run_serialized, run_static,
                                          summarize)
from perfwattlab.env import write_env


def isolated_reference(model, tok, prompt_lens, out_len=32):
    """Unloaded batch-1 TTFT(prompt_len) and TPOT, for per-request slowdown."""
    ttft, tpot = [], []
    for P in prompt_lens:
        r = generate(model, synthetic_prompt_ids(tok, P, 1), out_len, pad_id=tok.pad_token_id)
        m = r.batch_metrics()
        ttft.append(m["ttft_ms"]); tpot.append(m["tpot_ms"])
    a, b = np.polyfit(prompt_lens, ttft, 1) if len(prompt_lens) > 1 else (0.0, ttft[0])
    return (lambda p: max(a * p + b, 1e-3)), float(np.median(tpot))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--policies", default="serialized,static,continuous")
    ap.add_argument("--rates", default="0.25,0.5,1,2,4")
    ap.add_argument("--mix", default="short", choices=["short", "long", "heavy_tail"])
    ap.add_argument("--n-requests", type=int, default=64)
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--static-batch", type=int, default=8)
    ap.add_argument("--static-timeout", type=float, default=0.5)
    ap.add_argument("--max-prefill-tokens", type=int, default=2048)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--kv-budget-gb", type=float, default=None,
                    help="KV pool size; default = 60%% of free GPU memory after loading the model")
    ap.add_argument("--slo-ttft-ms", type=float, default=2000)
    ap.add_argument("--slo-tpot-ms", type=float, default=100)
    ap.add_argument("--triton-rmsnorm", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1,
                    help="independent workloads per (rate); seed, seed+1, ... → mean ± 95%% CI")
    ap.add_argument("--last-token-logits", action="store_true")
    ap.add_argument("--prefill-chunk", type=int, default=256,
                    help="chunk size for the continuous_chunked policy (prompt tokens per iteration)")
    ap.add_argument("--out-dir", default="results/v2/exp3_serving")
    args = ap.parse_args()

    model, tok = load_causal_lm(args.model)
    spec = model_spec(model, args.model)
    if args.triton_rmsnorm:
        from perfwattlab.kernels.rmsnorm_triton import patch_model_rmsnorm
        patch_model_rmsnorm(model)
    out = Path(args.out_dir) / f"{args.model.split('/')[-1]}_{args.mix}"
    out.mkdir(parents=True, exist_ok=True)
    write_env(out, vars(args))

    # KV pool sized once and reused across runs
    if args.kv_budget_gb:
        budget = int(args.kv_budget_gb * 2**30)
    elif torch.cuda.is_available():
        budget = int(torch.cuda.mem_get_info()[0] * 0.6)
    else:
        budget = 256 * 2**20
    n_blocks = PagedKVCache.blocks_for_budget(spec, budget, args.block_size)
    print(f"KV pool: {budget / 2**30:.2f} GiB → {n_blocks} blocks × {args.block_size} tokens "
          f"({n_blocks * args.block_size:,} token slots)")
    kv = PagedKVCache(spec, n_blocks, args.block_size, dtype=next(model.parameters()).dtype)

    generate(model, synthetic_prompt_ids(tok, 64, 2), 8, pad_id=tok.pad_token_id)   # warmup
    sample = make_workload(8, 1.0, args.mix, spec.vocab, spec.max_positions, seed=args.seed)
    plens = sorted({len(r.prompt_ids) for r in sample})
    iso_ttft, iso_tpot = isolated_reference(model, tok, [plens[0], plens[len(plens) // 2], plens[-1]])
    print(f"Isolated reference: TPOT {iso_tpot:.1f} ms; TTFT({plens[-1]}) {iso_ttft(plens[-1]):.1f} ms")

    meter = EnergyMeter()
    meter.measure_idle_power(5)
    summaries = []
    for rate, rep in [(float(x), r) for x in args.rates.split(",") for r in range(args.repeats)]:
        work = make_workload(args.n_requests, rate, args.mix, spec.vocab, spec.max_positions,
                             bos_id=tok.bos_token_id, seed=args.seed + rep)
        for pol in args.policies.split(","):
            reqs = fresh(work)
            t = meter.begin()
            if pol == "serialized":
                res = run_serialized(model, reqs, pad_id=tok.pad_token_id)
            elif pol == "static":
                res = run_static(model, reqs, pad_id=tok.pad_token_id, max_batch=args.static_batch,
                                 batch_timeout_s=args.static_timeout)
            elif pol in ("continuous", "continuous_chunked"):
                kv.reset()
                eng = ContinuousBatchingEngine(model, spec, kv, pad_id=tok.pad_token_id,
                                               max_batch=args.max_batch,
                                               max_prefill_tokens=args.max_prefill_tokens,
                                               last_token_logits=args.last_token_logits,
                                               prefill_chunk=args.prefill_chunk if pol == "continuous_chunked" else None)
                res = eng.run(reqs)
            else:
                raise ValueError(pol)
            er = meter.end(t)
            s, rdf, idf = summarize(res, pol, rate, args.slo_ttft_ms, args.slo_tpot_ms, iso_ttft, iso_tpot)
            tokens = int(rdf.output_tokens.sum())
            s.update({"energy_j": er.energy_j, "energy_j_per_out_token": er.energy_j / tokens,
                      "avg_power_w": er.avg_power_w, "energy_method": er.method})
            if er.active_energy_j is not None:
                s["active_energy_j_per_out_token"] = er.active_energy_j / tokens
            s["rep"] = rep
            s["seed"] = args.seed + rep
            summaries.append(s)
            tag = f"{pol}_{rate:g}" + (f"_r{rep}" if args.repeats > 1 else "")
            rdf.to_csv(out / f"requests_{tag}.csv", index=False)
            idf.to_csv(out / f"iterations_{tag}.csv", index=False)
            print(f"[rep {rep}] " * (args.repeats > 1) + f"rate {rate:5g} {pol:10s} thr {s['throughput_out_tok_s']:8.1f} tok/s | "
                  f"TTFT p50/p99 {s['ttft_ms_p50']:8.0f}/{s['ttft_ms_p99']:8.0f} ms | "
                  f"TPOT p50 {s['tpot_ms_p50']:6.1f} ms | batch {s.get('mean_batch', 1):5.1f} | "
                  f"preempt {s.get('preemptions', 0)} | {s['energy_j_per_out_token']:.3f} J/tok")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    meter.close()

    S = pd.DataFrame(summaries)
    S.to_csv(out / "summary_by_repeat.csv" if args.repeats > 1 else out / "summary.csv", index=False)
    if args.repeats > 1:
        C, R = with_ci(S), paired_ratios(S)
        C.to_csv(out / "summary_ci.csv", index=False)
        R.to_csv(out / "policy_ratios_ci.csv", index=False)
        # summary.csv keeps one row per (policy, rate) — the mean — for compare/plots
        S = S.groupby(["policy", "arrival_rate_rps"], as_index=False, sort=False).mean(numeric_only=True)
        S["slo"] = f"TTFT<={args.slo_ttft_ms:.0f}ms & TPOT<={args.slo_tpot_ms:.0f}ms"
        S.to_csv(out / "summary.csv", index=False)
    cols = ["policy", "arrival_rate_rps", "throughput_out_tok_s", "goodput_req_s", "ttft_ms_p50",
            "ttft_ms_p99", "tpot_ms_p50", "tpot_ms_p99", "itl_p99_ms_p99", "mean_batch",
            "mean_padding_waste", "wasted_decode_tokens", "preemptions", "gather_share_of_decode",
            "slowdown_median", "slowdown_max", "energy_j_per_out_token"]
    cols = [c for c in cols if c in S.columns]
    md = [f"# Experiment 3 — serving policies — {spec.name}, mix `{args.mix}`", "",
          f"{args.n_requests} requests per run, open-loop Poisson arrivals, latency from scheduled arrival. "
          f"KV pool {n_blocks} blocks × {args.block_size}. SLO: {S.slo.iloc[0]}.", "",
          S[cols].round(3).to_markdown(index=False), ""]
    if args.repeats > 1:
        md[2] = md[2].replace("per run,", f"per run, {args.repeats} independent workloads per rate (table = mean),")
        md += ["## 95% confidence intervals across repeats", "",
               C.round(3).to_markdown(index=False), "",
               "## Paired policy ratios (same workload per repeat; geometric mean, 95% CI)", "",
               R.round(3).to_markdown(index=False), ""]
    (out / "report.md").write_text("\n".join(md))
    plot(S, out)
    print(f"\nWrote {out}")


T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
CI_METRICS = ["throughput_out_tok_s", "goodput_req_s", "ttft_ms_p50", "ttft_ms_p99", "tpot_ms_p50",
              "itl_p99_ms_p99", "mean_batch", "energy_j_per_out_token"]


def with_ci(S: pd.DataFrame) -> pd.DataFrame:
    """Mean and t-based 95% CI across repeats for each (policy, rate)."""
    rows = []
    for (pol, rate), g in S.groupby(["policy", "arrival_rate_rps"], sort=False):
        row = {"policy": pol, "arrival_rate_rps": rate, "n": len(g)}
        for m in CI_METRICS:
            if m not in g:
                continue
            x = g[m].dropna().to_numpy()
            mean = float(x.mean()) if x.size else np.nan
            half = (T95.get(x.size - 1, 1.96) * x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else np.nan
            row[m] = f"{mean:.3g} ± {half:.2g}" if x.size > 1 else f"{mean:.3g}"
        rows.append(row)
    return pd.DataFrame(rows)


def paired_ratios(S: pd.DataFrame, base: str = "serialized", other: str = "continuous") -> pd.DataFrame:
    """Per-repeat ratio other/base on the same workload, geometric mean with t-CI in log space."""
    rows = []
    for alt_base in [base, "static"]:
        for rate, g in S.groupby("arrival_rate_rps"):
            a = g[g.policy == alt_base].set_index("rep")
            b = g[g.policy == other].set_index("rep")
            common = a.index.intersection(b.index)
            if len(common) < 2:
                continue
            row = {"comparison": f"{other} / {alt_base}", "arrival_rate_rps": rate, "n": len(common)}
            for m in ["throughput_out_tok_s", "ttft_ms_p50", "tpot_ms_p50", "energy_j_per_out_token"]:
                r = np.log(b.loc[common, m].to_numpy() / a.loc[common, m].to_numpy())
                half = T95.get(r.size - 1, 1.96) * r.std(ddof=1) / np.sqrt(r.size)
                row[m] = f"{np.exp(r.mean()):.3g}× [{np.exp(r.mean() - half):.3g}, {np.exp(r.mean() + half):.3g}]"
            rows.append(row)
    return pd.DataFrame(rows)


def plot(S, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for pol, g in S.groupby("policy"):
        ax[0].plot(g.arrival_rate_rps, g.throughput_out_tok_s, "o-", label=pol)
        ax[1].plot(g.throughput_out_tok_s, g.ttft_ms_p99, "o-", label=pol)
        ax[2].plot(g.arrival_rate_rps, g.energy_j_per_out_token, "o-", label=pol)
    ax[0].set(xlabel="offered load (req/s)", ylabel="output tok/s", title="Throughput vs load")
    ax[1].set(xlabel="output tok/s", ylabel="TTFT p99 (ms)", title="Latency–throughput", yscale="log")
    ax[2].set(xlabel="offered load (req/s)", ylabel="J / output token", title="Energy per token")
    for a in ax:
        a.grid(alpha=.3); a.legend()
    fig.tight_layout(); fig.savefig(out / "serving_curves.png", dpi=140); plt.close(fig)
    top = S[S.policy == "continuous"].arrival_rate_rps.max() if "continuous" in set(S.policy) else None
    f = out / f"iterations_continuous_{top:g}.csv" if top is not None else None
    if f is not None and f.exists():
        it = pd.read_csv(f)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(it.t_s, it.batch, label="running batch")
        ax.plot(it.t_s, it.waiting, label="waiting", alpha=.7)
        ax.set(xlabel="time (s)", ylabel="requests", title=f"Continuous batching timeline @ {top:g} req/s")
        ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out / "batch_timeline.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
