"""
reanalyze_v1.py — Recompute the corrected v1 numbers (README §2, §4.2, §10.2)
from the committed raw data in results/day4_runs_with_energy.csv. No GPU needed.

    python reanalyze_v1.py
"""
from pathlib import Path

import pandas as pd

from perfwattlab.stats import compare_configs, fmt_ratio

d = pd.read_csv("results/day4_runs_with_energy.csv")
d["ms_per_token"] = d.generation_ms / d.gen_tokens
d["prompt_id"] = d["query"]
base, opt = "baseline_pipeline", "optimized_direct"

print(f"runs per config: {d.groupby('config').size().to_dict()}  (v1 README claimed 1,000+)")
print("\nPer-prompt medians (note the different token counts on the median prompt):")
print(d.groupby(["query", "config"])[["gen_tokens", "generation_ms"]].median().unstack().round(1).to_string())

metrics = {"total_ms": "total latency (ms)", "generation_ms": "generation latency (ms)",
           "ms_per_token": "ms / output token", "energy_j": "energy / query (J)",
           "energy_j_per_token": "energy / token (J)", "avg_w_during_query": "avg board power (W)"}
t = compare_configs(d, metrics, "config", base, opt)
print("\nbaseline → optimized")
for _, r in t.iterrows():
    print(f"  {r['metric']:26s} pooled {fmt_ratio(r.pooled_ratio, r.pooled_lo, r.pooled_hi)}")
    print(f"  {'':26s} paired {fmt_ratio(r.paired_ratio, r.paired_lo, r.paired_hi)}")
out = Path("results/v2")
out.mkdir(parents=True, exist_ok=True)
t.to_csv(out / "v1_reanalysis.csv", index=False)
print(f"\nWrote {out / 'v1_reanalysis.csv'}")
