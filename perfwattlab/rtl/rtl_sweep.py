"""
rtl_sweep.py — Independent RTL experiment (README §11). Not related to the GPU work.

1. Simulates both MAC designs with iverilog over a sweep of valid duty cycle
   (+period) and input correlation (+hold), with a scoreboard checking that
   both produce a*b+c for every valid transaction.
2. Counts switching activity (events and bit-level toggles) per signal group.
3. Optionally synthesizes both designs with Yosys and reports cell counts
   (shows what the enable logic costs in area; still no power number).

    python perfwattlab/rtl/rtl_sweep.py --out-dir results/rtl
"""

import argparse
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from toggle_counter import compare

HERE = Path(__file__).resolve().parent


def sh(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--periods", default="1,2,4,8,16")
    ap.add_argument("--holds", default="0,1")
    ap.add_argument("--cycles", type=int, default=400)
    ap.add_argument("--out-dir", default="results/rtl")
    args = ap.parse_args()
    out = Path(args.out_dir).resolve()
    work = out / "sim"
    work.mkdir(parents=True, exist_ok=True)

    sh(["iverilog", "-g2012", "-o", str(work / "sim_base"), "-DUSE_BASELINE",
        str(HERE / "tb.v"), str(HERE / "mac_baseline.v")], work)
    sh(["iverilog", "-g2012", "-o", str(work / "sim_opt"),
        str(HERE / "tb.v"), str(HERE / "mac_optimized.v")], work)

    sweep, score = [], []
    for hold in [int(x) for x in args.holds.split(",")]:
        for period in [int(x) for x in args.periods.split(",")]:
            tag = f"p{period}_h{hold}"
            for design, exe in (("baseline", "sim_base"), ("optimized", "sim_opt")):
                o = sh(["vvp", "-n", str(work / exe), f"+period={period}", f"+hold={hold}",
                        f"+cycles={args.cycles}", f"+vcd={design}_{tag}.vcd"], work)
                m = re.search(r"transactions=(\d+) mismatches=(\d+) pending=(\d+)", o)
                score.append({"run": tag, "design": design, "period": period, "hold": hold,
                              "transactions": int(m[1]), "mismatches": int(m[2]), "pending": int(m[3])})
            s = compare(str(work / f"baseline_{tag}.vcd"), str(work / f"optimized_{tag}.vcd"),
                        out / "by_run", tag)
            s["valid_duty_pct"] = round(100 / period, 1)
            s["inputs_held_when_invalid"] = bool(hold)
            sweep.append(s)
            i = s[(s.scope == "internal") & (s.metric == "bit_toggles")].iloc[0]
            print(f"{tag:8s} duty {100 / period:5.1f}%  hold={hold}  internal bit toggles "
                  f"{i.baseline:6d} → {i.optimized:6d}  ({i.reduction_pct:+.1f}% reduction)")

    S = pd.concat(sweep, ignore_index=True)
    S.to_csv(out / "toggle_sweep.csv", index=False)
    SC = pd.DataFrame(score)
    SC.to_csv(out / "scoreboard.csv", index=False)
    ok = (SC.mismatches == 0).all() and (SC.pending == 0).all()
    print(f"\nScoreboard: {'PASS' if ok else 'FAIL'} — {SC.transactions.sum()} transactions checked")

    # default v1 configuration (25% duty, random inputs) as the headline table
    head = S[S.run == "p4_h0"].drop(columns=["run"])
    head.to_csv(out / "toggle_summary_p4_h0.csv", index=False)

    syn = synth(out)
    report(out, S, SC, head, syn, ok)
    plot(S, out)


def synth(out):
    if not shutil.which("yosys"):
        print("yosys not found — skipping synthesis stats")
        return None
    rows = []
    for top in ("mac_baseline", "mac_optimized"):
        log = subprocess.run(["yosys", "-q", "-p",
                              f"read_verilog {HERE / (top + '.v')}; synth -top {top}; tee -o /dev/stdout stat"],
                             capture_output=True, text=True).stdout
        cells = re.search(r"Number of cells:\s+(\d+)", log)
        dff = sum(int(n) for n in re.findall(r"\$_S?DFFE?_\w+\s+(\d+)", log))
        en = sum(int(n) for n in re.findall(r"\$_S?DFFE_\w+\s+(\d+)", log))
        rows.append({"design": top, "cells": int(cells[1]) if cells else None,
                     "flops": dff, "enable_flops": en})
    df = pd.DataFrame(rows)
    df.to_csv(out / "synth_stats.csv", index=False)
    return df


def report(out, S, SC, head, syn, ok):
    md = ["# RTL — valid-gated register updates (independent of the GPU experiments)", "",
          f"Scoreboard across all runs: **{'PASS' if ok else 'FAIL'}** "
          f"({SC.transactions.sum()} transactions, {SC.mismatches.sum()} mismatches).", "",
          "## Headline configuration: 25% valid duty, new random inputs every cycle", "",
          head.to_markdown(index=False), "",
          "## Sweep: internal-register bit toggles", ""]
    piv = S[(S.scope == "internal") & (S.metric == "bit_toggles")][
        ["valid_duty_pct", "inputs_held_when_invalid", "baseline", "optimized", "reduction_pct"]]
    md += [piv.to_markdown(index=False), ""]
    if syn is not None:
        md += ["## Yosys generic synthesis", "", syn.to_markdown(index=False), "",
               "`enable_flops` > 0 means the valid gating was mapped to enable flip-flops "
               "(hold muxes), not clock-gating cells.", ""]
    md += ["Switching activity is a proxy for dynamic power. No capacitance model, "
           "no gate-level power analysis.", ""]
    (out / "report.md").write_text("\n".join(md))
    print(f"Wrote {out / 'report.md'}")


def plot(S, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    d = S[(S.scope == "internal") & (S.metric == "bit_toggles")]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for hold, g in d.groupby("inputs_held_when_invalid"):
        ax.plot(g.valid_duty_pct, g.reduction_pct, "o-",
                label="inputs held when invalid" if hold else "random inputs every cycle")
    ax.set(xscale="log", xlabel="valid duty cycle (%)", ylabel="internal bit-toggle reduction (%)",
           title="Valid-gated updates: switching reduction vs duty cycle")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(out / "rtl_toggle_sweep.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
