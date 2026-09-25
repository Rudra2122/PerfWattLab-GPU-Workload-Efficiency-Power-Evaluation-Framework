"""
toggle_counter.py — Switching activity from VCD waveforms.

Two counts per signal:
  events       — number of value changes (a 16-bit bus changing value = 1 event).
                 This is what v1 reported as "toggles".
  bit_toggles  — number of individual bit flips (Hamming distance per change,
                 0<->1 only; transitions from/to x or z at reset are ignored).
                 Dynamic power tracks bit-level transitions weighted by the
                 switched capacitance, so this is the better proxy.

Signals are grouped so the design's own activity isn't diluted by signals it
doesn't control:
  clock     — clk (identical in both designs; valid-gating does not gate clocks)
  inputs    — ports driven by the testbench (a, b, c, valid, rst)
  internal  — the design's registers (a_r, b_r, c_r, c_d, prod, v)
  outputs   — y, out_valid

Switching activity is a PROXY. No capacitance, no synthesis, no power number.

    python toggle_counter.py baseline.vcd optimized.vcd --out-dir results/rtl
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

CLOCK = {"clk"}
INPUTS = {"a", "b", "c", "valid", "rst"}
OUTPUTS = {"y", "out_valid"}


def category(name: str) -> str:
    if name in CLOCK:
        return "clock"
    if name in INPUTS:
        return "inputs"
    if name in OUTPUTS:
        return "outputs"
    return "internal"


def _bits(val: str, width: int) -> str:
    """Left-extend a VCD vector value to its declared width (VCD rule: pad with 0,
    or with x/z if the leftmost digit is x/z)."""
    if len(val) >= width:
        return val[-width:]
    pad = val[0] if val[0] in "xz" else "0"
    return pad * (width - len(val)) + val


def count_vcd(vcd_path: str) -> pd.DataFrame:
    ids = {}                # vcd id -> (name, width)
    last = {}
    events = defaultdict(int)
    flips = defaultdict(int)
    scope = []
    in_defs = True
    with open(vcd_path, errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if in_defs:
                if line.startswith("$scope"):
                    scope.append(line.split()[2])
                elif line.startswith("$upscope"):
                    scope.pop()
                elif line.startswith("$var"):
                    p = line.split()
                    width, vid, name = int(p[2]), p[3], p[4]
                    ids.setdefault(vid, (name, width))   # aliased ids share activity
                elif line.startswith("$enddefinitions"):
                    in_defs = False
                continue
            if line[0] in "#$":
                continue
            if line[0] in "bB":
                val, vid = line[1:].split()
            elif line[0] in "01xzXZ":
                val, vid = line[0], line[1:]
            else:
                continue                      # real values etc. — not used here
            if vid not in ids:
                continue
            width = ids[vid][1]
            val = _bits(val.lower(), width)
            prev = last.get(vid)
            if prev is not None and prev != val:
                events[vid] += 1
                flips[vid] += sum(1 for p, q in zip(prev, val) if p in "01" and q in "01" and p != q)
            last[vid] = val
    rows = [{"signal": n, "width": w, "category": category(n),
             "events": events.get(v, 0), "bit_toggles": flips.get(v, 0)} for v, (n, w) in ids.items()]
    return pd.DataFrame(rows).sort_values(["category", "signal"]).reset_index(drop=True)


def compare(baseline_vcd: str, optimized_vcd: str, out_dir: Path = None, label: str = "") -> pd.DataFrame:
    b, o = count_vcd(baseline_vcd), count_vcd(optimized_vcd)
    b["design"], o["design"] = "baseline", "optimized"
    by_sig = pd.concat([b, o], ignore_index=True)
    rows = []
    for scope, filt in [("all_signals", lambda d: d),
                        ("internal", lambda d: d[d.category == "internal"]),
                        ("internal+outputs", lambda d: d[d.category.isin(["internal", "outputs"])]),
                        ("clock", lambda d: d[d.category == "clock"]),
                        ("inputs", lambda d: d[d.category == "inputs"])]:
        for metric in ("events", "bit_toggles"):
            vb, vo = int(filt(b)[metric].sum()), int(filt(o)[metric].sum())
            rows.append({"scope": scope, "metric": metric, "baseline": vb, "optimized": vo,
                         "reduction_pct": round(100 * (vb - vo) / vb, 2) if vb else 0.0})
    summ = pd.DataFrame(rows)
    if label:
        summ.insert(0, "run", label)
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        suf = f"_{label}" if label else ""
        summ.to_csv(out_dir / f"toggle_summary{suf}.csv", index=False)
        by_sig.to_csv(out_dir / f"toggles_by_signal{suf}.csv", index=False)
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Switching activity from VCD files")
    ap.add_argument("baseline")
    ap.add_argument("optimized")
    ap.add_argument("--out-dir", default="results/rtl")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    s = compare(a.baseline, a.optimized, Path(a.out_dir), a.label)
    print(s.to_string(index=False))
    print("\nSwitching activity is a proxy for dynamic power (no capacitance model, no synthesis).")
