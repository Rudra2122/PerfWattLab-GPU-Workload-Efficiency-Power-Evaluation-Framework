#!/usr/bin/env bash
# Independent RTL experiment (README §11): simulation sweep + scoreboard +
# bit-level switching activity + optional Yosys cell counts.
# Requires: iverilog (and optionally yosys).   Ubuntu: sudo apt-get install iverilog yosys
set -euo pipefail
cd "$(dirname "$0")"
python perfwattlab/rtl/rtl_sweep.py --out-dir results/rtl "$@"
