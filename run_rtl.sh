#!/usr/bin/env bash
# run_rtl.sh — Compile, simulate, and count toggle activity for both MAC designs
#
# Requirements: iverilog, vvp (apt install iverilog)
# Then run: python perfwattlab/rtl/toggle_counter.py baseline.vcd optimized.vcd
#
# What this validates:
#   The optimized MAC unit gates register updates on the valid signal.
#   When valid=0, registers hold state — no unnecessary switching.
#   This maps software execution gating to hardware dynamic power reduction.

set -e

RTL_DIR="perfwattlab/rtl"

echo "=== Compiling and simulating MAC designs ==="
echo ""

cd "$RTL_DIR"

echo "Cleaning previous artifacts..."
rm -f baseline.vcd optimized.vcd sim_baseline sim_opt

echo "Compiling baseline..."
iverilog -g2012 -o sim_baseline -DUSE_BASELINE tb.v mac_baseline.v
vvp sim_baseline

echo "Compiling optimized..."
iverilog -g2012 -o sim_opt tb.v mac_optimized.v
vvp sim_opt

echo ""
echo "VCD files generated:"
ls -lh baseline.vcd optimized.vcd

cd ../..

echo ""
echo "=== Counting toggle activity ==="
python perfwattlab/rtl/toggle_counter.py \
    perfwattlab/rtl/baseline.vcd \
    perfwattlab/rtl/optimized.vcd \
    --out-dir results/rtl

echo ""
echo "Done. Check results/rtl/ for toggle summary CSVs."
