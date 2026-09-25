"""
run_power.py — kept for backward compatibility.

v1 ran the baseline block, then the optimized block, with free-length
outputs and 5 Hz sample integration. Those choices produced a token-count
artifact and an order confound (README §2, §4.2). This entry point now runs
Experiment 0 with the v2 methodology, comparing only the two v1 paths:

    python run_power.py            ==  python run_exp0.py --variants pipeline,direct
"""
import sys

import run_exp0

if __name__ == "__main__":
    if not any(a.startswith("--variants") for a in sys.argv[1:]):
        sys.argv += ["--variants", "pipeline,direct"]
    run_exp0.main()
