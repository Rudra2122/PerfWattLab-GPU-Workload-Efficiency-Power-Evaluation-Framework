#!/usr/bin/env bash
# CPU-only smoke test: unit tests + every runner on a tiny random model.
# Checks that the code paths execute end to end. The numbers are meaningless.
set -euo pipefail
cd "$(dirname "$0")/.."
T=$(mktemp -d)
python -m pytest -q tests
python reanalyze_v1.py > /dev/null
python run_prefill_decode.py --model tiny-random --quick --out-dir "$T/e1"
python run_kv_cache.py --model tiny-random --quick --out-dir "$T/e2"
python run_serving.py --model tiny-random --n-requests 12 --rates 50 --out-dir "$T/e3"
TRITON_INTERPRET=1 python run_rmsnorm.py --check-only --out-dir "$T/e4"
python run_decode_opt.py --model tiny-random --batches 1,2 --prompt-len 32 --out-len 16 --repeats 1 --buckets 3:4 --out-dir "$T/e6"
python run_attention_check.py --model tiny-random --configs 2x32 --chunk 16 --profile-config 2x32 --out-dir "$T/ac"
python run_serving.py --model tiny-random --mix long --n-requests 8 --rates 50 --policies continuous,continuous_chunked --prefill-chunk 64 --out-dir "$T/e3c"
bash run_rtl.sh --out-dir "$T/rtl" --periods 1,4 --holds 0
echo "SMOKE TEST PASSED ($T)"
