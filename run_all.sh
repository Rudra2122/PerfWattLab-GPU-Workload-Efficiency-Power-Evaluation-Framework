#!/usr/bin/env bash
# Runs every PerfWattLab 2.0 experiment on one CUDA GPU (tested target: T4).
# Rough T4 wall time: 2.5–3.5 h total. Each step can be run on its own.
# Results land in results/v2/ (and results/rtl/); each directory has report.md + env.json.
set -euo pipefail
cd "$(dirname "$0")"

LONG_MODEL=${LONG_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}

step() { echo; echo "================ $* ================"; }

step "0. v1 re-analysis (CPU, seconds)";           python reanalyze_v1.py
step "Exp 0: pipeline vs direct + H1–H4 (~25 min)"; python run_exp0.py --repeats 3 --blocked --profile
step "Profiler evidence: decode at batch 1 vs 8";  python run_profiler.py --mode engine --batches 1,8
step "Exp 1: prefill/decode, TinyLlama ≤2K (~40 min)"
python run_prefill_decode.py --prompt-lens 128,512,1536 --out-lens 32,128,256 --batches 1,2,4,8,16
step "Exp 1: prefill/decode, long context 4K/8K ($LONG_MODEL)"
python run_prefill_decode.py --model "$LONG_MODEL" --prompt-lens 512,2048,4096,8192 --out-lens 128 --batches 1,2,4,8
step "Exp 1b: last-token-only logits (optimization A/B, Qwen)"
python run_prefill_decode.py --model "$LONG_MODEL" --prompt-lens 512,2048,4096,8192 --out-lens 128 --batches 1,4,16,32 --last-token-logits --tag lastlogits
step "Exp 2: KV cache (~10 min)";                   python run_kv_cache.py
step "Exp 3: serving, short mix, 3 repeats (~45 min)"; python run_serving.py --mix short --rates 1,4,16 --repeats 3 --max-batch 64
step "Exp 3: serving under KV pressure";            python run_serving.py --mix heavy_tail --policies continuous --rates 2,4,8 --kv-budget-gb 0.25 --out-dir results/v2/exp3_serving_kvpressure
step "Exp 4: RMSNorm kernel + end-to-end (~15 min)"; python run_rmsnorm.py --e2e
step "Nsight Compute + Nsight Systems (~15 min)"
bash scripts/install_nsight.sh && python run_nsight.py
step "RTL (CPU, ~1 min)";                           bash run_rtl.sh

cat <<'MSG'

Experiment 5 (vLLM) runs in a separate environment:
    pip install vllm aiohttp
    python run_vllm.py offline --mix short
    vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dtype half --port 8000 &
    python run_vllm.py server --mix short
    python run_vllm.py compare --mix short
MSG
