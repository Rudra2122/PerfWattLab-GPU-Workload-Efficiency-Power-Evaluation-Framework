"""
run_profiler.py — Quantitative profiler evidence.

  --mode exp0    profile pipeline / direct / direct_no_grad on the same prompt
                 with identical forced output length (Experiment 0, H1–H3)
  --mode engine  profile the explicit decode loop at several batch sizes
                 (Experiments 1 & 3: does batching raise GPU busy fraction?)

Outputs Chrome traces (open in https://ui.perfetto.dev), key_averages tables,
and trace_comparison.json with GPU busy fraction, idle-gap stats, kernels per
token and sync/memcpy API counts.

For Nsight Systems, wrap any runner (NVTX ranges "prefill"/"decode_step" are
emitted by the engine when --nvtx is passed to run_prefill_decode.py):
    nsys profile -t cuda,nvtx,osrt -o results/v2/nsys_decode \\
        python run_prefill_decode.py --quick --nvtx
"""

import argparse
from pathlib import Path

from perfwattlab.env import write_env
from perfwattlab.profiler import compare_paths

PROFILE_PROMPT = ("What is the difference between CUDA and Triton Inference Server, "
                  "and how does dynamic batching affect GPU utilization?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["exp0", "engine"], default="exp0")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--trace-dir", default="results/v2/traces")
    args = ap.parse_args()
    trace_dir = Path(args.trace_dir) / args.mode
    write_env(trace_dir, vars(args))
    N = args.max_tokens

    if args.mode == "exp0":
        from perfwattlab.pipeline import generate_direct, generate_pipeline, load_models
        _, _, tok, model, gen_pipe = load_models()
        fns = {
            "pipeline": lambda: generate_pipeline(PROFILE_PROMPT, gen_pipe, tok, N, min_new_tokens=N)[2],
            "direct": lambda: generate_direct(PROFILE_PROMPT, model, tok, N, min_new_tokens=N)[2],
            "direct_no_grad": lambda: generate_direct(PROFILE_PROMPT, model, tok, N, min_new_tokens=N,
                                                      grad_ctx="no_grad")[2],
        }
    else:
        from perfwattlab.engine.generate_loop import generate
        from perfwattlab.engine.model_utils import load_causal_lm, synthetic_prompt_ids
        model, tok = load_causal_lm()
        fns = {}
        for b in [int(x) for x in args.batches.split(",")]:
            ids = synthetic_prompt_ids(tok, args.prompt_len, n=b)
            fns[f"decode_b{b}"] = (lambda ids=ids: generate(model, ids, N, pad_id=tok.pad_token_id)
                                   .batch_metrics()["output_tokens_total"])

    res = compare_paths(fns, trace_dir)
    print(f"\nTraces and trace_comparison.json in {trace_dir}")
    for k, v in res.items():
        print(f"{k:16s} busy={v.get('gpu_busy_fraction')}  kernels/tok={v.get('kernels_per_token')}  "
              f"gap_total_ms={v.get('gap_total_ms')}  syncs={v.get('sync_calls')}")


if __name__ == "__main__":
    main()
