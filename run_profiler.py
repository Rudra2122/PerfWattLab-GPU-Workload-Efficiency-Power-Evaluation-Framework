"""
run_profiler.py — Entry point: kernel-level profiling of both generation paths.

This reproduces the core investigation of the project:
  1. Profile the baseline pipeline path
  2. Profile the optimized direct path
  3. Export Chrome traces and print kernel summary tables
  4. Show the latency delta between the two paths

Open the .json trace files at https://ui.perfetto.dev to see the GPU
execution timeline and identify synchronization stalls.

Usage:
    python run_profiler.py
    python run_profiler.py --trace-dir results/traces --max-tokens 96
"""

import argparse
from functools import partial
from pathlib import Path

from perfwattlab.pipeline import (
    load_models,
    load_index,
    build_index,
    generate_pipeline,
    generate_direct,
)
from perfwattlab.profiler import compare_profiles

INDEX_DIR = Path("index")
DATA_DIR  = Path("data")

PROFILE_PROMPT = (
    "What is the difference between CUDA and Triton, "
    "and how does dynamic batching affect GPU utilization?"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir",  default=str(INDEX_DIR))
    parser.add_argument("--trace-dir",  default="results/traces")
    parser.add_argument("--max-tokens", type=int, default=96)
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)

    print("Loading models...")
    embedder, reranker, tokenizer, model, gen_pipe = load_models()

    before_fn = partial(generate_pipeline, gen_pipe=gen_pipe, tokenizer=tokenizer)
    after_fn  = partial(generate_direct,   model=model,      tokenizer=tokenizer)

    # Wrap to match profiler's expected signature: fn(prompt, max_new_tokens)
    def before(prompt, max_new_tokens):
        return generate_pipeline(prompt, gen_pipe, tokenizer, max_new_tokens=max_new_tokens)

    def after(prompt, max_new_tokens):
        return generate_direct(prompt, model, tokenizer, max_new_tokens=max_new_tokens)

    result = compare_profiles(
        before_fn=before,
        after_fn=after,
        prompt=PROFILE_PROMPT,
        trace_dir=trace_dir,
        max_new_tokens=args.max_tokens,
    )

    print(f"\nImprovement: {result['delta_ms']} ms  ({result['pct_improvement']}%)")
    print("\nOpen traces in https://ui.perfetto.dev")
    print(f"  Before: {result['before']['trace_path']}")
    print(f"  After:  {result['after']['trace_path']}")


if __name__ == "__main__":
    main()
