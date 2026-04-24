import argparse
from functools import partial
from pathlib import Path

from perfwattlab.pipeline import (
    load_models,
    load_index,
    build_index,
    rag_once,
)
from perfwattlab.power import (
    PowerSampler,
    run_with_power,
    summarize_power_runs,
    plot_pareto,
    NVML_AVAILABLE,
)

import pandas as pd

INDEX_DIR = Path("index")
DATA_DIR  = Path("data")

QUERIES = [
    "What is CUDA and why is it useful?",
    "What is Triton Inference Server used for?",
    "Why do people use FAISS in RAG systems?",
    "What are Prometheus and Grafana used for?",
    "Explain dynamic batching in simple terms.",
] * 6  # 30 queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default=str(INDEX_DIR))
    parser.add_argument("--out-dir",   default="results/power")
    parser.add_argument("--runs",      type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not NVML_AVAILABLE:
        print("ERROR: pynvml not installed. Run: pip install nvidia-ml-py3")
        return

    print("Loading models...")
    embedder, reranker, tokenizer, model, gen_pipe = load_models()

    index_dir = Path(args.index_dir)
    if not (index_dir / "faiss.index").exists():
        print("ERROR: Run run_sweep.py first to build the FAISS index.")
        return

    print("Loading FAISS index...")
    index, chunks = load_index(index_dir)

    queries = (QUERIES * 10)[:args.runs]

    all_runs = []

    for gen_mode, config_name in [("pipeline", "baseline_pipeline"), ("direct", "optimized_direct")]:
        print(f"\nRunning: {config_name}")

        def make_rag(q):
            return rag_once(
                q, index=index, chunks=chunks,
                embedder=embedder, reranker=reranker,
                model=model, tokenizer=tokenizer,
                gen_pipe=gen_pipe, gen_mode=gen_mode,
                max_new_tokens=160,
            )

        sampler = PowerSampler(hz=5.0)

        # Warmup
        print("  Warming up...")
        make_rag(queries[0])
        make_rag(queries[1])

        sampler.start()
        runs_df = run_with_power(make_rag, queries, sampler, config_name=config_name)
        sampler.stop()

        power_df = sampler.to_df()
        power_df.to_csv(out_dir / f"power_samples_{config_name}.csv", index=False)
        runs_df.to_csv(out_dir / f"runs_{config_name}.csv", index=False)
        all_runs.append(runs_df)

        print(f"  p50 latency: {runs_df['total_ms'].median():.0f} ms")
        print(f"  p50 energy:  {runs_df['energy_j'].median():.1f} J/query")

    combined = pd.concat(all_runs, ignore_index=True)
    combined.to_csv(out_dir / "all_runs.csv", index=False)

    summary = summarize_power_runs(combined)
    summary.to_csv(out_dir / "summary.csv", index=False)

    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    plot_pareto(summary, out_dir / "latency_vs_energy_pareto.png")

    print(f"\nAll results saved to: {out_dir}")


if __name__ == "__main__":
    main()
