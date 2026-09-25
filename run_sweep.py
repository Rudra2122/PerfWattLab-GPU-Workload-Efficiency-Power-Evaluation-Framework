"""
run_sweep.py — v1 max_new_tokens sweep for both generation paths (single request
at a time). Kept for reproducing v1; the v2 experiments are run_exp0.py …
run_vllm.py (see README §14).
"""
import argparse
from functools import partial
from pathlib import Path

from perfwattlab.pipeline import (
    GEN_MODEL,
    load_models,
    ensure_index,
    rag_once,
)
from perfwattlab.sweep import run_sweep, DEFAULT_SWEEP, DEFAULT_QUERIES

DATA_DIR  = Path("data")
INDEX_DIR = Path("index")
OUT_DIR   = Path("results")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",  default=str(DATA_DIR))
    parser.add_argument("--index-dir", default=str(INDEX_DIR))
    parser.add_argument("--out-dir",   default=str(OUT_DIR))
    parser.add_argument("--runs",      type=int, default=30,
                        help="Number of queries per config (default 30)")
    args = parser.parse_args()

    data_dir  = Path(args.data_dir)
    index_dir = Path(args.index_dir)
    out_dir   = Path(args.out_dir)

    print("Loading models...")
    embedder, reranker, tokenizer, model, gen_pipe = load_models()

    index, chunks = ensure_index(data_dir, index_dir, embedder)

    print(f"Index loaded: {index.ntotal} vectors, {len(chunks)} chunks\n")

    # Sweep both generation paths
    for gen_mode in ["pipeline", "direct"]:
        print(f"\n{'='*50}")
        print(f"Generation mode: {gen_mode}")
        print(f"{'='*50}")

        queries = (DEFAULT_QUERIES * 10)[:args.runs]

        def make_rag_fn(cfg):
            return partial(
                rag_once,
                index=index,
                chunks=chunks,
                embedder=embedder,
                reranker=reranker,
                model=model,
                tokenizer=tokenizer,
                gen_pipe=gen_pipe,
                gen_mode=gen_mode,
                max_new_tokens=cfg["max_new_tokens"],
                do_sample=cfg["do_sample"],
                temperature=cfg.get("temperature", 0.0),
                top_p=cfg.get("top_p", 1.0),
            )

        run_sweep(make_rag_fn, out_dir / gen_mode, queries=queries)

    print("\nDone. Results written to:", out_dir)


if __name__ == "__main__":
    main()
