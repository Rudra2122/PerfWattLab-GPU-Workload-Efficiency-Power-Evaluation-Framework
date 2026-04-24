import argparse
from functools import partial
from pathlib import Path

from perfwattlab.pipeline import (
    GEN_MODEL,
    load_models,
    load_index,
    build_index,
    rag_once,
)
from perfwattlab.sweep import run_sweep, DEFAULT_SWEEP, DEFAULT_QUERIES

DATA_DIR  = Path("data")
INDEX_DIR = Path("index")
OUT_DIR   = Path("results")

SAMPLE_DOCS = {
    "doc1.txt": "CUDA is a parallel computing platform and programming model developed by NVIDIA. It enables dramatic increases in computing performance by harnessing the power of the GPU.",
    "doc2.txt": "Triton Inference Server is an open source inference serving software that simplifies deployment of AI models at scale. It supports multiple frameworks and backends.",
    "doc3.txt": "FAISS is a library for efficient similarity search and clustering of dense vectors. It is commonly used for vector search in retrieval augmented generation pipelines.",
    "doc4.txt": "Prometheus is a monitoring system and time series database. Grafana is used to visualize metrics and build dashboards for observability.",
    "doc5.txt": "Dynamic batching combines multiple inference requests into a single batch to improve GPU throughput while maintaining latency constraints.",
    "doc6.txt": "The KV cache stores attention key and value tensors during autoregressive generation, avoiding redundant recomputation of previous tokens.",
    "doc7.txt": "vLLM uses PagedAttention to manage the KV cache like virtual memory, enabling efficient serving of many concurrent sequences on a single GPU.",
}


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

    # Write sample docs if data dir is empty
    data_dir.mkdir(parents=True, exist_ok=True)
    if not list(data_dir.glob("*.txt")):
        print("Writing sample documents...")
        for name, text in SAMPLE_DOCS.items():
            (data_dir / name).write_text(text)

    print("Loading models...")
    embedder, reranker, tokenizer, model, gen_pipe = load_models()

    # Build or load index
    if not (index_dir / "faiss.index").exists():
        print("Building FAISS index...")
        index, chunks = build_index(data_dir, index_dir, embedder)
    else:
        print("Loading FAISS index...")
        index, chunks = load_index(index_dir)

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
