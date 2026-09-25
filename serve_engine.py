"""
serve_engine.py — Serve the PerfWattLab continuous batching engine over an
OpenAI-compatible streaming HTTP API (README §7.6).

    python serve_engine.py --port 8001 &
    python run_vllm.py server --base-url http://localhost:8001 --label perfwattlab_http --mix short

Benchmarking this server and `vllm serve` with the same client makes the
Experiment 5 comparison HTTP-to-HTTP.
"""

import argparse

import torch

from perfwattlab.engine.kv_cache import PagedKVCache
from perfwattlab.engine.model_utils import DEFAULT_MODEL, load_causal_lm, model_spec
from perfwattlab.engine.scheduler import ContinuousBatchingEngine
from perfwattlab.engine.server import EngineServer


def build(model_name, max_batch=64, max_prefill_tokens=2048, block_size=16, kv_budget_gb=None,
          last_token_logits=False):
    model, tok = load_causal_lm(model_name)
    spec = model_spec(model, model_name)
    if kv_budget_gb:
        budget = int(kv_budget_gb * 2**30)
    elif torch.cuda.is_available():
        budget = int(torch.cuda.mem_get_info()[0] * 0.6)
    else:
        budget = 256 * 2**20
    n_blocks = PagedKVCache.blocks_for_budget(spec, budget, block_size)
    kv = PagedKVCache(spec, n_blocks, block_size, dtype=next(model.parameters()).dtype)
    eng = ContinuousBatchingEngine(model, spec, kv, pad_id=tok.pad_token_id, max_batch=max_batch,
                                   max_prefill_tokens=max_prefill_tokens,
                                   last_token_logits=last_token_logits)
    print(f"{model_name}: KV pool {n_blocks} blocks × {block_size} tokens, max batch {max_batch}")
    return EngineServer(eng, tok, model_name, spec.max_positions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--max-prefill-tokens", type=int, default=2048)
    ap.add_argument("--kv-budget-gb", type=float, default=None)
    ap.add_argument("--last-token-logits", action="store_true")
    args = ap.parse_args()
    from aiohttp import web
    srv = build(args.model, args.max_batch, args.max_prefill_tokens, kv_budget_gb=args.kv_budget_gb,
                last_token_logits=args.last_token_logits)
    web.run_app(srv.app(), port=args.port, print=lambda *_: print(f"serving on :{args.port}"))


if __name__ == "__main__":
    main()
