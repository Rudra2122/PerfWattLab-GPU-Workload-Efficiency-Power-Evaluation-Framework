"""
nsight_targets.py — Small, deterministic workloads for Nsight Compute.

Each target warms up, times itself WITHOUT the profiler (so we know the real
wall time), then wraps exactly one iteration in cudaProfilerStart/Stop. Run it
under `ncu --profile-from-start off` so only that iteration's kernels are
measured.

    python -m perfwattlab.nsight_targets decode  --model M --batch B --context L --meta out.json
    python -m perfwattlab.nsight_targets prefill --model M --batch B --prompt P --meta out.json
    python -m perfwattlab.nsight_targets rmsnorm --impl eager|compile|triton --rows N --hidden H --meta out.json
"""

import argparse
import json
import time

import numpy as np
import torch


def _time_ms(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def _profile_once(fn):
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def target_decode(args):
    from perfwattlab.engine.kv_cache import as_cache
    from perfwattlab.engine.model_utils import load_causal_lm, model_spec
    model, tok = load_causal_lm(args.model)
    spec = model_spec(model, args.model)
    B, L = args.batch, args.context
    dt = next(model.parameters()).dtype
    shape = (B, spec.n_kv_heads, L, spec.head_dim)
    past = tuple((torch.randn(shape, dtype=dt, device="cuda"), torch.randn(shape, dtype=dt, device="cuda"))
                 for _ in range(spec.n_layers))
    inp = torch.full((B, 1), 100, device="cuda")
    mask = torch.ones((B, L + 1), dtype=torch.long, device="cuda")
    pos = torch.full((B, 1), L, device="cuda")

    if getattr(args, "static", False):
        # KV written into a StaticCache by a real prefill (the same path run_decode_opt.py and
        # the GPU graph test use); new tokens are then written in place, no torch.cat
        from transformers import StaticCache
        from perfwattlab.engine.model_utils import last_token_logits_kwargs, synthetic_prompt_ids
        del past
        # StaticCache writes each call's token at its own GPU-side counter, not at
        # cache_position, so every timed/profiled call of step() takes a new slot:
        # leave room for all of them (13 timing calls + 1 profiled call)
        cache = StaticCache(config=model.config, max_cache_len=L + 32)
        ids = torch.tensor(synthetic_prompt_ids(tok, L, B), device="cuda")
        with torch.inference_mode():
            model(input_ids=ids, past_key_values=cache, cache_position=torch.arange(L, device="cuda"),
                  use_cache=True, **last_token_logits_kwargs(model))
        cp = torch.tensor([L], device="cuda")

        @torch.inference_mode()
        def step():
            model(input_ids=inp, position_ids=pos, past_key_values=cache, cache_position=cp, use_cache=True)
    else:
        @torch.inference_mode()
        def step():
            model(input_ids=inp, attention_mask=mask, position_ids=pos,
                  past_key_values=as_cache(past), use_cache=True)

    wall = _time_ms(step)
    _profile_once(step)
    return {"target": "decode", "model": args.model, "batch": B, "context": L, "wall_ms": wall,
            "static_cache": bool(getattr(args, "static", False)),
            "model_bytes_weights": spec.weight_bytes,
            "model_bytes_kv": B * L * spec.kv_bytes_per_token(),
            "model_bytes_total": spec.decode_bytes_per_step(B, L)}


def target_prefill(args):
    from perfwattlab.engine.model_utils import (last_token_logits_kwargs, load_causal_lm, model_spec,
                                                synthetic_prompt_ids)
    model, tok = load_causal_lm(args.model)
    spec = model_spec(model, args.model)
    ids = torch.tensor(synthetic_prompt_ids(tok, args.prompt, args.batch), device="cuda")
    extra = last_token_logits_kwargs(model) if args.last_token_logits else {}

    @torch.inference_mode()
    def fwd():
        model(input_ids=ids, use_cache=True, **extra)

    wall = _time_ms(fwd, iters=5, warmup=2)
    _profile_once(fwd)
    return {"target": "prefill", "model": args.model, "batch": args.batch, "prompt": args.prompt,
            "wall_ms": wall, "model_flops": spec.prefill_flops(args.batch * args.prompt, args.batch,
                                                               args.last_token_logits),
            "model_bytes_weights": spec.weight_bytes}


def target_rmsnorm(args):
    from perfwattlab.kernels.rmsnorm_triton import min_bytes, rmsnorm_eager, rmsnorm_triton
    x = torch.randn(args.rows, args.hidden, device="cuda", dtype=torch.float16)
    w = torch.randn(args.hidden, device="cuda", dtype=torch.float16)
    fn = {"eager": lambda: rmsnorm_eager(x, w),
          "triton": lambda: rmsnorm_triton(x, w),
          "compile": (lambda c=torch.compile(rmsnorm_eager): (lambda: c(x, w)))()}[args.impl]
    wall = _time_ms(fn, iters=50, warmup=10)
    _profile_once(fn)
    return {"target": "rmsnorm", "impl": args.impl, "rows": args.rows, "hidden": args.hidden,
            "wall_ms": wall, "model_bytes_min": min_bytes(args.rows, args.hidden, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=["decode", "prefill", "rmsnorm"])
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--last-token-logits", action="store_true")
    ap.add_argument("--static", action="store_true", help="decode: use an in-place StaticCache")
    ap.add_argument("--impl", default="triton")
    ap.add_argument("--rows", type=int, default=16384)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--meta", required=True)
    args = ap.parse_args()
    meta = {"decode": target_decode, "prefill": target_prefill, "rmsnorm": target_rmsnorm}[args.target](args)
    with open(args.meta, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
