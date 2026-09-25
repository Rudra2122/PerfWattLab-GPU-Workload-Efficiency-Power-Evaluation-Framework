"""
static_decode.py — Decode with an in-place KV cache, optionally replayed as a
CUDA graph (README §12 → implemented in §5.6).

Why two steps
-------------
1. In-place KV writes. Hugging Face's DynamicCache appends each token with
   torch.cat, which allocates a new, larger cache every step and copies the old
   one (45% of decode DRAM traffic at batch 32, §5.5). StaticCache preallocates
   [batch, kv_heads, max_len, head_dim] once and writes each new token in place
   at `cache_position`.
2. CUDA graphs. With a static cache, every decode step has the same tensor
   shapes and the same memory addresses: input [B, 1], positions [B, 1],
   cache_position [1], a constant full-length attention mask. So one step can be
   captured once (~950 kernel launches recorded) and replayed with a single
   launch. New inputs are copied into fixed "static" buffers before each replay.

Batch size is part of the captured shapes, so a graph serves exactly one batch
size. Serving systems capture one graph per batch-size bucket and pad the batch
up to the bucket (run_decode_opt.py measures that padding cost).

Graph modes:  "none"    eager steps over the static cache
              "manual"  torch.cuda.graph capture of one step
              "compile" torch.compile(mode="reduce-overhead") fallback, which
                        builds CUDA graphs itself and tolerates graph breaks
"""

from __future__ import annotations

import time
from typing import List, Optional

import torch

from .generate_loop import GenResult, left_pad
from .model_utils import sync


def _step_inputs(nxt, pos, cp, mask):
    return dict(input_ids=nxt, attention_mask=mask, position_ids=pos, cache_position=cp, use_cache=True)


@torch.inference_mode()
def generate_static(model, batch_ids: List[List[int]], max_new_tokens: int, pad_id: int = 0,
                    graph_mode: str = "none", warmup_steps: int = 3) -> GenResult:
    """
    Greedy generation with a StaticCache. Returns a GenResult whose extra
    attributes record graph capture cost:
        res.graph_mode_used, res.capture_ms, res.graph_mem_mb, res.graph_error
    """
    from transformers import StaticCache

    dev = next(model.parameters()).device
    ids, mask, pos, lens = left_pad(batch_ids, pad_id, dev)
    B, P = ids.shape
    L = P + max_new_tokens
    padded = any(n != P for n in lens)
    # Constant full-length mask (1 = real token; future slots are hidden by the causal
    # mask via cache_position). None when there is no padding: HF then builds the
    # causal mask itself, which keeps the captured step simpler.
    full_mask = None
    if padded:
        full_mask = torch.zeros((B, L), dtype=torch.long, device=dev)
        full_mask[:, :P] = mask
        full_mask[:, P:] = 1

    cache = StaticCache(config=model.config, max_cache_len=L)
    sync()
    t_submit = time.perf_counter()
    out = model(input_ids=ids, attention_mask=full_mask, position_ids=pos, past_key_values=cache,
                cache_position=torch.arange(P, device=dev), use_cache=True)
    nxt = out.logits[:, -1, :].argmax(-1)
    first = nxt.tolist()
    t_first = time.perf_counter()
    del out

    outputs = [[t] for t in first]
    token_times = [t_first]

    # static buffers (fixed addresses for graph replay)
    tok_buf = nxt[:, None].clone()
    pos_buf = pos[:, -1:].clone()
    cp_buf = torch.tensor([P - 1], device=dev)

    def eager_step():
        pos_buf.add_(1)
        cp_buf.add_(1)
        o = model(past_key_values=cache, **_step_inputs(tok_buf, pos_buf, cp_buf, full_mask))
        return o.logits[:, -1, :].argmax(-1)

    capture_ms, graph_mem_mb, graph_error, used = 0.0, 0.0, None, "none"
    replay = None
    steps_done = 1

    def record(nxt_tokens):
        toks = nxt_tokens.tolist()                       # host sync: token exists for the client
        token_times.append(time.perf_counter())
        for i, t in enumerate(toks):
            outputs[i].append(t)
        tok_buf.copy_(nxt_tokens[:, None])

    if graph_mode != "none" and dev.type == "cuda":           # graphs need the model on the GPU
        # warm up with real steps (they produce real tokens) on a side stream
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            while steps_done < min(max_new_tokens, 1 + warmup_steps):
                record(eager_step())
                steps_done += 1
        torch.cuda.current_stream().wait_stream(s)

        if steps_done < max_new_tokens:
            m0 = torch.cuda.memory_allocated()
            t0 = time.perf_counter()
            if graph_mode == "manual":
                try:
                    g = torch.cuda.CUDAGraph()
                    # the captured step increments positions itself, exactly like eager_step
                    with torch.cuda.graph(g):
                        nxt_static = eager_step()
                    # capture records but does not execute: undo nothing, but the
                    # in-graph add_ ops will run on each replay
                    def replay():
                        g.replay()
                        return nxt_static
                    used = "manual"
                except Exception as e:                  # e.g. an op that syncs with the host
                    graph_error = f"{type(e).__name__}: {str(e)[:300]}"
                    torch.cuda.synchronize()
                    replay = None
                    graph_mode = "compile"
            if graph_mode == "compile" and replay is None:
                compiled = torch.compile(eager_step, mode="reduce-overhead", fullgraph=False)
                replay = compiled
                used = "compile"
            sync()
            capture_ms = (time.perf_counter() - t0) * 1000.0
            graph_mem_mb = (torch.cuda.memory_allocated() - m0) / 2**20

    step_fn = replay or eager_step
    while steps_done < max_new_tokens:
        record(step_fn())
        steps_done += 1

    import numpy as np
    res = GenResult(outputs, lens, t_submit, t_first, token_times, [max_new_tokens - 1] * B,
                    (np.diff(token_times) * 1000.0).tolist())
    res.graph_mode_used, res.capture_ms, res.graph_mem_mb, res.graph_error = used, capture_ms, graph_mem_mb, graph_error
    return res
