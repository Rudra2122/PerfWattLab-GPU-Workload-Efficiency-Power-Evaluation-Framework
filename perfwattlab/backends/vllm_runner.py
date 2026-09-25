"""
vllm_runner.py — Experiment 5: one real serving-backend comparison (README §9).

Two modes, both replaying the exact workload used by run_serving.py
(same make_workload(n, rate, mix, seed) → same prompt token ids, output
lengths and arrival times):

  offline  — vllm.LLM.generate() on all requests at once. Measures peak
             throughput with vLLM's own scheduler; no arrival process.
  server   — open-loop streaming client against a running
             `vllm serve <model>` OpenAI-compatible endpoint. TTFT/ITL are
             measured client-side from streamed chunk arrival times, from each
             request's SCHEDULED arrival time.

Fairness notes (also in the report):
  * Same model, fp16, greedy, forced output length (ignore_eos=True).
  * Server mode includes HTTP + JSON streaming overhead that the in-process
    PerfWattLab engine doesn't pay (typically ~ms per chunk). It penalizes vLLM
    slightly, which is the conservative direction for this comparison.
  * vLLM is installed separately (it pins its own torch). Turing (sm_75, T4)
    support depends on the vLLM version — check the release notes; pass
    --dtype half since T4 has no BF16.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import List

from ..engine.scheduler import Request


# ---------------------------------------------------------------------------
# Offline
# ---------------------------------------------------------------------------

def run_offline(model_name: str, requests: List[Request], gpu_memory_utilization: float = 0.85,
                max_model_len: int | None = None, dtype: str = "half", meter=None) -> dict:
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_name, dtype=dtype, gpu_memory_utilization=gpu_memory_utilization,
              max_model_len=max_model_len, enforce_eager=False)
    params = [SamplingParams(temperature=0.0, max_tokens=r.max_new_tokens, ignore_eos=True)
              for r in requests]
    prompts = [{"prompt_token_ids": r.prompt_ids} for r in requests]
    # warmup
    llm.generate(prompts[:2], params[:2], use_tqdm=False)
    # energy covers generate() only — NOT engine startup / CUDA-graph capture
    tok = meter.begin() if meter else None
    t0 = time.perf_counter()
    try:
        outs = llm.generate(prompts, params, use_tqdm=False)
    except TypeError:  # older vLLM API
        outs = llm.generate(prompt_token_ids=[r.prompt_ids for r in requests],
                            sampling_params=params, use_tqdm=False)
    wall = time.perf_counter() - t0
    energy = meter.end(tok) if meter else None
    n_out = sum(len(o.outputs[0].token_ids) for o in outs)
    n_in = sum(len(r.prompt_ids) for r in requests)
    r = {"backend": "vllm_offline", "n_requests": len(requests), "wall_s": wall,
         "output_tokens": n_out, "prompt_tokens": n_in,
         "throughput_out_tok_s": n_out / wall, "throughput_req_s": len(requests) / wall}
    if energy is not None:
        r.update(energy_j_per_out_token=energy.energy_j / n_out, avg_power_w=energy.avg_power_w,
                 energy_method=energy.method)
    return r


# ---------------------------------------------------------------------------
# Server (open-loop streaming client)
# ---------------------------------------------------------------------------

async def _one(session, url, model, r: Request, t0: float):
    delay = r.arrival_s - (time.perf_counter() - t0)
    if delay > 0:
        await asyncio.sleep(delay)
    body = {"model": model, "prompt": r.prompt_ids, "max_tokens": r.max_new_tokens,
            "temperature": 0.0, "ignore_eos": True, "stream": True}
    r.admit_s = time.perf_counter() - t0          # request sent
    async with session.post(url, json=body) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {await resp.text()}")
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if not chunk.get("choices"):
                continue
            now = time.perf_counter() - t0
            # One chunk normally carries one token. If a chunk carries several,
            # they are all stamped with its arrival time.
            text_tokens = chunk["choices"][0].get("token_ids") or [None]
            for _ in text_tokens:
                r.output_ids.append(-1)
                r.token_times.append(now)
    r.finish_s = r.token_times[-1] if r.token_times else time.perf_counter() - t0


async def _replay(base_url: str, model: str, requests: List[Request]):
    import aiohttp
    url = base_url.rstrip("/") + "/v1/completions"
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        t0 = time.perf_counter()
        await asyncio.gather(*[_one(s, url, model, r, t0) for r in requests])


def run_server(base_url: str, model: str, requests: List[Request]) -> dict:
    asyncio.run(_replay(base_url, model, requests))
    return {"requests": requests, "iterations": [], "kv_stats": None}
