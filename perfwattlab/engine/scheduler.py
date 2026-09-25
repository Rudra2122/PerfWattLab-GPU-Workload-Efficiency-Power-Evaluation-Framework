"""
scheduler.py — Serving engines for Experiment 3 (README §7).

Three policies, all driven by the same open-loop arrival schedule:

  serialized  — one request at a time, batch size 1 (what v1's run_fixed_rate
                actually was, now with correct latency accounting).
  static      — wait until `max_batch` requests are queued (or `batch_timeout_s`
                passes since the oldest arrived), run the whole batch to the
                longest request's length, then take the next batch.
  continuous  — iteration-level scheduling over a paged KV cache:
                every iteration (1) admits waiting requests if KV blocks and
                the prefill token budget allow, (2) prefills them, (3) runs one
                batched decode step for every running sequence, (4) grows KV
                one slot per sequence, preempting the most recently admitted
                request if blocks run out, (5) retires finished requests.
                Policy A: prefill runs as its own step before decode.

Time is wall-clock. Arrivals are scheduled relative to the engine start; a
request that "arrives" while the engine is busy simply waits in the queue.
Latency is always measured from the SCHEDULED arrival time, so queueing
delay is never hidden (no coordinated omission).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import torch

from .generate_loop import generate, left_pad
from .kv_cache import OutOfBlocks, PagedKVCache, as_cache, legacy_past
from .model_utils import ModelSpec, last_token_logits_kwargs, sync


@dataclass
class Request:
    rid: int
    prompt_ids: List[int]
    max_new_tokens: int
    arrival_s: float                          # relative to engine start
    output_ids: List[int] = field(default_factory=list)
    token_times: List[float] = field(default_factory=list)   # relative to engine start
    admit_s: Optional[float] = None
    finish_s: Optional[float] = None
    preemptions: int = 0
    prefilled: int = 0                        # context tokens whose KV exists (chunked prefill)

    @property
    def done(self) -> bool:
        return len(self.output_ids) >= self.max_new_tokens

    def metrics(self) -> dict:
        t = np.asarray(self.token_times)
        n = len(t)
        ttft = (t[0] - self.arrival_s) * 1000.0 if n else float("nan")
        e2e = (t[-1] - self.arrival_s) * 1000.0 if n else float("nan")
        itl = np.diff(t) * 1000.0
        return {
            "rid": self.rid, "prompt_tokens": len(self.prompt_ids), "output_tokens": n,
            "arrival_s": round(self.arrival_s, 4),
            "queue_ms": ((self.admit_s - self.arrival_s) * 1000.0) if self.admit_s is not None else float("nan"),
            "ttft_ms": ttft, "e2e_ms": e2e,
            "tpot_ms": (e2e - ttft) / (n - 1) if n > 1 else float("nan"),
            "itl_p50_ms": float(np.percentile(itl, 50)) if itl.size else float("nan"),
            "itl_p99_ms": float(np.percentile(itl, 99)) if itl.size else float("nan"),
            "itl_max_ms": float(itl.max()) if itl.size else float("nan"),
            "preemptions": self.preemptions,
        }


class _Clock:
    def __init__(self):
        self.t0 = time.perf_counter()

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def sleep_until(self, t_rel: float):
        dt = t_rel - self.now()
        if dt > 0:
            time.sleep(dt)


# ---------------------------------------------------------------------------
# Serialized (batch 1)
# ---------------------------------------------------------------------------

def run_serialized(model, requests: List[Request], pad_id: int, **_) -> dict:
    clock = _Clock()
    iters = []
    for r in sorted(requests, key=lambda x: x.arrival_s):
        clock.sleep_until(r.arrival_s)
        r.admit_s = clock.now()
        res = generate(model, [r.prompt_ids], r.max_new_tokens, pad_id=pad_id, ignore_eos=True)
        r.output_ids = res.output_ids[0]
        r.token_times = [t - clock.t0 for t in res.token_times]
        r.finish_s = r.token_times[-1]
        iters.append({"t_s": r.admit_s, "batch": 1, "n_prefill": 1, "padding_waste": 0.0,
                      "wasted_decode_tokens": 0})
    return {"requests": requests, "iterations": iters, "kv_stats": None}


# ---------------------------------------------------------------------------
# Static batching
# ---------------------------------------------------------------------------

def run_static(model, requests: List[Request], pad_id: int, max_batch: int = 8,
               batch_timeout_s: float = 0.5, **_) -> dict:
    clock = _Clock()
    pending = sorted(requests, key=lambda x: x.arrival_s)
    iters = []
    i = 0
    while i < len(pending):
        first = pending[i]
        clock.sleep_until(first.arrival_s)
        deadline = first.arrival_s + batch_timeout_s
        # close the batch when it is full, when the oldest request's timeout
        # expires, or when every remaining request has already arrived
        while True:
            now = clock.now()
            n_arr = sum(1 for r in pending[i:i + max_batch] if r.arrival_s <= now)
            if n_arr >= max_batch or now >= deadline or i + n_arr == len(pending):
                break
            clock.sleep_until(min(pending[i + n_arr].arrival_s, deadline))
        batch = [r for r in pending[i:i + max_batch] if r.arrival_s <= clock.now()] or [first]
        i += len(batch)
        now = clock.now()
        for r in batch:
            r.admit_s = now
        L = max(r.max_new_tokens for r in batch)
        lens = [len(r.prompt_ids) for r in batch]
        res = generate(model, [r.prompt_ids for r in batch], L, pad_id=pad_id, ignore_eos=True)
        times = [t - clock.t0 for t in res.token_times]
        for b, r in enumerate(batch):
            r.output_ids = res.output_ids[b][: r.max_new_tokens]
            r.token_times = times[: r.max_new_tokens]
            r.finish_s = r.token_times[-1]
        iters.append({
            "t_s": now, "batch": len(batch), "n_prefill": len(batch),
            "padding_waste": 1 - sum(lens) / (len(batch) * max(lens)),
            # decode tokens computed for requests that were already finished
            "wasted_decode_tokens": sum(L - r.max_new_tokens for r in batch),
        })
    return {"requests": requests, "iterations": iters, "kv_stats": None}


# ---------------------------------------------------------------------------
# Continuous batching over a paged KV cache
# ---------------------------------------------------------------------------

class ContinuousBatchingEngine:
    """
    Iteration-level scheduler over a paged KV cache.

    Two ways to drive it:
      run(requests)  — offline replay of a precomputed arrival schedule (Exp 3)
      add() + step() — online, one iteration at a time (used by the streaming
                       HTTP server in engine/server.py)

    on_token(request, token_id, t_s, finished) is called for every generated
    token as soon as it exists on the host — that is the streaming hook.
    """

    def __init__(self, model, spec: ModelSpec, kv: PagedKVCache, pad_id: int,
                 max_batch: int = 32, max_prefill_tokens: int = 2048,
                 watermark_blocks: Optional[int] = None, eos_id: Optional[int] = None,
                 on_token: Optional[Callable] = None, last_token_logits: bool = False,
                 prefill_chunk: Optional[int] = None):
        self.model = model
        self.spec = spec
        self.kv = kv
        self.pad_id = pad_id
        self.max_batch = max_batch
        self.max_prefill_tokens = max_prefill_tokens
        self.watermark = watermark_blocks if watermark_blocks is not None else max(1, kv.num_blocks // 100)
        self.dev = next(model.parameters()).device
        self.on_token = on_token
        self.prefill_extra = last_token_logits_kwargs(model) if last_token_logits else {}
        self.prefill_chunk = prefill_chunk
        self.iterations: List[dict] = []
        self.waiting: List[Request] = []
        self.running: List[Request] = []
        self.prefilling: List[Request] = []        # admitted, prefill not finished (chunked mode)

    # -- helpers -----------------------------------------------------------
    def _context(self, r: Request) -> List[int]:
        """Tokens whose KV must exist: prompt + all generated tokens except the last
        (the last generated token is fed as the next decode input)."""
        return r.prompt_ids + r.output_ids[:-1] if r.output_ids else r.prompt_ids

    def _emit(self, r: Request, tok: int, t: float):
        r.output_ids.append(tok)
        r.token_times.append(t)
        if self.on_token is not None:
            self.on_token(r, tok, t, r.done)

    @torch.inference_mode()
    def _prefill(self, reqs: List[Request], clock: _Clock) -> int:
        """Batched prefill. For fresh requests this produces output token #1.
        For preempted requests (recompute) it rebuilds KV for prompt+outputs[:-1]
        and discards the logits, since outputs[-1] is already known."""
        ctxs = [self._context(r) for r in reqs]
        ids, mask, pos, lens = left_pad(ctxs, self.pad_id, self.dev)
        out = self.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True,
                         **self.prefill_extra)
        nxt = out.logits[:, -1, :].argmax(-1).tolist()
        t = clock.now()
        past = legacy_past(out.past_key_values)
        for b, r in enumerate(reqs):
            self.kv.write_prefill(r.rid, past, b, lens[b])
            if not r.output_ids:                      # fresh request
                self._emit(r, nxt[b], t)
        return sum(lens)

    @torch.inference_mode()
    def _prefill_chunk(self, r: Request, take: int, clock: _Clock) -> bool:
        """
        Prefill the next `take` context tokens of r, attending to the KV already
        written for its earlier chunks. Returns True when r's prefill is complete.
        """
        ctx = self._context(r)
        s0 = r.prefilled
        ids = torch.tensor([ctx[s0:s0 + take]], device=self.dev)
        pos = torch.arange(s0, s0 + take, device=self.dev)[None]
        mask = torch.ones((1, s0 + take), dtype=torch.long, device=self.dev)
        past = as_cache(self.kv.gather([r.rid])[0]) if s0 > 0 else None
        out = self.model(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=past,
                         use_cache=True, **last_token_logits_kwargs(self.model))
        self.kv.write_prefill(r.rid, out.past_key_values, 0, take, start=s0)
        r.prefilled = s0 + take
        if r.prefilled < len(ctx):
            return False
        nxt = out.logits[:, -1, :].argmax(-1).tolist()
        if not r.output_ids:                        # fresh request: this is output token #1
            self._emit(r, nxt[0], clock.now())
        return True

    @torch.inference_mode()
    def _decode(self, reqs: List[Request], clock: _Clock) -> dict:
        past, mask, L = self.kv.gather([r.rid for r in reqs])
        B = len(reqs)
        mask = torch.cat([mask, mask.new_ones((B, 1))], dim=1)
        inp = torch.tensor([[r.output_ids[-1]] for r in reqs], device=self.dev)
        pos = torch.tensor([[self.kv.seq_lens[r.rid]] for r in reqs], device=self.dev)
        out = self.model(input_ids=inp, attention_mask=mask, position_ids=pos,
                         past_key_values=as_cache(past), use_cache=True)
        nxt = out.logits[:, -1, :].argmax(-1).tolist()
        t = clock.now()
        self.kv.write_decode([r.rid for r in reqs], out.past_key_values)
        for b, r in enumerate(reqs):
            self._emit(r, nxt[b], t)
        lens = [self.kv.seq_lens[r.rid] - 1 for r in reqs]
        return {"padding_waste": 1 - sum(lens) / (B * L) if L else 0.0, "context_max": L}

    def _preempt_one(self) -> Request:
        victim = max(self.running, key=lambda r: (r.admit_s, r.rid))   # most recently admitted
        self.running.remove(victim)
        self.kv.free(victim.rid)
        victim.preemptions += 1
        victim.prefilled = 0
        self.waiting.insert(0, victim)                                 # readmit first
        return victim

    def _retire(self) -> List[Request]:
        fin = [r for r in self.running if r.done]
        for r in fin:
            self.running.remove(r)
            self.kv.free(r.rid)
            r.finish_s = r.token_times[-1]
        return fin

    # -- online API ------------------------------------------------------------
    def add(self, r: Request):
        self.waiting.append(r)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running or self.prefilling)

    def step(self, clock: _Clock) -> List[Request]:
        """One scheduler iteration. Returns requests that finished in it."""
        it_t0 = time.perf_counter()
        finished: List[Request] = []

        # 1) admission
        admitted, budget = [], self.max_prefill_tokens
        while self.waiting and len(self.running) + len(self.prefilling) + len(admitted) < self.max_batch:
            r = self.waiting[0]
            n_ctx = len(self._context(r))
            if admitted and n_ctx > budget:
                break
            # keep a watermark of free blocks for running sequences to grow into,
            # unless nothing is running (otherwise we could deadlock)
            reserve = self.watermark if (self.running or self.prefilling or admitted) else 0
            if not self.kv.can_allocate(n_ctx + 1, reserve_blocks=reserve):
                break
            self.kv.allocate(r.rid, n_ctx + 1)
            self.waiting.pop(0)
            if r.admit_s is None:
                r.admit_s = clock.now()
            admitted.append(r)
            budget -= n_ctx
        if not self.running and not self.prefilling and not admitted and self.waiting:
            raise OutOfBlocks(f"request {self.waiting[0].rid} ({len(self._context(self.waiting[0]))} tokens) "
                              f"cannot fit in an empty KV pool of {self.kv.num_blocks} blocks")

        # 2) prefill (Policy A: separate step before decode)
        prefill_tokens, prefill_ms = 0, 0.0
        if self.prefill_chunk is None and admitted:
            t0 = time.perf_counter()
            prefill_tokens = self._prefill(admitted, clock)
            prefill_ms = (time.perf_counter() - t0) * 1000.0
            self.running.extend(admitted)
        elif self.prefill_chunk is not None:
            # chunked prefill: at most prefill_chunk prompt tokens per iteration,
            # so running sequences never wait for a whole long prompt
            self.prefilling.extend(admitted)
            budget = self.prefill_chunk
            t0 = time.perf_counter()
            while budget > 0 and self.prefilling:
                r = self.prefilling[0]
                take = min(budget, len(self._context(r)) - r.prefilled)
                done = self._prefill_chunk(r, take, clock)
                budget -= take
                prefill_tokens += take
                if done:
                    self.prefilling.pop(0)
                    self.running.append(r)
            prefill_ms = (time.perf_counter() - t0) * 1000.0
        finished += self._retire()                       # finished at prefill

        # 3-4) make room for one more token each, preempting if needed
        preempted, i = 0, 0
        while i < len(self.running):
            if self.kv.ensure_slot_for_next_token(self.running[i].rid):
                i += 1
                continue
            self._preempt_one()
            preempted += 1
            i = 0                                        # restart the pass; the pool changed

        decode_info, decode_ms, g0 = {}, 0.0, self.kv.gather_ms_total
        B = len(self.running)
        if self.running:
            t0 = time.perf_counter()
            decode_info = self._decode(self.running, clock)
            decode_ms = (time.perf_counter() - t0) * 1000.0

        # 5) retire
        finished += self._retire()

        st = self.kv.stats()
        self.iterations.append({
            "t_s": round(clock.now(), 5), "batch": B, "n_prefill": len(admitted),
            "prefilling": len(self.prefilling),
            "prefill_tokens": prefill_tokens, "prefill_ms": prefill_ms,
            "decode_ms": decode_ms, "gather_ms": self.kv.gather_ms_total - g0,
            "iteration_ms": (time.perf_counter() - it_t0) * 1000.0,
            "padding_waste": decode_info.get("padding_waste", 0.0),
            "context_max": decode_info.get("context_max", 0),
            "waiting": len(self.waiting), "preempted": preempted,
            "kv_blocks_used": st["kv_blocks_used"], "kv_slot_utilization": st["kv_slot_utilization"],
            "wasted_decode_tokens": 0,
        })
        return finished

    # -- offline replay ------------------------------------------------------
    def run(self, requests: List[Request]) -> dict:
        clock = _Clock()
        pending = sorted(requests, key=lambda x: x.arrival_s)
        self.waiting, self.running, self.prefilling, self.iterations = [], [], [], []
        n_done, pi = 0, 0
        while n_done < len(requests):
            now = clock.now()
            while pi < len(pending) and pending[pi].arrival_s <= now:
                self.add(pending[pi]); pi += 1
            if not self.has_work():
                clock.sleep_until(pending[pi].arrival_s)
                continue
            n_done += len(self.step(clock))
        return {"requests": requests, "iterations": self.iterations, "kv_stats": self.kv.stats(),
                "gather_ms_total": self.kv.gather_ms_total, "write_ms_total": self.kv.write_ms_total}


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def summarize(result: dict, policy: str, rate: float,
              slo_ttft_ms: float = 2000.0, slo_tpot_ms: float = 100.0,
              iso_ttft_fn: Optional[Callable[[int], float]] = None,
              iso_tpot_ms: Optional[float] = None) -> dict:
    import pandas as pd
    reqs = result["requests"]
    df = pd.DataFrame([r.metrics() for r in reqs])
    t_first_arrival = min(r.arrival_s for r in reqs)
    t_last = max(r.finish_s for r in reqs)
    makespan = max(t_last - t_first_arrival, 1e-9)
    out_tok = int(df.output_tokens.sum())
    good = df[(df.ttft_ms <= slo_ttft_ms) & ((df.tpot_ms <= slo_tpot_ms) | df.tpot_ms.isna())]
    it = pd.DataFrame(result["iterations"])
    s = {
        "policy": policy, "arrival_rate_rps": rate, "n_requests": len(reqs),
        "throughput_out_tok_s": out_tok / makespan,
        "throughput_req_s": len(reqs) / makespan,
        "goodput_req_s": len(good) / makespan,
        "slo": f"TTFT<={slo_ttft_ms:.0f}ms & TPOT<={slo_tpot_ms:.0f}ms",
    }
    for col in ("ttft_ms", "tpot_ms", "e2e_ms", "queue_ms", "itl_p99_ms"):
        s[f"{col}_p50"] = float(df[col].median())
        s[f"{col}_p99"] = float(df[col].quantile(0.99))
    if not it.empty:
        w = it[it.batch > 0]
        s["mean_batch"] = float(w.batch.mean()) if not w.empty else 0.0
        s["max_batch_seen"] = int(it.batch.max())
        s["mean_padding_waste"] = float(w.padding_waste.mean()) if not w.empty else 0.0
        s["wasted_decode_tokens"] = int(it.wasted_decode_tokens.sum())
        s["preemptions"] = int(it.get("preempted", pd.Series([0])).sum())
    if result.get("kv_stats"):
        s["kv_blocks_peak"] = result["kv_stats"]["kv_blocks_peak"]
        s["kv_blocks_total"] = result["kv_stats"]["kv_blocks_total"]
        s["gather_ms_total"] = result["gather_ms_total"]
        if "decode_ms" in it:
            s["gather_share_of_decode"] = result["gather_ms_total"] / max(it.decode_ms.sum(), 1e-9)
    if iso_ttft_fn is not None and iso_tpot_ms is not None:
        iso = df.prompt_tokens.map(iso_ttft_fn) + (df.output_tokens - 1) * iso_tpot_ms
        slow = df.e2e_ms / iso
        s["slowdown_median"] = float(slow.median())
        s["slowdown_max"] = float(slow.max())
    return s, df, it
