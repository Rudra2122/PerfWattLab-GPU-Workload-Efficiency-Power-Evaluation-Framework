"""
generate_loop.py — An explicit prefill + decode loop (README §5.3).

Why not model.generate()?  generate() hides the phase boundary. This loop
owns it, so we can time prefill and every decode step separately, force
exact output lengths, and later swap in our own KV cache and scheduler.

Timing model
------------
* Every decode step ends with `next_tokens.tolist()`, which copies the new
  token ids to the host and therefore synchronizes with the GPU. A token
  "exists" for a client at that moment, so host timestamps taken right after
  it are the correct basis for TTFT and ITL.
* prefill_ms = submit → first token on host (includes the prefill forward,
  the argmax over logits, and the device→host copy). For a batch submitted
  together, this is also each request's TTFT.
* decode step_ms[i] = gap between token i and token i+1 for the whole batch.

Decoding is greedy. Output length is forced (ignore_eos=True) by default so
configs are compared on identical token counts.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from .model_utils import last_token_logits_kwargs, sync


def _nvtx(name: str, enabled: bool):
    if enabled and torch.cuda.is_available():
        return torch.cuda.nvtx.range(name)
    return contextlib.nullcontext()


def grad_context(kind: str):
    """'inference_mode' | 'no_grad' — used by the Experiment 0 H1 ablation."""
    if kind == "inference_mode":
        return torch.inference_mode()
    if kind == "no_grad":
        return torch.no_grad()
    raise ValueError(kind)


def left_pad(batch_ids: List[List[int]], pad_id: int, dev: str):
    lens = [len(x) for x in batch_ids]
    L = max(lens)
    ids = torch.full((len(batch_ids), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch_ids), L), dtype=torch.long)
    for i, x in enumerate(batch_ids):
        ids[i, L - len(x):] = torch.tensor(x, dtype=torch.long)
        mask[i, L - len(x):] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return ids.to(dev), mask.to(dev), pos.to(dev), lens


@dataclass
class GenResult:
    output_ids: List[List[int]]
    prompt_lens: List[int]
    t_submit: float
    t_first: float
    token_times: List[float]           # host time each decode step's tokens landed (batch-wide)
    finish_step: List[int]             # index into token_times of each sequence's last token
    step_ms: List[float] = field(default_factory=list)

    @property
    def batch(self) -> int:
        return len(self.output_ids)

    @property
    def prefill_ms(self) -> float:
        return (self.t_first - self.t_submit) * 1000.0

    def per_request_metrics(self) -> List[dict]:
        rows = []
        for i in range(self.batch):
            times = self.token_times[: self.finish_step[i] + 1]
            n_out = len(times)
            e2e = (times[-1] - self.t_submit) * 1000.0
            ttft = (times[0] - self.t_submit) * 1000.0
            itl = np.diff(times) * 1000.0 if n_out > 1 else np.array([])
            rows.append({
                "request": i, "prompt_tokens": self.prompt_lens[i], "output_tokens": n_out,
                "ttft_ms": ttft, "e2e_ms": e2e,
                "tpot_ms": (e2e - ttft) / (n_out - 1) if n_out > 1 else float("nan"),
                "itl_p50_ms": float(np.percentile(itl, 50)) if itl.size else float("nan"),
                "itl_p95_ms": float(np.percentile(itl, 95)) if itl.size else float("nan"),
                "itl_p99_ms": float(np.percentile(itl, 99)) if itl.size else float("nan"),
            })
        return rows

    def batch_metrics(self) -> dict:
        step = np.asarray(self.step_ms)
        total_out = sum(f + 1 for f in self.finish_step)
        decode_s = (self.token_times[-1] - self.t_first)
        prompt_tokens = sum(self.prompt_lens)
        return {
            "batch": self.batch,
            "prompt_tokens_total": prompt_tokens,
            "output_tokens_total": total_out,
            "ttft_ms": self.prefill_ms,
            "prefill_ms": self.prefill_ms,
            "prefill_tok_s": prompt_tokens / (self.prefill_ms / 1000.0),
            "itl_p50_ms": float(np.percentile(step, 50)) if step.size else float("nan"),
            "itl_p95_ms": float(np.percentile(step, 95)) if step.size else float("nan"),
            "itl_p99_ms": float(np.percentile(step, 99)) if step.size else float("nan"),
            "tpot_ms": float(step.mean()) if step.size else float("nan"),
            # decode tok/s counts tokens produced after the first one
            "decode_tok_s": (total_out - self.batch) / decode_s if decode_s > 0 else float("nan"),
            "e2e_ms": (self.token_times[-1] - self.t_submit) * 1000.0,
        }


def generate(model, batch_ids: List[List[int]], max_new_tokens: int,
             pad_id: int = 0, eos_id: Optional[int] = None, ignore_eos: bool = True,
             nvtx: bool = False, grad_ctx: str = "inference_mode",
             last_token_logits: bool = False, prefill_chunk: Optional[int] = None) -> GenResult:
    """
    Greedy batched generation with explicit prefill and decode phases.
    last_token_logits: compute prefill logits only for the last position
    (see model_utils.last_token_logits_kwargs).
    prefill_chunk: process the prompt in chunks of this many positions, each
    chunk attending to the KV cache written by the previous ones. Bounds the
    activation memory of prefill to batch × chunk tokens (README §5.4).
    """
    extra = last_token_logits_kwargs(model) if last_token_logits else {}
    with grad_context(grad_ctx):
        dev = next(model.parameters()).device
        ids, mask, pos, lens = left_pad(batch_ids, pad_id, dev)
        B = ids.shape[0]
        sync()
        t_submit = time.perf_counter()

        with _nvtx("prefill", nvtx):
            L = ids.shape[1]
            if prefill_chunk and prefill_chunk < L:
                # every chunk except the last only needs its KV, not its logits
                keep1 = last_token_logits_kwargs(model)
                past = None
                for s0 in range(0, L, prefill_chunk):
                    s1 = min(L, s0 + prefill_chunk)
                    out = model(input_ids=ids[:, s0:s1], attention_mask=mask[:, :s1],
                                position_ids=pos[:, s0:s1], past_key_values=past, use_cache=True,
                                **(keep1 if s1 < L else extra))
                    past = out.past_key_values
            else:
                out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True, **extra)
            nxt = out.logits[:, -1, :].argmax(-1)
            first = nxt.tolist()                       # host sync
        t_first = time.perf_counter()
        past = out.past_key_values

        outputs = [[t] for t in first]
        token_times = [t_first]
        finished = [False] * B
        finish_step = [max_new_tokens - 1] * B
        if not ignore_eos and eos_id is not None:
            for i, t in enumerate(first):
                if t == eos_id:
                    finished[i], finish_step[i] = True, 0

        cur_pos = pos[:, -1]
        for step in range(1, max_new_tokens):
            if all(finished):
                break
            mask = torch.cat([mask, mask.new_ones((B, 1))], dim=1)
            cur_pos = cur_pos + 1
            with _nvtx("decode_step", nvtx):
                out = model(input_ids=nxt[:, None], attention_mask=mask,
                            position_ids=cur_pos[:, None], past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(-1)
                toks = nxt.tolist()                    # host sync
            token_times.append(time.perf_counter())
            for i, t in enumerate(toks):
                if finished[i]:
                    continue
                outputs[i].append(t)
                if not ignore_eos and eos_id is not None and t == eos_id:
                    finished[i], finish_step[i] = True, step
        # sequences that never hit EOS finished at the last step run
        last = len(token_times) - 1
        finish_step = [min(f, last) for f in finish_step]

    step_ms = (np.diff(token_times) * 1000.0).tolist()
    return GenResult(outputs, lens, t_submit, t_first, token_times, finish_step, step_ms)
