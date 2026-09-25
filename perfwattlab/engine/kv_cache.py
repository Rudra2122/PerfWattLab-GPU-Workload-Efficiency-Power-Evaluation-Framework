"""
kv_cache.py — A simplified paged KV-cache manager (README §6.3).

Memory layout
-------------
One preallocated pool per model:

    pool: [n_layers, 2 (K/V), num_slots, n_kv_heads, head_dim]
    num_slots = num_blocks * block_size

A *block* is `block_size` consecutive slots. Slot id = block_id * block_size + offset.
Each request owns an ordered list of blocks (its block table):

    block_table["A"] = [2, 7, 11]      # tokens 0-15 in block 2, 16-31 in 7, 32-47 in 11
    seq_lens["A"]    = 40              # block 11 is 8/16 full → 8 slots of internal fragmentation

Blocks are taken from `free_blocks` only when a sequence crosses a block
boundary, and returned when the request finishes or is preempted. There is
no per-request max-length preallocation, which is where the memory savings
over contiguous allocation come from.

Scope (stated honestly)
-----------------------
The model's stock attention expects contiguous K/V tensors, so before each
forward pass `gather()` copies the active sequences' blocks into a padded,
left-aligned contiguous buffer. That copy costs HBM bandwidth and is timed
separately (`gather_ms`). A paged-attention kernel that reads directly from
the block table would remove it; that is out of scope for v2.

K is cached *after* RoPE (as Hugging Face does), so left-padding the
gathered buffer does not disturb positions: the new token's position_ids
are passed explicitly by the scheduler.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import torch

from .model_utils import ModelSpec, sync


def legacy_past(past):
    """
    Normalize any Hugging Face KV cache to a tuple of (K, V) per layer,
    each [batch, n_kv_heads, seq, head_dim]. Works across transformers versions:
      5.x      DynamicCache.layers[i].keys / .values
      4.4x-4.5x DynamicCache.key_cache / value_cache lists, or to_legacy_cache()
      older    already a tuple
    """
    if isinstance(past, (tuple, list)):
        return tuple(past)
    if hasattr(past, "layers"):
        return tuple((l.keys, l.values) for l in past.layers)
    if hasattr(past, "to_legacy_cache"):
        return past.to_legacy_cache()
    if hasattr(past, "key_cache"):
        return tuple(zip(past.key_cache, past.value_cache))
    raise TypeError(f"unsupported cache type {type(past)}")


def as_cache(past_tuple):
    """
    Wrap (K, V)-per-layer tensors in a fresh DynamicCache for a forward pass.
    Recent transformers require a Cache object; update() works on every version
    since 4.36. A new object per call matters: the model appends to the cache in
    place, so reusing one would silently grow the context each call.
    """
    from transformers import DynamicCache
    c = DynamicCache()
    for i, (k, v) in enumerate(past_tuple):
        c.update(k, v, i)
    return c


def past_nbytes(past) -> int:
    """Actual bytes held by a HF past_key_values."""
    past = legacy_past(past)
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in past)


class OutOfBlocks(RuntimeError):
    pass


class PagedKVCache:
    def __init__(self, spec: ModelSpec, num_blocks: int, block_size: int = 16,
                 dtype: Optional[torch.dtype] = None, device: Optional[str] = None):
        self.spec = spec
        self.block_size = block_size
        self.num_blocks = num_blocks
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dt = dtype or (torch.float16 if dev == "cuda" else torch.float32)
        self.pool = torch.zeros(
            (spec.n_layers, 2, num_blocks * block_size, spec.n_kv_heads, spec.head_dim),
            dtype=dt, device=dev)
        self.free_blocks: deque = deque(range(num_blocks))
        self.block_table: Dict[Hashable, List[int]] = {}
        self.seq_lens: Dict[Hashable, int] = {}
        self.gather_ms_total = 0.0
        self.write_ms_total = 0.0
        self.peak_used_blocks = 0

    def reset(self):
        """Free every block and clear counters, keeping the pool allocated.
        Reusing one pool across runs avoids holding two multi-GB pools at once."""
        self.free_blocks = deque(range(self.num_blocks))
        self.block_table.clear()
        self.seq_lens.clear()
        self.gather_ms_total = 0.0
        self.write_ms_total = 0.0
        self.peak_used_blocks = 0

    # ------------------------------------------------------------------ sizing
    @staticmethod
    def blocks_for_budget(spec: ModelSpec, budget_bytes: int, block_size: int = 16) -> int:
        return int(budget_bytes // (block_size * spec.kv_bytes_per_token()))

    @property
    def bytes_per_block(self) -> int:
        return self.block_size * self.spec.kv_bytes_per_token()

    @property
    def pool_bytes(self) -> int:
        return self.pool.numel() * self.pool.element_size()

    def blocks_needed(self, n_tokens: int) -> int:
        return math.ceil(n_tokens / self.block_size)

    def num_free(self) -> int:
        return len(self.free_blocks)

    # ------------------------------------------------------------- allocation
    def can_allocate(self, n_tokens: int, reserve_blocks: int = 0) -> bool:
        return self.blocks_needed(n_tokens) + reserve_blocks <= self.num_free()

    def allocate(self, req: Hashable, n_tokens: int):
        need = self.blocks_needed(n_tokens)
        if need > self.num_free():
            raise OutOfBlocks(f"need {need}, free {self.num_free()}")
        self.block_table[req] = [self.free_blocks.popleft() for _ in range(need)]
        self.seq_lens[req] = 0
        self._track_peak()

    def ensure_slot_for_next_token(self, req: Hashable) -> bool:
        """Make sure there's room for one more token. False if out of blocks."""
        n = self.seq_lens[req]
        if n < len(self.block_table[req]) * self.block_size:
            return True
        if not self.free_blocks:
            return False
        self.block_table[req].append(self.free_blocks.popleft())
        self._track_peak()
        return True

    def free(self, req: Hashable):
        for b in self.block_table.pop(req, []):
            self.free_blocks.append(b)
        self.seq_lens.pop(req, None)

    def _track_peak(self):
        self.peak_used_blocks = max(self.peak_used_blocks, self.num_blocks - self.num_free())

    # ------------------------------------------------------------ slot math
    def _slot(self, req: Hashable, pos: int) -> int:
        blocks = self.block_table[req]
        return blocks[pos // self.block_size] * self.block_size + pos % self.block_size

    def slots(self, req: Hashable, start: int, end: int) -> List[int]:
        return [self._slot(req, p) for p in range(start, end)]

    # ---------------------------------------------------------------- writes
    def write_prefill(self, req: Hashable, past, batch_index: int, n_tokens: int, start: int = 0):
        """
        Copy the last `n_tokens` positions of row `batch_index` of a HF past
        (left-padded batch) into this request's blocks, at token positions
        start .. start+n_tokens-1 (start > 0 for chunked prefill).
        """
        t0 = time.perf_counter()
        past = legacy_past(past)
        slots = torch.tensor(self.slots(req, start, start + n_tokens), device=self.pool.device)
        # [L, 2, n_kv, T, hd] -> [L, 2, T, n_kv, hd]
        kv = torch.stack([torch.stack((k[batch_index, :, -n_tokens:, :],
                                       v[batch_index, :, -n_tokens:, :])) for k, v in past])
        self.pool[:, :, slots] = kv.transpose(2, 3).to(self.pool.dtype)
        self.seq_lens[req] = start + n_tokens
        self.write_ms_total += (time.perf_counter() - t0) * 1000.0

    def write_decode(self, reqs: Sequence[Hashable], past):
        """Append the newest position of every row of `past` to reqs (same order)."""
        t0 = time.perf_counter()
        past = legacy_past(past)
        slots = torch.tensor([self._slot(r, self.seq_lens[r]) for r in reqs],
                             device=self.pool.device)
        # [L, 2, B, n_kv, hd]
        kv = torch.stack([torch.stack((k[:, :, -1, :], v[:, :, -1, :])) for k, v in past])
        self.pool[:, :, slots] = kv.to(self.pool.dtype)
        for r in reqs:
            self.seq_lens[r] += 1
        self.write_ms_total += (time.perf_counter() - t0) * 1000.0

    # ---------------------------------------------------------------- gather
    def gather(self, reqs: Sequence[Hashable]) -> Tuple[tuple, torch.Tensor, int]:
        """
        Build a left-padded contiguous legacy past for `reqs`.
        Returns (past_tuple, attention_mask[B, Lmax] (long), Lmax).
        Padded positions point at slot 0 and are masked out.
        """
        t0 = time.perf_counter()
        lens = [self.seq_lens[r] for r in reqs]
        L = max(lens)
        B = len(reqs)
        idx = torch.zeros((B, L), dtype=torch.long)
        mask = torch.zeros((B, L), dtype=torch.long)
        for i, r in enumerate(reqs):
            n = lens[i]
            idx[i, L - n:] = torch.tensor(self.slots(r, 0, n), dtype=torch.long)
            mask[i, L - n:] = 1
        dev = self.pool.device
        idx, mask = idx.to(dev), mask.to(dev)
        g = self.pool.index_select(2, idx.flatten())                     # [Ly, 2, B*L, kvh, hd]
        g = g.view(self.spec.n_layers, 2, B, L, self.spec.n_kv_heads, self.spec.head_dim)
        g = g.transpose(3, 4).contiguous()                               # [Ly, 2, B, kvh, L, hd]
        past = tuple((g[l, 0], g[l, 1]) for l in range(self.spec.n_layers))
        sync()
        self.gather_ms_total += (time.perf_counter() - t0) * 1000.0
        return past, mask, L

    # ----------------------------------------------------------------- stats
    def stats(self) -> dict:
        used = self.num_blocks - self.num_free()
        tokens = sum(self.seq_lens.values())
        alloc_slots = used * self.block_size
        return {
            "kv_blocks_total": self.num_blocks,
            "kv_blocks_used": used,
            "kv_blocks_peak": self.peak_used_blocks,
            "kv_tokens": tokens,
            "kv_slots_allocated": alloc_slots,
            "kv_internal_frag_slots": alloc_slots - tokens,
            "kv_slot_utilization": tokens / alloc_slots if alloc_slots else 0.0,
            "kv_bytes_logical": tokens * self.spec.kv_bytes_per_token(),
            "kv_bytes_allocated": used * self.bytes_per_block,
            "active_sequences": len(self.block_table),
        }


# ---------------------------------------------------------------------------
# Capacity simulation: paged vs contiguous max-length preallocation
# ---------------------------------------------------------------------------

def capacity_comparison(spec: ModelSpec, budget_bytes: int, lengths: Sequence[int],
                        max_len: int, block_size: int = 16) -> dict:
    """
    How many of `lengths` (actual final sequence lengths, in arrival order)
    fit concurrently under a KV budget when
      (a) every request reserves max_len contiguously (naive serving), vs
      (b) requests take ceil(len / block_size) blocks (paged).
    Pure bookkeeping — no GPU needed.
    """
    per_tok = spec.kv_bytes_per_token()
    contiguous_slots = budget_bytes // (max_len * per_tok)
    n_blocks = budget_bytes // (block_size * per_tok)
    used, fit, frag = 0, 0, 0
    for n in lengths:
        need = math.ceil(n / block_size)
        if used + need > n_blocks:
            break
        used += need
        fit += 1
        frag += need * block_size - n
    fit_tokens = sum(lengths[:fit])
    return {
        "budget_gib": round(budget_bytes / 2**30, 3),
        "kv_bytes_per_token": per_tok,
        "max_len": max_len,
        "block_size": block_size,
        "contiguous_max_concurrent": int(min(contiguous_slots, len(lengths))),
        "contiguous_utilization": (sum(lengths[:int(contiguous_slots)]) /
                                   (min(contiguous_slots, len(lengths)) * max_len)) if contiguous_slots else 0.0,
        "paged_max_concurrent": fit,
        "paged_utilization": fit_tokens / (used * block_size) if used else 0.0,
        "paged_internal_frag_slots": frag,
    }
