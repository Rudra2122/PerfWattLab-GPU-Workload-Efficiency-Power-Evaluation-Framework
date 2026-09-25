"""
Experiment 2 — KV-cache instrumentation and the paged allocator (README §6).

Parts (all written to results/v2/exp2_kv_cache/<model>/):

  A. analytic.csv       KV bytes/token (GQA vs MHA-equivalent), per-sequence
                        footprint, sequences per GiB.
  B. measured_bytes.csv Actual bytes held by HF past_key_values after prefill
                        vs the formula; torch allocator delta vs logical KV.
  C. growth.csv         During decode with HF DynamicCache: logical KV bytes,
                        allocated memory and step time vs context length.
  D. step_vs_context.csv  Decode step latency vs context, HF DynamicCache vs
                        our paged cache (gather time broken out).
  E. step_vs_batch.csv  Decode step latency vs batch at fixed context, with
                        KV bytes read per step and estimated KV bandwidth.
  F. max_concurrency.csv  Largest batch that can run a decode step at context L
                        before OOM (HF contiguous cache) vs analytic ceiling.
  G. capacity.csv       Paged vs contiguous max-length preallocation for a
                        heavy-tailed length mix under the measured free memory.

    python run_kv_cache.py
    python run_kv_cache.py --model tiny-random --quick      # CPU smoke test
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from perfwattlab.engine.kv_cache import (PagedKVCache, as_cache, capacity_comparison, legacy_past,
                                         past_nbytes)
from perfwattlab.engine.loadgen import make_workload
from perfwattlab.engine.model_utils import (DEFAULT_MODEL, device, load_causal_lm, model_spec,
                                            peak_bw_gbps, sync, synthetic_prompt_ids)
from perfwattlab.env import write_env

GiB = 2 ** 30


def mem():
    return torch.cuda.memory_allocated() if torch.cuda.is_available() else 0


def synthetic_past(spec, B, L, dtype, dev):
    shape = (B, spec.n_kv_heads, L, spec.head_dim)
    return tuple((torch.randn(shape, dtype=dtype, device=dev), torch.randn(shape, dtype=dtype, device=dev))
                 for _ in range(spec.n_layers))


@torch.inference_mode()
def decode_step_time(model, past, B, L, iters=5):
    dev = device()
    inp = torch.full((B, 1), 100, device=dev)
    mask = torch.ones((B, L + 1), dtype=torch.long, device=dev)
    pos = torch.full((B, 1), L, device=dev)
    times = []
    for _ in range(iters):
        sync(); t0 = time.perf_counter()
        model(input_ids=inp, attention_mask=mask, position_ids=pos,
              past_key_values=as_cache(past), use_cache=True)
        sync(); times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def classify_failure(e: BaseException) -> str:
    """'oom' for out-of-memory, 'launch_limit' for kernel-configuration limits
    (e.g. attention grid too large at huge batch), else 'error'."""
    msg = str(e).lower()
    if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in msg:
        return "oom"
    if "invalid configuration" in msg or "invalid argument" in msg or "too large" in msg:
        return "launch_limit"
    return "error"


def max_batch_search(try_batch, cap: int = 8192):
    """
    Largest batch b in [1, cap] for which try_batch(b) succeeds, assuming
    success is monotone. Doubles from 1 until the first failure, then binary
    searches in between. Returns (max_ok, first_failure_kind or 'cap').
    Starting small matters: a KV-only ceiling can be ~10^6 for tiny models,
    and launching that directly hits CUDA grid limits before memory limits.
    """
    ok, bad, kind = 0, None, "cap"
    b = 1
    while b <= cap:
        res = try_batch(b)
        if res is True:
            ok, b = b, b * 2
        else:
            bad, kind = b, res
            break
    if bad is None:
        return ok, "cap"
    lo, hi = ok, bad - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        res = try_batch(mid)
        if res is True:
            lo = mid
        else:
            hi = mid - 1
    return lo, kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--contexts", default="128,256,512,1024,1536,2000")
    ap.add_argument("--batches", default="1,2,4,8,16,32,64")
    ap.add_argument("--batch-ctx", type=int, default=1024)
    ap.add_argument("--oom-contexts", default="512,1024,2000")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-batch-cap", type=int, default=8192)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out-dir", default="results/v2/exp2_kv_cache")
    args = ap.parse_args()
    if args.quick:
        args.contexts, args.batches, args.oom_contexts = "64,128,256", "1,4", "128"

    model, tok = load_causal_lm(args.model)
    spec = model_spec(model, args.model)
    dt = next(model.parameters()).dtype
    dev = device()
    out = Path(args.out_dir) / args.model.split("/")[-1]
    write_env(out, vars(args))
    ctxs = [int(x) for x in args.contexts.split(",") if int(x) < spec.max_positions]
    md = [f"# Experiment 2 — KV cache — {spec.name}", ""]

    # ---------------- A. analytic
    per, mha = spec.kv_bytes_per_token(), spec.kv_bytes_mha_equivalent_per_token()
    A = []
    for L in sorted({512, 2048, spec.max_positions}):
        A.append({"context": L, "gqa_kv_heads": spec.n_kv_heads, "mha_heads": spec.n_heads,
                  "gqa_bytes_per_token": per, "mha_bytes_per_token": mha,
                  "gqa_mib_per_seq": per * L / 2**20, "mha_mib_per_seq": mha * L / 2**20,
                  "gqa_seqs_per_gib": GiB / (per * L), "mha_seqs_per_gib": GiB / (mha * L)})
    A = pd.DataFrame(A); A.to_csv(out / "analytic.csv", index=False)
    md += ["## A. Analytical footprint", "",
           f"KV bytes/token = 2 × {spec.n_layers} layers × {spec.n_kv_heads} KV heads × "
           f"{spec.head_dim} head_dim × {spec.dtype_bytes} B = **{per:,} B** "
           f"(MHA-equivalent with {spec.n_heads} heads: {mha:,} B, {mha / per:.0f}×)", "",
           A.round(2).to_markdown(index=False), ""]

    # ---------------- B. measured bytes vs formula
    B_rows = []
    with torch.inference_mode():
        for L in ctxs:
            ids = torch.tensor(synthetic_prompt_ids(tok, L, 1), device=dev)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            m0 = mem()
            o = model(input_ids=ids, use_cache=True)
            past = legacy_past(o.past_key_values)
            del o
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            m1 = mem()
            nb = past_nbytes(past)
            B_rows.append({"context": L, "formula_bytes": per * L, "measured_past_bytes": nb,
                           "match": nb == per * L, "allocator_delta_bytes": m1 - m0,
                           "allocator_overhead_ratio": (m1 - m0) / nb if torch.cuda.is_available() else np.nan})
            del past
    Bdf = pd.DataFrame(B_rows); Bdf.to_csv(out / "measured_bytes.csv", index=False)
    md += ["## B. Measured past_key_values bytes vs formula", "", Bdf.to_markdown(index=False), ""]

    # ---------------- C. growth during decode with HF DynamicCache
    C = []
    P0, steps = ctxs[0], min(max(ctxs) - ctxs[0], 512)
    with torch.inference_mode():
        ids = torch.tensor(synthetic_prompt_ids(tok, P0, 1), device=dev)
        o = model(input_ids=ids, use_cache=True)
        past = o.past_key_values
        nxt = o.logits[:, -1:].argmax(-1)
        for s in range(steps):
            sync(); t0 = time.perf_counter()
            o = model(input_ids=nxt, past_key_values=past, use_cache=True)
            sync(); dt_ms = (time.perf_counter() - t0) * 1000
            past = o.past_key_values; nxt = o.logits[:, -1:].argmax(-1)
            if s % 16 == 0:
                C.append({"context": P0 + s + 1, "step_ms": dt_ms, "logical_kv_bytes": past_nbytes(past),
                          "allocated_bytes": mem()})
        del past, o
    Cdf = pd.DataFrame(C); Cdf.to_csv(out / "growth.csv", index=False)
    if len(Cdf) > 2:
        slope = np.polyfit(Cdf.context, Cdf.step_ms, 1)[0]
        md += ["## C. Decode with HF DynamicCache", "",
               f"Step time grows ≈ {slope * 1000:.2f} µs per extra context token (linear fit, batch 1). "
               "Logical KV grows exactly by bytes/token per step; see growth.csv for the allocator view.", ""]

    # ---------------- D. step time vs context: HF contiguous vs paged (gather)
    D = []
    for L in ctxs:
        past = synthetic_past(spec, 1, L, dt, dev)
        hf_ms = decode_step_time(model, past, 1, L)
        del past
        nblk = 2 * ((L + 1) // args.block_size + 2)
        kv = PagedKVCache(spec, nblk, args.block_size, dtype=dt, device=dev)
        kv.allocate("r", L + 1); kv.seq_lens["r"] = L
        g_times = []
        for _ in range(5):
            g0 = kv.gather_ms_total
            gp, gm, _ = kv.gather(["r"])
            g_times.append(kv.gather_ms_total - g0)
        paged_ms = decode_step_time(model, gp, 1, L) + float(np.median(g_times))
        D.append({"context": L, "hf_step_ms": hf_ms, "paged_gather_ms": float(np.median(g_times)),
                  "paged_step_ms_incl_gather": paged_ms,
                  "gather_share": float(np.median(g_times)) / paged_ms})
        del kv, gp
    Ddf = pd.DataFrame(D); Ddf.to_csv(out / "step_vs_context.csv", index=False)
    md += ["## D. Decode step vs context — HF contiguous vs paged + gather (batch 1)", "",
           Ddf.round(3).to_markdown(index=False), ""]

    # ---------------- E. step time vs batch at fixed context
    E, bw = [], peak_bw_gbps()
    L = min(args.batch_ctx, spec.max_positions - 1)
    for b in [int(x) for x in args.batches.split(",")]:
        try:
            past = synthetic_past(spec, b, L, dt, dev)
            ms = decode_step_time(model, past, b, L)
            del past
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            E.append({"batch": b, "context": L, "status": "OOM"}); break
        kvb = b * L * per
        row = {"batch": b, "context": L, "status": "ok", "step_ms": ms, "tok_s": b / (ms / 1000),
               "kv_bytes_read_mb": kvb / 1e6, "weight_bytes_mb": spec.weight_bytes / 1e6,
               "kv_share": kvb / (kvb + spec.weight_bytes),
               "est_total_gbps": (kvb + spec.weight_bytes) / (ms / 1000) / 1e9}
        if bw:
            row["pct_peak_bw"] = 100 * row["est_total_gbps"] / bw
        E.append(row)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    Edf = pd.DataFrame(E); Edf.to_csv(out / "step_vs_batch.csv", index=False)
    md += [f"## E. Decode step vs batch (context {L})", "", Edf.round(3).to_markdown(index=False), ""]

    # ---------------- F. max concurrent sequences before OOM
    F = []
    if torch.cuda.is_available():
        for L in [int(x) for x in args.oom_contexts.split(",") if int(x) < spec.max_positions]:
            torch.cuda.empty_cache()
            free = torch.cuda.mem_get_info()[0]
            analytic = int(free // (L * per))

            def try_batch(b, L=L):
                past = None
                try:
                    past = synthetic_past(spec, b, L, dt, dev)
                    decode_step_time(model, past, b, L, iters=1)
                    return True
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    kind = classify_failure(e)
                    if kind == "error":
                        raise
                    return kind
                finally:
                    del past
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

            best, limit = max_batch_search(try_batch, cap=args.max_batch_cap)
            F.append({"context": L, "free_mem_gib_before": free / GiB,
                      "analytic_kv_only_ceiling": analytic, "measured_max_batch": best,
                      "limited_by": limit,
                      "measured_vs_analytic": best / analytic if analytic else np.nan})
            print(f"context {L}: max decode batch {best} (limited by {limit}; KV-only ceiling {analytic})")
    Fdf = pd.DataFrame(F); Fdf.to_csv(out / "max_concurrency.csv", index=False)
    if not Fdf.empty:
        md += ["## F. Max concurrent sequences (HF contiguous cache, one decode step)", "",
               Fdf.round(3).to_markdown(index=False), "",
               "`limited_by`: `oom` = ran out of memory; `launch_limit` = a kernel's launch configuration "
               "exceeded CUDA limits first; `cap` = hit --max-batch-cap without failing. "
               "Below the KV-only ceiling, the gap is activation/workspace memory plus the copy "
               "DynamicCache makes when it appends the new token (old and new cache coexist).", ""]

    # ---------------- G. paged vs contiguous capacity
    budget = int(torch.cuda.mem_get_info()[0] * 0.8) if torch.cuda.is_available() else 256 * 2**20
    reqs = make_workload(20000, 1.0, "heavy_tail", spec.vocab, spec.max_positions, seed=1)
    lengths = [len(r.prompt_ids) + r.max_new_tokens for r in reqs]
    G = pd.DataFrame([capacity_comparison(spec, budget, lengths, max_len=spec.max_positions,
                                          block_size=args.block_size)])
    G.to_csv(out / "capacity.csv", index=False)
    md += ["## G. Paged vs contiguous max-length preallocation (heavy-tailed lengths)", "",
           G.round(3).T.rename(columns={0: "value"}).to_markdown(), ""]

    (out / "report.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
