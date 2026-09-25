"""
run_attention_check.py — Which attention kernel does prefill use, and what
actually limits prefill memory? (README §5.4)

Part 1: profiles one prefill through the same code path as Experiment 1
        (explicit attention mask + position ids) and lists the attention-related
        CUDA kernels: FlashAttention (flash_fwd), memory-efficient (fmha /
        efficient_attention), or the unfused math path (separate bmm + softmax).
Part 2: for each (batch, prompt) config, peak GPU memory and TTFT for
          full        one-shot prefill, logits for every position
          last        one-shot prefill, last-position logits only
          chunked     prefill in chunks of --chunk tokens, last-position logits
        Records OOM as a result. If memory scales with tokens in flight
        (batch × prompt), chunking should make the failing configs fit.

    python run_attention_check.py --model Qwen/Qwen2.5-7B-Instruct
    python run_attention_check.py --model tiny-random --configs 2x32,4x64 --chunk 16   # CPU smoke
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from perfwattlab.engine.generate_loop import generate
from perfwattlab.engine.model_utils import load_causal_lm, model_spec, synthetic_prompt_ids
from perfwattlab.env import write_env

ATTN_PATTERNS = ("flash", "fmha", "efficient_attention", "attention", "softmax", "bmm")


def attention_kernels(model, tok, B, P):
    from torch.profiler import ProfilerActivity, profile
    if not torch.cuda.is_available():
        return pd.DataFrame()
    ids = synthetic_prompt_ids(tok, P, B)
    generate(model, ids, 1, pad_id=tok.pad_token_id)            # warmup
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        generate(model, ids, 1, pad_id=tok.pad_token_id)
        torch.cuda.synchronize()
    rows = {}
    for e in prof.events():
        if e.device_type != torch.autograd.DeviceType.CUDA:
            continue
        n = e.name
        if any(p in n.lower() for p in ATTN_PATTERNS):
            r = rows.setdefault(n[:110], {"kernel": n[:110], "calls": 0, "total_us": 0.0})
            r["calls"] += 1
            r["total_us"] += e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
    return pd.DataFrame(list(rows.values())).sort_values("total_us", ascending=False) if rows else pd.DataFrame()


def prefill_peak(model, tok, B, P, mode, chunk):
    ids = synthetic_prompt_ids(tok, P, B, seed=1)
    kw = {"full": {}, "last": {"last_token_logits": True},
          "chunked": {"last_token_logits": True, "prefill_chunk": chunk}}[mode]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    try:
        t0 = time.perf_counter()
        res = generate(model, ids, 1, pad_id=tok.pad_token_id, **kw)
        ttft = res.prefill_ms
        status = "ok"
    except torch.cuda.OutOfMemoryError:
        ttft, status = float("nan"), "OOM"
    peak = (torch.cuda.max_memory_allocated() - base) / 2**30 if torch.cuda.is_available() else float("nan")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"batch": B, "prompt": P, "tokens_in_flight": B * P if mode != "chunked" else B * chunk,
            "mode": mode, "status": status, "ttft_ms": ttft,
            "peak_extra_gib": peak if status == "ok" else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--configs", default="4x2048,16x4096,32x2048,32x4096,16x8192,32x8192")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--profile-config", default="4x2048")
    ap.add_argument("--out-dir", default="results/v2/exp1_attention_check")
    args = ap.parse_args()
    model, tok = load_causal_lm(args.model)
    spec = model_spec(model, args.model)
    out = Path(args.out_dir) / args.model.split("/")[-1]
    write_env(out, vars(args))
    md = [f"# Prefill attention backend and memory — {spec.name}", ""]

    pb, pp = (int(x) for x in args.profile_config.split("x"))
    K = attention_kernels(model, tok, pb, pp)
    if not K.empty:
        K.to_csv(out / "attention_kernels.csv", index=False)
        md += [f"## Part 1 — attention-related kernels in one prefill ({pb} × {pp}, Exp 1 code path)", "",
               K.round(1).to_markdown(index=False), ""]
        names = " ".join(K.kernel).lower()
        backend = ("FlashAttention" if "flash" in names else
                   "memory-efficient" if ("fmha" in names or "efficient" in names) else "math (unfused)")
        md += [f"**Backend: {backend}.** Layers: {spec.n_layers}.", ""]
        print(f"prefill attention backend: {backend}")

    rows = []
    for cfg in args.configs.split(","):
        B, P = (int(x) for x in cfg.split("x"))
        if P + 1 > spec.max_positions:
            continue
        for mode in ("full", "last", "chunked"):
            r = prefill_peak(model, tok, B, P, mode, args.chunk)
            rows.append(r)
            print(f"{B:3d} × {P:5d} {mode:8s} {r['status']:4s} TTFT {r['ttft_ms']:9.1f} ms  "
                  f"peak +{r['peak_extra_gib']:.2f} GiB")
    D = pd.DataFrame(rows)
    D.to_csv(out / "prefill_memory.csv", index=False)
    md += [f"## Part 2 — prefill peak memory (above weights) and TTFT; chunk = {args.chunk} tokens", "",
           D.round(2).to_markdown(index=False), "",
           "`tokens_in_flight` = tokens processed by one forward pass (batch × prompt, or batch × chunk).", ""]
    (out / "report.md").write_text("\n".join(md))
    print(f"\nWrote {out}/report.md")


if __name__ == "__main__":
    main()
