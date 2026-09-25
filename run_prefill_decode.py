"""
Experiment 1 — Prefill vs decode separation (README §5).

Sweeps prompt length × output length × batch size with the explicit
generate loop and reports, per point:

  TTFT, prefill latency, prefill tok/s, prefill achieved TFLOP/s,
  ITL p50/p95/p99, TPOT, decode tok/s, estimated decode bytes/step and GB/s
  (weights + KV), % of peak bandwidth, NVML GPU util during decode,
  energy per output token, peak allocated memory.

Points whose prompt+output exceed the model's context window are recorded as
"context_limit"; points that run out of memory are recorded as "OOM". Both are
results, not errors.

    python run_prefill_decode.py --quick
    python run_prefill_decode.py                                   # TinyLlama, up to 2K
    python run_prefill_decode.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt-lens 4096,8192
    python run_prefill_decode.py --triton-rmsnorm --tag triton     # §8 end-to-end
    python run_prefill_decode.py --model Qwen/Qwen2.5-7B-Instruct --last-token-logits --tag lastlogits
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from perfwattlab.energy import EnergyMeter
from perfwattlab.engine.generate_loop import generate
from perfwattlab.engine.model_utils import (DEFAULT_MODEL, load_causal_lm, model_spec,
                                            peak_bw_gbps, synthetic_prompt_ids)
from perfwattlab.env import write_env


def ints(s):
    return [int(x) for x in s.split(",") if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt-lens", default="128,512,2048,4096,8192")
    ap.add_argument("--out-lens", default="32,128,256")
    ap.add_argument("--batches", default="1,2,4,8,16")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="prompt 128,512 × out 32 × batch 1,4, 2 repeats")
    ap.add_argument("--nvtx", action="store_true", help="emit NVTX ranges for Nsight Systems")
    ap.add_argument("--triton-rmsnorm", action="store_true", help="swap in the §8 Triton RMSNorm")
    ap.add_argument("--attn", default=None, help="attn_implementation: sdpa | eager")
    ap.add_argument("--prefill-chunk", type=int, default=None,
                    help="prefill in chunks of N positions (bounds prefill activation memory)")
    ap.add_argument("--last-token-logits", action="store_true",
                    help="prefill computes logits for the last position only (§5 optimization)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default="results/v2/exp1_prefill_decode")
    args = ap.parse_args()
    if args.quick:
        args.prompt_lens, args.out_lens, args.batches, args.repeats = "128,512", "32", "1,4", 2

    model, tok = load_causal_lm(args.model, attn_implementation=args.attn)
    spec = model_spec(model, args.model)
    if args.triton_rmsnorm:
        from perfwattlab.kernels.rmsnorm_triton import patch_model_rmsnorm
        print(f"Patched {patch_model_rmsnorm(model)} RMSNorm modules with the Triton kernel")
    name = args.model.split("/")[-1] + (f"_{args.tag}" if args.tag else "")
    out = Path(args.out_dir) / name
    write_env(out, vars(args))
    bw = peak_bw_gbps()
    print(f"{spec.name}: {spec.n_params / 1e9:.2f}B params, max context {spec.max_positions}, "
          f"KV {spec.kv_bytes_per_token()} B/token")

    # warmup
    generate(model, synthetic_prompt_ids(tok, 64, 1), 8, pad_id=tok.pad_token_id)
    meter = EnergyMeter()
    meter.measure_idle_power(5)
    s0 = meter.sampler

    rows = []
    for P in ints(args.prompt_lens):
        for O in ints(args.out_lens):
            for B in ints(args.batches):
                base = {"model": spec.name, "prompt_len": P, "out_len": O, "batch": B}
                if P + O > spec.max_positions:
                    rows.append({**base, "status": "context_limit"})
                    print(f"P={P:5d} O={O:3d} B={B:2d}  skipped (context {spec.max_positions})")
                    continue
                for rep in range(args.repeats):
                    ids = synthetic_prompt_ids(tok, P, n=B, seed=rep)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                    try:
                        t = meter.begin()
                        res = generate(model, ids, O, pad_id=tok.pad_token_id, nvtx=args.nvtx,
                                       last_token_logits=args.last_token_logits,
                                       prefill_chunk=args.prefill_chunk)
                        er = meter.end(t)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        rows.append({**base, "rep": rep, "status": "OOM"})
                        print(f"P={P:5d} O={O:3d} B={B:2d}  OOM")
                        break
                    m = res.batch_metrics()
                    row = {**base, "rep": rep, "status": "ok", **m}
                    # prefill compute rate
                    row["last_token_logits"] = args.last_token_logits
                    row["prefill_chunk"] = args.prefill_chunk or 0
                    row["prefill_tflops"] = (spec.prefill_flops(B * P, B, args.last_token_logits)
                                             / (m["prefill_ms"] / 1000) / 1e12)
                    # decode traffic model: weights + KV of all sequences at mean context
                    mean_ctx = P + O / 2
                    step_bytes = spec.decode_bytes_per_step(B, int(mean_ctx))
                    row["decode_bytes_per_step_mb"] = step_bytes / 1e6
                    row["kv_share_of_step_bytes"] = 1 - spec.weight_bytes / step_bytes
                    row["decode_est_gbps"] = step_bytes / (m["tpot_ms"] / 1000) / 1e9 if O > 1 else np.nan
                    if bw:
                        row["decode_pct_peak_bw"] = 100 * row["decode_est_gbps"] / bw
                    row["energy_j_per_out_token"] = er.energy_j / m["output_tokens_total"]
                    row["avg_power_w"] = er.avg_power_w
                    if s0 and s0.enabled and O > 1:
                        dec = s0.window_stats(res.t_first - s0.t0, res.token_times[-1] - s0.t0)
                        row["decode_gpu_util_pct"] = dec.get("mean_gpu_util_pct")
                        row["decode_sm_clock_mhz"] = dec.get("mean_sm_clock_mhz")
                    if torch.cuda.is_available():
                        row["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                    rows.append(row)
                ok = [r for r in rows if r.get("status") == "ok" and r["prompt_len"] == P
                      and r["out_len"] == O and r["batch"] == B]
                if ok:
                    d = pd.DataFrame(ok)
                    print(f"P={P:5d} O={O:3d} B={B:2d}  TTFT {d.ttft_ms.median():8.1f} ms | "
                          f"ITL p50 {d.itl_p50_ms.median():6.2f} ms | decode {d.decode_tok_s.median():8.1f} tok/s | "
                          f"prefill {d.prefill_tok_s.median():9.0f} tok/s")
    meter.close()

    df = pd.DataFrame(rows)
    df.to_csv(out / "runs.csv", index=False)
    ok = df[df.status == "ok"]
    num = ok.select_dtypes("number").columns.difference(["prompt_len", "out_len", "batch", "rep"])
    summary = ok.groupby(["prompt_len", "out_len", "batch"])[list(num)].median().reset_index()
    limits = df[df.status != "ok"][["prompt_len", "out_len", "batch", "status"]].drop_duplicates()
    summary.to_csv(out / "summary.csv", index=False)
    limits.to_csv(out / "limits.csv", index=False)

    cols = ["prompt_len", "batch", "out_len", "ttft_ms", "prefill_tok_s", "prefill_tflops",
            "itl_p50_ms", "itl_p95_ms", "tpot_ms", "decode_tok_s", "decode_gpu_util_pct",
            "decode_est_gbps", "decode_pct_peak_bw", "kv_share_of_step_bytes", "energy_j_per_out_token"]
    cols = [c for c in cols if c in summary.columns]
    md = [f"# Experiment 1 — {spec.name}", "", summary[cols].round(3).to_markdown(index=False), ""]
    if not limits.empty:
        md += ["Limits (context window / OOM):", "", limits.to_markdown(index=False), ""]
    (out / "report.md").write_text("\n".join(md))
    plot(summary, out)
    print(f"\nWrote {out}")


def plot(s: pd.DataFrame, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    O = s.out_len.max()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for P, g in s[s.out_len == O].groupby("prompt_len"):
        ax[0].plot(g.batch, g.itl_p50_ms, "o-", label=f"ctx {P}")
        ax[1].plot(g.batch, g.decode_tok_s, "o-", label=f"ctx {P}")
    b1 = s[(s.batch == s.batch.min()) & (s.out_len == O)]
    ax[2].plot(b1.prompt_len, b1.prefill_tok_s, "o-")
    ax[0].set(xlabel="batch", ylabel="ITL p50 (ms)", title="Decode ITL vs batch", xscale="log", xticks=sorted(s.batch.unique()))
    ax[1].set(xlabel="batch", ylabel="decode tok/s", title="Decode throughput vs batch", xscale="log")
    ax[2].set(xlabel="prompt tokens", ylabel="prefill tok/s", title="Prefill throughput vs prompt length", xscale="log")
    for a in ax:
        a.grid(alpha=.3)
    ax[0].legend(); ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "prefill_decode.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
