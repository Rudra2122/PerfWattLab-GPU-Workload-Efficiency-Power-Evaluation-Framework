"""
Experiment 5 — HF generate vs PerfWattLab continuous batching vs vLLM (README §9).

vLLM is not in requirements.txt (it pins its own torch). Use a separate env:
    pip install vllm aiohttp

Offline peak throughput (all requests submitted at once):
    python run_vllm.py offline --mix short --n-requests 64

Online, same open-loop workload as run_serving.py:
    vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dtype half --port 8000 &
    python run_vllm.py server --rates 0.25,0.5,1,2,4 --mix short --n-requests 64

The PerfWattLab engine over HTTP, measured by the same client:
    python serve_engine.py --port 8001 &
    python run_vllm.py server --base-url http://localhost:8001 --label perfwattlab_http --mix short

Then build the comparison table:
    python run_vllm.py compare --mix short
"""

import argparse
from pathlib import Path

import pandas as pd

from perfwattlab.engine.loadgen import fresh, make_workload
from perfwattlab.engine.model_utils import DEFAULT_MODEL
from perfwattlab.env import write_env


def workload_meta(model_name):
    """Vocab / context for make_workload without loading weights."""
    from transformers import AutoConfig, AutoTokenizer
    cfg = AutoConfig.from_pretrained(model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    return cfg.vocab_size, cfg.max_position_embeddings, tok.bos_token_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["offline", "server", "compare"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--mix", default="short")
    ap.add_argument("--n-requests", type=int, default=64)
    ap.add_argument("--rates", default="0.25,0.5,1,2,4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--served-model-name", default=None)
    ap.add_argument("--label", default="vllm_server",
                    help="policy name in results; use perfwattlab_http when --base-url points at serve_engine.py")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--slo-ttft-ms", type=float, default=2000)
    ap.add_argument("--slo-tpot-ms", type=float, default=100)
    ap.add_argument("--out-dir", default="results/v2/exp5_vllm")
    ap.add_argument("--serving-dir", default="results/v2/exp3_serving")
    args = ap.parse_args()
    out = Path(args.out_dir) / f"{args.model.split('/')[-1]}_{args.mix}"
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "compare":
        return compare(args, out)

    write_env(out, vars(args))
    vocab, maxpos, bos = workload_meta(args.model)

    if args.mode == "offline":
        from perfwattlab.backends.vllm_runner import run_offline
        from perfwattlab.energy import EnergyMeter
        reqs = make_workload(args.n_requests, 1.0, args.mix, vocab, maxpos, bos_id=bos, seed=args.seed)
        meter = EnergyMeter()
        r = run_offline(args.model, reqs, args.gpu_memory_utilization, meter=meter)
        meter.close()
        pd.DataFrame([r]).to_csv(out / "offline.csv", index=False)
        print(r)
        return

    from perfwattlab.backends.vllm_runner import run_server
    from perfwattlab.energy import EnergyMeter
    from perfwattlab.engine.scheduler import summarize
    meter = EnergyMeter()
    rows = []
    for rate in [float(x) for x in args.rates.split(",")]:
        reqs = fresh(make_workload(args.n_requests, rate, args.mix, vocab, maxpos, bos_id=bos, seed=args.seed))
        t = meter.begin()
        res = run_server(args.base_url, args.served_model_name or args.model, reqs)
        er = meter.end(t)
        short = [r.rid for r in reqs if len(r.token_times) != r.max_new_tokens]
        if short:
            print(f"  WARNING: {len(short)} requests streamed a chunk count != max_tokens "
                  f"(multi-token chunks); TPOT/ITL for those are approximate")
        s, rdf, _ = summarize(res, args.label, rate, args.slo_ttft_ms, args.slo_tpot_ms)
        s["energy_j_per_out_token"] = er.energy_j / max(int(rdf.output_tokens.sum()), 1)
        rows.append(s)
        prefix = "vllm" if args.label == "vllm_server" else args.label
        rdf.to_csv(out / f"requests_{prefix}_{rate:g}.csv", index=False)
        print(f"rate {rate:g}: {s['throughput_out_tok_s']:.1f} tok/s, TTFT p50/p99 "
              f"{s['ttft_ms_p50']:.0f}/{s['ttft_ms_p99']:.0f} ms, TPOT p50 {s['tpot_ms_p50']:.1f} ms")
    meter.close()
    name = "server_summary.csv" if args.label == "vllm_server" else f"server_summary_{args.label}.csv"
    pd.DataFrame(rows).to_csv(out / name, index=False)


def compare(args, out):
    ours = Path(args.serving_dir) / f"{args.model.split('/')[-1]}_{args.mix}" / "summary.csv"
    frames = []
    if ours.exists():
        frames.append(pd.read_csv(ours))
    else:
        print(f"missing {ours} — run run_serving.py with the same --mix/--n-requests/--seed first")
    for vs in sorted(out.glob("server_summary*.csv")):     # vLLM and/or perfwattlab_http
        frames.append(pd.read_csv(vs))
    if not frames:
        return
    S = pd.concat(frames, ignore_index=True)
    cols = ["policy", "arrival_rate_rps", "throughput_out_tok_s", "goodput_req_s", "ttft_ms_p50",
            "ttft_ms_p99", "tpot_ms_p50", "tpot_ms_p99", "energy_j_per_out_token"]
    S = S[[c for c in cols if c in S.columns]].sort_values(["arrival_rate_rps", "policy"])
    S.to_csv(out / "comparison.csv", index=False)
    md = ["# Experiment 5 — serialized HF vs PerfWattLab continuous batching vs vLLM", "",
          S.round(3).to_markdown(index=False), ""]
    off = out / "offline.csv"
    if off.exists():
        md += ["Offline peak throughput (vLLM, all requests at once):", "",
               pd.read_csv(off).round(3).to_markdown(index=False), ""]
    md += ["`vllm_server` and `perfwattlab_http` are measured with the same HTTP streaming client; "
           "`serialized`, `static` and `continuous` are in-process.", ""]
    (out / "report.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
