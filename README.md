# PerfWattLab: GPU Inference Performance and Power Evaluation Framework

## The Problem I Was Actually Solving

Standard benchmarks told me throughput looked fine. p50 latency looked acceptable. But p95 wasn't moving, even after I added micro-batching, which should have moved it.

Something was wrong at a layer the high-level metrics couldn't see.

I traced it to the kernel level using `torch.profiler`. Found CPU-GPU synchronization overhead at the pipeline boundary. The GPU was sitting idle, waiting on the CPU between kernel launches. Invisible at the metric level. Completely visible in the kernel timeline.

Removing the high-level pipeline wrapper and using direct inference mode cut p50 latency 16%, from 3.27s to 2.76s, across 1,000+ runs. The throughput was never the problem. The synchronization was.

That investigation is what PerfWattLab became: a reproducible framework for measuring, explaining, and improving GPU inference performance and energy efficiency for LLM workloads. Built to answer questions that standard monitoring can't.

---

## Key Results

| Metric | Baseline | Optimized | Delta |
|---|---|---|---|
| p50 Latency | 3.27s | 2.76s | **−16%** |
| Energy per Query | 191.5J | 185.3J | **−3.2%** |
| MAC Toggle Count | 4,219 | 2,719 | **−35.5%** |

The RTL result connects software execution efficiency directly to dynamic power reduction at the silicon level. Toggle count is proportional to dynamic power consumption. This isn't just a software optimization story.

---

## Architecture

```
Request
  │
  ▼
RAG Pipeline
  ├── FAISS retrieval (MiniLM embeddings)
  ├── Cross-encoder reranker
  └── LLM generation (PyTorch, direct inference mode)
  │
  ▼
Instrumentation Layer
  ├── torch.profiler        — kernel-level execution tracing
  ├── NVML power sampling   — board-level power at 5Hz
  └── Latency measurement   — p50/p95 across concurrent runs
  │
  ▼
Experiment Runner
  ├── Config sweep (tokens, concurrency, execution path)
  ├── Workload profiles benchmarked against TensorRT-LLM and Triton-style baselines
  └── Auto-generated markdown reports
  │
  ▼
Results
  ├── CSV reports
  ├── Latency vs Energy Pareto curves
  ├── Profiler traces
  └── RTL toggle analysis
```

---

## What I Built and Why

### 1. Baseline RAG Pipeline

Built a full RAG pipeline with a FAISS index, cross-encoder reranker, and TinyLlama generator, with end-to-end timing breakdown at each stage. The goal was a reproducible baseline where every millisecond was accounted for.

Measured: retrieval time, rerank time, generation time, and total latency per query.

### 2. Config Sweep and Concurrency Evaluation

Built an experiment runner that sweeps `max_new_tokens`, concurrency, and execution path. Generated p50/p95 latency, throughput (RPS), and tokens/sec across all configurations automatically.

This is where I noticed the anomaly. Micro-batching wasn't moving p95 the way it should have.

### 3. Kernel-Level Profiling and Bottleneck Discovery

This is the core of the project.

Used `torch.profiler` to trace execution at the kernel level. The timeline showed the GPU going dark between launches. Synchronization points at the pipeline boundary where the CPU was forcing the GPU to wait unnecessarily.

**Fix:** Removed the high-level pipeline wrapper. Used `inference_mode()` directly. Placed tensors explicitly on device before the generation loop.

**Result:** 16% latency reduction. Not from a batching change. From removing synchronization that didn't need to be there.

### 4. Power Measurement and Energy per Query

Instrumented GPU board power using NVML at 5Hz. Computed average watts, joules per query, and energy per token across all configurations.

Key finding: two configurations with nearly identical latency had meaningfully different energy profiles. Latency alone doesn't tell the full story of serving efficiency.

Generated a Latency vs Energy Pareto curve to make the tradeoff visible.

### 5. Production Serving Simulation

Simulated production serving conditions including warmup phase, fixed request rate, and concurrency sweep, mirroring internal evaluation workflows used in TensorRT-LLM and Triton benchmarking.

Auto-generated markdown reports comparing baseline vs optimized across latency, throughput, and energy per query.

### 6. RTL Bridge: Connecting Software to Silicon

To validate that software-level optimization actually changes hardware-level power behavior:

- Wrote a baseline Verilog MAC block
- Implemented an optimized operand isolation variant with clock gating logic
- Simulated both using `iverilog`, dumped VCD waveforms
- Counted switching activity via Python VCD parser

First attempt increased toggles due to a design mistake in the gating logic. After redesigning:

**Switching activity reduced 35.5% (4,219 to 2,719 toggles).**

Dynamic power scales directly with switching activity. This closes the loop between a Python optimization and its effect at the silicon level.

---

## Repository Structure

```
perfwattlab/
├── perfwattlab/
│   ├── pipeline.py          # RAG pipeline, direct inference mode
│   ├── profiler.py          # torch.profiler wrapper, kernel trace parsing
│   ├── power.py             # NVML power sampling, energy per query
│   ├── sweep.py             # Experiment runner, config sweep logic
│   └── rtl/
│       ├── mac_baseline.v
│       ├── mac_optimized.v
│       └── toggle_counter.py
├── run_sweep.py             # Entry point: run full experiment
├── run_power.py             # Entry point: power sampling
├── run_profiler.py          # Entry point: kernel profiling
├── run_rtl.sh               # RTL simulation and toggle analysis
├── results/                 # CSV reports, profiler traces
├── figures/                 # Pareto curves, before/after plots
├── notebooks/               # Exploratory analysis (not primary workflow)
└── requirements.txt
```

---

## Reproduce

```bash
pip install -r requirements.txt

# Run full config sweep
python run_sweep.py

# Run power sampling
python run_power.py

# Run kernel profiling
python run_profiler.py

# Run RTL toggle analysis
bash run_rtl.sh
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Framework | PyTorch |
| Retrieval | FAISS |
| Profiling | torch.profiler, NVIDIA Nsight Systems |
| Power Metrics | NVML |
| Serving Baselines | TensorRT-LLM, Triton-style evaluation |
| RTL Simulation | iverilog |
| Toggle Analysis | Python VCD parser |

---

## Limitations

- Evaluated on Colab T4 GPU, not a production A100/H100 environment
- NVML power sampling at 5Hz, not high-resolution power rail measurement
- RTL power estimated via toggle count, not a signoff tool like PrimeTime PX
- Single-node setup, multi-GPU behavior not captured

---

## What I'd Do Next

- Integrate Triton Inference Server directly and profile at the kernel level across serving backends
- Add DCGM exporter with Prometheus integration for production-grade GPU observability
- Extend to multi-GPU workloads with cross-device memory pressure analysis
- Add KV cache hit rate tracking per session, the metric that actually explains throughput collapse under high concurrency

---

## Author

**Rudra Brahmbhatt**
MS Computer Science, Texas State University, May 2026
ML Inference Infrastructure
[LinkedIn](#) · [GitHub](#)
