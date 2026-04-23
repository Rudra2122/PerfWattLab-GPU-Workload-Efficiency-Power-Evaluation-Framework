"""
PerfWattLab — GPU Inference Performance & Power Evaluation Framework

Core modules:
  pipeline  — RAG pipeline, retrieval, reranking, and two generation paths
  profiler  — torch.profiler wrapper for kernel-level trace analysis
  power     — NVML power sampling and energy-per-query computation
  sweep     — Config sweep and benchmark runner
  rtl/      — Verilog MAC units and VCD toggle counter
"""
