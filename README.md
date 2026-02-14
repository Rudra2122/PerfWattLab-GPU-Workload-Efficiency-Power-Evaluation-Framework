# ⚡ PerfWattLab — GPU Workload Efficiency & Power Evaluation Framework

## 🚀 Executive Summary

PerfWattLab is a reproducible GPU workload efficiency evaluation framework built to measure, explain, and improve inference performance and energy consumption for RAG-style LLM workloads.

The goal is simple:

- Measure performance like a systems engineer  
- Measure energy like a power architect  
- Connect software optimization to hardware switching activity  

---

## 📊 Key Results (Measured)

### ⚡ Inference Performance

- p50 latency reduced from **~3.27s → ~2.76s** (~15.6% improvement)  
- Throughput improved under optimized execution path  
- Reduced kernel launch overhead by removing high-level pipeline wrapper  

### 🔋 Energy Efficiency

- Energy per query reduced from **~191.5J → ~185.3J**  
- Lower average GPU board power under optimized execution  
- Clear latency vs energy Pareto tradeoff curve generated  

### 🧠 RTL Switching Activity

- Baseline MAC toggle count: **4219**  
- Optimized MAC toggle count: **2719**  
- **35.5% reduction in switching activity**  

This connects software execution efficiency directly to dynamic power reduction at the silicon level.

---

## 🧠 Problem

Modern LLM evaluation focuses heavily on:

- Model accuracy  
- Model size  
- Parameter scaling  

But production systems fail due to:

- Kernel launch inefficiencies  
- GPU underutilization  
- Energy cost per inference  
- Lack of reproducible performance baselines  
- No visibility into perf-per-watt tradeoffs  

PerfWattLab was built to evaluate GPU workloads end-to-end with measurable evidence.

---

## 🏗 System Overview

```text
Query
  │
  ▼
RAG Pipeline
  ├── FAISS retrieval
  ├── Cross-encoder rerank
  ├── LLM generation (PyTorch)
  │
  ▼
Instrumentation Layer
  ├── torch.profiler
  ├── NVML power sampling
  ├── Latency measurement
  │
  ▼
Experiment Runner
  ├── Config sweep
  ├── Concurrency variation
  ├── Token length variation
  │
  ▼
Results Artifacts
  ├── CSV reports
  ├── Power samples
  ├── Pareto plots
  ├── Profiler traces
