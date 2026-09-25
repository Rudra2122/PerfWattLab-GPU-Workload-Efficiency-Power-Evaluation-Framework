"""
PerfWattLab 2.0 — LLM inference performance engineering lab.

  pipeline   RAG pipeline; pipeline() vs direct generate() paths (Experiment 0)
  engine/    explicit prefill/decode loop, paged KV cache, continuous batching
             scheduler, open-loop load generator (Experiments 1–3)
  kernels/   Triton-language RMSNorm (Experiment 4)
  backends/  vLLM comparison (Experiment 5)
  profiler   torch.profiler capture + quantitative trace analysis
  power      NVML power sampling;  energy  NVML energy counter + idle subtraction
  stats      bootstrap CIs and paired comparisons
  rtl/       independent RTL switching-activity experiment
"""
