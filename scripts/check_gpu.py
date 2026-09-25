"""
check_gpu.py — 20-second preflight: which PerfWattLab experiments can run on this GPU?

    python scripts/check_gpu.py
"""
import torch

print(f"torch {torch.__version__}  CUDA build {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU visible. On Kaggle: Settings → Accelerator → GPU.")

name = torch.cuda.get_device_name(0)
cc = torch.cuda.get_device_capability(0)
print(f"GPU {name}  sm_{cc[0]}{cc[1]}  {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB  "
      f"(visible GPUs: {torch.cuda.device_count()}, PerfWattLab uses GPU 0)")

# 1. Does this torch build actually have kernels for this GPU?
try:
    x = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    (x @ x).sum().item()
    print("[ok]   torch runs FP16 matmul on this GPU")
except RuntimeError as e:
    print(f"[FAIL] torch can't run on this GPU: {e}")
    print("       Newer torch builds dropped Pascal (sm_60). Install one that includes it, e.g.:")
    print("       pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121")
    raise SystemExit(1)

# 2. NVML energy counter (Volta+)
try:
    import pynvml as nv
    nv.nvmlInit()
    h = nv.nvmlDeviceGetHandleByIndex(0)
    try:
        nv.nvmlDeviceGetTotalEnergyConsumption(h)
        print("[ok]   NVML energy counter available → energy_method = counter")
    except Exception:
        print("[warn] NVML energy counter NOT supported (pre-Volta) → energy falls back to sampled power "
              "(energy_method = sampler); coarser, still valid over long windows")
    print(f"[ok]   NVML power reading: {nv.nvmlDeviceGetPowerUsage(h) / 1000:.1f} W")
    nv.nvmlShutdown()
except Exception as e:
    print(f"[warn] NVML unavailable ({e}) → no power/energy numbers")

# 3. Triton / torch.compile / vLLM need sm_70+
ok70 = cc >= (7, 0)
print(f"[{'ok' if ok70 else 'NO'}]   {'' if ok70 else '  '}OpenAI Triton + torch.compile (Experiment 4)")
print(f"[{'ok' if ok70 else 'NO'}]   {'' if ok70 else '  '}vLLM (Experiment 5)")
print(f"[{'ok' if cc >= (8, 0) else 'no'}]   BF16 (Ampere+ only; everything here uses FP16)")

print("\nRunnable here: Exp 0, 1, 2, 3, RTL" + (", 4, 5" if ok70 else
      "\nRun elsewhere (sm_70+, e.g. Kaggle 'GPU T4 x2'): Exp 4 (Triton RMSNorm), Exp 5 (vLLM)"))
