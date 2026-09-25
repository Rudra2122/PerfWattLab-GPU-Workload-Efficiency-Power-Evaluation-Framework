"""
rmsnorm_triton.py — Fused RMSNorm in OpenAI Triton (the kernel language).
Not related to NVIDIA Triton Inference Server.  (README §8)

    y = x / sqrt(mean(x^2) + eps) * weight

Eager Hugging Face LlamaRMSNorm:

    h = x.to(fp32)                      # kernel 1: cast, writes an fp32 copy
    var = h.pow(2).mean(-1, keepdim)    # kernels 2-3: square (fp32 tensor), reduce
    h = h * torch.rsqrt(var + eps)      # kernels 4-6: add, rsqrt, multiply (fp32 tensor)
    return w * h.to(x.dtype)            # kernels 7-8: cast, multiply

Each op launches a kernel and round-trips an N×H intermediate through HBM
(several of them in fp32, i.e. 2× the bytes of the fp16 input).

This kernel: one program per row. Load the row once, square/reduce in fp32
registers, normalize, cast, scale, store once. Minimum traffic:

    bytes = N·H·2 (read x) + H·2 (read w) + N·H·2 (write y)      [fp16]

Numerics match HF exactly in structure: fp32 reduction, cast normalized
value to the input dtype, then multiply by weight in the input dtype.
"""

# NOTE: no `from __future__ import annotations` here — it turns the
# `tl.constexpr` annotation into a string and Triton stops treating BLOCK as constexpr.
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _rmsnorm_fwd(X, W, Y, stride_x, stride_y, n_cols, eps,
                     BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)
        xn = (x * rstd).to(Y.dtype.element_ty)
        w = tl.load(W + cols, mask=mask, other=0.0)
        tl.store(Y + row * stride_y + cols, xn * w, mask=mask)


def triton_gpu_supported() -> bool:
    """OpenAI Triton (and therefore torch.compile's Inductor backend) needs
    compute capability >= 7.0. Pascal GPUs such as the P100 (sm_60) are too old."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0) >= (7, 0)


def require_triton_gpu():
    if torch.cuda.is_available() and not triton_gpu_supported():
        name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        raise SystemExit(
            f"{name} is sm_{cc[0]}{cc[1]}; OpenAI Triton and torch.compile need sm_70+ (Volta or newer). "
            f"Run Experiment 4 on a T4/L4/A100 (e.g. Kaggle 'GPU T4 x2').")


def _default_warps(block: int) -> int:
    # 1 warp per 256 fp16 elements is a reasonable starting point; tuned in the benchmark.
    return max(1, min(16, block // 256))


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6,
                   num_warps: Optional[int] = None) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("triton is not installed")
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    n_rows, n_cols = x2.shape
    y = torch.empty_like(x2)
    block = triton.next_power_of_2(n_cols)
    if block * x.element_size() > 65536:
        raise ValueError(f"hidden size {n_cols} too large for single-block kernel")
    _rmsnorm_fwd[(n_rows,)](x2, weight, y, x2.stride(0), y.stride(0), n_cols, eps,
                            BLOCK=block, num_warps=num_warps or _default_warps(block))
    return y.view(shape)


def rmsnorm_eager(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Same math as transformers' LlamaRMSNorm.forward."""
    dt = x.dtype
    h = x.to(torch.float32)
    var = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(var + eps)
    return weight * h.to(dt)


def rmsnorm_reference_fp32(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    h = x.float()
    return (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)) * weight.float()


def min_bytes(n_rows: int, hidden: int, elem_bytes: int = 2) -> int:
    return 2 * n_rows * hidden * elem_bytes + hidden * elem_bytes


# ---------------------------------------------------------------------------
# Swap the kernel into a Hugging Face model (for end-to-end measurement)
# ---------------------------------------------------------------------------

def patch_model_rmsnorm(model, num_warps: Optional[int] = None) -> int:
    """
    Replace forward() of every *RMSNorm module in `model` with the Triton
    kernel. Returns the number of modules patched. Reversible via
    unpatch_model_rmsnorm().
    """
    if next(model.parameters()).is_cuda:
        require_triton_gpu()
    n = 0
    for mod in model.modules():
        if type(mod).__name__.endswith("RMSNorm") and hasattr(mod, "weight"):
            eps = getattr(mod, "variance_epsilon", getattr(mod, "eps", 1e-6))
            if not hasattr(mod, "_pwl_orig_forward"):
                mod._pwl_orig_forward = mod.forward
            mod.forward = (lambda m, e: (lambda h: rmsnorm_triton(h, m.weight, e, num_warps)))(mod, eps)
            n += 1
    return n


def unpatch_model_rmsnorm(model) -> int:
    n = 0
    for mod in model.modules():
        if hasattr(mod, "_pwl_orig_forward"):
            mod.forward = mod._pwl_orig_forward
            del mod._pwl_orig_forward
            n += 1
    return n
