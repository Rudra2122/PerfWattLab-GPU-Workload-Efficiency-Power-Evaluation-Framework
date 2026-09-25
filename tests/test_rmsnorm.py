"""Triton RMSNorm correctness. Runs on CPU via Triton's interpreter."""
import os

import pytest
import torch

# conftest.py sets TRITON_INTERPRET=1 when there is no sm_70+ GPU.
_GPU_OK = os.environ.get("TRITON_INTERPRET") != "1" and torch.cuda.is_available()
triton = pytest.importorskip("triton")

from perfwattlab.kernels.rmsnorm_triton import (patch_model_rmsnorm, rmsnorm_eager,  # noqa: E402
                                                rmsnorm_triton, unpatch_model_rmsnorm)
from tests._tiny import tiny_llama  # noqa: E402

DEV = "cuda" if _GPU_OK else "cpu"


@pytest.mark.parametrize("H", [64, 100, 2048])
def test_matches_eager(H):
    x = torch.randn(5, 3, H, device=DEV)
    w = torch.randn(H, device=DEV)
    assert torch.allclose(rmsnorm_triton(x, w, 1e-6), rmsnorm_eager(x, w, 1e-6), atol=1e-5)


def test_patched_model_same_logits():
    m = tiny_llama().to(DEV)
    ids = torch.randint(3, 512, (2, 9), device=DEV)
    with torch.inference_mode():
        a = m(ids).logits
        n = patch_model_rmsnorm(m)
        b = m(ids).logits
        unpatch_model_rmsnorm(m)
        c = m(ids).logits
    assert n == 2 * 2 + 1
    assert torch.allclose(a, b, atol=1e-4) and torch.equal(a, c)
