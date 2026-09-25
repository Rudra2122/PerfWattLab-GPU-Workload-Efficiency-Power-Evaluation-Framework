import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Triton decides compiled-vs-interpreted when it is first imported, and
# transformers 5.x imports Triton itself. So pick the mode here, before any test
# module imports transformers: interpreter on CPU or on GPUs older than sm_70.
import torch  # noqa: E402  (torch alone does not import triton)

_gpu_ok = torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (7, 0)
if not _gpu_ok:
    os.environ.setdefault("TRITON_INTERPRET", "1")
