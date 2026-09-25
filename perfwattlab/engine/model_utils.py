"""
model_utils.py — Model loading, architecture spec, and analytical byte counts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import torch

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# A longer-context model for the 4K/8K sweep points (TinyLlama is 2K max).
LONG_CONTEXT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def now() -> float:
    """Host timestamp after draining the GPU queue."""
    sync()
    return time.perf_counter()


def dtype_kwarg(dt: torch.dtype) -> dict:
    """from_pretrained() dtype keyword: `dtype` since transformers 4.56, `torch_dtype` before."""
    import transformers
    from packaging.version import Version
    return {"dtype": dt} if Version(transformers.__version__) >= Version("4.56") else {"torch_dtype": dt}


def default_dtype() -> torch.dtype:
    # T4 (sm_75) has no native BF16 — FP16 everywhere on CUDA.
    return torch.float16 if torch.cuda.is_available() else torch.float32


TINY = "tiny-random"   # offline smoke-test model: random 2-layer GQA Llama + word-level tokenizer


def tiny_random_lm(attn_implementation: Optional[str] = None, max_positions: int = 1024):
    """A random 2-layer Llama with GQA and an offline tokenizer, for smoke tests
    (`--model tiny-random`). Numbers from it are meaningless; it only checks
    that the pipeline of a script runs end to end."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=max_positions,
                      attn_implementation=attn_implementation or "sdpa")
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, **{f"w{i}": i for i in range(3, 512)}}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", bos_token="<s>",
                                  eos_token="</s>", pad_token="</s>",
                                  model_input_names=["input_ids", "attention_mask"])
    model = LlamaForCausalLM(cfg).to(device=device(), dtype=default_dtype()).eval()
    return model, tok


def load_causal_lm(name: str = DEFAULT_MODEL, dtype: Optional[torch.dtype] = None,
                   attn_implementation: Optional[str] = None):
    if name == TINY:
        return tiny_random_lm(attn_implementation)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = dtype or default_dtype()
    kw = dtype_kwarg(dtype)
    if attn_implementation:
        kw["attn_implementation"] = attn_implementation
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, **kw).to(device())
    model.eval()
    return model, tok


@dataclass
class ModelSpec:
    name: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden: int
    vocab: int
    max_positions: int
    n_params: int
    dtype_bytes: int
    lm_head_params: int = 0

    @property
    def weight_bytes(self) -> int:
        return self.n_params * self.dtype_bytes

    def kv_bytes_per_token(self, n_kv_heads: Optional[int] = None) -> int:
        """2 (K,V) × layers × kv_heads × head_dim × bytes."""
        h = self.n_kv_heads if n_kv_heads is None else n_kv_heads
        return 2 * self.n_layers * h * self.head_dim * self.dtype_bytes

    def kv_bytes_mha_equivalent_per_token(self) -> int:
        return self.kv_bytes_per_token(self.n_heads)

    def prefill_flops(self, prompt_tokens: int, n_sequences: int = 1,
                      last_token_logits: bool = False) -> float:
        """
        ≈ 2 × params × tokens (ignores attention's quadratic term).
        With last_token_logits, the LM head (hidden × vocab) runs once per
        sequence instead of once per prompt token.
        """
        if not last_token_logits or not self.lm_head_params:
            return 2.0 * self.n_params * prompt_tokens
        body = self.n_params - self.lm_head_params
        return 2.0 * body * prompt_tokens + 2.0 * self.lm_head_params * n_sequences

    def decode_bytes_per_step(self, batch: int, context: int) -> int:
        """Weights read once per step + KV read for every active sequence."""
        return self.weight_bytes + batch * context * self.kv_bytes_per_token()


def model_spec(model, name: str = "") -> ModelSpec:
    c = model.config
    n_heads = c.num_attention_heads
    n_kv = getattr(c, "num_key_value_heads", None) or n_heads
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // n_heads
    p = next(model.parameters())
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    lm_head = head.weight.numel() if head is not None else 0
    return ModelSpec(
        name=name or getattr(c, "_name_or_path", "model"),
        n_layers=c.num_hidden_layers, n_heads=n_heads, n_kv_heads=n_kv,
        head_dim=head_dim, hidden=c.hidden_size, vocab=c.vocab_size,
        max_positions=getattr(c, "max_position_embeddings", 2048),
        n_params=sum(x.numel() for x in model.parameters()),
        dtype_bytes=p.element_size(),
        lm_head_params=lm_head,
    )


def last_token_logits_kwargs(model) -> dict:
    """
    Forward kwargs that make the LM head run only on the last position.
    Without this, prefill materializes [batch, prompt_len, vocab] logits
    (e.g. 32 × 4096 × 152k × 2 B ≈ 40 GB for Qwen2.5) when only the last row
    is used. Name differs by transformers version.
    """
    import inspect
    params = inspect.signature(model.forward).parameters
    if "logits_to_keep" in params:
        return {"logits_to_keep": 1}
    if "num_logits_to_keep" in params:
        return {"num_logits_to_keep": 1}
    return {}


def synthetic_prompt_ids(tokenizer, length: int, n: int = 1, seed: int = 0) -> List[List[int]]:
    """
    Random token-id prompts of exact length (for controlled length sweeps).
    Special tokens are excluded. Content doesn't matter for timing because
    output length is forced.
    """
    g = torch.Generator().manual_seed(seed)
    special = set(tokenizer.all_special_ids or [])
    vocab = tokenizer.vocab_size
    lo = min(100, vocab // 4)
    out = []
    for _ in range(n):
        ids = torch.randint(lo, vocab, (length,), generator=g).tolist()
        ids = [i if i not in special else lo for i in ids]
        if tokenizer.bos_token_id is not None:
            ids[0] = tokenizer.bos_token_id
        out.append(ids)
    return out


# Hardware reference numbers used for roofline estimates (README §4.3).
# Checked in order, so "A100" must come before "A10".
GPU_PEAK_BW_GBPS = {"T4": 320.0, "P100": 732.0, "V100": 900.0, "A100": 1555.0, "H100": 3350.0,
                    "A10": 600.0, "L4": 300.0}


def peak_bw_gbps() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(0)
    if "A100" in name and torch.cuda.get_device_properties(0).total_memory > 60e9:
        return 2039.0                     # A100 80GB (HBM2e) vs 1,555 GB/s on the 40GB part
    for k, v in GPU_PEAK_BW_GBPS.items():
        if k in name:
            return v
    return None
