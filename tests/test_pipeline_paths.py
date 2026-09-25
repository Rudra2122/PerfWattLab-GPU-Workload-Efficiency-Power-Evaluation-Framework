"""Experiment 0 code paths on a tiny offline model (API correctness, not timing)."""
import torch
from transformers import pipeline

from perfwattlab.pipeline import (GenerateRecorder, generate_direct, generate_pipeline,
                                  generate_pipeline_staged)
from tests._tiny import tiny_llama, tiny_tokenizer


def _setup():
    m, t = tiny_llama(), tiny_tokenizer()
    m.generation_config.eos_token_id = 2
    m.generation_config.pad_token_id = 2
    return m, t, pipeline("text-generation", model=m, tokenizer=t, device=-1)


def test_fixed_length_paths_agree():
    m, t, p = _setup()
    prompt = " ".join(f"w{i}" for i in range(10, 30))
    _, _, n_pipe, _ = generate_pipeline(prompt, p, t, max_new_tokens=12, min_new_tokens=12)
    txt_d, _, n_dir, _ = generate_direct(prompt, m, t, max_new_tokens=12, min_new_tokens=12)
    txt_n, _, n_ng, _ = generate_direct(prompt, m, t, max_new_tokens=12, min_new_tokens=12,
                                         grad_ctx="no_grad")
    assert n_dir == n_ng == 12
    assert txt_d == txt_n
    assert n_pipe == 12          # re-tokenized without special tokens


def test_staged_and_recorder():
    m, t, p = _setup()
    prompt = "w5 w6 w7 w8"
    st = generate_pipeline_staged(prompt, p, max_new_tokens=5, min_new_tokens=5)
    assert st["forward_ms"] > 0 and "inference_context" in st
    with GenerateRecorder(m) as rec:
        generate_pipeline(prompt, p, t, max_new_tokens=5, min_new_tokens=5)
        generate_direct(prompt, m, t, max_new_tokens=5, min_new_tokens=5)
    assert len(rec.calls) == 2
    assert m.generate.__name__ != "wrapped"
