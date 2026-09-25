"""End-to-end smoke test of run_exp0.py with a tiny offline model (no GPU, no hub)."""
import sys

from transformers import pipeline

import run_exp0
from tests._tiny import tiny_llama, tiny_tokenizer


def test_exp0_runs(tmp_path, monkeypatch):
    m, t = tiny_llama(), tiny_tokenizer()
    m.generation_config.eos_token_id = 2
    m.generation_config.pad_token_id = 2
    gp = pipeline("text-generation", model=m, tokenizer=t, device=-1)
    monkeypatch.setattr(run_exp0, "load_models", lambda: (None, None, t, m, gp))
    monkeypatch.setattr(run_exp0, "ensure_index", lambda *a: (None, None))
    monkeypatch.setattr(run_exp0, "retrieve", lambda q, *a: ([], 1.0))
    monkeypatch.setattr(run_exp0, "rerank", lambda q, r, *a: ([], 1.0))
    monkeypatch.setattr(run_exp0, "build_prompt", lambda q, c: "w5 w6 w7 " + " ".join(f"w{len(q) + i}" for i in range(8)))
    monkeypatch.setattr(sys, "argv", ["run_exp0.py", "--n-prompts", "4", "--repeats", "2",
                                      "--max-new-tokens", "6", "--idle-seconds", "0.1",
                                      "--h2-runs", "2", "--blocked", "--out-dir", str(tmp_path)])
    run_exp0.main()
    for f in ("runs.csv", "report.md", "comparison.csv", "h3_generate_kwargs.json",
              "h2_pipeline_stages.csv", "env.json"):
        assert (tmp_path / f).exists(), f
    rep = (tmp_path / "report.md").read_text()
    assert "H1" in rep and "H3" in rep and "H4" in rep
