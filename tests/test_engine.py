import random

import pytest
import torch

from perfwattlab.engine.generate_loop import generate
from perfwattlab.engine.kv_cache import PagedKVCache, capacity_comparison
from perfwattlab.engine.loadgen import fresh, make_workload
from perfwattlab.engine.model_utils import model_spec
from perfwattlab.engine.scheduler import (ContinuousBatchingEngine, Request, run_serialized,
                                          run_static)
from tests._tiny import hf_greedy, tiny_llama


@pytest.fixture(scope="module", params=["eager", "sdpa"])
def model(request):
    return tiny_llama(attn=request.param)


def _prompts(n, seed=0):
    rnd = random.Random(seed)
    return [[1] + [rnd.randrange(3, 512) for _ in range(rnd.randrange(3, 40))] for _ in range(n)]


def test_generate_loop_matches_hf(model):
    prompts = _prompts(4)
    res = generate(model, prompts, 12, pad_id=0)
    for p, out in zip(prompts, res.output_ids):
        assert out == hf_greedy(model, p, 12)
    m = res.batch_metrics()
    assert m["output_tokens_total"] == 4 * 12 and len(res.step_ms) == 11


def test_generate_loop_no_grad_matches(model):
    prompts = _prompts(2, seed=3)
    a = generate(model, prompts, 8, pad_id=0, grad_ctx="inference_mode").output_ids
    b = generate(model, prompts, 8, pad_id=0, grad_ctx="no_grad").output_ids
    assert a == b


def _reqs(prompts, outs):
    return [Request(i, p, o, 0.0) for i, (p, o) in enumerate(zip(prompts, outs))]


@pytest.mark.parametrize("num_blocks", [256, 12])   # 12 blocks x 4 slots forces preemption
def test_continuous_matches_hf(model, num_blocks):
    spec = model_spec(model)
    prompts = _prompts(6, seed=1)
    outs = [5, 9, 3, 12, 7, 10]
    kv = PagedKVCache(spec, num_blocks=num_blocks, block_size=4, dtype=torch.float32, device="cpu")
    eng = ContinuousBatchingEngine(model, spec, kv, pad_id=0, max_batch=4, watermark_blocks=1)
    res = eng.run(_reqs(prompts, outs))
    for r in res["requests"]:
        assert r.output_ids == hf_greedy(model, r.prompt_ids, r.max_new_tokens), r.rid
        assert len(r.token_times) == r.max_new_tokens
    assert kv.num_free() == num_blocks            # everything returned to the free list


def test_preemption_recompute_is_exact(model):
    """Short prompts + long outputs in a tiny pool: sequences must be preempted
    mid-decode, recomputed, and still produce exactly HF's greedy tokens."""
    spec = model_spec(model)
    rnd = random.Random(7)
    prompts = [[1] + [rnd.randrange(3, 512) for _ in range(4)] for _ in range(4)]
    kv = PagedKVCache(spec, num_blocks=12, block_size=4, dtype=torch.float32, device="cpu")
    eng = ContinuousBatchingEngine(model, spec, kv, pad_id=0, max_batch=4, watermark_blocks=0)
    res = eng.run(_reqs(prompts, [20] * 4))
    assert sum(r.preemptions for r in res["requests"]) > 0
    for r in res["requests"]:
        assert r.output_ids == hf_greedy(model, r.prompt_ids, 20)
    assert kv.num_free() == 12


def test_static_and_serialized_match_hf(model):
    prompts = _prompts(5, seed=2)
    outs = [4, 8, 6, 3, 7]
    for fn in (run_serialized, run_static):
        res = fn(model, _reqs(prompts, outs), pad_id=0, max_batch=3, batch_timeout_s=0.0)
        for r in res["requests"]:
            assert r.output_ids == hf_greedy(model, r.prompt_ids, r.max_new_tokens)


def test_workload_and_arrivals(model):
    reqs = make_workload(10, rate_rps=50, mix="short", vocab_size=512, max_positions=512, seed=0)
    assert reqs[0].arrival_s == 0.0 and all(b.arrival_s >= a.arrival_s for a, b in zip(reqs, reqs[1:]))
    spec = model_spec(model)
    kv = PagedKVCache(spec, num_blocks=2048, block_size=16, dtype=torch.float32, device="cpu")
    res = ContinuousBatchingEngine(model, spec, kv, pad_id=0, max_batch=8).run(fresh(reqs))
    for r in res["requests"]:
        assert r.token_times[0] >= r.arrival_s          # never emits before it arrives
    assert all(r.output_ids == [] for r in fresh(reqs))


def test_allocator_accounting():
    m = tiny_llama()
    spec = model_spec(m)
    kv = PagedKVCache(spec, num_blocks=8, block_size=4, dtype=torch.float32, device="cpu")
    kv.allocate("A", 5)                               # 2 blocks
    assert len(kv.block_table["A"]) == 2 and kv.num_free() == 6
    kv.seq_lens["A"] = 8
    assert kv.ensure_slot_for_next_token("A") and len(kv.block_table["A"]) == 3
    s = kv.stats()
    assert s["kv_internal_frag_slots"] == 12 - 8
    kv.free("A")
    assert kv.num_free() == 8
    assert spec.kv_bytes_per_token() == 2 * 2 * 2 * 16 * 4   # 2 x layers x kv_heads x head_dim x fp32


def test_capacity_comparison():
    spec = model_spec(tiny_llama())
    per = spec.kv_bytes_per_token()
    c = capacity_comparison(spec, budget_bytes=per * 1000, lengths=[100] * 50, max_len=500, block_size=16)
    assert c["contiguous_max_concurrent"] == 2
    assert c["paged_max_concurrent"] == 1000 // 16 // 7    # 7 blocks per 100-token seq


def test_max_batch_search():
    from run_kv_cache import max_batch_search
    calls = []

    def fake(limit, kind):
        def f(b):
            calls.append(b)
            return True if b <= limit else kind
        return f
    assert max_batch_search(fake(37, "oom")) == (37, "oom")
    assert max(calls) <= 64                       # never jumps to huge batches
    assert max_batch_search(fake(5000, "launch_limit"), cap=8192) == (5000, "launch_limit")
    assert max_batch_search(fake(10**9, "oom"), cap=1024) == (1024, "cap")
    assert max_batch_search(fake(0, "oom")) == (0, "oom")


def test_ratio_ci_zero_safe():
    import numpy as np
    from perfwattlab.stats import ratio_ci
    r = ratio_ci([0.0, 0.0], [1.0, 2.0])
    assert all(np.isnan(x) for x in r)


def test_kv_reset_reuses_pool():
    spec = model_spec(tiny_llama())
    kv = PagedKVCache(spec, num_blocks=8, block_size=4, dtype=torch.float32, device="cpu")
    pool = kv.pool
    kv.allocate("A", 10); kv.gather_ms_total = 5.0
    kv.reset()
    assert kv.pool is pool and kv.num_free() == 8 and not kv.block_table and kv.gather_ms_total == 0


def test_last_token_logits_same_tokens(model):
    prompts = _prompts(3, seed=5)
    a = generate(model, prompts, 6, pad_id=0).output_ids
    b = generate(model, prompts, 6, pad_id=0, last_token_logits=True).output_ids
    assert a == b


def test_prefill_flops_last_token():
    spec = model_spec(tiny_llama())
    assert spec.lm_head_params == 512 * 64
    full = spec.prefill_flops(100, 2)
    last = spec.prefill_flops(100, 2, last_token_logits=True)
    assert last == full - 2 * spec.lm_head_params * 98


@pytest.mark.parametrize("chunk", [1, 4, 7])
def test_chunked_prefill_same_tokens(model, chunk):
    prompts = _prompts(3, seed=9)                      # different lengths → left padding
    a = generate(model, prompts, 6, pad_id=0).output_ids
    b = generate(model, prompts, 6, pad_id=0, prefill_chunk=chunk).output_ids
    assert a == b


@pytest.mark.parametrize("chunk", [1, 5, 16])
def test_continuous_chunked_prefill_matches_hf(model, chunk):
    spec = model_spec(model)
    prompts = _prompts(6, seed=11)
    outs = [5, 9, 3, 12, 7, 10]
    kv = PagedKVCache(spec, num_blocks=256, block_size=4, dtype=torch.float32, device="cpu")
    eng = ContinuousBatchingEngine(model, spec, kv, pad_id=0, max_batch=4, prefill_chunk=chunk)
    res = eng.run(_reqs(prompts, outs))
    for r in res["requests"]:
        assert r.output_ids == hf_greedy(model, r.prompt_ids, r.max_new_tokens), r.rid
    assert kv.num_free() == 256
    assert max(it["prefill_tokens"] for it in res["iterations"]) <= chunk


def test_chunked_prefill_with_preemption(model):
    spec = model_spec(model)
    rnd = random.Random(7)
    prompts = [[1] + [rnd.randrange(3, 512) for _ in range(9)] for _ in range(4)]
    kv = PagedKVCache(spec, num_blocks=14, block_size=4, dtype=torch.float32, device="cpu")
    eng = ContinuousBatchingEngine(model, spec, kv, pad_id=0, max_batch=4, watermark_blocks=0, prefill_chunk=3)
    res = eng.run(_reqs(prompts, [20] * 4))
    assert sum(r.preemptions for r in res["requests"]) > 0
    for r in res["requests"]:
        assert r.output_ids == hf_greedy(model, r.prompt_ids, 20)


@pytest.mark.parametrize("mode", ["none", "manual"])        # "manual" runs eager on CPU
def test_static_cache_decode_matches_hf(model, mode):
    from perfwattlab.engine.static_decode import generate_static
    prompts = _prompts(3, seed=13)                              # uneven → padding path
    res = generate_static(model, prompts, 9, pad_id=0, graph_mode=mode)
    for p, out in zip(prompts, res.output_ids):
        assert out == hf_greedy(model, p, 9)
    same = [[1, 4, 9, 16, 25], [1, 7, 8, 9, 10]]                # even → no-mask path
    res = generate_static(model, same, 6, pad_id=0, graph_mode=mode)
    for p, out in zip(same, res.output_ids):
        assert out == hf_greedy(model, p, 6)
    assert len(res.step_ms) == 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_cuda_graph_decode_matches_hf_on_gpu():
    """Real graph capture + replay on the GPU must give HF's exact greedy tokens."""
    from perfwattlab.engine.static_decode import generate_static
    m = tiny_llama(attn="sdpa").cuda()
    prompts = [[1, 4, 9, 16, 25, 36], [1, 7, 8, 9, 10, 11]]
    res = generate_static(m, prompts, 16, pad_id=0, graph_mode="manual")
    assert res.graph_mode_used == "manual", f"graph capture failed: {res.graph_error}"
    for p, out in zip(prompts, res.output_ids):
        ref = m.generate(torch.tensor([p], device="cuda"), max_new_tokens=16, min_new_tokens=16,
                         do_sample=False, eos_token_id=None, pad_token_id=0)[0, len(p):].tolist()
        assert out == ref
