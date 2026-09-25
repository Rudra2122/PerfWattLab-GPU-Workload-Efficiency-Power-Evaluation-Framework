"""Streaming server: tokens arrive incrementally and equal HF greedy output."""
import asyncio
import json
import socket
import threading
import time

import pytest

aiohttp = pytest.importorskip("aiohttp")

from perfwattlab.backends.vllm_runner import run_server  # noqa: E402
from perfwattlab.engine.kv_cache import PagedKVCache  # noqa: E402
from perfwattlab.engine.model_utils import model_spec  # noqa: E402
from perfwattlab.engine.scheduler import ContinuousBatchingEngine, Request, summarize  # noqa: E402
from perfwattlab.engine.server import EngineServer  # noqa: E402
from tests._tiny import hf_greedy, tiny_llama, tiny_tokenizer  # noqa: E402


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


@pytest.fixture(scope="module")
def server():
    import torch
    from aiohttp import web
    m, t = tiny_llama(), tiny_tokenizer()
    spec = model_spec(m)
    kv = PagedKVCache(spec, num_blocks=64, block_size=4, dtype=torch.float32, device="cpu")
    eng = ContinuousBatchingEngine(m, spec, kv, pad_id=0, max_batch=4, watermark_blocks=1)
    srv = EngineServer(eng, t, "tiny", max_positions=512)
    port = _free_port()
    loop = asyncio.new_event_loop()

    def serve():
        asyncio.set_event_loop(loop)
        runner = web.AppRunner(srv.app())
        loop.run_until_complete(runner.setup())
        loop.run_until_complete(web.TCPSite(runner, "127.0.0.1", port).start())
        loop.run_forever()
    threading.Thread(target=serve, daemon=True).start()
    time.sleep(0.5)
    return m, f"http://127.0.0.1:{port}"


async def _stream(url, ids, n):
    chunks, times = [], []
    async with aiohttp.ClientSession() as s:
        async with s.post(url + "/v1/completions",
                          json={"prompt": ids, "max_tokens": n, "stream": True}) as r:
            assert r.status == 200
            async for raw in r.content:
                line = raw.decode().strip()
                if line.startswith("data:") and line[5:].strip() != "[DONE]":
                    chunks.append(json.loads(line[5:])["choices"][0]["token_ids"][0])
                    times.append(time.perf_counter())
    return chunks, times


def test_stream_matches_hf_and_is_incremental(server):
    m, url = server
    ids = [1, 7, 42, 99, 3, 250]
    toks, times = asyncio.run(_stream(url, ids, 12))
    assert toks == hf_greedy(m, ids, 12)
    assert len(set(round(t, 4) for t in times)) > 1      # arrived over time, not in one blob


def test_concurrent_streams_and_errors(server):
    m, url = server

    async def many():
        return await asyncio.gather(*[_stream(url, [1, 5 + i, 9, 11 + i], 6 + i) for i in range(5)])
    for i, (toks, _) in enumerate(asyncio.run(many())):
        assert toks == hf_greedy(m, [1, 5 + i, 9, 11 + i], 6 + i)

    async def bad():
        async with aiohttp.ClientSession() as s:
            async with s.post(url + "/v1/completions", json={"prompt": [1, 2], "max_tokens": 10_000}) as r:
                return r.status
    assert asyncio.run(bad()) == 400


def test_benchmark_client_against_engine_server(server):
    _, url = server
    reqs = [Request(i, [1, 3 + i, 8, 20], 5, arrival_s=0.02 * i) for i in range(6)]
    res = run_server(url, "tiny", reqs)
    s, df, _ = summarize(res, "perfwattlab_http", 50)
    assert (df.output_tokens == 5).all() and s["throughput_out_tok_s"] > 0
