"""
server.py — OpenAI-compatible streaming HTTP server in front of the
continuous batching engine (README §7.6).

    POST /v1/completions   {"prompt": str | [token ids], "max_tokens": N, "stream": true}
    GET  /v1/models

Design
------
* One engine thread owns the GPU and calls ContinuousBatchingEngine.step() in
  a loop. New requests arrive through a thread-safe inbox; the thread blocks
  on it only when there is no work.
* Each generated token fires the engine's on_token hook, which hands the
  token to the request's asyncio.Queue on the server's event loop
  (call_soon_threadsafe). The HTTP handler turns each queue item into one
  server-sent-events chunk, so clients see tokens as soon as they exist.
* Decoding is greedy and output length is forced to max_tokens (EOS is
  ignored), matching how every PerfWattLab benchmark runs. Requests that
  would exceed the model's context window are rejected with HTTP 400.

Because the same OpenAI-style client (backends/vllm_runner.run_server) drives
both this server and `vllm serve`, the Experiment 5 comparison can be made
HTTP-to-HTTP rather than in-process vs HTTP.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import queue
import threading
import time
from typing import Optional

from .scheduler import ContinuousBatchingEngine, Request, _Clock


class EngineServer:
    def __init__(self, engine: ContinuousBatchingEngine, tokenizer, model_name: str,
                 max_positions: int):
        self.engine = engine
        self.tok = tokenizer
        self.model_name = model_name
        self.max_positions = max_positions
        self.inbox: "queue.Queue[Request]" = queue.Queue()
        self.streams = {}                      # rid -> (loop, asyncio.Queue)
        self.clock = _Clock()
        self._ids = itertools.count()
        self.error: Optional[BaseException] = None
        engine.on_token = self._on_token
        self.thread = threading.Thread(target=self._engine_loop, daemon=True, name="engine")
        self.thread.start()

    # ---------------------------------------------------------------- engine side
    def _on_token(self, r: Request, tok: int, t: float, finished: bool):
        entry = self.streams.get(r.rid)
        if entry is None:                       # client went away; the request still completes
            return
        loop, q = entry
        loop.call_soon_threadsafe(q.put_nowait, (tok, finished))

    def _engine_loop(self):
        eng = self.engine
        while True:
            if not eng.has_work():
                eng.add(self.inbox.get())       # block until something arrives
            while True:
                try:
                    eng.add(self.inbox.get_nowait())
                except queue.Empty:
                    break
            try:
                eng.step(self.clock)
            except BaseException as e:          # surface to every open stream
                self.error = e
                for loop, q in list(self.streams.values()):
                    loop.call_soon_threadsafe(q.put_nowait, e)
                eng.waiting.clear()
                eng.running.clear()
                eng.prefilling.clear()
                eng.kv.reset()

    # ---------------------------------------------------------------- HTTP side
    def submit(self, prompt_ids, max_tokens: int, loop) -> Request:
        r = Request(rid=next(self._ids), prompt_ids=list(prompt_ids),
                    max_new_tokens=max_tokens, arrival_s=self.clock.now())
        q: asyncio.Queue = asyncio.Queue()
        self.streams[r.rid] = (loop, q)
        self.inbox.put(r)
        return r

    def _parse(self, body: dict):
        prompt = body.get("prompt")
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], list):
            if len(prompt) != 1:
                raise ValueError("batched prompts are not supported; send one request per prompt")
            prompt = prompt[0]
        if isinstance(prompt, str):
            ids = self.tok(prompt)["input_ids"]
        elif isinstance(prompt, list) and all(isinstance(x, int) for x in prompt):
            ids = prompt
        else:
            raise ValueError("prompt must be a string or a list of token ids")
        max_tokens = int(body.get("max_tokens", 16))
        if not ids or max_tokens < 1:
            raise ValueError("empty prompt or max_tokens < 1")
        if len(ids) + max_tokens > self.max_positions:
            raise ValueError(f"prompt ({len(ids)}) + max_tokens ({max_tokens}) exceeds "
                             f"context window {self.max_positions}")
        return ids, max_tokens

    def app(self):
        from aiohttp import web

        async def models(_request):
            return web.json_response({"object": "list", "data": [
                {"id": self.model_name, "object": "model", "owned_by": "perfwattlab"}]})

        async def completions(request):
            try:
                body = await request.json()
                ids, max_tokens = self._parse(body)
            except (ValueError, json.JSONDecodeError) as e:
                return web.json_response({"error": {"message": str(e)}}, status=400)
            loop = asyncio.get_running_loop()
            r = self.submit(ids, max_tokens, loop)
            q = self.streams[r.rid][1]
            cid = f"cmpl-pwl-{r.rid}"
            created = int(time.time())
            try:
                if body.get("stream"):
                    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream",
                                                       "Cache-Control": "no-cache"})
                    await resp.prepare(request)
                    while True:
                        item = await q.get()
                        if isinstance(item, BaseException):
                            await resp.write(f"data: {json.dumps({'error': str(item)})}\n\n".encode())
                            break
                        tok, fin = item
                        chunk = {"id": cid, "object": "text_completion", "created": created,
                                 "model": self.model_name,
                                 "choices": [{"index": 0, "text": self.tok.decode([tok]),
                                              "token_ids": [tok],
                                              "finish_reason": "length" if fin else None}]}
                        await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        if fin:
                            break
                    await resp.write(b"data: [DONE]\n\n")
                    return resp
                toks = []
                while True:
                    item = await q.get()
                    if isinstance(item, BaseException):
                        return web.json_response({"error": {"message": str(item)}}, status=500)
                    tok, fin = item
                    toks.append(tok)
                    if fin:
                        break
                return web.json_response({
                    "id": cid, "object": "text_completion", "created": created, "model": self.model_name,
                    "choices": [{"index": 0, "text": self.tok.decode(toks), "token_ids": toks,
                                 "finish_reason": "length"}],
                    "usage": {"prompt_tokens": len(ids), "completion_tokens": len(toks),
                              "total_tokens": len(ids) + len(toks)}})
            finally:
                self.streams.pop(r.rid, None)

        app = web.Application()
        app.router.add_get("/v1/models", models)
        app.router.add_post("/v1/completions", completions)
        return app
