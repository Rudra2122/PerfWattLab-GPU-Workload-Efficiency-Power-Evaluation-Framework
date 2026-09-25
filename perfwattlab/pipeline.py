import json
import math
import re
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEN_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models():
    """Load and return (embedder, reranker, tokenizer, model, gen_pipe)."""
    embedder = SentenceTransformer(EMBED_MODEL)
    if torch.cuda.is_available():
        embedder = embedder.to("cuda")

    reranker = CrossEncoder(RERANKER_MODEL, device=_DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    from perfwattlab.engine.model_utils import dtype_kwarg
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL,
        device_map="auto",
        **dtype_kwarg(torch.float16 if torch.cuda.is_available() else torch.float32),
    )
    model.eval()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    return embedder, reranker, tokenizer, model, gen_pipe


SAMPLE_DOCS = {
    "doc1.txt": "CUDA is a parallel computing platform and programming model developed by NVIDIA. It enables dramatic increases in computing performance by harnessing the power of the GPU.",
    "doc2.txt": "Triton Inference Server is an open source inference serving software that simplifies deployment of AI models at scale. It supports multiple frameworks and backends.",
    "doc3.txt": "FAISS is a library for efficient similarity search and clustering of dense vectors. It is commonly used for vector search in retrieval augmented generation pipelines.",
    "doc4.txt": "Prometheus is a monitoring system and time series database. Grafana is used to visualize metrics and build dashboards for observability.",
    "doc5.txt": "Dynamic batching combines multiple inference requests into a single batch to improve GPU throughput while maintaining latency constraints.",
    "doc6.txt": "The KV cache stores attention key and value tensors during autoregressive generation, avoiding redundant recomputation of previous tokens.",
    "doc7.txt": "vLLM uses PagedAttention to manage the KV cache like virtual memory, enabling efficient serving of many concurrent sequences on a single GPU.",
}


def ensure_index(data_dir: Path, index_dir: Path, embedder) -> tuple:
    """Write sample docs if data_dir is empty, then build or load the FAISS index."""
    data_dir.mkdir(parents=True, exist_ok=True)
    if not list(data_dir.glob("*.txt")):
        for name, text in SAMPLE_DOCS.items():
            (data_dir / name).write_text(text)
    if (index_dir / "faiss.index").exists():
        return load_index(index_dir)
    return build_index(data_dir, index_dir, embedder)


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def build_index(data_dir: Path, index_dir: Path, embedder) -> tuple:
    """
    Chunk documents in data_dir, embed them, build a FAISS index,
    and save both to index_dir. Returns (index, chunks).
    """
    index_dir.mkdir(parents=True, exist_ok=True)

    def clean_text(s: str) -> str:
        s = s.replace("\u00a0", " ")
        return re.sub(r"\s+", " ", s).strip()

    def chunk_text(text: str, chunk_size: int = 450, overlap: int = 80):
        chunks_out = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk = text[start:end].strip()
            if chunk:
                chunks_out.append(chunk)
            if end == len(text):
                break
            start = end - overlap
        return chunks_out

    docs = []
    for p in sorted(data_dir.glob("*.txt")):
        text = clean_text(p.read_text(errors="ignore"))
        if text:
            docs.append({"doc_id": p.name, "text": text})

    chunks = []
    for d in docs:
        for i, c in enumerate(chunk_text(d["text"])):
            chunks.append({"chunk_id": f"{d['doc_id']}::chunk{i}", "doc_id": d["doc_id"], "text": c})

    texts = [c["text"] for c in chunks]
    emb = embedder.encode(texts, batch_size=64, show_progress_bar=True,
                          convert_to_numpy=True, normalize_embeddings=True)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype(np.float32))

    faiss.write_index(index, str(index_dir / "faiss.index"))
    with open(index_dir / "chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)

    return index, chunks


def load_index(index_dir: Path) -> tuple:
    """Load a previously built FAISS index and chunk list."""
    index = faiss.read_index(str(index_dir / "faiss.index"))
    with open(index_dir / "chunks.json") as f:
        chunks = json.load(f)
    return index, chunks


# ---------------------------------------------------------------------------
# Retrieval and reranking
# ---------------------------------------------------------------------------

def retrieve(query: str, index, chunks, embedder, top_k: int = 10) -> tuple:
    t0 = time.perf_counter()
    q_emb = embedder.encode([query], convert_to_numpy=True,
                             normalize_embeddings=True).astype(np.float32)
    scores, idxs = index.search(q_emb, top_k)
    t1 = time.perf_counter()

    results = []
    for score, i in zip(scores[0], idxs[0]):
        c = chunks[int(i)]
        results.append({"chunk_id": c["chunk_id"], "doc_id": c["doc_id"],
                         "text": c["text"], "score": float(score)})
    return results, (t1 - t0) * 1000.0


def rerank(query: str, retrieved: list, reranker, top_k: int = 5) -> tuple:
    t0 = time.perf_counter()
    pairs = [(query, r["text"]) for r in retrieved]
    scores = reranker.predict(pairs)
    for r, s in zip(retrieved, scores):
        r["rerank_score"] = float(s)
    reranked = sorted(retrieved, key=lambda x: x["rerank_score"], reverse=True)

    seen, deduped = set(), []
    for r in reranked:
        if r["chunk_id"] not in seen:
            seen.add(r["chunk_id"])
            deduped.append(r)

    t1 = time.perf_counter()
    return deduped[:top_k], (t1 - t0) * 1000.0


def build_prompt(query: str, context_chunks: list) -> str:
    context = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(context_chunks)])
    return (
        "You are a helpful assistant. Use the context to answer the question.\n"
        "If the context is not enough, say you are not sure.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    )


# ---------------------------------------------------------------------------
# Generation — baseline path (pipeline API)
# ---------------------------------------------------------------------------

def _gen_kwargs(max_new_tokens, do_sample, temperature, top_p, min_new_tokens):
    kw = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample:
        kw.update(temperature=temperature, top_p=top_p)
    if min_new_tokens:
        kw["min_new_tokens"] = min_new_tokens
    return kw


def generate_pipeline(prompt: str, gen_pipe, tokenizer,
                      max_new_tokens: int = 96,
                      do_sample: bool = False,
                      temperature: float = 0.0,
                      top_p: float = 1.0,
                      min_new_tokens: Optional[int] = None) -> tuple:
    """
    Baseline generation through transformers' TextGenerationPipeline.

    The pipeline wraps the same model.generate() call with its own
    preprocess (tokenize), forward (device placement + inference context +
    generate) and postprocess (decode) stages. Which part of that wrapper
    costs time is an open question (README §4.4, hypotheses H1–H3), so this
    docstring deliberately makes no claim about synchronization.

    Output tokens are counted by re-tokenizing the generated text WITHOUT
    special tokens (v1 added a BOS token here, overcounting by one).
    """
    kw = _gen_kwargs(max_new_tokens, do_sample, temperature, top_p, min_new_tokens)
    t0 = time.perf_counter()
    out = gen_pipe(prompt, return_full_text=False, **kw)[0]["generated_text"]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    gen_tokens = max(1, len(tokenizer.encode(out, add_special_tokens=False)))
    secs = max(t1 - t0, 1e-9)
    return out.strip(), (t1 - t0) * 1000.0, gen_tokens, gen_tokens / secs


def generate_pipeline_staged(prompt: str, gen_pipe, max_new_tokens: int = 96,
                             min_new_tokens: Optional[int] = None) -> dict:
    """
    H2 instrumentation: run the pipeline's three stages by hand and time each.
    Uses the pipeline's own _sanitize_parameters so the kwargs routing matches
    what gen_pipe(prompt, ...) does.
    """
    kw = _gen_kwargs(max_new_tokens, False, 0.0, 1.0, min_new_tokens)
    pre_p, fwd_p, post_p = gen_pipe._sanitize_parameters(return_full_text=False, **kw)
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    t0 = time.perf_counter()
    model_inputs = gen_pipe.preprocess(prompt, **pre_p)
    t1 = time.perf_counter()
    model_outputs = gen_pipe.forward(model_inputs, **fwd_p)
    sync()
    t2 = time.perf_counter()
    gen_pipe.postprocess(model_outputs, **post_p)
    t3 = time.perf_counter()
    return {"preprocess_ms": (t1 - t0) * 1e3, "forward_ms": (t2 - t1) * 1e3,
            "postprocess_ms": (t3 - t2) * 1e3, "total_ms": (t3 - t0) * 1e3,
            "inference_context": getattr(gen_pipe.get_inference_context(), "__name__",
                                         str(gen_pipe.get_inference_context()))}


# ---------------------------------------------------------------------------
# Generation — direct path (model.generate)
# ---------------------------------------------------------------------------

def generate_direct(prompt: str, model, tokenizer,
                    max_new_tokens: int = 96,
                    do_sample: bool = False,
                    temperature: float = 0.0,
                    top_p: float = 1.0,
                    min_new_tokens: Optional[int] = None,
                    grad_ctx: str = "inference_mode") -> tuple:
    """
    Direct model.generate() with explicit device placement.

    grad_ctx: "inference_mode" (the v1 "optimized" path) or "no_grad"
    (the H1 ablation: same path, different autograd context).

    This is still Hugging Face's generate(): attention, KV cache, kernels
    and batching are identical to the pipeline path.
    """
    ctx = torch.inference_mode() if grad_ctx == "inference_mode" else torch.no_grad()
    kw = _gen_kwargs(max_new_tokens, do_sample, temperature, top_p, min_new_tokens)
    with ctx:
        t0 = time.perf_counter()
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        out_ids = model.generate(**inputs, use_cache=True, **kw)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    n_in = inputs["input_ids"].shape[1]
    gen_tokens = max(int(out_ids.shape[1] - n_in), 1)
    secs = max(t1 - t0, 1e-9)
    # decode only the new tokens, matching the pipeline's return_full_text=False
    text = tokenizer.decode(out_ids[0, n_in:], skip_special_tokens=True)
    return text, (t1 - t0) * 1000.0, gen_tokens, gen_tokens / secs


class GenerateRecorder:
    """
    H3 instrumentation: wrap model.generate to record the kwargs each path
    actually passes, so the two paths' generation configs can be diffed.
    """

    def __init__(self, model):
        self.model = model
        self.calls = []
        self._orig = model.generate

    def __enter__(self):
        def wrapped(*args, **kwargs):
            rec = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    rec[k] = f"Tensor{tuple(v.shape)} {v.dtype} {v.device}"
                elif k == "generation_config" and v is not None:
                    rec[k] = v.to_diff_dict()
                else:
                    rec[k] = repr(v)
            self.calls.append(rec)
            return self._orig(*args, **kwargs)
        self.model.generate = wrapped
        return self

    def __exit__(self, *exc):
        self.model.generate = self._orig


# ---------------------------------------------------------------------------
# Full RAG call
# ---------------------------------------------------------------------------

def rag_once(query: str, index, chunks, embedder, reranker, model, tokenizer,
             gen_pipe=None, gen_mode: str = "direct",
             max_new_tokens: int = 160, do_sample: bool = False,
             temperature: float = 0.0, top_p: float = 1.0,
             min_new_tokens: Optional[int] = None, grad_ctx: str = "inference_mode") -> dict:
    """Run one full RAG query and return timing breakdown."""
    retrieved, t_retr = retrieve(query, index, chunks, embedder)
    reranked, t_rer = rerank(query, retrieved, reranker)
    prompt = build_prompt(query, reranked)

    if gen_mode == "pipeline":
        text, t_gen, gen_tokens, tps = generate_pipeline(
            prompt, gen_pipe, tokenizer,
            max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature, top_p=top_p, min_new_tokens=min_new_tokens,
        )
    else:
        text, t_gen, gen_tokens, tps = generate_direct(
            prompt, model, tokenizer,
            max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature, top_p=top_p, min_new_tokens=min_new_tokens,
            grad_ctx=grad_ctx,
        )

    return {
        "query": query,
        "retrieval_ms": round(t_retr, 2),
        "rerank_ms": round(t_rer, 2),
        "generation_ms": round(t_gen, 2),
        "total_ms": round(t_retr + t_rer + t_gen, 2),
        "gen_tokens": gen_tokens,
        "toks_per_sec": round(tps, 2),
        "prompt": prompt,
        "answer_preview": text[-300:],
    }
