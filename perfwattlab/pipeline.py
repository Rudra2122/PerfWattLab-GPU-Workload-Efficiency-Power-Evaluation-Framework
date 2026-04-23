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
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    return embedder, reranker, tokenizer, model, gen_pipe


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

def generate_pipeline(prompt: str, gen_pipe, tokenizer,
                      max_new_tokens: int = 96,
                      do_sample: bool = False,
                      temperature: float = 0.0,
                      top_p: float = 1.0) -> tuple:
    """
    Baseline generation using the HuggingFace pipeline API.

    This path adds CPU-side overhead and extra tensor handling around the model
    call. In torch.profiler traces, this shows up as synchronization stalls
    between CPU and GPU — the GPU sits idle waiting for the CPU wrapper to
    hand off the next operation.
    """
    t0 = time.perf_counter()
    out = gen_pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 0.0,
        top_p=top_p if do_sample else 1.0,
        return_full_text=False,
    )[0]["generated_text"]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    gen_tokens = max(1, len(tokenizer.encode(out)))
    secs = max(t1 - t0, 1e-9)
    return out.strip(), (t1 - t0) * 1000.0, gen_tokens, gen_tokens / secs


# ---------------------------------------------------------------------------
# Generation — optimized path (direct model.generate)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_direct(prompt: str, model, tokenizer,
                    max_new_tokens: int = 96,
                    do_sample: bool = False,
                    temperature: float = 0.0,
                    top_p: float = 1.0) -> tuple:
    """
    Optimized generation calling model.generate() directly under inference_mode.

    This removes the pipeline wrapper overhead, reduces unnecessary CPU-GPU
    transfers, and gives the GPU a continuous execution path. The fix that
    produced the 16% latency reduction in this project.
    """
    t0 = time.perf_counter()
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        use_cache=True,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    gen_tokens = max(int(out_ids.shape[1] - inputs["input_ids"].shape[1]), 1)
    secs = max(t1 - t0, 1e-9)
    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return text, (t1 - t0) * 1000.0, gen_tokens, gen_tokens / secs


# ---------------------------------------------------------------------------
# Full RAG call
# ---------------------------------------------------------------------------

def rag_once(query: str, index, chunks, embedder, reranker, model, tokenizer,
             gen_pipe=None, gen_mode: str = "direct",
             max_new_tokens: int = 160, do_sample: bool = False,
             temperature: float = 0.0, top_p: float = 1.0) -> dict:
    """Run one full RAG query and return timing breakdown."""
    retrieved, t_retr = retrieve(query, index, chunks, embedder)
    reranked, t_rer = rerank(query, retrieved, reranker)
    prompt = build_prompt(query, reranked)

    if gen_mode == "pipeline":
        text, t_gen, gen_tokens, tps = generate_pipeline(
            prompt, gen_pipe, tokenizer,
            max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature, top_p=top_p,
        )
    else:
        text, t_gen, gen_tokens, tps = generate_direct(
            prompt, model, tokenizer,
            max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature, top_p=top_p,
        )

    return {
        "query": query,
        "retrieval_ms": round(t_retr, 2),
        "rerank_ms": round(t_rer, 2),
        "generation_ms": round(t_gen, 2),
        "total_ms": round(t_retr + t_rer + t_gen, 2),
        "gen_tokens": gen_tokens,
        "toks_per_sec": round(tps, 2),
        "answer_preview": text[-300:],
    }
