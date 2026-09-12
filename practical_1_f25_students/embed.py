#!/usr/bin/env python3
"""
embed.py — shared helper functions for Lab 03 / RAG.
Used by rag.py to load the precomputed index, embed queries,
and compute cosine similarity scores.
"""

from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import ollama

# --- Paths ---
INDEX_DIR   = Path("index")
EMB_PATH    = INDEX_DIR / "embeddings.npy"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
META_PATH   = INDEX_DIR / "meta.json"

# --- Core helpers ---

def load_index():
    """Load embeddings, chunks, and metadata from index/ folder."""
    if not (EMB_PATH.exists() and CHUNKS_PATH.exists() and META_PATH.exists()):
        print("ERROR: missing index files. Run build_index.py first.", file=sys.stderr)
        sys.exit(1)
    embs = np.load(EMB_PATH)  # (N, d) float32
    chunks = [json.loads(line) for line in open(CHUNKS_PATH, "r", encoding="utf-8")]
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    if embs.shape[0] != len(chunks):
        print("ERROR: embeddings count != chunks count.", file=sys.stderr)
        sys.exit(1)
    return embs, chunks, meta

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec if n == 0 else vec / n

def embed_query(model: str, text: str) -> np.ndarray:
    """Generate an embedding vector for the given text via Ollama."""
    resp = ollama.embed(model=model, input=text)
    q = np.array(resp["embeddings"][0], dtype=np.float32)
    return q

def cosine_scores(E: np.ndarray, q: np.ndarray, assume_normalized: bool) -> np.ndarray:
    """Return cosine similarity scores for each row of E vs query vector q."""
    qn = l2_normalize(q)
    if assume_normalized:
        return E @ qn
    En = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)
    return En @ qn

def overlaps(a: dict, b: dict) -> bool:
    """True if character spans intersect."""
    return not (a["end"] <= b["start"] or b["end"] <= a["start"])

def collapse_overlapping_hits(ranked_idxs: list[int], chunks: list[dict], k: int) -> list[int]:
    """Remove overlapping chunk hits and return top-k distinct indices."""
    result = []
    for idx in ranked_idxs:
        if len(result) >= k:
            break
        this_chunk = chunks[idx]
        if any(overlaps(this_chunk, chunks[j]) for j in result):
            continue
        result.append(idx)
    return result

# --- New: retrieval that returns TEXT STRINGS ONLY ---------------------

def retrieve_texts(E, chunks, meta, question: str, k: int = 4, min_sim: float = 0.25) -> list[str]:
    """
    Return up to k non-overlapping chunk texts relevant to the question,
    filtered by cosine similarity >= min_sim. Strings only; no metadata.
    """
    emb_model    = meta.get("model", "mxbai-embed-large")
    assume_norm  = bool(meta.get("normalize", True))
    q            = embed_query(emb_model, question)
    scores       = cosine_scores(E, q, assume_norm)
    ranked       = np.argsort(-scores)  # descending

    # Threshold + de-overlap
    ranked_filtered = [i for i in ranked.tolist() if scores[i] >= min_sim]
    top_idxs        = collapse_overlapping_hits(ranked_filtered, chunks, k)

    # Just the text strings
    return [chunks[i]["text"] for i in top_idxs]
