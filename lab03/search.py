import json
import numpy as np
import os
import sys
import ollama
from pathlib import Path

INDEX_DIR = "index/"
TOP_K = 5

def load_index(index_dir=INDEX_DIR):
    meta_path = Path(index_dir) / "meta.json"
    chunks_path = Path(index_dir) / "chunks.jsonl"
    embeddings_path = Path(index_dir) / "embeddings.npy"

    if not meta_path.exists() or not chunks_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError("One or more index files are missing in 'index/' directory")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    chunks = []
    with open(chunks_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            chunks.append(json.loads(line))

    embeddings = np.load(embeddings_path)

    return meta, chunks, embeddings


def l2_normalize_rows(mat, eps=1e-12):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return mat / norms


def embed_query(text, model="mxbai-embed-large"):
    resp = ollama.embed(model=model, input=text)
    vec = np.array(resp["embeddings"][0], dtype=np.float32)
    return vec


def dedupe_by_span(top_indices, chunks):
    """
    Deduplicate overlapping chunks by span. Rule:
    - If two chunks overlap in (start,end) span, keep the one with the higher score.
    - We assume `top_indices` is ordered by descending score.
    """
    kept = []
    spans = []  # list of (start,end)
    for idx in top_indices:
        c = chunks[idx]
        start = c.get("start")
        end = c.get("end")
        overlap = False

        if start is None or end is None:
            # If no span info, keep the chunk (cannot dedupe)
            kept.append(idx)
            continue

        for s, e in spans:
            # overlap if ranges intersect
            if not (end <= s or start >= e):
                overlap = True
                break

        if not overlap:
            kept.append(idx)
            spans.append((start, end))
            
    return kept


def run_repl(index_dir=INDEX_DIR, top_k=TOP_K):
    meta, chunks, embeddings = load_index(index_dir)

    # If embeddings were saved non-normalized but meta.normalize says True/False, handle accordingly
    normalize_flag = meta.get("normalize", False)
    if normalize_flag:
        # assume rows already normalized
        E = embeddings.astype(np.float32)
    else:
        E = l2_normalize_rows(embeddings.astype(np.float32))

    dim = E.shape[1]
    print(f"Loaded index: N={E.shape[0]}, d={dim}, model={meta.get('model')}, normalize={normalize_flag}")

    try:
        while True:
            query = input("query> ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("bye")
                break

            query_vector = embed_query(query, model=meta.get("model", "mxbai-embed-large"))
            query_vector = query_vector.astype(np.float32)

            # normalize query
            query_norm = np.linalg.norm(query_vector)
            if query_norm == 0:
                print("zero-length query embedding")
                continue
            query_vector = query_vector / query_norm

            # compute scores: E @ q
            scores = E @ query_vector
            # get top indices sorted by descending score
            top_idx = np.argsort(-scores)

            # take more than top_k to allow deduping
            candidate = top_idx[: min(len(top_idx), top_k * 4)]
            deduped = dedupe_by_span(candidate, chunks)
            # finally select top_k from deduped
            final = deduped[:top_k]

            print("\nResults:")
            for rank, idx in enumerate(final, start=1):
                score = float(scores[idx])
                c = chunks[idx]
                text = c.get("text", "")
                # snippet = text.replace("\n", " ")[:200]
                snippet = text.replace("\n", " ")[:]
                start = c.get("start")
                end = c.get("end")
                print(f"{rank}. id={idx} score={score:.4f} span=({start},{end})")
                print(f"   {snippet}")
            print("")
    except KeyboardInterrupt:
        print("\nbye")


def execute_query(query, index_dir, top_k=TOP_K):
    meta, chunks, embeddings = load_index(index_dir)
    
    # If embeddings were saved non-normalized but meta.normalize says True/False, handle accordingly
    normalize_flag = meta.get("normalize", False)
    if normalize_flag:
        # assume rows already normalized
        E = embeddings.astype(np.float32)
    else:
        E = l2_normalize_rows(embeddings.astype(np.float32))
    

    print(f"Processing query {query}")
    query_vector = embed_query(query['text'], model=meta["model"]).astype(np.float32)
    query_normalized = query_vector / np.linalg.norm(query_vector)

    scores = E @ query_normalized
    top_idx = np.argsort(-scores)

    candidate = top_idx[: min(len(top_idx), top_k * 4)]
    deduped = dedupe_by_span(candidate, chunks)
    final = deduped[:top_k]

    results = []
    for rank, idx in enumerate(final, start=1):
        result = {
            "rank": rank,
            "chunk": chunks[idx],
            "score": float(scores[idx])
        }
        results.append(result)

    return results






if __name__ == "__main__":
    run_repl()
