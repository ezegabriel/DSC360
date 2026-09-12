# === STUDENT INSTRUCTIONS =======================================================
# ask.py — Doc2Query RAG: Query the built index
#
# Implement:
#   embed_questions(texts: list[str], model: str) -> list[np.ndarray]
#
# What you write (list-only embedding function):
#   • Accept a list of strings and a model name.
#   • Call the provided Ollama embedding backend; pass the model name through.
#   • Return a list of NumPy arrays (dtype float32), one per input, preserving order.
#   • Do NOT normalize/reshape vectors; return them as provided by the backend.
#   • If the backend returns no embeddings, return [].
#
# What to do (overall flow for this problem pair):
#   1) Implement embed_questions() here in ask.py.
#   2) Ensure embed_questions() is also implemented in index.py.
#   3) Run:  python3 index.py   (builds questions + embeddings + on-disk index)
#   4) Then: python3 ask.py     (loads index, retrieves, answers with citations)
#
# Hints:
#   • You can iterate quickly by first building a small index (e.g., after slicing
#     in index.py). For your final run, restore the original behavior and rebuild.
#
# Only your embed_questions() implementation is graded in this file.
# =================================================================================

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from pydantic import BaseModel, Field, ValidationError
import ollama


# ---------------------------
# Config / schema
# ---------------------------

class QAResult(BaseModel):
    abstain: bool = Field(..., description="True if the answer cannot be supported by the provided excerpts.")
    answer: str = Field(..., description="Short, direct answer or 'I don't know' if abstaining.")
    supporting_chunks: List[int] = Field(
        default_factory=list,
        description="List of chunk_idx values (from the provided context) that directly support the answer."
    )


SYSTEM_PROMPT = """You are a precise literature tutor answering questions ONLY from the provided excerpts of
Edgar Allan Poe's 'The Tell-Tale Heart'. Output strictly valid JSON for the given schema.

Decision rules:
- Answer whenever any excerpt contains evidence (even if wording differs via synonyms).
- Only set abstain=true when NONE of the excerpts contain information that addresses the question.
- Treat everyday phrasing and synonyms as equivalent.
- For “how many / how much / which / who” questions, extract the value directly from the excerpts when present.
- Be concise: 1–2 sentences for explanations; for count/name questions, a single short phrase is fine.
- If you answer, include at least one supporting_chunks id that points to the excerpt used.
- Do NOT invent facts beyond the excerpts. If you are speculating, say so.
- You can be a little bit imaginative. However, if the question or request is completely off-topic, you should abstain."""


# =========================
# Embedding
# =========================

# ===== STUDENT TODO ======================================================
def embed_questions(texts: List[str], model: str) -> List[np.ndarray]:
    """
    Implement a list-only embedding function.
    Accept a list of strings and a model name; return a list of np.ndarray
    (dtype float32), one per input, preserving order. Use the provided
    embedding backend; do not hard-code the model.
    """

    embeddings = []
    for text in texts:
        response = ollama.embed(model=model, input=text)
        embeddings.append(np.array(response["embeddings"][0], dtype=np.float32))
    # Placeholder stub — replace with your implementation.
    return embeddings
# =========================================================================


# ---------------------------
# I/O helpers
# ---------------------------

def load_index(index_dir: str) -> Tuple[List[str], np.ndarray, List[Dict[str, Any]], Dict[int, int]]:
    """
    Load chunks (list[str]), embeddings matrix (N x d), metadata rows (list[dict]).
    Also returns a dict: first_line_for_chunk[chunk_idx] -> approx_line (min per chunk).
    """
    idx = Path(index_dir)
    chunks = json.loads((idx / "chunks.json").read_text(encoding="utf-8"))["chunks"]
    M = np.load(idx / "embeddings.npy")
    rows: List[Dict[str, Any]] = []
    first_line_for_chunk: Dict[int, int] = {}
    with (idx / "metadata.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rows.append(rec)
            cidx = rec["chunk_idx"]
            line_no = rec.get("approx_line")
            if line_no is not None:
                if cidx not in first_line_for_chunk:
                    first_line_for_chunk[cidx] = int(line_no)
                else:
                    first_line_for_chunk[cidx] = min(first_line_for_chunk[cidx], int(line_no))
    return chunks, M, rows, first_line_for_chunk


# ---------------------------
# Retrieval
# ---------------------------

def top_k_rows(M: np.ndarray, q: np.ndarray, k: int = 10) -> List[int]:
    """
    Given row-normalized M (N x d) and a query vector q, return the top-k row indices by cosine (= dot product).
    """
    if M.size == 0:
        return []
    q = q.astype(np.float32)
    q /= (np.linalg.norm(q) + 1e-12)
    sims = M @ q  # (N,)
    if k >= len(sims):
        order = np.argsort(-sims)
    else:
        part = np.argpartition(-sims, k - 1)[:k]
        order = part[np.argsort(-sims[part])]
    return order.tolist()


def gather_context(top_rows_idx: List[int],
                   meta_rows: List[Dict[str, Any]],
                   chunks: List[str],
                   first_line_for_chunk: Dict[int, int],
                   max_chunks: int = 10,
                   max_chars_per_chunk: int = 800) -> List[Dict[str, Any]]:
    """
    Map top question rows to unique chunk_idx entries (preserving rank order).
    Returns a list of context dicts: {chunk_idx, approx_line, text}
    """
    seen = set()
    context: List[Dict[str, Any]] = []
    for r in top_rows_idx:
        rec = meta_rows[r]
        cidx = rec["chunk_idx"]
        if cidx in seen:
            continue
        seen.add(cidx)
        approx_line = first_line_for_chunk.get(cidx, None)
        text = chunks[cidx]
        if max_chars_per_chunk and len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + " …"
        context.append({"chunk_idx": cidx, "approx_line": approx_line, "text": text})
        if len(context) >= max_chunks:
            break
    return context


# ---------------------------
# Generation
# ---------------------------

def answer_with_context(question: str,
                        context: List[Dict[str, Any]],
                        gen_model: str,
                        allowed_chunk_ids: List[int]) -> QAResult | None:
    """
    Ask the LLM for a grounded answer constrained to provided chunks.
    Returns a parsed QAResult or None on failure.
    """
    # Build a compact context block
    lines = []
    for c in context:
        hdr = f"[chunk_idx {c['chunk_idx']} | line~{c.get('approx_line','?')}]"
        lines.append(hdr)
        lines.append(c["text"])
    context_block = "\n\n".join(lines)

    user_prompt = textwrap.dedent(f"""\
    You are given EXCERPTS (unordered) from the story. Only use these to answer.

    EXCERPTS START
    {context_block}
    EXCERPTS END

    QUESTION: {question}

    Respond with valid JSON per schema. supporting_chunks must be a subset of: {allowed_chunk_ids}.
    If the excerpts don't answer, abstain.
    """)

    try:
        resp = ollama.chat(
            model=gen_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            format=QAResult.model_json_schema(),
            options={"temperature": 0.0},
        )
        raw = getattr(resp, "message", None)
        raw_content = getattr(raw, "content", None) if raw else None
        if raw_content is None and isinstance(resp, dict):
            raw_content = resp.get("message", {}).get("content")
        if not raw_content:
            print("[chat] Empty response.", file=sys.stderr)
            return None
        return QAResult.model_validate_json(raw_content)
    except ValidationError as e:
        print(f"[parse] ValidationError: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[chat] Error: {e}", file=sys.stderr)
        return None


# ---------------------------
# Rendering
# ---------------------------

def render_result(res: QAResult, context: List[Dict[str, Any]]) -> None:
    print("\n=== Answer ===")
    if res.abstain:
        print("This is not a question I can answer. [INSUFFICIENT EVIDENCE]")
    else:
        print(res.answer.strip())

    if res.supporting_chunks:
        print("\n--- Supporting chunk(s) ---")
        by_idx = {c["chunk_idx"]: c for c in context}
        for cidx in res.supporting_chunks:
            c = by_idx.get(cidx)
            if not c:
                continue
            line_info = f"~line {c.get('approx_line')}" if c.get('approx_line') else "line ?"
            snippet = c["text"].replace("\n", " ")
            if len(snippet) > 240:
                snippet = snippet[:240] + " …"
            print(f"* chunk_idx {cidx} ({line_info}): {snippet}")
    print()


# ---------------------------
# REPL
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Ask grounded questions about 'The Tell-Tale Heart' using the doc2query index.")
    ap.add_argument("--index", default="index_tell_tale", help="Path to the built index directory")
    ap.add_argument("--gen", default="gemma3:4b", help="Ollama model for question generation")
    ap.add_argument("--embed", default="mxbai-embed-large", help="Ollama embedding model")
    ap.add_argument("--k", type=int, default=10, help="Number of nearest question-embeddings to consider")
    ap.add_argument("--threshold", type=float, default=0.25, help="Min cosine similarity for top-1 to proceed; else abstain")
    args = ap.parse_args()

    # Load index
    try:
        chunks, M, meta_rows, first_line_for_chunk = load_index(args.index)
    except FileNotFoundError as e:
        print(f"[load] {e}", file=sys.stderr)
        sys.exit(1)

    if M.size == 0:
        print("[load] Empty embeddings matrix.", file=sys.stderr)
        sys.exit(1)

    print(f"Index loaded. Embedder: {args.embed}. Generator: {args.gen}")

    print("RAG REPL ready. Type your question, or /q to quit.\n")
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            print()
            break
        if not q:
            continue
        if q.lower() in {"/q", "/quit"}:
            break

        # Embed query
        q_vecs = embed_questions([q], model=args.embed)
        if not q_vecs:
            print("(!) Could not embed your query.")
            continue
        qv = q_vecs[0]

        # Retrieve (cosine via dot since M rows are ~unit length)
        qv_n = qv / (np.linalg.norm(qv) + 1e-12)
        sims = M @ qv_n                                # (num_rows,)
        order = np.argsort(sims)[::-1]                 # descending
        topk = order[:args.k]
        top1_sim = float(sims[topk[0]])

        # r0 = int(topk[0])
        # print(f"[debug] top1 row={r0} sim={top1_sim:.3f}")
        # print(f"[debug] matched question: {meta_rows[r0]['question']!r}")
        # print(f"[debug] matched chunk snippet: {chunks[meta_rows[r0]['chunk_idx']][:140].replace('\n',' ')}…")

        # Threshold on top-1
        if top1_sim < args.threshold:
            print("(abstain) Your question doesn't match the indexed material closely enough.")
            continue

        # Gather context from TOP-K (unique chunk_idx, preserve rank)
        cand_row_ids = topk.tolist()
        context = gather_context(cand_row_ids, meta_rows, chunks, first_line_for_chunk, max_chunks=args.k)

        allowed_chunk_ids = [c["chunk_idx"] for c in context]
        res = answer_with_context(q, context, gen_model=args.gen, allowed_chunk_ids=allowed_chunk_ids)
        if res is None:
            print("(!) Unable to produce a structured answer.")
            continue

        render_result(res, context)

    print("bye.")


if __name__ == "__main__":
    main()
