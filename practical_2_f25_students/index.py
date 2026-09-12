# === STUDENT INSTRUCTIONS =======================================================
# index.py — Doc2Query RAG: Build the question index
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
#   1) Implement embed_questions() here in index.py.
#   2) Implement embed_questions() in ask.py.
#   3) Fully run:  python3 index.py   (builds questions + embeddings + on-disk index)
#   4) Then run:   python3 ask.py     (loads index, retrieves, answers with citations)
#
# Hints:
#   • For faster iteration while editing, you may TEMPORARILY limit work to the
#     first few chunks (e.g., slice the texts list) to speed up indexing runs.
#     Remember to restore the original behavior before your final submission.
#
# What is Doc2Query?
#   Doc2Query prompts an LLM to generate likely user questions for each chunk,
#   then embeds those questions (instead of raw text) to improve recall for
#   paraphrased queries. It often boosts matching when wording differs.
#
# Only your embed_questions() implementation is graded in this file.
# =================================================================================

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from pydantic import BaseModel, Field, ValidationError

# --- Ollama client(s) ---
import ollama

# =========================
# Models / Schemas
# =========================

class Question(BaseModel):
    text: str = Field(..., description="A single, specific recall/comprehension question answerable strictly from the current excerpt.")

class QuestionList(BaseModel):
    questions: List[Question]

# =========================
# Chunking helpers
# =========================

def chunker(full_text: str) -> List[str]:
    """
    Split by blank lines into trimmed paragraph chunks.
    Return list[str]. main() will drop chunk[0] assuming it's the title line(s).
    """
    # Normalize newlines, split on >=1 blank lines
    parts = [p.strip() for p in full_text.replace("\r\n", "\n").replace("\r", "\n").split("\n\n") if p.strip()]
    return parts

def approx_start_lines(full_text: str, chunks: List[str]) -> List[int]:
    """
    Roughly map each chunk to a 1-based starting line number in full_text.
    We scan forward to find each chunk in sequence to avoid false matches.
    """
    lines_prefix_count = 0
    cursor = 0
    line_numbers = []
    haystack = full_text

    for ch in chunks:
        # Find next occurrence of the chunk text after 'cursor'
        idx = haystack.find(ch, cursor)
        if idx < 0:
            # Fallback: if not found, reuse last known line number + 1
            line_numbers.append(max(1, (line_numbers[-1] + 1) if line_numbers else 1))
            continue
        # Count newlines before idx
        segment = haystack[cursor:idx]
        lines_prefix_count += segment.count("\n")
        line_numbers.append(max(1, lines_prefix_count + 1))
        # Advance cursor to end of this chunk
        cursor = idx + len(ch)

    return line_numbers

# =========================
# LLM question generation
# =========================

SYSTEM_PROMPT = """You are a meticulous English teacher writing short, specific recall/comprehension questions.
Rules:
- ONLY ask about the CURRENT EXCERPT.
- Each question must be answerable unambiguously from that excerpt alone.
- Avoid interpretation, opinion, or spoilers from other parts of the story.
- Keep questions concise (one sentence each).
- Return ONLY valid JSON per the provided schema.
"""

def make_questions_for_chunk(
    current_chunk: str,
    prev_chunks: List[str],
    model: str,
    k_ctx: int = 5,
    temperature: float = 0.0,
) -> List[str]:
    """
    Use Ollama chat() with Pydantic schema to get a list of questions for the current chunk.
    We provide up to K previous chunks as 'context' but strictly instruct to ask only about CURRENT.
    """
    # Compose the user prompt with previous excerpts (context only) + current
    prev_text = "\n\n---\n\n".join(prev_chunks[-k_ctx:]) if prev_chunks else ""
    user_parts = []
    if prev_text:
        user_parts.append("Previous excerpts (context only; do NOT ask about these):\n" + prev_text)
    user_parts.append("CURRENT EXCERPT (ask questions ONLY about this):\n" + current_chunk)
    user_parts.append(
        "Task: Write 5–8 precise recall questions that a student can answer using ONLY the CURRENT EXCERPT."
    )
    user_prompt = "\n\n".join(user_parts)

    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            # Ask Ollama to adhere to our JSON schema
            format=QuestionList.model_json_schema(),
            options={"temperature": temperature},
        )
        raw = resp.message.content  # ollama >=0.3 returns an object with .message.content
        qlist = QuestionList.model_validate_json(raw)
        return [q.text.strip() for q in qlist.questions if q.text.strip()]
    except ValidationError as e:
        print(f"[parse] ValidationError for chunk: {e}", file=sys.stderr)
        # If the model emitted non-JSON or wrong schema, fall back to best-effort strip
        return []
    except Exception as e:
        print(f"[chat] Error generating questions: {e}", file=sys.stderr)
        return []

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

# =========================
# Index builder
# =========================

@dataclass
class BuildConfig:
    gen_model: str
    embed_model: str
    k_history: int
    outdir: Path

def save_index(outdir: Path,
               chunks: List[str],
               all_embs: List[np.ndarray],
               meta_records: List[dict],
               cfg: BuildConfig,
               n_total_q: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Write chunks
    with (outdir / "chunks.json").open("w", encoding="utf-8") as f:
        json.dump({"chunks": chunks}, f, ensure_ascii=False, indent=2)

    # Write metadata
    with (outdir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for rec in meta_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write embeddings (L2-normalized rows)
    if not all_embs:
        np.save(outdir / "embeddings.npy", np.zeros((0, 0), dtype=np.float32))
    else:
        M = np.vstack(all_embs).astype(np.float32)
        norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
        M /= norms
        np.save(outdir / "embeddings.npy", M)

    # Manifest
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gen_model": cfg.gen_model,
        "embed_model": cfg.embed_model,
        "k_history": cfg.k_history,
        "n_chunks": len(chunks),
        "n_questions": int(n_total_q),
        "files": ["chunks.json", "metadata.jsonl", "embeddings.npy"],
        "notes": "Embeddings are row-normalized; use dot-product for cosine similarity.",
    }
    with (outdir / "index_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def doc2query(chunks: List[str], full_text: str, cfg: BuildConfig) -> None:
    """
    Two-pass build:
      Pass A: generate all questions (generator stays resident).
      Pass B: batch-embed all questions once (embedder stays resident).
    """
    line_map = approx_start_lines(full_text, chunks)

    # ---- Pass A: question generation (no embeddings yet) ----
    history = deque(maxlen=cfg.k_history)
    records: List[dict] = []   # [{chunk_idx, approx_line, question}, ...]
    n_total_q = 0

    for idx, paragraph in enumerate(chunks):
        questions = make_questions_for_chunk(
            current_chunk=paragraph,
            prev_chunks=list(history),
            model=cfg.gen_model,
            k_ctx=cfg.k_history,
            temperature=0.0,
        )

        if not questions:
            print(f"[{idx:03d}] q=0")
            history.append(paragraph)
            continue

        for q in questions:
            records.append({
                "chunk_idx": idx,
                "approx_line": int(line_map[idx]),
                "question": q,
            })
        n_q = len(questions)
        n_total_q += n_q
        print(f"[{idx:03d}] q={n_q}")
        history.append(paragraph)

    if not records:
        # Nothing to embed; persist an empty index for completeness.
        save_index(cfg.outdir, chunks, [], [], cfg, 0)
        print(f"[done] 0 questions over {len(chunks)} chunks → {cfg.outdir}")
        return

    # ---- Pass B: batch embeddings (single embedder load) ----
    BATCH = 32 # tune as needed (256–1024 fine on lab machines)
    all_embs: List[np.ndarray] = []
    total = len(records)

    # Embed in order to keep row alignment trivial
    for start in range(0, total, BATCH):
        end = min(start + BATCH, total)
        batch_texts = [rec["question"] for rec in records[start:end]]

        vecs = embed_questions(batch_texts, model=cfg.embed_model)
        if not vecs:
            # Backend returned nothing; skip but keep going (rare).
            print(f"[emb] {end}/{total} (warning: empty batch)")
            continue

        all_embs.extend(vecs)
        print(f"[emb] {end}/{total}")

    # Safety: ensure 1:1 mapping questions↔embeddings
    n = min(len(all_embs), len(records))
    if n != len(records):
        print(f"[warn] embeddings count mismatch ({len(all_embs)} vs {len(records)}); truncating to {n}.")
        all_embs = all_embs[:n]
        records = records[:n]
        n_total_q = n

    # Build meta rows aligned with all_embs order
    meta_records: List[dict] = []
    for row, rec in enumerate(records):
        meta_records.append({
            "row": row,
            "chunk_idx": rec["chunk_idx"],
            "approx_line": rec["approx_line"],
            "question": rec["question"],
        })

    save_index(cfg.outdir, chunks, all_embs, meta_records, cfg, n_total_q)
    print(f"[done] {n_total_q} questions over {len(chunks)} chunks → {cfg.outdir}")

    
# =========================
# CLI / main
# =========================

def main():
    ap = argparse.ArgumentParser(description="Build a doc2query index for Poe's 'The Tell-Tale Heart'.")
    ap.add_argument("input", nargs="?", default="heart.txt", help="Path to heart.txt")
    ap.add_argument("--outdir", default="index_tell_tale", help="Output directory for index files")
    ap.add_argument("--gen", default="gemma3:4b", help="Ollama model for question generation")
    ap.add_argument("--embed", default="mxbai-embed-large", help="Ollama embedding model")
    ap.add_argument("--k", type=int, default=5, help="# of previous chunks to include as context (history)")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Indexing file {args.input}. Embedder: {args.embed}. Generator: {args.gen}")
        
    full_text = in_path.read_text(encoding="utf-8")
    raw_chunks = chunker(full_text)

    if not raw_chunks:
        print("No chunks found; is the input empty?", file=sys.stderr)
        sys.exit(1)

    # Drop the first chunk (title)
    chunks = raw_chunks[1:] if len(raw_chunks) > 1 else []

    if not chunks:
        print("After dropping the first chunk (title), there are no paragraphs to index.", file=sys.stderr)
        sys.exit(1)

    cfg = BuildConfig(
        gen_model=args.gen,
        embed_model=args.embed,
        k_history=args.k,
        outdir=Path(args.outdir),
    )

    doc2query(chunks=chunks, full_text=full_text, cfg=cfg)

if __name__ == "__main__":
    main()
