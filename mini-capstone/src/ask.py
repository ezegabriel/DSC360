"""
ask.py

REPL for the Student Handbook doc2query system.

Per query:
  1. Embed user question.
  2. Find top-k synthetic doc2query questions (cosine similarity).
  3. Map to chunks and build a context from the best few chunks.
  4. If similarity too low: say "Insufficient context".
  5. Else: call generator LLM with context and print answer + citations.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import requests
import re

# ---- configuration ---------------------------------------------------

OLLAMA_URL = "http://localhost:11434"

EMBED_MODEL = "mxbai-embed-large:latest"
GENERATOR_MODEL = "qwen2.5:1.5b"

TOP_K_QUESTIONS = 5      # how many synthetic questions to consider
MAX_CHUNKS = 1           # how many distinct chunks to feed to the LLM
MIN_SIMILARITY = 0.68    # below this: "insufficient context"




# ---- basic helpers ---------------------------------------------------

def load_questions(path: Path) -> List[Dict]:
    questions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions


def load_chunks(path: Path) -> Dict[str, Dict]:
    chunks_by_id: Dict[str, Dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chunks_by_id[rec["chunk_id"]] = rec
    return chunks_by_id


def load_meta(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ollama_embed(model: str, text: str) -> List[float]:
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["embedding"]


def ollama_chat(model: str, prompt: str) -> str:
    # non-streaming response so resp.json() works
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


def cosine_sim_matrix(vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """
    vec: shape (d,)
    mat: shape (N, d)
    returns: shape (N,) similarities
    """
    # add a tiny epsilon to avoid division by zero
    vec_norm = np.linalg.norm(vec) + 1e-8
    mat_norms = np.linalg.norm(mat, axis=1) + 1e-8
    sims = (mat @ vec) / (mat_norms * vec_norm)
    return sims


# ---- retrieval + answer generation -----------------------------------

def retrieve_chunks_for_query(
    user_q: str,
    emb_matrix: np.ndarray,
    questions: List[Dict],
    chunks_by_id: Dict[str, Dict],
) -> Tuple[float, List[Dict]]:
    """
    Returns:
        max_sim: maximum similarity
        context_chunks: list of distinct chunk dicts (up to MAX_CHUNKS)
    """
    q_vec = np.array(ollama_embed(EMBED_MODEL, user_q), dtype="float32")
    sims = cosine_sim_matrix(q_vec, emb_matrix)

    # top-k indices
    top_indices = np.argsort(-sims)[:TOP_K_QUESTIONS]

    max_sim = float(sims[top_indices[0]]) if len(top_indices) > 0 else 0.0

    # map to chunks, dedupe by chunk_id, preserve order
    seen = set()
    context_chunks: List[Dict] = []

    for idx in top_indices:
        q_rec = questions[int(idx)]
        chunk_id = q_rec["chunk_id"]
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        chunk = chunks_by_id.get(chunk_id)
        if chunk is not None:
            context_chunks.append(chunk)
        if len(context_chunks) >= MAX_CHUNKS:
            break
    return max_sim, context_chunks


def build_context_prompt(user_q: str, context_chunks: List[Dict]) -> str:
    ch = context_chunks[0]
    context_text = ch["text"]

    prompt = f"""
Follow the rules below exactly. Do not violate or ignore any rule.

RULES (read carefully):

1. You must output exactly ONE of:
   (a) an answer based ONLY on user enquiry and all relevant details in the excerpt,
   (b) the exact string: "I cannot help with that as it would violate college policy.",
   (c) the exact string: "Insufficient context in the handbook sections I know."


2. If the question explicitly seeks tips, strategies, or methods to break or sneak around college guidelines,
   respond ONLY with line (b).


3. If the question seeks to manipulate a specific person or group,
   respond ONLY with line (b).


4. If the excerpt includes policy language that directly relates to the question,
   respond with line (a).


5. If the excerpt does NOT include policy language that directly relates to the
   question, respond ONLY with line (c).


6. You must NOT guess, improvise, add assumptions, or supply information not in
   the excerpt.


7. You must NOT mention anything about the rules, excerpt, or knowledge base.


--- EXCERPT START ---
{context_text}
--- EXCERPT END ---

USER QUESTION:
{user_q}

YOUR RESPONSE:
"""
    return prompt





def answer_query(user_q: str,
                 emb_matrix: np.ndarray,
                 questions: List[Dict],
                 chunks_by_id: Dict[str, Dict]) -> None:
    max_sim, context_chunks = retrieve_chunks_for_query(
        user_q, emb_matrix, questions, chunks_by_id
    )

    if max_sim < MIN_SIMILARITY or not context_chunks:
        print("Insufficient context in the handbook sections I know.\n")
        return

    
    # if we get here, both gates are satisfied → call LLM
    prompt = build_context_prompt(user_q, context_chunks)
    answer = ollama_chat(GENERATOR_MODEL, prompt)

    print(answer.strip())
    # citations
    print("\nSources:")
    for ch in context_chunks:
        title = ch.get("section_title", "")
        url = ch.get("url", "")
        if url:
            print(f"- {title} – Student Handbook ({url})")
        else:
            print(f"- {title} – Student Handbook")
    print()




# ---- main REPL -------------------------------------------------------

def main():
    root = Path(__file__).resolve().parent.parent
    index_dir = root / "index"

    questions_path = index_dir / "questions.jsonl"
    chunks_path = index_dir / "chunks.jsonl"
    emb_path = index_dir / "embeddings.npy"
    meta_path = index_dir / "meta.json"

    questions = load_questions(questions_path)
    chunks_by_id = load_chunks(chunks_path)
    emb_matrix = np.load(emb_path)
    meta = load_meta(meta_path)

    print("Centre Handbook Chatbot")
    # print(f"- Embedding model: {meta.get('embedding_model', EMBED_MODEL)}")
    print("Type /exit or /quit to stop.\n")

    while True:
        try:
            user_q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_q:
            continue
        if user_q.lower() in {"/exit", "/quit"}:
            break

        answer_query(user_q, emb_matrix, questions, chunks_by_id)


if __name__ == "__main__":
    main()


