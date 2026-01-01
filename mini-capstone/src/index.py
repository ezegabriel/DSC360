"""
index.py

Doc2Query + embeddings for the Student Handbook project.

Pipeline (matches the Doc2Query in Practice notes):
1. Load chunks from ../index/chunks.jsonl.
2. For each chunk, ask an LLM to generate a small set of student-style questions.
3. Save all questions to ../index/questions.jsonl.
4. Embed each question with a local embedding model via Ollama.
5. Save embeddings to ../index/embeddings.npy and basic index/meta.json.
"""

import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import requests
import re

# ---- configuration ---------------------------------------------------

OLLAMA_URL = "http://localhost:11434"

DOC2QUERY_MODEL = "qwen2.5:1.5b"          # LLM used for synthetic question & answer generation
EMBED_MODEL = "mxbai-embed-large:latest"  # embedding model
N_QUESTIONS_PER_CHUNK = 3                 # 3–5 is fine
EMBED_NORMALIZED = False                  # set True if you want L2 normalization

# minimal stopword list for overlap check
STOPWORDS = {
    "the", "and", "or", "a", "an", "of", "to", "in", "for", "on",
    "at", "by", "is", "are", "be", "this", "that", "it", "with",
    "as", "from", "about",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "can", "could", "would", "should", "may"
}




# ---- basic I/O -------------------------------------------------------

def load_chunks(chunks_path: Path) -> List[Dict]:
    chunks = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def save_questions(questions: List[Dict], questions_path: Path) -> None:
    with questions_path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")
    print(f"Wrote {len(questions)} questions to {questions_path}")


def save_embeddings(embeddings: np.ndarray, emb_path: Path, meta_path: Path) -> None:
    np.save(emb_path, embeddings)
    print(f"Wrote embeddings with shape {embeddings.shape} to {emb_path}")

    meta = {
        "embedding_model": EMBED_MODEL,
        "dimension": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "normalized": EMBED_NORMALIZED,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote embedding metadata to {meta_path}")


# ---- Ollama helpers --------------------------------------------------

def ollama_chat(model: str, prompt: str) -> str:
    """Call Ollama /api/chat and return the response content as a string without streaming."""
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
    # Some versions stream; here we assume final message in "message"
    return data["message"]["content"]


def ollama_embed(model: str, text: str) -> List[float]:
    """Call Ollama /api/embeddings and return the embedding vector."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["embedding"]


# ---- doc2query + embedding logic ------------------------------------

def generate_questions_for_chunk(section_title: str,                                 
                                 chunk_text: str) -> List[str]:
    """
    Use an LLM to generate N_QUESTIONS_PER_CHUNK student-style questions.

    Matches the style in the Doc2Query notes:
    - natural student language
    - answerable from this passage alone
    - no invented numbers or facts
    """
    prompt = f"""
You are helping build a search index for the Centre College Student Handbook.

You will be given:
- the title of a handbook section
- an excerpt from that section

Write {N_QUESTIONS_PER_CHUNK} different questions that a student, parent, incoming student, or staff member might ask when trying to understand how these rules apply in real situations.

Guidelines:
- Sound natural and conversational, not like a textbook.
- Do NOT prefix question statements with roles, categories, labels, or markdown of any kind.
- You may ask about fines or sanctions only when the excerpt clearly mentions them, but do NOT invent additional penalties not stated.
- Do NOT describe fictional accidents, mishaps, or made-up stories involving friends "sneaking" into a violation.
- Do NOT ask about personal struggles, emotional support, family issues, relationship problems, or everyday mishaps; focus only on policy-related questions that this excerpt can answer.
- Do NOT ask about household chores or routine personal tasks; only ask about enforceable policies stated in the excerpt.
- Do NOT refer to any meta-level structure; questions must be written as if the user only knows general campus topics, not the handbook text you were shown.
- Do NOT introduce sensitive topics unless they are explicitly stated in the excerpt.
- Do NOT copy long phrases from the excerpt; paraphrase in your own words.
- Each question must focus on a different situation or angle.
- Keep each question to one sentence.


Section title: {section_title}

Handbook excerpt:
{chunk_text}
"""

    raw = ollama_chat(DOC2QUERY_MODEL, prompt)

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    questions: List[str] = []

    for line in lines:
        # Strip any leading numbering/bullets the model might add anyway
        while line and line[0] in "-*0123456789":
            line = line.lstrip("-*0123456789. ").strip()
        if line:
            questions.append(line)

    if len(questions) > N_QUESTIONS_PER_CHUNK:
        questions = questions[:N_QUESTIONS_PER_CHUNK]

    return questions

def _tokenize(text: str) -> List[str]:
    """Lowercase word tokens, strip punctuation, drop stopwords."""
    # simple split + strip; good enough here
    raw_tokens = re.findall(r"[A-Za-z]+", text.lower())
    return [t for t in raw_tokens if t not in STOPWORDS]


def _has_basic_overlap(question: str, chunk_text: str, min_overlap: int = 2) -> bool:
    """
    Require at least min_overlap content words in common between
    the question and the chunk.
    """
    q_tokens = set(_tokenize(question))
    c_tokens = set(_tokenize(chunk_text))
    overlap = q_tokens & c_tokens
    return len(overlap) >= min_overlap


def _question_supported_by_chunk(question: str, chunk_text: str) -> bool:
    """
    Heuristic filter from Doc2Query notes:
    - numbers in the question must appear in the chunk
    - require basic word overlap

    If either check fails, return False.
    """
    # number consistency: all digit substrings in question should appear in chunk
    nums_q = re.findall(r"\d+", question)
    if nums_q:
        for n in nums_q:
            if n not in chunk_text:
                return False

    # basic content word overlap
    if not _has_basic_overlap(question, chunk_text, min_overlap=2):
        return False

    return True



def embed_text(text: str) -> List[float]:
    """Embed a single question using the chosen embedding model."""
    return ollama_embed(EMBED_MODEL, text)


def build_questions(chunks: List[Dict]) -> List[Dict]:
    """
    For each chunk, generate question records like:

    {
      "question_id": "chunk_12_q0",
      "chunk_id": "chunk_12",
      "section_title": "...",
      "question": "How many unexcused absences..."
    }
    """
    all_questions: List[Dict] = []

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        section_title = chunk.get("section_title", "")
        text = chunk["text"]

        print(f"Generating questions for {chunk_id} ({section_title})...")
        raw_questions = generate_questions_for_chunk(section_title, text)

        # basic cleaning
        clean = []
        for q in raw_questions:
            q = q.strip()
            if not q:
                continue
            # filter: must be supported by the chunk (overlap + numbers)
            if not _question_supported_by_chunk(q, text):
                print(f"  Dropping hallucinated/unsupported question: {q!r}")
                continue
            clean.append(q)

        if not clean:
            print(f"  Warning: no supported questions for {chunk_id}, skipping this chunk.")
            continue

        clean = clean[:N_QUESTIONS_PER_CHUNK]

        for i, q in enumerate(clean):
            q_id = f"{chunk_id}_q{i}"
            all_questions.append(
                {
                    "question_id": q_id,
                    "chunk_id": chunk_id,
                    "section_title": section_title,
                    "question": q,
                }
            )

    return all_questions



def build_embeddings(questions: List[Dict]) -> np.ndarray:
    """
    Embed each question.question into a vector and return an (N, d) matrix.
    """
    vectors: List[List[float]] = []

    for i, q in enumerate(questions):
        text = q["question"]
        print(f"Embedding question {i+1}/{len(questions)}")
        vec = embed_text(text)
        vectors.append(vec)

    arr = np.array(vectors, dtype="float32")

    if EMBED_NORMALIZED:
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
        arr = arr / norms

    return arr


def main():
    root = Path(__file__).resolve().parent.parent
    index_dir = root / "index"
    chunks_path = index_dir / "chunks.jsonl"
    questions_path = index_dir / "questions.jsonl"
    emb_path = index_dir / "embeddings.npy"
    meta_path = index_dir / "meta.json"

    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    # Step 1: doc2query question generation
    questions = build_questions(chunks)
    save_questions(questions, questions_path)

    # Step 2: embeddings
    emb_matrix = build_embeddings(questions)
    save_embeddings(emb_matrix, emb_path, meta_path)


if __name__ == "__main__":
    main()



