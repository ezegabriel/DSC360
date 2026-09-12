# === STUDENT INSTRUCTIONS ================================================
# moby.py — Retrieval-Augmented Generation (RAG) for the novel Moby Dick
#
# Implement:
#   generate(question, passages) -> str
#
# You will receive up to K short passages (strings) already retrieved
# from your instructor's Lab 03 index. Build a concise prompt that
# answers using ONLY those passages. If the passages don’t support an
# answer, say so briefly (e.g., “I don’t have enough information from
# these passages.”). Keep answers to at most a few sentences.
# =========================================================================

from __future__ import annotations
import sys
from embed import load_index, retrieve_texts
import ollama

# Config
TOP_K    = 4        # max passages to use
MIN_SIM  = 0.25     # cosine similarity threshold (adjustable)
MODEL = "gemma3:4b"

# ===== STUDENT TODO ======================================================
def generate(question: str, passages: list[str]) -> str:
    """
    Generate a brief answer using only the provided passages.

    Args:
        question (str): The user's query.
        passages (list[str]): Chunked text passages from *Moby-Dick* to use as context.

    Returns:
        str: The model’s response text (1–3 sentences).
    """

    prompt = f"""
        Answer the following question using only information from the provided passages. 
        If the passages does not support an answer, response with: "I don't have enough information from these passages".
        Question: {question} 
        Passages: {passages}
    """

    try:
        r = ollama.chat(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            options={"temperature": 0}
        )
        return r["message"]["content"].strip()
    except Exception as e:
        print("Error:", e)
        return "Error"

# =========================================================================

def rag(question: str, E, chunks, meta) -> str:
    """Retrieve up to K relevant passage strings, then call generate()."""
    passages = retrieve_texts(E, chunks, meta, question, k=TOP_K, min_sim=MIN_SIM)
    return generate(question, passages)

def main():
    E, chunks, meta = load_index()
    print(f"[moby] Index ready ({E.shape[0]} chunks). Ask about Moby Dick.")
    print("Example: Who is Captain Ahab?   (Type '/quit' to exit.)")

    while True:
        try:
            q = input("\nquestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not q:
            continue
        if q.lower() in {"/q", "/quit", "/exit"}:
            print("Bye.")
            break

        answer = rag(q, E, chunks, meta)
        print("\n=== Answer ===")
        print(answer)

if __name__ == "__main__":
    main()
