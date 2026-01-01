"""
score.py


Runs the same retrieval + answer logic as ask.py, but controlled and silent.
Computes chunk-level Hit@1 and saves all results to out/eval_results.csv.


This DOES NOT call answer_query() because that function only prints.
Instead, it reuses retrieve_chunks_for_query(), build_context_prompt(),
and ollama_chat() exactly the way your CLI does.
"""


import csv
import json
from pathlib import Path


import numpy as np


# Import your actual functions from ask.py
try:
    from .ask import (
        retrieve_chunks_for_query,
        build_context_prompt,
        ollama_chat,
        MIN_SIMILARITY,
        GENERATOR_MODEL,
    )
except ImportError:
    from ask import (
        retrieve_chunks_for_query,
        build_context_prompt,
        ollama_chat,
        MIN_SIMILARITY,
        GENERATOR_MODEL,
    )




def load_gold(path):
    rows = []
    with open(path, newline="", encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows




def load_index(index_dir):
    emb = np.load(index_dir / "embeddings.npy")


    questions = []
    with open(index_dir / "questions.jsonl", encoding="latin-1") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))


    chunks = {}
    with open(index_dir / "chunks.jsonl", encoding="latin-1") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                chunks[rec["chunk_id"]] = rec


    return emb, questions, chunks




def safe_generate_answer(user_q, max_sim, context_chunks):
    """
    Reproduce exactly what answer_query() does,
    but return the text instead of printing anything.
    """
    if max_sim < MIN_SIMILARITY or not context_chunks:
        return "Insufficient context in the handbook sections I know."


    prompt = build_context_prompt(user_q, context_chunks)
    answer = ollama_chat(GENERATOR_MODEL, prompt).strip()
    return answer




def main():
    root = Path(__file__).resolve().parents[1]  # project root
    tests_dir = root / "tests"
    index_dir = root / "index"
    out_dir = root / "out"
    out_dir.mkdir(exist_ok=True)


    gold_path = tests_dir / "gold.csv"
    results_path = out_dir / "results.csv"


    gold = load_gold(gold_path)
    emb_matrix, questions, chunks_by_id = load_index(index_dir)


    total_normals = 0
    total_hits = 0


    out_rows = []


    for row in gold:
        qid = row["qid"]
        qtext = row["question"]
        qtype = row["type"]
        gold_chunk = row["gold_chunk"].strip() if row["gold_chunk"] else ""
        notes = row["notes"]


        # --- RETRIEVAL (the real pipeline) ---
        max_sim, context_chunks = retrieve_chunks_for_query(
            qtext, emb_matrix, questions, chunks_by_id
        )


        # top-1 predicted chunk
        if context_chunks:
            pred_chunk = context_chunks[0]["chunk_id"]
        else:
            pred_chunk = ""


        # compute hit1 only for answerable normals
        if qtype == "normal" and gold_chunk:
            total_normals += 1
            hit1 = 1 if pred_chunk == gold_chunk else 0
            total_hits += hit1
        else:
            hit1 = ""


        # --- GET ANSWER TEXT (no printing) ---
        answer_text = safe_generate_answer(qtext, max_sim, context_chunks)


        out_rows.append(
            {
                "qid": qid,
                "question": qtext,
                "type": qtype,
                "gold_chunk": gold_chunk,
                "pred_chunk": pred_chunk,
                "hit1": hit1,
                "answer_text": answer_text,
                "notes": notes,
            }
        )


    # write results
    fieldnames = [
        "qid",
        "question",
        "type",
        "gold_chunk",
        "pred_chunk",
        "hit1",
        "answer_text",
        "notes",
    ]


    with open(results_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)


    # summary
    if total_normals > 0:
        hit_rate = total_hits / total_normals
        print(f"Normal questions with gold chunk: {total_normals}")
        # print(f"Hits at 1: {total_hits}")
        print(f"Hit@1 = {hit_rate:.3f}")
    else:
        print("No answerable normal questions found.")




if __name__ == "__main__":
    main()





