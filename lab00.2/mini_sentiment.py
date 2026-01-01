#!/usr/bin/env python3
#
# DSC 360 Mini-Lab 00.3: Movie Reviews
#
# Goal: Use an LLM to classify movie reviews as "positive" or "negative".
# You will:
#   1) implement query_ollama() if you have not already done so
#   2) implement classify_review() (including the prompt)
# The rest is provided for you.

# How to run:
#   python3 mini_sentiment.py            # uses imdb20.csv
#   python3 mini_sentiment.py custom.csv # (optional) another CSV
#
# Input CSV columns:
#   review,sentiment
# Output:
#   results.csv with your predictions

import sys
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import ollama

# ************************************************************************
# Your task in this lab is to modify the code in this section as needed:

MODEL = "gemma3:1b"
INPUT_CSV = "imdb20.csv"
OUT_CSV = "results.csv"

def query_ollama(prompt: str, model: str = MODEL) -> str:
    """Call Ollama chat with the user prompt and return the reply text.

    Note: You should have written this for the previous mini-lab.
    If it meets the specifications, you should be able to paste it in."""

    # YOUR CODE HERE
    try:
        response = ollama.chat(model = MODEL,
                               messages = [{'role':'user',
                                            'content':prompt}])
        return response.message.content
    except ollama.ResponseError as e:
        print('Error: ', e.error)
    """Call Ollama with the user prompt and return the reply text."""
    return ''  # placeholder  

def classify_review(review_text: str, model: str = MODEL) -> str:
    """TODO: Build the prompt and call query_ollama().
    Return exactly 'positive' or 'negative' if possible; otherwise 'unknown'.

    Hints for your prompt:
      - Force ONE WORD exactly: "positive" or "negative".
      - Include the review text via {review}.
      - Keep it short and unambiguous.

    Minimal normalization idea (optional but helpful):
      - If the reply is exactly 'positive' or 'negative', return it.
      - If the reply rambles (e.g., 'positive because ...'), pick the word that
        appears WITHOUT the other one. Otherwise return 'unknown'.
      - Hint: Use the keyword `in` or `not in` along with Boolean operators.
    """

    # YOUR CODE HERE
    # prompt = f"Write your prompt here including the review: {review_text}."
    # call query_ollama with your prompt and store the result using a variable
    # depending on the result, return "positive" or "negative" as a string
    
    prompt = "Classify this movie: {review} as either positive or negative. Give adjective answer."

    response = query_ollama(prompt, model).strip().lower()

    return response  # placeholder


# ************************************************************************
# This is scaffolded code provided to you.
# Do NOT modify it without checking with your instructor.

def run_experiment(input_csv: str, model: str = MODEL) -> pd.DataFrame:
    """Read the CSV, classify each review, and compute metrics."""
    df = pd.read_csv(input_csv)  # expects columns: review, sentiment

    # Collect predictions
    predictions = []
    for text in df["review"]:
        label = classify_review(str(text), model)  # students implement this
        predictions.append(label)
    df["predicted"] = predictions

    # Scoring helper: force non-binary predictions to be wrong on purpose
    def coerce_for_scoring(true_label: str, pred_label: str) -> str:
        """If pred is not 'positive'/'negative', flip it vs the truth so it's wrong."""
        if pred_label in ("positive", "negative"):
            return pred_label
        return "negative" if true_label == "positive" else "positive"

    preds_for_scoring = [coerce_for_scoring(t, p)
                         for t, p in zip(df["sentiment"], df["predicted"])]

    # Compare predictions with ground truth.
    acc = accuracy_score(df["sentiment"], preds_for_scoring)
    f1  = f1_score(df["sentiment"], preds_for_scoring, pos_label="positive")

    # Display metrics. Return object to allow further (human) inspection.
    print(f"Accuracy: {acc:.2f}")
    print(f"F1 (positive): {f1:.2f}")

    return df

def main() -> None:
    """Decide which CSV to use, run the experiment, and save results."""
    if len(sys.argv) > 1:
        in_csv = sys.argv[1]
    else:
        in_csv = INPUT_CSV

    results = run_experiment(in_csv, MODEL)
    results.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")
# ************************************************************************

if __name__ == "__main__":
    main()
