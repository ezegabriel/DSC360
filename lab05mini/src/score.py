"""
Part 2 — Automated Scoring

This script compares your extracted test output (out/sections_test.csv)
to the instructor’s gold standard (tests/gold.csv) using **Exact Match (EM)**.

Instructions:

- Do NOT run this until you have finished and perfected Part 1.

- In Part 1, run extract.py on the training set (raw/training.txt)
  and manually inspect your output (out/sections_train.csv).
  Refine your prompt and schema until your extraction looks correct.

- When you are satisfied, uncomment the test-set lines at the bottom
  of extract.py and run it again to generate out/sections_test.csv.

- Only then should you run this scoring script.

Definition of Exact Match (EM):

- Field-level EM — the proportion of individual fields (columns)
  that exactly match the gold data, averaged over all rows.

- Row-level EM — the proportion of complete rows that match the
  gold data across *every* field.

Example:

- If 9 of 10 'program' values match → Field-level EM for 'program' = 0.9

- If 7 of 10 rows match in *all* columns → Row-level EM = 0.7
"""

import os
import pandas as pd


def main():
    # Check that the student has completed Part 1 and generated test output
    if not os.path.exists("out/sections_test.csv"):
        print("⚠️  No test output found.")
        print("Please finish Part 1 first, refine your extraction app,")
        print("then uncomment the test-set lines at the bottom of extract.py")
        print("to generate out/sections_test.csv before scoring.")
        return

    if not os.path.exists("tests/gold.csv"):
        print("⚠️  Missing gold standard file in tests/gold.csv.")
        return

    pred = pd.read_csv("out/sections_test.csv", sep=";")
    gold = pd.read_csv("tests/gold.csv", sep=";")

    if len(pred) != len(gold):
        print(f"⚠️  Output has {len(pred)} rows; gold has {len(gold)}")

    # Field-level EM
    matches = (pred.fillna("") == gold.fillna(""))
    field_em = matches.mean(axis=0)

    print("Field-level Exact Match (EM):")
    for field, value in field_em.items():
        print(f"  {field:10s}: {value:.2f}")

    # Row-level EM
    row_em = matches.all(axis=1).mean()
    print(f"\nRow-level EM: {row_em:.2f}")


if __name__ == "__main__":
    main()
