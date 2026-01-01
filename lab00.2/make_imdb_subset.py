import pandas as pd
import re

SRC = "IMDB Dataset.csv"   # or IMDB-Dataset.csv
OUT = "imdb_subset_trimmed.csv"
N = 20                     # how many rows to sample
KEEP_WORDS_EACH_SIDE = 80  # ~160 words total ≈ ~250–300 tokens
ELLIPSIS = " […] "

def clean(text: str) -> str:
    # Replace IMDB's HTML breaks, collapse whitespace
    s = str(text).replace("<br />", " ")
    return re.sub(r"\s+", " ", s).strip()

def truncate_start_end(text: str, k: int) -> str:
    s = clean(text)
    words = s.split()
    if len(words) <= 2 * k:
        return s
    return " ".join(words[:k]) + ELLIPSIS + " ".join(words[-k:])

df = pd.read_csv(SRC)
# Note: Random state of 42 leads to content that may be problematic
subset = df.sample(n=N, random_state=41).copy()
subset["review"] = subset["review"].apply(lambda t: truncate_start_end(t, KEEP_WORDS_EACH_SIDE))

subset.to_csv(OUT, index=False)

# (Optional) sanity check: lengths look reasonable
print(subset["review"].str.len().describe())
