This zip contains two Python stubs and one dataset for the mini-lab:

- mini_sentiment.py
  Classify movie reviews as “positive” or “negative.” You will complete
  query_ollama() (copy from Part 1) and classify_review(). The rest is provided.

- imdb20.csv
  A small subset of 20 reviews used in mini_sentiment.py.

- make_imdb_subset.py
  Script that creates imdb20.csv
  To run this script you will need to download the dataset from Kaggle.
  
------------------------------------------------------------
Python environment setup
------------------------------------------------------------

If you already have a course environment (for example at ~/.venvs/dsc):

  source ~/.venvs/dsc/bin/activate

Then install the required packages (only once per environment):

  pip install pandas scikit-learn ollama

If you do NOT have an environment yet, you can create one like this:

  python3 -m venv ~/.venvs/dsc
  source ~/.venvs/dsc/bin/activate
  pip install pandas scikit-learn ollama

If that doesn’t work or you prefer not to use environments, you can install
packages into your global Python with:

  pip install --user pandas scikit-learn ollama

------------------------------------------------------------
Where did imdb20.csv come from?
------------------------------------------------------------

The full dataset is the IMDB Movie Reviews Dataset (50,000 reviews with labels).
It is publicly available from Stanford and also mirrored on Kaggle:

- Stanford site: https://ai.stanford.edu/~amaas/data/sentiment/
- Kaggle mirror: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

We used a helper script (make_imdb_subset.py) to take a random sample of 20
reviews and trim them to a manageable length for this lab.

If you want to try downloading the Kaggle version directly (requires a Kaggle
account and API key), you can use the kaggle CLI. Alternatively, the raw file
can sometimes be fetched with curl:

  curl -L -o imdb-dataset-of-50k-movie-reviews.zip \
    https://www.kaggle.com/api/v1/datasets/download/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Unzip the file to find IMDB Dataset.csv, which contains the full 50,000 reviews.

------------------------------------------------------------
Requirements
------------------------------------------------------------
- Python 3.9+
- pandas
- scikit-learn
- ollama
