from string import Template
from pathlib import Path
import time
import pandas as pd
import ollama
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    classification_report
)

MODELS = ["gemma3:270m", "gemma3:1b", "gemma3:4b"]
INPUT_CSV = Path("data.csv")


def query_ollama(prompt: str, model: str) -> str:
    """Call Ollama chat with the user prompt and return the reply text."""
    try:
        response =  ollama.chat(model=model, messages=[
            {
                'role':'user',
                'content': prompt,
            },
        ])
        return response.message.content or ''
    except ollama.ResponseError as e:
        return f'Error: {e.error}'


def classify_review(prompt: str, model: str) -> str:

    result = query_ollama(prompt=prompt, model=model).strip().lower()

    # print(f"DEBUG:\nPrompt:{prompt}\nresult='{result}'")  # you can comment this out later

    has_positive = "positive" in result
    has_negative = "negative" in result
    has_neutral  = "neutral"  in result

    if has_positive and not has_negative and not has_neutral: return "positive"
    if has_negative and not has_positive and not has_neutral: return "negative"
    if has_neutral and not has_positive and not has_negative: return "neutral"

    return "unknown"


# Scoring helper: force non-binary predictions to be wrong on purpose
def coerce_for_scoring(true_label: str, pred_label: str) -> str:
    """If pred is not 'positive'/'negative', flip it vs the truth so it's wrong."""

    if pred_label in ("positive", "negative", "neutral"):
        return pred_label
    
    return "neutral"


def run_experiment(OUTPUT_FILE_PATH, df: pd.DataFrame, model: str, prompt_template: Template, examples: str = '') -> pd.DataFrame:

    has_examples = False if examples == '' else True
    start_time = time.time()

    predictions = []
    for text in df["sentence"]:
        print(f"Progress: {len(predictions)}/{len(df)}", end='\r')
        prompt = prompt_template.substitute(sentence=text, example=examples)
        label = classify_review(prompt, model)
        predictions.append(label)
    df["predicted"] = predictions

    preds_for_scoring = [coerce_for_scoring(t, p) for t, p in zip(df["sentiment"], df["predicted"])]
    # preds_for_scoring = [coerce_for_scoring(t, p) for t, p in zip(df["sentiment"], predictions)]

    classification_report_results = classification_report(df["sentiment"], preds_for_scoring)
    print("Classification Report:\n", classification_report_results)

    end_time = time.time()
    elapsed_time = end_time - start_time

    with open(OUTPUT_FILE_PATH, 'a') as f:
        f.write(f"Model: {model}\n")
        f.write(f"Examples included: {has_examples}\n")
        f.write(f"Elapsed time: {elapsed_time:.2f} seconds\n")
        f.write(classification_report_results)
        f.write("\n\n")

    return df


def create_splits(subsample_proportion: float = 1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(INPUT_CSV)

    # Split the data into train, dev, and test sets (60% train, 20% dev, 20% test)
    train_df, temp_def = train_test_split(df, test_size=0.4, random_state=42, stratify=df['sentiment'])
    dev_df, test_df = train_test_split(temp_def, test_size=0.5, random_state=42)

    # portion of training sample to use
    train_df = train_df.sample(frac=subsample_proportion, random_state=42)

    return train_df, dev_df, test_df


def main() -> None:
    prompt_template = Template('You are a financial sentiment analyzer. Please classify the sentence as one of the following: positive, negative, or neutral. Only respond with one answer. $example Sentence: $sentence')

    examples = """
        Here are some examples with sentiment: ["$ESI on lows, down $1.50 to $2.50 BK a real possibility", negative],
        ["For the last quarter of 2010 , Componenta 's net sales doubled to EUR131m from EUR76m for the same period a year earlier , while it moved to a zero pre-tax profit from a pre-tax loss of EUR7m .", positive],
        ["According to the Finnish-Russian Chamber of Commerce , all the major construction companies of Finland are operating in Russia .", neutral]. "
    """

    num_iterations = 1

    for iteration in range(num_iterations):

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        OUTPUT_FILE_PATH = Path(f"results_{str(timestamp)}.txt")

        train_df, dev_df, test_df = create_splits()

        current_df = test_df

        # Experiment 1: gemma3:270m without examples
        current_model = MODELS[0]
        run_experiment(OUTPUT_FILE_PATH, current_df.copy(), current_model, prompt_template)

        # Experiment 2: gemma3:270m with examples
        run_experiment(OUTPUT_FILE_PATH, current_df.copy(), current_model, prompt_template, examples)

        # Experiment 3: gemma3:1b without examples
        current_model = MODELS[1]
        run_experiment(OUTPUT_FILE_PATH, current_df.copy(), current_model, prompt_template)

        # Experiment 4: gemma3:1b with examples
        run_experiment(OUTPUT_FILE_PATH, current_df.copy(), current_model, prompt_template, examples)

        # Experiment 5: gemma3:4b without examples
        current_model = MODELS[2]
        run_experiment(OUTPUT_FILE_PATH, current_df.copy(), current_model, prompt_template)

        #Experiment 6: gemma3:4b with examples
        run_experiment(OUTPUT_FILE_PATH, current_df.copy(), current_model, prompt_template, examples)


if __name__ == "__main__":
    main()