# === STUDENT INSTRUCTIONS =====================================================
# clickbait.py — Evaluate an LLM-powered clickbait detector.
#
# You are given:
#   `data`: instances with headlines and correct labels (clickbait or news)
#
# Implement:
#   1. classify_headline(text) -> str      # either "clickbait" or "news"
#   2. evaluate_accuracy(sample) -> float  # in [0, 1]
# ==============================================================================

import ollama
import random

MODEL = "gemma3:4b"

# (headline, true class label)
data = [
    ("10 secrets you won’t believe about cats!", "clickbait"),
    ("Mayor announces new recycling plan for city", "news"),
    ("This one trick could save you thousands on groceries!", "clickbait"),
    ("Study reveals benefits of daily walking", "news"),
    ("California police pull over a self-driving Waymo for an illegal U-turn, but they can’t ticket", "news"),
    ("Local team wins state championship", "news"),
    ("Doctors hate him! See the simple cure he discovered", "clickbait"),
    ("New legislation aims to cut carbon emissions by 2030", "news"),
    ("She opened the door—and you won’t believe what she found!", "clickbait"),
    ("Museum unveils exhibit of ancient artifacts", "news"),
    ("You Won’t Believe What This Small Town Did to Save Its Schools", "clickbait"),
    ("Has International Travel to the U.S. Really Collapsed?", "news"),
    ("8 hidden tricks that help flight attendants live better", "clickbait"),
    ("Doctors reveal the #1 habit that changes your health", "clickbait"),
    ("City council approves new zoning laws", "news"),
    ("They Tried to Shut Her Down — Here’s What Happened", "clickbait"),
    ("Doctors reveal the #1 habit that changes your health", "clickbait"),
    ("The secret to living longer might be in your gut", "news"),
    ("City council approves new zoning laws", "news"),
    ("They Tried to Shut Her Down — Here’s What Happened", "clickbait"),
]

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
    return 'Error'  # placeholder  

# ===== STUDENT TODO #1 ========================================================
def classify_headline(text: str) -> str:
    """
    Return exactly 'clickbait' or 'news' for the given headline.
    Requirements:
      - Write your prompt.
      - Call Ollama with MODEL and your prompt.
      - Normalize the model's reply so this function ALWAYS returns exactly
        'clickbait' or 'news' (handle case, punctuation, extra words, etc.).
    """
    
    # YOUR CODE HERE
    prompt = (
        'Classify the following headlines as either "clickbait" or "news".\n'
        'Only reply with your one word label prediction'
        f'Headline: {text}'
    )
    response = query_ollama(prompt, MODEL)

    has_clickbait = "clickbait" in response
    has_news = "news" in response

    if has_clickbait and not has_news: return "clickbait"
    if not has_clickbait and has_news: return "news"
    if has_clickbait and has_news: return ""

    if 'clickbait' in response:
        return 'clickbait'
    if 'news' in response:
        return 'news'

    return "news"  # placeholder

# ===== STUDENT TODO #2 ========================================================
def evaluate_accuracy(sample) -> float:
    """
    Display predictions on a sample and return accuracy score in [0, 1].
    Required print format for each item:
        Headline: <headline>
        Model: <predicted_class>, Actual: <actual_class>
    """
    # YOUR CODE HERE
    total_predictions_clickbait = 0
    total_predictions_news = 0

    total_actual_clickbait = 0
    total_actual_news = 0

    correct = 0

    for i, item in enumerate(sample):

        if item[1] == 'clickbait':
            total_actual_clickbait += 1
        elif item[1] == "news":
            total_actual_news += 1

        prediction = classify_headline(item[0])

        if prediction == "clickbait":
            total_predictions_clickbait += 1
        elif prediction == "news":
            total_predictions_news += 1
        
        if prediction == item[1]: correct += 1

        print(f"Headline: {item[0]}\nModel: {prediction}, Actual: {item[1]}")

    accuracy_score = correct/len(sample)

    return accuracy_score



# ==============================================================================

def main():
    # Select random sample of the data set (headlines and class labels).
    sample_size = 10
    sample = random.sample(data, k=sample_size)

    # Evaluate model accuracy on sample
    accuracy_score = evaluate_accuracy(sample)
    print(f"Accuracy score: {accuracy_score:.2f}")

if __name__ == "__main__":
    main()
