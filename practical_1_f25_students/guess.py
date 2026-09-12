# === STUDENT INSTRUCTIONS =====================================================
# guess.py — Guess-the-number game with conversation memory.
#
# Extend function play_game() so the Model remembers prior turns.
#   Hint: You did something similar in an early lab.
#
# Players:
#   - Host: Knows the secret number and replies with feedback
#   - Model (LLM): Has up to 5 attempts to guess the secret number
#
# Example interaction:
#
#   Host: I'm thinking of a number between 1 and 32.
#   Guess the number by replying with a single numeral.
#   You can have up to 5 guesses.
#   Model: 16
#   Host: Too low.
#   Model: 24
#   Host: Too high.
#   Model: 20
#   Host: Too low.
#   Model: 22
#   Host: Correct!
#   Success: True
# ==============================================================================

import ollama
import re
from random import randint

MODEL = "gemma3:4b"

def call_ollama(messages):
    """Send the message list to the model and return its reply text."""
    try:
        r = ollama.chat(
            model=MODEL,
            messages=messages,
            options={"temperature": 0}
        )
        return r["message"]["content"].strip()
    except Exception as e:
        print("Error:", e)
        return "Error"

def extract_integer(text):
    """Return the first integer in text, or 0 if none is found."""
    m = re.search(r"\d+", text)
    return int(m.group()) if m else 0

# ===== FUNCTION PLAY_GAME() for you to complete =====
def play_game(secret, max_guesses=5) -> bool:
    """Simulate the Guess-the-Number Game.

    The Host (user) is given a `secret` number.
    
    The Model (LLM) tries to guess the `secret` with the help of
    conversation history (`messages`) and `feedback` from the Host about
    whether each `guess` is too high or too low.
    
    Game ends when Host guesses the number or exceeds `max_guesses`.
    Returns True if Host correctly guessed number, or False otherwise.

    """
    instructions = (
        "I'm thinking of a number between 1 and 32.\n"
        "Guess the number by replying with a single numeral.\n"
        f"You can have up to {max_guesses} guesses."
    )
    messages = [{"role": "user", "content": instructions}]
    print("Host:", instructions)

    for i in range(max_guesses):
        reply = call_ollama(messages)  # Send conversation history to LLM
        print("Model:", reply)

        guess = extract_integer(reply)  # Convert response from LLM to int
        messages.append({
            "role": "assistant",
            "content": str(guess)
        })
        
        if guess == 0:
            feedback = "I didn’t see a number. Try again."
        elif guess == secret:  # Host guessed number: Game Over
            print("Host:", "Correct!")
            return True
        elif guess < secret:
            feedback = "Too low."
        else:
            feedback = "Too high."

        print("Host:", feedback)

        # ===== STUDENT TODO ==============================================
        # Append the Model's reply and Host's feedback to `messages`
        # so the next turn has full conversation context.
        # =================================================================

        # YOUR CODE HERE
        messages.append({
            "role": "user",
            "content": feedback
        })
    
    print(f"Host: Out of guesses. My number was {secret}.")
    return False  # Reached guess limit: Game Over

def main():
    # Simulate one round of game play
    # Feel free to adapt this, if necessary, to test your code
    secret = randint(1, 32)
    success = play_game(secret)
    print("Success:", success)

if __name__ == "__main__":
    main()
