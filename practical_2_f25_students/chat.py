# === STUDENT INSTRUCTIONS =======================================================
# chat.py: Trimming Chat History for a Simple LLM Chatbot
#
# Your task: implement trim_history(messages, max_turns) so that our chatbot
# keeps the system prompt and only the last few turns of conversation before
# calling the local Ollama model.
#
# We provide:
#   - A working REPL that reads user input and calls the model
#   - The call_ollama(...) helper that talks to a local gemma model
#   - The main() function that uses trim_history(...) on each turn
#
# You should:
#   - Only edit trim_history(...)
#   - Assume messages[0] is always the system message
#   - Return a NEW list (do not rely on modifying the original in place)
# ===============================================================================

from ollama import chat

MODEL_NAME = "gemma3:1b"   # adjust if needed
MAX_TURNS = 4              # max number of non-system messages to keep


# ===== STUDENT TODO ============================================================
def trim_history(messages, max_turns):
    """
    Keep the system prompt (messages[0]) and only the last max_turns
    non-system messages from messages[1:].
    """
    # messages: list of dicts like {"role": "system" | "user" | "assistant",
    #                              "content": str}
    # max_turns: maximum number of non-system messages to keep
    #
    # Return a new list with:
    #   - the original system message (the first message in messages)
    #   - only the last max_turns messages from the other messages
    #
    # Example: if messages has 1 system + 6 other messages and max_turns == 4,
    # your result should have 1 system + the last 4 of those 6 messages.
    #
    # Your code here

    system_message = messages[0]
    non_system_messages = messages[1:]

    if len(non_system_messages) <= max_turns:
        return messages.copy()

    trimmed_messages = non_system_messages[-max_turns:]
    result = [system_message] + trimmed_messages

    return result  # placeholder so the chatbot still works
# ============================================================================

def call_ollama(model, messages):
    """
    Call a local Ollama model and return a single assistant message dict.

    messages: full chat history (already trimmed) as a list of {role, content} dicts.
    """
    response = chat(model=model, messages=messages)
    content = response.message.content

    return {
        "role": "assistant",
        "content": content,
    }


def main():
    system_prompt = (
        "You are a simple chatbot assistant. "
        "You have been tasked to help students with course registration. "
        "Keep replies to 1–2 sentences."
    )

    # messages[0] is always the system message
    messages = [
        {"role": "system", "content": system_prompt},
    ]

    print("Bot: Hello! I'm a friendly course registration assistant. How can I help you today? ")

    running = True
    while running:
        user_text = input("You: ").strip()
        lower_text = user_text.lower()

        if lower_text in {"/quit", "/q"}:
            # Case 1: user asked to quit
            running = False

        elif user_text != "":
            # Case 2: normal message → update history and call the model
            messages.append({"role": "user", "content": user_text})

            # Trim BEFORE calling the model
            messages = trim_history(messages, MAX_TURNS)

            # Call the model and get one assistant reply
            assistant_msg = call_ollama(MODEL_NAME, messages)

            print("Bot:", assistant_msg["content"])
            messages.append(assistant_msg)
            
        # Do nothing if student enters nothing ("") since that will break the bot


if __name__ == "__main__":
    main()
