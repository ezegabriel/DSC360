import ollama

def main():
    model = 'gemma3:1b'
    language = input("Enter a programming language (Python): ")
    prompt = f"Write hello world in {language}. Code only, no explanation."
    try:
        response = ollama.chat(model=model,
                               messages=[{"role": "user",
                                          "content": prompt}])
        print(response.message.content)
    except ollama.ResponseError as e:
        print("Error: ", e.error)

main()
