import sys

def main():
    from query_generate import query_and_generate_answer  
    print("Welcome to the DeepwokenBot. Type 'exit' to quit.")
    
    while True:
        query_text = input("You: ").strip()
        if query_text.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        try:
            print("\nDeepwokenBot is typing...\n")
            # Call the streaming function, which prints tokens as they come
            query_and_generate_answer(query_text)
            print()  # Add a newline after the answer is complete.
        except Exception as e:
            print("Error generating answer:", e)

if __name__ == "__main__":
    main()
