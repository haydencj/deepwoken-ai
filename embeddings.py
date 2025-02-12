from openai import OpenAI
from config import OPENAI_API_KEY, client
import tiktoken
 
def truncate_to_max_tokens(text: str, max_tokens: int = 8000) -> str:
    """
    Truncate the input text to a maximum number of tokens using tiktoken.
    """
    # Get an encoding for the embedding model. You can choose the encoding for the model
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = encoding.decode(tokens)
    return text

def get_embedding(text: str, model: str = "text-embedding-ada-002") -> list[float]:
    """
    Generate an embedding for the provided text using OpenAI's Embedding API.
    This function first truncates the text to ensure it does not exceed the model's context length.
    
    Args:
        text (str): The input text to embed.
        model (str): The embedding model to use. Defaults to "text-embedding-ada-002".
        
    Returns:
        list[float]: The embedding vector as a list of floats.
    """
    # Truncate text to a safe token limit
    safe_text = truncate_to_max_tokens(text, max_tokens=8000)
    
    response = client.embeddings.create(
        input=[safe_text],
        model=model
    )
    
    # Extract the embedding from the response.
    embedding = response.data[0].embedding
    return embedding

# Example usage:
if __name__ == "__main__":
    sample_text = "Deepwoken is a challenging and immersive game on Roblox."
    embedding = get_embedding(sample_text)
    print("Embedding vector:", embedding)