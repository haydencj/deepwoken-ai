import tiktoken

def split_text_into_token_chunks(text: str, max_tokens: int = 7000, overlap: int = 100, model: str = "text-embedding-ada-002") -> list[str]:
    """
    Splits text into chunks based on token counts using tiktoken.
    
    Args:
        text (str): The input text to split.
        max_tokens (int): The maximum number of tokens per chunk.
        overlap (int): The number of tokens to overlap between consecutive chunks.
        model (str): The model for which to get the tokenizer.
        
    Returns:
        list[str]: A list of text chunks, each within the token limit.
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        # Advance start by max_tokens - overlap to provide an overlap between chunks.
        start += max_tokens - overlap
    return chunks
