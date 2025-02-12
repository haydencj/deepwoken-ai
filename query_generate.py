from embeddings import get_embedding
from pinecone_client import index
from config import client 
from prompt import SYSTEM_PROMPT
import sys

def query_and_generate_answer(query_text: str, top_k: int = 5) -> str:
    """
    Embed the user query, retrieve top-k similar documents from Pinecone,
    and generate an answer using OpenAI's Chat Completions API with the new SDK.
    """
    # Generate an embedding for the user query.
    query_embedding = get_embedding(query_text)
    
    # Query Pinecone for the top-k similar vectors, including metadata.
    query_response = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    
    # Extract document texts from the metadata (assumes "content" is stored there).
    retrieved_texts = []
    for match in query_response.get("matches", []):
        content = match.get("metadata", {}).get("content", "")
        if content:
            retrieved_texts.append(content)
    
    # Join the retrieved document texts as context.
    context = "\n\n".join(retrieved_texts)
    
    #print(f"Context: {context}")

    # Construct messages for the chat completion.
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuery: {query_text}"
        }
    ]
    
    # Create the chat completion using the new API.
    stream = client.chat.completions.create(
        model="gpt-4o-mini",  # Use "gpt-4o-mini" (or "gpt-4o" if desired)
        messages=messages,
        max_tokens=500,
        temperature=0.2,
        stream=True
    )
    
        # Iterate over the streamed response and print tokens as they arrive.
    for chunk in stream:
        # For each chunk, extract the new token if available.
        token = chunk.choices[0].delta.content
        if token is not None:
            print(token, end="")
            yield token
        sys.stdout.flush()
    print()  # for final newline