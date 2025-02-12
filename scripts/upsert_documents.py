import json
import re
import unicodedata
from embeddings import get_embedding
from pinecone_client import init_pc, index  # Ensure index is imported from pinecone_client.py
from chunker import split_text_into_token_chunks
import sys

def sanitize_id(s: str) -> str:
    """
    Normalize and sanitize the string to create an ASCII-only ID.
    """
    normalized = unicodedata.normalize("NFKD", s)
    ascii_str = normalized.encode("ascii", "ignore").decode("ascii")
    sanitized = re.sub(r"\W+", "_", ascii_str)
    return sanitized

def upsert_documents(file_path: str, max_tokens: int = 7000, overlap: int = 100):
    """
    Load documents from a JSONL file, split them into token-based chunks, and upsert into Pinecone.
    Each JSON line should contain keys "name", "content", and "metadata".
    """    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            original_id = doc.get("name")
            content = doc.get("content")
            if not original_id or not content:
                print("Skipping document due to missing 'name' or 'content'")
                continue
            
            # Sanitize the document ID.
            doc_id = sanitize_id(original_id)
            
            # Split the document content into token-based chunks.
            chunks = split_text_into_token_chunks(content, max_tokens=max_tokens, overlap=overlap)
            for i, chunk in enumerate(chunks):
                # Create a compound ID for each chunk.
                chunk_id = f"{doc_id}_chunk_{i}"
                # Build metadata to include the chunk text and reference to the original document.
                metadata = {
                    "content": chunk,  # Save the actual chunk text.
                    "original_doc": doc_id,
                    "chunk_index": i,
                    "source": doc.get("metadata", {}).get("source", "web"),
                    "url": doc.get("metadata", {}).get("url", "")
                }
                # Generate embedding for the chunk.
                embedding = get_embedding(chunk)
                # Upsert the chunk into Pinecone.
                index.upsert([(chunk_id, embedding, metadata)])
                print(f"Upserted chunk {i} of document: {doc_id}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python upsert_documents.py <docs_file>")
        sys.exit(1)
    upsert_documents(sys.argv[1])
