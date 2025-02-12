import json
from embeddings import get_embedding
from pinecone_client import index

def test_pinecone_query(query_text: str, top_k: int = 5):
    """
    Test a Pinecone query by generating an embedding for the provided query,
    querying the Pinecone index, and printing the results.
    """    
    # Generate an embedding for the query.
    query_embedding = get_embedding(query_text)
    
    # Query the Pinecone index for the top_k matching vectors with metadata.
    query_response = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    
    # Print the complete query response for debugging.
    print("Pinecone Query Response:")
    print(query_response)
    
    # Iterate over matches and print each match's details.
    matches = query_response.get("matches", [])
    if not matches:
        print("No matches found for the query.")
    else:
        for match in matches:
            match_id = match.get("id")
            score = match.get("score")
            metadata = match.get("metadata", {})
            print(f"\nID: {match_id}")
            print(f"Score: {score}")
            print("Metadata:")
            print(json.dumps(metadata, indent=2))
            print("-" * 40)

if __name__ == "__main__":
    # Replace with any query you want to test.
    sample_query = "how do i get the ghost talent?"
    test_pinecone_query(sample_query)
