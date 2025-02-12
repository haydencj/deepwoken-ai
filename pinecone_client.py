from pinecone import Pinecone
from pinecone import ServerlessSpec
from config import PINECONE_API_KEY, INDEX_NAME, VECTOR_DIM

def init_pc():
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # check if index already exists (it shouldn't if this is first time)
    if INDEX_NAME not in pc.list_indexes().names():
        # if does not exist, create index
        pc.create_index(
            INDEX_NAME,
            dimension=VECTOR_DIM,  # dimensionality of text-embedding-ada-002
            metric='cosine',
            spec=ServerlessSpec(
            cloud="aws", 
            region="us-east-1"
            ) 
        )

    # connect to index
    index = pc.Index(name=INDEX_NAME)
    # view index stats
    stats = index.describe_index_stats()
    #print(stats)

    return index

index = init_pc()

if __name__ == "__main__":
    init_pc()
