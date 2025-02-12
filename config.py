import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env


# Configuration for API keys and index details.
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")  # e.g., "us-west1-gcp"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

INDEX_NAME = "deepwoken-docs"
VECTOR_DIM = 1536  # typical dimension for text-embedding-ada-002
DOCS_FILE = "docs.jsonl"  # path to your JSONL documents file

client = OpenAI(api_key=OPENAI_API_KEY)
