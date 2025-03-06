# src/chunking.py - Chunking & Relevance Filtering
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import requests
import os

# Load environment variables from .env file
load_dotenv()

# Load embedding model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# embedding_model = SentenceTransformer("mxbai-embed-large", use_auth_token=True)
API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5"
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY") 
HEADERS = {"Authorization": "Bearer {HUGGINGFACE_API_KEY}"}  # Get from huggingface.co/settings/tokens
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

def get_embedding(text):
    """Fetches embeddings from Hugging Face API for a given text."""
    response = requests.post(API_URL, headers=HEADERS, json={"inputs": text})
    if response.status_code != 200:
        raise ValueError(f"Error {response.status_code}: {response.text}")
    return response.json()["embeddings"][0]  # Extract embedding vector

# Spliting text into small token-sized chunks with default value as 1000
def chunk_text(text, chunk_size=1000):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Ranking chunks by relevance using embeddings
def rank_chunks(problem_statement, chunks):
    # problem_embed = get_embedding(problem_statement)
    # chunk_embeds = [get_embedding(chunk) for chunk in chunks]
    problem_embed = embedding_model.encode(problem_statement).reshape(1,-1)
    chunk_embeds = np.array([embedding_model.encode(chunk) for chunk in chunks])

    scores = cosine_similarity(problem_embed, chunk_embeds)[0]
    ranked_chunks = [x for _, x in sorted(zip(scores, chunks), reverse=True)]

    return ranked_chunks[:5]  # Retaining top 5 most relevant chunks
