"""Smoke test: verify all packages installed and API key works."""
import os
from google import genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
import pypdf

load_dotenv()

# Test 1: Gemini API
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
r = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents="Reply with just 'ok'.",
)
print(f"[1/3] Gemini API: {r.text.strip()}")

# Test 2: Embedding model (this downloads on first use, ~90MB)
print("[2/3] Loading embedding model (first run downloads ~90MB)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
sample_vec = model.encode("hello world")
print(f"      Embedding vector dimension: {len(sample_vec)}")

# Test 3: ChromaDB
client_db = chromadb.PersistentClient(path="./chroma_db")
collection = client_db.get_or_create_collection("smoke_test")
collection.upsert(ids=["1"], documents=["hello"], embeddings=[sample_vec.tolist()])
result = collection.query(query_embeddings=[sample_vec.tolist()], n_results=1)
print(f"[3/3] ChromaDB: retrieved {result['documents'][0][0]}")

print("\nAll systems working.")