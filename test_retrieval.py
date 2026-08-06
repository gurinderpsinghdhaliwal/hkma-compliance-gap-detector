"""
Smoke test for the vector index. Run a few realistic compliance queries
and print the top matching regulatory chunks.
"""
from sentence_transformers import SentenceTransformer
import chromadb

DB_DIR = "chroma_db"
COLLECTION_NAME = "hkma_amlcft"
EMBED_MODEL = "all-MiniLM-L6-v2"

QUERIES = [
    "What are the HKMA's requirements for handling politically exposed persons?",
    "Transaction monitoring using artificial intelligence and machine learning",
    "Customer due diligence for high-risk jurisdictions",
    "Reporting obligations for suspicious transactions",
    "Sanctions screening controls",
]

TOP_K = 3


def main():
    print(f"Loading embedding model ({EMBED_MODEL})...")
    embedder = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"Connected to collection with {collection.count()} chunks.\n")

    for q in QUERIES:
        print(f"{'=' * 78}")
        print(f"QUERY: {q}")
        print(f"{'=' * 78}")
        q_embed = embedder.encode(q).tolist()
        results = collection.query(query_embeddings=[q_embed], n_results=TOP_K)

        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ), 1):
            print(f"\n  [{i}] source: {meta['source_pdf']}  (distance: {dist:.3f})")
            # Print first 400 chars of the matched chunk
            snippet = doc[:400].replace("\n", " ")
            print(f"      {snippet}...")
        print()


if __name__ == "__main__":
    main()