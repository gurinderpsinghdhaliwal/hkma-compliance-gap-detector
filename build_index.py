"""
Build a searchable vector index of HKMA AML/CFT circulars.

For each extracted text file in corpus/texts/, chunk it into overlapping
passages, embed each chunk with sentence-transformers, and store the
whole thing in a local ChromaDB collection.

Idempotent: safe to re-run. Existing chunks with the same IDs are upserted.
"""
import os
import json
import re
from typing import Iterator
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

TEXT_DIR = "corpus/texts"
DB_DIR = "chroma_db"
COLLECTION_NAME = "hkma_amlcft"

# Chunking parameters — tuned for regulatory text
CHUNK_WORDS = 400          # ~500-600 tokens; comfortable for retrieval
OVERLAP_WORDS = 80         # 20% overlap so ideas that span a boundary aren't lost

# Embedding model — small, fast, good enough for English regulatory text
EMBED_MODEL = "all-MiniLM-L6-v2"


def chunk_text(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    """
    Split text into overlapping chunks of roughly chunk_words words each.
    Overlaps preserve context around chunk boundaries.
    """
    # Normalise whitespace first — collapse runs of whitespace to single spaces
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split(" ")
    if len(words) <= chunk_words:
        return [text] if text else []

    chunks = []
    step = chunk_words - overlap_words
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_words])
        if chunk.strip():
            chunks.append(chunk)
        if start + chunk_words >= len(words):
            break
    return chunks


def iter_documents(text_dir: str) -> Iterator[tuple[str, str, dict]]:
    """Yield (doc_id, text, metadata) for each extracted regulatory document."""
    for fname in sorted(os.listdir(text_dir)):
        if not fname.endswith(".txt"):
            continue
        base = os.path.splitext(fname)[0]
        txt_path = os.path.join(text_dir, fname)
        json_path = os.path.join(text_dir, f"{base}.json")

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        meta = {}
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        yield base, text, meta


def main():
    print(f"Loading embedding model ({EMBED_MODEL})...")
    embedder = SentenceTransformer(EMBED_MODEL)

    print(f"Opening ChromaDB at ./{DB_DIR}...")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    print(f"\nChunking + embedding documents from ./{TEXT_DIR}...\n")
    total_chunks = 0
    docs = list(iter_documents(TEXT_DIR))

    for doc_id, text, meta in tqdm(docs, ncols=90):
        chunks = chunk_text(text, CHUNK_WORDS, OVERLAP_WORDS)
        if not chunks:
            continue

        # Batch-embed all chunks for this doc
        embeddings = embedder.encode(chunks, show_progress_bar=False)

        ids = [f"{doc_id}::chunk_{i:03d}" for i in range(len(chunks))]
        metadatas = [{
            "doc_id": doc_id,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "source_pdf": meta.get("filename", f"{doc_id}.pdf"),
            "source_pages": meta.get("pages", 0),
        } for i in range(len(chunks))]

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=[e.tolist() for e in embeddings],
            metadatas=metadatas,
        )
        total_chunks += len(chunks)

    print(f"\n{'=' * 62}")
    print(f"INDEX SUMMARY")
    print(f"{'=' * 62}")
    print(f"Documents indexed:   {len(docs)}")
    print(f"Chunks created:      {total_chunks}")
    print(f"Avg chunks per doc:  {total_chunks / max(1, len(docs)):.1f}")
    print(f"Total vectors in DB: {collection.count()}")
    print(f"Storage location:    {os.path.abspath(DB_DIR)}")


if __name__ == "__main__":
    main()