"""
Topic-scan gap detector: finds SILENT OMISSIONS.

The section-by-section detector (gap_detector.py) can only find gaps in
topics the policy already talks about. This scanner does the reverse:
it lists the major regulatory topics that appear in the HKMA corpus, then
checks whether each one is addressed anywhere in the policy at all.

If a corpus topic has no strong match in the policy, it's flagged as a
silent omission.
"""
import os
import json
import re
import time
from collections import defaultdict
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

# --- Configuration ---
POLICY_PATH = "sample_policy.md"
DB_DIR = "chroma_db"
COLLECTION_NAME = "hkma_amlcft"
EMBED_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.1-flash-lite"

# Silent-omission threshold: if the closest policy chunk to a corpus topic
# is further than this, the topic is considered absent from the policy.
OMISSION_DISTANCE = 1.15

# Chunk the policy for reverse-search
POLICY_CHUNK_WORDS = 200
POLICY_OVERLAP = 40

# How many corpus topics to test
MAX_TOPICS = 30

PAUSE = 4.5


TOPIC_DISTILL_PROMPT = """You are cataloguing the topics covered by a set of HKMA AML/CFT regulatory documents.

You will be given a list of document filenames and short excerpts from each. Produce a JSON array of the DISTINCT regulatory topics these documents address, from the perspective of an internal bank compliance policy.

Focus on topics that an AML policy would need to address explicitly. Examples of good topic labels:
- "Authorized payment scams (APP fraud) detection and response"
- "FATF high-risk jurisdictions and public statements"
- "Politically exposed persons (PEPs) enhanced due diligence"
- "Transaction monitoring system governance and model validation"
- "AI/ML in transaction monitoring: explainability and human oversight"
- "Sanctions screening: UNSC lists and alerting"

Return STRICT JSON, no prose:
{
  "topics": [
    {
      "label": "short human-readable topic label (10-15 words)",
      "description": "one sentence describing the regulatory expectation on this topic",
      "example_sources": ["source_pdf_name_1", "source_pdf_name_2"]
    }
  ]
}

Aim for 20-30 distinct topics. Merge near-duplicates. Prefer specific over generic (prefer "AI/ML in transaction monitoring" over "technology in compliance")."""


def chunk_policy(policy_text: str, chunk_words: int, overlap: int) -> list[str]:
    """Chunk the policy so we can do reverse semantic search into it."""
    text = re.sub(r"\s+", " ", policy_text).strip()
    words = text.split(" ")
    if len(words) <= chunk_words:
        return [text]
    step = chunk_words - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_words]))
        if start + chunk_words >= len(words):
            break
    return chunks


def sample_corpus_titles_and_openings(collection) -> str:
    """
    Pull one representative chunk (chunk_index 0) per unique document to
    give Gemini a broad view of the corpus for topic distillation.
    """
    # Chroma doesn't have a native "distinct doc" op, so pull everything then dedupe
    all_docs = collection.get()  # get everything (limit is high for our size)
    by_doc = {}
    for i, meta in enumerate(all_docs["metadatas"]):
        doc_id = meta["doc_id"]
        # Prefer the earliest chunk (chunk_index 0)
        if doc_id not in by_doc or meta["chunk_index"] < by_doc[doc_id][1]:
            by_doc[doc_id] = (all_docs["documents"][i], meta["chunk_index"], meta["source_pdf"])

    lines = []
    for doc_id, (text, _, source) in sorted(by_doc.items()):
        # First 400 chars of the opening chunk
        opening = text[:400].replace("\n", " ")
        lines.append(f"=== {source} ===\n{opening}\n")
    return "\n".join(lines)


def distill_topics(client, corpus_overview: str) -> list[dict]:
    """Ask Gemini to distill the corpus into a list of regulatory topics."""
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=TOPIC_DISTILL_PROMPT,
            response_mime_type="application/json",
            temperature=0.2,
        ),
        contents=f"Corpus overview:\n\n{corpus_overview}",
    )
    return json.loads(response.text)["topics"]


def main():
    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)

    print(f"Opening ChromaDB collection...")
    db = chromadb.PersistentClient(path=DB_DIR)
    collection = db.get_collection(name=COLLECTION_NAME)
    print(f"  {collection.count()} regulatory chunks available")

    print(f"Loading policy: {POLICY_PATH}")
    with open(POLICY_PATH, "r", encoding="utf-8") as f:
        policy = f.read()

    policy_chunks = chunk_policy(policy, POLICY_CHUNK_WORDS, POLICY_OVERLAP)
    print(f"  Split policy into {len(policy_chunks)} chunks for reverse search")
    policy_embeddings = embedder.encode(policy_chunks)

    print(f"\nStep 1: distilling regulatory topics from the corpus...")
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    overview = sample_corpus_titles_and_openings(collection)
    # Truncate if too long — flash-lite can handle plenty, but be safe
    if len(overview) > 60_000:
        overview = overview[:60_000]

    topics = distill_topics(client, overview)
    topics = topics[:MAX_TOPICS]
    print(f"  Distilled {len(topics)} distinct topics from the corpus\n")

    print(f"Step 2: reverse-searching each topic against the policy...\n")

    findings = []
    for i, topic in enumerate(topics, 1):
        label = topic["label"]
        desc = topic["description"]
        # Embed the topic and find the closest policy chunk
        query = f"{label}. {desc}"
        q_embed = embedder.encode(query)

        # Cosine-style distance approximation using vector similarity
        # (Chroma uses L2 by default, so we replicate with numpy here)
        import numpy as np
        pol_arr = np.array(policy_embeddings)
        q_arr = np.array(q_embed)
        dists = np.linalg.norm(pol_arr - q_arr, axis=1)
        best_dist = float(dists.min())
        best_chunk_idx = int(dists.argmin())

        status = "OMITTED" if best_dist > OMISSION_DISTANCE else "addressed"
        print(f"  [{i}/{len(topics)}] [{status}] {label[:65]}  (best d={best_dist:.2f})")

        findings.append({
            "topic": label,
            "description": desc,
            "example_sources": topic.get("example_sources", []),
            "closest_policy_distance": round(best_dist, 3),
            "closest_policy_snippet": policy_chunks[best_chunk_idx][:200],
            "status": status,
        })

    omitted = [f for f in findings if f["status"] == "OMITTED"]

    print(f"\n{'=' * 62}")
    print(f"SILENT-OMISSION SCAN SUMMARY")
    print(f"{'=' * 62}")
    print(f"Corpus topics tested:        {len(findings)}")
    print(f"Topics addressed in policy:  {len(findings) - len(omitted)}")
    print(f"Topics silently omitted:     {len(omitted)}")
    print()
    if omitted:
        print("Silent omissions (topics HKMA covers but the policy does not):")
        for f in omitted:
            print(f"  - {f['topic']}")
            if f["example_sources"]:
                print(f"      sources: {', '.join(f['example_sources'][:2])}")

    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"results/topic_scan_{ts}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "policy_file": POLICY_PATH,
            "model": GEMINI_MODEL,
            "timestamp": datetime.now().isoformat(),
            "omission_threshold": OMISSION_DISTANCE,
            "findings": findings,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()