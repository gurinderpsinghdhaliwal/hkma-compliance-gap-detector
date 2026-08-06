"""
HKMA AML/CFT Compliance Gap Detector.

For each section of an internal AML policy, retrieve the most relevant
chunks from the HKMA regulatory corpus and ask Gemini to identify gaps.

Outputs a structured JSON report plus a per-section terminal summary.
"""
import os
import json
import re
import time
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

# Retrieval: how many regulatory chunks to consider per policy section.
TOP_K = 5

# Retrieval quality filter: discard chunks with distance above this.
# (Recall from test_retrieval: 0.6-0.75 = strong match, 0.85+ = drift)
MAX_DISTANCE = 1.10

# Rate-limit pacing (~15 req/min free tier on flash-lite)
PAUSE = 4.5


# --- Prompt: the whole product ---
SYSTEM_PROMPT = """You are an AML/CFT compliance analyst reviewing an internal bank policy against HKMA regulatory requirements.

You will be given:
1. One section of a bank's internal AML/CFT policy.
2. Several excerpts from HKMA circulars and guidance papers that were retrieved as most relevant to that section.

Your task: identify where the policy section fully addresses HKMA requirements, where it is silent on requirements HKMA has specifically laid out, and where it partially addresses them.

Only flag gaps or partial coverage where the HKMA excerpts you have been given contain a concrete requirement or expectation. Do NOT invent regulatory requirements. Do NOT flag general best practices as gaps unless they are explicit in the excerpts. If the excerpts do not contain a clear requirement relevant to the policy section, report no findings rather than speculating.

Return STRICT JSON matching this schema, with no prose before or after:
{
  "section_summary": "one-sentence description of what this policy section covers",
  "covered": [
    {
      "requirement": "the HKMA requirement, short phrasing",
      "policy_evidence": "short quote or paraphrase from the policy showing coverage",
      "source_chunks": ["source_pdf_name_1", "source_pdf_name_2"]
    }
  ],
  "gaps": [
    {
      "requirement": "the HKMA requirement that is missing",
      "why_gap": "one sentence on why the policy fails to address this",
      "hkma_quote": "short direct quote from an HKMA excerpt that establishes the requirement",
      "source_chunks": ["source_pdf_name_1"],
      "severity": "HIGH | MEDIUM | LOW"
    }
  ],
  "partial": [
    {
      "requirement": "the HKMA requirement partially addressed",
      "what_is_covered": "which part of the requirement the policy addresses",
      "what_is_missing": "which part is silent or vague",
      "source_chunks": ["source_pdf_name_1"]
    }
  ]
}

If a category has no entries, return an empty list."""


# --- Policy parser ---
SECTION_HEADING = re.compile(r"^##\s+(\d+)\.\s+(.+?)\s*$", re.MULTILINE)


def parse_policy_sections(policy_text: str) -> list[dict]:
    """Split the markdown policy into numbered sections."""
    matches = list(SECTION_HEADING.finditer(policy_text))
    sections = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(policy_text)
        sections.append({
            "number": m.group(1),
            "title": m.group(2).strip(),
            "body": policy_text[start:end].strip(),
        })
    return sections


# --- Retrieval ---
def retrieve_relevant_chunks(query: str, embedder, collection, top_k: int, max_dist: float) -> list[dict]:
    """Semantic search: return top_k chunks with distance < max_dist."""
    q_embed = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[q_embed], n_results=top_k)
    kept = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if dist <= max_dist:
            kept.append({"text": doc, "source": meta["source_pdf"], "distance": round(dist, 3)})
    return kept


# --- Gap detection call ---
def analyse_section(client, section: dict, relevant_chunks: list[dict]) -> dict:
    """Ask Gemini to compare the policy section against retrieved HKMA chunks."""
    if not relevant_chunks:
        return {
            "section_summary": "",
            "covered": [],
            "gaps": [],
            "partial": [],
            "note": "No relevant HKMA chunks retrieved within distance threshold.",
        }

    # Format the retrieved chunks with clear source attribution
    excerpts_block = "\n\n".join(
        f"--- HKMA Excerpt {i+1} (source: {c['source']}) ---\n{c['text']}"
        for i, c in enumerate(relevant_chunks)
    )

    user_msg = (
        f"POLICY SECTION (from bank's internal AML/CFT policy):\n\n"
        f"{section['body']}\n\n"
        f"{'=' * 60}\n\n"
        f"RELEVANT HKMA EXCERPTS:\n\n"
        f"{excerpts_block}"
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.1,
        ),
        contents=user_msg,
    )
    return json.loads(response.text)


# --- Main pipeline ---
def main():
    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)

    print(f"Opening ChromaDB collection '{COLLECTION_NAME}'...")
    db = chromadb.PersistentClient(path=DB_DIR)
    collection = db.get_collection(name=COLLECTION_NAME)
    print(f"  {collection.count()} regulatory chunks available\n")

    print(f"Loading policy: {POLICY_PATH}")
    with open(POLICY_PATH, "r", encoding="utf-8") as f:
        policy = f.read()
    sections = parse_policy_sections(policy)
    print(f"  Parsed {len(sections)} sections: {[s['title'] for s in sections]}\n")

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    report = {
        "policy_file": POLICY_PATH,
        "model": GEMINI_MODEL,
        "timestamp": datetime.now().isoformat(),
        "sections": [],
    }

    for i, section in enumerate(sections, 1):
        title = f"Section {section['number']}: {section['title']}"
        print(f"[{i}/{len(sections)}] {title}")

        # Use section title + first 500 chars of body as the retrieval query
        query = f"{section['title']}. {section['body'][:500]}"
        chunks = retrieve_relevant_chunks(
            query, embedder, collection, TOP_K, MAX_DISTANCE
        )
        print(f"    retrieved {len(chunks)} relevant chunks (top source: {chunks[0]['source'] if chunks else 'none'})")

        try:
            analysis = analyse_section(client, section, chunks)
        except Exception as e:
            print(f"    ! analysis failed: {e}")
            time.sleep(PAUSE)
            continue

        n_covered = len(analysis.get("covered", []))
        n_gaps = len(analysis.get("gaps", []))
        n_partial = len(analysis.get("partial", []))
        print(f"    -> {n_covered} covered | {n_gaps} gaps | {n_partial} partial")

        for g in analysis.get("gaps", []):
            severity = g.get("severity", "?")
            print(f"       GAP [{severity}]: {g.get('requirement', '')[:80]}")

        report["sections"].append({
            "section_number": section["number"],
            "section_title": section["title"],
            "retrieved_chunks": chunks,
            "analysis": analysis,
        })

        time.sleep(PAUSE)

    # Save the full report
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"results/gap_report_{ts}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Summary
    total_covered = sum(len(s["analysis"].get("covered", [])) for s in report["sections"])
    total_gaps = sum(len(s["analysis"].get("gaps", [])) for s in report["sections"])
    total_partial = sum(len(s["analysis"].get("partial", [])) for s in report["sections"])
    high_gaps = sum(
        1 for s in report["sections"]
        for g in s["analysis"].get("gaps", [])
        if g.get("severity") == "HIGH"
    )

    print(f"\n{'=' * 62}")
    print(f"GAP DETECTION SUMMARY")
    print(f"{'=' * 62}")
    print(f"Policy sections analysed:  {len(report['sections'])}")
    print(f"Requirements covered:      {total_covered}")
    print(f"Gaps identified:           {total_gaps} ({high_gaps} HIGH severity)")
    print(f"Partial coverage:          {total_partial}")
    print(f"\nFull report saved to: {out}")


if __name__ == "__main__":
    main()