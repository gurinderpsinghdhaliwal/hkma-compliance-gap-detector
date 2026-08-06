# HKMA AML/CFT Compliance Gap Detector

A working prototype that reads a bank's internal Anti-Money Laundering policy, cross-references it against 62 HKMA regulatory circulars and guidance papers, and produces a compliance gap report — the kind of first-pass analysis a compliance team would otherwise assign to a junior analyst for a full afternoon.

It's a small RAG (retrieval-augmented generation) system built specifically for Hong Kong AML/CFT compliance, tested against a sample bank policy with deliberately-seeded gaps to see whether the detector would catch them. It caught most of them cleanly. Where it didn't, the *reason* it didn't was itself an interesting finding — which I'll explain below.

## Why I built this

I'm applying for finance-adjacent AI roles in Hong Kong, and RegTech (regulatory technology) is one of the more active corners of the HK financial-services market. Almost every product being built in this space right now uses some flavour of RAG over regulatory documents. I wanted to build a working example over the actual HKMA corpus — not a generic tutorial, not a corpus I could have downloaded pre-processed from Hugging Face — to see how it holds up on a real regulatory dataset with a real compliance question.

This is a companion to my [Hindi Adverse Media Screener](https://github.com/gurinderpsinghdhaliwal/hindi-adverse-media-screener) — same idea (build the thing, benchmark it, be honest about what it does and doesn't do), different corner of RegTech.

## What it does

Two detectors working together over the same regulatory corpus:

**1. Section-by-section detector.** Walks through a bank's AML policy one section at a time. For each section, retrieves the 5 most relevant chunks from the HKMA corpus using semantic search, then asks the language model: "given these HKMA requirements, does this policy section fully address them? What's missing?" Returns a structured list of covered requirements, gaps (with severity), and partial coverage. Each finding is anchored to a specific HKMA circular by filename, and each gap includes a direct quote from the source.

**2. Topic-scan detector.** Does the reverse. Distills the ~15 most distinct regulatory topics from the corpus, then checks each topic against the policy. If a topic has no semantic match anywhere in the policy, it's flagged as a **silent omission** — a topic HKMA has issued guidance on but the policy doesn't discuss at all.

Both are needed. The section-by-section detector finds gaps in areas the policy already talks about. The topic scan finds areas the policy is entirely blind to. Real compliance analysts implicitly do both.

## The corpus

62 HKMA AML/CFT circulars, guidance papers, and thematic reviews. Fetched from three HKMA index pages: current AML/CFT circulars for Authorized Institutions, the archive, and the parallel set for Stored Value Facility licensees. All PDFs, all text-based (zero scanned images — HKMA publishes clean documents), all successfully extracted. Coverage spans 2018 to 2025, weighted toward more recent guidance.

Chunked into ~500 pieces of ~400 words each, embedded with `all-MiniLM-L6-v2`, stored in ChromaDB.

## The sample policy

I couldn't test this against a real bank's confidential AML policy, so I wrote a realistic one: a 10-section AML/CFT policy for a fictional mid-tier HK bank ("Meridian Bank Hong Kong Limited"). Each section is written the way a real policy would be — covering the basic requirements at a surface level, with deliberately-seeded gaps of three kinds:

- **Complete gaps** — topics HKMA requires that the policy doesn't mention at all (Authorized Payment Scams, FATF high-risk jurisdictions, AI/ML governance in transaction monitoring)
- **Partial coverage** — topics the policy touches but leaves substantive requirements unaddressed (ongoing monitoring, third-party outsourcing)
- **Genuine coverage** — topics the policy handles adequately (governance, basic CDD, sanctions screening against UN/OFAC/EU, STR to MLRO, record retention periods)

The seeded gaps aren't marked in the file — they're just baked into how the policy was written, so the detector has to find them without hints.

## What I found

Running both detectors against the sample policy:

- **10 policy sections analysed**
- **16 requirements identified as covered**
- **21 gaps flagged** (7 HIGH severity)
- **5 partial-coverage findings**
- **2 silent omissions** identified by the topic scan

The section-by-section detector produced substantive, well-cited findings. Not vague "policy could say more about X" comments — actual regulatory requirements like *"Appoint a Compliance Officer at management level (SVF Guideline 5.16)"* or *"Identify and verify persons acting on behalf of the customer (SVF Guideline paragraph on section 4.7)"*, each anchored to a direct quote from the source circular. These are the kinds of gaps a bank could genuinely fail an inspection on.

The topic scan caught the Authorized Payment Scams omission cleanly, with the April 2025 APP circular correctly cited as the source.

## What it didn't catch — and why

The topic scan misclassified two of the seeded gaps as "addressed":

- **AI/ML in transaction monitoring** (distance 0.93 to the closest policy chunk) — marked addressed
- **FATF high-risk jurisdictions** (distance 0.96) — marked addressed

Both misses have the same underlying cause. The policy Section 6 says *"The Bank operates an automated transaction monitoring system,"* and Section 5 lists UN, OFAC, and EU sanctions databases. Those sentences embed close in vector space to *any* query about transaction monitoring or lists of jurisdictions — even when the substantive requirement (AI model validation, FATF-specific enhanced due diligence) isn't remotely addressed.

This is a genuine limitation of similarity-only scanning: **proximity in vector space is not the same as substantive coverage**. The policy is in the *neighbourhood* of the topic, so the scanner assumes it's covered. Real production RegTech tools hit this exact problem.

The fix — which I've scoped but haven't built yet — is a two-stage check. First identify topics the policy is in the neighbourhood of; then use the LLM to verify whether the policy actually addresses the specific regulatory requirement, not just talks *near* it. That's a next iteration, not something wrong with the design.

## What I think this means

The value of this project isn't the specific gaps it caught. It's that the pipeline structure — semantic retrieval → LLM analysis with strict grounding constraints → structured output → readable report — actually works over a real regulatory corpus. The findings are specific, citation-backed, and reproducible.

The value of the mistakes it made is even higher, honestly. Any working RAG system in RegTech has to deal with the "close in vector space, not close in substance" problem. Discovering it on a controlled sample where I could see exactly why is a much better learning than reading about it in a paper.

## Limitations I want to be upfront about

- **Sample policy only.** I built and tested this against a policy I wrote myself, so I know where the gaps are. Testing against a real, unseen bank policy would be the next step — and I don't have permission to do that.
- **The corpus mixes Authorized Institution and SVF Licensee guidance.** HKMA's AML frameworks for the two entity types mirror each other closely, but a production system should filter by the regulated-entity type it's checking against. Several of the findings in the sample report cite SVF-specific guidance for what would be an AI policy — technically fine because the requirements match, but architecturally you'd want a cleaner split.
- **Vector-similarity scanning has known limits** (see above). The two-stage LLM verifier that fixes it is scoped but unbuilt.
- **No human-labelled ground truth.** I read the outputs myself and cross-referenced against the source circulars for accuracy. A real evaluation would need a compliance analyst to label the data properly.
- **Retrieval quality varies by topic.** Some topics (transaction monitoring, sanctions) retrieve strong matches (distance ~0.7). Others (specific enforcement procedures) drift into weaker territory (>1.0). The pipeline currently accepts up to distance 1.10, which the prompt tells the model to handle gracefully — but a stricter threshold with fallback logic would be better.
- **First-pass triage, not final judgment.** Everything in the report should be reviewed by a qualified compliance professional. This is stated in the report itself. It matters.

## Files

- `fetch_circulars.py` — scrapes HKMA AML/CFT index pages, downloads PDFs
- `extract_text.py` — extracts clean text from PDFs, saves per-document metadata
- `build_index.py` — chunks, embeds, and stores the corpus in ChromaDB
- `test_retrieval.py` — smoke test for retrieval quality
- `gap_detector.py` — section-by-section detector
- `topic_scan.py` — silent-omission detector
- `render_report.py` — turns the JSON outputs into a readable Markdown report
- `sample_policy.md` — the fictional bank policy used for testing
- `gap_report.md` — a sample output showing what the pipeline produces

## How to run it

Python 3.10+ and a free Google AI Studio API key.

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1     # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root with `GEMINI_API_KEY=your-key-here`.

Then, in order:

```bash
python fetch_circulars.py    # downloads ~60 PDFs into corpus/pdfs/
python extract_text.py       # extracts text into corpus/texts/
python build_index.py        # builds the vector store in chroma_db/
python gap_detector.py       # section-by-section analysis
python topic_scan.py         # silent-omission scan
python render_report.py      # produces gap_report.md
```

Everything after the first three steps is fast and can be re-run repeatedly; the fetch, extract, and index steps are one-off.

## A note on tooling

Free tier of Google's Gemini API on `gemini-3.1-flash-lite`. Chose it deliberately — the whole project should be reproducible without a credit card. Google shuffled available models several times during development, so the model name is a single constant at the top of each script.

Embedding model is `all-MiniLM-L6-v2` from sentence-transformers, running on CPU. Small, fast, good enough for English regulatory text. A larger model like `bge-large` would give sharper retrieval but for this corpus size the improvement isn't worth the complexity.

ChromaDB persistent client stores everything locally in `chroma_db/`. Not committed to the repo (gitignored) since it's regenerable and large.

---

Built by Gary Singh · Hong Kong