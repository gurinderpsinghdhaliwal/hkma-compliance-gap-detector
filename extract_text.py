"""
Extract text from HKMA AML/CFT circular PDFs into clean .txt files.

For each PDF in corpus/pdfs/, produce:
  corpus/texts/{filename}.txt   - plain extracted text
  corpus/texts/{filename}.json  - metadata (source URL, char count, page count)

Skips PDFs that look like scanned images (very low char/page ratio) since
we don't have OCR wired up. Reports what got skipped so you can see the shape
of the corpus.
"""
import os
import json
import re
from pypdf import PdfReader
from tqdm import tqdm

PDF_DIR = "corpus/pdfs"
TEXT_DIR = "corpus/texts"

# If a PDF averages fewer than this many characters per page,
# assume it's a scanned image and skip.
MIN_CHARS_PER_PAGE = 100


def clean_text(text: str) -> str:
    """Light cleanup: collapse whitespace, remove obvious page-noise patterns."""
    # Normalise all whitespace
    text = re.sub(r"\r\n?", "\n", text)
    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing spaces on each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def extract_one(pdf_path: str, out_txt: str, out_json: str) -> dict:
    """Extract text from one PDF, write outputs, return metadata dict."""
    reader = PdfReader(pdf_path)
    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception as e:
            pages_text.append(f"[extraction error on page: {e}]")
    full_text = clean_text("\n\n".join(pages_text))

    meta = {
        "filename": os.path.basename(pdf_path),
        "pages": len(reader.pages),
        "chars": len(full_text),
        "chars_per_page": round(len(full_text) / max(1, len(reader.pages)), 1),
        "looks_scanned": len(full_text) / max(1, len(reader.pages)) < MIN_CHARS_PER_PAGE,
    }

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(full_text)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def main():
    os.makedirs(TEXT_DIR, exist_ok=True)
    pdfs = sorted(f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf"))
    if not pdfs:
        print(f"No PDFs in {PDF_DIR}. Run fetch_circulars.py first.")
        return

    print(f"Extracting text from {len(pdfs)} PDFs...\n")
    ok = []
    scanned = []
    failed = []

    for fname in tqdm(pdfs, ncols=90):
        pdf_path = os.path.join(PDF_DIR, fname)
        base = os.path.splitext(fname)[0]
        out_txt = os.path.join(TEXT_DIR, f"{base}.txt")
        out_json = os.path.join(TEXT_DIR, f"{base}.json")
        try:
            meta = extract_one(pdf_path, out_txt, out_json)
            if meta["looks_scanned"]:
                scanned.append(fname)
            else:
                ok.append(fname)
        except Exception as e:
            failed.append((fname, str(e)))

    print(f"\n{'=' * 62}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'=' * 62}")
    print(f"Text-based PDFs (usable):    {len(ok)}")
    print(f"Scanned/image PDFs (skipped for now): {len(scanned)}")
    print(f"Failed:                       {len(failed)}")
    if scanned:
        print(f"\nScanned PDFs (will not be searchable without OCR):")
        for s in scanned[:10]:
            print(f"  - {s}")
        if len(scanned) > 10:
            print(f"  ... and {len(scanned) - 10} more")
    if failed:
        print(f"\nFailures:")
        for fname, err in failed[:5]:
            print(f"  - {fname}: {err[:100]}")
    print(f"\nOutput: {os.path.abspath(TEXT_DIR)}")


if __name__ == "__main__":
    main()