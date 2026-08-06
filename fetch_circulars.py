"""
Fetch AML/CFT circulars from the HKMA website.

Walks the two index pages (current + archive) for AML/CFT-related circulars
and guidance papers, extracts all PDF links, and downloads them locally
into corpus/pdfs/.

We do this respectfully: identify ourselves in the User-Agent, rate-limit
between downloads, and skip anything already fetched.
"""
import os
import time
import re
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

# HKMA AML/CFT circular index pages
INDEX_PAGES = [
    "https://www.hkma.gov.hk/eng/key-functions/banking/anti-money-laundering-and-counter-financing-of-terrorism/guidance-papers-circulars/",
    "https://www.hkma.gov.hk/eng/key-functions/banking/anti-money-laundering-and-counter-financing-of-terrorism/circulars-guidance-papers-archive/",
    "https://www.hkma.gov.hk/eng/key-functions/banking/anti-money-laundering-and-counter-financing-of-terrorism/aml-cft-related-information-for-stored-value-facility-licensees/circulars-2/",
]

HEADERS = {
    "User-Agent": "hkma-compliance-gap-research/1.0 (educational; contact via github.com/gurinderpsinghdhaliwal)"
}

CORPUS_DIR = "corpus/pdfs"
PAUSE_SECONDS = 1.0  # be polite to HKMA servers


def find_pdf_links(index_url: str) -> list[str]:
    """Load an HKMA index page and return all .pdf URLs it contains."""
    print(f"  scanning {index_url}")
    try:
        r = requests.get(index_url, headers=HEADERS, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print(f"    ! failed to load index: {e}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    pdf_urls = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if ".pdf" in href.lower():
            # Convert relative URLs to absolute
            full = urljoin(index_url, href)
            pdf_urls.add(full)
    print(f"    -> found {len(pdf_urls)} pdf links")
    return sorted(pdf_urls)


def safe_filename(url: str) -> str:
    """Make a safe local filename from a PDF URL."""
    path = urlparse(url).path
    name = os.path.basename(path)
    # Strip anything problematic; keep alphanumerics, dots, hyphens
    name = re.sub(r"[^A-Za-z0-9.\-_]", "_", name)
    return name or "unknown.pdf"


def download_pdf(url: str, target_dir: str) -> str | None:
    """Download a single PDF into target_dir. Skip if already present."""
    fname = safe_filename(url)
    path = os.path.join(target_dir, fname)
    if os.path.exists(path):
        return path  # already downloaded
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        # Basic sanity check that this is really a PDF
        if not r.content.startswith(b"%PDF"):
            print(f"    ! {fname} is not a PDF (got {r.content[:8]!r}), skipping")
            return None
        with open(path, "wb") as f:
            f.write(r.content)
        return path
    except Exception as e:
        print(f"    ! failed to download {fname}: {e}")
        return None


def main():
    os.makedirs(CORPUS_DIR, exist_ok=True)

    print("Step 1: discovering PDF links from HKMA AML/CFT index pages...")
    all_pdf_urls: set[str] = set()
    for index_url in INDEX_PAGES:
        all_pdf_urls.update(find_pdf_links(index_url))
        time.sleep(PAUSE_SECONDS)

    print(f"\nStep 2: downloading {len(all_pdf_urls)} unique PDFs...")
    downloaded = 0
    skipped = 0
    for i, url in enumerate(sorted(all_pdf_urls), 1):
        fname = safe_filename(url)
        target = os.path.join(CORPUS_DIR, fname)
        if os.path.exists(target):
            skipped += 1
            continue
        print(f"  [{i}/{len(all_pdf_urls)}] {fname}")
        result = download_pdf(url, CORPUS_DIR)
        if result:
            downloaded += 1
        time.sleep(PAUSE_SECONDS)

    print(f"\nDone. Downloaded {downloaded} new, skipped {skipped} already-present.")
    print(f"Corpus location: {os.path.abspath(CORPUS_DIR)}")


if __name__ == "__main__":
    main()