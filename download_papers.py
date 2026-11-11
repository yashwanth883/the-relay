import os
import sys
import subprocess
import urllib.request

try:
    import pypdf
except ImportError:
    print("Installing pypdf...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])
    import pypdf

PAPERS = {
    "attention": {
        "title": "Attention Is All You Need",
        "arxiv_id": "1706.03762v7",
        "url": "https://arxiv.org/pdf/1706.03762v7.pdf",
        "filename": "attention_1706.03762v7.pdf"
    },
    "moshi": {
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "arxiv_id": "2410.00037v1",
        "url": "https://arxiv.org/pdf/2410.00037v1.pdf",
        "filename": "moshi_2410.00037v1.pdf"
    },
    "dynamic_chunking": {
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "arxiv_id": "2507.07955v2",
        "url": "https://arxiv.org/pdf/2507.07955v2.pdf",
        "filename": "dynamic_chunking_2507.07955v2.pdf"
    }
}

os.makedirs("data/papers", exist_ok=True)
os.makedirs("data/extracted", exist_ok=True)

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TakeHomeResearch/1.0"}

for key, meta in PAPERS.items():
    pdf_path = os.path.join("data/papers", meta["filename"])
    if not os.path.exists(pdf_path):
        print(f"Downloading {meta['title']} ({meta['arxiv_id']})...")
        req = urllib.request.Request(meta["url"], headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp, open(pdf_path, "wb") as f:
            f.write(resp.read())
        print(f"Saved {pdf_path} ({os.path.getsize(pdf_path)} bytes)")
    else:
        print(f"Already exists: {pdf_path}")

    # Extract text by page
    print(f"Extracting text from {pdf_path}...")
    reader = pypdf.PdfReader(pdf_path)
    pages_text = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages_text.append(f"--- [Page {idx + 1}] ---\n{text}")

    full_text = "\n\n".join(pages_text)
    txt_path = os.path.join("data/extracted", f"{key}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Title: {meta['title']}\nArXiv ID: {meta['arxiv_id']}\nTotal Pages: {len(reader.pages)}\n\n")
        f.write(full_text)
    print(f"Extracted {len(reader.pages)} pages to {txt_path}")

print("All papers downloaded and extracted successfully!")

