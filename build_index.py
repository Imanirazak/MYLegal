# build_index_numpy.py — one-time, FAISS-free index
import os, json
from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
INDEX_DIR = Path("index")
INDEX_DIR.mkdir(exist_ok=True)

EMB_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

def read_pdf(pdf: Path) -> List[Dict]:
    out = []
    r = PdfReader(str(pdf))
    for i, page in enumerate(r.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        t = t.replace("\u0000", " ").strip()
        if t:
            out.append({"page": i+1, "text": t})
    return out

def chunk(t: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    res, start = [], 0
    while start < len(t):
        end = min(start+size, len(t))
        res.append(t[start:end])
        if end == len(t): break
        start = max(0, end-overlap)
    return res

def main():
    if not any(DATA_DIR.glob("*.pdf")):
        raise SystemExit("No PDFs in ./data. Add Malaysian law PDFs and re-run.")

    docs = []
    for pdf in DATA_DIR.glob("*.pdf"):
        for p in read_pdf(pdf):
            for ch in chunk(p["text"]):
                docs.append({"file": pdf.name, "page": p["page"], "text": ch})

    print(f"Loaded {len(docs)} chunks. Embedding…")
    model = SentenceTransformer(EMB_MODEL_NAME)
    embs = model.encode(
        [d["text"] for d in docs],
        batch_size=64, show_progress_bar=True, normalize_embeddings=True
    ).astype("float32")

    np.save("index/embeddings.npy", embs)
    with open("index/meta.jsonl","w",encoding="utf-8") as f:
        for d in docs: f.write(json.dumps(d, ensure_ascii=False)+"\n")
    with open("index/config.json","w",encoding="utf-8") as f:
        json.dump({
            "emb_model": EMB_MODEL_NAME,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP
        }, f)
    print("✅ Saved index to ./index (embeddings.npy, meta.jsonl, config.json)")

if __name__ == "__main__":
    main()
