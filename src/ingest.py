# src/ingest.py

import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

DATA_PATH = "../data/raw"
VECTORSTORE_PATH = "../vectorstore"

CHUNK_SIZE = 200
CHUNK_OVERLAP = 100


def load_pdfs():
    documents = []

    for file in os.listdir(DATA_PATH):
        if not file.endswith(".pdf"):
            continue

        path = os.path.join(DATA_PATH, file)
        reader = PdfReader(path)

        # Select pages smartly
        if "CLRS" in file:
            pages = reader.pages[:250]  # algorithms
        elif "OS" in file:
            pages = reader.pages[300:500]  # deadlock section
        else:
            pages = reader.pages[:200]

        for page_num, page in enumerate(pages):
            text = page.extract_text()
            if text and len(text.strip()) > 300:
                documents.append({
                    "text": text,
                    "source": file,
                    "page": page_num + 1
                })

    return documents


def chunk_text(text):
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = words[i:i + CHUNK_SIZE]
        chunks.append(" ".join(chunk))
    return chunks


def create_chunks(docs):
    all_chunks = []
    for doc in docs:
        for chunk in chunk_text(doc["text"]):
            all_chunks.append({
                "chunk": chunk,
                "source": doc["source"],
                "page": doc["page"]
            })
    return all_chunks


def build_faiss(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["chunk"] for c in chunks]

    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, chunks


def save_index(index, metadata):
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    faiss.write_index(index, VECTORSTORE_PATH + "/faiss.index")

    with open(VECTORSTORE_PATH + "/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)


if __name__ == "__main__":
    print("Loading PDFs...")
    docs = load_pdfs()
    print(f"Loaded {len(docs)} pages")

    print("Chunking...")
    chunks = create_chunks(docs)
    print(f"Created {len(chunks)} chunks")

    print("Building FAISS index...")
    index, metadata = build_faiss(chunks)

    print("Saving vectorstore...")
    save_index(index, metadata)

    print("DONE.")
