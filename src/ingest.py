# src/ingest.py

import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

DATA_PATH = "../data/raw"
VECTORSTORE_PATH = "../vectorstore"

# Optimized for full textbooks
CHUNK_SIZE = 300
CHUNK_OVERLAP = 100
BATCH_SIZE = 64   # embedding batch size (safe for Mac)


def load_pdfs():
    documents = []

    for file in os.listdir(DATA_PATH):
        if not file.endswith(".pdf"):
            continue

        path = os.path.join(DATA_PATH, file)
        reader = PdfReader(path)
        pages = reader.pages  # LOAD ALL PAGES

        print(f"\n📘 Loading {file} with {len(pages)} pages")

        for page_num, page in enumerate(pages):
            try:
                text = page.extract_text()
            except:
                continue

            if text and len(text.strip()) > 300:
                documents.append({
                    "text": text,
                    "source": file,
                    "page": page_num + 1
                })

            # Progress log
            if page_num % 50 == 0:
                print(f"  Loaded page {page_num}")

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
    print("\n🧠 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [c["chunk"] for c in chunks]
    embeddings = []

    print(f"⚙️ Creating embeddings in batches of {BATCH_SIZE}...")

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        emb = model.encode(batch, show_progress_bar=False)
        embeddings.append(emb)

        if i % 1000 == 0:
            print(f"  Embedded {i}/{len(texts)} chunks")

    embeddings = np.vstack(embeddings).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, chunks


def save_index(index, metadata):
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)

    print("\n💾 Saving FAISS index...")
    faiss.write_index(index, VECTORSTORE_PATH + "/faiss.index")

    with open(VECTORSTORE_PATH + "/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)


if __name__ == "__main__":
    print("\n🚀 FULL TEXTBOOK INGESTION STARTED")

    docs = load_pdfs()
    print(f"\n✅ Loaded {len(docs)} pages")

    print("\n✂️ Chunking text...")
    chunks = create_chunks(docs)
    print(f"✅ Created {len(chunks)} chunks")

    index, metadata = build_faiss(chunks)

    save_index(index, metadata)

    print("\n🎉 FULL VECTOR DATABASE READY")
