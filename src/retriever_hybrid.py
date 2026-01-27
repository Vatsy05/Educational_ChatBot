# src/retriever_hybrid.py

import faiss
import pickle
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
import os

# Absolute path fix
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_PATH = os.path.join(BASE_DIR, "..", "vectorstore")


# Load FAISS + metadata
def load_vectorstore():
    index_path = os.path.join(VECTORSTORE_PATH, "faiss.index")
    meta_path = os.path.join(VECTORSTORE_PATH, "metadata.pkl")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")

    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata


# Build BM25 index
def build_bm25(metadata):
    corpus = [m["chunk"].lower().split() for m in metadata]
    return BM25Okapi(corpus)


# Clean garbage chunks
def clean_chunk(text):
    bad_words = ["contents", "index", "bibliography", "references", "exercise"]
    for w in bad_words:
        if w in text.lower():
            return None
    return text


# Hybrid Search
def hybrid_search(query, top_k=5):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, metadata = load_vectorstore()
    bm25 = build_bm25(metadata)

    # -------- Dense FAISS Search ----------
    dense_query = query + " operating systems algorithms data structures memory paging virtual memory"
    q_emb = model.encode([dense_query]).astype("float32")
    faiss.normalize_L2(q_emb)

    _, dense_indices = index.search(q_emb, top_k)
    dense_results = [metadata[i] for i in dense_indices[0]]

    # -------- Sparse BM25 Search ----------
    tokenized_query = query.lower().split()
    sparse_scores = bm25.get_scores(tokenized_query)
    sparse_indices = np.argsort(sparse_scores)[::-1][:top_k]
    sparse_results = [metadata[i] for i in sparse_indices]

    # -------- Merge + Deduplicate ----------
    combined = dense_results + sparse_results
    seen = set()
    final_results = []

    for r in combined:
        key = (r["source"], r["page"], r["chunk"][:50])
        if key not in seen:
            cleaned = clean_chunk(r["chunk"])
            if cleaned:
                final_results.append(r)
                seen.add(key)

    return final_results[:top_k]
