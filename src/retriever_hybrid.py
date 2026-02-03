# src/retriever_hybrid.py

import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from src.router import route_query

# ---------- PATHS ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_PATH = os.path.join(BASE_DIR, "..", "vectorstore")

# ---------- LOAD VECTORSTORE ----------
def load_vectorstore():
    index_path = os.path.join(VECTORSTORE_PATH, "faiss.index")
    meta_path = os.path.join(VECTORSTORE_PATH, "metadata.pkl")

    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata

# ---------- BM25 ----------
def build_bm25(metadata):
    corpus = [m["chunk"].lower().split() for m in metadata]
    return BM25Okapi(corpus)

# ---------- FILTER BY DOMAIN ----------
def filter_by_domain(results, domain):
    if domain == "general":
        return results

    filtered = []
    for r in results:
        text = r["chunk"].lower()

        if domain == "dsa" and any(
            k in text for k in ["algorithm", "sort", "tree", "graph"]
        ):
            filtered.append(r)

        elif domain == "os" and any(
            k in text for k in ["memory", "process", "paging", "deadlock"]
        ):
            filtered.append(r)

    return filtered if filtered else results

# ---------- HYBRID SEARCH ----------
def hybrid_search(query, top_k=5):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, metadata = load_vectorstore()
    bm25 = build_bm25(metadata)

    domain = route_query(query)

    # ----- DOMAIN-AWARE QUERY EXPANSION -----
    if domain == "dsa":
        dense_query = query + " algorithms data structures sorting trees graphs"
    elif domain == "os":
        dense_query = query + " operating systems memory management processes"
    else:
        dense_query = query

    # ----- FAISS SEARCH -----
    q_emb = model.encode([dense_query]).astype("float32")
    faiss.normalize_L2(q_emb)
    _, dense_idx = index.search(q_emb, top_k)

    dense_results = [metadata[i] for i in dense_idx[0]]

    # ----- BM25 SEARCH -----
    scores = bm25.get_scores(query.lower().split())
    sparse_idx = np.argsort(scores)[::-1][:top_k]
    sparse_results = [metadata[i] for i in sparse_idx]

    # ----- MERGE -----
    combined = dense_results + sparse_results
    seen = set()
    merged = []

    for r in combined:
        key = (r["source"], r["page"], r["chunk"][:50])
        if key not in seen:
            merged.append(r)
            seen.add(key)

    # ----- DOMAIN FILTER -----
    final_results = filter_by_domain(merged, domain)

    return final_results[:top_k]
