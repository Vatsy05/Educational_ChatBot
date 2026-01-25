# src_retriever

import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

VECTORSTORE_PATH = "../vectorstore"


def load_vectorstore():
    index = faiss.read_index(VECTORSTORE_PATH + "/faiss.index")
    with open(VECTORSTORE_PATH + "/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def clean_chunk(text):
    bad_words = ["contents", "index", "bibliography", "references", "exercise"]
    for w in bad_words:
        if w in text.lower():
            return None
    return text


def search(query, top_k=5):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, metadata = load_vectorstore()

    # Query expansion (huge boost)
    query = query + " definition explanation operating systems algorithms data structures"

    query_embedding = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        chunk = metadata[idx]["chunk"]
        cleaned = clean_chunk(chunk)
        if cleaned:
            results.append(metadata[idx])

    return results
