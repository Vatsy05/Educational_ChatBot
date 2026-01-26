# src/rag_pipeline.py

from retriever_hybrid import hybrid_search as search
from llm import generate_answer


def rag_query(query):
    results = search(query, top_k=5)

    context = "\n\n".join(
        [f"Source: {r['source']} Page: {r['page']}\n{r['chunk']}" for r in results]
    )

    answer = generate_answer(context, query)
    return answer, results


if __name__ == "__main__":
    print("🔥 Hybrid RAG Chatbot Ready (BM25 + FAISS). Type 'exit' to quit.\n")

    while True:
        query = input("Ask: ")
        if query.lower() == "exit":
            break

        answer, sources = rag_query(query)

        print("\n=========== ANSWER ===========\n")
        print(answer)

        print("\n=========== SOURCES ===========")
        for s in sources:
            print(f"{s['source']} | Page {s['page']}")
        print("================================\n")
