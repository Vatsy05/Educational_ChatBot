# app.py

import streamlit as st
from src.retriever_hybrid import hybrid_search
from src.llm import generate_answer

# Page config
st.set_page_config(page_title="Educational RAG Chatbot", page_icon="🤖", layout="wide")

# Sidebar
st.sidebar.title("⚙️ System Info")
st.sidebar.write("Model: Llama 3 (Ollama)")
st.sidebar.write("Retriever: FAISS + BM25 Hybrid")
st.sidebar.write("Books: CLRS + Silberschatz OS")

# Title
st.title("🤖 Educational RAG Chatbot")
st.caption("Full Textbook AI Tutor (Hybrid RAG)")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# RAG function
def rag_chat(query):
    results = hybrid_search(query, top_k=5)

    context = "\n\n".join(
        [f"Source: {r['source']} Page: {r['page']}\n{r['chunk']}" for r in results]
    )

    answer = generate_answer(context, query)
    return answer, results


# Input box
user_input = st.chat_input("Ask me anything about DSA or Operating Systems...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        answer, sources = rag_chat(user_input)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })


# Display chat
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        with st.chat_message("user"):
            st.write(chat["content"])

    else:
        with st.chat_message("assistant"):
            st.write(chat["content"])

            with st.expander("📚 Sources"):
                for s in chat["sources"]:
                    st.write(f"{s['source']} | Page {s['page']}")
