# app.py

import streamlit as st
from src.retriever_hybrid import hybrid_search
from src.llm import generate_answer
from src.voice import listen_voice

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Educational RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ System Info")
st.sidebar.write("Model: Llama 3 (Ollama)")
st.sidebar.write("Retriever: Hybrid FAISS + BM25")
st.sidebar.write("Input Modes: Text + Voice")

# ---------------- TITLE ----------------
st.title("🤖 Educational RAG Chatbot")
st.caption("Textbook-grounded AI tutor with voice input")

# ---------------- SESSION STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- RAG FUNCTION ----------------
def rag_chat(query: str):
    results = hybrid_search(query, top_k=5)

    context = "\n\n".join(
        [r["chunk"] for r in results]
    )

    answer = generate_answer(context, query)
    return answer, results

# ---------------- INPUT CONTROLS ----------------
col1, col2 = st.columns([4, 1])

with col1:
    user_input = st.chat_input("Type your question...")

with col2:
    speak_clicked = st.button("🎤 Speak")

# ---------------- VOICE INPUT ----------------
if speak_clicked:
    with st.spinner("🎤 Listening… speak now"):
        voice_text = listen_voice()

    if voice_text == "__NO_MIC__":
        st.error("🎧 No microphone detected.")

    elif voice_text == "__UNRECOGNIZED__":
        st.warning("🎧 I heard something, but couldn't understand it.")

    elif voice_text == "__API_ERROR__":
        st.error("🌐 Speech recognition service error.")

    elif voice_text:
        st.success(f"🗣️ Captured: {voice_text}")
        user_input = voice_text

# ---------------- PROCESS QUERY ----------------
if user_input:
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    with st.spinner("Thinking..."):
        answer, sources = rag_chat(user_input)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

# ---------------- DISPLAY CHAT ----------------
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
