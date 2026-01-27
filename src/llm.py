# src/llm.py

import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"


def generate_answer(context, query):
    prompt = f"""
You are a computer science professor.

TASK:
Explain the concept clearly to a student.

RULES:
1. Do NOT mention pages, chapters, or books.
2. Do NOT quote the text directly.
3. Explain in your own words like teaching.
4. Use the context only as knowledge.
5. If missing, say: "I don't know based on the documents."

Context:
{context}

Question: {query}

Explain clearly:
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    return response.json()["response"]
