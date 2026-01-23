## Software Requirements Specification (SRS)

# 1. Introduction

## 1.1 Purpose

The purpose of this document is to specify the requirements for the **RAG-Based Educational Chatbot**, a system designed to provide accurate and context-aware answers to computer science academic queries using Retrieval-Augmented Generation (RAG). The system integrates document retrieval with large language models (LLMs) to ensure grounded and reliable responses.

## 1.2 Scope

The RAG-Based Educational Chatbot will:

* Ingest academic documents such as textbooks and lecture notes.
* Convert documents into semantic embeddings.
* Store embeddings in a FAISS vector database.
* Retrieve relevant content based on user queries.
* Use an LLM to generate answers grounded in retrieved content.
* Provide an interactive web-based interface using Streamlit.

The system is intended for students and learners seeking conceptual explanations in computer science subjects such as Data Structures, Algorithms, and Operating Systems.

## 1.3 Definitions, Acronyms, and Abbreviations

* **RAG**: Retrieval-Augmented Generation
* **LLM**: Large Language Model
* **FAISS**: Facebook AI Similarity Search
* **Embedding**: Vector representation of text
* **Chunking**: Splitting documents into smaller segments
* **UI**: User Interface

## 1.4 References

* Sentence Transformers Documentation
* FAISS Documentation
* Streamlit Documentation
* CLRS (Introduction to Algorithms)

---

# 2. Overall Description

## 2.1 Product Perspective

The system is a standalone AI-based academic assistant built using Python, FAISS, Sentence Transformers, and Streamlit. It follows a modular architecture separating ingestion, retrieval, and generation.

## 2.2 Product Functions

* Document ingestion and preprocessing
* Text chunking and embedding generation
* Vector database indexing
* Semantic search and retrieval
* LLM-based answer generation
* Web-based chat interface

## 2.3 User Classes and Characteristics

* **Students**: Seek conceptual explanations and academic help.
* **Developers/Researchers**: Explore RAG architectures and NLP applications.

## 2.4 Operating Environment

* Programming Language: Python 3.10+
* OS: Windows, macOS, Linux
* Libraries: SentenceTransformers, FAISS, Streamlit, PyPDF, OpenAI/Local LLM

## 2.5 Design and Implementation Constraints

* Limited to static document knowledge (no real-time data)
* LLM API limits or local compute constraints
* Memory constraints for large vector indexes

## 2.6 Assumptions and Dependencies

* Users provide valid academic PDFs
* Internet access for model downloads (if not cached)
* GPU optional for faster embedding generation

---

# 3. System Requirements

## 3.1 Functional Requirements

### FR1: Document Ingestion

The system shall allow users to upload academic PDF documents for indexing.

### FR2: Text Chunking

The system shall split documents into overlapping text chunks for embedding.

### FR3: Embedding Generation

The system shall convert text chunks into vector embeddings using a pretrained embedding model.

### FR4: Vector Storage

The system shall store embeddings in a FAISS vector database.

### FR5: Semantic Retrieval

The system shall retrieve the top-k most relevant chunks for a given query.

### FR6: Context-Aware Answer Generation

The system shall use an LLM to generate answers using retrieved context.

### FR7: User Interface

The system shall provide a chat-based web interface using Streamlit.

### FR8: Source Attribution

The system shall display retrieved document sources alongside answers.

---

## 3.2 Non-Functional Requirements

### Performance Requirements

* Query response time should be less than 5 seconds for moderate dataset sizes.
* Embedding generation should support batch processing.

### Security Requirements

* No personal data storage
* API keys must be stored securely in environment variables.

### Usability Requirements

* Simple web UI with chat history
* Clear instructions for document ingestion

### Scalability Requirements

* System should support incremental document ingestion.
* Vector index should scale to thousands of document chunks.

### Reliability Requirements

* The system should handle missing or corrupted documents gracefully.

---

# 4. System Architecture

## 4.1 High-Level Architecture

1. Document Loader
2. Text Chunker
3. Embedding Generator
4. FAISS Vector Store
5. Retriever Module
6. LLM Generator
7. Streamlit UI

## 4.2 Data Flow

1. User uploads documents
2. Documents are chunked and embedded
3. Embeddings stored in FAISS
4. User query converted to embedding
5. Relevant chunks retrieved
6. LLM generates answer using context

---

# 5. External Interface Requirements

## 5.1 User Interface

* Web-based chat interface
* Display retrieved context snippets
* Input box for user queries

## 5.2 Hardware Interfaces

* Optional GPU for embedding acceleration

## 5.3 Software Interfaces

* FAISS for vector search
* SentenceTransformers for embeddings
* OpenAI or local LLM APIs

---

# 6. Future Enhancements

* Hybrid retrieval (BM25 + embeddings)
* Reranking using cross-encoders
* Multi-document upload via UI
* Conversation memory
* Citation scoring and confidence estimation

---

# 7. Appendix

## 7.1 Project Directory Structure

```
rag-edu-bot/
├── app.py
├── ingest.py
├── retriever.py
├── llm.py
├── rag_pipeline.py
├── data/
│   ├── raw/
│   └── processed/
├── vectorstore/
│   └── faiss.index
└── requirements.txt
```

## 7.2 Glossary

* **Chunk**: A segment of text used for embedding.
* **Embedding Model**: Neural network converting text to vectors.
* **Retriever**: Component that fetches relevant chunks from FAISS.
* **Generator**: LLM that produces final answers.
