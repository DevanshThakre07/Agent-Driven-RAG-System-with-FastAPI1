# Agent-Driven RAG System with Hybrid Search & Reranking

A production-style **Retrieval-Augmented Generation (RAG)** system built using **FastAPI**, enhanced with **Hybrid Search (FAISS + BM25)** and **Cross-Encoder Reranking** to deliver highly relevant and accurate answers from documents.

A production-style **Retrieval-Augmented Generation (RAG)** system built using **FastAPI**, enabling users to upload documents and perform intelligent question-answering using LLMs.

---

## Features

- 📄 Upload PDF documents dynamically
- 🔍 Hybrid Search (FAISS + BM25)
- 🧠 Cross-Encoder Reranking for improved relevance
- 🤖 Context-aware answer generation using LLM (Gemini API)
- ⚡ FastAPI-based REST API for real-time interaction
- 🔁 Intelligent fallback to direct LLM responses
- 📚 Source attribution for transparency

---

## 🧱 Architecture

User Query  
↓  
FastAPI (/ask, /upload)  
↓  
Hybrid Retrieval (FAISS + BM25)  
↓  
Top-K Candidate Chunks  
↓  
Cross-Encoder Reranking  
↓  
Filtered Context  
↓  
LLM (Gemini API)  
↓  
Final Answer + Sources

---

## ⚙️ Tech Stack

- **Backend:** FastAPI
- **LLM:** Gemini API
- **Embeddings:** Hugging Face Transformers
- **Vector DB:** FAISS
- **Language:** Python

---

## 📂 Project Structure

Agent-Driven-RAG-System-with-FastAPI/
│── main.py # FastAPI application
│── rag_pipeline.py # RAG logic (retrieval + generation)
│── requirements.txt # Dependencies
│── uploads/ # Uploaded documents

---

## 🚀 API Endpoints

### 🔹 1. Upload Document

**POST** `/upload`

Upload a PDF file for processing.

---

### 🔹 2. Ask Question

**POST** `/ask`

#### Request:

```json
{
  "question": "What is this document about?"
}
Response:

{
  "answer": "...",
  "sources": ["file.pdf (page 2)"]
}
```

## How It Works

1. User uploads a PDF document
2. Document is split into chunks
3. Embeddings are generated using MiniLM
4. FAISS performs semantic search
5. BM25 performs keyword-based retrieval
6. Results are combined (Hybrid Search)
7. Cross-encoder reranks the results
8. Top chunks are passed to LLM
9. LLM generates context-aware answer

▶️ Run Locally

1. Clone the repository
   git clone https://github.com/your-username/Agent-Driven-RAG-System-with-FastAPI.git
   cd rag-api
2. Install dependencies
   pip install -r requirements.txt
3. Run FastAPI server
   uvicorn main:app --reload

#Future Improvements

- [x] Hybrid search (FAISS + BM25)
- [x] Cross-encoder reranking
- [ ] FAISS persistence
- [ ] LangGraph-based agent workflow
- [ ] Docker deployment

## 🌐 Live Demo

🔗 API: https://agent-driven-rag-system-with-fastapi1.onrender.com/docs

#Author

Devansh Thakre

GitHub: https://github.com/DevanshThakre07
LinkedIn: https://www.linkedin.com/in/devansh-thakre/

⭐ If you like this project
Give it a ⭐ on GitHub!
