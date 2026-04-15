# Agent-Driven-RAG-System-with-FastAPI1

Deployed FastAPI RAG system with document upload, FAISS-based retrieval, and LLM-powered question answering.

A production-style **Retrieval-Augmented Generation (RAG)** system built using **FastAPI**, enabling users to upload documents and perform intelligent question-answering using LLMs.

---

## Features

- 📄 Upload PDF documents dynamically
- 🔍 Semantic search using FAISS
- 🤖 Context-aware answer generation using LLM (Gemini API)
- ⚡ FastAPI-based REST API for real-time interaction
- 🧠 Hybrid system with fallback to direct LLM responses
- 📚 Source attribution for transparency

---

## 🧱 Architecture

User Query
↓
FastAPI (/ask, /upload)
↓
RAG Pipeline
↓
FAISS Vector Store (Embeddings)
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

#How It Works
1.User uploads a PDF document
2.Document is split into chunks
3.Embeddings are generated
4.Stored in FAISS vector database
5.User asks a question
6.Relevant chunks are retrieved
7.LLM generates context-aware answer

▶️ Run Locally

1. Clone the repository
   git clone https://github.com/your-username/Agent-Driven-RAG-System-with-FastAPI.git
   cd rag-api
2. Install dependencies
   pip install -r requirements.txt
3. Run FastAPI server
   uvicorn main:app --reload

#Future Improvements
1.Add authentication system
2.Improve retrieval using re-ranking
3.Add support for multiple documents
4.Implement advanced agent-based routing

#Author

Devansh Thakre

GitHub: https://github.com/DevanshThakre07
LinkedIn: https://www.linkedin.com/in/devansh-thakre/

⭐ If you like this project
Give it a ⭐ on GitHub!
