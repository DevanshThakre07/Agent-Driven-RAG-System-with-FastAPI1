from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
# from google import genai 
import google.genai as genai
# change
from rank_bm25 import BM25Okapi
# change
from sentence_transformers import CrossEncoder

model_id = 'gemini-2.5-flash-lite'
# change
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
def get_client():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("GOOGLE_API_KEY not set")
    return genai.Client(api_key=key)

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def initialize_vector_store(file_paths):
    global vector_store

    documents = []

    for file in file_paths:
        loader = PyPDFLoader(file)
        docs = loader.load()

        for i, doc in enumerate(docs):
            doc.metadata["source"] = file
            doc.metadata["page"] = i + 1

        documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300
    )

    chunks = splitter.split_documents(documents)

    vector_store = FAISS.from_documents(chunks, get_embedding_model())

# Change 
    # 🔹 STORE CHUNKS
    global bm25, chunks_list
    chunks_list = chunks

    # 🔹 BM25 SETUP
    texts = [doc.page_content for doc in chunks]
    tokenized_corpus = [text.split() for text in texts]

    bm25 = BM25Okapi(tokenized_corpus)


vector_store = None
# change
bm25 = None
chunks_list = None


# FORMAT CONTEXT
def format_context(docs):
    context = ""
    sources = []

    for doc in docs:
        context += doc.page_content + "\n\n"
        sources.append(f"{doc.metadata['source']} (page: {doc.metadata['page']})")

    return context, list(set(sources))


# GENERATE ANSWER
def generate_answer(query, context):
    prompt = f"""
    You are an expert assistant.

    Answer the question using the context below.

    Even if the answer is partial, try to explain based on the context.

    DO NOT say "I don't know".

    Context:
    {context}

    Question:
    {query}
    """
    client = get_client()
    response = client.models.generate_content(
    model=model_id,
    contents=prompt)
    return response.text

# Change
def hybrid_search(query, k=5):
    global vector_store, bm25, chunks_list

    # 🔹 FAISS
    faiss_docs = vector_store.similarity_search(query, k=k)

    # 🔹 BM25
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]

    bm25_docs = [chunks_list[i] for i in top_indices]

    # 🔹 Combine
    combined = faiss_docs + bm25_docs

    # 🔹 Remove duplicates
    seen = set()
    unique_docs = []

    for doc in combined:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    return unique_docs[:k]

# change
def rerank(query, docs, top_k=3):
    pairs = [(query, doc.page_content) for doc in docs]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    top_docs = [doc for doc, score in ranked[:top_k]]

    return top_docs


# MAIN FUNCTION
def answer_query(query: str):
    global vector_store
    client = get_client()
    #  Step 1: If no document → LLM
    if vector_store is None:
        response = client.models.generate_content(
        model=model_id,
        contents=query)
        return {"answer": response.text, "sources": []}

    #  Step 2: Try retrieval
    # this commented line is the origional
    # docs = vector_store.similarity_search(query, k=5)

    # changed
    docs = hybrid_search(query, k=10)
    docs = rerank(query, docs, top_k=3)

    #  Step 3: If weak retrieval → fallback
    if len(docs) == 0:
        
        response = client.models.generate_content(
        model=model_id,
        contents=query)
        return {"answer": response.text, "sources": []}

    #  Step 4: Use RAG
    context, sources = format_context(docs)
    answer = generate_answer(query, context)

    return {"answer": answer, "sources": sources}