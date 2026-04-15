from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from google import genai 

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


key = os.getenv('GOOGLE_API_KEY')
client = genai.Client(api_key= key)
model_id = 'gemini-2.5-flash-lite'


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

    vector_store = FAISS.from_documents(chunks, embedding_model)



vector_store = None


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

    response = client.models.generate_content(
    model=model_id,
    contents=prompt)
    return response.text


# MAIN FUNCTION
def answer_query(query: str):
    global vector_store

    #  Step 1: If no document → LLM
    if vector_store is None:
        response = client.models.generate_content(
        model=model_id,
        contents=query)
        return {"answer": response.text, "sources": []}

    #  Step 2: Try retrieval
    docs = vector_store.similarity_search(query, k=5)

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