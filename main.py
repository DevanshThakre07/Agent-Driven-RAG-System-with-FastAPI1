from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "working"}
# from fastapi import FastAPI
# from models import QueryRequest
# from rag_pipeline import answer_query, initialize_vector_store
# from fastapi import UploadFile, File
# import os
# app = FastAPI()



# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# @app.get("/")
# def home():
#     return {"message": "RAG API running"}

# @app.post("/upload")
# async def upload(file: UploadFile = File(...)):
#     file_path = os.path.join(UPLOAD_DIR, file.filename)

#     # Save file
#     with open(file_path, "wb") as f:
#         content = await file.read()
#         f.write(content)

#     # Initialize vector store with this file
#     initialize_vector_store([file_path])

#     return {"message": f"{file.filename} uploaded and processed"}

# @app.post("/ask")
# def ask(request: QueryRequest):
#     result = answer_query(request.question)

#     return {
#         "question": request.question,
#         "answer": result["answer"],
#         "sources": result["sources"]
#     }