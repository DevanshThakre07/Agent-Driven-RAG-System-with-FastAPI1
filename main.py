from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import UploadFile, File
import os

app = FastAPI()


class QueryRequest(BaseModel):
    question: str

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def home():
    return {"message": "RAG API running"}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        from rag_pipeline import initialize_vector_store
        
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        initialize_vector_store([file_path])

        return {"message": f"{file.filename} uploaded and processed"}

    except Exception as e:
        return {"error": str(e)}

@app.post("/ask")
def ask(request: QueryRequest):
    try:
        from rag_pipeline import answer_query

        result = answer_query(request.question)

        return {
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except Exception as e:
        return {"error": str(e)}