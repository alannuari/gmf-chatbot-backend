import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from services.rag_service import ask_question
from services.ingest_service import process_file

app = FastAPI(title="RAG API using LangChain + FastAPI")

UPLOAD_DIR = "./docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "RAG API is running 🚀"}

@app.post("/ask")
def ask(req: QuestionRequest):
    try:
        return {"answer": ask_question(req.question)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Upload and process a document (PDF/DOCX)."""
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        chunks = process_file(file_path)
        return {
            "filename": file.filename,
            "status": "success",
            "chunks_stored": chunks,
            "message": f"{chunks} chunks embedded to PostgreSQL"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
