import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel
from services.knowledge_base_service import get_embedded_sources
from services.rag_service import ask_question
from services.ingest_service import process_file, process_from_url
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

app = FastAPI(title="RAG API using LangChain + FastAPI")

origins = [
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # daftar origin yang diizinkan
    allow_credentials=True,
    allow_methods=["*"],          # semua metode HTTP (GET, POST, dll)
    allow_headers=["*"],          # semua header
)

# Mount folder docs agar bisa diakses lewat URL /docs/...
app.mount("/docs", StaticFiles(directory="docs"), name="docs")

UPLOAD_DIR = "./docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QuestionRequest(BaseModel):
    question: str

class URLRequest(BaseModel):
    url: str

@app.get("/")
def root():
    return {"message": "RAG API is running 🚀"}

@app.get("/embedded-sources")
def list_embedded_sources():
    """Get all unique embedded document sources and titles."""
    try:
        sources = get_embedded_sources()
        return {"count": len(sources), "data": sources}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask")
def ask(req: QuestionRequest):
    try:
        return ask_question(req.question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/ingest/file",
    summary="Unggah file PDF atau DOCX untuk diingest"
)
async def ingest_file(file: UploadFile = File(...)):
    """Upload and process a document (PDF/DOCX)."""
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        chunks = process_file(file_path)
        return {"message": f"File berhasil diproses, total chunk: {chunks}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/ingest/url",
    summary="Masukkan URL dokumen atau halaman web untuk diingest"
)
async def ingest_url(request: URLRequest):
    url = request.url
    if not url:
        raise HTTPException(status_code=400, detail="URL tidak boleh kosong")
    chunks, src = process_from_url(url)
    return {"message": f"Dokumen dari URL {src} berhasil diproses, total chunk: {chunks}"}