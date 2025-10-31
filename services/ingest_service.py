import os
import requests
import tempfile
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

load_dotenv()

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

CONNECTION_STRING = (
    f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# Buat koneksi SQLAlchemy
engine = create_engine(CONNECTION_STRING)


def _process_documents(documents):
    """Gunakan untuk semua dokumen (file maupun web)."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = PGVector.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection=engine
    )

    return len(splits)

def process_file(file_path: str, source_url: str = None):
    """Load, split, embed, and store a document."""
    if file_path.endswith('.pdf'):
        loader = PyMuPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type")

    documents = loader.load()

    if source_url:
        # Tambahkan metadata agar tahu asalnya dari upload
        for doc in documents:
            doc.metadata["source"] = source_url

    return _process_documents(documents)


def process_from_url(url: str):
    """Download dan proses dokumen dari URL."""
    response = requests.get(url, timeout=15)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")

    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        suffix = ".pdf"
    elif "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in content_type or url.lower().endswith(".docx"):
        suffix = ".docx"
    elif "text/html" in content_type:
        # === CASE: web page ===
        loader = WebBaseLoader(url)
        documents = loader.load()

        return _process_documents(documents), url
    else:
        raise ValueError(f"Tipe konten tidak didukung: {content_type}")

    # === CASE: PDF atau DOCX dari URL ===
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name

    chunks = process_file(tmp_path, url)

    os.remove(tmp_path)
    return chunks, url