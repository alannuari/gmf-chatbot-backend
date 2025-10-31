import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from urllib.parse import quote
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError
from psycopg.errors import UndefinedTable

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")

BASE_URL = os.getenv("BASE_URL")
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
collection_name = os.getenv("COLLECTION_NAME")

# Buat koneksi SQLAlchemy
engine = create_engine(CONNECTION_STRING)

error_messages = [
    "Apologies, I don’t have that specific information in my Knowledge Base right now.",
    "It seems that information hasn’t been added to the Knowledge Base yet. Please check back soon.",
    "I couldn’t find any related information in the Knowledge Base. Please try rephrasing your question or asking it in another way."
]

# --- Helper: Ubah dokumen jadi string + metadata ---
def docs2str(docs):
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
        for doc in docs
    )

# --- Buat RAG chain dan retriever ---
def get_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Cek apakah tabel dan koleksi sudah ada
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT DISTINCT c.name 
                FROM langchain_pg_collection c
                JOIN langchain_pg_embedding e ON e.collection_id = c.uuid
            """))
            collections = [row[0] for row in result.fetchall()]

        if collection_name not in collections:
            raise ValueError(f"Collection '{collection_name}' belum dibuat. Jalankan proses embedding dulu.")

    except (UndefinedTable, ProgrammingError):
        # Tangkap jika tabel embedding belum ada
        raise ValueError("Belum ada tabel embedding. Jalankan proses embedding dulu untuk membuatnya.")

    # Load vectorstore hanya jika tabel & koleksi sudah ada
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=engine,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = """
        You are a precise and literal assistant.
        You MUST answer ONLY using information from the provided context.
    
        Instructions:
        - First, detect the main language used in the context.
        - ALWAYS respond in that same language, even if the question is in a different language.
        - If the context is in English, your answer MUST be entirely in English.
        - If the answer is not found in the context, respond with ONE of the following messages (choose one naturally):
            1. "Apologies, I don’t have that specific information in my Knowledge Base right now."
            2. "It seems that information hasn’t been added to the Knowledge Base yet. Please check back soon."
            3. "I couldn’t find any related information in the Knowledge Base. Please try rephrasing your question or asking it in another way."
        - Do NOT use external knowledge or translate the answer to another language.

        Context:
        {context}

        Question:
        {question}

        Answer (use the same language as the context):
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(
        model="mistralai/mistral-7b-instruct:free",
        temperature=0.7
    )

    chain = prompt | llm | StrOutputParser()

    return retriever, chain

def format_sources(docs):
    sources_map = {}

    for doc in docs:
        path = doc.metadata.get("source")
        page = doc.metadata.get("page")

        if not path:
            continue

        filename = os.path.basename(path.rstrip("/"))
        encoded_filename = quote(filename, safe="")

        # Buat URL file
        if path.startswith("http://") or path.startswith("https://"):
            base_url = path
        else:
            base_url = f"{BASE_URL}/docs/{encoded_filename}"

        # Jika file sudah pernah masuk, tambahkan halamannya
        if path in sources_map:
            if page and page not in sources_map[path]["pages"]:
                sources_map[path]["pages"].append(page)
        else:
            sources_map[path] = {
                "name": filename,
                "source": base_url,
                "pages": [page] if page else []
            }

    # Ubah ke list untuk output akhir
    unique_sources = list(sources_map.values())

    # Sort daftar halaman untuk rapi
    for src in unique_sources:
        src["pages"] = sorted([p for p in src["pages"] if p is not None])

    return unique_sources

# --- Fungsi utama untuk QA ---
def ask_question(question: str):
    retriever, chain = get_rag_chain()

    # Ambil dokumen terkait
    docs = retriever.invoke(question)

    # Gabungkan jadi konteks
    context = docs2str(docs)

    # Jalankan chain
    response = chain.invoke({"context": context, "question": question})

    # Format sumber jadi URL yang bisa diklik
    sources = format_sources(docs)

    # Pastikan ada jawaban
    if not response.strip():
        response = random.choice(error_messages)

    # Jika LLM tidak tahu, kosongkan sources
    if any(msg in response for msg in error_messages):
        sources = []

    return {
        "answer": response,
        "sources": sources
    }
