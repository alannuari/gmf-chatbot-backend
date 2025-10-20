import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError
from psycopg.errors import UndefinedTable

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")

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

def docs2str(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    template = """
    You are an intelligent assistant.
    Answer the question based only on the following context:
    
    Context: {context}

    Question: {question}

    If the answer is not found in the context, say "I don't know based on the given context".
    Answer clearly and concisely.
    """
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(
        model="mistralai/mistral-7b-instruct:free",
        temperature=0.7
    )

    chain = (
        {"context": retriever | docs2str, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

def ask_question(question: str):
    chain = get_rag_chain()
    response = chain.invoke(question)
    return response
