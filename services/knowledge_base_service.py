import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB")

CONNECTION_STRING = (
    f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

engine = create_engine(CONNECTION_STRING)

def get_embedded_sources():
    query = text("""
        SELECT DISTINCT
            e.cmetadata->>'source' AS source,
            e.cmetadata->>'title' as title,
            e.cmetadata->>'author' as author,
            (e.cmetadata->>'total_pages')::int as total_pages
        FROM langchain_pg_embedding e
        WHERE e.cmetadata->>'source' IS NOT NULL
        ORDER BY title;
    """)

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    # Convert hasil ke list of dict
    results = [
        {
            "source": row[0],
            "title": row[1],
            "author": row[2],
            "totalPages": row[3],
        }
        for row in rows
        if row[0] is not None
    ]

    return results
