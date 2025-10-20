# 🧠 FastAPI + LangChain RAG Project

Proyek ini adalah implementasi **Retrieval-Augmented Generation (RAG)** menggunakan **FastAPI** sebagai backend framework dan **LangChain** sebagai framework untuk integrasi model bahasa dan penyimpanan vektor.

---

## 🚀 Fitur
- **Ingest dokumen** ke dalam vector store.
- **RAG service** untuk menjawab pertanyaan berbasis dokumen.

---

## ⚙️ Instalasi & Menjalankan

### 1️⃣ Clone repository
```bash
git clone https://github.com/alannuari/gmf-chatbot-backend.git
cd gmf-chatbot-backend
```

### 2️⃣ Buat virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Untuk Linux/Mac
venv\Scripts\activate     # Untuk Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Buat file .env
Copy .env.example ke .env

### 4️⃣ Menjalankan Server FastAPI
```bash
uvicorn main:app --reload
```

Server akan berjalan di:
```bash
http://127.0.0.1:8000
```

## 📘 Dokumentasi API (Swagger & ReDoc)
| Jenis          | URL                                                        | Deskripsi                                            |
| -------------- | ---------------------------------------------------------- | ---------------------------------------------------- |
| **Swagger UI** | [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)   | Antarmuka interaktif untuk mencoba endpoint langsung |
| **ReDoc**      | [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc) | Dokumentasi API dengan tampilan profesional          |
