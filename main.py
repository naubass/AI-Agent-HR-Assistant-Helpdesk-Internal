import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# inisialisasi FastAPI
app = FastAPI()

# vectordb
persist_directory = "./chroma_db"

# get embeddings
def get_embeddings():
    """Coba Google embeddings, fallback ke HuggingFace."""
    if GOOGLE_API_KEY:
        try:
            print("🔑 Using Google Generative AI Embeddings...")
            # Lakukan tes kecil untuk memicu error jika ada masalah kuota/koneksi
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY
            )
            embeddings.embed_query("test") 
            return embeddings
        except Exception as e:
            print(f"⚠️ Google embeddings gagal: {e}")
            print("👉 Fallback ke HuggingFace Embeddings...")

    print("✅ Using HuggingFace Embeddings...")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# embeddings & vectordb
embeddings = get_embeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# LLM
if GOOGLE_API_KEY:
    llm = GoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=GOOGLE_API_KEY)
else:
    raise ValueError("GOOGLE_API_KEY tidak ditemukan! Harap set API key di .env")

# prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        """
        Anda adalah asisten AI untuk menjawab pertanyaan tentang dokumen kebijakan perusahaan.
        Gunakan potongan konteks berikut untuk menjawab pertanyaan.
        Jika Anda tidak tahu jawabannya, katakan saja Anda tidak tahu, jangan mencoba mengarang jawaban.
        Jawablah dalam Bahasa Indonesia.

        Konteks: {context}

        Pertanyaan: {question}

        Jawaban:
        """
    )
)

# helper format docs
def format_docs(docs):
    safe_texts = []
    for d in docs:
        if hasattr(d, "page_content") and d.page_content:
            safe_texts.append(str(d.page_content))
    return "\n\n".join(safe_texts) if safe_texts else "Tidak ada konteks relevan ditemukan."

# RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# model request dan response
class QueryRequest(BaseModel):
    question: str

# route untuk tanya
@app.post("/ask")
async def ask_question(request: QueryRequest):
    question = request.question
    print(f"Pertanyaan diterima: {question}")

    try:
        # --- PERBAIKAN DI SINI ---
        # Kirim hanya string pertanyaan ke dalam chain
        answer = rag_chain.invoke(question)
        # -------------------------
        
        print(f"Jawaban: {answer}")
        return {"answer": answer}
    except Exception as e:
        print(f"❌ Error saat memproses pertanyaan: {e}")
        return {"error": "Terjadi kesalahan saat memproses pertanyaan."}

# static files (Asumsi folder frontend bernama 'static')
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_homepage():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    # Pastikan host adalah "0.0.0.0" agar bisa diakses dari luar container jika pakai Docker
    uvicorn.run(app, host="localhost", port=8000)
