import os
from dotenv import load_dotenv
import datetime

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.prebuilt import create_react_agent

# caching
import langchain
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()

# load environment variables
load_dotenv()

# set google api key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# inisialisasi FastAPI app
app = FastAPI()

# konfigurasi path data dan vectordb
FAISS_INDEX_PATH = "faiss_index"

# get embeddings
def get_embeddings():
    """Memilih model embedding, dengan fallback ke HuggingFace."""
    if GOOGLE_API_KEY:
        try:
            print("🔑 Menggunakan Google Generative AI Embeddings...")
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
            embeddings.embed_query("test")
            return embeddings
        except Exception as e:
            print(f"Google embeddings gagal: {e}")
            print("Fallback ke HuggingFace Embeddings...")
    
    print("Menggunakan HuggingFace Embeddings...")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = get_embeddings()

# setting LLM model
if GOOGLE_API_KEY:
    print("🔑 Menggunakan Google Generative AI...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=GOOGLE_API_KEY, temperature=0)
else:
    raise ValueError("Google API Key tidak ditemukan. Silakan set GOOGLE_API_KEY di file .env")

# membuat tools dari Agent
print("Membuat tools dari FAISS vector store...")
if not os.path.exists(FAISS_INDEX_PATH):
    raise ValueError(f"FAISS index tidak ditemukan di {FAISS_INDEX_PATH}. Silakan jalankan conf_doc.py terlebih dahulu.")
vectordb = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

retriever_tool = create_retriever_tool(
    retriever,
    "pencarian_kebijakan_perusahaan",
    "Gunakan alat ini untuk mencari informasi spesifik tentang kebijakan HR, prosedur onboarding, aturan kerja, klaim medis, dan panduan IT perusahaan."
)

# buat tool custom ticket bantuan IT
@tool
def create_it_ticket(issue_description: str) -> str:
    """
    Gunakan alat ini untuk membuat tiket bantuan IT.
    Alat ini membutuhkan deskripsi masalah yang jelas sebagai input.
    """
    print(f"--- FUNGSI ALAT DIPANGGIL: Membuat tiket IT ---")
    print(f"--- Deskripsi Masalah: {issue_description} ---")
    # logika untuk membuat tiket IT (misalnya, menyimpan ke database atau mengirim email)
    ticket_id = f"IT-{hash(issue_description) % 1000}"  # contoh sederhana untuk ID tiket
    return f"Tiket bantuan IT telah dibuat dengan ID: {ticket_id}. Deskripsi masalah Anda: {issue_description}"

# tool aksi untuk penjadwalan wawancara
@tool
def schedule_interview(candidate_name: str, position: str, interview_date: str) -> str:
    """
    Gunakan alat ini untuk menjadwalkan wawancara kandidat.
    Membutuhkan nama kandidat, posisi yang dilamar, dan tanggal wawancara dalam format YYYY-MM-DD.
    """
    print(f"--- FUNGSI ALAT DIPANGGIL: Menjadwalkan Wawancara ---")
    print(f"--- Nama Kandidat: {candidate_name} ---")
    print(f"--- Posisi: {position} ---")
    print(f"--- Tanggal: {interview_date} ---")

    # logika untuk validasi tanggal dan menjadwalkan wawancara
    try:
        datetime.datetime.strptime(interview_date, "%Y-%m-%d")
        return f"Wawancara untuk {candidate_name} pada posisi {position} telah dijadwalkan pada {interview_date}."
    except ValueError:
        return "Format tanggal tidak valid. Harap gunakan format YYYY-MM-DD."

# gabungkan semua tools
tools = [retriever_tool, create_it_ticket, schedule_interview]

agent_executor = create_react_agent(llm, tools)

# memori chat history
class QueryRequest(BaseModel):
    question: str
    session_id: str

session_histories = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_histories:
        session_histories[session_id] = ChatMessageHistory()
    return session_histories[session_id]

conversational_agent = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="messages",
    history_messages_key="chat_history",
)

# endpoint untuk menerima query
@app.post("/ask")
async def ask_question(request: QueryRequest):
    session_id = request.session_id
    question = request.question
    config = {"configurable": {"session_id": session_id}}
    print(f"--- Pertanyaan diterima: {question} (Session: {session_id}) ---")

    try:
        history = get_session_history(session_id)
        messages = []

        if len(history.messages) == 0:
            messages.append(SystemMessage(
                content=(
                    "Anda adalah asisten AI yang memiliki dua kemampuan utama: "
                    "(1) Menjawab pertanyaan dengan mencari informasi, dan "
                    "(2) Melakukan aksi dengan menggunakan alat. "
                    "Analisis permintaan pengguna dengan cermat. "
                    "Selalu jawab dalam Bahasa Indonesia."
                )
            ))

        messages.append(HumanMessage(content=question))

        response = conversational_agent.invoke(
            {"messages": messages},
            config=config
        )
        answer = response["messages"][-1].content
        return {"answer": answer}
    except Exception as e:
        print(f"--- Error saat memproses pertanyaan: {e} (Session: {session_id}) ---")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan saat memproses pertanyaan Anda.")
    
# serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

# jalankan app dengan: uvicorn hr-agent-app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)







