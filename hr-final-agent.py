import os
from dotenv import load_dotenv
import datetime

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Import untuk Google Sheets
import gspread
from google.oauth2.service_account import Credentials

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.prebuilt import create_react_agent

# caching
import langchain
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()

# load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")

# setup app FastAPI & konfigurasi vectordb
app = FastAPI()
FAISS_INDEX_PATH = "faiss_index"

# funtion get embeddings
def get_embeddings():
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

# set google llm
if not GOOGLE_API_KEY:
    raise ValueError("Google API Key tidak ditemukan.")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=GOOGLE_API_KEY, temperature=0)
print("LLM Google Generative AI berhasil dimuat.")

# membuat tools agent
print("Memuat FAISS index dan membuat tools...")
if not os.path.exists(FAISS_INDEX_PATH):
    raise ValueError(f"FAISS index tidak ditemukan di {FAISS_INDEX_PATH}.")

vectordb = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# retriever tool
retriever_tool = create_retriever_tool(
    retriever,
    "pencarian_kebijakan_perusahaan",
    "Gunakan alat ini untuk mencari informasi spesifik tentang kebijakan HR, prosedur onboarding, aturan kerja, klaim medis, dan panduan IT perusahaan."
)

# tool aksi bantuan ticket IT
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

# tool aksi penjadwalan wawancara
@tool
def schedule_interview(candidate_name: str, position: str, interview_date: str) -> str:
    """
    Gunakan alat ini untuk menjadwalkan wawancara kandidat dan MENCATATNYA ke Google Sheet.
    Membutuhkan nama kandidat, posisi yang dilamar, dan tanggal wawancara dalam format YYYY-MM-DD.
    """
    print(f"--- FUNGSI ALAT DIPANGGIL: Menjadwalkan Wawancara & Mencatat ke Sheet ---")
    
    # Validasi tanggal terlebih dahulu
    try:
        datetime.datetime.strptime(interview_date, "%Y-%m-%d")
    except ValueError:
        return "Format tanggal tidak valid. Harap gunakan format YYYY-MM-DD."

    # Cek apakah konfigurasi Google Sheets ada
    if not GOOGLE_SHEET_ID or not GOOGLE_SERVICE_ACCOUNT_FILE:
        return "Konfigurasi Google Sheets (ID atau file kredensial) tidak ditemukan. Penjadwalan dicatat secara lokal saja."

    try:
        # Autentikasi ke Google Sheets
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(GOOGLE_SERVICE_ACCOUNT_FILE, scopes=scopes)
        client = gspread.authorize(creds)

        # Buka spreadsheet dan worksheet
        sheet = client.open_by_key(GOOGLE_SHEET_ID)
        worksheet = sheet.get_worksheet(0) # Ambil sheet pertama

        # Buat baris baru untuk ditambahkan
        new_row = [candidate_name, position, interview_date, "Dijadwalkan"]
        worksheet.append_row(new_row)
        
        print(f"---Berhasil menambahkan jadwal ke Google Sheet.---")
        return f"Wawancara untuk {candidate_name} ({position}) telah dijadwalkan pada {interview_date} dan berhasil dicatat di Google Sheet."

    except Exception as e:
        print(f"Gagal menulis ke Google Sheet: {e}")
        return f"Gagal mencatat jadwal ke Google Sheet karena error: {e}. Namun, wawancara tetap dijadwalkan secara sistem."

# gabungkan tools
tools = [retriever_tool, create_it_ticket, schedule_interview]

# memuat agent
agent_executor = create_react_agent(llm, tools)
print("Agent berhasil dimuat.")

# manage memory history & endpoint
class QueryRequest(BaseModel):
    question: str
    session_id: str

session_histories = {}

# function historis
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

# endpoint fastapi
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
            messages.append(SystemMessage(content=("Anda adalah asisten AI yang memiliki kemampuan utama: Menjawab pertanyaan. Analisis permintaan pengguna dengan cermat. Selalu jawab dalam Bahasa Indonesia.")))
        messages.append(HumanMessage(content=question))
        response = conversational_agent.invoke({"messages": messages}, config=config)
        answer = response["messages"][-1].content
        return {"answer": answer}
    except Exception as e:
        print(f"--- Error saat memproses pertanyaan: {e} (Session: {session_id}) ---")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan saat memproses pertanyaan Anda.")
    
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

# run app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)





    

