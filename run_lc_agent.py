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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Import komponen inti untuk membangun agen LangChain klasik
from langchain.agents import create_tool_calling_agent, AgentExecutor

import langchain
from langchain.cache import InMemoryCache

langchain.llm_cache = InMemoryCache()

# setup environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
app = FastAPI()
FAISS_INDEX_PATH = "faiss_index"

# initialize model and embeddings
def get_embeddings():
    if GOOGLE_API_KEY:
        try:
            print("🔑 Menggunakan Google Generative AI Embeddings...")
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
            embeddings.embed_query("test")
            return embeddings
        except Exception as e:
            print(f"⚠️ Google embeddings gagal: {e}")
    print("✅ Menggunakan HuggingFace Embeddings...")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = get_embeddings()

if not GOOGLE_API_KEY:
    raise ValueError("Google API Key tidak ditemukan.")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=GOOGLE_API_KEY, temperature=0)
print("✅ LLM Google Generative AI berhasil dimuat.")

# membuat tools
print("Memuat FAISS index dan membuat tools...")
if not os.path.exists(FAISS_INDEX_PATH):
    raise ValueError(f"FAISS index tidak ditemukan di {FAISS_INDEX_PATH}.")
vectordb = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

retriever_tool = create_retriever_tool(retriever, "pencarian_informasi_kebijakan_perusahaan", "Gunakan alat ini HANYA untuk MENJAWAB PERTANYAAN tentang kebijakan HR, prosedur onboarding, aturan kerja, klaim medis, dan panduan IT. Jangan gunakan untuk melakukan aksi seperti membuat tiket atau menjadwalkan sesuatu.")

@tool
def create_it_ticket(issue_description: str) -> str:
    """Gunakan alat ini UNTUK MELAKUKAN AKSI membuat tiket bantuan IT jika pengguna melaporkan masalah teknis. Butuh deskripsi masalah yang jelas dari pengguna."""
    ticket_id = f"IT-{hash(issue_description) % 1000}"
    return f"Baik, saya telah membuatkan tiket bantuan IT dengan ID: {ticket_id} untuk masalah: '{issue_description}'. Tim IT akan segera menghubungi Anda."

@tool
def schedule_interview(candidate_name: str, position: str, interview_date: str) -> str:
    """Gunakan alat ini UNTUK MELAKUKAN AKSI menjadwalkan wawancara kandidat baru. Butuh nama kandidat, posisi yang dilamar, dan tanggal wawancara (format YYYY-MM-DD)."""
    try:
        datetime.datetime.strptime(interview_date, "%Y-%m-%d")
        return f"Wawancara untuk {candidate_name} ({position}) telah berhasil dijadwalkan pada {interview_date}."
    except ValueError:
        return "Format tanggal tidak valid. Harap gunakan format YYYY-MM-DD."

tools = [retriever_tool, create_it_ticket, schedule_interview]
print("✅ Tools berhasil dibuat.")

# membuat prompt agent langsung
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Anda adalah asisten AI yang ramah dan kompeten. Analisis permintaan pengguna dan pilih alat yang paling sesuai untuk menyelesaikannya."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Buat "otak" agen
agent = create_tool_calling_agent(llm, tools, prompt)

# Buat "pelaksana" agen yang akan menjalankan alat
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
print("✅ LangChain Agent Executor berhasil dibuat.")

# manage memory history
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
    input_messages_key="input",
    history_messages_key="chat_history",
)
print("✅ Agen dengan memori percakapan siap.")

# endpoint untuk menerima query
@app.post("/ask")
async def ask_question(request: QueryRequest):
    session_id = request.session_id
    question = request.question
    
    history = get_session_history(session_id)
    is_first_turn = len(history.messages) == 0

    final_input = question
    
    if is_first_turn:
        print("--- Ini adalah giliran pertama, menambahkan instruksi sistem. ---")
        system_instruction = (
            "Anda adalah asisten AI yang memiliki dua kemampuan utama: (1) Menjawab pertanyaan dengan mencari informasi, dan (2) Melakukan aksi dengan menggunakan alat. "
            "Tugas Anda adalah menganalisis permintaan pengguna dengan cermat. "
            "Jika pengguna bertanya tentang informasi (misal: 'apa', 'bagaimana', 'berapa'), gunakan alat 'pencarian_informasi_kebijakan_perusahaan'. "
            "Jika pengguna meminta untuk melakukan sesuatu atau melaporkan masalah (misal: 'tolong buatkan', 'jadwalkan', 'laptop saya rusak'), pilih dan gunakan alat aksi yang sesuai seperti 'create_it_ticket' atau 'schedule_interview'. "
            "Selalu jawab dalam Bahasa Indonesia."
        )
        final_input = f"{system_instruction}\n\nPertanyaan saya: {question}"

    print(f"--- Pertanyaan diterima: {question} (Session: {session_id}) ---")

    try:
        config = {"configurable": {"session_id": session_id}}
        response = conversational_agent.invoke(
            {"input": final_input},
            config=config
        )
        answer = response["output"]

        print(f"--- Jawaban: {answer} ---")
        return {"answer": answer}
    except Exception as e:
        print(f"Error saat memproses pertanyaan: {e}")
        raise HTTPException(
            status_code=500,
            detail="Terjadi kesalahan saat memproses pertanyaan Anda."
        )

# Sajikan file statis (HTML)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

# Jalankan aplikasi
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

