import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Muat environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Inisialisasi FastAPI
app = FastAPI()

# konfigurasi vectordb
persist_directory = "./chroma_db"

def get_embeddings():
    if GOOGLE_API_KEY:
        try:
            print("🔑 Menggunakan Google Generative AI Embeddings...")
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
            embeddings.embed_query("test")
            return embeddings
        except Exception as e:
            print(f"⚠️ Google embeddings gagal: {e}")
            print("👉 Fallback ke HuggingFace Embeddings...")
    print("✅ Menggunakan HuggingFace Embeddings...")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = get_embeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

if GOOGLE_API_KEY:
    llm = GoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=GOOGLE_API_KEY)
else:
    raise ValueError("GOOGLE_API_KEY tidak ditemukan! Harap set API key di .env")

# Prompt untuk mengubah pertanyaan (Question Rephrasing)
contextualize_q_system_prompt = """
Mengingat riwayat percakapan dan pertanyaan tindak lanjut dari pengguna, ubah pertanyaan tindak lanjut tersebut menjadi pertanyaan yang berdiri sendiri (standalone question).
"""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)
# Chain ini akan membuat pertanyaan baru yang lebih baik untuk retrieval
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Prompt untuk menjawab pertanyaan setelah mendapatkan konteks
qa_system_prompt = """
Anda adalah asisten AI yang ramah dan membantu untuk menjawab pertanyaan tentang dokumen kebijakan perusahaan.
Gunakan potongan konteks yang relevan untuk menjawab pertanyaan. JANGAN gunakan riwayat percakapan untuk menjawab.
Jika Anda tidak tahu jawabannya berdasarkan konteks yang diberikan, katakan saja Anda tidak tahu. Jangan mengarang jawaban.
Jawablah dalam Bahasa Indonesia.

Konteks:
{context}
"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)
# Chain ini akan mengambil konteks dan pertanyaan, lalu menghasilkan jawaban
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Gabungkan semuanya menjadi Retrieval Chain utama
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# riwayat sesi untuk setiap pengguna
session_histories = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_histories:
        session_histories[session_id] = ChatMessageHistory()
    return session_histories[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer", # Menentukan kunci output
)

# model request dan response
class QueryRequest(BaseModel):
    question: str
    session_id: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    session_id = request.session_id
    question = request.question
    print(f"Pertanyaan diterima (Session: {session_id}): {question}")

    try:
        # Kita memanggil chain dengan input dictionary yang sesuai
        config = {"configurable": {"session_id": session_id}}
        # Chain baru menghasilkan dictionary, kita ambil 'answer'-nya
        response_dict = conversational_rag_chain.invoke({"input": question}, config=config)
        
        answer = response_dict.get("answer", "Maaf, terjadi kesalahan dalam menghasilkan jawaban.")
        
        print(f"Jawaban: {answer}")
        return {"answer": answer}
    except Exception as e:
        print(f"❌ Error saat memproses pertanyaan: {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan saat memproses pertanyaan.")

# sajikan static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_homepage():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

