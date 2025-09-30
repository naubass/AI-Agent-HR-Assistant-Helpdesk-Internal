import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# konfigurasi path data dan vectordb
DATA_PATH = "docs/"
FAISS_INDEX_PATH = "faiss_index"

# fungsi untuk mendapatkan embeddings
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

def ingest_data():
    """Fungsi untuk ingest data dari folder docs/ ke FAISS vector store."""
    # load dokumen dari folder
    pdf_loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    txt_loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
    documents = pdf_loader.load() + txt_loader.load()
    print(f"Loaded {len(documents)} documents from {DATA_PATH} berhasil di load")

    # split dokumen menjadi chunk
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text")

    # dapatkan embeddings
    embeddings = get_embeddings()

    # hapus index lama jika ada
    if os.path.exists(FAISS_INDEX_PATH):
        shutil.rmtree(FAISS_INDEX_PATH)

    # buat FAISS vector store
    vectordb = FAISS.from_documents(texts, embedding=embeddings)
    vectordb.save_local(FAISS_INDEX_PATH)
    print(f"Ingest selesai. FAISS index disimpan di {FAISS_INDEX_PATH}")

if __name__ == "__main__":
    ingest_data()