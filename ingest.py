import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  

# load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

DATA_PATH = "data/"
pdf_loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
txt_loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader, show_progress=True)

documents = pdf_loader.load() + txt_loader.load()
print(f"Loaded {len(documents)} documents from {DATA_PATH} berhasil di load")

def ingest_data():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text")

    # coba Google embeddings
    if GOOGLE_API_KEY:
        try:
            print("🔑 Using Google Generative AI Embeddings...")
            google_embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY
            )
            vectordb = Chroma.from_documents(texts, embedding=google_embeddings, persist_directory="./chroma_db")
            print("✅ Ingest selesai dengan Google Embeddings.")
            return
        except Exception as e:
            print(f"⚠️ Google embeddings gagal: {e}")
            print("👉 Fallback ke HuggingFace Embeddings...")

    # fallback HuggingFace
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(texts, embedding=hf_embeddings, persist_directory="./chroma_db")
    print("✅ Ingest selesai dengan HuggingFace Embeddings.")

if __name__ == "__main__":
    ingest_data()
