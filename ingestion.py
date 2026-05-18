import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURATION LOCALE (DEEPSEEK) ---
DATA_DIR = Path("data")
DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def load_pdf_documents(data_dir: Path):
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError("❌ Aucun PDF trouvé dans le dossier 'data'.")
    
    documents = []
    for pdf_path in pdf_paths:
        print(f"📄 Lecture de : {pdf_path.name}...")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        
        doc_title = pdf_path.stem.replace("_", " ").strip() 
        doc_category = "Législation Tunisienne"
        
        for page in pages:
            page.metadata["source"] = pdf_path.name
            page.metadata["doc_title"] = doc_title
            page.metadata["doc_category"] = doc_category
        documents.extend(pages)
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        chunk.page_content = f"Loi: {chunk.metadata.get('doc_title', '')}\n{chunk.page_content}"
    return chunks

def create_vectorstore(chunks):
    print("🧠 Chargement des Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print("💾 Sauvegarde dans ChromaDB...")
    return Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)

def main():
    print("🚀 DÉMARRAGE DE L'INGESTION 🚀")
    documents = load_pdf_documents(DATA_DIR)
    chunks = split_documents(documents)
    create_vectorstore(chunks)
    print("🎉 INGESTION TERMINÉE ! La base est prête pour DeepSeek. 🎉")

if __name__ == "__main__":
    main()