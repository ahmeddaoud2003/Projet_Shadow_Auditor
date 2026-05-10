import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Configuration des chemins et modèles
DATA_DIR = Path("data")
DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # Modèle vu en TP

# Pour les textes juridiques, on prend des morceaux assez grands pour garder le sens d'un article de loi
CHUNK_SIZE = 1000 
CHUNK_OVERLAP = 200

def load_and_process_pdfs():
    print("1. Recherche des documents juridiques...")
    pdf_paths = list(DATA_DIR.glob("*.pdf"))
    
    if not pdf_paths:
        print("Erreur : Aucun PDF trouvé dans le dossier 'data'.")
        return None

    documents = []
    
    # 2. Chargement des PDF avec Métadonnées (Crucial pour l'Auditeur)
    for pdf_path in pdf_paths:
        print(f"Chargement de : {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        
        # Ajout du nom du document dans les métadonnées pour que l'IA puisse citer ses sources
        for page in pages:
            page.metadata["source"] = pdf_path.name
            
        documents.extend(pages)
        
    print(f"Total de pages chargées : {len(documents)}")
    
    # 3. Découpage du texte (Chunking)
    print("2. Découpage des documents en chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""] # Séparateurs adaptés au texte juridique
    )
    chunks = splitter.split_documents(documents)
    print(f"Total de morceaux générés : {len(chunks)}")
    
    return chunks

def create_vector_db(chunks):
    print("3. Création des embeddings et de la base vectorielle ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Création et sauvegarde de la base sur le disque
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print(f"Terminé ! Base vectorielle sauvegardée dans : {DB_DIR}")
    return vectorstore

if __name__ == "__main__":
    print("=== DÉMARRAGE DE L'INGESTION THE SHADOW AUDITOR ===")
    chunks = load_and_process_pdfs()
    
    if chunks:
        create_vector_db(chunks)