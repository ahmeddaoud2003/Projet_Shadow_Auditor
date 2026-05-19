import os
import re
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
DATA_DIR = Path("data")
DB_DIR = "./chroma_db"

# ✅ FIX 1 : Modèle multilingue (français/arabe) au lieu de all-MiniLM (anglais seulement)
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# Chunks plus grands pour capturer des articles complets avec leurs alinéas
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400


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

        # ✅ FIX 2 : Catégories explicites par fichier pour le filtrage multi-code
        category_map = {
            "CODE DU TRAVAIL": "Droit du Travail",
            "CODE SOCIETE": "Droit des Sociétés",
            "Code des Obligations et Contrats": "Droit des Obligations",
            "INPDP": "Protection des Données",
            "JORT": "Journal Officiel",
        }
        # Cherche une correspondance partielle dans le nom du fichier
        doc_category = next(
            (cat for key, cat in category_map.items() if key.lower() in doc_title.lower()),
            "Législation Tunisienne"
        )

        for page in pages:
            page.metadata["source"] = pdf_path.name
            page.metadata["doc_title"] = doc_title
            page.metadata["doc_category"] = doc_category
        documents.extend(pages)

    print(f"✅ {len(documents)} pages chargées depuis {len(pdf_paths)} PDF(s).")
    return documents


def split_documents(documents):
    # ✅ FIX 3 : Séparateurs conscients de la structure juridique tunisienne
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\nArticle",     # Séparateur principal : début d'article
            "\nart.",
            "\nArt.",
            "\nCHAPITRE",
            "\nSECTION",
            "\nTITRE",
            "\n\n",
            "\n",
            " ",
            ""
        ]
    )
    chunks = splitter.split_documents(documents)

    processed = []
    for chunk in chunks:
        title = chunk.metadata.get("doc_title", "Loi")
        category = chunk.metadata.get("doc_category", "")
        page = chunk.metadata.get("page", 0) + 1

        # ✅ FIX 4 : Préfixe enrichi — le modèle d'embedding sait d'où vient chaque chunk
        chunk.page_content = (
            f"[SOURCE: {title} | CATÉGORIE: {category} | PAGE: {page}]\n"
            f"{chunk.page_content}"
        )
        processed.append(chunk)

    print(f"✅ {len(processed)} chunks créés.")
    return processed


def create_vectorstore(chunks):
    # Supprime l'ancienne DB pour éviter les conflits de modèle d'embedding
    if os.path.exists(DB_DIR):
        import shutil
        shutil.rmtree(DB_DIR)
        print("🗑️  Ancienne ChromaDB supprimée.")

    print(f"🧠 Chargement du modèle d'embedding : {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
        # multilingual-e5 nécessite ce préfixe — on le passe via model_kwargs
        model_kwargs={"prompts": {"query": "query: ", "passage": "passage: "}}
    )

    print("💾 Création et sauvegarde dans ChromaDB (peut prendre quelques minutes)...")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)
    print(f"🎉 Base créée avec {vectorstore._collection.count()} vecteurs.")
    return vectorstore


def main():
    print("=" * 50)
    print("🚀 DÉMARRAGE DE L'INGESTION AMÉLIORÉE 🚀")
    print("=" * 50)
    documents = load_pdf_documents(DATA_DIR)
    chunks = split_documents(documents)
    create_vectorstore(chunks)
    print("=" * 50)
    print("✅ INGESTION TERMINÉE. Lancez maintenant : streamlit run app.py")
    print("=" * 50)


if __name__ == "__main__":
    main()