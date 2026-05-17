import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma  # Remplacement de FAISS par Chroma
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
DATA_DIR = Path("data")
DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300
TOP_K = 12

def load_pdf_documents(data_dir: Path):
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            "Aucun PDF trouvé dans le dossier 'data'. "
            "Ajoutez vos documents (Code du Travail, COC, CSC, JORT, INPDP)."
        )
    documents = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        
        # --- NOUVEAUTÉ ICI : Extraction d'informations plus riches du nom de fichier ---
        # Exemple: "Code_du_Travail_Tunisien.pdf" -> "Code du Travail"
        doc_title = pdf_path.stem.replace("_", " ").strip() 
        # On peut aussi catégoriser
        if "travail" in doc_title.lower():
            doc_category = "Droit du Travail"
        elif "obligations" in doc_title.lower() or "contrat" in doc_title.lower():
            doc_category = "Droit des Obligations et Contrats"
        elif "societes" in doc_title.lower():
            doc_category = "Droit des Sociétés Commerciales"
        elif "jort" in doc_title.lower() or "officiel" in doc_title.lower():
            doc_category = "Journal Officiel (Jurisprudence/Textes Législatifs)"
        elif "inpdp" in doc_title.lower():
            doc_category = "Protection des Données Personnelles"
        else:
            doc_category = "Législation Générale"
        # --- FIN NOUVEAUTÉ ---

        for page in pages:
            page.metadata["source"] = pdf_path.name
            # --- AJOUT DES NOUVELLES METADONNEES ---
            page.metadata["doc_title"] = doc_title
            page.metadata["doc_category"] = doc_category
            # --- FIN AJOUT ---
        documents.extend(pages)
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    # --- NOUVEAUTÉ ICI : Ajout du titre du document au chunk ---
    # Pour que chaque chunk sache d'où il vient de manière plus explicite
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        chunk.page_content = f"Document: {chunk.metadata.get('doc_title', 'Inconnu')}\nCatégorie: {chunk.metadata.get('doc_category', 'Générale')}\nContenu: {chunk.page_content}"
    # --- FIN NOUVEAUTÉ ---

    return chunks

def create_vectorstore(chunks):
    """Crée la base de données vectorielle avec Chroma et la sauvegarde sur le disque."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)

def build_prompt():
    """Prompt hautement personnalisé pour 'The Shadow Auditor' et le droit tunisien."""
    template = """
Tu es "The Shadow Auditor", un expert juridique impitoyable, direct et extrêmement précis, spécialisé dans le droit tunisien.
Ta mission est d'analyser la question de l'utilisateur en utilisant UNIQUEMENT le contexte fourni ci-dessous.

Consignes strictes pour ton analyse :
1. Sois direct, confiant et affirmatif. Si la réponse se trouve dans le texte, donne-la clairement sans utiliser des termes hésitants comme "il semble", "cela suggère" ou "pourrait être".
2. Va droit au but. Donne la réponse exacte puis explique-la brièvement avec le texte de loi.
3. Référencie toujours la Catégorie et le Document (ex: "Selon l'article X du Code du Travail...").
4. Si l’information n’est absolument pas dans le contexte, dis clairement : "En tant qu'auditeur, je ne trouve aucune disposition à ce sujet." Ne sois jamais créatif.

Contexte juridique extrait :
{context}

Question de l'utilisateur :
{question}
"""
    return PromptTemplate.from_template(template)

def format_context(docs):
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "source_inconnue")
        page = doc.metadata.get("page", "?")
        doc_title = doc.metadata.get("doc_title", "Titre Inconnu") # Nouvelle métadonnée
        doc_category = doc.metadata.get("doc_category", "Catégorie Générale") # Nouvelle métadonnée

        if isinstance(page, int):
            page = page + 1
        
        # Le formatage pour le LLM est plus riche
        parts.append(
            f"--- Extrait Juridique {i} ---\n"
            f"Document : {doc_title} (Catégorie : {doc_category})\n"
            f"Source Fichier : {source} (Page {page})\n"
            f"Contenu :\n{doc.page_content}\n" # Le chunk contient déjà Document/Catégorie grâce à split_documents
            f"--------------------------\n"
        )
    return "\n\n".join(parts)

# Et assure-toi que format_sources utilise aussi les nouvelles métadonnées pour être plus descriptif
def format_sources(docs):
    unique_sources_details = {} # Utilise un dict pour gérer l'unicité et le détail
    for doc in docs:
        source = doc.metadata.get("source", "source_inconnue")
        page = doc.metadata.get("page", "?")
        doc_title = doc.metadata.get("doc_title", "Titre Inconnu")
        
        if isinstance(page, int):
            page = page + 1
        
        key = f"{doc_title} - {source}" # Clé unique par document/fichier
        
        if key not in unique_sources_details:
            unique_sources_details[key] = {"title": doc_title, "source_file": source, "pages": set()}
        unique_sources_details[key]["pages"].add(str(page)) # Ajouter la page comme string

    formatted_sources = []
    for key, details in unique_sources_details.items():
        pages_str = ", ".join(sorted(list(details["pages"])))
        formatted_sources.append(f"{details['title']} ({details['source_file']}, Pages: {pages_str})")
        
    return "\n".join(formatted_sources) # Retourne les sources sur plusieurs lignes

def answer_question(question, retriever, llm, prompt):
    """La fonction principale qui orchestre le RAG."""
    docs = retriever.invoke(question)
    context = format_context(docs)
    sources = format_sources(docs)

    final_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(final_prompt).content

    # Forcer l'ajout des sources si le LLM les a oubliées
    if "Sources :" not in response:
        response = response.strip() + f"\n\nSources utilisées :\n{sources}"

    return response, docs

def main():
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise EnvironmentError("La variable GROQ_API_KEY est absente.")

    # Etape 1: Traitement et Vectorisation
    print("Chargement des PDF juridiques tunisiens...")
    documents = load_pdf_documents(DATA_DIR)
    
    print("Découpage en chunks...")
    chunks = split_documents(documents)
    
    print("Création de l'index ChromaDB (Sauvegarde sur disque)...")
    vectorstore = create_vectorstore(chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=groq_api_key)
    prompt = build_prompt()

    print("\n✅ Base de données vectorisée ! Assistant RAG Terminal prêt.")
    print("Tapez votre question juridique ou 'quit' pour quitter.\n")

    # Etape 2: La boucle de Chat (CLI)
    while True:
        question = input("The Shadow Auditor > ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Fin de l'audit.")
            break

        answer, docs = answer_question(question, retriever, llm, prompt)

        print("\n--- Rapport d'Audit ---")
        print(answer)
        print("\n--- Chunks récupérés (Thought Process) ---")
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "source_inconnue")
            page = doc.metadata.get("page", "?")
            if isinstance(page, int): page += 1
            print(f"[{i}] {source} | Page {page} : {doc.page_content[:150].replace(chr(10), ' ')}...")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()