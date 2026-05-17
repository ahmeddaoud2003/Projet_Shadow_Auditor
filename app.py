import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import  MultiQueryRetriever

# IMPORT DE TES FONCTIONS DEPUIS ingestion.py
from ingestion import build_prompt, format_context, format_sources, DB_DIR, EMBEDDING_MODEL, GROQ_MODEL, TOP_K

# --- CONFIGURATION STREAMLIT ---
st.set_page_config(page_title="The Shadow Auditor", page_icon="⚖️", layout="wide")

# --- INITIALISATION DU MOTEUR RAG ---
@st.cache_resource(show_spinner=False)
def init_rag_system():
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Clé API Groq manquante dans le fichier .env")
        st.stop()

    if not os.path.exists(DB_DIR):
        st.error("⚠️ Base ChromaDB introuvable. Exécutez d'abord `python ingestion.py` dans le terminal.")
        st.stop()

    # Connexion à la base de données
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    # 1. On augmente le nombre de résultats (Top 10 au lieu de 5)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # Initialisation du LLM
    llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=groq_api_key)

    # 2. LA MAGIE DE L'AGENTIC RAG : Le Multi-Query Retriever
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )
    
    return retriever, llm
# --- INTERFACE UTILISATEUR ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA;}
    .css-1d391kg { background-color: #161B22; }
    h1 { color: #00FF41; font-family: 'Courier New', Courier, monospace; }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.title("⚖️ The Shadow Auditor")
    st.markdown("---")
    st.markdown("### 📚 Base de connaissances")
    st.success("✅ Code du Travail\n\n✅ Code des Obligations (COC)\n\n✅ Code des Sociétés (CSC)\n\n✅ JORT\n\n✅ INPDP")
    st.markdown("---")
    st.markdown("### ⚙️ Paramètres Moteur")
    st.code(f"Modèle : {GROQ_MODEL}\nBase : ChromaDB\nTop K : {TOP_K}")

st.title("⚖️ Interface d'Audit Juridique Tunisien")
st.markdown("Posez vos questions sur le droit des sociétés, le travail ou la protection des données.")

with st.spinner("Connexion sécurisée à la base juridique..."):
    retriever, llm = init_rag_system()
    prompt = build_prompt() # Utilisation de ton prompt personnalisé

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Bonjour. Je suis The Shadow Auditor. Que souhaitez-vous vérifier dans la loi tunisienne ?"}]

# Affichage de l'historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "chunks" in msg:
            with st.expander("🔍 Voir les textes de lois utilisés (Agentic RAG Thought Process)"):
                for i, doc in enumerate(msg["chunks"], 1):
                    source = doc.metadata.get("source", "Inconnu")
                    page = doc.metadata.get("page", "?")
                    if isinstance(page, int): page += 1
                    st.info(f"**Document : {source} (Page {page})**\n\n{doc.page_content}")

# Gestion de la question utilisateur
if user_question := st.chat_input("Ex: Quelles sont les conditions de licenciement abusif selon le Code du Travail ?"):
    
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Analyse approfondie des codes tunisiens en cours..."):
            
            # Utilisation de tes fonctions de formatage
            docs = retriever.invoke(user_question)
            context = format_context(docs)
            sources = format_sources(docs)
            
            final_prompt = prompt.format(context=context, question=user_question)
            response = llm.invoke(final_prompt).content
            
            if "Sources :" not in response:
                response = response.strip() + f"\n\n**Sources confirmées :**\n{sources}"
            
            st.markdown(response)
            
            with st.expander("🔍 Voir les textes de lois utilisés (Agentic RAG Thought Process)"):
                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get("source", "Inconnu")
                    page = doc.metadata.get("page", "?")
                    if isinstance(page, int): page += 1
                    st.info(f"**Document : {source} (Page {page})**\n\n{doc.page_content}")
            
    st.session_state.messages.append({"role": "assistant", "content": response, "chunks": docs})