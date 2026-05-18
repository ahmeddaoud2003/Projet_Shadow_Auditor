import os
import re
import streamlit as st
from io import BytesIO
from docx import Document
from pypdf import PdfReader

# --- IMPORTS LOCAUX ---
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

# CONFIGURATION
DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "deepseek-r1:8b" # NOTRE NOUVEAU MODELE RAISONNEUR !
TOP_K = 6

# --- FONCTIONS DE FORMATAGE ---
def format_context(docs):
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Inconnu")
        page = doc.metadata.get("page", 0) + 1
        parts.append(f"--- Extrait {i} ({source} - Page {page}) ---\n{doc.page_content}\n")
    return "\n".join(parts)

def format_sources(docs):
    sources = set([f"{doc.metadata.get('doc_title', 'Loi')} (Page {doc.metadata.get('page', 0) + 1})" for doc in docs])
    return "\n".join(sources)

# --- FONCTION POUR EXTRAIRE LE CERVEAU DE DEEPSEEK (<think>) ---
def extract_thinking_and_answer(text):
    """DeepSeek réfléchit dans des balises <think>. On va les extraire pour impressionner le jury."""
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = text.replace(f'<think>{think_match.group(1)}</think>', '').strip()
        return thinking, answer
    return "Réflexion instantanée.", text.strip()

# --- GENERATION WORD ---
def creer_document_word(contenu):
    doc = Document()
    doc.add_heading('Document Juridique Officiel', 0)
    doc.add_paragraph(contenu)
    doc.add_paragraph("\n\nGénéré par The Shadow Auditor (Powered by DeepSeek-R1)")
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# --- INITIALISATION ---
st.set_page_config(page_title="The Shadow Auditor", page_icon="⚖️", layout="wide")

@st.cache_resource(show_spinner=False)
def init_rag_system():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if not os.path.exists(DB_DIR):
        st.error("⚠️ Base ChromaDB introuvable. Lancez 'python ingestion.py' d'abord.")
        st.stop()
        
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    
    # LE CERVEAU DEEPSEEK (Local, Gratuit, Intelligent)
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    
    return retriever, llm

# --- UI STYLE ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA;}
    h1, h2, h3 { color: #00FF41; font-family: 'Courier New', Courier, monospace; }
    .stButton>button { background-color: #00FF41; color: black; font-weight: bold; border-radius: 8px;}
    </style>
    """, unsafe_allow_html=True)

with st.spinner("Activation de DeepSeek-R1 (Local AI)..."):
    retriever, base_llm = init_rag_system()

tab1, tab2 = st.tabs(["💬 L'Agent DeepSeek (Chat & Actions)", "🕵️‍♂️ Scanner de Contrat (Audit)"])

# ==========================================
# ONGLET 1 : L'AGENT DEEPSEEK
# ==========================================
with tab1:
    st.title("⚖️ Shadow Auditor (DeepSeek-R1)")
    st.markdown("⚠️ *Analyse 100% Locale et Confidentielle.*")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_container = st.container()

    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg.get("thinking"):
                    with st.expander("🧠 Voir la réflexion de l'IA (DeepSeek Thought Process)"):
                        st.write(msg["thinking"])
                st.markdown(msg["content"])
                
                if msg["role"] == "assistant" and "Objet : Mise en demeure" in msg["content"]:
                    st.download_button("📥 Télécharger l'Acte (.docx)", data=creer_document_word(msg["content"]), file_name="Acte.docx")

    if user_question := st.chat_input("Ex: Calcule l'indemnité pour 2500 TND (8 ans)... OU Rédige une mise en demeure..."):
        with chat_container:
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("DeepSeek réfléchit à la solution juridique..."):
                    docs = retriever.invoke(user_question)
                    context = format_context(docs)
                    
                    system_instruction = f"""Tu es The Shadow Auditor, expert juridique tunisien.
Contexte lois tunisiennes : 
{context}

RÈGLES :
1. Si l'utilisateur demande de CALCULER une indemnité (salaire et ancienneté), utilise la règle de l'Art 23 bis du Code du Travail (entre 1 et 2 mois de salaire par année d'ancienneté, max 3 ans de salaire). Fais le calcul mathématique toi-même de manière très précise.
2. Si l'utilisateur demande de RÉDIGER un acte juridique, écris le texte avec l'en-tête 'Objet : Mise en demeure'.
3. Réponds de manière sûre, en citant le code exact. Ne parle pas de toi.
"""
                    messages = [SystemMessage(content=system_instruction), HumanMessage(content=user_question)]
                    
                    # On appelle DeepSeek
                    raw_response = base_llm.invoke(messages).content
                    
                    # On sépare la "pensée" de la "réponse finale"
                    thinking, final_answer = extract_thinking_and_answer(raw_response)
                    
                    # On affiche le cerveau
                    with st.expander("🧠 Voir la réflexion de l'IA (DeepSeek Thought Process)"):
                        st.write(thinking)
                        
                    # On affiche la réponse
                    st.markdown(final_answer)
                    
                    if "Objet : Mise en demeure" in final_answer:
                        st.download_button("📥 Télécharger l'Acte (.docx)", data=creer_document_word(final_answer), file_name="Acte.docx")

                    with st.expander("📚 Sources de la loi récupérées par RAG"):
                        st.markdown(format_sources(docs))

        st.session_state.messages.append({"role": "assistant", "content": final_answer, "thinking": thinking})

# ==========================================
# ONGLET 2 : SCANNER DE CONTRAT
# ==========================================
with tab2:
    st.title("🕵️‍♂️ Scanner de Contrat par DeepSeek")
    
    uploaded_file = st.file_uploader("Contrat (PDF)", type="pdf")
    if uploaded_file and st.button("🚀 Lancer l'Audit"):
        with st.spinner("DeepSeek analyse le contrat clause par clause..."):
            pdf_reader = PdfReader(uploaded_file)
            texte_contrat = "".join([page.extract_text() for page in pdf_reader.pages])
            
            docs_loi = retriever.invoke("Contrat de travail, horaires, licenciement.")
            contexte_loi = format_context(docs_loi)
            
            audit_prompt = f"""Tu es un Auditeur Juridique.
Lois : {contexte_loi}
Contrat à auditer : "{texte_contrat[:2000]}"

Génère UNIQUEMENT un tableau Markdown (sans blabla avant ni après) avec ces 3 colonnes :
| Clause du Contrat | Statut (Conforme ✅ / Illégal ❌) | Explication Légale |"""
            
            raw_response = base_llm.invoke(audit_prompt).content
            thinking, final_answer = extract_thinking_and_answer(raw_response)
            
            with st.expander("🧠 Voir comment l'IA a scanné le contrat (Pensée)"):
                st.write(thinking)
                
            st.success("Audit terminé avec succès !")
            st.markdown(final_answer)