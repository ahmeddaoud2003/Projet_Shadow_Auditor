import os
import re
from collections import defaultdict
from io import BytesIO

import streamlit as st
from docx import Document
from pypdf import PdfReader

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# ==========================================
# CONFIGURATION
# ==========================================
DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
OLLAMA_MODEL = "deepseek-r1:8b"

# ==========================================
# FONCTIONS UTILITAIRES
# ==========================================

def format_context(docs):
    parts = []
    for i, doc in enumerate(docs, start=1):
        parts.append(f"--- Extrait {i} ---\n{doc.page_content}\n")
    return "\n".join(parts)

def format_sources(docs):
    seen = set()
    lines = []
    for doc in docs:
        title = doc.metadata.get("doc_title", "Loi")
        page = doc.metadata.get("page", 0) + 1
        key = f"{title} – Page {page}"
        if key not in seen:
            seen.add(key)
            lines.append(f"• {key}")
    return "\n".join(lines) if lines else "Sources non identifiées"

def extract_thinking_and_answer(text: str):
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = text.replace(f"<think>{think_match.group(1)}</think>", "").strip()
        return thinking, answer
    return "Aucune réflexion intermédiaire.", text.strip()

def creer_document_word(contenu: str) -> BytesIO:
    doc = Document()
    doc.add_heading("Document Juridique Officiel", 0)
    for line in contenu.split("\n"):
        doc.add_paragraph(line)
    doc.add_paragraph("\n\nGénéré par The Shadow Auditor (Powered by DeepSeek-R1)")
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

def retrieve_multi_source(query: str, retriever, top_k_per_source: int = 3) -> list:
    all_docs = retriever.invoke(query)
    by_source = defaultdict(list)
    for doc in all_docs:
        title = doc.metadata.get("doc_title", "Inconnu")
        by_source[title].append(doc)

    keywords = {
        "travail": "CODE DU TRAVAIL",
        "licenciement": "CODE DU TRAVAIL",
        "contrat de travail": "CODE DU TRAVAIL",
        "société": "CODE SOCIETE",
        "sarl": "CODE SOCIETE",
        "sa ": "CODE SOCIETE",
        "obligations": "Code des Obligations et Contrats",
        "coc": "Code des Obligations et Contrats",
        "vente": "Code des Obligations et Contrats",
        "inpdp": "INPDP",
        "données personnelles": "INPDP",
        "jort": "JORT",
        "journal officiel": "JORT",
    }

    q_lower = query.lower()
    requested_sources = {v for k, v in keywords.items() if k in q_lower}

    if len(requested_sources) > 1:
        balanced = []
        for src in requested_sources:
            candidates = by_source.get(src, [])
            if len(candidates) < top_k_per_source:
                try:
                    extra = retriever.vectorstore.similarity_search(
                        query, k=top_k_per_source, filter={"doc_title": src}
                    )
                    candidates = (candidates + extra)[:top_k_per_source]
                except Exception:
                    pass
            balanced.extend(candidates[:top_k_per_source])
        seen_ids = {id(d) for d in balanced}
        for doc in all_docs:
            if id(doc) not in seen_ids:
                balanced.append(doc)
                if len(balanced) >= 12:
                    break
        return balanced

    return all_docs


# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="The Shadow Auditor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==========================================
# GLOBAL CSS — Luxury Dark Legal Theme
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Lato:wght@300;400;700&display=swap');

:root {
    --gold:      #C9A84C;
    --gold-light:#E8C97A;
    --gold-dim:  #8A6E30;
    --bg-deep:   #0D0F14;
    --bg-card:   #141720;
    --bg-input:  #1A1E2A;
    --bg-hover:  #1E2330;
    --border:    rgba(201,168,76,0.18);
    --border-hi: rgba(201,168,76,0.45);
    --text-main: #E8E4D9;
    --text-muted:#8A8680;
    --text-dim:  #555050;
    --teal:      #2A7B6F;
    --teal-light:#3AADA0;
}

html, body, [data-testid="stAppViewContainer"], .main {
    background: var(--bg-deep) !important;
    color: var(--text-main) !important;
    font-family: 'Lato', sans-serif !important;
}

.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

#MainMenu, footer, header,
[data-testid="stDecoration"],
[data-testid="stToolbar"] { display: none !important; }

/* ── TABS NAV ── */
[data-testid="stTabs"] {
    background: var(--bg-card) !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 0 2rem !important;
    position: sticky; top: 0; z-index: 100;
}
button[data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-family: 'Lato', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    padding: 1.1rem 1.8rem !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    transition: all 0.2s ease !important;
}
button[data-baseweb="tab"]:hover {
    color: var(--gold-light) !important;
    background: rgba(201,168,76,0.04) !important;
}
button[aria-selected="true"][data-baseweb="tab"] {
    color: var(--gold) !important;
    border-bottom: 2px solid var(--gold) !important;
    background: transparent !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background: var(--gold) !important;
    height: 2px !important;
}
[data-testid="stTabsContent"] {
    background: var(--bg-deep) !important;
    padding: 0 !important;
}

/* ── HEADER ── */
.shadow-header {
    background: linear-gradient(135deg, #0D0F14 0%, #141720 60%, #0D0F14 100%);
    border-bottom: 1px solid var(--border);
    padding: 1.5rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.shadow-header .brand { display: flex; align-items: center; gap: 1rem; }
.shadow-header .seal {
    width: 46px; height: 46px;
    border: 1.5px solid var(--gold-dim);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
    background: rgba(201,168,76,0.07);
}
.shadow-header h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: var(--gold) !important;
    margin: 0 !important;
    letter-spacing: 0.02em;
}
.shadow-header .tagline {
    font-size: 0.68rem;
    color: var(--text-dim);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 3px;
}
.shadow-header .status-pill {
    background: rgba(42,123,111,0.12);
    border: 1px solid rgba(58,173,160,0.28);
    color: var(--teal-light);
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 0.28rem 0.85rem;
    border-radius: 20px;
    font-weight: 700;
}

/* ── PAGE HEADERS ── */
.page-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-main);
    margin: 0 0 0.15rem;
}
.page-title span { color: var(--gold); }
.page-subtitle {
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 0.06em;
    margin-bottom: 1.6rem;
    padding-bottom: 1.1rem;
    border-bottom: 1px solid var(--border);
}
.tab-body { padding: 1.8rem 2.5rem; max-width: 1200px; margin: 0 auto; }

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.4rem 0 !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: rgba(201,168,76,0.05) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.85rem 1.2rem !important;
    margin: 0.35rem 0 !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--gold-dim) !important;
    border-radius: 10px !important;
    padding: 0.85rem 1.2rem !important;
    margin: 0.35rem 0 !important;
}
[data-testid="stChatMessageAvatarUser"] {
    background: var(--gold-dim) !important;
    border-radius: 50% !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
    background: var(--bg-input) !important;
    border: 1px solid var(--gold-dim) !important;
    border-radius: 50% !important;
}

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 12px !important;
    box-shadow: 0 0 24px rgba(201,168,76,0.05) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px rgba(201,168,76,0.09) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text-main) !important;
    font-family: 'Lato', sans-serif !important;
    font-size: 0.92rem !important;
    caret-color: var(--gold) !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--text-dim) !important; }
[data-testid="stChatInput"] button {
    background: var(--gold) !important;
    border-radius: 8px !important;
    border: none !important;
    color: #0D0F14 !important;
}

/* ── EXPANDERS ── */
[data-testid="stExpander"] {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    margin: 0.45rem 0 !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 1rem !important;
}
[data-testid="stExpander"] summary:hover { color: var(--gold-light) !important; }
[data-testid="stExpander"] > div > div {
    padding: 0.75rem 1rem !important;
    font-size: 0.84rem !important;
    color: var(--text-muted) !important;
    line-height: 1.7 !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--border-hi) !important;
    color: var(--gold) !important;
    font-family: 'Lato', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.13em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.5rem !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: rgba(201,168,76,0.09) !important;
    border-color: var(--gold) !important;
    box-shadow: 0 0 18px rgba(201,168,76,0.13) !important;
}
.stButton > button[kind="primary"] {
    background: var(--gold) !important;
    color: #0D0F14 !important;
    border-color: var(--gold) !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--gold-light) !important;
    box-shadow: 0 4px 18px rgba(201,168,76,0.28) !important;
}

/* ── DOWNLOAD BUTTON ── */
[data-testid="stDownloadButton"] button {
    background: rgba(201,168,76,0.07) !important;
    border: 1px solid var(--gold-dim) !important;
    color: var(--gold) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
    border-radius: 6px !important;
    transition: all 0.2s !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: var(--gold) !important;
    color: #0D0F14 !important;
}

/* ── TEXT AREA ── */
textarea, [data-testid="stTextArea"] textarea {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-main) !important;
    font-family: 'Lato', sans-serif !important;
    font-size: 0.9rem !important;
    line-height: 1.65 !important;
    transition: border-color 0.2s !important;
}
textarea:focus { border-color: var(--gold-dim) !important; box-shadow: 0 0 0 3px rgba(201,168,76,0.07) !important; }

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1.5px dashed var(--border-hi) !important;
    border-radius: 12px !important;
    padding: 2.2rem !important;
    transition: all 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--gold) !important;
    background: rgba(201,168,76,0.03) !important;
}
[data-testid="stFileUploader"] label { color: var(--text-muted) !important; font-size: 0.88rem !important; }

/* ── RADIO ── */
[data-testid="stRadio"] label, [data-testid="stRadio"] p { color: var(--text-muted) !important; font-size: 0.85rem !important; }

/* ── SPINNER ── */
[data-testid="stSpinner"] p {
    color: var(--gold-dim) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}

/* ── MARKDOWN ── */
[data-testid="stMarkdownContainer"] { color: var(--text-main) !important; font-size: 0.92rem !important; line-height: 1.78 !important; }
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--gold-light) !important;
    font-weight: 600 !important;
}
[data-testid="stMarkdownContainer"] h2 {
    font-size: 1.05rem !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 0.35rem !important;
    margin-top: 1.1rem !important;
}
[data-testid="stMarkdownContainer"] strong { color: var(--gold-light) !important; }
[data-testid="stMarkdownContainer"] table { border-collapse: collapse !important; width: 100% !important; font-size: 0.84rem !important; }
[data-testid="stMarkdownContainer"] th {
    background: rgba(201,168,76,0.1) !important;
    color: var(--gold) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 0.85rem !important;
    border: 1px solid var(--border) !important;
}
[data-testid="stMarkdownContainer"] td { padding: 0.5rem 0.85rem !important; border: 1px solid var(--border) !important; }
[data-testid="stMarkdownContainer"] tr:nth-child(even) td { background: rgba(255,255,255,0.02) !important; }
[data-testid="stMarkdownContainer"] code {
    background: rgba(201,168,76,0.08) !important;
    color: var(--gold-light) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    padding: 0.1rem 0.4rem !important;
    font-size: 0.8rem !important;
}

hr { border-color: var(--border) !important; margin: 1.4rem 0 !important; }

/* ── AGENT CARDS ── */
.agent-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem 0.8rem;
    margin: 1.2rem 0 0.6rem;
    position: relative;
    overflow: hidden;
}
.agent-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.agent-card.plaignant::before { background: var(--teal-light); }
.agent-card.defense::before   { background: #C0392B; }
.agent-card.juge::before      { background: var(--gold); }

.agent-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}
.agent-label.plaignant { color: var(--teal-light); }
.agent-label.defense   { color: #E74C3C; }
.agent-label.juge      { color: var(--gold); }

.agent-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-main);
}

/* ── VERDICT BOX ── */
.verdict-box {
    background: linear-gradient(135deg, rgba(139,32,32,0.12), rgba(13,15,20,0.9));
    border: 1.5px solid rgba(192,57,43,0.4);
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    position: relative;
    margin-top: 0.6rem;
}
.verdict-box::before {
    content: 'VERDICT';
    position: absolute;
    top: -0.58rem; left: 1.4rem;
    background: var(--bg-deep);
    padding: 0 0.55rem;
    font-size: 0.62rem;
    letter-spacing: 0.28em;
    color: #E74C3C;
    font-weight: 700;
}

/* ── WELCOME SCREEN ── */
.welcome-screen {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 420px;
    gap: 1rem;
    opacity: 0.45;
}
.welcome-icon { font-size: 3rem; }
.welcome-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    color: var(--gold);
}
.welcome-sub { font-size: 0.78rem; color: var(--text-dim); text-align: center; max-width: 380px; line-height: 1.7; }

/* ── COLUMNS ── */
[data-testid="stColumns"] { gap: 1.2rem !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--gold-dim); }

::selection { background: rgba(201,168,76,0.22); color: var(--text-main); }

[data-testid="stWidgetLabel"] p,
[data-testid="stCaptionContainer"] p,
label { color: var(--text-muted) !important; font-size: 0.8rem !important; }

h1, h2, h3 { font-family: 'Playfair Display', serif !important; color: var(--text-main) !important; }

/* Info box */
[data-testid="stInfo"] {
    background: rgba(42,123,111,0.09) !important;
    border-color: rgba(58,173,160,0.25) !important;
    border-radius: 8px !important;
    color: var(--text-muted) !important;
    font-size: 0.84rem !important;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================
st.markdown("""
<div class="shadow-header">
    <div class="brand">
        <div class="seal">⚖️</div>
        <div>
            <h1>The Shadow Auditor</h1>
            <div class="tagline">Intelligence Juridique Tunisienne &nbsp;·&nbsp; Powered by DeepSeek-R1</div>
        </div>
    </div>
    <div class="status-pill">● Système Actif</div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# INITIALISATION RAG
# ==========================================
@st.cache_resource(show_spinner=False)
def init_rag_system():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"prompts": {"query": "query: ", "passage": "passage: "}}
    )
    if not os.path.exists(DB_DIR):
        st.error("⚠️ Base ChromaDB introuvable. Lancez 'python ingestion.py' d'abord.")
        st.stop()
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 30}
    )
    retriever.vectorstore = vectorstore
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    return retriever, llm

with st.spinner("Connexion à la base juridique..."):
    retriever, base_llm = init_rag_system()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3 = st.tabs([
    "💬  Agent Juridique",
    "🕵️  Scanner de Contrat",
    "⚔️  Tribunal Virtuel",
])


# ==========================================
# ONGLET 1 — AGENT JURIDIQUE
# ==========================================
with tab1:
    st.markdown("""
    <div class="tab-body" style="padding-bottom:0.5rem">
        <div class="page-title">Agent <span>Juridique</span></div>
        <div class="page-subtitle">
            Posez vos questions en français ou en arabe — droit du travail, sociétés, obligations, INPDP
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container(height=560, border=False):
        if not st.session_state.messages:
            st.markdown("""
            <div class="welcome-screen">
                <div class="welcome-icon">⚖️</div>
                <div class="welcome-title">Posez votre première question</div>
                <div class="welcome-sub">
                    Calculs d'indemnités · Rédaction d'actes · Analyse multi-codes<br>
                    حساب التعويضات · صياغة العقود · تحليل متعدد القوانين
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    if msg.get("thinking"):
                        with st.expander("🧠 Chaîne de raisonnement — DeepSeek R1"):
                            st.write(msg["thinking"])
                    st.markdown(msg["content"])
                    if msg.get("sources"):
                        with st.expander("📚 Sources juridiques consultées"):
                            st.markdown(msg["sources"])

if user_question := st.chat_input(
    "Ex: Calcule l'indemnité de licenciement abusif (Art 23 bis) pour 3500 DT et 14 ans..."
):
    st.session_state.messages.append({"role": "user", "content": user_question})

    with tab1:
        with st.container(height=560, border=False):
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Consultation des codes juridiques..."):
                    docs = retrieve_multi_source(user_question, retriever, top_k_per_source=3)
                    context = format_context(docs)
                    sources_text = format_sources(docs)

                    system_instruction = f"""أنت "The Shadow Auditor"، خبير قانوني ومالي تونسي.
You are "The Shadow Auditor", a Tunisian legal and financial expert.

════════════════════════════════════════
RÈGLE LINGUISTIQUE ABSOLUE — قاعدة اللغة المطلقة
════════════════════════════════════════
- Si la question est en FRANÇAIS → réponds ENTIÈREMENT en français.
- Si la question est en ARABE ou en dialecte tunisien (دارجة) → réponds ENTIÈREMENT en arabe.
- Ne mélange JAMAIS les deux langues. Cette règle est PRIORITAIRE.
إذا كان السؤال بالعربية أو بالدارجة التونسية، يجب أن تكون إجابتك كاملة بالعربية.

════════════════════════════════════════
TEXTES JURIDIQUES TUNISIENS DISPONIBLES
════════════════════════════════════════
{context}

════════════════════════════════════════
RÈGLES IMPÉRATIVES
════════════════════════════════════════

POUR LES CALCULS : dans <think>, fais dans l'ordre :
1. Article applicable (numéro exact)
2. Formule littérale
3. Remplacement par les valeurs
4. Calcul pas à pas
5. Vérification du plafond légal
6. Résultat arrondi final

Réponse finale : article + formule + calcul + résultat **en gras** + plafond.

POUR QUESTIONS MULTI-CODES : sous-titre ## par code, indiquer complémentarité ou contradiction.

POUR RÉDACTION D'ACTES : format professionnel, en-tête, objet, corps, signature, articles cités.

SI ABSENT : "⚠️ Les textes fournis ne contiennent pas cet article. Consultez un avocat spécialisé."
NE fabrique PAS de chiffres ou d'articles inexistants.
"""
                    messages_llm = [
                        SystemMessage(content=system_instruction),
                        HumanMessage(content=user_question)
                    ]
                    raw_response = base_llm.invoke(messages_llm).content
                    thinking, final_answer = extract_thinking_and_answer(raw_response)

                    with st.expander("🧠 Chaîne de raisonnement — DeepSeek R1"):
                        st.write(thinking)

                    st.markdown(final_answer)

                    with st.expander("📚 Sources juridiques consultées"):
                        st.markdown(sources_text)

                    doc_keywords = ["Objet :", "Mise en demeure", "Monsieur", "Madame", "À l'attention"]
                    if any(kw in final_answer for kw in doc_keywords):
                        st.download_button(
                            "📥 Télécharger l'acte en Word",
                            data=creer_document_word(final_answer),
                            file_name="Acte_Juridique.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer,
        "thinking": thinking,
        "sources": sources_text
    })


# ==========================================
# ONGLET 2 — SCANNER DE CONTRAT
# ==========================================
with tab2:
    st.markdown("""
    <div class="tab-body">
        <div class="page-title">Scanner de <span>Contrat</span></div>
        <div class="page-subtitle">
            Uploadez un PDF · L'IA détecte les clauses illégales selon les codes tunisiens
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        uploaded_file = st.file_uploader(
            "Glissez votre contrat ici",
            type="pdf",
            label_visibility="visible"
        )
        if uploaded_file:
            st.markdown(f"""
            <div style="background:rgba(201,168,76,0.06);border:1px solid rgba(201,168,76,0.2);
                        border-radius:8px;padding:0.75rem 1rem;margin:0.6rem 0;">
                <div style="font-size:0.65rem;letter-spacing:0.18em;text-transform:uppercase;
                            color:#8A6E30;margin-bottom:0.2rem;">Fichier chargé</div>
                <div style="font-size:0.85rem;color:#E8E4D9;font-weight:700;">📄 {uploaded_file.name}</div>
                <div style="font-size:0.7rem;color:#555050;margin-top:0.15rem;">{uploaded_file.size // 1024} Ko</div>
            </div>
            """, unsafe_allow_html=True)

            audit_mode = st.radio(
                "Profondeur d'analyse",
                ["⚡ Rapide (2 500 mots)", "🔍 Complet (5 000 mots)"],
                index=0
            )
            run_audit = st.button("🔍 Lancer l'Audit", type="primary", use_container_width=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;opacity:0.35;">
                <div style="font-size:2.5rem;margin-bottom:0.7rem;">📄</div>
                <div style="font-size:0.8rem;color:#555050;line-height:1.6;">
                    Glissez votre contrat PDF<br>ou cliquez pour parcourir
                </div>
            </div>
            """, unsafe_allow_html=True)
            run_audit = False

    with col_right:
        if uploaded_file and run_audit:
            with st.spinner("Analyse juridique en cours..."):
                pdf_reader = PdfReader(uploaded_file)
                texte_contrat = "".join([p.extract_text() or "" for p in pdf_reader.pages])
                word_limit = 2500 if "Rapide" in audit_mode else 5000
                texte_analyse = texte_contrat[:word_limit]

                docs_loi = retrieve_multi_source(texte_contrat[:600], retriever, top_k_per_source=3)
                contexte_loi = format_context(docs_loi)

                audit_prompt = f"""Tu es un auditeur juridique tunisien expert.

LOIS TUNISIENNES DE RÉFÉRENCE :
{contexte_loi}

CONTRAT À AUDITER :
{texte_analyse}

MISSION :
1. Identifie les clauses potentiellement problématiques ou illégales.
2. Cite l'article tunisien applicable pour chaque clause.
3. Tableau Markdown :

| # | Clause / Extrait | Statut | Article de loi | Explication |
|---|-----------------|--------|---------------|-------------|
| 1 | ... | ✅ Conforme / ⚠️ Risque / ❌ Illégal | Art. X | ... |

4. Section "## Recommandations" avec actions prioritaires.
Si conforme, indique-le clairement."""

                res = base_llm.invoke(audit_prompt).content
                thinking, clean_res = extract_thinking_and_answer(res)

            with st.expander("🧠 Raisonnement de l'Auditeur — DeepSeek R1"):
                st.write(thinking)

            st.markdown(clean_res)

            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "📥 Rapport Word",
                    data=creer_document_word(clean_res),
                    file_name="Rapport_Audit_Contrat.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True
                )
            with c2:
                with st.expander("📚 Lois consultées"):
                    st.markdown(format_sources(docs_loi))

        elif not uploaded_file:
            st.markdown("""
            <div style="display:flex;align-items:center;justify-content:center;
                        height:320px;opacity:0.22;">
                <div style="text-align:center;">
                    <div style="font-size:2rem;margin-bottom:0.5rem;">🔍</div>
                    <div style="font-size:0.82rem;color:#555050;">Le rapport apparaîtra ici</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ==========================================
# ONGLET 3 — TRIBUNAL VIRTUEL
# ==========================================
with tab3:
    st.markdown("""
    <div class="tab-body">
        <div class="page-title">Tribunal <span>Virtuel</span></div>
        <div class="page-subtitle">
            Trois agents IA s'affrontent &nbsp;·&nbsp;
            Avocat du Plaignant &nbsp;·&nbsp; Avocat de la Défense &nbsp;·&nbsp; Le Juge tranche
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "tribunal_result" not in st.session_state:
        st.session_state.tribunal_result = None

    col_form, col_opts = st.columns([3, 1], gap="large")

    with col_form:
        litige = st.text_area(
            "Décrivez le litige",
            placeholder=(
                "Ex: Mon patron m'a licencié car j'ai refusé de travailler le dimanche. "
                "J'ai 8 ans d'ancienneté et un salaire de 2000 DT.\n\n"
                "مثال: صاحب العمل طردني من الخدمة باش فرض عليا نخدم يوم الأحد بالرغم من رفضي."
            ),
            height=155,
            label_visibility="collapsed"
        )

    with col_opts:
        st.markdown("<div style='height:0.2rem'></div>", unsafe_allow_html=True)
        langue_tribunal = st.radio(
            "Langue du débat",
            ["🇫🇷 Français", "🇹🇳 Arabe / Dارجة"],
        )
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        launch = st.button(
            "⚔️ Ouvrir l'Audience",
            type="primary",
            disabled=not (litige and litige.strip()),
            use_container_width=True
        )

    is_arabic = "Arabe" in langue_tribunal

    if launch and litige.strip():

        with st.spinner("Recherche des articles de loi applicables..."):
            docs_tribunal = retrieve_multi_source(litige, retriever, top_k_per_source=4)
            contexte_tribunal = format_context(docs_tribunal)
            sources_tribunal = format_sources(docs_tribunal)

        if is_arabic:
            lang_rule = "⚠️ قاعدة مطلقة: مرافعتك كاملة بالعربية أو الدارجة التونسية فقط."
            lang_label = "العربية / الدارجة"
        else:
            lang_rule = "⚠️ Règle absolue : plaidoirie ENTIÈREMENT en français."
            lang_label = "français"

        base_ctx = f"""TEXTES JURIDIQUES TUNISIENS :
{contexte_tribunal}

LITIGE :
{litige}

LANGUE : {lang_label} — {lang_rule}
"""

        # ── AGENT 1 ─────────────────────────────────────────────────────
        st.markdown("""
        <div class="agent-card plaignant">
            <div class="agent-label plaignant">Agent 01 — Défense du Plaignant</div>
            <div class="agent-name">🧑‍⚖️ Avocat du Salarié / Victime</div>
        </div>
        """, unsafe_allow_html=True)

        spin1 = "المحامي الأول يحضّر مرافعته..." if is_arabic else "L'Avocat du Plaignant rédige sa plaidoirie..."
        with st.spinner(spin1):
            if is_arabic:
                p1 = f"""{base_ctx}
أنت محامٍ متمرّس يدافع عن المدّعي. في <think>: خطة المرافعة والمواد.
ردّك النهائي بالعربية فقط:
## استراتيجية الدفاع
- الوقائع الداعمة لموكلك
- المواد القانونية (مع أرقامها)
- المطالب والتعويضات
"""
            else:
                p1 = f"""{base_ctx}
Tu défends le Plaignant. Dans <think> : stratégie et articles favorables.
Réponse finale en français uniquement :
## Plaidoirie — Défense du Plaignant
- Faits favorables au client
- Articles invoqués (numéros exacts)
- Demandes et réparations
"""
            raw1 = base_llm.invoke(p1).content
            think1, plea1 = extract_thinking_and_answer(raw1)

        with st.expander("🧠 Raisonnement Agent 1 — DeepSeek R1"):
            st.write(think1)
        st.markdown(plea1)

        # ── AGENT 2 ─────────────────────────────────────────────────────
        st.markdown("""
        <div class="agent-card defense">
            <div class="agent-label defense">Agent 02 — Contre-attaque</div>
            <div class="agent-name">🏢 Avocat de l'Employeur / Entreprise</div>
        </div>
        """, unsafe_allow_html=True)

        spin2 = "محامي الدفاع يردّ على المرافعة الأولى..." if is_arabic else "L'Avocat de la Défense prépare sa contre-attaque..."
        with st.spinner(spin2):
            if is_arabic:
                p2 = f"""{base_ctx}
مرافعة الخصم: \"\"\"{plea1}\"\"\"
أنت تدافع عن صاحب العمل. فنّد حجج الخصم. ردّك النهائي بالعربية فقط:
## مرافعة الدفاع
- الردّ نقطة بنقطة
- المواد القانونية الداعمة
- طلبات الرفض أو التخفيف
"""
            else:
                p2 = f"""{base_ctx}
Plaidoirie adverse : \"\"\"{plea1}\"\"\"
Tu défends l'Employeur. Réfute les arguments adverses. Réponse en français :
## Plaidoirie — Défense de l'Employeur
- Réfutation point par point
- Articles de loi pour la défense
- Demandes de rejet / réduction
"""
            raw2 = base_llm.invoke(p2).content
            think2, plea2 = extract_thinking_and_answer(raw2)

        with st.expander("🧠 Raisonnement Agent 2 — DeepSeek R1"):
            st.write(think2)
        st.markdown(plea2)

        # ── AGENT 3 : JUGE ───────────────────────────────────────────────
        st.markdown("""
        <div class="agent-card juge">
            <div class="agent-label juge">Agent 03 — Délibéré Final</div>
            <div class="agent-name">👨‍⚖️ Le Juge — Verdict Motivé</div>
        </div>
        """, unsafe_allow_html=True)

        spin3 = "القاضي يدرس الملف ويصدر حكمه..." if is_arabic else "Le Juge délibère et rend son verdict..."
        with st.spinner(spin3):
            if is_arabic:
                p3 = f"""{base_ctx}
مرافعة المدّعي: \"\"\"{plea1}\"\"\"
مرافعة الدفاع: \"\"\"{plea2}\"\"\"
أنت قاضٍ تونسي نزيه. حكمك المسبّب بالعربية فقط:
## الحكم القضائي
### أولاً — في الوقائع
### ثانياً — في القانون
### ثالثاً — الحكم (يُقضى / يُرفض + التعويضات)
### ملاحظة الجلسة
"""
            else:
                p3 = f"""{base_ctx}
Plaignant : \"\"\"{plea1}\"\"\"
Défense : \"\"\"{plea2}\"\"\"
Tu es un Juge impartial. Verdict motivé en français :
## Verdict Judiciaire
### I — En fait
### II — En droit (articles appliqués)
### III — Par ces motifs (CONDAMNE / DÉBOUTE + réparations)
### Note d'audience
"""
            raw3 = base_llm.invoke(p3).content
            think3, verdict = extract_thinking_and_answer(raw3)

        with st.expander("🧠 Délibéré du Juge — DeepSeek R1"):
            st.write(think3)

        st.markdown(f'<div class="verdict-box">{verdict}</div>', unsafe_allow_html=True)
        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

        ca, cb = st.columns(2)
        with ca:
            rapport = (
                f"LITIGE :\n{litige}\n\n"
                f"PLAIDOIRIE DU PLAIGNANT :\n{plea1}\n\n"
                f"PLAIDOIRIE DE LA DÉFENSE :\n{plea2}\n\n"
                f"VERDICT :\n{verdict}"
            )
            st.download_button(
                "📥 Télécharger le procès-verbal",
                data=creer_document_word(rapport),
                file_name="Tribunal_Virtuel.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )
        with cb:
            with st.expander("📚 Articles de loi consultés"):
                st.markdown(sources_tribunal)