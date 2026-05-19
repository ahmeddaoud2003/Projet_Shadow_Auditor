import os
import re
from collections import defaultdict
from io import BytesIO

import streamlit as st
from docx import Document
from pypdf import PdfReader

# --- IMPORTS LOCAUX ---
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# ==========================================
# CONFIGURATION
# ==========================================
DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"   # ✅ même modèle que ingestion.py
OLLAMA_MODEL = "deepseek-r1:8b"

# ==========================================
# FONCTIONS UTILITAIRES
# ==========================================

def format_context(docs):
    """Formate les chunks RAG en bloc lisible pour le LLM."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        parts.append(f"--- Extrait {i} ---\n{doc.page_content}\n")
    return "\n".join(parts)


def format_sources(docs):
    """Retourne la liste des sources uniques trouvées."""
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
    """Sépare le bloc <think> de DeepSeek de la réponse finale."""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = text.replace(f"<think>{think_match.group(1)}</think>", "").strip()
        return thinking, answer
    return "Aucune réflexion intermédiaire.", text.strip()


def creer_document_word(contenu: str) -> BytesIO:
    """Génère un fichier .docx en mémoire."""
    doc = Document()
    doc.add_heading("Document Juridique Officiel", 0)
    for line in contenu.split("\n"):
        doc.add_paragraph(line)
    doc.add_paragraph("\n\nGénéré par The Shadow Auditor (Powered by DeepSeek-R1)")
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer


# ==========================================
# ✅ FIX PRINCIPAL : Retriever multi-sources
# ==========================================
def retrieve_multi_source(query: str, retriever, top_k_per_source: int = 3) -> list:
    """
    Stratégie en deux passes :
    1. Récupère 30 candidats avec MMR classique.
    2. Rééquilibre pour garantir au moins `top_k_per_source` chunks
       par code juridique mentionné dans la question.
    """
    # Passe 1 : récupération large
    all_docs = retriever.invoke(query)  # MMR k=8, fetch_k=30

    # Groupe par doc_title
    by_source = defaultdict(list)
    for doc in all_docs:
        title = doc.metadata.get("doc_title", "Inconnu")
        by_source[title].append(doc)

    # Passe 2 : détecte les codes mentionnés explicitement dans la question
    keywords = {
        "travail": "Code du Travail",
        "licenciement": "Code du Travail",
        "contrat de travail": "Code du Travail",
        "société": "code societes fr",
        "sarl": "code societes fr",
        "sa ": "code societes fr",
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

    # Si plusieurs codes sont détectés, on force des chunks depuis chaque source
    if len(requested_sources) > 1:
        balanced = []
        for src in requested_sources:
            # Cherche les chunks déjà récupérés
            candidates = by_source.get(src, [])
            if len(candidates) < top_k_per_source:
                # Fallback : recherche ciblée avec filtre metadata
                try:
                    extra = retriever.vectorstore.similarity_search(
                        query,
                        k=top_k_per_source,
                        filter={"doc_title": src}
                    )
                    candidates = (candidates + extra)[:top_k_per_source]
                except Exception:
                    pass
            balanced.extend(candidates[:top_k_per_source])

        # Complète avec les autres docs non sélectionnés (pour le contexte général)
        seen_ids = {id(d) for d in balanced}
        for doc in all_docs:
            if id(doc) not in seen_ids:
                balanced.append(doc)
                if len(balanced) >= 12:
                    break
        return balanced

    return all_docs  # Question simple : retour MMR standard


# ==========================================
# INITIALISATION
# ==========================================
st.set_page_config(page_title="The Shadow Auditor", page_icon="⚖️", layout="wide")


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

    # MMR large : fetch 30 candidats, garde 8 diversifiés
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 30}
    )
    # Attache le vectorstore au retriever pour la recherche filtrée
    retriever.vectorstore = vectorstore

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    return retriever, llm


with st.spinner("⚖️ Activation de The Shadow Auditor..."):
    retriever, base_llm = init_rag_system()

if "messages" not in st.session_state:
    st.session_state.messages = []

tab1, tab2 = st.tabs(["💬 Agent Juridique", "🕵️‍♂️ Scanner de Documents"])

# ==========================================
# ONGLET 1 : CHAT
# ==========================================
with tab1:
    st.title("⚖️ The Shadow Auditor — Expert Droit Tunisien")
    chat_container = st.container(height=520)

    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg.get("thinking"):
                    with st.expander("🧠 Processus de réflexion (DeepSeek R1)"):
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
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("🔍 Consultation des codes juridiques..."):

                    # ✅ Retrieval multi-source équilibré
                    docs = retrieve_multi_source(user_question, retriever, top_k_per_source=3)
                    context = format_context(docs)
                    sources_text = format_sources(docs)

                    # ✅ FIX CALCUL : Prompt structuré avec étapes obligatoires
                    system_instruction = f"""Tu es "The Shadow Auditor", expert juridique et financier tunisien.
Tu raisonnes en français. Tu es PRÉCIS, RIGOUREUX et tu montres TOUJOURS tes calculs étape par étape.

════════════════════════════════════════
TEXTES JURIDIQUES TUNISIENS DISPONIBLES
════════════════════════════════════════
{context}

════════════════════════════════════════
RÈGLES IMPÉRATIVES
════════════════════════════════════════

### POUR LES CALCULS (indemnités, amendes, délais…) :
Dans ta balise <think>, tu DOIS faire dans l'ordre :
1. Identifier l'article applicable (cite son numéro exact)
2. Écrire la formule littérale : ex. Indemnité = Salaire_journalier × Jours_par_année × Années
3. Remplacer par les valeurs numériques fournies
4. Calculer pas à pas (multiplication, division, addition)
5. Vérifier si un PLAFOND légal s'applique et l'appliquer
6. Donner le résultat final arrondi

Dans ta réponse finale, présente :
- L'article juridique de base
- La formule utilisée
- Le calcul détaillé
- Le résultat final **en gras**
- Le plafond appliqué le cas échéant

### POUR LES QUESTIONS MULTI-CODES (ex: licenciement + COC + société) :
- Traite CHAQUE code séparément avec un sous-titre ## 
- Indique clairement si les dispositions se complètent ou se contredisent
- Donne une conclusion de synthèse

### POUR LA RÉDACTION D'ACTES :
- Rédige en format professionnel avec en-tête, objet, corps, signature
- Cite les articles de loi dans le corps du document

### SI L'INFORMATION EST ABSENTE :
- Dis exactement : "⚠️ Les textes fournis ne contiennent pas cet article. Consultez un avocat spécialisé."
- NE fabrique PAS de chiffres ou d'articles inexistants.
"""
                    messages = [
                        SystemMessage(content=system_instruction),
                        HumanMessage(content=user_question)
                    ]

                    raw_response = base_llm.invoke(messages).content
                    thinking, final_answer = extract_thinking_and_answer(raw_response)

                    with st.expander("🧠 Analyse & Calculs (DeepSeek R1 — Chaîne de pensée)"):
                        st.write(thinking)

                    st.markdown(final_answer)

                    with st.expander("📚 Sources juridiques consultées"):
                        st.markdown(sources_text)

                    # Bouton Word si acte juridique détecté
                    doc_keywords = ["Objet :", "Mise en demeure", "Monsieur", "Madame", "À l'attention"]
                    if any(kw in final_answer for kw in doc_keywords):
                        st.download_button(
                            "📥 Télécharger le document Word",
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
# ONGLET 2 : SCANNER UNIVERSEL
# ==========================================
with tab2:
    st.title("🕵️‍♂️ Scanner Universel de Contrat")
    st.caption("Uploadez un contrat PDF — l'IA détecte les clauses illégales selon le droit tunisien.")

    uploaded_file = st.file_uploader("Contrat à auditer (PDF)", type="pdf")

    if uploaded_file:
        col1, col2 = st.columns([1, 3])
        with col1:
            audit_mode = st.radio(
                "Type d'audit",
                ["Rapide (2 500 mots)", "Complet (5 000 mots)"],
                index=0
            )
        with col2:
            st.info(
                "L'audit rapide analyse les premières clauses. "
                "L'audit complet couvre la majorité du contrat."
            )

        if st.button("🚀 Lancer l'Audit", type="primary"):
            with st.spinner("🔍 Analyse du contrat en cours..."):
                pdf_reader = PdfReader(uploaded_file)
                texte_contrat = "".join([p.extract_text() or "" for p in pdf_reader.pages])

                word_limit = 2500 if "Rapide" in audit_mode else 5000
                texte_analyse = texte_contrat[:word_limit]

                # RAG sur les 500 premiers caractères du contrat pour cibler les codes
                docs_loi = retrieve_multi_source(texte_contrat[:600], retriever, top_k_per_source=3)
                contexte_loi = format_context(docs_loi)

                audit_prompt = f"""Tu es un auditeur juridique tunisien expert.

LOIS TUNISIENNES DE RÉFÉRENCE :
{contexte_loi}

CONTRAT À AUDITER :
{texte_analyse}

MISSION :
1. Identifie les clauses potentiellement problématiques ou illégales.
2. Pour chaque clause, cite l'article de loi tunisienne applicable.
3. Présente l'analyse dans un tableau Markdown :

| # | Clause / Extrait | Statut | Article de loi | Explication |
|---|-----------------|--------|---------------|-------------|
| 1 | ... | ✅ Conforme / ⚠️ Risque / ❌ Illégal | Art. X | ... |

4. Ajoute une section "## Recommandations" avec les actions prioritaires.

Si le contrat est conforme, indique-le clairement."""

                res = base_llm.invoke(audit_prompt).content
                thinking, clean_res = extract_thinking_and_answer(res)

                with st.expander("🧠 Réflexion de l'Auditeur"):
                    st.write(thinking)

                st.markdown(clean_res)

                st.download_button(
                    "📥 Télécharger le rapport Word",
                    data=creer_document_word(clean_res),
                    file_name="Rapport_Audit_Contrat.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

                with st.expander("📚 Lois consultées"):
                    st.markdown(format_sources(docs_loi))