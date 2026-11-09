"""
app.py
------
Chatbot RAG (Retrieval-Augmented Generation) 100% fonctionnel :
- LLM gratuit via l'API Groq (ultra rapide, quota gratuit généreux)
- Embeddings locaux gratuits (sentence-transformers, pas de clé API)
- Index vectoriel FAISS en mémoire
- Interface Streamlit

Lancement : streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from rag_utils import (
    load_document,
    split_documents,
    build_vectorstore,
    retrieve_relevant_chunks,
    format_context,
)

# ---------------------------------------------------------------------------
# Configuration générale
# ---------------------------------------------------------------------------
load_dotenv()

st.set_page_config(page_title="RAG Chatbot (Groq - Gratuit)", page_icon="💬", layout="wide")

GROQ_MODEL = "openai/gpt-oss-20b"  # modèle gratuit et rapide sur Groq

SYSTEM_PROMPT = """Tu es un assistant qui répond UNIQUEMENT à partir du contexte fourni.
Si la réponse ne se trouve pas dans le contexte, dis clairement que tu ne sais pas
à partir des documents fournis. Ne pas inventer d'information.
Réponds dans la même langue que la question posée."""

USER_PROMPT_TEMPLATE = """Contexte extrait des documents :
---------------------
{context}
---------------------

Question : {question}

Réponse (en te basant uniquement sur le contexte ci-dessus) :"""


# ---------------------------------------------------------------------------
# État de session
# ---------------------------------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


# ---------------------------------------------------------------------------
# Barre latérale : clé API + upload de documents
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuration")

    api_key = st.text_input(
        "Clé API Groq (gratuite)",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Obtiens une clé gratuite sur https://console.groq.com/keys",
    )

    st.markdown("---")
    st.subheader("📄 Documents")

    uploaded_files = st.file_uploader(
        "Ajoute un ou plusieurs fichiers (PDF ou TXT)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("🔨 Construire / mettre à jour l'index", use_container_width=True):
        if not uploaded_files:
            st.warning("Ajoute au moins un fichier avant de construire l'index.")
        else:
            with st.spinner("Lecture, découpage et indexation des documents..."):
                all_docs = []
                for f in uploaded_files:
                    all_docs.extend(load_document(f))

                chunks = split_documents(all_docs)
                st.session_state.vectorstore = build_vectorstore(chunks)
                st.session_state.indexed_files = [f.name for f in uploaded_files]

            st.success(f"Index créé : {len(chunks)} chunks à partir de {len(uploaded_files)} fichier(s).")

    if st.session_state.indexed_files:
        st.markdown("**Fichiers indexés :**")
        for name in st.session_state.indexed_files:
            st.markdown(f"- {name}")

    st.markdown("---")
    if st.button("🗑️ Réinitialiser la conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Zone principale : chat
# ---------------------------------------------------------------------------
st.title("💬 Chatbot RAG — LLM gratuit (Groq)")
st.caption(
    "Uploade tes documents dans la barre latérale, construis l'index, "
    "puis pose tes questions. Les réponses sont générées uniquement à "
    "partir du contenu de tes documents."
)

# Affiche l'historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📚 Sources utilisées"):
                st.markdown(msg["sources"])

# Saisie utilisateur
question = st.chat_input("Pose ta question sur les documents...")

if question:
    if not api_key:
        st.error("Merci de renseigner ta clé API Groq dans la barre latérale (gratuite sur console.groq.com).")
        st.stop()

    if st.session_state.vectorstore is None:
        st.error("Merci d'uploader des documents et de construire l'index d'abord.")
        st.stop()

    # Affiche la question de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Récupération des chunks pertinents (retrieval)
    with st.spinner("Recherche des passages pertinents..."):
        relevant_chunks = retrieve_relevant_chunks(st.session_state.vectorstore, question, k=4)
        context = format_context(relevant_chunks)

    # Génération de la réponse (augmented generation) via Groq
    with st.chat_message("assistant"):
        try:
            llm = ChatGroq(
                api_key=api_key,
                model=GROQ_MODEL,
                temperature=0.2,
            )

            prompt = [
                ("system", SYSTEM_PROMPT),
                ("user", USER_PROMPT_TEMPLATE.format(context=context, question=question)),
            ]

            with st.spinner("Génération de la réponse..."):
                response = llm.invoke(prompt)
                answer = response.content

            st.markdown(answer)

            sources_md = "\n\n".join(
                f"**Extrait {i+1}** — *{c.metadata.get('source', 'document')}*\n\n> {c.page_content[:300]}..."
                for i, c in enumerate(relevant_chunks)
            )
            with st.expander("📚 Sources utilisées"):
                st.markdown(sources_md)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources_md}
            )

        except Exception as e:
            st.error(f"Erreur lors de l'appel à l'API Groq : {e}")