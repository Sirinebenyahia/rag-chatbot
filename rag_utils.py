"""
rag_utils.py
------------
Fonctions utilitaires pour le pipeline RAG :
1. Chargement de documents (PDF / TXT)
2. Découpage en chunks
3. Génération d'embeddings (modèle local gratuit, via sentence-transformers)
4. Construction et interrogation d'un index vectoriel FAISS
"""

import os
import tempfile
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Modèle d'embeddings gratuit, léger, tourne en local sur CPU (pas de clé API nécessaire)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_document(uploaded_file) -> List[Document]:
    """
    Charge un fichier uploadé via Streamlit (PDF ou TXT) et retourne une liste
    de Documents LangChain.
    """
    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    # Streamlit fournit un objet en mémoire : on l'écrit temporairement sur disque
    # car les loaders LangChain attendent un chemin de fichier.
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix in (".txt", ".md"):
            loader = TextLoader(tmp_path, encoding="utf-8")
        else:
            raise ValueError(f"Format de fichier non supporté : {suffix}")

        documents = loader.load()
    finally:
        os.remove(tmp_path)

    # On ajoute le nom du fichier d'origine dans les métadonnées (utile pour les sources)
    for doc in documents:
        doc.metadata["source"] = uploaded_file.name

    return documents


def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Document]:
    """
    Découpe les documents en chunks plus petits, avec un léger chevauchement
    pour préserver le contexte entre les morceaux.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def get_embeddings_model():
    """
    Retourne le modèle d'embeddings local (gratuit, pas de clé API).
    Le modèle est téléchargé automatiquement au premier lancement
    (environ 80 Mo) puis mis en cache.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks: List[Document]) -> FAISS:
    """
    Construit un index vectoriel FAISS en mémoire à partir des chunks fournis.
    """
    embeddings = get_embeddings_model()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def retrieve_relevant_chunks(vectorstore: FAISS, query: str, k: int = 4) -> List[Document]:
    """
    Recherche les k chunks les plus pertinents pour une question donnée.
    """
    return vectorstore.similarity_search(query, k=k)


def format_context(chunks: List[Document]) -> str:
    """
    Formate les chunks récupérés en un bloc de texte à insérer dans le prompt,
    avec indication de la source de chaque extrait.
    """
    formatted = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "document")
        page = chunk.metadata.get("page")
        page_info = f", page {page + 1}" if page is not None else ""
        formatted.append(f"[Extrait {i} - {source}{page_info}]\n{chunk.page_content}")
    return "\n\n".join(formatted)
