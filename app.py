#!/usr/bin/env python3

import os

import requests
import streamlit as st

from llm_streams import ollama_chat_stream, openai_compatible_chat_stream
from pageindex_adapter import build_pageindex_chunks
from rag_core import build_context


st.set_page_config(page_title="PDF → Ollama/OpenAI (RAG local)", layout="wide")
st.title("PDF → Q&A avec Ollama / OpenAI-compatible")

default_ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

with st.sidebar:
    st.header("Paramètres")
    st.info("Indexation principale: `pageindex` (package Python installé).")

    provider = st.selectbox("Provider", options=["Ollama", "OpenAI-compatible"], index=0)
    if provider == "Ollama":
        host = st.text_input("Ollama host", value=default_ollama_host)
        model = st.text_input("Modèle", value="gpt-oss", key="model_ollama")
        api_key = ""
        openai_base_url = ""
    else:
        openai_base_url = st.text_input(
            "OpenAI base URL",
            value=os.getenv("OPENAI_API_BASE", os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")),
            help="Exemples: https://api.openai.com ou URL d'un serveur compatible OpenAI.",
        )
        api_key = st.text_input(
            "OPENAI API key",
            value=os.getenv("OPENAI_API_KEY", os.getenv("CHATGPT_API_KEY", "")),
            type="password",
            help="Clé API utilisée avec l'en-tête Bearer.",
        )
        model = st.text_input(
            "Modèle",
            value=os.getenv("OPENAI_API_MODEL", os.getenv("OPENAI_MODEL", "gpt-oss")),
            key="model_openai",
        )
        host = ""

    k = st.slider("Top-K chunks injectés", min_value=1, max_value=10, value=4)
    show_context = st.checkbox("Afficher le contexte injecté", value=False)

uploaded_files = st.file_uploader("Upload un ou plusieurs PDF", type=["pdf"], accept_multiple_files=True)

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "documents" not in st.session_state:
    st.session_state.documents = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chat" not in st.session_state:
    st.session_state.chat = []

colL, colR = st.columns([1, 1], gap="large")

with colL:
    st.subheader("1) Document")
    if not uploaded_files:
        st.info("Charge un ou plusieurs PDF pour commencer.")
    else:
        st.caption(f"{len(uploaded_files)} fichier(s) sélectionné(s).")
        if st.button("Indexer les PDF", type="primary"):
            all_texts = []
            all_chunks = []
            docs = []

            with st.spinner("Extraction du texte…"):
                if provider != "OpenAI-compatible":
                    st.error(
                        "Le mode PageIndex nécessite un endpoint OpenAI-compatible "
                        "(provider = OpenAI-compatible)."
                    )
                elif not api_key.strip():
                    st.error("Le mode PageIndex nécessite une OPENAI API key.")
                else:
                    try:
                        docs, all_chunks, all_texts, warnings = build_pageindex_chunks(
                            uploaded_files=uploaded_files,
                            model=model,
                            api_key=api_key,
                            openai_base_url=openai_base_url,
                        )
                        for w in warnings:
                            st.warning(w)
                    except Exception as e:
                        st.error(f"Erreur PageIndex: {type(e).__name__}: {e}")

            if not all_chunks:
                st.error(
                    "Aucun texte exploitable trouvé. Probable PDF scanné (images) → il faut un OCR avant. "
                    "Sinon, essaie d'autres PDF."
                )
            else:
                st.session_state.pdf_text = "\n\n".join(all_texts)
                st.session_state.documents = docs
                st.session_state.chunks = all_chunks
                st.success(
                    f"OK. {len(st.session_state.documents)} document(s) indexé(s), "
                    f"{len(st.session_state.chunks)} chunks générés."
                )

        if st.session_state.documents:
            with st.expander("Documents indexés", expanded=False):
                for doc in st.session_state.documents:
                    st.markdown(f"- **{doc['name']}** — {doc['chunks']} chunks, {doc['text_chars']} caractères")
        if st.session_state.pdf_text:
            with st.expander("Aperçu texte extrait", expanded=False):
                st.text_area("Texte", st.session_state.pdf_text[:8000], height=260)

with colR:
    st.subheader("2) Chat")
    if not st.session_state.chunks:
        st.warning("Indexe d’abord les documents (bouton « Indexer les PDF »).")
    else:
        for m in st.session_state.chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        question = st.chat_input("Pose une question sur ces documents…")
        if question:
            st.session_state.chat.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            context, _selected = build_context(question, st.session_state.chunks, k=k)
            system = (
                "Tu es un assistant qui répond uniquement à partir du CONTEXTE fourni. "
                "Si le contexte ne permet pas de répondre, dis-le clairement et propose ce qu'il manque."
            )
            user_prompt = f"""CONTEXTE (extraits du PDF) :
{context}

QUESTION :
{question}

Réponds en français, de façon concise et factuelle. Appuie-toi uniquement sur le contexte."""

            if show_context:
                with st.expander("Contexte injecté au modèle", expanded=False):
                    st.code(context[:20000])

            with st.chat_message("assistant"):
                placeholder = st.empty()
                acc = ""
                messages = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]

                try:
                    if provider == "Ollama":
                        stream = ollama_chat_stream(model=model, host=host, messages=messages)
                    else:
                        stream = openai_compatible_chat_stream(
                            model=model,
                            base_url=openai_base_url,
                            api_key=api_key,
                            messages=messages,
                        )

                    for tok in stream:
                        acc += tok
                        placeholder.markdown(acc)
                except requests.exceptions.ConnectionError:
                    if provider == "Ollama":
                        st.error(
                            "Impossible de joindre Ollama. Vérifie qu’Ollama tourne et que l’URL est bonne "
                            f"({host})."
                        )
                    else:
                        st.error(
                            "Impossible de joindre le endpoint OpenAI-compatible. Vérifie l'URL de base "
                            f"({openai_base_url})."
                        )
                except requests.HTTPError as e:
                    st.error(f"Erreur HTTP provider ({provider}): {e}")
                except ValueError as e:
                    st.error(str(e))

            if acc.strip():
                st.session_state.chat.append({"role": "assistant", "content": acc})

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Vider le chat"):
                st.session_state.chat = []
                st.rerun()
        with c2:
            if st.button("Réinitialiser PDF + chat"):
                st.session_state.pdf_text = ""
                st.session_state.chunks = []
                st.session_state.chat = []
                st.rerun()
        with c3:
            st.caption("Indexation: PageIndex uniquement, chat Ollama/OpenAI API-like.")
