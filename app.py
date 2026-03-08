#!/usr/bin/env python3
# app_streamlit_pdf_ollama.py
#
# Streamlit UI: upload PDF -> extract text -> chunk -> retrieve top-K -> chat with Ollama (local)
#
# Install:
#   pip install streamlit pypdf requests
# Run:
#   streamlit run app_streamlit_pdf_ollama.py
#
# Prérequis:
#   Ollama lancé en local (par défaut http://localhost:11434)
#   Exemple: ollama pull llama3.2

import os
import re
from io import BytesIO
from typing import List, Tuple

import requests
import streamlit as st
from pypdf import PdfReader


# ----------------------------
# PDF -> texte
# ----------------------------
def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        t = page.extract_text() or ""
        t = t.replace("\x00", " ")
        pages.append(t)
    return "\n\n".join(pages)


# ----------------------------
# Chunking simple
# ----------------------------
def chunk_text(text: str, max_chars: int = 1800, overlap: int = 200) -> List[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    # Prevent non-progressing windows when overlap is too large.
    overlap = max(0, min(overlap, max_chars - 1))

    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(0, end - overlap)
        if end == n:
            break
    return chunks


# ----------------------------
# Retrieval lexical minimal (sans vecteurs)
# ----------------------------
def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-zA-ZÀ-ÿ0-9]{2,}", s.lower())


def score_chunk(query: str, chunk: str) -> int:
    q = set(tokenize(query))
    c = tokenize(chunk)
    if not q or not c:
        return 0
    hits = sum(1 for w in c if w in q)
    uniq = len(set(c) & q)
    return hits + 2 * uniq


def top_k_chunks(query: str, chunks: List[dict], k: int = 4) -> List[Tuple[int, dict]]:
    scored = [(score_chunk(query, item["text"]), item) for item in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(s, item) for s, item in scored[:k] if s > 0]


# ----------------------------
# Ollama streaming
# ----------------------------
def ollama_chat_stream(model: str, host: str, messages: List[dict]):
    """
    Yield incremental tokens from Ollama /api/chat streaming response.
    """
    url = f"{host.rstrip('/')}/api/chat"
    payload = {"model": model, "stream": True, "messages": messages}
    with requests.post(url, json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            data = None
            try:
                data = requests.models.complexjson.loads(line)  # type: ignore[attr-defined]
            except Exception:
                # fallback json
                import json

                data = json.loads(line)

            # Ollama envoie souvent {"message": {"content": "..."} , ...}
            if "message" in data and data["message"] and "content" in data["message"]:
                chunk = data["message"]["content"]
                if chunk:
                    yield chunk

            # fin
            if data.get("done") is True:
                break


def build_context(question: str, chunks: List[dict], k: int) -> Tuple[str, List[Tuple[int, dict]]]:
    selected = top_k_chunks(question, chunks, k=k)
    if selected:
        ctx = "\n\n".join(
            [
                f"[CHUNK score={s} doc={item['doc_name']} idx={item['chunk_id']}]\n{item['text']}"
                for s, item in selected
            ]
        )
        return ctx, selected
    # fallback: on injecte le début
    fallback_items = chunks[:2]
    fallback_ctx = "\n\n".join(
        [
            f"[CHUNK score=0 doc={item['doc_name']} idx={item['chunk_id']}]\n{item['text']}"
            for item in fallback_items
        ]
    )
    return fallback_ctx, []


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="PDF → Ollama (RAG local)", layout="wide")
st.title("PDF → Q&A avec Ollama (local)")

default_ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

with st.sidebar:
    st.header("Paramètres")
    host = st.text_input("Ollama host", value=default_ollama_host)
    model = st.text_input("Modèle Ollama", value="gpt-oss")
    k = st.slider("Top-K chunks injectés", min_value=1, max_value=10, value=4)
    max_chars = st.slider("Taille chunk (chars)", min_value=600, max_value=4000, value=1800, step=100)
    overlap = st.slider("Overlap (chars)", min_value=0, max_value=800, value=200, step=50)
    show_context = st.checkbox("Afficher le contexte injecté", value=False)

if overlap >= max_chars:
    st.sidebar.warning(
        "Overlap doit être inférieur à la taille de chunk. La valeur sera automatiquement réduite."
    )

uploaded_files = st.file_uploader("Upload un ou plusieurs PDF", type=["pdf"], accept_multiple_files=True)

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "documents" not in st.session_state:
    st.session_state.documents = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of {"role":..., "content":...}

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
                for file in uploaded_files:
                    text = pdf_bytes_to_text(file.read())
                    if not text.strip():
                        continue

                    doc_chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)
                    docs.append(
                        {
                            "name": file.name,
                            "text_chars": len(text),
                            "chunks": len(doc_chunks),
                        }
                    )
                    all_texts.append(f"===== {file.name} =====\n{text}")

                    for idx, ch in enumerate(doc_chunks, start=1):
                        all_chunks.append(
                            {
                                "doc_name": file.name,
                                "chunk_id": idx,
                                "text": ch,
                            }
                        )

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
                    st.markdown(
                        f"- **{doc['name']}** — {doc['chunks']} chunks, {doc['text_chars']} caractères"
                    )
        if st.session_state.pdf_text:
            with st.expander("Aperçu texte extrait", expanded=False):
                st.text_area("Texte", st.session_state.pdf_text[:8000], height=260)

with colR:
    st.subheader("2) Chat")
    if not st.session_state.chunks:
        st.warning("Indexe d’abord les documents (bouton « Indexer les PDF »).")
    else:
        # afficher historique
        for m in st.session_state.chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        question = st.chat_input("Pose une question sur ces documents…")
        if question:
            # push user msg
            st.session_state.chat.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            # construire contexte
            context, selected = build_context(question, st.session_state.chunks, k=k)

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

            # streaming response
            with st.chat_message("assistant"):
                placeholder = st.empty()
                acc = ""

                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ]

                try:
                    for tok in ollama_chat_stream(model=model, host=host, messages=messages):
                        acc += tok
                        placeholder.markdown(acc)
                except requests.exceptions.ConnectionError:
                    st.error(
                        "Impossible de joindre Ollama. Vérifie qu’Ollama tourne et que l’URL est bonne "
                        f"({host})."
                    )
                    acc = ""
                except requests.HTTPError as e:
                    st.error(f"Erreur HTTP Ollama: {e}")
                    acc = ""

            if acc.strip():
                st.session_state.chat.append({"role": "assistant", "content": acc})

        # boutons utilitaires
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
            st.caption("RAG lexical simple (sans embeddings).")
