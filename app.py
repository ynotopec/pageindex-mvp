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
import importlib
import sys
from io import BytesIO
from types import SimpleNamespace
from typing import Any, Generator, List, Tuple

import requests
import streamlit as st
import tiktoken
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
def ollama_chat_stream(model: str, host: str, messages: List[dict]) -> Generator[str, None, None]:
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


def openai_compatible_chat_stream(
    model: str,
    base_url: str,
    api_key: str,
    messages: List[dict],
) -> Generator[str, None, None]:
    """
    Yield incremental tokens from OpenAI-compatible /v1/chat/completions streaming response.
    """
    if not api_key.strip():
        raise ValueError("OPENAI API key manquante")

    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {"model": model, "stream": True, "messages": messages}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    with requests.post(url, json=payload, headers=headers, stream=True, timeout=300) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue

            line = raw.strip()
            if not line.startswith("data:"):
                continue

            data_part = line[5:].strip()
            if data_part == "[DONE]":
                break

            try:
                data = requests.models.complexjson.loads(data_part)  # type: ignore[attr-defined]
            except Exception:
                import json

                data = json.loads(data_part)

            choices = data.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            chunk = delta.get("content")
            if chunk:
                yield chunk


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


def load_pageindex_lib() -> Tuple[Any, Any, Any]:
    """
    Load the installed PageIndex library package and return:
    - page_index_main function
    - config factory
    - pageindex.utils module
    """
    # Prefer installed package. Some versions expose `page_index_main` at package root,
    # others only in `pageindex.page_index`.
    page_index_main = None
    config_factory = None
    pageindex_utils = None

    try:
        pkg = importlib.import_module("pageindex")
        page_index_main = getattr(pkg, "page_index_main", None)
    except Exception:
        pkg = None

    if page_index_main is None:
        try:
            mod = importlib.import_module("pageindex.page_index")
            page_index_main = getattr(mod, "page_index_main", None)
        except Exception:
            page_index_main = None

    try:
        pageindex_utils = importlib.import_module("pageindex.utils")
        config_factory = getattr(pageindex_utils, "config", None)
    except Exception:
        pageindex_utils = None
        config_factory = None

    # Fallback for packages without exported `config`
    if config_factory is None:
        config_factory = SimpleNamespace

    # Last-resort fallback to bundled legacy source.
    # IMPORTANT: we cannot re-import "pageindex" here because it may already
    # point to the installed package in sys.modules.
    if page_index_main is None:
        old_root = os.path.join(os.path.dirname(__file__), "old")
        if old_root not in sys.path:
            sys.path.insert(0, old_root)

        # Ensure we import legacy `pageindex` and not the already-cached installed one.
        cached_pageindex_modules = {
            name: module
            for name, module in list(sys.modules.items())
            if name == "pageindex" or name.startswith("pageindex.")
        }
        for name in cached_pageindex_modules:
            sys.modules.pop(name, None)

        try:
            legacy_mod = importlib.import_module("pageindex.page_index")
            page_index_main = getattr(legacy_mod, "page_index_main", None)
            if pageindex_utils is None:
                pageindex_utils = importlib.import_module("pageindex.utils")
        finally:
            # Restore previously cached package modules so rest of app behavior
            # remains stable after fallback probing.
            for name, module in cached_pageindex_modules.items():
                sys.modules.setdefault(name, module)

        if pageindex_utils is not None and getattr(pageindex_utils, "config", None) is not None:
            config_factory = pageindex_utils.config

    if page_index_main is None:
        raise ImportError(
            "`page_index_main` introuvable dans le package `pageindex` installé et fallback legacy indisponible."
        )

    return page_index_main, config_factory, pageindex_utils


def build_pageindex_chunks(
    uploaded_files: List[Any],
    model: str,
    api_key: str,
    openai_base_url: str,
    lexical_max_chars: int,
    lexical_overlap: int,
) -> Tuple[List[dict], List[dict], List[str]]:
    try:
        page_index_main, config, pageindex_utils = load_pageindex_lib()
    except Exception as e:
        raise RuntimeError(
            "Impossible d'importer la librairie PageIndex (package `pageindex`). "
            "Vérifie que `pageindex` est bien installé depuis requirements.txt."
        ) from e

    os.environ["CHATGPT_API_KEY"] = api_key
    if openai_base_url.strip():
        os.environ["OPENAI_BASE_URL"] = openai_base_url

    # PageIndex relies on tiktoken.encoding_for_model(model). Some model names
    # used in local runtimes (ex: gpt-oss) are unknown to tiktoken.
    # Fallback to a tokenizer-compatible OpenAI model name to prevent KeyError.
    pageindex_model = model
    try:
        tiktoken.encoding_for_model(pageindex_model)
    except KeyError:
        pageindex_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Patch runtime client construction to inject API key/base URL when the
    # package exposes an `openai` module reference (legacy-compatible behavior).
    if pageindex_utils is not None and hasattr(pageindex_utils, "openai"):
        original_sync_openai = pageindex_utils.openai.OpenAI
        original_async_openai = pageindex_utils.openai.AsyncOpenAI

        def patched_sync_openai(*args, **kwargs):
            if not kwargs.get("api_key"):
                kwargs["api_key"] = api_key
            if openai_base_url.strip() and not kwargs.get("base_url"):
                kwargs["base_url"] = openai_base_url
            return original_sync_openai(*args, **kwargs)

        def patched_async_openai(*args, **kwargs):
            if not kwargs.get("api_key"):
                kwargs["api_key"] = api_key
            if openai_base_url.strip() and not kwargs.get("base_url"):
                kwargs["base_url"] = openai_base_url
            return original_async_openai(*args, **kwargs)

        pageindex_utils.openai.OpenAI = patched_sync_openai
        pageindex_utils.openai.AsyncOpenAI = patched_async_openai

    opt = config(
        model=pageindex_model,
        toc_check_page_num=20,
        max_page_num_each_node=10,
        max_token_num_each_node=20000,
        if_add_node_id="yes",
        if_add_node_summary="yes",
        if_add_doc_description="no",
        if_add_node_text="yes",
    )

    docs = []
    all_chunks = []
    all_texts = []

    for file in uploaded_files:
        raw_pdf = file.read()
        if not raw_pdf:
            continue

        node_chunks = []
        try:
            result = page_index_main(BytesIO(raw_pdf), opt)
            structure = result.get("structure", [])

            def visit(nodes: List[dict]) -> None:
                for node in nodes:
                    node_text = (node.get("text") or "").strip()
                    node_summary = (node.get("summary") or "").strip()
                    if node_text:
                        node_chunks.append(node_text)
                    elif node_summary:
                        node_chunks.append(node_summary)
                    children = node.get("nodes") or []
                    if children:
                        visit(children)

            visit(structure)
        except Exception as e:
            # Guard against brittle upstream parser failures (ex: KeyError on toc_detected)
            # and keep the app usable by falling back to lexical chunking for this document.
            fallback_text = pdf_bytes_to_text(raw_pdf)
            node_chunks = chunk_text(
                fallback_text,
                max_chars=lexical_max_chars,
                overlap=lexical_overlap,
            )
            if node_chunks:
                st.warning(
                    f"PageIndex a échoué pour '{file.name}' ({type(e).__name__}: {e}). "
                    "Fallback automatique vers chunking lexical."
                )
            else:
                st.warning(
                    f"PageIndex a échoué pour '{file.name}' ({type(e).__name__}: {e}) "
                    "et le fallback lexical n'a extrait aucun texte."
                )

        docs.append(
            {
                "name": file.name,
                "text_chars": sum(len(c) for c in node_chunks),
                "chunks": len(node_chunks),
            }
        )

        all_texts.append(f"===== {file.name} =====\n" + "\n\n".join(node_chunks))

        for idx, ch in enumerate(node_chunks, start=1):
            all_chunks.append(
                {
                    "doc_name": file.name,
                    "chunk_id": idx,
                    "text": ch,
                }
            )

    return docs, all_chunks, all_texts


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="PDF → Ollama (RAG local)", layout="wide")
st.title("PDF → Q&A avec Ollama (local)")

default_ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

with st.sidebar:
    st.header("Paramètres")
    indexing_backend = st.selectbox(
        "Indexation",
        options=["Chunking lexical (rapide)", "PageIndex package (structure + raisonnement)"],
        index=0,
    )
    if indexing_backend.startswith("PageIndex"):
        st.info("Librairie utilisée: `pageindex` (package Python installé).")
    provider = st.selectbox("Provider", options=["Ollama", "OpenAI-compatible"], index=0)
    if provider == "Ollama":
        host = st.text_input("Ollama host", value=default_ollama_host)
        model = st.text_input("Modèle", value="gpt-oss", key="model_ollama")
        api_key = ""
        openai_base_url = ""
    else:
        openai_base_url = st.text_input(
            "OpenAI base URL",
            value=os.getenv("OPENAI_BASE_URL", "http://localhost:8000"),
            help="Exemples: https://api.openai.com ou URL d'un serveur compatible OpenAI.",
        )
        api_key = st.text_input(
            "OPENAI API key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="Clé API utilisée avec l'en-tête Bearer.",
        )
        model = st.text_input(
            "Modèle",
            value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            key="model_openai",
        )
        host = ""
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
                if indexing_backend.startswith("Chunking"):
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
                else:
                    if provider != "OpenAI-compatible":
                        st.error(
                            "Le mode PageIndex package nécessite un endpoint OpenAI-compatible "
                            "(provider = OpenAI-compatible)."
                        )
                    elif not api_key.strip():
                        st.error("Le mode PageIndex package nécessite une OPENAI API key.")
                    else:
                        docs, all_chunks, all_texts = build_pageindex_chunks(
                            uploaded_files=uploaded_files,
                            model=model,
                            api_key=api_key,
                            openai_base_url=openai_base_url,
                            lexical_max_chars=max_chars,
                            lexical_overlap=overlap,
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
                    acc = ""
                except requests.HTTPError as e:
                    st.error(f"Erreur HTTP provider ({provider}): {e}")
                    acc = ""
                except ValueError as e:
                    st.error(str(e))
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
            st.caption(
                "Backends: chunking lexical rapide ou PageIndex package (structure), "
                "avec chat Ollama/OpenAI API-like."
            )
