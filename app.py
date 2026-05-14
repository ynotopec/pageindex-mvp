#!/usr/bin/env python3
# app_streamlit_pdf_ollama.py
#
# Streamlit UI: upload PDF -> extract text -> build a PageIndex-inspired tree -> retrieve top-K -> chat
#
# Install:
#   pip install streamlit pymupdf pypdf requests openai
# Run:
#   streamlit run app_streamlit_pdf_ollama.py
#
# Prérequis:
#   Ollama lancé en local (par défaut http://localhost:11434)
#   Exemple: ollama pull llama3.2

import json
import os
import re
from io import BytesIO
from typing import Iterable, List, Tuple

import fitz  # PyMuPDF, also used by the PageIndex project for PDF parsing.
import requests
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader


# ----------------------------
# PDF -> pages texte
# ----------------------------
def pdf_bytes_to_pages(pdf_bytes: bytes) -> List[dict]:
    """Extract page-aware text with PyMuPDF, then fall back to pypdf if needed."""
    pages: List[dict] = []
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page_index, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                text = text.replace("\x00", " ").strip()
                pages.append({"page": page_index, "text": text})
    except Exception:
        # Keep the app usable for PDFs that PyMuPDF cannot open but pypdf can parse.
        reader = PdfReader(BytesIO(pdf_bytes))
        pages = []
        for page_index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.replace("\x00", " ").strip()
            pages.append({"page": page_index, "text": text})

    return pages


def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    return "\n\n".join(page["text"] for page in pdf_bytes_to_pages(pdf_bytes))


# ----------------------------
# Nettoyage, chunking fallback et scoring lexical
# ----------------------------
def normalize_text(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, max_chars: int = 1800, overlap: int = 200) -> List[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    # Prevent non-progressing windows when overlap is too large.
    overlap = max(0, min(overlap, max_chars - 1))

    text = normalize_text(text)

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


def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-zA-ZÀ-ÿ0-9]{2,}", s.lower())


def text_excerpt(text: str, max_chars: int = 360) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"


def score_text(query_terms: set, text: str, weight: int = 1) -> int:
    words = tokenize(text)
    if not query_terms or not words:
        return 0
    hits = sum(1 for word in words if word in query_terms)
    uniq = len(set(words) & query_terms)
    return weight * (hits + 2 * uniq)


def score_item(query: str, item: dict) -> int:
    q = set(tokenize(query))
    score = score_text(q, item.get("title", ""), weight=8)
    score += score_text(q, item.get("summary", ""), weight=4)
    score += score_text(q, item.get("text", ""), weight=1)

    query_norm = " ".join(tokenize(query))
    haystack = " ".join(
        [item.get("title", ""), item.get("summary", ""), item.get("text", "")]
    )
    haystack_norm = " ".join(tokenize(haystack))
    if query_norm and query_norm in haystack_norm:
        score += 25
    return score


def top_k_items(query: str, items: List[dict], k: int = 4) -> List[Tuple[int, dict]]:
    scored = [(score_item(query, item), item) for item in items]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(s, item) for s, item in scored[:k] if s > 0]


# ----------------------------
# PageIndex-inspired hierarchical tree
# ----------------------------
HEADING_PATTERNS = [
    re.compile(
        r"^(?:chapter|chapitre|section|article|appendix|annexe|part|partie)\s+[\wIVXLCDM.-]+\b",
        re.I,
    ),
    re.compile(
        r"^(?:\d{1,3}(?:\.\d{1,3}){0,5}|[IVXLCDM]{1,8})(?:[.)\-–:]|\s+)\s*\S", re.I
    ),
    re.compile(r"^[A-ZÀ-Ÿ0-9][A-ZÀ-Ÿ0-9 '&/.,:;()\-–]{7,90}$"),
]


def is_heading(line: str) -> bool:
    line = re.sub(r"\s+", " ", line).strip()
    if not line or len(line) > 140:
        return False
    if line.endswith(".") and len(line.split()) > 8:
        return False
    return any(pattern.match(line) for pattern in HEADING_PATTERNS)


def heading_level(title: str) -> int:
    clean = title.strip()
    numbered = re.match(r"^(\d{1,3}(?:\.\d{1,3})*)", clean)
    if numbered:
        return min(6, 1 + numbered.group(1).count("."))
    if re.match(r"^(?:chapter|chapitre|part|partie)\b", clean, re.I):
        return 1
    if re.match(r"^(?:section|article|appendix|annexe)\b", clean, re.I):
        return 2
    return 2


def new_node(node_id: str, title: str, level: int, page: int) -> dict:
    return {
        "node_id": node_id,
        "title": title,
        "level": level,
        "start_page": page,
        "end_page": page,
        "summary": "",
        "text": "",
        "nodes": [],
    }


def append_line_to_node(node: dict, line: str, page: int) -> None:
    line = line.strip()
    if not line:
        return
    node["text"] = f"{node['text']}\n{line}".strip()
    node["end_page"] = max(node["end_page"], page)


def iter_page_lines(pages: Iterable[dict]) -> Iterable[Tuple[int, str]]:
    for page in pages:
        for line in page["text"].splitlines():
            clean = re.sub(r"\s+", " ", line).strip()
            if clean:
                yield page["page"], clean


def add_summaries(node: dict) -> None:
    own_text = node.get("text", "")
    child_summaries = " ".join(
        child.get("summary", "") for child in node.get("nodes", [])
    )
    node["summary"] = text_excerpt(
        own_text or child_summaries or node.get("title", ""), 420
    )
    for child in node.get("nodes", []):
        add_summaries(child)
    if not own_text and node.get("nodes"):
        node["summary"] = text_excerpt(
            " ".join(child.get("summary", "") for child in node["nodes"]), 420
        )


def prune_empty_nodes(node: dict) -> bool:
    node["nodes"] = [
        child for child in node.get("nodes", []) if prune_empty_nodes(child)
    ]
    return bool(node.get("text", "").strip() or node.get("nodes"))


def build_pageindex_tree(
    doc_name: str, pages: List[dict], max_chars_per_node: int
) -> dict:
    """Build a lightweight, local tree inspired by PageIndex's table-of-contents index."""
    root = new_node("root", doc_name, 0, pages[0]["page"] if pages else 1)
    root["doc_name"] = doc_name
    stack = [root]
    node_counter = 0
    current = root

    for page, line in iter_page_lines(pages):
        if is_heading(line):
            node_counter += 1
            level = heading_level(line)
            node = new_node(f"{node_counter:04d}", line, level, page)
            while len(stack) > 1 and stack[-1]["level"] >= level:
                stack.pop()
            stack[-1]["nodes"].append(node)
            stack.append(node)
            current = node
            continue
        append_line_to_node(current, line, page)

    if root["text"].strip() and root["nodes"]:
        preamble = new_node("0000", "Préambule", 1, root["start_page"])
        preamble["end_page"] = root["end_page"]
        preamble["text"] = root["text"]
        root["text"] = ""
        root["nodes"].insert(0, preamble)

    # Fallback for documents where no heading-like lines were detected.
    if not root["nodes"]:
        full_text = normalize_text("\n\n".join(page["text"] for page in pages))
        for idx, chunk in enumerate(
            chunk_text(full_text, max_chars=max_chars_per_node, overlap=200), start=1
        ):
            node = new_node(f"{idx:04d}", f"Extrait {idx}", 1, 1)
            node["text"] = chunk
            root["nodes"].append(node)
    else:
        # Split very large leaf sections into smaller page-aware child extracts.
        split_large_leaf_nodes(root, max_chars_per_node)

    prune_empty_nodes(root)
    add_summaries(root)
    return root


def split_large_leaf_nodes(node: dict, max_chars_per_node: int) -> None:
    for child in list(node.get("nodes", [])):
        split_large_leaf_nodes(child, max_chars_per_node)

    if node.get("nodes") or len(node.get("text", "")) <= max_chars_per_node:
        return

    chunks = chunk_text(node["text"], max_chars=max_chars_per_node, overlap=200)
    node["text"] = ""
    node["nodes"] = []
    for idx, chunk in enumerate(chunks, start=1):
        child = new_node(
            f"{node['node_id']}.{idx}",
            f"{node['title']} — extrait {idx}",
            node["level"] + 1,
            node["start_page"],
        )
        child["end_page"] = node["end_page"]
        child["text"] = chunk
        node["nodes"].append(child)


def flatten_tree(node: dict, doc_name: str, path: str = "") -> List[dict]:
    title = node.get("title", "")
    next_path = title if not path else f"{path} > {title}"
    items = []
    if node.get("node_id") != "root":
        inherited_text = node.get("text", "") or "\n".join(
            child.get("summary", "") for child in node.get("nodes", [])
        )
        items.append(
            {
                "doc_name": doc_name,
                "node_id": node["node_id"],
                "title": title,
                "path": next_path,
                "start_page": node.get("start_page"),
                "end_page": node.get("end_page"),
                "summary": node.get("summary", ""),
                "text": inherited_text,
            }
        )
    for child in node.get("nodes", []):
        items.extend(flatten_tree(child, doc_name, next_path))
    return items


def pages_label(item: dict) -> str:
    start = item.get("start_page")
    end = item.get("end_page")
    if not start and not end:
        return "pages n/a"
    if start == end or not end:
        return f"page {start}"
    return f"pages {start}-{end}"


# ----------------------------
# LLM streaming
# ----------------------------
def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def ollama_chat_stream(
    model: str, host: str, messages: List[dict], temperature: float, max_tokens: int
):
    """
    Yield incremental tokens from Ollama /api/chat streaming response.
    """
    url = f"{host.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "stream": True,
        "messages": messages,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    with requests.post(url, json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
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
    temperature: float,
    max_tokens: int,
):
    """Yield tokens from the popular OpenAI-compatible Chat Completions API."""
    client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/"))
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for event in stream:
        if not event.choices:
            continue
        delta = event.choices[0].delta.content
        if delta:
            yield delta


def chat_stream(
    provider: str,
    messages: List[dict],
    ollama_host: str,
    ollama_model: str,
    openai_base_url: str,
    openai_api_key: str,
    openai_model: str,
    temperature: float,
    max_tokens: int,
):
    if provider == "openai_compatible":
        if not openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when LLM_PROVIDER=openai_compatible."
            )
        yield from openai_compatible_chat_stream(
            model=openai_model,
            base_url=openai_base_url,
            api_key=openai_api_key,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return

    yield from ollama_chat_stream(
        model=ollama_model,
        host=ollama_host,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def build_context(
    question: str, items: List[dict], k: int
) -> Tuple[str, List[Tuple[int, dict]]]:
    selected = top_k_items(question, items, k=k)
    if selected:
        ctx = "\n\n".join(
            [
                (
                    f"[SECTION score={score} doc={item['doc_name']} node={item['node_id']} "
                    f"{pages_label(item)}]\n"
                    f"Chemin: {item['path']}\n"
                    f"Résumé: {item['summary']}\n"
                    f"Texte:\n{item['text']}"
                )
                for score, item in selected
            ]
        )
        return ctx, selected
    # fallback: on injecte le début de l'arbre plat
    fallback_items = items[:2]
    fallback_ctx = "\n\n".join(
        [
            (
                f"[SECTION score=0 doc={item['doc_name']} node={item['node_id']} {pages_label(item)}]\n"
                f"Chemin: {item['path']}\n"
                f"Résumé: {item['summary']}\n"
                f"Texte:\n{item['text']}"
            )
            for item in fallback_items
        ]
    )
    return fallback_ctx, []


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="PDF → Q&A (RAG)", layout="wide")
st.title("PDF → Q&A (RAG local + API)")
st.caption(
    "Index hiérarchique inspiré de PageIndex: extraction par pages, sections naturelles, références page/section."
)

default_provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
default_ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
default_ollama_model = os.getenv("OLLAMA_MODEL", "gpt-oss")
default_openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
default_openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
default_openai_api_key = os.getenv("OPENAI_API_KEY", "")
default_temperature = env_float("LLM_TEMPERATURE", 0.0)
default_max_tokens = env_int("LLM_MAX_TOKENS", 1024)

with st.sidebar:
    st.header("Paramètres")
    provider_options = ["ollama", "openai_compatible"]
    provider_index = (
        provider_options.index(default_provider)
        if default_provider in provider_options
        else 0
    )
    provider = st.selectbox("Provider LLM", provider_options, index=provider_index)

    with st.expander("Ollama", expanded=(provider == "ollama")):
        ollama_host = st.text_input("Ollama host", value=default_ollama_host)
        ollama_model = st.text_input("Modèle Ollama", value=default_ollama_model)

    with st.expander(
        "API OpenAI-compatible", expanded=(provider == "openai_compatible")
    ):
        openai_base_url = st.text_input("Base URL", value=default_openai_base_url)
        openai_model = st.text_input("Modèle API", value=default_openai_model)
        openai_api_key = st.text_input(
            "Token API", value=default_openai_api_key, type="password"
        )

    k = st.slider(
        "Top-K sections injectées", min_value=1, max_value=10, value=env_int("TOP_K", 4)
    )
    max_chars = st.slider(
        "Taille max d'une section/extrait",
        min_value=600,
        max_value=8000,
        value=env_int("CHUNK_MAX_CHARS", 2400),
        step=100,
    )
    temperature = st.slider(
        "Température", min_value=0.0, max_value=2.0, value=default_temperature, step=0.1
    )
    max_tokens = st.slider(
        "Max tokens", min_value=128, max_value=8192, value=default_max_tokens, step=128
    )
    show_context = st.checkbox("Afficher le contexte injecté", value=False)
    show_tree = st.checkbox("Afficher l'arbre d'index", value=False)

uploaded_files = st.file_uploader(
    "Upload un ou plusieurs PDF", type=["pdf"], accept_multiple_files=True
)

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "documents" not in st.session_state:
    st.session_state.documents = []
if "chunks" not in st.session_state:
    st.session_state.chunks = (
        []
    )  # Backward-compatible name: flattened tree retrieval items.
if "trees" not in st.session_state:
    st.session_state.trees = []
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
            all_items = []
            docs = []
            trees = []

            with st.spinner("Extraction par pages et construction de l'arbre…"):
                for file in uploaded_files:
                    pages = pdf_bytes_to_pages(file.read())
                    pages = [page for page in pages if page["text"].strip()]
                    if not pages:
                        continue

                    tree = build_pageindex_tree(
                        file.name, pages, max_chars_per_node=max_chars
                    )
                    items = flatten_tree(tree, file.name)
                    text = "\n\n".join(
                        f"--- page {page['page']} ---\n{page['text']}" for page in pages
                    )
                    docs.append(
                        {
                            "name": file.name,
                            "text_chars": len(text),
                            "pages": len(pages),
                            "sections": len(items),
                        }
                    )
                    all_texts.append(f"===== {file.name} =====\n{text}")
                    trees.append(tree)
                    all_items.extend(items)

            if not all_items:
                st.error(
                    "Aucun texte exploitable trouvé. Probable PDF scanné (images) → il faut un OCR avant. "
                    "Sinon, essaie d'autres PDF."
                )
            else:
                st.session_state.pdf_text = "\n\n".join(all_texts)
                st.session_state.documents = docs
                st.session_state.chunks = all_items
                st.session_state.trees = trees
                st.success(
                    f"OK. {len(st.session_state.documents)} document(s) indexé(s), "
                    f"{len(st.session_state.chunks)} section(s)/extrait(s) généré(s)."
                )

        if st.session_state.documents:
            with st.expander("Documents indexés", expanded=False):
                for doc in st.session_state.documents:
                    st.markdown(
                        f"- **{doc['name']}** — {doc['pages']} pages, {doc['sections']} sections/extraits, "
                        f"{doc['text_chars']} caractères"
                    )
        if show_tree and st.session_state.trees:
            with st.expander("Arbre PageIndex-style", expanded=True):
                st.json(st.session_state.trees)
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
                "Tu es un assistant de RAG documentaire. Réponds uniquement à partir du CONTEXTE fourni. "
                "Cite les documents, pages et sections utiles quand ils sont disponibles. "
                "Si le contexte ne permet pas de répondre, dis-le clairement et indique quelle information manque."
            )

            user_prompt = f"""CONTEXTE (sections extraites du PDF) :
{context}

QUESTION :
{question}

Réponds en français, de façon concise et factuelle. Appuie-toi uniquement sur le contexte."""

            if selected:
                with st.expander("Sections retenues", expanded=False):
                    for score, item in selected:
                        st.markdown(
                            f"- **{item['doc_name']}**, `{item['node_id']}`, {pages_label(item)}, "
                            f"score `{score}` — {item['path']}"
                        )

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
                    for tok in chat_stream(
                        provider=provider,
                        messages=messages,
                        ollama_host=ollama_host,
                        ollama_model=ollama_model,
                        openai_base_url=openai_base_url,
                        openai_api_key=openai_api_key,
                        openai_model=openai_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ):
                        acc += tok
                        placeholder.markdown(acc)
                except requests.exceptions.ConnectionError:
                    st.error(
                        "Impossible de joindre Ollama. Vérifie qu’Ollama tourne et que l’URL est bonne "
                        f"({ollama_host})."
                    )
                    acc = ""
                except requests.HTTPError as e:
                    st.error(f"Erreur HTTP Ollama: {e}")
                    acc = ""
                except ValueError as e:
                    st.error(str(e))
                    acc = ""
                except Exception as e:
                    st.error(f"Erreur API LLM: {e}")
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
                st.session_state.documents = []
                st.session_state.chunks = []
                st.session_state.trees = []
                st.session_state.chat = []
                st.rerun()
        with c3:
            st.caption("RAG hiérarchique local (sans embeddings ni vector DB).")
