#!/usr/bin/env python3
"""
Streamlit PDF Q&A MVP.

Version orientée vrai PageIndex :
1. upload PDF ;
2. indexation via VectifyAI/PageIndex si disponible ;
3. lecture de la structure PageIndex avec get_document_structure() ;
4. sélection de pages par le LLM ;
5. récupération ciblée avec get_page_content() ;
6. réponse uniquement depuis le contexte récupéré.

Fallback automatique : si PageIndex n'est pas installé ou échoue, l'application
revient au RAG lexical simple sans embeddings ni vector DB.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

PAGEINDEX_IMPORT_ERROR = None

try:
    from pageindex import PageIndexClient  # type: ignore
except Exception as exc:
    PageIndexClient = None  # type: ignore
    PAGEINDEX_IMPORT_ERROR = repr(exc)


# ----------------------------
# Env helpers
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


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


# ----------------------------
# PDF -> texte fallback
# ----------------------------
def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = text.replace("\x00", " ")
        pages.append(text)
    return "\n\n".join(pages)


# ----------------------------
# Chunking lexical fallback
# ----------------------------
def normalize_text(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, max_chars: int = 1800, overlap: int = 200) -> List[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")

    overlap = max(0, min(overlap, max_chars - 1))
    text = normalize_text(text)

    chunks: List[str] = []
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


def tokenize(value: str) -> List[str]:
    return re.findall(r"[a-zA-ZÀ-ÿ0-9]{2,}", value.lower())


def score_chunk(query: str, chunk: str) -> int:
    query_terms = set(tokenize(query))
    chunk_words = tokenize(chunk)
    if not query_terms or not chunk_words:
        return 0

    hits = sum(1 for word in chunk_words if word in query_terms)
    uniq = len(set(chunk_words) & query_terms)

    query_norm = " ".join(tokenize(query))
    chunk_norm = " ".join(chunk_words)
    phrase_bonus = 25 if query_norm and query_norm in chunk_norm else 0

    return hits + 2 * uniq + phrase_bonus


def top_k_chunks(query: str, chunks: List[dict], k: int = 4) -> List[Tuple[int, dict]]:
    scored = [(score_chunk(query, item["text"]), item) for item in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(score, item) for score, item in scored[:k] if score > 0]


def build_lexical_context(question: str, chunks: List[dict], k: int) -> Tuple[str, List[dict]]:
    selected = top_k_chunks(question, chunks, k=k)
    if selected:
        context = "\n\n".join(
            f"[CHUNK score={score} doc={item['doc_name']} idx={item['chunk_id']}]\n{item['text']}"
            for score, item in selected
        )
        traces = [
            {
                "mode": "lexical",
                "doc": item["doc_name"],
                "chunk": item["chunk_id"],
                "score": score,
            }
            for score, item in selected
        ]
        return context, traces

    fallback_items = chunks[:2]
    fallback_context = "\n\n".join(
        f"[CHUNK score=0 doc={item['doc_name']} idx={item['chunk_id']}]\n{item['text']}"
        for item in fallback_items
    )
    traces = [
        {
            "mode": "lexical_fallback_start",
            "doc": item["doc_name"],
            "chunk": item["chunk_id"],
            "score": 0,
        }
        for item in fallback_items
    ]
    return fallback_context, traces


# ----------------------------
# LLM providers
# ----------------------------
def ollama_chat_stream(
    model: str,
    host: str,
    messages: List[dict],
    temperature: float,
    max_tokens: int,
) -> Iterable[str]:
    url = f"{host.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "stream": True,
        "messages": messages,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }

    with requests.post(url, json=payload, stream=True, timeout=300) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            data = json.loads(line)
            chunk = (data.get("message") or {}).get("content")
            if chunk:
                yield chunk
            if data.get("done") is True:
                break


def openai_compatible_chat_stream(
    model: str,
    base_url: str,
    api_key: str,
    messages: List[dict],
    temperature: float,
    max_tokens: int,
) -> Iterable[str]:
    if not api_key:
        raise ValueError("OPENAI_API_KEY est requis avec le provider openai_compatible.")

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
    *,
    provider: str,
    messages: List[dict],
    ollama_host: str,
    ollama_model: str,
    openai_base_url: str,
    openai_api_key: str,
    openai_model: str,
    temperature: float,
    max_tokens: int,
) -> Iterable[str]:
    if provider == "openai_compatible":
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


def chat_complete(*, max_tokens: int = 512, **kwargs: Any) -> str:
    return "".join(chat_stream(max_tokens=max_tokens, **kwargs)).strip()


# ----------------------------
# PageIndex integration
# ----------------------------
def safe_filename(name: str) -> str:
    base = Path(name).name
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", base).strip("._") or "document.pdf"


def file_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def write_uploaded_pdf(workspace: Path, file_name: str, data: bytes) -> Path:
    upload_dir = workspace / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{file_digest(data)}_{safe_filename(file_name)}"
    if not path.exists():
        path.write_bytes(data)
    return path


def get_pageindex_client(workspace: Path):
    if PageIndexClient is None:
        raise RuntimeError(
            "PageIndex n'est pas installé/importable. "
            f"Erreur import: {PAGEINDEX_IMPORT_ERROR}"
        )

    workspace.mkdir(parents=True, exist_ok=True)

    # PageIndex dev attend storage_path=..., pas workspace=...
    # PAGEINDEX_MODEL peut être utile si tu ne veux pas le modèle par défaut.
    model = os.getenv("PAGEINDEX_MODEL") or None
    retrieve_model = os.getenv("PAGEINDEX_RETRIEVE_MODEL") or None

    return PageIndexClient(
        model=model,
        retrieve_model=retrieve_model,
        storage_path=str(workspace),
    )


def get_pageindex_collection(client: Any):
    collection_name = os.getenv("PAGEINDEX_COLLECTION", "default")
    return client.collection(collection_name)


def _doc_name_from_metadata(doc: dict) -> str:
    for key in ("doc_name", "name", "file_name", "filename", "title"):
        value = doc.get(key)
        if value:
            return str(value)
    path = doc.get("path") or doc.get("file_path") or doc.get("source")
    if path:
        return Path(str(path)).name
    return ""


def _doc_id_from_metadata(doc: dict) -> Optional[str]:
    for key in ("doc_id", "id", "document_id"):
        value = doc.get(key)
        if value:
            return str(value)
    return None


def find_cached_doc_id(collection: Any, pdf_path: Path) -> Optional[str]:
    try:
        for doc in collection.list_documents():
            if _doc_name_from_metadata(doc) == pdf_path.name:
                return _doc_id_from_metadata(doc)
    except Exception:
        return None
    return None


def index_with_pageindex(collection: Any, pdf_path: Path) -> str:
    cached = find_cached_doc_id(collection, pdf_path)
    if cached:
        return cached
    return str(collection.add(str(pdf_path)))


def compact_structure(structure: str, limit: int) -> str:
    structure = structure.strip()
    if len(structure) <= limit:
        return structure
    return structure[:limit] + "\n... [structure tronquée]"


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}
    return {}


def normalize_pages(value: Any, default_pages: str) -> str:
    if isinstance(value, list):
        value = ",".join(str(v) for v in value)
    value = str(value or "").strip()

    # Formes acceptées : 1, 1-3, 1,3,7-8.
    candidates = re.findall(r"\d+(?:\s*-\s*\d+)?", value)
    if not candidates:
        return default_pages

    return ",".join(part.replace(" ", "") for part in candidates[:8])


def choose_pageindex_ranges(
    *,
    question: str,
    doc_name: str,
    structure: str,
    default_pages: str,
    llm_kwargs: Dict[str, Any],
    max_structure_chars: int,
) -> Tuple[str, str]:
    system = (
        "Tu sélectionnes les pages ou lignes utiles dans une structure PageIndex. "
        "Réponds uniquement en JSON valide."
    )
    user = f"""
DOCUMENT: {doc_name}
QUESTION: {question}

STRUCTURE PAGEINDEX:
{compact_structure(structure, max_structure_chars)}

Retourne uniquement ce JSON :
{{"pages":"1-3,5", "reason":"raison très courte"}}

Règles :
- Choisis des plages serrées.
- Ne récupère jamais tout le document.
- Si la structure ne suffit pas, prends le début logique du document : {default_pages}.
""".strip()

    raw = chat_complete(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=256,
        **llm_kwargs,
    )
    data = extract_json_object(raw)
    pages = normalize_pages(data.get("pages"), default_pages=default_pages)
    reason = str(data.get("reason") or "Sélection automatique via structure PageIndex.").strip()
    return pages, reason


def build_pageindex_context(
    *,
    client: Any,
    question: str,
    documents: List[dict],
    llm_kwargs: Dict[str, Any],
    max_docs: int,
    default_pages: str,
    max_structure_chars: int,
    max_context_chars: int,
) -> Tuple[str, List[dict]]:
    parts: List[str] = []
    traces: List[dict] = []

    for doc in documents[:max_docs]:
        doc_id = doc["doc_id"]
        doc_name = doc["name"]

        structure = client.get_document_structure(doc_id)
        pages, reason = choose_pageindex_ranges(
            question=question,
            doc_name=doc_name,
            structure=structure,
            default_pages=default_pages,
            llm_kwargs=llm_kwargs,
            max_structure_chars=max_structure_chars,
        )
        content = client.get_page_content(doc_id, pages)

        traces.append(
            {
                "mode": "pageindex",
                "doc": doc_name,
                "doc_id": doc_id,
                "pages": pages,
                "reason": reason,
            }
        )
        parts.append(f"[PAGEINDEX doc={doc_name} pages={pages} reason={reason}]\n{content}")

    context = "\n\n".join(parts).strip()
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n... [contexte tronqué]"
    return context, traces


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="PDF → Q&A PageIndex", layout="wide")
st.title("PDF → Q&A avec PageIndex")
st.caption("Vrai PageIndex si disponible. Fallback lexical simple sans embeddings.")

workspace = Path(os.getenv("PAGEINDEX_WORKSPACE", ".pageindex_workspace"))
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

    retrieval_mode = st.selectbox(
        "Mode retrieval",
        ["pageindex", "lexical"],
        index=0 if env_bool("USE_PAGEINDEX", True) else 1,
        help="PageIndex utilise get_document_structure/get_page_content. Lexical garde le fallback top-k chunks.",
    )

    provider_options = ["ollama", "openai_compatible"]
    provider_index = provider_options.index(default_provider) if default_provider in provider_options else 0
    provider = st.selectbox("Provider LLM", provider_options, index=provider_index)

    with st.expander("Ollama", expanded=(provider == "ollama")):
        ollama_host = st.text_input("Ollama host", value=default_ollama_host)
        ollama_model = st.text_input("Modèle Ollama", value=default_ollama_model)

    with st.expander("API OpenAI-compatible", expanded=(provider == "openai_compatible")):
        openai_base_url = st.text_input("Base URL", value=default_openai_base_url)
        openai_model = st.text_input("Modèle API", value=default_openai_model)
        openai_api_key = st.text_input("Token API", value=default_openai_api_key, type="password")

    st.divider()
    st.subheader("PageIndex")
    pageindex_max_docs = st.slider("Documents PageIndex interrogés", 1, 5, env_int("PAGEINDEX_MAX_DOCS", 3))
    pageindex_default_pages = st.text_input("Pages fallback", os.getenv("PAGEINDEX_DEFAULT_PAGES", "1-3"))
    max_structure_chars = st.slider(
        "Structure max chars",
        2_000,
        50_000,
        env_int("PAGEINDEX_MAX_STRUCTURE_CHARS", 16_000),
        step=1_000,
    )
    max_context_chars = st.slider(
        "Contexte max chars",
        4_000,
        80_000,
        env_int("MAX_CONTEXT_CHARS", 24_000),
        step=1_000,
    )

    st.divider()
    st.subheader("Fallback lexical")
    k = st.slider("Top-K chunks", 1, 10, env_int("TOP_K", 4))
    max_chars = st.slider("Taille chunk", 600, 8000, env_int("CHUNK_MAX_CHARS", 2400), step=100)
    overlap = st.slider("Overlap", 0, 1000, env_int("CHUNK_OVERLAP", 200), step=50)

    st.divider()
    temperature = st.slider("Température", 0.0, 2.0, default_temperature, step=0.1)
    max_tokens = st.slider("Max tokens", 128, 8192, default_max_tokens, step=128)
    show_context = st.checkbox("Afficher contexte / traces", value=False)

if overlap >= max_chars:
    st.sidebar.warning("Overlap doit être inférieur à la taille de chunk. Il sera automatiquement réduit.")

llm_kwargs = {
    "provider": provider,
    "ollama_host": ollama_host,
    "ollama_model": ollama_model,
    "openai_base_url": openai_base_url,
    "openai_api_key": openai_api_key,
    "openai_model": openai_model,
}

uploaded_files = st.file_uploader("Upload un ou plusieurs PDF", type=["pdf"], accept_multiple_files=True)

for key, default in {
    "pdf_text": "",
    "documents": [],
    "chunks": [],
    "chat": [],
    "pageindex_enabled": False,
    "pageindex_error": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1) Indexation")

    if PageIndexClient is None and retrieval_mode == "pageindex":
        st.warning(
            "PageIndex n'est pas importable. "
            f"Erreur: {PAGEINDEX_IMPORT_ERROR}. "
            "Vérifie que Streamlit est lancé avec `uv run streamlit run app.py`."
        )

    if not uploaded_files:
        st.info("Charge un ou plusieurs PDF pour commencer.")
    else:
        st.caption(f"{len(uploaded_files)} fichier(s) sélectionné(s).")
        if st.button("Indexer les PDF", type="primary"):
            all_texts: List[str] = []
            all_chunks: List[dict] = []
            docs: List[dict] = []
            pageindex_error = ""
            pageindex_enabled = False
            pageindex_collection = None

            if retrieval_mode == "pageindex" and PageIndexClient is not None:
                try:
                    pageindex_client = get_pageindex_client(workspace)
                    pageindex_collection = get_pageindex_collection(pageindex_client)
                except Exception as exc:
                    pageindex_error = str(exc)

            with st.spinner("Indexation des documents…"):
                for file in uploaded_files:
                    pdf_bytes = file.read()
                    pdf_path = write_uploaded_pdf(workspace, file.name, pdf_bytes)

                    doc_entry: Dict[str, Any] = {
                        "name": file.name,
                        "path": str(pdf_path),
                        "mode": "lexical",
                    }

                    if pageindex_collection is not None:
                        try:
                            doc_id = index_with_pageindex(pageindex_collection, pdf_path)
                            doc_entry.update({"doc_id": doc_id, "mode": "pageindex"})
                            pageindex_enabled = True
                        except Exception as exc:
                            pageindex_error = f"PageIndex a échoué pour {file.name}: {exc}"

                    text = pdf_bytes_to_text(pdf_bytes)
                    if text.strip():
                        doc_chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)
                        doc_entry.update({"text_chars": len(text), "chunks": len(doc_chunks)})
                        all_texts.append(f"===== {file.name} =====\n{text}")
                        for idx, chunk in enumerate(doc_chunks, start=1):
                            all_chunks.append({"doc_name": file.name, "chunk_id": idx, "text": chunk})
                    else:
                        doc_entry.update({"text_chars": 0, "chunks": 0})

                    docs.append(doc_entry)

            if not docs:
                st.error("Aucun document exploitable trouvé.")
            elif not pageindex_enabled and not all_chunks:
                st.error("Aucun texte exploitable trouvé. Probable PDF scanné : il faut un OCR.")
            else:
                st.session_state.pdf_text = "\n\n".join(all_texts)
                st.session_state.documents = docs
                st.session_state.chunks = all_chunks
                st.session_state.pageindex_enabled = pageindex_enabled
                st.session_state.pageindex_error = pageindex_error

                mode_label = "PageIndex" if pageindex_enabled else "lexical"
                st.success(
                    f"OK. {len(docs)} document(s) indexé(s), "
                    f"{len(all_chunks)} chunks fallback, mode actif: {mode_label}."
                )
                if pageindex_error:
                    st.warning(pageindex_error)

    if st.session_state.documents:
        with st.expander("Documents indexés", expanded=False):
            for doc in st.session_state.documents:
                st.markdown(
                    f"- **{doc['name']}** — mode `{doc.get('mode')}` — "
                    f"{doc.get('chunks', 0)} chunks — {doc.get('text_chars', 0)} caractères"
                )
                if doc.get("doc_id"):
                    st.caption(f"doc_id: {doc['doc_id']}")

    if st.session_state.pdf_text:
        with st.expander("Aperçu texte extrait", expanded=False):
            st.text_area("Texte", st.session_state.pdf_text[:8000], height=260)

with col_right:
    st.subheader("2) Chat")

    if not st.session_state.documents:
        st.warning("Indexe d'abord les documents.")
    else:
        for message in st.session_state.chat:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Pose une question sur ces documents…")
        if question:
            st.session_state.chat.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            context = ""
            traces: List[dict] = []
            active_mode = "lexical"

            try:
                if retrieval_mode == "pageindex" and st.session_state.pageindex_enabled:
                    pageindex_client = get_pageindex_client(workspace)
                    pageindex_collection = get_pageindex_collection(pageindex_client)
                    pageindex_docs = [doc for doc in st.session_state.documents if doc.get("doc_id")]
                    context, traces = build_pageindex_context(
                        client=pageindex_collection,
                        question=question,
                        documents=pageindex_docs,
                        llm_kwargs=llm_kwargs,
                        max_docs=pageindex_max_docs,
                        default_pages=pageindex_default_pages,
                        max_structure_chars=max_structure_chars,
                        max_context_chars=max_context_chars,
                    )
                    active_mode = "pageindex"
                else:
                    context, traces = build_lexical_context(question, st.session_state.chunks, k=k)
            except Exception as exc:
                st.warning(f"PageIndex indisponible pour cette question, fallback lexical: {exc}")
                context, traces = build_lexical_context(question, st.session_state.chunks, k=k)
                active_mode = "lexical"

            if show_context:
                with st.expander(f"Contexte injecté ({active_mode})", expanded=False):
                    st.json(traces)
                    st.code(context[:30000])

            system = (
                "Tu es un assistant documentaire. Réponds uniquement à partir du CONTEXTE fourni. "
                "Si le contexte ne contient pas la réponse, dis-le clairement. "
                "Réponds en français, de façon concise, factuelle et utile."
            )
            user_prompt = f"""CONTEXTE:
{context}

QUESTION:
{question}

Réponse attendue : concise, sourcée par le contexte, sans invention."""

            with st.chat_message("assistant"):
                placeholder = st.empty()
                answer = ""
                try:
                    for token in chat_stream(
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **llm_kwargs,
                    ):
                        answer += token
                        placeholder.markdown(answer)
                except requests.exceptions.ConnectionError:
                    st.error(f"Impossible de joindre Ollama ({ollama_host}).")
                    answer = ""
                except requests.HTTPError as exc:
                    st.error(f"Erreur HTTP LLM: {exc}")
                    answer = ""
                except Exception as exc:
                    st.error(f"Erreur LLM: {exc}")
                    answer = ""

            if answer.strip():
                st.session_state.chat.append({"role": "assistant", "content": answer})

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
                st.session_state.chat = []
                st.session_state.pageindex_enabled = False
                st.session_state.pageindex_error = ""
                st.rerun()
        with c3:
            if st.session_state.pageindex_enabled:
                st.caption("Retrieval: PageIndex vectorless")
            else:
                st.caption("Retrieval: fallback lexical")
