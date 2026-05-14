#!/usr/bin/env python3
"""
Streamlit PageIndex document Q&A app.

This version intentionally removes the former lexical/chunk fallback. All
indexing and retrieval now go through VectifyAI/PageIndex's Collection API:

1. upload PDF or Markdown files;
2. index them with collection.add(...), which builds PageIndex's hierarchical
   tree index;
3. answer questions with collection.query(...), letting the PageIndex agent use
   list_documents, get_document, get_document_structure and get_page_content.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import streamlit as st
from pageindex import PageIndexClient  # type: ignore


# ----------------------------
# Env helpers
# ----------------------------
def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


# ----------------------------
# File / PageIndex helpers
# ----------------------------
def safe_filename(name: str) -> str:
    base = Path(name).name
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", base).strip("._") or "document"


def file_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def write_uploaded_document(workspace: Path, file_name: str, data: bytes) -> Path:
    upload_dir = workspace / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{file_digest(data)}_{safe_filename(file_name)}"
    if not path.exists():
        path.write_bytes(data)
    return path


def pageindex_value_to_text(value: Any) -> str:
    """Convert PageIndex SDK responses/events into prompt- or UI-safe text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        return str(value)


def compact_text(value: Any, limit: int = 2_000) -> str:
    text = pageindex_value_to_text(value).strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [sortie tronquée]"


def get_pageindex_client(*, api_key: str, model: str, retrieve_model: str, storage_path: Path):
    kwargs: Dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    else:
        storage_path.mkdir(parents=True, exist_ok=True)
        kwargs["storage_path"] = str(storage_path)
        if model:
            kwargs["model"] = model
        if retrieve_model:
            kwargs["retrieve_model"] = retrieve_model
    return PageIndexClient(**kwargs)


def get_pageindex_collection(client: Any, collection_name: str):
    return client.collection(collection_name)


def _doc_name_from_metadata(doc: dict) -> str:
    for key in ("doc_name", "name", "file_name", "filename", "title"):
        value = doc.get(key)
        if value:
            return str(value)
    path = doc.get("file_path") or doc.get("path") or doc.get("source")
    if path:
        return Path(str(path)).name
    return ""


def _doc_id_from_metadata(doc: dict) -> Optional[str]:
    for key in ("doc_id", "id", "document_id"):
        value = doc.get(key)
        if value:
            return str(value)
    return None


def find_cached_doc_id(collection: Any, stored_path: Path) -> Optional[str]:
    for doc in collection.list_documents():
        doc_id = _doc_id_from_metadata(doc)
        if doc_id and _doc_name_from_metadata(doc) == stored_path.name:
            return doc_id
    return None


def index_with_pageindex(collection: Any, stored_path: Path) -> str:
    cached = find_cached_doc_id(collection, stored_path)
    if cached:
        return cached
    return str(collection.add(str(stored_path)))


def run_pageindex_query(
    *,
    collection: Any,
    question: str,
    doc_ids: List[str],
    on_answer_delta: Callable[[str], None],
    on_trace: Callable[[dict], None],
) -> str:
    """Run the PageIndex agentic query stream from synchronous Streamlit code."""

    async def consume_stream() -> str:
        final_answer_parts: List[str] = []
        final_answer = ""
        stream = collection.query(question, doc_ids=doc_ids or None, stream=True)
        async for event in stream:
            event_type = getattr(event, "type", "")
            data = getattr(event, "data", None)
            if event_type == "answer_delta":
                delta = str(data or "")
                final_answer_parts.append(delta)
                on_answer_delta(delta)
            elif event_type == "answer_done":
                final_answer = str(data or "")
            elif event_type in {"reasoning", "tool_call", "tool_result"}:
                trace = {"type": event_type}
                if event_type == "tool_result":
                    trace["data"] = compact_text(data)
                else:
                    trace["data"] = data
                on_trace(trace)
        return final_answer or "".join(final_answer_parts)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(consume_stream()).strip()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(consume_stream())).result().strip()


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="PDF/Markdown → Q&A PageIndex", layout="wide")
st.title("PDF/Markdown → Q&A avec PageIndex")
st.caption("Retrieval 100% PageIndex : index hiérarchique, agent tools, sans fallback lexical ni chunks locaux.")

workspace = Path(os.getenv("PAGEINDEX_WORKSPACE", ".pageindex_workspace"))
default_api_key = os.getenv("PAGEINDEX_API_KEY", "")
default_collection = os.getenv("PAGEINDEX_COLLECTION", "default")
default_model = os.getenv("PAGEINDEX_MODEL", "")
default_retrieve_model = os.getenv("PAGEINDEX_RETRIEVE_MODEL", default_model)

with st.sidebar:
    st.header("PageIndex")
    pageindex_api_key = st.text_input(
        "PageIndex API key (cloud, optionnel)",
        value=default_api_key,
        type="password",
        help="Si renseigné, PageIndex fonctionne en mode cloud. Sinon l'app utilise le mode local self-hosted.",
    )
    collection_name = st.text_input("Collection", value=default_collection)
    storage_path_text = st.text_input("Storage local", value=str(workspace))

    st.subheader("Modèles locaux")
    pageindex_model = st.text_input(
        "PAGEINDEX_MODEL",
        value=default_model,
        help="Modèle utilisé pour construire l'index en mode local. Vide = défaut PageIndex.",
        disabled=bool(pageindex_api_key),
    )
    pageindex_retrieve_model = st.text_input(
        "PAGEINDEX_RETRIEVE_MODEL",
        value=default_retrieve_model,
        help="Modèle agentique utilisé par collection.query(...). Vide = défaut PageIndex.",
        disabled=bool(pageindex_api_key),
    )

    st.divider()
    show_traces = st.checkbox("Afficher appels outils PageIndex", value=env_bool("PAGEINDEX_SHOW_TRACES", False))

uploaded_files = st.file_uploader(
    "Upload un ou plusieurs documents",
    type=["pdf", "md", "markdown"],
    accept_multiple_files=True,
)

for key, default in {
    "documents": [],
    "chat": [],
    "last_traces": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1) Indexation PageIndex")

    if not uploaded_files:
        st.info("Charge un ou plusieurs PDF/Markdown pour commencer.")
    else:
        st.caption(f"{len(uploaded_files)} fichier(s) sélectionné(s).")
        if st.button("Indexer avec PageIndex", type="primary"):
            docs: List[dict] = []
            storage_path = Path(storage_path_text)

            with st.spinner("Construction de l'index hiérarchique PageIndex…"):
                try:
                    pageindex_client = get_pageindex_client(
                        api_key=pageindex_api_key,
                        model=pageindex_model,
                        retrieve_model=pageindex_retrieve_model,
                        storage_path=storage_path,
                    )
                    pageindex_collection = get_pageindex_collection(pageindex_client, collection_name)

                    for file in uploaded_files:
                        data = file.read()
                        stored_path = write_uploaded_document(storage_path, file.name, data)
                        doc_id = index_with_pageindex(pageindex_collection, stored_path)
                        metadata = pageindex_collection.get_document(doc_id)
                        docs.append(
                            {
                                "name": file.name,
                                "stored_name": stored_path.name,
                                "path": str(stored_path),
                                "doc_id": doc_id,
                                "metadata": metadata,
                            }
                        )
                except Exception as exc:
                    st.error(f"Indexation PageIndex impossible : {exc}")
                    docs = []

            if docs:
                st.session_state.documents = docs
                st.session_state.chat = []
                st.session_state.last_traces = []
                st.success(f"OK. {len(docs)} document(s) indexé(s) avec PageIndex.")

    if st.session_state.documents:
        with st.expander("Documents indexés", expanded=True):
            for doc in st.session_state.documents:
                metadata = doc.get("metadata") or {}
                doc_type = metadata.get("doc_type") or metadata.get("type") or "document"
                page_count = metadata.get("page_count")
                line_count = metadata.get("line_count")
                size_label = ""
                if page_count:
                    size_label = f" — {page_count} pages"
                elif line_count:
                    size_label = f" — {line_count} lignes"
                st.markdown(f"- **{doc['name']}** — `{doc_type}`{size_label}")
                st.caption(f"doc_id: {doc['doc_id']}")

        if show_traces and st.session_state.last_traces:
            with st.expander("Derniers appels outils PageIndex", expanded=False):
                st.json(st.session_state.last_traces)

with col_right:
    st.subheader("2) Chat PageIndex")

    if not st.session_state.documents:
        st.warning("Indexe d'abord les documents avec PageIndex.")
    else:
        for message in st.session_state.chat:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Pose une question sur ces documents…")
        if question:
            st.session_state.chat.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            traces: List[dict] = []
            doc_ids = [doc["doc_id"] for doc in st.session_state.documents if doc.get("doc_id")]

            with st.chat_message("assistant"):
                placeholder = st.empty()
                answer = ""

                def append_delta(delta: str) -> None:
                    nonlocal_answer["value"] += delta
                    placeholder.markdown(nonlocal_answer["value"])

                def append_trace(trace: dict) -> None:
                    traces.append(trace)

                nonlocal_answer = {"value": ""}
                try:
                    pageindex_client = get_pageindex_client(
                        api_key=pageindex_api_key,
                        model=pageindex_model,
                        retrieve_model=pageindex_retrieve_model,
                        storage_path=Path(storage_path_text),
                    )
                    pageindex_collection = get_pageindex_collection(pageindex_client, collection_name)
                    answer = run_pageindex_query(
                        collection=pageindex_collection,
                        question=question,
                        doc_ids=doc_ids,
                        on_answer_delta=append_delta,
                        on_trace=append_trace,
                    )
                    if answer and answer != nonlocal_answer["value"]:
                        placeholder.markdown(answer)
                except Exception as exc:
                    answer = ""
                    st.error(f"Erreur PageIndex : {exc}")

            st.session_state.last_traces = traces
            if answer.strip():
                st.session_state.chat.append({"role": "assistant", "content": answer})

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Vider le chat"):
                st.session_state.chat = []
                st.session_state.last_traces = []
                st.rerun()
        with c2:
            if st.button("Réinitialiser documents + chat"):
                st.session_state.documents = []
                st.session_state.chat = []
                st.session_state.last_traces = []
                st.rerun()
