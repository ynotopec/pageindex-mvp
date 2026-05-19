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
import inspect
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Disable OpenAI Agents trace export by default before PageIndex imports the
# agents SDK. Local/OpenAI-compatible keys are often not valid OpenAI platform
# keys, so exporting traces can otherwise create noisy 401 warnings.
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

from agents import set_tracing_disabled
import streamlit as st
from pageindex import PageIndexClient  # type: ignore
from pageindex.config import IndexConfig  # type: ignore

set_tracing_disabled(os.getenv("OPENAI_AGENTS_DISABLE_TRACING", "1").strip().lower() in {"1", "true", "yes", "y", "on"})


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



def sanitize_error_text(value: Any) -> str:
    """Return a UI-safe error message without leaking API keys."""
    text = pageindex_value_to_text(value).strip()
    if not text:
        return "Erreur inconnue."

    text = re.sub(
        r"(?i)(api key(?: provided)?\s*[:=]\s*)[^\s,}\]]+",
        r"\1[redacted]",
        text,
    )
    text = re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "sk-[redacted]", text)
    return text


def pageindex_query_result_to_text(result: Any) -> str:
    """Extract answer text from non-streaming PageIndex query responses."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        for key in ("answer", "content", "text", "response", "result"):
            value = result.get(key)
            if value:
                return pageindex_value_to_text(value).strip()
    for attr in ("answer", "content", "text", "response", "result"):
        value = getattr(result, attr, None)
        if value:
            return pageindex_value_to_text(value).strip()
    return pageindex_value_to_text(result).strip()


def is_pageindex_processing_failure(exc: BaseException) -> bool:
    """Detect PageIndex's generic local PDF tree-building failure."""
    return "processing failed" in sanitize_error_text(exc).lower()


def build_pageindex_indexing_error_message(
    *,
    stored_path: Path,
    error: BaseException,
    retried_without_toc_detection: bool = False,
    retry_error: Optional[BaseException] = None,
) -> str:
    """Build an actionable, sanitized message for PageIndex indexing failures."""
    detail = sanitize_error_text(error)
    message = f"Failed to index {stored_path}: {detail}"
    if retry_error is not None:
        message += f" Retry toc_check_page_num=0: {sanitize_error_text(retry_error)}"

    retry_failed_processing = (
        retry_error is not None and is_pageindex_processing_failure(retry_error)
    )
    if is_pageindex_processing_failure(error) or retry_failed_processing:
        if retried_without_toc_detection:
            message += (
                " PageIndex a aussi échoué après un retry automatique avec "
                "toc_check_page_num=0. Vérifie que le PDF contient du texte sélectionnable "
                "(pas uniquement des scans/images), utilise un modèle d'indexation plus fiable "
                "ou convertis/OCR le document avant de le réimporter."
            )
        else:
            message += (
                " PageIndex signale souvent cette erreur quand la table des matières du PDF "
                "est mal détectée ou mal alignée avec les pages. En mode local, réessaie avec "
                "toc_check_page_num=0 dans IndexConfig pour forcer l'extraction sans TOC."
            )
    return message


def is_internal_server_error(exc: BaseException) -> bool:
    """Detect generic 5xx message surfaced by OpenAI-compatible backends."""
    return "internal server error" in sanitize_error_text(exc).lower()




def is_litellm_model_group_error(exc: BaseException) -> bool:
    """Detect LiteLLM proxy routing errors about missing model groups."""
    text = sanitize_error_text(exc).lower()
    return "model group" in text and ("available model group fallbacks" in text or "received model group" in text)

def build_openai_compatibility_hint(exc: BaseException) -> str:
    """Explain the backend capability expected by PageIndex/OpenAI Agents."""
    if not is_internal_server_error(exc):
        return ""
    message = (
        " Le backend ciblé par OPENAI_BASE_URL doit être pleinement compatible "
        "OpenAI Responses + tool calling (utilisés par openai-agents/PageIndex). "
        "Certains endpoints OpenAI-compatibles partiels (chat/completions only) "
        "répondent 500 sur ces appels. Les modèles open source restent compatibles "
        "si tu passes par un serveur/proxy qui expose correctement Responses API + "
        "tool calls (ex. LiteLLM, vLLM récent, Ollama OpenAI mode selon version). "
        "Pour isoler le problème, vérifie d'abord avec OPENAI officiel "
        "(https://api.openai.com/v1)."
    )
    if is_litellm_model_group_error(exc):
        message += (
            " Avec LiteLLM proxy, l'erreur 'Received Model Group=...' indique souvent "
            "que ce model group n'est pas routé dans la config (ou sans fallback). "
            "Ajoute un mapping pour ce group (ex. ai-tools) vers un modèle/provider "
            "valide, ou aligne PAGEINDEX_MODEL/PAGEINDEX_RETRIEVE_MODEL sur un model "
            "group existant côté proxy."
        )
    return message


def build_pageindex_error_message(*, stream_error: BaseException, retry_error: Optional[BaseException] = None) -> str:
    message = "Le streaming PageIndex a échoué."
    message += f" Détail streaming: {sanitize_error_text(stream_error)}"
    if retry_error is not None:
        message += f" Détail retry non-stream: {sanitize_error_text(retry_error)}"
    message += (
        " Vérifie OPENAI_BASE_URL/OPENAI_API_BASE, OPENAI_API_KEY, PAGEINDEX_MODEL "
        "et PAGEINDEX_RETRIEVE_MODEL. Si LiteLLM affiche Provider List, utilise un modèle "
        "préfixé litellm/provider/... (ex. litellm/openai/..., litellm/ollama_chat/..., "
        "litellm/vllm/...)."
    )
    message += build_openai_compatibility_hint(stream_error)
    if retry_error is not None:
        message += build_openai_compatibility_hint(retry_error)
    return message


def normalize_retrieve_model_name(model: str) -> str:
    """Normalize retrieval model names the same way PageIndex ``dev`` does.

    PageIndex passes the indexing model to LiteLLM as-is for provider
    validation, then normalizes only the agent QA model: plain model names such
    as ``gpt-4o-mini`` stay plain, ``openai/...`` and ``litellm/...`` are
    preserved, and other provider paths are routed through the Agents SDK
    LiteLLM provider.
    """
    value = model.strip()
    passthrough_prefixes = ("litellm/", "openai/")
    if not value or "/" not in value or value.startswith(passthrough_prefixes):
        return value
    return f"litellm/{value}"


def build_pageindex_query_prompt(question: str) -> str:
    """Add the official PageIndex agent guidance around the user's question."""
    clean_question = question.strip()
    return (
        "Réponds à la question ci-dessous avec PageIndex uniquement. "
        "Commence par vérifier les métadonnées du document, puis utilise la "
        "structure PageIndex pour choisir les sections pertinentes, puis récupère "
        "seulement les pages/lignes nécessaires avec des plages serrées. "
        "Base la réponse exclusivement sur le contenu récupéré; si l'information "
        "n'est pas dans les documents, dis-le clairement. Donne une réponse concise "
        "et cite les pages, lignes ou sections quand elles sont disponibles.\n\n"
        f"Question utilisateur : {clean_question}"
    )


def configure_llm_environment(
    *,
    llm_api_key: str,
    llm_base_url: str,
    disable_tracing: bool = True,
) -> Dict[str, str]:
    """Expose the selected OpenAI-compatible endpoint to PageIndex/LiteLLM.

    PageIndex local mode delegates model calls to LiteLLM/OpenAI-compatible
    clients, which read their endpoint from environment variables. Set both
    commonly used names so vLLM, LiteLLM proxy, Ollama OpenAI mode and OpenAI
    SDK-compatible providers all receive the same base URL.
    """
    updates: Dict[str, str] = {}
    api_key = llm_api_key.strip()
    base_url = llm_base_url.strip().rstrip("/")

    updates["OPENAI_AGENTS_DISABLE_TRACING"] = "1" if disable_tracing else "0"

    if api_key:
        updates["OPENAI_API_KEY"] = api_key
    if base_url:
        updates["OPENAI_BASE_URL"] = base_url
        updates["OPENAI_API_BASE"] = base_url

    os.environ.update(updates)
    set_tracing_disabled(disable_tracing)
    return updates


def get_pageindex_client(
    *,
    api_key: str,
    model: str,
    retrieve_model: str,
    storage_path: Path,
    llm_api_key: str,
    llm_base_url: str,
    disable_tracing: bool,
    index_config: Optional[IndexConfig] = None,
):
    kwargs: Dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    else:
        configure_llm_environment(
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            disable_tracing=disable_tracing,
        )
        storage_path.mkdir(parents=True, exist_ok=True)
        kwargs["storage_path"] = str(storage_path)
        if index_config is not None:
            kwargs["index_config"] = index_config
        if model:
            kwargs["model"] = model.strip()
        if retrieve_model:
            kwargs["retrieve_model"] = normalize_retrieve_model_name(retrieve_model)
    return PageIndexClient(**kwargs)


def get_pageindex_collection(client: Any, collection_name: str):
    return client.collection(collection_name)


def build_index_config(
    *,
    toc_check_page_num: int,
    max_page_num_each_node: int,
    max_token_num_each_node: int,
    if_add_node_id: bool,
    if_add_node_summary: bool,
    if_add_doc_description: bool,
    if_add_node_text: bool,
) -> IndexConfig:
    """Build PageIndex's official local indexing configuration."""
    return IndexConfig(
        toc_check_page_num=toc_check_page_num,
        max_page_num_each_node=max_page_num_each_node,
        max_token_num_each_node=max_token_num_each_node,
        if_add_node_id=if_add_node_id,
        if_add_node_summary=if_add_node_summary,
        if_add_doc_description=if_add_doc_description,
        if_add_node_text=if_add_node_text,
    )


def build_no_toc_retry_index_config(index_config: IndexConfig) -> IndexConfig:
    """Copy the active local IndexConfig but disable TOC detection for one retry."""
    return build_index_config(
        toc_check_page_num=0,
        max_page_num_each_node=int(index_config.max_page_num_each_node),
        max_token_num_each_node=int(index_config.max_token_num_each_node),
        if_add_node_id=bool(index_config.if_add_node_id),
        if_add_node_summary=bool(index_config.if_add_node_summary),
        if_add_doc_description=bool(index_config.if_add_doc_description),
        if_add_node_text=bool(index_config.if_add_node_text),
    )


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


def run_pageindex_query_non_stream(*, collection: Any, question: str, doc_ids: List[str]) -> str:
    """Run PageIndex non-streaming query from a context without an active event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        retry_result = collection.query(question, doc_ids=doc_ids or None, stream=False)
        if inspect.isawaitable(retry_result):
            retry_result = asyncio.run(retry_result)
        return pageindex_query_result_to_text(retry_result)

    raise RuntimeError(
        "run_pageindex_query_non_stream doit être exécuté hors de la boucle asyncio active."
    )


def _run_pageindex_query_non_stream_any_context(*, collection: Any, question: str, doc_ids: List[str]) -> str:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return run_pageindex_query_non_stream(collection=collection, question=question, doc_ids=doc_ids)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(
            run_pageindex_query_non_stream,
            collection=collection,
            question=question,
            doc_ids=doc_ids,
        ).result()


def run_pageindex_query(
    *,
    collection: Any,
    question: str,
    doc_ids: List[str],
    on_answer_delta: Callable[[str], None],
    on_trace: Callable[[dict], None],
    prefer_stream: bool,
) -> str:
    """Run PageIndex query, optionally streaming, with non-stream fallback."""

    prompt = build_pageindex_query_prompt(question)

    if not prefer_stream:
        on_trace({"type": "non_stream", "data": "Streaming désactivé: requête PageIndex non-stream directe."})
        return _run_pageindex_query_non_stream_any_context(
            collection=collection,
            question=prompt,
            doc_ids=doc_ids,
        ).strip()

    async def consume_stream() -> str:
        final_answer_parts: List[str] = []
        final_answer = ""
        try:
            stream = collection.query(prompt, doc_ids=doc_ids or None, stream=True)
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
                elif event_type in {"error", "exception"}:
                    raise RuntimeError(sanitize_error_text(data))
            return final_answer or "".join(final_answer_parts)
        except Exception as stream_exc:
            on_trace({"type": "stream_error", "data": sanitize_error_text(stream_exc)})
            try:
                retry_answer = await asyncio.to_thread(
                    run_pageindex_query_non_stream,
                    collection=collection,
                    question=prompt,
                    doc_ids=doc_ids,
                )
                if retry_answer:
                    on_trace({"type": "non_stream_retry", "data": "Réponse récupérée après échec du streaming."})
                    return retry_answer
                raise RuntimeError("Le retry PageIndex non-stream n'a retourné aucune réponse.")
            except Exception as retry_exc:
                raise RuntimeError(
                    build_pageindex_error_message(stream_error=stream_exc, retry_error=retry_exc)
                ) from retry_exc

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
default_llm_api_key = os.getenv("OPENAI_API_KEY", "")
default_llm_base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
default_disable_tracing = env_bool("OPENAI_AGENTS_DISABLE_TRACING", True)

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

    st.subheader("LLM local / OpenAI-compatible")
    st.caption(
        "Oui, les modèles open source peuvent fonctionner, mais l'endpoint doit "
        "supporter OpenAI Responses API + tool calling (requis par PageIndex/openai-agents)."
    )
    llm_api_key = st.text_input(
        "OPENAI_API_KEY",
        value=default_llm_api_key,
        type="password",
        help="Token envoyé au serveur LLM en mode local PageIndex.",
        disabled=bool(pageindex_api_key),
    )
    llm_base_url = st.text_input(
        "OPENAI_BASE_URL / OPENAI_API_BASE",
        value=default_llm_base_url,
        help="URL du serveur LLM OpenAI-compatible, par exemple https://api.openai.com/v1, http://localhost:8000/v1 ou http://localhost:11434/v1.",
        disabled=bool(pageindex_api_key),
    )

    st.subheader("Modèles locaux")
    pageindex_model = st.text_input(
        "PAGEINDEX_MODEL",
        value=default_model,
        help="Modèle LiteLLM utilisé pour construire l'index. Comme PageIndex dev, il est transmis tel quel; exemples: openai/gpt-4o-mini, ollama_chat/llama3.1, vllm/<model>.",
        disabled=bool(pageindex_api_key),
    )
    pageindex_retrieve_model = st.text_input(
        "PAGEINDEX_RETRIEVE_MODEL",
        value=default_retrieve_model,
        help="Modèle agentique utilisé par collection.query(...). Comme PageIndex dev, l'app ajoute litellm/ seulement aux chemins provider/model non OpenAI.",
        disabled=bool(pageindex_api_key),
    )

    with st.expander("IndexConfig PageIndex local", expanded=False):
        st.caption("Paramètres officiels de pageindex.config.IndexConfig, utilisés uniquement en mode local.")
        toc_check_page_num = st.number_input(
            "toc_check_page_num",
            min_value=0,
            value=int(os.getenv("PAGEINDEX_TOC_CHECK_PAGE_NUM", "20")),
            step=1,
            disabled=bool(pageindex_api_key),
        )
        max_page_num_each_node = st.number_input(
            "max_page_num_each_node",
            min_value=1,
            value=int(os.getenv("PAGEINDEX_MAX_PAGE_NUM_EACH_NODE", "10")),
            step=1,
            disabled=bool(pageindex_api_key),
        )
        max_token_num_each_node = st.number_input(
            "max_token_num_each_node",
            min_value=1,
            value=int(os.getenv("PAGEINDEX_MAX_TOKEN_NUM_EACH_NODE", "20000")),
            step=1000,
            disabled=bool(pageindex_api_key),
        )
        if_add_node_id = st.checkbox(
            "if_add_node_id",
            value=env_bool("PAGEINDEX_IF_ADD_NODE_ID", True),
            disabled=bool(pageindex_api_key),
        )
        if_add_node_summary = st.checkbox(
            "if_add_node_summary",
            value=env_bool("PAGEINDEX_IF_ADD_NODE_SUMMARY", True),
            disabled=bool(pageindex_api_key),
        )
        if_add_doc_description = st.checkbox(
            "if_add_doc_description",
            value=env_bool("PAGEINDEX_IF_ADD_DOC_DESCRIPTION", True),
            disabled=bool(pageindex_api_key),
        )
        if_add_node_text = st.checkbox(
            "if_add_node_text",
            value=env_bool("PAGEINDEX_IF_ADD_NODE_TEXT", False),
            help="Option avancée PageIndex; peut augmenter fortement la taille stockée.",
            disabled=bool(pageindex_api_key),
        )

    disable_tracing = st.checkbox(
        "Désactiver tracing OpenAI Agents",
        value=default_disable_tracing,
        help="Recommandé avec Ollama/vLLM/LiteLLM proxy : évite que l'Agents SDK tente d'envoyer des traces à OpenAI avec une clé locale/non-OpenAI.",
    )

    st.divider()
    prefer_stream = st.checkbox(
        "Activer streaming des réponses",
        value=env_bool("PAGEINDEX_STREAMING", False),
        help=(
            "Désactivé par défaut pour éviter une double requête lente quand le "
            "serveur local/OpenAI-compatible ne gère pas correctement le streaming "
            "PageIndex. Active-le seulement si ton endpoint supporte bien le streaming."
        ),
    )
    show_traces = st.checkbox("Afficher appels outils PageIndex", value=env_bool("PAGEINDEX_SHOW_TRACES", False))

uploaded_files = st.file_uploader(
    "Upload un ou plusieurs documents",
    type=["pdf", "md", "markdown"],
    accept_multiple_files=True,
)

local_index_config = build_index_config(
    toc_check_page_num=int(toc_check_page_num),
    max_page_num_each_node=int(max_page_num_each_node),
    max_token_num_each_node=int(max_token_num_each_node),
    if_add_node_id=if_add_node_id,
    if_add_node_summary=if_add_node_summary,
    if_add_doc_description=if_add_doc_description,
    if_add_node_text=if_add_node_text,
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
                        llm_api_key=llm_api_key,
                        llm_base_url=llm_base_url,
                        disable_tracing=disable_tracing,
                        index_config=local_index_config,
                    )
                    pageindex_collection = get_pageindex_collection(pageindex_client, collection_name)

                    for file in uploaded_files:
                        data = file.read()
                        stored_path = write_uploaded_document(storage_path, file.name, data)
                        active_collection = pageindex_collection
                        try:
                            doc_id = index_with_pageindex(active_collection, stored_path)
                        except Exception as index_exc:
                            can_retry_without_toc = (
                                not pageindex_api_key
                                and is_pageindex_processing_failure(index_exc)
                                and int(local_index_config.toc_check_page_num) != 0
                            )
                            if not can_retry_without_toc:
                                raise RuntimeError(
                                    build_pageindex_indexing_error_message(
                                        stored_path=stored_path,
                                        error=index_exc,
                                    )
                                ) from index_exc

                            st.warning(
                                f"{file.name}: PageIndex a renvoyé 'Processing failed'. "
                                "Retry automatique avec toc_check_page_num=0 "
                                "(détection TOC désactivée)."
                            )
                            retry_config = build_no_toc_retry_index_config(local_index_config)
                            retry_client = get_pageindex_client(
                                api_key=pageindex_api_key,
                                model=pageindex_model,
                                retrieve_model=pageindex_retrieve_model,
                                storage_path=storage_path,
                                llm_api_key=llm_api_key,
                                llm_base_url=llm_base_url,
                                disable_tracing=disable_tracing,
                                index_config=retry_config,
                            )
                            active_collection = get_pageindex_collection(retry_client, collection_name)
                            try:
                                doc_id = index_with_pageindex(active_collection, stored_path)
                            except Exception as retry_exc:
                                raise RuntimeError(
                                    build_pageindex_indexing_error_message(
                                        stored_path=stored_path,
                                        error=index_exc,
                                        retried_without_toc_detection=True,
                                        retry_error=retry_exc,
                                    )
                                ) from retry_exc

                        metadata = active_collection.get_document(doc_id)
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
                    st.error(f"Indexation PageIndex impossible : {sanitize_error_text(exc)}{build_openai_compatibility_hint(exc)}")
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
                        llm_api_key=llm_api_key,
                        llm_base_url=llm_base_url,
                        disable_tracing=disable_tracing,
                        index_config=local_index_config,
                    )
                    pageindex_collection = get_pageindex_collection(pageindex_client, collection_name)
                    answer = run_pageindex_query(
                        collection=pageindex_collection,
                        question=question,
                        doc_ids=doc_ids,
                        on_answer_delta=append_delta,
                        on_trace=append_trace,
                        prefer_stream=prefer_stream,
                    )
                    if answer and answer != nonlocal_answer["value"]:
                        placeholder.markdown(answer)
                    if any(trace.get("type") == "stream_error" for trace in traces):
                        st.warning("Le streaming PageIndex a échoué; une réponse non-streaming PageIndex a été utilisée.")
                except Exception as exc:
                    answer = ""
                    st.error(f"Erreur PageIndex : {sanitize_error_text(exc)}{build_openai_compatibility_hint(exc)}")

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
