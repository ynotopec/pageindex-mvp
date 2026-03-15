import importlib
import os
import sys
from io import BytesIO
from types import SimpleNamespace
from typing import Any, List, Tuple

import tiktoken

from rag_core import chunk_text, pdf_bytes_to_text


def load_pageindex_lib() -> Tuple[Any, Any, Any]:
    page_index_main = None
    config_factory = None
    pageindex_utils = None

    try:
        pkg = importlib.import_module("pageindex")
        page_index_main = getattr(pkg, "page_index_main", None)
    except Exception:
        pass

    if page_index_main is None:
        try:
            mod = importlib.import_module("pageindex.page_index")
            page_index_main = getattr(mod, "page_index_main", None)
        except Exception:
            pass

    try:
        pageindex_utils = importlib.import_module("pageindex.utils")
        config_factory = getattr(pageindex_utils, "config", None)
    except Exception:
        pageindex_utils = None

    if config_factory is None:
        config_factory = SimpleNamespace

    if page_index_main is None:
        old_root = os.path.join(os.path.dirname(__file__), "old")
        if old_root not in sys.path:
            sys.path.insert(0, old_root)

        cached = {
            name: module
            for name, module in list(sys.modules.items())
            if name == "pageindex" or name.startswith("pageindex.")
        }
        for name in cached:
            sys.modules.pop(name, None)

        try:
            legacy_mod = importlib.import_module("pageindex.page_index")
            page_index_main = getattr(legacy_mod, "page_index_main", None)
            if pageindex_utils is None:
                pageindex_utils = importlib.import_module("pageindex.utils")
        finally:
            for name, module in cached.items():
                sys.modules.setdefault(name, module)

        if pageindex_utils is not None and getattr(pageindex_utils, "config", None) is not None:
            config_factory = pageindex_utils.config

    if page_index_main is None:
        raise ImportError("`page_index_main` introuvable dans `pageindex`.")

    return page_index_main, config_factory, pageindex_utils


def _safe_pageindex_model(model: str) -> str:
    try:
        tiktoken.encoding_for_model(model)
        return model
    except KeyError:
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _patch_pageindex_openai(pageindex_utils: Any, api_key: str, openai_base_url: str) -> None:
    if pageindex_utils is None or not hasattr(pageindex_utils, "openai"):
        return

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


def build_pageindex_chunks(
    uploaded_files: List[Any],
    model: str,
    api_key: str,
    openai_base_url: str,
    lexical_max_chars: int,
    lexical_overlap: int,
) -> Tuple[List[dict], List[dict], List[str], List[str]]:
    page_index_main, config, pageindex_utils = load_pageindex_lib()

    os.environ["CHATGPT_API_KEY"] = api_key
    if openai_base_url.strip():
        os.environ["OPENAI_BASE_URL"] = openai_base_url

    _patch_pageindex_openai(pageindex_utils, api_key=api_key, openai_base_url=openai_base_url)

    opt = config(
        model=_safe_pageindex_model(model),
        toc_check_page_num=20,
        max_page_num_each_node=10,
        max_token_num_each_node=20000,
        if_add_node_id="yes",
        if_add_node_summary="yes",
        if_add_doc_description="no",
        if_add_node_text="yes",
    )

    docs: List[dict] = []
    all_chunks: List[dict] = []
    all_texts: List[str] = []
    warnings: List[str] = []

    for file in uploaded_files:
        raw_pdf = file.read()
        if not raw_pdf:
            continue

        node_chunks: List[str] = []
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
            fallback_text = pdf_bytes_to_text(raw_pdf)
            node_chunks = chunk_text(fallback_text, max_chars=lexical_max_chars, overlap=lexical_overlap)
            if node_chunks:
                warnings.append(
                    f"PageIndex a échoué pour '{file.name}' ({type(e).__name__}: {e}). "
                    "Fallback automatique vers chunking lexical."
                )
            else:
                warnings.append(
                    f"PageIndex a échoué pour '{file.name}' ({type(e).__name__}: {e}) "
                    "et le fallback lexical n'a extrait aucun texte."
                )

        docs.append({"name": file.name, "text_chars": sum(len(c) for c in node_chunks), "chunks": len(node_chunks)})
        all_texts.append(f"===== {file.name} =====\n" + "\n\n".join(node_chunks))

        for idx, ch in enumerate(node_chunks, start=1):
            all_chunks.append({"doc_name": file.name, "chunk_id": idx, "text": ch})

    return docs, all_chunks, all_texts, warnings
