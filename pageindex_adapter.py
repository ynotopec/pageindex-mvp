import importlib
import os
import sys
from io import BytesIO
from types import SimpleNamespace
from typing import Any, List, Tuple


class _OfflineEncoding:
    def encode(self, text: str):
        if not text:
            return []
        # Deterministic offline approximation (~4 chars per token).
        return list(range(max(1, len(text) // 4)))


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
    return model or os.getenv("OPENAI_API_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-oss"


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


def _build_safe_count_tokens(original_count_tokens):
    def patched_count_tokens(text, model=None):
        try:
            return original_count_tokens(text, model=model)
        except Exception:
            if not text:
                return 0
            # Keep fallback fully offline and deterministic (no tokenizer downloads).
            # Approximation: ~4 chars/token for Latin text.
            return max(1, len(text) // 4)

    return patched_count_tokens


def _patch_pageindex_token_counter(pageindex_utils: Any, page_index_main: Any) -> None:
    if pageindex_utils is None or not hasattr(pageindex_utils, "count_tokens"):
        return

    original_count_tokens = pageindex_utils.count_tokens
    patched_count_tokens = _build_safe_count_tokens(original_count_tokens)
    pageindex_utils.count_tokens = patched_count_tokens

    # `pageindex.page_index` imports `count_tokens` with `from .utils import *`.
    # Replace its module-level reference too, otherwise KeyError can still occur.
    try:
        if hasattr(page_index_main, "__globals__") and "count_tokens" in page_index_main.__globals__:
            page_index_main.__globals__["count_tokens"] = patched_count_tokens
    except Exception:
        pass


def _patch_pageindex_tokenizer_mapping(pageindex_utils: Any, page_index_main: Any) -> None:
    """
    Some PageIndex versions call tiktoken.encoding_for_model directly.
    Patch that path to avoid KeyError with open-source model names.
    """

    def patch_module_tiktoken(module_like: Any) -> None:
        if module_like is None or not hasattr(module_like, "tiktoken"):
            return
        tk = module_like.tiktoken
        if not hasattr(tk, "encoding_for_model"):
            return

        original = tk.encoding_for_model

        def safe_encoding_for_model(model_name):
            try:
                return original(model_name)
            except Exception:
                return _OfflineEncoding()

        tk.encoding_for_model = safe_encoding_for_model

    patch_module_tiktoken(pageindex_utils)

    try:
        if hasattr(page_index_main, "__globals__") and "tiktoken" in page_index_main.__globals__:
            tk = page_index_main.__globals__["tiktoken"]
            if hasattr(tk, "encoding_for_model"):
                original = tk.encoding_for_model

                def safe_encoding_for_model(model_name):
                    try:
                        return original(model_name)
                    except Exception:
                        return _OfflineEncoding()

                tk.encoding_for_model = safe_encoding_for_model
    except Exception:
        pass


def build_pageindex_chunks(
    uploaded_files: List[Any],
    model: str,
    api_key: str,
    openai_base_url: str,
) -> Tuple[List[dict], List[dict], List[str], List[str]]:
    page_index_main, config, pageindex_utils = load_pageindex_lib()

    os.environ["OPENAI_API_MODEL"] = model
    os.environ["CHATGPT_API_KEY"] = api_key
    os.environ["OPENAI_API_KEY"] = api_key
    if openai_base_url.strip():
        os.environ["OPENAI_BASE_URL"] = openai_base_url
        os.environ["OPENAI_API_BASE"] = openai_base_url

    _patch_pageindex_openai(pageindex_utils, api_key=api_key, openai_base_url=openai_base_url)
    _patch_pageindex_token_counter(pageindex_utils, page_index_main=page_index_main)
    _patch_pageindex_tokenizer_mapping(pageindex_utils, page_index_main=page_index_main)

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
            # Retry once with an alternate model if provided in env.
            retry_model = os.getenv("OPENAI_API_MODEL") or os.getenv("OPENAI_MODEL") or ""
            if retry_model != opt.model:
                retry_opt = config(
                    model=retry_model,
                    toc_check_page_num=20,
                    max_page_num_each_node=10,
                    max_token_num_each_node=20000,
                    if_add_node_id="yes",
                    if_add_node_summary="yes",
                    if_add_doc_description="no",
                    if_add_node_text="yes",
                )
                try:
                    retry_result = page_index_main(BytesIO(raw_pdf), retry_opt)
                    retry_structure = retry_result.get("structure", [])

                    def visit_retry(nodes: List[dict]) -> None:
                        for node in nodes:
                            node_text = (node.get("text") or "").strip()
                            node_summary = (node.get("summary") or "").strip()
                            if node_text:
                                node_chunks.append(node_text)
                            elif node_summary:
                                node_chunks.append(node_summary)
                            children = node.get("nodes") or []
                            if children:
                                visit_retry(children)

                    visit_retry(retry_structure)
                    warnings.append(
                        f"PageIndex a échoué pour '{file.name}' avec '{opt.model}'. "
                        f"Relance réussie avec '{retry_model}'."
                    )
                    e = None
                except Exception as retry_err:
                    e = retry_err

            if not node_chunks:
                warnings.append(
                    f"PageIndex a échoué pour '{file.name}' ({type(e).__name__}: {e}). "
                    "Document ignoré (pas de fallback lexical)."
                )

        if not node_chunks:
            continue

        docs.append({"name": file.name, "text_chars": sum(len(c) for c in node_chunks), "chunks": len(node_chunks)})
        all_texts.append(f"===== {file.name} =====\n" + "\n\n".join(node_chunks))

        for idx, ch in enumerate(node_chunks, start=1):
            all_chunks.append({"doc_name": file.name, "chunk_id": idx, "text": ch})

    return docs, all_chunks, all_texts, warnings
