import ast
import asyncio
import inspect
import os
import unittest
from pathlib import Path


HELPER_NAMES = {
    "safe_filename",
    "file_digest",
    "pageindex_value_to_text",
    "compact_text",
    "_doc_name_from_metadata",
    "_doc_id_from_metadata",
    "configure_llm_environment",
    "sanitize_error_text",
    "pageindex_query_result_to_text",
    "build_pageindex_error_message",
    "is_internal_server_error",
    "build_openai_compatibility_hint",
    "is_pageindex_processing_failure",
    "build_pageindex_indexing_error_message",
    "build_no_toc_retry_index_config",
    "run_pageindex_query_non_stream",
    "normalize_retrieve_model_name",
    "build_pageindex_query_prompt",
    "build_index_config",
    "_run_pageindex_query_non_stream_any_context",
    "run_pageindex_query",
}


def load_helpers():
    source = Path(__file__).resolve().parents[1] / "app.py"
    tree = ast.parse(source.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in HELPER_NAMES]
    module = ast.Module(
        body=[
            ast.Import(names=[ast.alias(name="asyncio")]),
            ast.Import(names=[ast.alias(name="hashlib")]),
            ast.Import(names=[ast.alias(name="inspect")]),
            ast.Import(names=[ast.alias(name="json")]),
            ast.Import(names=[ast.alias(name="os")]),
            ast.Import(names=[ast.alias(name="re")]),
            ast.Import(names=[ast.alias(name="concurrent.futures")]),
            ast.ImportFrom(module="pathlib", names=[ast.alias(name="Path")], level=0),
            ast.ImportFrom(module="typing", names=[ast.alias(name="Any"), ast.alias(name="Callable"), ast.alias(name="Dict"), ast.alias(name="List"), ast.alias(name="Optional")], level=0),
            *selected,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    class FakeIndexConfig:
        def __init__(self, **kwargs):
            defaults = {
                "toc_check_page_num": 20,
                "max_page_num_each_node": 10,
                "max_token_num_each_node": 20000,
                "if_add_node_id": True,
                "if_add_node_summary": True,
                "if_add_doc_description": True,
                "if_add_node_text": False,
            }
            defaults.update(kwargs)
            for key, value in defaults.items():
                setattr(self, key, value)

    namespace = {"set_tracing_disabled": lambda disabled: None, "IndexConfig": FakeIndexConfig}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


class PageIndexHelperTests(unittest.TestCase):
    def test_safe_filename_removes_path_and_unsafe_characters(self):
        helpers = load_helpers()

        self.assertEqual(helpers["safe_filename"]("../Rapport final éco.pdf"), "Rapport_final_co.pdf")

    def test_file_digest_is_stable_and_short(self):
        helpers = load_helpers()

        self.assertEqual(helpers["file_digest"](b"abc"), helpers["file_digest"](b"abc"))
        self.assertEqual(len(helpers["file_digest"](b"abc")), 16)

    def test_pageindex_value_to_text_serializes_json_values(self):
        helpers = load_helpers()

        text = helpers["pageindex_value_to_text"]([{"page": 1, "content": "Bonjour"}])

        self.assertIn('"page": 1', text)
        self.assertIn("Bonjour", text)

    def test_doc_metadata_helpers_accept_pageindex_shapes(self):
        helpers = load_helpers()
        metadata = {"doc_id": "doc-1", "file_path": "/tmp/report.pdf"}

        self.assertEqual(helpers["_doc_id_from_metadata"](metadata), "doc-1")
        self.assertEqual(helpers["_doc_name_from_metadata"](metadata), "report.pdf")

    def test_compact_text_truncates_long_values(self):
        helpers = load_helpers()

        text = helpers["compact_text"]("abcdef", limit=3)

        self.assertEqual(text, "abc\n... [sortie tronquée]")

    def test_configure_llm_environment_sets_base_url_aliases(self):
        helpers = load_helpers()
        original = {
            key: os.environ.get(key)
            for key in (
                "OPENAI_API_KEY",
                "OPENAI_BASE_URL",
                "OPENAI_API_BASE",
                "OPENAI_AGENTS_DISABLE_TRACING",
            )
        }
        try:
            updates = helpers["configure_llm_environment"](
                llm_api_key="sk-test",
                llm_base_url="http://localhost:8000/v1/",
                disable_tracing=True,
            )

            self.assertEqual(updates["OPENAI_API_KEY"], "sk-test")
            self.assertEqual(updates["OPENAI_BASE_URL"], "http://localhost:8000/v1")
            self.assertEqual(updates["OPENAI_API_BASE"], "http://localhost:8000/v1")
            self.assertEqual(updates["OPENAI_AGENTS_DISABLE_TRACING"], "1")
            self.assertEqual(os.environ["OPENAI_API_BASE"], "http://localhost:8000/v1")
            self.assertEqual(os.environ["OPENAI_AGENTS_DISABLE_TRACING"], "1")
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_configure_llm_environment_can_enable_tracing(self):
        helpers = load_helpers()
        original = os.environ.get("OPENAI_AGENTS_DISABLE_TRACING")
        try:
            updates = helpers["configure_llm_environment"](
                llm_api_key="",
                llm_base_url="",
                disable_tracing=False,
            )

            self.assertEqual(updates["OPENAI_AGENTS_DISABLE_TRACING"], "0")
            self.assertEqual(os.environ["OPENAI_AGENTS_DISABLE_TRACING"], "0")
        finally:
            if original is None:
                os.environ.pop("OPENAI_AGENTS_DISABLE_TRACING", None)
            else:
                os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = original

    def test_sanitize_error_text_redacts_api_keys(self):
        helpers = load_helpers()

        text = helpers["sanitize_error_text"]("Incorrect API key provided: sk-secret123456")

        self.assertIn("[redacted]", text)
        self.assertNotIn("sk-secret123456", text)

    def test_pageindex_query_result_to_text_extracts_answer_shapes(self):
        helpers = load_helpers()

        self.assertEqual(helpers["pageindex_query_result_to_text"]({"answer": "Bonjour"}), "Bonjour")

    def test_build_pageindex_error_message_includes_provider_hint(self):
        helpers = load_helpers()

        text = helpers["build_pageindex_error_message"](stream_error=RuntimeError("Internal Server Error"))

        self.assertIn("OPENAI_BASE_URL", text)
        self.assertIn("Provider List", text)

    def test_openai_compatibility_hint_detects_internal_server_error(self):
        helpers = load_helpers()

        exc = RuntimeError("Internal Server Error")
        hint = helpers["build_openai_compatibility_hint"](exc)

        self.assertTrue(helpers["is_internal_server_error"](exc))
        self.assertIn("Responses", hint)
        self.assertIn("tool", hint)

    def test_build_pageindex_error_message_includes_compatibility_hint_on_500(self):
        helpers = load_helpers()

        text = helpers["build_pageindex_error_message"](stream_error=RuntimeError("Internal Server Error"))

        self.assertIn("OpenAI Responses", text)
        self.assertIn("OPENAI_BASE_URL", text)


    def test_processing_failed_indexing_message_recommends_no_toc_retry(self):
        helpers = load_helpers()

        error = RuntimeError("Failed to index /tmp/report.pdf: Processing failed")
        text = helpers["build_pageindex_indexing_error_message"](
            stored_path=Path("/tmp/report.pdf"),
            error=error,
        )

        self.assertTrue(helpers["is_pageindex_processing_failure"](error))
        self.assertIn("toc_check_page_num=0", text)
        self.assertIn("sans TOC", text)

    def test_retry_index_config_only_disables_toc_detection(self):
        helpers = load_helpers()

        config = helpers["build_index_config"](
            toc_check_page_num=20,
            max_page_num_each_node=8,
            max_token_num_each_node=12000,
            if_add_node_id=True,
            if_add_node_summary=False,
            if_add_doc_description=True,
            if_add_node_text=False,
        )
        retry_config = helpers["build_no_toc_retry_index_config"](config)

        self.assertEqual(retry_config.toc_check_page_num, 0)
        self.assertEqual(retry_config.max_page_num_each_node, 8)
        self.assertEqual(retry_config.max_token_num_each_node, 12000)
        self.assertTrue(retry_config.if_add_node_id)
        self.assertFalse(retry_config.if_add_node_summary)
        self.assertTrue(retry_config.if_add_doc_description)
        self.assertFalse(retry_config.if_add_node_text)

    def test_non_stream_query_helper_requires_no_running_loop(self):
        helpers = load_helpers()

        class FakeCollection:
            def query(self, question, doc_ids=None, stream=False):
                self.question = question
                self.doc_ids = doc_ids
                self.stream = stream
                return {"answer": "Réponse non-stream"}

        collection = FakeCollection()
        answer = helpers["run_pageindex_query_non_stream"](
            collection=collection,
            question="Question ?",
            doc_ids=["doc-1"],
        )

        self.assertEqual(answer, "Réponse non-stream")
        self.assertEqual(collection.question, "Question ?")
        self.assertEqual(collection.doc_ids, ["doc-1"])
        self.assertFalse(collection.stream)

    def test_non_stream_query_helper_rejects_running_event_loop(self):
        helpers = load_helpers()

        async def call_helper():
            with self.assertRaisesRegex(RuntimeError, "hors de la boucle asyncio"):
                helpers["run_pageindex_query_non_stream"](
                    collection=object(),
                    question="Question ?",
                    doc_ids=[],
                )

        asyncio.run(call_helper())

    def test_build_pageindex_query_prompt_enforces_tight_pageindex_retrieval(self):
        helpers = load_helpers()

        text = helpers["build_pageindex_query_prompt"]("Quel est le revenu ?")

        self.assertIn("Question utilisateur : Quel est le revenu ?", text)
        self.assertIn("plages serrées", text)
        self.assertIn("exclusivement sur le contenu récupéré", text)

    def test_run_pageindex_query_can_skip_streaming(self):
        helpers = load_helpers()

        class FakeCollection:
            def query(self, question, doc_ids=None, stream=False):
                self.question = question
                self.doc_ids = doc_ids
                self.stream = stream
                return "Réponse directe"

        traces = []
        answer = helpers["run_pageindex_query"](
            collection=FakeCollection(),
            question="Question ?",
            doc_ids=["doc-1"],
            on_answer_delta=lambda delta: None,
            on_trace=traces.append,
            prefer_stream=False,
        )

        self.assertEqual(answer, "Réponse directe")
        self.assertEqual(traces[0]["type"], "non_stream")

    def test_normalize_retrieve_model_name_matches_pageindex_dev(self):
        helpers = load_helpers()

        self.assertEqual(
            helpers["normalize_retrieve_model_name"]("openai/gpt-4o-mini"),
            "openai/gpt-4o-mini",
        )
        self.assertEqual(
            helpers["normalize_retrieve_model_name"]("ollama_chat/llama3.1"),
            "litellm/ollama_chat/llama3.1",
        )
        self.assertEqual(
            helpers["normalize_retrieve_model_name"]("litellm/vllm/model"),
            "litellm/vllm/model",
        )
        self.assertEqual(helpers["normalize_retrieve_model_name"]("gpt-4o-mini"), "gpt-4o-mini")
        self.assertEqual(helpers["normalize_retrieve_model_name"](""), "")

    def test_build_index_config_uses_pageindex_dev_fields(self):
        helpers = load_helpers()

        config = helpers["build_index_config"](
            toc_check_page_num=30,
            max_page_num_each_node=8,
            max_token_num_each_node=12000,
            if_add_node_id=True,
            if_add_node_summary=False,
            if_add_doc_description=True,
            if_add_node_text=False,
        )

        self.assertEqual(config.toc_check_page_num, 30)
        self.assertEqual(config.max_page_num_each_node, 8)
        self.assertEqual(config.max_token_num_each_node, 12000)
        self.assertTrue(config.if_add_node_id)
        self.assertFalse(config.if_add_node_summary)
        self.assertTrue(config.if_add_doc_description)
        self.assertFalse(config.if_add_node_text)


if __name__ == "__main__":
    unittest.main()
