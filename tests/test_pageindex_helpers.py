import ast
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
}


def load_helpers():
    source = Path(__file__).resolve().parents[1] / "app.py"
    tree = ast.parse(source.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in HELPER_NAMES]
    module = ast.Module(
        body=[
            ast.Import(names=[ast.alias(name="hashlib")]),
            ast.Import(names=[ast.alias(name="json")]),
            ast.Import(names=[ast.alias(name="os")]),
            ast.Import(names=[ast.alias(name="re")]),
            ast.ImportFrom(module="pathlib", names=[ast.alias(name="Path")], level=0),
            ast.ImportFrom(module="typing", names=[ast.alias(name="Any"), ast.alias(name="Dict"), ast.alias(name="Optional")], level=0),
            *selected,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {}
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
        original = {key: os.environ.get(key) for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE")}
        try:
            updates = helpers["configure_llm_environment"](
                llm_api_key="sk-test",
                llm_base_url="http://localhost:8000/v1/",
            )

            self.assertEqual(updates["OPENAI_API_KEY"], "sk-test")
            self.assertEqual(updates["OPENAI_BASE_URL"], "http://localhost:8000/v1")
            self.assertEqual(updates["OPENAI_API_BASE"], "http://localhost:8000/v1")
            self.assertEqual(os.environ["OPENAI_API_BASE"], "http://localhost:8000/v1")
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
