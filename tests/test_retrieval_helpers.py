import ast
import unittest
from pathlib import Path


HELPER_NAMES = {
    "tokenize",
    "score_chunk",
    "top_k_chunks",
    "build_matched_lexical_context",
    "build_lexical_context",
    "append_context_part",
}


def load_helpers():
    source = Path(__file__).resolve().parents[1] / "app.py"
    tree = ast.parse(source.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in HELPER_NAMES]
    module = ast.Module(
        body=[
            ast.Import(names=[ast.alias(name="re")]),
            ast.ImportFrom(
                module="typing",
                names=[
                    ast.alias(name="List"),
                    ast.alias(name="Tuple"),
                ],
                level=0,
            ),
            *selected,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


class RetrievalHelperTests(unittest.TestCase):
    def test_matched_lexical_context_keeps_multiple_relevant_chunks(self):
        helpers = load_helpers()
        chunks = [
            {"doc_name": "sample.pdf", "chunk_id": 1, "text": "Alpha est un élément de conformité."},
            {"doc_name": "sample.pdf", "chunk_id": 2, "text": "Beta est un autre élément de conformité."},
            {"doc_name": "sample.pdf", "chunk_id": 3, "text": "Gamma est le troisième élément de conformité."},
        ]

        context, traces = helpers["build_matched_lexical_context"](
            "Quels sont les éléments de conformité ?",
            chunks,
            k=3,
            mode="pageindex_lexical_supplement",
        )

        self.assertIn("Alpha", context)
        self.assertIn("Beta", context)
        self.assertIn("Gamma", context)
        self.assertEqual(
            [trace["mode"] for trace in traces],
            ["pageindex_lexical_supplement"] * 3,
        )

    def test_matched_lexical_context_does_not_fallback_to_unrelated_start(self):
        helpers = load_helpers()
        context, traces = helpers["build_matched_lexical_context"](
            "zèbre introuvable",
            [{"doc_name": "sample.pdf", "chunk_id": 1, "text": "Première page générique."}],
            k=3,
        )

        self.assertEqual(context, "")
        self.assertEqual(traces, [])

    def test_append_context_part_applies_global_limit(self):
        helpers = load_helpers()
        context = helpers["append_context_part"]("abc", "defghijkl", max_chars=8)

        self.assertTrue(context.startswith("abc\n\nde"))
        self.assertTrue(context.endswith("... [contexte tronqué]"))


if __name__ == "__main__":
    unittest.main()
