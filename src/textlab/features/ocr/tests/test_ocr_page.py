"""The OCR page and its parts import only what the feature provides.

The page is a Streamlit script and cannot be imported in tests, so the
imports of the page and of ``ui/streamlit/ocr/`` are read from the source.
"""

import ast
from pathlib import Path

from textlab.features.ocr import service

STREAMLIT = Path(__file__).resolve().parents[3] / "ui" / "streamlit"
SOURCES = [STREAMLIT / "pages" / "OCR.py", *(STREAMLIT / "ocr").glob("*.py")]


def parse(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def feature_imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "textlab.features"
        ):
            for alias in node.names:
                yield node.module, alias.name


def test_only_the_service_and_the_result_model_are_imported():
    for path in SOURCES:
        for module, name in feature_imports(parse(path)):
            assert module == "textlab.features.ocr", (path.name, module)
            assert name in {"service", "doc_ir"}, (path.name, name)


def test_only_names_the_service_exports_are_used():
    for path in SOURCES:
        used = {
            node.attr
            for node in ast.walk(parse(path))
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "service"
        }
        missing = used - set(service.__all__)
        assert not missing, (path.name, missing)
