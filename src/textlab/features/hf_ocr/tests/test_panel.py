"""The tool's panel on the OCR page uses only the feature's public API.

Streamlit scripts cannot be imported in tests, so the panel's source is
read instead.
"""

import ast
from pathlib import Path

from textlab.features.hf_ocr import service

STREAMLIT = Path(__file__).resolve().parents[3] / "ui" / "streamlit"
PANEL = STREAMLIT / "hf_ocr" / "panel.py"


def tree(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def test_only_the_service_and_the_store_are_imported():
    for node in ast.walk(tree(PANEL)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "textlab.features"
        ):
            assert node.module == "textlab.features.hf_ocr"
            assert sorted(alias.name for alias in node.names) == [
                "service",
                "store",
            ]


def test_only_names_the_service_exports_are_used():
    used = {
        node.attr
        for node in ast.walk(tree(PANEL))
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "service"
    }
    assert used
    assert not used - set(service.__all__)


def test_the_panel_uses_no_other_ocr_feature():
    source = PANEL.read_text(encoding="utf-8")
    assert "features.ocr" not in source
    assert "features.handwriting" not in source
