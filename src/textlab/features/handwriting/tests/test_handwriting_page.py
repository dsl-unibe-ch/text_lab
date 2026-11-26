"""The handwriting mode of the OCR page uses only the feature's service.

Streamlit scripts cannot be imported in tests, so the sources of the
handwriting panel and of the OCR page are read instead.
"""

import ast
from pathlib import Path

from textlab.features.handwriting import service

STREAMLIT = Path(__file__).resolve().parents[3] / "ui" / "streamlit"
PANEL = STREAMLIT / "handwriting" / "panel.py"
OCR_PAGE = STREAMLIT / "pages" / "OCR.py"


def tree(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def test_only_the_service_is_imported():
    for node in ast.walk(tree(PANEL)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "textlab.features"
        ):
            assert node.module == "textlab.features.handwriting"
            assert [alias.name for alias in node.names] == ["service"]


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


def test_the_panel_does_not_use_the_ocr_feature():
    assert "features.ocr" not in PANEL.read_text(encoding="utf-8")


def test_there_is_no_separate_page():
    assert not (STREAMLIT / "pages" / "Historic_Handwriting.py").exists()


def test_the_ocr_page_draws_it_below_the_legacy_engines():
    """The OCR page's own workflows come first and are unchanged.

    The Hugging Face OCR models tool follows below it.
    """
    calls = [
        node.value.func.id
        for node in tree(OCR_PAGE).body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
    ]
    assert calls[-3:] == [
        "manual_engines_expander",
        "handwriting_expander",
        "hf_ocr_expander",
    ]
