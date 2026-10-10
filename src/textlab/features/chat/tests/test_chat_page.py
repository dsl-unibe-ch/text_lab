"""The Chat page imports only the Chat service.

The page is a Streamlit script and cannot be imported in tests, so its
imports are read from the source.
"""

import ast
from pathlib import Path

from textlab.features.chat import service

PAGE = (
    Path(__file__).resolve().parents[3]
    / "ui"
    / "streamlit"
    / "pages"
    / "Chat.py"
)


def page_tree():
    return ast.parse(PAGE.read_text(encoding="utf-8"))


def test_the_page_imports_only_the_service():
    for node in ast.walk(page_tree()):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "textlab.features"
        ):
            assert node.module == "textlab.features.chat"
            assert [alias.name for alias in node.names] == ["service"]


def test_the_page_uses_only_names_the_service_exports():
    used = {
        node.attr
        for node in ast.walk(page_tree())
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "service"
    }
    assert used <= set(service.__all__), used - set(service.__all__)
