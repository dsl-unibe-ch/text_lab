"""The Translate page imports only names the feature provides.

The page is a Streamlit script and cannot be imported in tests, so its
imports are read from the source and resolved here.
"""

import ast
import importlib
from pathlib import Path

PAGE = (
    Path(__file__).resolve().parents[3]
    / "ui"
    / "streamlit"
    / "pages"
    / "Translate.py"
)


def feature_imports():
    module = ast.parse(PAGE.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "textlab.features.translation"
        ):
            for alias in node.names:
                yield node.module, alias.name


def test_page_imports_resolve():
    for module_name, name in feature_imports():
        module = importlib.import_module(module_name)
        if not hasattr(module, name):  # A submodule, such as ``service``.
            importlib.import_module(f"{module_name}.{name}")


def test_page_imports_only_the_service():
    assert list(feature_imports()) == [
        ("textlab.features.translation", "service")
    ]


def test_page_uses_only_names_the_service_exports():
    from textlab.features.translation import service

    used = {
        node.attr
        for node in ast.walk(ast.parse(PAGE.read_text(encoding="utf-8")))
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "service"
    }
    assert used <= set(service.__all__), used - set(service.__all__)
