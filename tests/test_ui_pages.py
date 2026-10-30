"""Every Streamlit page shows the disclaimer about AI output.

The pages are Streamlit scripts and cannot be imported in tests, so their
calls are read from the source.
"""

import ast
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parents[1] / "src/textlab/ui/streamlit"
PAGES = [UI / "Home.py", *sorted((UI / "pages").glob("*.py"))]


def called_names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.name)
def test_every_page_shows_the_disclaimer(page):
    calls = called_names(page)
    assert calls.count("render_disclaimer") == 1
    assert calls.count("check_token") == 1
