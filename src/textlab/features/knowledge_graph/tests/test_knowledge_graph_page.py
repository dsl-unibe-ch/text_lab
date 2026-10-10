"""The Knowledge Graph page imports only the service, and nothing heavy.

The page is a Streamlit script and cannot be imported in tests, so its
imports are read from the source.
"""

import ast
import subprocess
import sys
from pathlib import Path

from textlab.common.jobs import worker_environment
from textlab.features.knowledge_graph import service

PAGE = (
    Path(__file__).resolve().parents[3]
    / "ui"
    / "streamlit"
    / "pages"
    / "Knowledge_Graph.py"
)


def page_tree():
    return ast.parse(PAGE.read_text(encoding="utf-8"))


def test_the_page_imports_only_the_service():
    for node in ast.walk(page_tree()):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "textlab.features"
        ):
            names = {alias.name for alias in node.names}
            if node.module == "textlab.features.knowledge_graph":
                assert names == {"service"}
            else:
                assert (
                    node.module == "textlab.features.knowledge_graph.service"
                )
                assert names <= set(service.__all__), names


def test_the_page_uses_only_names_the_service_exports():
    used = {
        node.attr
        for node in ast.walk(page_tree())
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "service"
    }
    assert used <= set(service.__all__), used - set(service.__all__)


def test_the_service_does_not_import_the_graph_libraries():
    """Opening the page must not load NetworkX, Pyvis or the OpenAI client."""
    code = (
        "import sys\n"
        "import textlab.features.knowledge_graph.service\n"
        "heavy = {'networkx', 'pyvis', 'openai'}\n"
        "print(sorted(heavy & set(sys.modules)))\n"
    )
    output = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=worker_environment(),
    ).stdout
    # Libraries may print warnings first; the answer is the last line.
    assert output.strip().splitlines()[-1] == "[]"
