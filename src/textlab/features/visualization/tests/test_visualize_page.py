"""The Visualize Data page imports only the service, and no agent code.

The page is a Streamlit script and cannot be imported in tests, so its
imports are read from the source.
"""

import ast
import subprocess
import sys
from pathlib import Path

from textlab.common.jobs import worker_environment
from textlab.features.visualization import service

PAGE = (
    Path(__file__).resolve().parents[3]
    / "ui"
    / "streamlit"
    / "pages"
    / "Visualize_Data.py"
)


def page_tree():
    return ast.parse(PAGE.read_text(encoding="utf-8"))


def test_the_page_imports_only_the_service():
    for node in ast.walk(page_tree()):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "textlab.features"
        ):
            assert node.module == "textlab.features.visualization"
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


def test_the_services_do_not_load_the_agents():
    """Opening Visualize Data or Chat must not import the agent or MCP."""
    code = (
        "import sys\n"
        "import textlab.features.visualization.service\n"
        "import textlab.features.chat.service\n"
        "heavy = {'mcp', 'textlab.features.visualization.viz_agent'}\n"
        "print(sorted(heavy & set(sys.modules)))\n"
    )
    output = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=worker_environment(),
    ).stdout
    # Libraries may print warnings first (PyMuPDF does when imported as
    # fitz); the answer is the last line.
    assert output.strip().splitlines()[-1] == "[]"
