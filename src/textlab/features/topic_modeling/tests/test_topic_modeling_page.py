"""The Topic Modeling page imports only the service, and nothing heavy.

The page is a Streamlit script and cannot be imported in tests, so its
imports are read from the source.
"""

import ast
import subprocess
import sys
from pathlib import Path

from textlab.common.jobs import worker_environment
from textlab.features.topic_modeling import service

PAGE = (
    Path(__file__).resolve().parents[3]
    / "ui"
    / "streamlit"
    / "pages"
    / "Topic_Modeling.py"
)


def page_tree():
    return ast.parse(PAGE.read_text(encoding="utf-8"))


def test_the_page_imports_only_the_service():
    for node in ast.walk(page_tree()):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "textlab.features"
        ):
            names = {alias.name for alias in node.names}
            if node.module == "textlab.features.topic_modeling":
                assert names == {"service"}
            else:
                assert node.module == "textlab.features.topic_modeling.service"
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


def test_the_service_does_not_import_the_modeling_libraries():
    """Opening the page must not load BERTopic, Top2Vec, gensim or spaCy."""
    code = (
        "import sys\n"
        "import textlab.features.topic_modeling.service\n"
        "heavy = {'bertopic', 'top2vec', 'gensim', 'spacy', 'nltk', 'umap'}\n"
        "print(sorted(heavy & set(sys.modules)))\n"
    )
    output = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=worker_environment(),
    ).stdout
    assert output.strip() == "[]"
