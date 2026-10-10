"""User data stays in the job workspace, and site values stay in site.env.

Text Lab promises that uploaded files and results are not left behind after
a session. Temporary files therefore go into the per-job workspace
(``textlab.common.storage``), never into the user's home directory. These
checks scan the source for the patterns that broke that promise before:
paths built from the home directory, and cluster-specific paths and host
names hardcoded in the code instead of the site configuration.

A use that is legitimate is listed in an allow-list together with the
reason. An entry that no longer matches anything fails the test as well, so
the lists stay accurate.
"""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "textlab"

#: Files allowed to use the home directory, with the reason.
HOME_ALLOWED = {
    "features/topic_modeling/topic_utils.py": (
        "custom embedding models the user picks are downloaded to their own "
        "Hugging Face cache; these are model files, not user data"
    ),
    "ui/streamlit/pages/OCR.py": (
        "PaddleX model cache default (~/.paddlex); the launch script mounts "
        "the shared model store there"
    ),
    "ui/streamlit/pages/Knowledge_Graph.py": (
        "suggests ~/papers as the folder holding the user's own papers and "
        "lets the user choose where to save the output"
    ),
}

#: Files allowed to contain site-specific strings, with the reason.
SITE_ALLOWED = {
    "ui/streamlit/Home.py": (
        "support e-mail, documentation and mailing-list links on the home "
        "page; to be moved to the site configuration"
    ),
    "ui/streamlit/pages/Translate.py": (
        "page caption naming UBELIX; reworded when Translate is refactored"
    ),
}

#: Strings that identify the UBELIX deployment.
SITE_MARKERS = ("/storage/", "unibe.ch", "ubelix")


def source_files():
    """Yield (relative path, syntax tree) for every non-test module."""
    for path in sorted(PACKAGE.rglob("*.py")):
        if "tests" in path.relative_to(PACKAGE).parts:
            continue
        rel = path.relative_to(PACKAGE).as_posix()
        yield rel, ast.parse(path.read_text(encoding="utf-8"))


def _is_home_use(node):
    """True for ``Path.home()``, ``expanduser("~")`` or the HOME variable."""
    if isinstance(node, ast.Call):
        func = node.func
        name = getattr(func, "attr", getattr(func, "id", ""))
        if name == "home":
            return True
        if name == "expanduser" and node.args:
            arg = node.args[0]
            return isinstance(arg, ast.Constant) and str(arg.value).startswith(
                "~"
            )
        if name in ("get", "getenv") and node.args:
            arg = node.args[0]
            return isinstance(arg, ast.Constant) and arg.value in (
                "HOME",
                "XDG_CACHE_HOME",
            )
    if isinstance(node, ast.Subscript) and isinstance(
        node.slice, ast.Constant
    ):
        return node.slice.value in ("HOME", "XDG_CACHE_HOME")
    return False


def _docstrings(tree):
    """Return the ids of the docstring nodes in a syntax tree."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(
            node,
            ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
            ):
                found.add(id(body[0].value))
    return found


def _site_strings(tree):
    """Yield (line, text) for string constants naming the UBELIX site.

    Docstrings are skipped: they may mention a site as an example.
    """
    docstrings = _docstrings(tree)
    for node in ast.walk(tree):
        if id(node) in docstrings:
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.lower()
            if any(marker in lowered for marker in SITE_MARKERS):
                yield node.lineno, node.value


def test_no_temporary_data_is_written_to_the_home_directory():
    found = {}
    for rel, tree in source_files():
        lines = [n.lineno for n in ast.walk(tree) if _is_home_use(n)]
        if lines:
            found[rel] = lines
    unexpected = {
        f: lines for f, lines in found.items() if f not in HOME_ALLOWED
    }
    assert not unexpected, (
        "Use the job workspace (textlab.common.storage.get_workspace) for "
        f"temporary files instead of the home directory: {unexpected}"
    )
    stale = set(HOME_ALLOWED) - set(found)
    assert not stale, f"Remove stale HOME_ALLOWED entries: {sorted(stale)}"


def test_no_site_specific_values_in_the_code():
    found = {}
    for rel, tree in source_files():
        hits = [line for line, _ in _site_strings(tree)]
        if hits:
            found[rel] = hits
    unexpected = {
        f: hits for f, hits in found.items() if f not in SITE_ALLOWED
    }
    assert not unexpected, (
        "Move site-specific paths and hosts to deploy/site.env and read them "
        f"through textlab.common.config: {unexpected}"
    )
    stale = set(SITE_ALLOWED) - set(found)
    assert not stale, f"Remove stale SITE_ALLOWED entries: {sorted(stale)}"
