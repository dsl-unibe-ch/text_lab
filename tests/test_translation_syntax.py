"""Syntax checks that also catch PEP 701 f-strings on newer dev hosts."""

import ast
import io
from pathlib import Path
import token
import tokenize

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted((ROOT / "src" / "core" / "translation").glob("*.py"))
SOURCES.append(ROOT / "src" / "pages" / "Translate.py")


def _check_syntax(source, filename):
    # feature_version alone can miss 3.12-only f-string line breaks when
    # validation runs on a newer interpreter than the deployed Python 3.11.
    stack = []
    for item in tokenize.generate_tokens(io.StringIO(source).readline):
        kind = token.tok_name[item.type]
        if kind == "FSTRING_START":
            stack.append((
                item.start[0], item.string.endswith(('"""', "'''")),
            ))
        elif kind == "FSTRING_END":
            start, triple_quoted = stack.pop()
            if not triple_quoted and item.end[0] != start:
                raise SyntaxError(
                    f"{filename}:{start}: multiline single-quoted f-string "
                    "is incompatible with Python 3.11"
                )
    ast.parse(source, filename=filename, feature_version=(3, 11))
    compile(source, filename, "exec")


@pytest.mark.parametrize("path", SOURCES, ids=lambda path: path.name)
def test_translation_source_syntax(path):
    _check_syntax(path.read_text(encoding="utf-8"), str(path))


def test_multiline_single_quoted_fstring_is_rejected():
    with pytest.raises((SyntaxError, tokenize.TokenError)):
        _check_syntax('message = f"GPU ({\n name})"\n', "regression.py")


def test_adjacent_and_triple_quoted_fstrings_are_allowed():
    _check_syntax(
        'message = (f"GPU ({name}, "\n f"{memory} GB)")\n'
        'html = f"""\n<p>{message}</p>\n"""\n', "valid.py",
    )
