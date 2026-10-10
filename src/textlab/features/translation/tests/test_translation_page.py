"""Exercise document dispatch without importing or launching Streamlit."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


PAGE = (
    Path(__file__).resolve().parents[3]
    / "ui" / "streamlit" / "pages" / "Translate.py"
)


def test_page_translation_imports_are_exported():
    from textlab.features import translation

    module = ast.parse(PAGE.read_text(encoding="utf-8"))
    for node in module.body:
        if (isinstance(node, ast.ImportFrom)
                and node.module == "textlab.features.translation"):
            for alias in node.names:
                assert hasattr(translation, alias.name), alias.name


def dispatch_namespace():
    module = ast.parse(PAGE.read_text(encoding="utf-8"))
    function = next(node for node in module.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "_translate_one")
    namespace = {
        "os": os, "glossary": {"Bern": "Berne"},
        "glossary_case_sensitive": True,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]),
                 str(PAGE), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("extension,function", [
    (".md", "translate_markdown"), (".docx", "translate_docx"),
    (".xlsx", "translate_xlsx"), (".pptx", "translate_pptx"),
])
def test_page_forwards_glossary_case_to_every_document_format(
    extension, function,
):
    namespace = dispatch_namespace()
    calls = []

    def translate(data, translator, **options):
        calls.append(options)
        return "result" if extension == ".md" else b"result"

    namespace[function] = translate
    output = namespace["_translate_one"](
        "source" + extension, b"input", None, lambda *args: None, "en",
    )
    assert output == [("source.en" + extension, b"result")]
    assert calls[0]["glossary_case_sensitive"] is True


def test_page_keeps_valid_pdf_sibling_and_includes_report():
    namespace = dispatch_namespace()
    result = SimpleNamespace(
        outputs=[("source.en.md", b"complete translation")],
        blocked=[{"output": "PDF", "reason": "Overflow", "pages": [1]}],
        report_bytes=lambda: b'{"status":"partial"}',
    )
    namespace["translate_pdf_outputs"] = lambda *args, **kwargs: result
    reports = []
    outputs = namespace["_translate_one"](
        "source.pdf", b"input", None, lambda *args: None, "en",
        pdf_result_cb=reports.append,
    )
    assert outputs == result.outputs + [
        ("source.en.translation-report.json", b'{"status":"partial"}'),
    ]
    assert reports == [result]


def test_page_reports_all_blocked_outputs_without_a_fake_success():
    namespace = dispatch_namespace()
    result = SimpleNamespace(
        outputs=[], blocked=[
            {"output": "Markdown", "reason": "OCR failed", "pages": [2]},
            {"output": "PDF", "reason": "Scanned page", "pages": [2]},
        ],
    )
    namespace["translate_pdf_outputs"] = lambda *args, **kwargs: result
    reports = []
    with pytest.raises(ValueError, match="OCR failed.*Scanned page"):
        namespace["_translate_one"](
            "source.pdf", b"input", None, lambda *args: None, "en",
            pdf_result_cb=reports.append,
        )
    assert reports == [result]
