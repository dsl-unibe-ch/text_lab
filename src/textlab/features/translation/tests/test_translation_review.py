"""Side-by-side review files, actionable messages and document detection."""

import io

import pytest

from textlab.features.translation import lang_detect
from textlab.features.translation.chunking import (
    InputTooLongError,
    OutputTruncatedError,
)
from textlab.features.translation.lang_detect import DetectionResult
from textlab.features.translation.messages import describe_error
from textlab.features.translation.review import (
    build_review_docx,
    build_review_html,
    review_rows,
)
from textlab.features.translation.shield import (
    ProtectedContentError,
    record_translations,
    shielded_translate_many,
)


def test_recorder_collects_pairs_only_inside_its_scope():
    shielded_translate_many(["outside"], str.upper)
    with record_translations() as pairs:
        shielded_translate_many(["one", "", "two `x`"], str.upper)
        with record_translations() as inner:
            shielded_translate_many(["three"], str.upper)
    assert pairs == [("one", "ONE"), ("two `x`", "TWO `x`")]
    assert inner == [("three", "THREE")]


def test_review_rows_are_readable_and_deduplicated():
    rows = review_rows(
        [
            ("Header", "Kopf"),
            (
                "<table><tr><td>Cell</td></tr></table>",
                "<table><tr><td>Zelle</td></tr></table>",
            ),
            ("Header", "Kopf"),
            ("12.5", "12.5"),
            ("~~~text\nx = 1\n~~~", "~~~text\nx = 1\n~~~"),
        ]
    )
    assert rows == [("Header", "Kopf"), ("Cell", "Zelle")]


def test_review_html_escapes_text_and_names_languages():
    page = build_review_html(
        [("<b>bold</b> & more", "fett & mehr")],
        title="a<b>.pdf",
        source_language="English",
        target_language="German",
    ).decode()
    assert '<td dir="auto">bold &amp; more</td>' in page
    assert "a&lt;b&gt;.pdf" in page
    assert "English" in page and "German" in page


def test_review_docx_has_a_two_column_table():
    docx = pytest.importorskip("docx")
    data = build_review_docx(
        [("Hello", "Hallo")],
        title="t",
        source_language="English",
        target_language="German",
    )
    table = docx.Document(io.BytesIO(data)).tables[0]
    assert [cell.text for cell in table.rows[1].cells] == ["Hello", "Hallo"]


@pytest.mark.parametrize(
    "error,backend,expected",
    [
        (OutputTruncatedError("x"), "nllb", "NLLB-200 3.3B"),
        (OutputTruncatedError("x"), "nllb-large", "LLM (Ollama)"),
        (InputTooLongError("x"), "nllb", "LLM (Ollama)"),
        (ProtectedContentError("x"), "madlad-3b", "LLM (Ollama)"),
        (ProtectedContentError("x"), "ollama", "larger LLM"),
    ],
)
def test_errors_name_a_concrete_backend_to_try(error, backend, expected):
    message = describe_error(error, backend)
    assert expected in message
    assert "different model" not in message


def test_unknown_errors_stay_generic_and_own_messages_pass_through():
    assert "unexpected" in describe_error(KeyError("internal detail"))
    assert describe_error(ValueError("OPUS-MT has no pair.")) == (
        "OPUS-MT has no pair."
    )


def _result(code, confidence):
    return DetectionResult(code[:2], confidence, code, code.upper())


def test_document_language_is_voted_across_paragraphs(monkeypatch):
    english = "This abstract is written in English for the journal. " * 3
    german = "Dieser Abschnitt ist auf Deutsch geschrieben und lang. " * 3
    text = "\n\n".join([english] + [german] * 4)
    monkeypatch.setattr(
        lang_detect,
        "detect_language",
        lambda paragraph: _result(
            "eng_Latn" if "English" in paragraph else "deu_Latn",
            0.95,
        ),
    )
    result = lang_detect.detect_document_language("a.md", text.encode())
    assert result.flores_code == "deu_Latn"


def test_uncertain_or_textless_documents_are_not_guessed(monkeypatch):
    monkeypatch.setattr(
        lang_detect,
        "detect_language",
        lambda paragraph: _result(
            "deu_Latn",
            0.3,
        ),
    )
    long_text = (
        "Ein ziemlich langer Absatz mit genug Buchstaben darin. " * 3
    ).encode()
    assert lang_detect.detect_document_language("a.md", long_text) is None
    assert lang_detect.detect_document_language("a.md", b"12 34") is None
    assert lang_detect.detect_document_language("a.xlsx", b"...") is None
