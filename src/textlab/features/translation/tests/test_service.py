"""The translation service, with the backends replaced by plain functions."""

import io
import json
import zipfile
from types import SimpleNamespace

import pytest

from textlab.features.translation import service
from textlab.features.translation.lang_detect import DetectionResult

OPTIONS = service.TranslationOptions(
    backend="nllb",
    source_code="deu_Latn",
    source_name="German",
    target_code="eng_Latn",
    target_name="English",
    glossary={"Bern": "Berne"},
    glossary_case_sensitive=True,
)


def upper(text):
    return text.upper()


@pytest.fixture
def translators(monkeypatch):
    """Replace model-backed translators by upper-casing ones."""
    made = []

    def make_translate_fn(**kwargs):
        made.append(kwargs)

        def translate(text):
            if kwargs["progress_cb"] is not None:
                kwargs["progress_cb"](1, 2)
                kwargs["progress_cb"](2, 2)
            return text.upper()

        return translate

    monkeypatch.setattr(service, "make_translate_fn", make_translate_fn)
    return made


# --- One document ---------------------------------------------------------


@pytest.mark.parametrize(
    "extension,function",
    [
        (".md", "translate_markdown"),
        (".docx", "translate_docx"),
        (".xlsx", "translate_xlsx"),
        (".pptx", "translate_pptx"),
    ],
)
def test_document_formats_get_the_glossary_and_keep_their_type(
    monkeypatch, extension, function
):
    calls = []

    def translate(data, translator, **options):
        calls.append(options)
        return "result" if extension == ".md" else b"result"

    monkeypatch.setattr(service, function, translate)
    stages = []
    output = service.translate_document(
        "folder/source" + extension,
        b"input",
        upper,
        "en",
        glossary={"Bern": "Berne"},
        glossary_case_sensitive=True,
        progress_cb=lambda *args: stages.append(args[2]),
    )
    assert output == [("source.en" + extension, b"result")]
    assert calls[0]["glossary"] == {"Bern": "Berne"}
    assert calls[0]["glossary_case_sensitive"] is True
    assert stages[0].startswith("parsing")
    assert stages[-1].startswith("reconstructing")


def test_plain_text_is_reflowed_but_subtitles_keep_their_lines():
    wrapped = b"Die Kommission hat\nbeschlossen."
    text = service.translate_document("a.txt", wrapped, upper, "en")
    subtitles = service.translate_document("a.srt", wrapped, upper, "en")
    assert text == [("a.en.txt", b"DIE KOMMISSION HAT BESCHLOSSEN.")]
    assert subtitles == [("a.en.srt", b"DIE KOMMISSION HAT\nBESCHLOSSEN.")]


def test_unsupported_type_is_refused():
    with pytest.raises(ValueError, match="Unsupported file type: .rtf"):
        service.translate_document("a.rtf", b"", upper, "en")


def test_pdf_keeps_a_valid_sibling_and_adds_the_report(monkeypatch):
    result = SimpleNamespace(
        outputs=[("source.en.md", b"complete translation")],
        blocked=[{"output": "PDF", "reason": "Overflow", "pages": [1]}],
        report_bytes=lambda: b'{"status":"partial"}',
    )
    monkeypatch.setattr(
        service, "translate_pdf_outputs", lambda *args, **kwargs: result
    )
    reports = []
    outputs = service.translate_document(
        "source.pdf", b"input", upper, "en", pdf_result_cb=reports.append
    )
    assert outputs == result.outputs + [
        ("source.en.translation-report.json", b'{"status":"partial"}'),
    ]
    assert reports == [result]


def test_pdf_without_any_valid_output_fails_with_the_reasons(monkeypatch):
    result = SimpleNamespace(
        outputs=[],
        blocked=[
            {"output": "Markdown", "reason": "OCR failed", "pages": [2]},
            {"output": "PDF", "reason": "Scanned page", "pages": [2]},
        ],
    )
    monkeypatch.setattr(
        service, "translate_pdf_outputs", lambda *args, **kwargs: result
    )
    reports = []
    with pytest.raises(ValueError, match="OCR failed.*Scanned page"):
        service.translate_document(
            "source.pdf", b"input", upper, "en", pdf_result_cb=reports.append
        )
    assert reports == [result]


# --- Batches --------------------------------------------------------------


def zip_bytes(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def test_uploads_are_unpacked_and_unsupported_members_reported():
    archive = zip_bytes(
        {"docs/": b"", "docs/a.md": b"# A", "b.exe": b"x", "c.txt": b"c"}
    )
    unpacked = service.unpack_uploads(
        [("one.docx", b"d"), ("batch.zip", archive), ("bad.zip", b"no")]
    )
    assert unpacked.files == [
        ("one.docx", b"d"),
        ("docs/a.md", b"# A"),
        ("c.txt", b"c"),
    ]
    assert unpacked.skipped == ["b.exe"]
    assert unpacked.invalid == ["bad.zip"]


def test_batch_translates_each_file_and_reports_failures(
    translators, monkeypatch
):
    def translate_document(name, data, translate_fn, target_code, **kwargs):
        if name == "broken.md":
            raise RuntimeError("parser crashed")
        kwargs["progress_cb"](1, 1, "done")
        return [
            (f"{name}.{target_code}", translate_fn(data.decode()).encode())
        ]

    monkeypatch.setattr(service, "translate_document", translate_document)
    updates = []
    results = service.translate_documents(
        [("a.txt", b"hallo"), ("broken.md", b"x")],
        OPTIONS,
        service.DocumentOptions(detect_source=False, review=False),
        on_progress=updates.append,
    )
    assert results.outputs == [("a.txt.eng_Latn", b"HALLO")]
    assert [name for name, _ in results.errors] == ["broken.md"]
    assert "parser crashed" in results.errors[0][1]
    assert results.total == 2
    assert results.target_code == "eng_Latn"
    messages = [update.message for update in updates]
    assert messages[0] == "[1/2] a.txt"
    assert "[2/2] broken.md" in messages
    assert (messages[-1], updates[-1].fraction) == ("done", 1.0)
    assert len(translators) == 1  # One translator per source language.
    assert translators[0]["src_lang"] == "deu_Latn"
    assert translators[0]["tgt_lang"] == "eng_Latn"


def test_batch_uses_detected_languages_and_skips_target_language(
    translators, monkeypatch
):
    detected = {
        "fr.txt": DetectionResult("fr", 0.99, "fra_Latn", "French"),
        "en.txt": DetectionResult("en", 0.99, "eng_Latn", "English"),
        "unknown.txt": None,
    }
    monkeypatch.setattr(
        service,
        "detect_document_language",
        lambda name, data: detected[name],
    )
    results = service.translate_documents(
        [(name, b"text") for name in detected],
        OPTIONS,
        service.DocumentOptions(review=False),
    )
    assert [name for name, _ in results.file_outputs] == [
        "fr.txt",
        "unknown.txt",
    ]
    assert results.errors == [
        ("en.txt", "This document already appears to be in English."),
    ]
    assert results.languages == {
        "fr.txt": "French (detected)",
        "en.txt": "English (detected)",
        "unknown.txt": "German (not detected, using your selection)",
    }
    assert [made["src_lang"] for made in translators] == [
        "fra_Latn",
        "deu_Latn",
    ]


def test_batch_adds_review_files(translators):
    results = service.translate_documents(
        [("notes.txt", b"Guten Tag.")],
        OPTIONS,
        service.DocumentOptions(detect_source=False),
    )
    names = [name for name, _ in results.outputs]
    assert names[:2] == [
        "notes.eng_Latn.txt",
        "notes.eng_Latn.side-by-side.html",
    ]
    html = dict(results.outputs)["notes.eng_Latn.side-by-side.html"]
    assert b"Guten Tag." in html and b"GUTEN TAG." in html


def test_outputs_zip_nests_several_sources_and_unpacks_bundles():
    bundle = zip_bytes({"a.en.md": b"# A", "assets/fig.png": b"png"})
    archive = zipfile.ZipFile(
        io.BytesIO(
            service.outputs_zip(
                [("a.pdf", [("a.en.md.zip", bundle), ("a.en.pdf", b"pdf")])],
                [("b.docx", "Unsupported font")],
            )
        )
    )
    assert sorted(archive.namelist()) == [
        "a/a.en.md",
        "a/a.en.pdf",
        "a/assets/fig.png",
        "b.docx.ERROR.txt",
    ]
    assert archive.read("b.docx.ERROR.txt") == (
        b"Failed to translate: Unsupported font"
    )


def test_outputs_zip_puts_a_single_source_at_the_root():
    archive = zipfile.ZipFile(
        io.BytesIO(service.outputs_zip([("a.md", [("a.en.md", b"x")])], []))
    )
    assert archive.namelist() == ["a.en.md"]


def test_review_files_need_pairs():
    assert service.review_files("a.pdf", [], "German", "English", "en") == []


@pytest.mark.parametrize(
    "name,expected",
    [
        ("a.PDF", "application/pdf"),
        ("a.md.zip", "application/zip"),
        ("a.srt", "text/plain"),
        ("a.side-by-side.html", "text/html"),
    ],
)
def test_mime_type(name, expected):
    assert service.mime_type(name) == expected


# --- Text -----------------------------------------------------------------


def test_text_is_reflowed_and_progress_reported(translators):
    updates = []
    translated = service.translate_text(
        "Die Kommission hat\nbeschlossen.",
        OPTIONS,
        on_progress=updates.append,
    )
    assert translated == "DIE KOMMISSION HAT BESCHLOSSEN."
    assert [(u.message, u.fraction) for u in updates] == [
        ("Translating chunk 1/2", 0.5),
        ("Translating chunk 2/2", 1.0),
    ]


def test_glossary_terms_are_forced(translators):
    assert service.translate_text("Bern ist schön.", OPTIONS).startswith(
        "Berne"
    )


# --- Backends and progress ------------------------------------------------


def test_only_opus_mt_is_limited_to_mapped_pairs():
    assert service.supports_pair("opus-mt", "deu_Latn", "eng_Latn")
    assert not service.supports_pair("opus-mt", "deu_Latn", "gsw_Latn")
    assert service.supports_pair("nllb", "deu_Latn", "gsw_Latn")


def test_load_signature_follows_the_model_not_the_glossary():
    other = service.TranslationOptions(
        backend="nllb",
        source_code="fra_Latn",
        source_name="French",
        target_code="ita_Latn",
        target_name="Italian",
    )
    assert service.load_signature(OPTIONS) == service.load_signature(other)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def test_progress_limits_updates_of_one_stage():
    updates = []
    clock = Clock()
    progress = service._DocumentProgress(updates.append, clock)
    progress.step(1, 10, "parsing markdown")
    progress.step(2, 10, "parsing markdown")  # Too soon: dropped.
    progress.step(10, 10, "parsing markdown")  # Final: always shown.
    progress.step(0, 3, "translating markdown")  # New stage: shown.
    clock.now += 1
    progress.step(1, 3, "translating markdown")
    assert [(u.message, u.fraction) for u in updates] == [
        ("parsing markdown", 0.1),
        ("parsing markdown", 1.0),
        ("translating markdown", 0.0),
        ("translating markdown", pytest.approx(1 / 3)),
    ]


def test_progress_estimates_the_time_left():
    updates = []
    clock = Clock()
    progress = service._DocumentProgress(updates.append, clock)
    progress.sentences(0, 100)
    clock.now += 60
    progress.sentences(10, 100)
    clock.now += 1
    progress.sentences(90, 100)
    assert [u.message for u in updates] == [
        "translating sentences 0/100",
        "translating sentences 10/100 · about 9 min left",
        "translating sentences 90/100 · less than a minute left",
    ]


def test_retries_are_reported_as_a_stage():
    updates = []
    progress = service._DocumentProgress(updates.append)
    progress.retrying("Ollama reached its output-token limit. Retrying.")
    assert updates[0].message == "retrying some text in smaller pieces"
    assert updates[0].fraction is None


def test_pdf_report_is_valid_json():
    report = service.PDFTranslationResult(outputs=[("a.md", b"x")])
    assert json.loads(report.report_bytes())["status"] == "passed_checks"
