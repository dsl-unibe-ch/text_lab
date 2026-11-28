"""The Translation page: the editor, and documents as jobs.

The model is faked: translators upper-case their text (and can be held at
a gate, to test waiting and cancelling); loading a backend is recorded;
language detection returns what the test sets. The rest of the service
runs for real (shielding, document formats, review files), so the tests
check the page with the service it calls.
"""

import io
import re
import threading
import zipfile
from types import SimpleNamespace

import pytest

from textlab.features.translation import service
from textlab.ui.web.jobs import JobState
from textlab.ui.web.pages import translation as page
from textlab.ui.web.tests.conftest import BASE, wait_for
from textlab.ui.web.tests.test_job_pages import htmx
from textlab.ui.web.tests.test_jobs import Gate

GERMAN = service.DetectionResult("de", 0.98, "deu_Latn", "German")

SETTINGS = {
    "source": "deu_Latn",
    "target": "eng_Latn",
    "backend": "nllb",
    "pdf_markdown": "1",
    "pdf_pdf": "1",
    "review": "1",
}


@pytest.fixture
def engine(monkeypatch):
    """Fake the models; record what was loaded and translated."""
    calls = SimpleNamespace(
        loaded=set(),
        loads=[],
        made=[],
        texts=[],
        gate=None,
        detected=GERMAN,
        documents={},
    )

    def make_translate_fn(**kwargs):
        calls.made.append(kwargs)

        def translate(text):
            calls.texts.append(text)
            progress = kwargs["progress_cb"]
            if calls.gate is not None:
                while not calls.gate.wait(0.01):
                    if progress is not None:
                        progress(0, 1)
            if progress is not None:
                progress(1, 1)
            return text.upper()

        return translate

    def load_backend(options):
        calls.loads.append(options.backend)
        calls.loaded.add(options.backend)

    monkeypatch.setattr(service, "make_translate_fn", make_translate_fn)
    monkeypatch.setattr(service, "load_backend", load_backend)
    monkeypatch.setattr(
        service,
        "backend_ready",
        lambda options: options.backend in calls.loaded,
    )
    monkeypatch.setattr(
        service, "detect_language", lambda text: calls.detected
    )
    monkeypatch.setattr(
        service,
        "detect_document_language",
        lambda name, data: calls.documents.get(name, calls.detected),
    )
    monkeypatch.setattr(service, "ollama_models", lambda: ["small:8b"])
    monkeypatch.setattr(service, "server_ready", lambda attempts=1: True)
    monkeypatch.setattr(page, "SETTLE_SECONDS", 0.2)
    return calls


def translate(client, text, **fields):
    return client.post(
        f"{BASE}/translation/text",
        data={**SETTINGS, "text": text, **fields},
        headers=htmx(client),
    )


def card_id(html):
    return re.search(r'id="job-([0-9a-f]+)"', html).group(1)


def editor_job(runner, response):
    assert response.status_code == 200, response.text[:500]
    job = runner.get(card_id(response.text))
    wait_for(lambda: not job.active)
    return job


def card(client, job):
    return client.get(f"{BASE}/jobs/{job.id}").text


def upload(client, files, **fields):
    return client.post(
        f"{BASE}/translation",
        data={**SETTINGS, **fields},
        files=[("files", item) for item in files] or None,
        headers=htmx(client),
    )


def documents_job(runner, response):
    assert response.status_code == 200, response.text[:500]
    job = runner.get(card_id(response.text))
    wait_for(lambda: not job.active)
    return job


# --- The page -------------------------------------------------------------


def test_the_page_shows_the_editor_the_languages_and_the_engines(
    client, engine
):
    html = client.get(f"{BASE}/translation").text
    assert 'name="mode" value="text" checked' in html
    assert '<option value="auto" selected>Detect language</option>' in html
    assert '<option value="eng_Latn" selected>English</option>' in html
    for backend in service.BACKENDS.values():
        assert backend.name in html
    assert "The translation appears here." in html
    assert "data-text-submit" in html
    assert "<option selected>small:8b</option>" in html
    assert "Its server does not answer yet." not in html


def test_the_language_model_engine_is_off_while_its_server_is_down(
    client, engine, monkeypatch
):
    monkeypatch.setattr(service, "server_ready", lambda attempts=1: False)
    html = client.get(f"{BASE}/translation").text
    assert "Its server does not answer yet." in html
    response = translate(client, "Hallo", backend="ollama", llm="small:8b")
    assert response.status_code == 422
    assert "does not answer" in response.text


def test_translation_is_ready_in_the_navigation(client):
    html = client.get(f"{BASE}/").text
    assert re.search(r"Translation</span>\s*</a>", html)


# --- The editor -----------------------------------------------------------


def test_text_is_translated_into_the_output_pane(
    client, runner, engine, released
):
    response = translate(client, "Guten Tag.")
    job = editor_job(runner, response)
    assert job.state is JobState.DONE
    html = card(client, job)
    assert "data-output-text" in html
    assert "GUTEN TAG." in html
    assert "German to English" in html
    assert "NLLB-200, 600M" in html
    assert engine.loads == ["nllb"]
    assert released == ["translation"]


def test_short_texts_come_back_with_the_answer(client, engine):
    response = translate(client, "Guten Tag.")
    assert "GUTEN TAG." in response.text
    assert "data-job-active" not in response.text


def test_a_loaded_engine_is_not_loaded_again(client, runner, engine):
    editor_job(runner, translate(client, "Eins."))
    editor_job(runner, translate(client, "Zwei."))
    assert engine.loads == ["nllb"]


def test_the_language_is_detected_when_asked(client, runner, engine):
    job = editor_job(runner, translate(client, "Guten Tag.", source="auto"))
    html = card(client, job)
    assert "German, detected to English" in html
    assert 'data-detected="deu_Latn"' in html
    assert 'data-detected-name="German"' in html
    assert engine.made[-1]["src_lang"] == "deu_Latn"


@pytest.mark.parametrize(
    "detected,message",
    [
        (None, "could not be told"),
        (
            service.DetectionResult("de", 0.4, "deu_Latn", "German"),
            "not certain",
        ),
        (
            service.DetectionResult("en", 0.99, "eng_Latn", "English"),
            "in English already",
        ),
    ],
)
def test_a_language_that_cannot_be_used_is_explained(
    client, runner, engine, detected, message
):
    engine.detected = detected
    job = editor_job(runner, translate(client, "Text.", source="auto"))
    assert job.state is JobState.FAILED
    assert message in job.error
    assert message in card(client, job)


def test_editor_translations_are_not_listed_with_jobs(client, runner, engine):
    editor_job(runner, translate(client, "Guten Tag."))
    assert runner.jobs(listed=True) == []
    home = client.get(f"{BASE}/").text
    assert "GUTEN TAG." not in home
    assert "No jobs yet." in home


def test_a_new_translation_replaces_the_one_before(client, runner, engine):
    first = editor_job(runner, translate(client, "Erster Text."))
    engine.gate = threading.Event()
    response = translate(client, "Zweiter Text.")
    assert response.status_code == 200
    html = response.text
    assert "data-job-active" in html
    assert "output__text--stale" in html
    assert "ERSTER TEXT." in html
    assert runner.get(first.id) is None
    engine.gate.set()
    job = editor_job(runner, response)
    assert "ZWEITER TEXT." in card(client, job)
    assert len(runner.jobs("translation", listed=False)) == 1


def test_a_translation_says_which_job_it_waits_for(client, runner, engine):
    work = Gate()
    other = runner.start(
        work, title="lecture.mp3", page="transcription", gpu="transcribe"
    )
    wait_for(work.started.is_set)
    response = translate(client, "Guten Tag.")
    assert 'Waits for "lecture.mp3" to finish' in response.text
    assert f"/jobs/{other.id}/cancel" in response.text
    work.opened.set()
    job = editor_job(runner, response)
    assert job.state is JobState.DONE


def test_stop_cancels_a_running_translation(client, runner, engine):
    engine.gate = threading.Event()
    response = translate(client, "Guten Tag.")
    job = runner.get(card_id(response.text))
    wait_for(lambda: job.state is JobState.RUNNING)
    stopped = client.post(
        f"{BASE}/jobs/{job.id}/cancel", headers=htmx(client)
    ).text
    assert 'class="output output--' in stopped
    wait_for(lambda: not job.active)
    assert job.state is JobState.CANCELLED
    assert "Stopped." in card(client, job)


def test_the_page_shows_the_last_text_and_settings_again(
    client, runner, engine
):
    editor_job(
        runner,
        translate(
            client,
            "Bern ist schön.",
            target="fra_Latn",
            term=["Bern", "", "Uni"],
            translation=["Berne", "x", ""],
            glossary_case="1",
        ),
    )
    html = client.get(f"{BASE}/translation").text
    assert ">Bern ist schön.</textarea>" in html
    assert '<option value="fra_Latn" selected>French</option>' in html
    assert 'name="term" value="Bern"' in html
    assert 'name="translation" value="Berne"' in html
    assert 'name="glossary_case" value="1" checked' in html
    assert "1 term<" in html


def test_glossary_terms_reach_the_translation(client, runner, engine):
    job = editor_job(
        runner,
        translate(
            client, "Bern ist schön.", term=["Bern"], translation=["Berne"]
        ),
    )
    assert job.result.text.startswith("Berne")
    assert job.result.glossary_terms == 1


def test_the_language_model_keeps_its_model_loaded(
    client, runner, engine, released
):
    job = editor_job(
        runner,
        translate(
            client,
            "Grüezi.",
            backend="ollama",
            llm="small:8b",
            formality="formal",
        ),
    )
    assert job.state is JobState.DONE
    assert released == ["translation:small:8b"]
    assert engine.made[-1]["formality"] == "formal"
    assert engine.made[-1]["ollama_model"] == "small:8b"


@pytest.mark.parametrize(
    "fields,message",
    [
        ({"text": "   "}, "Type or paste"),
        ({"text": "x" * (page.TEXT_LIMIT + 1)}, "as a document"),
        ({"target": "deu_Latn"}, "same language"),
        ({"target": "xxx_Latn"}, "translate into"),
        ({"source": "xxx_Latn"}, "translate from"),
        ({"backend": "deepl"}, "engine"),
        ({"backend": "ollama", "llm": "huge:405b"}, "language model"),
        (
            {"backend": "opus-mt", "source": "hye_Armn", "target": "kat_Geor"},
            "OPUS-MT has no model",
        ),
    ],
)
def test_bad_requests_are_explained(client, engine, fields, message):
    data = {"text": "Guten Tag.", **fields}
    response = translate(client, **data)
    assert response.status_code == 422
    assert message in response.text


def test_glossary_rows_without_both_halves_are_left_out():
    from starlette.datastructures import FormData

    form = FormData(
        [
            ("term", " Bern "),
            ("translation", " Berne "),
            ("term", "Uni"),
            ("translation", ""),
            ("term", "Bern"),
            ("translation", "Bärn"),
        ]
    )
    assert page.read_glossary(form) == (("Bern", "Bärn"),)


def test_too_many_glossary_terms_are_refused():
    from starlette.datastructures import FormData

    rows = []
    for number in range(page.GLOSSARY_LIMIT + 1):
        rows += [("term", f"t{number}"), ("translation", f"u{number}")]
    with pytest.raises(page.uploads.UploadError, match="at most"):
        page.read_glossary(FormData(rows))


# --- Documents ------------------------------------------------------------


def test_documents_are_translated_with_their_files(client, runner, engine):
    job = documents_job(
        runner,
        upload(
            client,
            [
                ("brief.txt", b"Guten Tag.", "text/plain"),
                ("notes.md", b"# Titel\n\nEin Satz.", "text/markdown"),
            ],
        ),
    )
    assert job.state is JobState.DONE, job.error_details
    html = card(client, job)
    assert "2 of 2 documents translated" in html
    assert "German to English" in html
    assert "brief.eng_Latn.txt" in html
    assert "notes.eng_Latn.md" in html
    assert "Side by side, HTML" in html
    assert "Compare source and translation" in html
    assert "GUTEN TAG." in html
    # The record lists the ZIP; each document lists its own files.
    assert [item.name for item in job.listed_downloads] == [
        "translated_eng_Latn.zip"
    ]
    archive = client.get(
        f"{BASE}/jobs/{job.id}/files/translated_eng_Latn.zip"
    ).content
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        assert "brief/brief.eng_Latn.txt" in zipped.namelist()
    text = client.get(f"{BASE}/jobs/{job.id}/files/brief.eng_Latn.txt")
    assert text.content == b"GUTEN TAG."
    assert runner.jobs(listed=True) == [job]


def test_one_output_is_offered_directly(client, runner, engine):
    job = documents_job(
        runner,
        upload(
            client, [("brief.txt", b"Guten Tag.", "text/plain")], review=""
        ),
    )
    assert [item.name for item in job.listed_downloads] == [
        "brief.eng_Latn.txt"
    ]


def test_each_documents_language_is_detected(client, runner, engine):
    engine.documents = {
        "fr.txt": service.DetectionResult("fr", 0.99, "fra_Latn", "French")
    }
    job = documents_job(
        runner,
        upload(
            client,
            [
                ("fr.txt", b"Bonjour.", "text/plain"),
                ("de.txt", b"Hallo.", "text/plain"),
            ],
            source="auto",
        ),
    )
    html = card(client, job)
    assert "Detected to English" in html
    assert "French (detected)" in html
    assert "German (detected)" in html
    assert {made["src_lang"] for made in engine.made} == {
        "fra_Latn",
        "deu_Latn",
    }


def test_zip_members_of_other_types_are_noted(client, runner, engine):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("a.txt", "Guten Tag.")
        archive.writestr("b.exe", "x")
    job = documents_job(
        runner,
        upload(client, [("batch.zip", buffer.getvalue(), "application/zip")]),
    )
    html = card(client, job)
    assert "b.exe was left out" in html
    assert "a.eng_Latn.txt" in html


def test_a_job_with_no_translated_document_fails(client, runner, engine):
    engine.detected = None
    job = documents_job(
        runner,
        upload(client, [("a.txt", b"???", "text/plain")], source="auto"),
    )
    assert job.state is JobState.FAILED
    assert "No document could be translated" in job.error
    assert "could not be told" in job.error


def test_documents_can_be_cancelled(client, runner, engine):
    engine.gate = threading.Event()
    response = upload(client, [("a.txt", b"Guten Tag.", "text/plain")])
    job = runner.get(card_id(response.text))
    wait_for(lambda: job.state is JobState.RUNNING)
    client.post(f"{BASE}/jobs/{job.id}/cancel", headers=htmx(client))
    wait_for(lambda: not job.active)
    assert job.state is JobState.CANCELLED
    assert job.downloads == []


@pytest.mark.parametrize(
    "files,fields,message",
    [
        ([], {}, "Choose the documents"),
        ([("a.rtf", b"x", "text/rtf")], {}, "not a file this page reads"),
        (
            [("a.pdf", b"%PDF", "application/pdf")],
            {"pdf_markdown": "", "pdf_pdf": ""},
            "at least one output",
        ),
    ],
)
def test_bad_uploads_are_explained(client, engine, files, fields, message):
    response = upload(client, files, **fields)
    assert response.status_code == 422
    assert message in response.text


def test_output_files_are_named_by_kind():
    assert (
        page.file_kind("a.deu_Latn.side-by-side.docx") == "Side by side, Word"
    )
    assert page.file_kind("a.deu_Latn.md.zip") == "Markdown with images"
    assert page.file_kind("a.deu_Latn.pdf") == "PDF"
    assert page.file_kind("a.bin") == "File"
