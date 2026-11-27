"""Historic handwriting on the OCR page, with the two workers faked.

Line detection finds two lines on every page and recognition reads each
line as its crop's name (and can be held at a gate, to test cancelling);
the rest of the service runs for real: pages, crops, the downloads with
PAGE XML, the saved page images, the batch ZIP.
"""

import io
import re
import threading
import zipfile
from types import SimpleNamespace

import pytest

from textlab.common.progress import Progress, check_cancelled
from textlab.features.handwriting import service
from textlab.features.handwriting.models import HandwritingModel
from textlab.features.handwriting.tests.test_service import png
from textlab.ui.web.jobs import JobState
from textlab.ui.web.tests.conftest import BASE, wait_for
from textlab.ui.web.tests.test_job_pages import htmx

MODELS = [
    HandwritingModel(
        "kurrent",
        "Kurrent, 19th century",
        "Letters and records in Kurrent.",
        "org/base",
        "org/kurrent-adapter",
        "Read.",
    ),
    HandwritingModel(
        "medieval", "Medieval", "Charters.", "org/medieval", None, "Read."
    ),
]


@pytest.fixture
def models(monkeypatch):
    """The configured models; a test may empty the list."""
    configured = list(MODELS)
    monkeypatch.setattr(service, "available_models", lambda: configured)

    def get_model(key):
        for model in configured:
            if model.key == key:
                return model
        raise service.HandwritingError(f"Unknown model {key}")

    monkeypatch.setattr(service, "get_model", get_model)
    return configured


@pytest.fixture
def workers(monkeypatch, workspace, models):
    """Fake the workers; record how they were asked to run."""
    calls = SimpleNamespace(detect=[], recognize=[], cancel=[], gate=None)

    def detect(images, *, on_progress, cancel=None):
        calls.detect.append(list(images))
        calls.cancel.append(cancel)
        on_progress(Progress("Finding lines...", 1.0))
        if calls.gate is not None:
            while not calls.gate.wait(0.01):
                check_cancelled(cancel)
        two_lines = [
            [[10, 60], [300, 60], [300, 90], [10, 90]],
            [[10, 10], [300, 10], [300, 40], [10, 40]],
        ]
        return [two_lines for _ in images]

    def recognize(crops, spec, *, on_progress, cancel=None):
        calls.recognize.append((list(crops), spec))
        return [f"read {crop.stem} <b>" for crop in crops]

    monkeypatch.setattr(service, "detect_lines", detect)
    monkeypatch.setattr(service, "recognize_lines", recognize)
    monkeypatch.setattr(service, "get_workspace", lambda: workspace)
    return calls


def submit(client, name, data, **fields):
    return client.post(
        f"{BASE}/ocr/handwriting",
        data=fields,
        files={"file": (name, data, "application/octet-stream")},
        headers=htmx(client),
    )


def finished(runner, response):
    assert response.status_code == 200, response.text[:500]
    job_id = re.search(r'id="job-([0-9a-f]+)"', response.text).group(1)
    job = runner.get(job_id)
    wait_for(lambda: not job.active)
    return job


def card(client, job):
    return client.get(f"{BASE}/jobs/{job.id}").text


def test_the_form_offers_the_models(client, models):
    html = client.get(f"{BASE}/ocr").text
    assert 'data-when="handwriting"' in html
    assert f'hx-post="{BASE}/ocr/handwriting"' in html
    assert re.search(r'name="model" value="kurrent"[^>]*checked', html)
    assert 'name="model" value="medieval"' in html
    assert "Letters and records in Kurrent." in html
    assert "org/kurrent-adapter (LoRA adapter on org/base)" in html
    assert service.LINE_DETECTOR in html
    assert 'id="handwriting-file"' in html


def test_without_models_the_form_says_so(client, models):
    models.clear()
    html = client.get(f"{BASE}/ocr").text
    assert "No handwriting models are configured." in html
    assert re.search(r'type="submit" disabled>Transcribe', html)


def test_a_document_is_transcribed_line_by_line(
    client, runner, workers, released
):
    job = finished(
        runner, submit(client, "letter.png", png(), model="kurrent")
    )
    assert job.state is JobState.DONE, job.error_details
    assert workers.recognize[0][1].key == "kurrent"
    assert workers.cancel[0] is job.cancel
    assert released == ["handwriting"]

    html = card(client, job)
    assert "Kurrent, 19th century" in html
    assert "2 lines" in html
    assert (
        '<rect class="box rt-text" x="10" y="60" width="290" height="30"'
        in (html)
    )
    assert re.search(
        r'<span class="ocr-line__number">1</span><span>read line-000000 '
        r"&lt;b&gt;</span>",
        html,
    )
    assert "Text recognition" in html and "org/kurrent-adapter" in html

    names = [item.name for item in job.listed_downloads]
    assert names == ["letter_handwriting.zip", "letter.txt", "letter.json"]
    archive = client.get(f"{BASE}/jobs/{job.id}/files/letter_handwriting.zip")
    listed = zipfile.ZipFile(io.BytesIO(archive.content)).namelist()
    assert "page_xml/page_0001.xml" in listed
    image = client.get(f"{BASE}/jobs/{job.id}/files/letter_page-0001.png")
    assert image.headers["content-type"] == "image/png"


def test_a_zip_is_transcribed_as_a_batch(client, runner, workers):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("box/a.png", png())
        archive.writestr("b.png", png())
    job = finished(
        runner,
        submit(client, "letters.zip", buffer.getvalue(), model="medieval"),
    )
    assert job.state is JobState.DONE, job.error_details
    assert len(workers.detect) == 1 and len(workers.detect[0]) == 2
    html = card(client, job)
    assert "2 files" in html and "4 lines" in html
    assert "box/a.png" in html
    assert "org/medieval" in html
    assert [item.name for item in job.listed_downloads] == [
        "letters_handwriting.zip"
    ]


def test_a_transcription_can_be_cancelled(client, runner, workers):
    workers.gate = threading.Event()
    response = submit(client, "a.png", png(), model="kurrent")
    job_id = re.search(r'id="job-([0-9a-f]+)"', response.text).group(1)
    job = runner.get(job_id)
    wait_for(lambda: len(workers.detect) == 1)
    client.post(f"{BASE}/jobs/{job.id}/cancel", headers=htmx(client))
    wait_for(lambda: not job.active)
    assert job.state is JobState.CANCELLED
    assert workers.recognize == []


@pytest.mark.parametrize(
    "name,fields,message",
    [
        ("a.png", {"model": "unknown"}, "Choose a handwriting model"),
        ("a.docx", {"model": "kurrent"}, "not a file this page reads"),
    ],
)
def test_bad_requests_are_explained(client, workers, name, fields, message):
    response = submit(client, name, b"x", **fields)
    assert response.status_code == 422
    assert message in response.text
