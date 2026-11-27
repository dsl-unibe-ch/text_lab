"""The OCR page, with the automatic pipeline faked.

``process_document`` returns a page with a title, a table, a figure and a
check box (and can be held at a gate, to test cancelling); the rest of the
service runs for real: the downloads, the saved page images and figures,
the batch ZIP.
"""

import io
import re
import threading
import zipfile
from types import SimpleNamespace

import pytest

from textlab.common.progress import check_cancelled
from textlab.features.ocr import service
from textlab.features.ocr.tests.test_service import rich_document
from textlab.ui.web.jobs import JobState
from textlab.ui.web.pages import ocr as page
from textlab.ui.web.tests.conftest import BASE, wait_for
from textlab.ui.web.tests.test_job_pages import htmx


@pytest.fixture
def pipeline(monkeypatch, workspace):
    """Fake the recognition; record how it was asked to run."""
    calls = SimpleNamespace(runs=[], gate=None, fail=None)

    def process_document(input_path, workspace_dir, **kwargs):
        calls.runs.append({"name": input_path.name, **kwargs})
        kwargs["progress"](0.5, "Reading page 1")
        if calls.gate is not None:
            while not calls.gate.wait(0.01):
                check_cancelled(kwargs["cancel"])
        if calls.fail is not None:
            raise calls.fail
        document = rich_document()
        document.source_name = input_path.name
        return document

    monkeypatch.setattr(service, "process_document", process_document)
    monkeypatch.setattr(service, "get_workspace", lambda: workspace)
    return calls


def submit(client, name, data, **fields):
    return client.post(
        f"{BASE}/ocr/standard",
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


def zip_of(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def test_the_page_offers_the_tools_and_the_options(client):
    html = client.get(f"{BASE}/ocr").text
    assert 'name="tool" value="standard" checked' in html
    for value in ("engine", "handwriting", "huggingface"):
        assert f'name="tool" value="{value}" disabled' in html
    assert "Soon" in html
    assert 'name="searchable_pdf" value="1" checked' in html
    assert '<option value="auto" selected>Detect page by page</option>' in html
    assert ".zip" in html
    assert "questionnaire" not in html.lower()


def test_a_document_is_read_and_shown_beside_its_page(
    client, runner, pipeline, released
):
    job = finished(
        runner,
        submit(
            client,
            "report.pdf",
            b"%PDF",
            searchable_pdf="1",
            ocr_lang="deu",
            highest_quality="1",
            describe_images="1",
        ),
    )
    assert job.state is JobState.DONE, job.error_details
    run = pipeline.runs[0]
    assert run["native_fast_lane"] is False
    assert run["describe_images"] is True
    assert run["searchable_pdf"] is True
    assert run["ocr_lang"] == "deu"
    assert run["extract_survey"] is False
    assert run["cancel"] is job.cancel
    assert released == ["ocr"]

    html = card(client, job)
    assert "1 read by the model" in html
    assert 'class="scan__image"' in html
    assert re.search(r'<rect class="box rt-title"', html)
    assert 'class="ocr-block rt-table"' in html
    assert "<td>a</td>" in html
    assert "alert(1)" not in html
    assert "Tables (1)" in html and "Figures (1)" in html
    assert "A small white square." in html
    assert "mark mark--checked" in html
    assert "vision:7b" in html

    names = [item.name for item in job.listed_downloads]
    assert names[0] == "report_bundle.zip"
    assert "report.txt" in names
    image = client.get(f"{BASE}/jobs/{job.id}/files/report_page-0001.png")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    assert "report_page-0001.png" not in names
    csv = client.get(f"{BASE}/jobs/{job.id}/files/report_table-p1_r1.csv")
    assert csv.status_code == 200


def test_the_language_follows_the_searchable_pdf(client, runner, pipeline):
    finished(runner, submit(client, "a.png", b"png", ocr_lang="deu"))
    assert pipeline.runs[0]["searchable_pdf"] is False
    assert pipeline.runs[0]["ocr_lang"] == service.DEFAULT_TESSERACT_LANG
    finished(runner, submit(client, "b.png", b"png", searchable_pdf="1"))
    assert pipeline.runs[1]["ocr_lang"] == "auto"


def test_a_zip_is_read_as_a_batch(client, runner, pipeline):
    job = finished(
        runner,
        submit(
            client,
            "scans.zip",
            zip_of({"a.pdf": b"a", "more/b.png": b"b", "notes.txt": b"x"}),
        ),
    )
    assert job.state is JobState.DONE, job.error_details
    assert [run["name"] for run in pipeline.runs] == ["a.pdf", "b.png"]
    assert all(run["extract_survey"] is False for run in pipeline.runs)
    html = card(client, job)
    assert "2 files" in html
    assert "more/b.png" in html
    assert [item.name for item in job.listed_downloads] == ["scans_ocr.zip"]
    archive = client.get(f"{BASE}/jobs/{job.id}/files/scans_ocr.zip").content
    names = zipfile.ZipFile(io.BytesIO(archive)).namelist()
    assert "a/document.md" in names
    assert "more/b/document.md" in names


def test_a_zip_without_documents_fails_with_the_reason(
    client, runner, pipeline
):
    job = finished(runner, submit(client, "x.zip", zip_of({"a.txt": b"x"})))
    assert job.state is JobState.FAILED
    assert "No valid documents" in job.error


def test_reading_can_be_cancelled(client, runner, pipeline):
    pipeline.gate = threading.Event()
    response = submit(client, "a.pdf", b"%PDF")
    job_id = re.search(r'id="job-([0-9a-f]+)"', response.text).group(1)
    job = runner.get(job_id)
    wait_for(lambda: len(pipeline.runs) == 1)
    client.post(f"{BASE}/jobs/{job.id}/cancel", headers=htmx(client))
    wait_for(lambda: not job.active)
    assert job.state is JobState.CANCELLED
    assert job.downloads == []


def test_a_failure_is_shown(client, runner, pipeline):
    pipeline.fail = RuntimeError("PaddleOCR-VL backend failed.")
    job = finished(runner, submit(client, "a.pdf", b"%PDF"))
    assert job.state is JobState.FAILED
    assert "PaddleOCR-VL backend failed." in card(client, job)


@pytest.mark.parametrize(
    "name,fields,message",
    [
        ("a.docx", {}, "not a file this page reads"),
        ("a.pdf", {"searchable_pdf": "1", "ocr_lang": "klingon"}, "language"),
    ],
)
def test_bad_requests_are_explained(client, pipeline, name, fields, message):
    response = submit(client, name, b"x", **fields)
    assert response.status_code == 422
    assert message in response.text


def test_region_counts_read_well():
    assert page.region_counts({"text": 12, "table": 1, "checkbox": 2}) == [
        "12 text blocks",
        "2 check boxes",
        "1 table",
    ]
    assert page.region_label("checkbox") == "Check box"
    assert page.region_label("odd") == "Odd"
