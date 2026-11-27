"""The OCR page, with the recognition faked.

For Standard, ``process_document`` returns a page with a title, a table, a
figure and a check box (and can be held at a gate, to test cancelling);
for Choose an engine, the engines are replaced by fakes that keep their
names and settings and return outlined lines, a table or one text. The
rest of the service runs for real: the downloads, the saved page images,
figures and tables, the batch ZIPs.
"""

import io
import re
import threading
import zipfile
from types import SimpleNamespace

import pytest

from textlab.common.progress import Progress, check_cancelled
from textlab.features.ocr import manual, service
from textlab.features.ocr.engines import base, easy_ocr, glm_ocr, olm_ocr
from textlab.features.ocr.tests.test_service import png, rich_document
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


@pytest.fixture(autouse=True)
def ollama_running(monkeypatch):
    """GLM-OCR's server is running, unless a test says otherwise."""
    state = SimpleNamespace(running=True)
    monkeypatch.setattr(
        glm_ocr, "check_ollama_server", lambda attempts: state.running
    )
    return state


def submit(client, name, data, *, route="standard", **fields):
    return client.post(
        f"{BASE}/ocr/{route}",
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
    assert 'name="tool" value="engine">' in html
    for value in ("handwriting", "huggingface"):
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


# --- Choose an engine -------------------------------------------------------

LINES = [
    base.TextLine("Hello", ((1, 2), (11, 2), (11, 6), (1, 6))),
    base.TextLine("there <b>", ((1, 8), (11, 8), (11, 12), (1, 12))),
]
TABLE = "<table><tr><th>a</th></tr><tr><td>1</td></tr></table>"


@pytest.fixture
def engines(monkeypatch, workspace):
    """Fake the engines' recognition; record how each was asked to run.

    EasyOCR and PaddleOCR return outlined lines, GLM-OCR a heading and a
    table, OlmOCR one text without a page image.
    """
    calls = SimpleNamespace(runs=[], gate=None)

    def recognize(
        self, input_path, work_dir, options, on_progress=None, cancel=None
    ):
        calls.runs.append(
            {
                "engine": self.name,
                "name": input_path.name,
                "options": options,
                "cancel": cancel,
            }
        )
        if on_progress is not None:
            on_progress(Progress("Running page 1", 0.5))
        if calls.gate is not None:
            while not calls.gate.wait(0.01):
                check_cancelled(cancel)
        if isinstance(self, olm_ocr.OlmOcr):
            page = base.PageText(1, "# Paper\n\nText.", {"text": "x"})
            return base.EngineOutput(pages=[page], record='{"text": "x"}')
        if isinstance(self, glm_ocr.GlmOcr):
            page = base.PageText(1, f"# Prices\n{TABLE}", {}, image=png())
            return base.EngineOutput(pages=[page])
        page = base.PageText(1, "Hello\nthere <b>", [], LINES, png())
        return base.EngineOutput(pages=[page])

    for engine in service.ENGINES.values():
        kind = type(engine)
        monkeypatch.setattr(kind, "prepare", lambda *args: None)
        monkeypatch.setattr(kind, "recognize", recognize)
    monkeypatch.setattr(manual, "get_workspace", lambda: workspace)
    return calls


def test_the_engine_form_offers_each_engine_with_its_settings(client):
    html = client.get(f"{BASE}/ocr").text
    assert 'data-when="engine" data-switch="engine"' in html
    assert 'name="engine" value="EasyOCR" data-readout="engine-name"' in html
    assert re.search(r'value="EasyOCR"[^>]*checked', html)
    for name in ("PaddleOCR", "OlmOCR", "GLM-OCR"):
        assert f'name="engine" value="{name}"' in html
    assert 'name="language_easyocr"' in html
    assert '<option value="en" selected>English</option>' in html
    assert 'name="language_paddleocr"' in html
    assert 'name="mode_glm_ocr" value="Table Recognition"' in html
    assert "OlmOCR has no settings" in html
    assert (
        'id="engine-file"' in html and 'data-file-list="engine-file"' in html
    )
    assert easy_ocr.EasyOcr.description in html


def test_an_engine_that_cannot_run_is_offered_switched_off(
    client, ollama_running
):
    ollama_running.running = False
    html = client.get(f"{BASE}/ocr").text
    assert re.search(r'value="GLM-OCR"[^>]*disabled', html)
    assert "The language model server is not running." in html


def test_a_document_is_read_with_the_chosen_engine(
    client, runner, engines, released
):
    job = finished(
        runner,
        submit(
            client,
            "scan.png",
            png(),
            route="engine",
            engine="EasyOCR",
            language_easyocr="de",
            language_paddleocr="fr",
        ),
    )
    assert job.state is JobState.DONE, job.error_details
    (run,) = engines.runs
    assert run["engine"] == "EasyOCR"
    assert run["options"].language == "de"
    assert run["options"].previews is False
    assert run["cancel"] is job.cancel
    assert released == ["ocr"]

    html = card(client, job)
    assert "EasyOCR · German" in html
    assert "2 text blocks outlined" in html
    assert '<polygon class="box rt-text" points="1.0,2.0 11.0,2.0' in html
    assert re.search(
        r'<p class="ocr-line rt-text" data-region="[^"]+">Hello', html
    )
    assert "there &lt;b&gt;" in html and "there <b>" not in html
    assert f'data-copy-from="{job.id}-text-1"' in html

    names = [item.name for item in job.listed_downloads]
    assert names == ["scan_outputs.zip", "scan.txt", "scan.json"]
    image = client.get(f"{BASE}/jobs/{job.id}/files/scan_page-0001.png")
    assert image.headers["content-type"] == "image/png"


def test_tables_an_engine_writes_are_shown_as_tables(
    client, runner, engines, released
):
    job = finished(
        runner,
        submit(
            client,
            "prices.pdf",
            b"%PDF",
            route="engine",
            engine="GLM-OCR",
            mode_glm_ocr="Table Recognition",
        ),
    )
    assert job.state is JobState.DONE, job.error_details
    assert engines.runs[0]["options"].mode == "Table Recognition"
    assert released == [f"ocr:{glm_ocr.MODEL}"]
    html = card(client, job)
    assert "GLM-OCR · Table Recognition" in html
    assert "1 table" in html
    assert "<td>1</td>" in html
    assert "<h1>Prices</h1>" in html
    assert "prices_page-0001_table-1.csv" in html
    csv = client.get(
        f"{BASE}/jobs/{job.id}/files/prices_page-0001_table-1.csv"
    )
    assert csv.text.splitlines() == ["a", "1"]


def test_an_engine_that_reads_whole_documents_gives_one_text(
    client, runner, engines
):
    job = finished(
        runner,
        submit(client, "paper.pdf", b"%PDF", route="engine", engine="OlmOCR"),
    )
    assert job.state is JobState.DONE, job.error_details
    html = card(client, job)
    assert "ocr-page ocr-page--text" in html
    assert "The document" in html
    assert "scan__image" not in html
    assert "<h1>Paper</h1>" in html
    names = [item.name for item in job.listed_downloads]
    assert "paper.jsonl" in names


def test_a_zip_is_read_with_the_chosen_engine_as_a_batch(
    client, runner, engines
):
    job = finished(
        runner,
        submit(
            client,
            "scans.zip",
            zip_of({"a.png": png(), "more/b.pdf": b"%PDF"}),
            route="engine",
            engine="PaddleOCR",
            language_paddleocr="de",
        ),
    )
    assert job.state is JobState.DONE, job.error_details
    assert [run["name"] for run in engines.runs] == ["a.png", "b.pdf"]
    assert all(run["options"].language == "de" for run in engines.runs)
    html = card(client, job)
    assert "PaddleOCR · German" in html
    assert "2 files" in html and "more/b.pdf" in html
    assert [item.name for item in job.listed_downloads] == [
        "scans_paddleocr.zip"
    ]
    archive = client.get(f"{BASE}/jobs/{job.id}/files/scans_paddleocr.zip")
    names = zipfile.ZipFile(io.BytesIO(archive.content)).namelist()
    assert "a/a.txt" in names and "more/b/b.txt" in names


def test_reading_with_an_engine_can_be_cancelled(client, runner, engines):
    engines.gate = threading.Event()
    response = submit(client, "a.png", png(), route="engine", engine="EasyOCR")
    job_id = re.search(r'id="job-([0-9a-f]+)"', response.text).group(1)
    job = runner.get(job_id)
    wait_for(lambda: len(engines.runs) == 1)
    client.post(f"{BASE}/jobs/{job.id}/cancel", headers=htmx(client))
    wait_for(lambda: not job.active)
    assert job.state is JobState.CANCELLED
    assert job.downloads == []


@pytest.mark.parametrize(
    "fields,message",
    [
        ({"engine": "Tesseract"}, "Choose an engine"),
        ({"engine": "EasyOCR", "language_easyocr": "xx"}, "language"),
        ({"engine": "GLM-OCR", "mode_glm_ocr": "Poems"}, "what to read"),
    ],
)
def test_bad_engine_requests_are_explained(client, engines, fields, message):
    response = submit(client, "a.png", b"x", route="engine", **fields)
    assert response.status_code == 422
    assert message in response.text


def test_an_engine_that_cannot_run_is_refused(client, engines, ollama_running):
    ollama_running.running = False
    response = submit(client, "a.png", b"x", route="engine", engine="GLM-OCR")
    assert response.status_code == 422
    assert "GLM-OCR cannot run now" in response.text
    assert engines.runs == []


def test_engine_names_become_field_names():
    assert page.engine_slug("GLM-OCR") == "glm_ocr"
    assert page.engine_slug("EasyOCR") == "easyocr"
