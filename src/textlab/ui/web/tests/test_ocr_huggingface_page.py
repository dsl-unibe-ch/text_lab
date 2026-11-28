"""Hugging Face models on the OCR page, with the model's worker faked.

The worker reads each page as its image's file name (and can be held at
a gate, to test cancelling and deleting models while a job runs); the rest
of the service runs for real: pages, the downloads, the saved page images,
the batch ZIP, the user's model folder.
"""

import io
import re
import threading
import zipfile
from types import SimpleNamespace

import pytest

from textlab.common.progress import Progress, check_cancelled
from textlab.features.hf_ocr import service
from textlab.features.hf_ocr.tests.test_service import png
from textlab.ui.web.jobs import JobState
from textlab.ui.web.tests.conftest import BASE, wait_for
from textlab.ui.web.tests.test_job_pages import htmx


@pytest.fixture
def folder(tmp_path, monkeypatch):
    """The user's model folder."""
    path = tmp_path / "hf_models"
    monkeypatch.setenv("TEXT_LAB_HF_OCR_DIR", str(path))
    return path


def store_model(folder, repo, size=2048):
    model = folder / ("models--" + repo.replace("/", "--"))
    model.mkdir(parents=True)
    (model / "weights").write_bytes(b"x" * size)


@pytest.fixture
def worker(monkeypatch, workspace, folder):
    """Fake the worker; record its requests."""
    calls = SimpleNamespace(requests=[], cancel=[], gate=None)

    def run_worker(module, request, **kwargs):
        calls.requests.append(request)
        calls.cancel.append(kwargs.get("cancel"))
        kwargs["on_progress"](Progress("Reading pages...", 0.5))
        if calls.gate is not None:
            while not calls.gate.wait(0.01):
                check_cancelled(kwargs.get("cancel"))
        return {
            "texts": [
                image.rsplit("/", 1)[-1] + " <i>"
                for image in request["images"]
            ],
            "model": {"repo_id": request["repo_id"], "revision": "abc123"},
        }

    monkeypatch.setattr(service.jobs, "run_worker", run_worker)
    monkeypatch.setattr(service, "get_workspace", lambda: workspace)
    monkeypatch.delenv("HF_OCR_BACKEND_PYTHON", raising=False)
    return calls


def submit(client, name, data, **fields):
    fields = {"repo": "org/model", "max_new_tokens": "1024", **fields}
    return client.post(
        f"{BASE}/ocr/huggingface",
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


def test_the_form_asks_for_a_model_and_instructions(client, folder):
    html = client.get(f"{BASE}/ocr").text
    assert 'data-when="huggingface"' in html
    assert 'name="repo" type="text" required' in html
    assert service.DEFAULT_PROMPT in html
    assert (
        f'max="{service.MAX_NEW_TOKENS_LIMIT}" '
        f'value="{service.DEFAULT_MAX_NEW_TOKENS}"' in html
    )
    assert 'id="hf-file"' in html
    assert "None yet." in html
    assert str(folder) in html


def test_downloaded_models_are_listed_to_use_or_delete(client, folder):
    store_model(folder, "org/small")
    html = client.get(f"{BASE}/ocr").text
    assert "Your downloaded models · 1, 0 MB" in html
    assert 'data-fill="hf-repo" data-fill-value="org/small"' in html
    assert """hx-vals='{"repo": "org/small"}' hx-params="repo\"""" in html


def test_a_document_is_read_with_the_named_model(
    client, runner, worker, released
):
    job = finished(
        runner,
        submit(
            client,
            "letter.png",
            png(),
            repo="https://huggingface.co/org/model/tree/main",
            prompt="  Read it.  ",
            max_new_tokens="512",
        ),
    )
    assert job.state is JobState.DONE, job.error_details
    (request,) = worker.requests
    assert request["repo_id"] == "org/model"
    assert request["prompt"] == "Read it."
    assert request["max_new_tokens"] == 512
    assert worker.cancel[0] is job.cancel
    assert released == ["hf_ocr"]

    html = card(client, job)
    assert "org/model" in html and "1 page" in html
    assert "page-0001.png &lt;i&gt;" in html
    assert "org/model (revision abc123)" in html
    assert "Read it." in html
    assert 'href="https://huggingface.co/org/model"' in html
    names = [item.name for item in job.listed_downloads]
    assert names == ["letter_hf_ocr.zip", "letter.txt", "letter.json"]
    image = client.get(f"{BASE}/jobs/{job.id}/files/letter_page-0001.png")
    assert image.headers["content-type"] == "image/png"


def test_a_zip_is_read_as_a_batch(client, runner, worker):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("box/a.png", png())
        archive.writestr("b.png", png())
    job = finished(runner, submit(client, "scans.zip", buffer.getvalue()))
    assert job.state is JobState.DONE, job.error_details
    assert len(worker.requests) == 1
    html = card(client, job)
    assert "2 files" in html and "box/a.png" in html
    assert [item.name for item in job.listed_downloads] == ["scans_hf_ocr.zip"]


def test_reading_can_be_cancelled(client, runner, worker):
    worker.gate = threading.Event()
    response = submit(client, "a.png", png())
    job_id = re.search(r'id="job-([0-9a-f]+)"', response.text).group(1)
    job = runner.get(job_id)
    wait_for(lambda: len(worker.requests) == 1)
    client.post(f"{BASE}/jobs/{job.id}/cancel", headers=htmx(client))
    wait_for(lambda: not job.active)
    assert job.state is JobState.CANCELLED


def test_a_model_is_deleted_and_the_list_redrawn(client, folder):
    store_model(folder, "org/small")
    store_model(folder, "org/large")
    response = client.post(
        f"{BASE}/ocr/huggingface/models/delete",
        data={"repo": "org/small"},
        headers=htmx(client),
    )
    assert response.status_code == 200
    assert 'id="hf-models"' in response.text
    assert "org/small" not in response.text and "org/large" in response.text
    assert [model.repo_id for model in service.stored_models()] == [
        "org/large"
    ]


def test_no_model_is_deleted_while_one_is_in_use(
    client, runner, worker, folder
):
    store_model(folder, "org/model")
    worker.gate = threading.Event()
    submit(client, "a.png", png())
    wait_for(lambda: len(worker.requests) == 1)
    response = client.post(
        f"{BASE}/ocr/huggingface/models/delete",
        data={"repo": "org/model"},
        headers=htmx(client),
    )
    assert response.status_code == 422
    assert "in use" in response.text
    assert service.stored_models()[0].repo_id == "org/model"
    worker.gate.set()


@pytest.mark.parametrize(
    "fields,message",
    [
        ({"repo": "not a model"}, "not the name of a Hugging Face model"),
        ({"max_new_tokens": "0"}, "token limit"),
        ({"max_new_tokens": "many"}, "token limit"),
    ],
)
def test_bad_requests_are_explained(client, worker, fields, message):
    response = submit(client, "a.png", png(), **fields)
    assert response.status_code == 422
    assert message in response.text
    assert worker.requests == []
