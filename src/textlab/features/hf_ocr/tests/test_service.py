"""The service with a fake worker: outputs, batch layout, errors, cleanup."""

import io
import json
import sys
import zipfile

import pytest
from PIL import Image

from textlab.common import jobs
from textlab.common.storage import Workspace
from textlab.features.hf_ocr import service

OPTIONS = service.HfOcrOptions("org/model", prompt="Read.")


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path / "workspace")
    monkeypatch.setattr(service, "get_workspace", lambda: workspace)
    return workspace


@pytest.fixture
def worker_calls(monkeypatch, tmp_path):
    """A fake worker: each page is read as its image's file name."""
    calls = []

    def run_worker(module, request, **kwargs):
        calls.append({"module": module, "request": request, **kwargs})
        for image in request["images"]:
            assert image.endswith(".png")
        kwargs["on_progress"](service.Progress("Reading pages...", 1.0))
        return {
            "texts": [image.rsplit("/", 1)[-1] for image in request["images"]],
            "model": {"repo_id": request["repo_id"], "revision": "r1"},
        }

    monkeypatch.setattr(service.jobs, "run_worker", run_worker)
    monkeypatch.setenv("TEXT_LAB_HF_OCR_DIR", str(tmp_path / "models"))
    monkeypatch.delenv("HF_OCR_BACKEND_PYTHON", raising=False)
    return calls


def png(width=60, height=40):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def names_in(zip_bytes):
    return sorted(zipfile.ZipFile(io.BytesIO(zip_bytes)).namelist())


def test_a_document_is_read_and_offered_for_download(
    workspace, worker_calls, tmp_path
):
    updates = []
    run = service.read_document(
        "../x/letter.png", png(), OPTIONS, on_progress=updates.append
    )
    assert [page.text for page in run.document.pages] == ["page-0001.png"]
    assert run.stem == "letter"
    assert run.model.citation() == "org/model (revision r1)"
    assert json.loads(run.to_json())["prompt"] == "Read."
    assert len(run.previews) == 1
    assert names_in(run.zip_bytes) == [
        "letter.json",
        "letter.txt",
        "model_used.txt",
    ]
    (call,) = worker_calls
    assert call["module"] == "textlab.features.hf_ocr.worker"
    assert call["python"] == "/opt/conda/envs/hf_ocr_backend/bin/python"
    assert call["request"]["repo_id"] == "org/model"
    assert call["request"]["max_new_tokens"] == OPTIONS.max_new_tokens
    # The model goes into the user's folder, created for the worker.
    assert call["request"]["cache_dir"] == str(tmp_path / "models")
    assert (tmp_path / "models").is_dir()
    fractions = [u.fraction for u in updates if u.fraction is not None]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    assert not any(workspace.dir(service.WORKSPACE_AREA).iterdir())


def test_a_batch_mirrors_the_archive_with_one_worker_run(
    workspace, worker_calls
):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("box1/a.png", png())
        archive.writestr("b.png", png())
        archive.writestr("readme.txt", b"ignored")
    buffer.seek(0)
    result = service.read_batch(buffer, OPTIONS)
    assert (result.files, result.pages) == (2, 2)
    assert len(worker_calls) == 1
    assert len(worker_calls[0]["request"]["images"]) == 2
    names = names_in(result.zip_bytes)
    assert names == [
        "b/b.json",
        "b/b.txt",
        "box1/a/a.json",
        "box1/a/a.txt",
        "model_used.txt",
    ]


def test_an_archive_without_documents_is_an_error(workspace, worker_calls):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("notes.txt", b"x")
    buffer.seek(0)
    with pytest.raises(service.HfOcrError, match="no supported"):
        service.read_batch(buffer, OPTIONS)
    with pytest.raises(service.HfOcrError, match="ZIP"):
        service.read_batch(io.BytesIO(b"not a zip"), OPTIONS)


def test_an_unreadable_file_is_an_error(workspace, worker_calls):
    with pytest.raises(service.HfOcrError, match="broken.png"):
        service.read_document("broken.png", b"not an image", OPTIONS)
    assert worker_calls == []


def test_the_workers_message_for_users_is_shown_as_it_is(
    workspace, monkeypatch, tmp_path
):
    def refuse(module, request, **kwargs):
        raise jobs.WorkerError(
            "OcrModelError: org/model is gated",
            "traceback",
            error_type="OcrModelError",
            error_message="org/model is gated",
        )

    monkeypatch.setattr(service.jobs, "run_worker", refuse)
    with pytest.raises(service.HfOcrError) as raised:
        service.read_pages([], OPTIONS, cache_dir=tmp_path)
    assert str(raised.value) == "org/model is gated"


def test_other_worker_failures_keep_their_details(
    workspace, monkeypatch, tmp_path
):
    def crash(module, request, **kwargs):
        raise jobs.WorkerError("RuntimeError: boom", "traceback")

    monkeypatch.setattr(service.jobs, "run_worker", crash)
    with pytest.raises(jobs.WorkerError) as raised:
        service.read_pages([], OPTIONS, cache_dir=tmp_path)
    assert raised.value.details == "traceback"


def test_the_worker_interpreter_can_be_overridden(
    workspace, worker_calls, monkeypatch, tmp_path
):
    monkeypatch.setenv("HF_OCR_BACKEND_PYTHON", "/env/bin/python")
    service.read_pages([], OPTIONS, cache_dir=tmp_path)
    service.read_pages(
        [], OPTIONS, cache_dir=tmp_path, backend_python=sys.executable
    )
    assert [call["python"] for call in worker_calls] == [
        "/env/bin/python",
        sys.executable,
    ]
