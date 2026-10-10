"""Worker processes: results, errors, crashes, cancellation and cleanup."""

import os
import threading
import time
from pathlib import Path

import pytest

import textlab
from textlab.common import jobs, storage
from textlab.common.config import get_settings
from textlab.common.progress import CancelledError, Progress

STUB = "textlab.common.tests.stub_worker"


@pytest.fixture(autouse=True)
def workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("TEXT_LAB_WORKDIR", str(tmp_path / "work"))
    get_settings.cache_clear()
    storage.get_workspace.cache_clear()
    yield tmp_path / "work"
    get_settings.cache_clear()
    storage.get_workspace.cache_clear()


def run(request, **kwargs):
    kwargs.setdefault("poll_interval", 0.05)
    return jobs.run_worker(STUB, request, area="tests", **kwargs)


def test_the_result_and_progress_come_back(workspace):
    updates = []
    result = run({"mode": "ok", "value": "hi"}, on_progress=updates.append)
    assert result == {"echo": "hi", "number": 1.5}
    assert Progress("working", 0.5) in updates


def test_the_job_folder_is_removed(workspace):
    run({"mode": "ok"})
    assert list((workspace / "tests").iterdir()) == []


def test_a_failure_carries_the_worker_traceback():
    expected = "ValueError: bad input"
    with pytest.raises(jobs.WorkerError, match=expected) as error:
        run({"mode": "fail"})
    assert "Traceback" in error.value.details


def test_a_crash_without_result_is_reported():
    with pytest.raises(jobs.WorkerError, match="exit code 3"):
        run({"mode": "crash"})


def test_cancel_stops_the_worker(workspace):
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(CancelledError):
        run({"mode": "hang"}, cancel=cancel)
    assert time.monotonic() - started < jobs.STOP_TIMEOUT + 5
    assert list((workspace / "tests").iterdir()) == []


def test_an_error_in_the_progress_callback_stops_the_worker():
    def broken(progress):
        raise RuntimeError("page closed")

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="page closed"):
        run({"mode": "hang"}, on_progress=broken)
    assert time.monotonic() - started < jobs.STOP_TIMEOUT + 5


def test_the_worker_can_import_textlab_without_pythonpath(monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    assert run({"mode": "ok", "value": 1})["echo"] == 1


def test_worker_environment_for_another_interpreter(tmp_path, monkeypatch):
    python = tmp_path / "envs" / "paddle" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    env = jobs.worker_environment(str(python))
    assert env["PATH"].split(os.pathsep)[:2] == [
        str(python.parent.resolve()),
        "/usr/bin",
    ]
    assert env["LD_LIBRARY_PATH"] == str(
        python.parent.resolve().parent / "lib"
    )
    src_root = str(Path(textlab.__file__).resolve().parents[1])
    assert env["PYTHONPATH"].split(os.pathsep)[0] == src_root


def test_worker_environment_keeps_path_for_this_interpreter(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    assert jobs.worker_environment()["PATH"] == "/usr/bin"
