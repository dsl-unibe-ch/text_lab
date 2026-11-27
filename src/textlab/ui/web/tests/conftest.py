"""Fixtures for the web UI's tests: an app under a session's base path, a
job runner with a fake GPU release, clients with and without login, and a
fake transcription backend.

The test app also serves the demo job of ``demo_job.py``, which the tests
of the job pages and of the app run instead of a feature's job.
"""

import re
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from textlab.common import storage
from textlab.common.config import get_settings
from textlab.common.storage import Workspace
from textlab.features.transcription import service as transcription
from textlab.ui.web import auth, pages
from textlab.ui.web.app import create_app
from textlab.ui.web.jobs import JobRunner
from textlab.ui.web.settings import WebSettings
from textlab.ui.web.tests import demo_job

BASE = "/node/gnode99.example.org/31234"
PASSWORD = "Correct7Horse8Battery"
PORTAL = "ondemand.example.org"
OWNER = "alice"
UPLOAD_LIMIT = 10 * 1000 * 1000


def wait_for(condition, timeout=10.0):
    """Wait until ``condition()`` is true; fail after ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for a condition")


def csrf_token(page_html):
    """Return the CSRF token a page hands to htmx."""
    return re.search(r'"X-CSRF-Token": "([^"]+)"', page_html).group(1)


@pytest.fixture(autouse=True)
def no_login_delay(monkeypatch):
    monkeypatch.setattr(auth, "FAILED_LOGIN_DELAY", 0)


class ReleaseLog(list):
    """A fake GPU release that notes what it released the GPU for.

    Each call adds the feature, or ``"feature:model"`` when an Ollama model
    is to stay loaded.
    """

    def __call__(self, feature, ollama_model=None):
        self.append(
            feature if ollama_model is None else f"{feature}:{ollama_model}"
        )


@pytest.fixture
def released():
    """The features the fake GPU release was called for, in order."""
    return ReleaseLog()


@pytest.fixture
def workspace(tmp_path):
    return Workspace(tmp_path / "workspace")


@pytest.fixture
def runner(released, workspace):
    runner = JobRunner(prepare_gpu=released, workspace=workspace)
    yield runner
    runner.close()


@pytest.fixture
def settings():
    return WebSettings(
        token=PASSWORD,
        base_path=BASE,
        owner=OWNER,
        public_host=PORTAL,
        upload_limit=UPLOAD_LIMIT,
    )


@pytest.fixture
def app(settings, runner, monkeypatch):
    routers = pages.routers
    monkeypatch.setattr(
        pages, "routers", lambda: [*routers(), demo_job.router]
    )
    return create_app(settings, runner=runner)


def make_client(app, **kwargs):
    # Not used as a context manager: the app's lifespan would close the
    # shared runner; the runner fixture closes it instead.
    return TestClient(
        app, base_url="https://testserver", follow_redirects=False, **kwargs
    )


@pytest.fixture
def anonymous(app):
    """A client without login."""
    return make_client(app)


@pytest.fixture
def client(app):
    """A logged-in client; ``client.csrf`` is its session's CSRF token."""
    client = make_client(app)
    response = client.post(f"{BASE}/login", data={"password": PASSWORD})
    assert response.status_code == 303
    client.csrf = csrf_token(client.get(f"{BASE}/").text)
    return client


def transcript_for(name):
    """A transcript with two speakers, as WhisperX gives it."""
    return transcription.Transcript(
        name=name,
        language="de",
        duration_seconds=12.5,
        has_speakers=True,
        segments=[
            {
                "start": 0.0,
                "end": 1.6,
                "text": "Guten Morgen. Hallo.",
                "words": [
                    {
                        "word": "Guten",
                        "start": 0.0,
                        "end": 0.4,
                        "speaker": "SPEAKER_00",
                    },
                    {
                        "word": "Morgen.",
                        "start": 0.5,
                        "end": 1.0,
                        "speaker": "SPEAKER_00",
                    },
                    {
                        "word": "Hallo.",
                        "start": 1.2,
                        "end": 1.6,
                        "speaker": "SPEAKER_01",
                    },
                ],
            }
        ],
    )


@pytest.fixture
def backend(monkeypatch, tmp_path):
    """Fake the transcription and the playback; record the calls."""
    monkeypatch.setenv("TEXT_LAB_WORKDIR", str(tmp_path / "work"))
    get_settings.cache_clear()
    storage.get_workspace.cache_clear()
    calls = SimpleNamespace(runs=[], gate=None)

    def prepare_playback(source, folder, name, *, wav=False, cancel=None):
        assert source.is_file()
        mp3 = folder / f"{name}.mp3"
        mp3.write_bytes(b"ID3" + bytes(997))
        wav_path = None
        if wav:
            wav_path = folder / f"{name}.wav"
            wav_path.write_bytes(b"RIFF" + bytes(96))
        return transcription.Playback(
            path=mp3, duration_seconds=12.5, levels=(1, 4, 8, 2), wav=wav_path
        )

    def run_transcription(files, options, *, on_progress, cancel=None):
        calls.runs.append(([file.name for file in files], options))
        for file in files:
            assert file.path.is_file()
        if calls.gate is not None:
            assert calls.gate.wait(10)
        return transcription.TranscriptionResult(
            transcripts=[
                transcript_for(file.name)
                for file in files
                if not file.name.startswith("broken")
            ],
            skipped=[
                transcription.SkippedFile(file.name, "could not decode audio")
                for file in files
                if file.name.startswith("broken")
            ],
        )

    monkeypatch.setattr(transcription, "prepare_playback", prepare_playback)
    monkeypatch.setattr(transcription, "run_transcription", run_transcription)
    yield calls
    get_settings.cache_clear()
    storage.get_workspace.cache_clear()
