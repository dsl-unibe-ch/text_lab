"""Fixtures for the web UI's tests: an app under a session's base path, a
job runner with a fake GPU release, and clients with and without login."""

import re
import time

import pytest
from fastapi.testclient import TestClient

from textlab.common.storage import Workspace
from textlab.ui.web import auth
from textlab.ui.web.app import create_app
from textlab.ui.web.jobs import JobRunner
from textlab.ui.web.settings import WebSettings

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


@pytest.fixture
def released():
    """The features the fake GPU release was called for, in order."""
    return []


@pytest.fixture
def workspace(tmp_path):
    return Workspace(tmp_path / "workspace")


@pytest.fixture
def runner(released, workspace):
    runner = JobRunner(prepare_gpu=released.append, workspace=workspace)
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
        dev=True,
    )


@pytest.fixture
def app(settings, runner):
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
