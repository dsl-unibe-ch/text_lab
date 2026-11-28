import re
import time

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from textlab.ui.web.app import SECURITY_HEADERS, create_app, create_pages
from textlab.ui.web.pages import TOOLS
from textlab.ui.web.render import (
    NOTICE_HEADER,
    duration,
    file_size,
    markdown,
    time_left,
)
from textlab.ui.web.settings import WebSettings
from textlab.ui.web.tests.conftest import BASE, PASSWORD, UPLOAD_LIMIT
from textlab.ui.web.tests.demo_job import PATH

PAGES = ["/", *(tool.path for tool in TOOLS)]


@pytest.mark.parametrize("path", PAGES)
def test_every_page_renders_in_the_layout(client, path):
    response = client.get(f"{BASE}{path}")
    assert response.status_code == 200, response.text[:500]
    page = response.text
    assert "AI models can produce inaccurate or misleading output" in page
    assert f'href="{BASE}/"' in page
    assert f"{BASE}/static/css/tokens.css?v=" in page
    assert 'id="job-count"' in page
    assert f'data-root="{BASE}"' in page


def test_the_navigation_numbers_the_tools_in_groups(client):
    page = client.get(f"{BASE}/").text
    number = '<span class="index__num">01</span>'
    assert f'{number}<span class="index__name">Transcription</span>' in page
    assert page.index(">Audio<") < page.index(">Documents<")


def test_the_status_bar_shows_when_the_session_ends(runner):
    settings = WebSettings(
        token=PASSWORD, base_path=BASE, session_end=time.time() + 3 * 3600
    )
    client = TestClient(
        create_app(settings, runner=runner),
        base_url="https://testserver",
        follow_redirects=False,
    )
    client.post(f"{BASE}/login", data={"password": PASSWORD})
    page = client.get(f"{BASE}/").text
    assert "Session ends" in page
    assert re.search(r"2 h 5\d min|3 h 00 min", page)


def test_the_current_page_is_marked_in_the_navigation(client):
    page = client.get(f"{BASE}/transcription").text
    assert f'href="{BASE}/transcription" aria-current="page"' in page


def test_security_headers_are_sent(client):
    for path in ("/", "/login", "/nowhere", "/static/js/app.js"):
        response = client.get(f"{BASE}{path}")
        for name in ("Content-Security-Policy", "X-Content-Type-Options"):
            assert response.headers[name] == SECURITY_HEADERS[name], path
    csp = SECURITY_HEADERS["Content-Security-Policy"]
    assert "script-src 'self';" in csp
    assert "unsafe" not in csp


def test_pages_are_not_cached_but_versioned_files_are(client):
    assert client.get(f"{BASE}/").headers["cache-control"] == "no-store"
    versioned = client.get(f"{BASE}/static/js/app.js?v=1")
    assert "immutable" in versioned.headers["cache-control"]
    plain = client.get(f"{BASE}/static/js/app.js")
    assert plain.headers["cache-control"] == "no-cache"
    # Vendored files carry their release in their path.
    font = client.get(
        f"{BASE}/static/vendor/ibm-plex-mono-5.3.0/"
        "ibm-plex-mono-latin-400-normal.woff2"
    )
    assert font.status_code == 200
    assert "immutable" in font.headers["cache-control"]


def test_the_base_path_without_a_slash_redirects(anonymous):
    response = anonymous.get(BASE)
    assert response.status_code == 307
    assert response.headers["location"] == f"{BASE}/"


def test_paths_outside_the_base_path_name_it(anonymous):
    response = anonymous.get("/elsewhere")
    assert response.status_code == 404
    assert f"{BASE}/" in response.text


def test_an_unknown_page_is_explained(client):
    response = client.get(f"{BASE}/nowhere")
    assert response.status_code == 404
    assert "Not found" in response.text
    assert "<nav" in response.text


def test_errors_of_htmx_requests_are_notices(client):
    response = client.get(
        f"{BASE}/jobs/unknown", headers={"HX-Request": "true"}
    )
    assert response.status_code == 404
    assert response.headers[NOTICE_HEADER] == "error"
    assert 'class="notice notice--error"' in response.text
    assert "<nav" not in response.text


def test_the_home_page_names_no_unfinished_or_developer_pages(client):
    page = client.get(f"{BASE}/").text
    for text in ("under construction", "Soon", "/dev", "Developer"):
        assert text not in page
    assert client.get(f"{BASE}/dev").status_code == 404
    assert client.get(f"{BASE}/dev/demo").status_code == 404


def test_unexpected_errors_show_an_error_id(settings, runner):
    pages = create_pages(settings, runner)
    broken = APIRouter()

    @broken.get("/broken")
    def fail():
        raise RuntimeError("something broke")

    pages.include_router(broken)
    client = TestClient(pages, raise_server_exceptions=False)
    response = client.get("/broken")
    assert response.status_code == 500
    assert "error id" in response.text
    assert "RuntimeError: something broke" in response.text


def test_requests_over_the_upload_limit_are_refused(client):
    headers = {"X-CSRF-Token": client.csrf, "HX-Request": "true"}
    response = client.post(
        f"{BASE}{PATH}",
        files={"file": ("big.bin", bytes(UPLOAD_LIMIT + 1))},
        data={"seconds": "1"},
        headers=headers,
    )
    assert response.status_code == 413
    assert "larger than Text Lab accepts" in response.text


def test_bodies_without_a_size_are_counted(client):
    headers = {
        "X-CSRF-Token": client.csrf,
        "Content-Type": "multipart/form-data; boundary=x",
    }
    start = (
        b'--x\r\nContent-Disposition: form-data; name="file"; '
        b'filename="big.bin"\r\n\r\n'
    )
    chunk = bytes(1024 * 1024)

    def body():
        # A generator has no size, so it is sent in chunks.
        yield start
        for _ in range(UPLOAD_LIMIT // len(chunk) + 2):
            yield chunk

    response = client.post(f"{BASE}{PATH}", content=body(), headers=headers)
    assert response.status_code == 413


@pytest.mark.parametrize(
    ("size", "text"),
    [(0, "0 bytes"), (999, "999 bytes"), (48_200_000, "48.2 MB"),
     (10**10, "10 GB"), (1_000, "1 kB")],
)  # fmt: skip
def test_file_sizes_read_well(size, text):
    assert file_size(size) == text


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(0, "0 s"), (45, "45 s"), (125, "2 min 05 s"), (3720, "1 h 02 min")],
)
def test_durations_read_well(seconds, text):
    assert duration(seconds) == text


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(30, "under 1 min"), (14 * 60 + 59, "14 min"), (7500, "2 h 05 min")],
)
def test_the_time_left_reads_well(seconds, text):
    assert time_left(seconds) == text


def test_model_output_is_rendered_as_safe_markdown():
    html = markdown(
        "# Notes\n\n<script>alert(1)</script> **bold** "
        "![a picture](https://example.org/a.png) "
        "[bad](javascript:alert(1)) [good](https://example.org)\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n"
    )
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "<strong>bold</strong>" in html
    assert "<img" not in html
    assert 'href="javascript' not in html
    assert (
        '<a href="https://example.org" target="_blank" '
        'rel="noopener noreferrer">good</a>'
    ) in html
    assert "<table>" in html
