from urllib.parse import quote

import pytest

from textlab.ui.web import auth, jobs, pages
from textlab.ui.web.tests.conftest import (
    BASE,
    OWNER,
    PASSWORD,
    PORTAL,
    csrf_token,
    make_client,
)


def test_pages_send_a_browser_without_login_to_the_login(anonymous):
    response = anonymous.get(f"{BASE}/transcription")
    assert response.status_code == 303
    assert response.headers["location"] == (
        f"{BASE}/login?next={quote(f'{BASE}/transcription')}"
    )


def test_htmx_requests_without_login_get_a_redirect_header(anonymous):
    response = anonymous.get(f"{BASE}/", headers={"HX-Request": "true"})
    assert response.status_code == 401
    assert response.headers["HX-Redirect"] == f"{BASE}/login"


def test_the_login_page_points_to_the_connect_button(anonymous):
    response = anonymous.get(f"{BASE}/login")
    assert response.status_code == 200
    assert "Connect to Text Lab" in response.text
    assert f"https://{PORTAL}/pun/sys/dashboard/batch_connect/sessions" in (
        response.text
    )


def test_the_right_password_starts_a_protected_session(anonymous):
    response = anonymous.post(f"{BASE}/login", data={"password": PASSWORD})
    assert response.status_code == 303
    assert response.headers["location"] == f"{BASE}/"
    cookie = response.headers["set-cookie"]
    flags = cookie.lower()
    assert cookie.startswith(f"{auth.COOKIE}=")
    assert "httponly" in flags
    assert "secure" in flags
    assert "samesite=strict" in flags
    assert f"path={BASE.lower()}/" in flags
    assert anonymous.get(f"{BASE}/").status_code == 200


def test_a_wrong_password_returns_to_the_login(anonymous):
    response = anonymous.post(f"{BASE}/login", data={"password": "wrong"})
    assert response.status_code == 303
    assert response.headers["location"] == f"{BASE}/login?failed=true"
    assert "set-cookie" not in response.headers
    page = anonymous.get(response.headers["location"])
    assert "not this session's password" in page.text


def test_a_non_ascii_password_is_refused_not_an_error(anonymous):
    response = anonymous.post(f"{BASE}/login", data={"password": "Grüße"})
    assert response.status_code == 303
    assert "failed=true" in response.headers["location"]


def test_login_returns_to_the_page_asked_for(anonymous):
    target = f"{BASE}/transcription"
    response = anonymous.post(
        f"{BASE}/login", data={"password": PASSWORD, "next": target}
    )
    assert response.headers["location"] == target


@pytest.mark.parametrize(
    "target",
    [
        "https://evil.example.org/",
        "//evil.example.org/",
        "/node/other/1/",
        f"{BASE}//evil.example.org",
        f"{BASE}/login",
        f"{BASE}\\evil",
    ],
)
def test_login_never_redirects_elsewhere(anonymous, target):
    response = anonymous.post(
        f"{BASE}/login", data={"password": PASSWORD, "next": target}
    )
    assert response.headers["location"] == f"{BASE}/"


def test_the_connect_button_of_the_portal_can_log_in(anonymous):
    response = anonymous.post(
        f"{BASE}/login",
        data={"password": PASSWORD},
        headers={"Origin": f"https://{PORTAL}"},
    )
    assert response.status_code == 303
    assert "set-cookie" in response.headers


def test_logins_from_other_sites_are_refused(anonymous):
    response = anonymous.post(
        f"{BASE}/login",
        data={"password": PASSWORD},
        headers={"Origin": "https://evil.example.org"},
    )
    assert response.status_code == 403
    assert "set-cookie" not in response.headers


def test_a_login_with_a_file_is_refused(anonymous):
    response = anonymous.post(
        f"{BASE}/login",
        data={"password": PASSWORD},
        files={"file": ("a.txt", b"x")},
    )
    assert response.status_code == 400


def test_a_forged_cookie_is_ignored(anonymous):
    forged = {"Cookie": f"{auth.COOKIE}=forged"}
    assert anonymous.get(f"{BASE}/", headers=forged).status_code == 303


def test_the_login_page_moves_on_when_logged_in(client):
    response = client.get(f"{BASE}/login")
    assert response.status_code == 303
    assert response.headers["location"] == f"{BASE}/"


def test_changes_without_the_csrf_token_are_refused(client):
    url = f"{BASE}/jobs/unknown/cancel"
    assert client.post(url).status_code == 403
    wrong = {"X-CSRF-Token": "wrong"}
    assert client.post(url, headers=wrong).status_code == 403
    right = {"X-CSRF-Token": client.csrf}
    assert client.post(url, headers=right).status_code == 404


def test_signing_out_ends_the_session(client):
    assert client.post(f"{BASE}/logout").status_code == 403
    response = client.post(
        f"{BASE}/logout", data={auth.CSRF_FIELD: client.csrf}
    )
    assert response.status_code == 200
    assert "Signed out" in response.text
    assert "max-age=0" in response.headers["set-cookie"].lower()
    assert client.get(f"{BASE}/").status_code == 303


def test_sessions_are_separate(app):
    first, second = make_client(app), make_client(app)
    for client in (first, second):
        client.post(f"{BASE}/login", data={"password": PASSWORD})
    first_token = csrf_token(first.get(f"{BASE}/").text)
    second_token = csrf_token(second.get(f"{BASE}/").text)
    assert first_token != second_token
    response = second.post(
        f"{BASE}/jobs/unknown/cancel", headers={"X-CSRF-Token": first_token}
    )
    assert response.status_code == 403


def test_requests_of_another_user_are_refused(anonymous):
    other = {"X-Forwarded-User": "mallory"}
    assert anonymous.get(f"{BASE}/login", headers=other).status_code == 403
    response = anonymous.post(
        f"{BASE}/login", data={"password": PASSWORD}, headers=other
    )
    assert response.status_code == 403
    owner = {"X-Forwarded-User": OWNER}
    assert anonymous.get(f"{BASE}/login", headers=owner).status_code == 200


def test_every_route_but_the_login_needs_a_session(settings, app):
    anonymous = make_client(app)
    routers = [auth.router, jobs.router, *pages.routers()]
    checked = 0
    for router in routers:
        for route in router.routes:
            if route.path == "/login":
                continue
            path = route.path.replace("{job_id}", "x").replace("{name}", "x")
            for method in route.methods:
                response = anonymous.request(method, f"{BASE}{path}")
                assert response.status_code in (303, 401), (method, path)
                checked += 1
    assert checked > 10
