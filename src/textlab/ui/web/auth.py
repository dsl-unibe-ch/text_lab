"""Login with the session password, the session cookie and CSRF tokens.

Each Open OnDemand session has its own password (``TOKEN``), which the
session's card on the dashboard holds: its *Connect* button posts it to
:func:`login`, so nobody else can use the session even though the server
listens on the node's network. A successful login starts a session: a
random id in a signed cookie (``HttpOnly``, ``Secure``, ``SameSite=Strict``
and limited to the session's path), and a CSRF token that every request
changing something must send back, because all Open OnDemand apps share
one origin.

Routes use the dependencies :data:`LoggedIn` (a valid session) and
:data:`Checked` (a valid session and its CSRF token). Routers made with
:func:`protected_router` require a session for every route.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hmac
import secrets
import threading
from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse, Response
from itsdangerous import BadSignature, URLSafeSerializer

from textlab.ui.web.render import renderer, root
from textlab.ui.web.settings import WebSettings

#: Name of the session cookie.
COOKIE = "textlab_session"

#: Header and form field that carry the CSRF token.
CSRF_HEADER = "X-CSRF-Token"
CSRF_FIELD = "csrf_token"

#: Seconds a wrong password waits before the answer, to slow down guessing.
FAILED_LOGIN_DELAY = 1.0


@dataclasses.dataclass(frozen=True)
class Session:
    """A logged-in browser.

    Attributes:
        id: Random id, kept signed in the cookie.
        csrf_token: Token the browser sends back with every change.
    """

    id: str
    csrf_token: str = dataclasses.field(repr=False)


class LoginRequiredError(Exception):
    """The request has no valid session cookie."""


class CsrfError(Exception):
    """A request that changes something lacks the session's CSRF token."""


class Auth:
    """The password check and the sessions of one app."""

    def __init__(self, settings: WebSettings):
        """Set up an app's sessions.

        Args:
            settings: The app's settings: the password and the signing key.
        """
        self._password = settings.token.encode()
        self._serializer = URLSafeSerializer(
            settings.secret_key, salt="textlab-session"
        )
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def password_matches(self, password: str) -> bool:
        """Return whether ``password`` is the session password.

        Args:
            password: The password a login sent.

        Returns:
            The result of a constant-time comparison.
        """
        return hmac.compare_digest(password.encode(), self._password)

    def start(self) -> tuple[Session, str]:
        """Start a session.

        Returns:
            The session and the signed value of its cookie.
        """
        session = Session(secrets.token_urlsafe(32), secrets.token_urlsafe(32))
        with self._lock:
            self._sessions[session.id] = session
        return session, self._serializer.dumps(session.id)

    def find(self, cookie: str | None) -> Session | None:
        """Return the session a cookie belongs to.

        Args:
            cookie: The cookie's value, or ``None``.

        Returns:
            The session, or ``None`` if the cookie is missing, forged or
            belongs to a session that ended.
        """
        if not cookie:
            return None
        try:
            session_id = self._serializer.loads(cookie)
        except BadSignature:
            return None
        with self._lock:
            return self._sessions.get(session_id)

    def end(self, session: Session) -> None:
        """End a session; its cookie no longer works.

        Args:
            session: The session.
        """
        with self._lock:
            self._sessions.pop(session.id, None)


def current_session(request: Request) -> Session:
    """Return the request's session; the dependency behind :data:`LoggedIn`.

    Args:
        request: The current request. The session is also stored in
            ``request.state.session`` for the templates.

    Returns:
        The session.

    Raises:
        LoginRequiredError: If the request has no valid session cookie.
    """
    session = request.app.state.auth.find(request.cookies.get(COOKIE))
    if session is None:
        raise LoginRequiredError
    request.state.session = session
    return session


async def checked_session(
    request: Request, session: Annotated[Session, Depends(current_session)]
) -> Session:
    """Return the session of a request that sends its CSRF token.

    The token comes in the ``X-CSRF-Token`` header, which htmx adds to
    every request, or in the ``csrf_token`` field of a plain form. The
    header is checked before any body is read, so an upload without it is
    refused before it is received.

    Args:
        request: The current request.
        session: The request's session.

    Returns:
        The session.

    Raises:
        CsrfError: If the token is missing or wrong.
    """
    sent = request.headers.get(CSRF_HEADER)
    content_type = request.headers.get("content-type", "")
    if sent is None and content_type.startswith(
        "application/x-www-form-urlencoded"
    ):
        sent = (await request.form()).get(CSRF_FIELD)
    if not isinstance(sent, str) or not hmac.compare_digest(
        sent.encode(), session.csrf_token.encode()
    ):
        raise CsrfError
    return session


#: A route parameter that requires a valid session.
LoggedIn = Annotated[Session, Depends(current_session)]

#: A route parameter that requires a valid session and its CSRF token.
Checked = Annotated[Session, Depends(checked_session)]


def protected_router(**kwargs) -> APIRouter:
    """Return a router whose routes all require a valid session.

    Args:
        **kwargs: Passed on to :class:`fastapi.APIRouter`, such as
            ``prefix``.

    Returns:
        The router.
    """
    return APIRouter(dependencies=[Depends(current_session)], **kwargs)


def safe_next(request: Request, target: str | None) -> str:
    """Return ``target`` if it is a path of this app to return to.

    Args:
        request: The current request.
        target: The path to go to after logging in.

    Returns:
        ``target``, or ``""`` if it leads elsewhere, back to the login, or
        is not a plain path.
    """
    base = root(request)
    if (
        not target
        or not target.startswith(f"{base}/")
        or target.startswith(f"{base}/login")
        or "//" in target
        or "\\" in target
        or not target.isprintable()
    ):
        return ""
    return target


def origin_allowed(request: Request) -> bool:
    """Return whether a login comes from a page of the portal itself.

    Browsers name the page a form was sent from in the ``Origin`` header.
    Requests without it (not from a browser) pass; the password is checked
    anyway.

    Args:
        request: The login request.

    Returns:
        ``False`` if the request names another origin than the portal.
    """
    portal = request.app.state.settings.public_host
    origin = request.headers.get("origin")
    return not portal or origin is None or origin == f"https://{portal}"


def login_redirect(request: Request) -> Response:
    """Answer a request without a valid session: send it to the login.

    Args:
        request: The request.

    Returns:
        For htmx requests, status 401 with ``HX-Redirect`` (htmx then
        loads the login page); otherwise a redirect to the login page,
        which returns to the requested page afterwards.
    """
    login = f"{root(request)}/login"
    if request.headers.get("HX-Request") == "true":
        return Response(status_code=401, headers={"HX-Redirect": login})
    if request.method == "GET" and safe_next(request, request.url.path):
        login += f"?next={quote(request.url.path)}"
    return RedirectResponse(login, status_code=303)


router = APIRouter()


@router.get("/login")
def login_page(
    request: Request,
    failed: bool = False,
    next_path: Annotated[str, Query(alias="next")] = "",
) -> Response:
    """Show the login page, or go on if the browser is logged in.

    Args:
        request: The current request.
        failed: Whether the last password was wrong.
        next_path: Path to go to after logging in (``?next=``).

    Returns:
        The login page, or a redirect.
    """
    target = safe_next(request, next_path)
    if request.app.state.auth.find(request.cookies.get(COOKIE)):
        return RedirectResponse(target or f"{root(request)}/", status_code=303)
    return renderer(request).response(
        request, "auth/login.html", failed=failed, next_path=target
    )


@router.post("/login")
async def login(request: Request) -> Response:
    """Check the posted password and start a session.

    The *Connect* button on the session's card posts the password here, as
    does the login page.

    Args:
        request: The login request, a form with ``password`` and,
            optionally, ``next``.

    Returns:
        A redirect to the app (with the session cookie) or back to the
        login page.

    Raises:
        HTTPException: If the form comes from another origin (403).
    """
    if not origin_allowed(request):
        raise HTTPException(
            status_code=403,
            detail="Logins are accepted only from the Open OnDemand portal.",
        )
    form = await request.form(max_files=0, max_fields=5)
    password = form.get("password")
    target = safe_next(request, str(form.get("next") or ""))
    auth: Auth = request.app.state.auth
    if not isinstance(password, str) or not auth.password_matches(password):
        await asyncio.sleep(FAILED_LOGIN_DELAY)
        again = f"{root(request)}/login?failed=true"
        if target:
            again += f"&next={quote(target)}"
        return RedirectResponse(again, status_code=303)

    _, cookie = auth.start()
    response = RedirectResponse(target or f"{root(request)}/", status_code=303)
    response.set_cookie(
        COOKIE,
        cookie,
        httponly=True,
        secure=True,
        samesite="strict",
        path=f"{root(request)}/",
    )
    return response


@router.post("/logout")
def logout(request: Request, session: Checked) -> Response:
    """End the session and show that the browser is signed out.

    Args:
        request: The current request.
        session: The session to end.

    Returns:
        The signed-out page, which removes the cookie.
    """
    request.app.state.auth.end(session)
    request.state.session = None
    response = renderer(request).response(request, "auth/signed_out.html")
    response.delete_cookie(
        COOKIE,
        path=f"{root(request)}/",
        secure=True,
        httponly=True,
        samesite="strict",
    )
    return response
