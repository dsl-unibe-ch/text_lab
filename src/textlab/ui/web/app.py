"""The web UI's app factory: routes, middleware, static files, error pages.

:func:`create_app` builds the ASGI app that :mod:`~textlab.ui.web.server`
serves. Its pages form a FastAPI app mounted under the session's base path
(``/node/<host>/<port>``); every link is built from that path, never from
the host and scheme of the request, which behind the Open OnDemand proxy
are the compute node's. Around it, as plain ASGI middleware so that event
streams and downloads pass unchanged:

- :class:`SecurityHeaders` adds the Content Security Policy and the other
  security headers to every response;
- :class:`ProxyUserCheck` refuses requests that the proxy marks as another
  user's;
- :class:`~textlab.ui.web.uploads.BodyLimit` refuses bodies over the upload
  limit.

Errors are shown as a page, or for htmx requests as a notice that
``static/js/app.js`` puts into the page (marked with the header
:data:`~textlab.ui.web.render.NOTICE_HEADER`).
"""

from __future__ import annotations

import contextlib
import logging
import os
import secrets

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.applications import Starlette
from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException
from starlette.routing import Mount, Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from textlab.ui.web import auth, jobs, pages
from textlab.ui.web.render import (
    NOTICE_HEADER,
    STATIC_DIR,
    Renderer,
    is_htmx,
    renderer,
)
from textlab.ui.web.settings import WebSettings
from textlab.ui.web.uploads import BodyLimit, BodyTooLargeError, UploadError

LOGGER = logging.getLogger(__name__)

#: Headers added to every response that does not set them itself.
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data: blob:; media-src 'self' blob:; "
        "connect-src 'self'; object-src 'none'; form-action 'self'; "
        "frame-ancestors 'none'; base-uri 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
    # Pages and results hold the user's data: keep them out of the
    # browser's cache. Static files set their own header.
    "Cache-Control": "no-store",
}

#: Titles of error pages, by status.
ERROR_TITLES = {
    400: "The request could not be read",
    403: "Not allowed",
    404: "Not found",
    405: "Not allowed",
    409: "Not possible now",
    413: "Upload too large",
    422: "Check your input",
    500: "Something went wrong",
}


class SecurityHeaders:
    """ASGI middleware that adds :data:`SECURITY_HEADERS` to responses."""

    def __init__(self, app: ASGIApp):
        """Wrap ``app``.

        Args:
            app: The ASGI app.
        """
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        """Add the headers when the response starts."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS.items():
                    headers.setdefault(name, value)
            await send(message)

        await self.app(scope, receive, send_with_headers)


class ProxyUserCheck:
    """ASGI middleware that refuses requests marked as another user's.

    The Open OnDemand proxy names the logged-in user in
    ``X-Forwarded-User``. A request through the proxy from another user is
    refused before anything else, even with the right password. Requests
    without the header (not through the proxy) pass to the login as usual:
    the header can be forged on a direct connection, so it cannot replace
    the password.
    """

    def __init__(self, app: ASGIApp, owner: str):
        """Wrap ``app``.

        Args:
            app: The ASGI app.
            owner: The session's user; empty turns the check off.
        """
        self.app = app
        self.owner = owner

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        """Refuse the request if it names another user."""
        if scope["type"] == "http" and self.owner:
            user = Headers(scope=scope).get("x-forwarded-user")
            if user is not None and user != self.owner:
                LOGGER.warning("Refused a request of another user")
                response = PlainTextResponse(
                    "This Text Lab session belongs to another user.",
                    status_code=403,
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


class StaticAssets(StaticFiles):
    """Static files, cached for good when their link names a version.

    :meth:`Renderer.context`'s ``static()`` adds ``?v=`` and a hash of the
    file, so a changed file gets a new link. Vendored files carry their
    release in their path (``vendor/ibm-plex-sans-5.3.0/``), so they are
    cached for good as well; style sheets link fonts that way.
    """

    def file_response(
        self,
        full_path: os.PathLike | str,
        stat_result: os.stat_result,
        scope: Scope,
        status_code: int = 200,
    ) -> Response:
        """Return the file with a ``Cache-Control`` header."""
        response = super().file_response(
            full_path, stat_result, scope, status_code
        )
        versioned = b"v=" in scope.get("query_string", b"") or str(
            full_path
        ).startswith(str(STATIC_DIR / "vendor"))
        response.headers["Cache-Control"] = (
            "public, max-age=31536000, immutable" if versioned else "no-cache"
        )
        return response


def error_response(
    request: Request,
    status_code: int,
    message: str,
    *,
    title: str = "",
    details: str = "",
    headers: dict[str, str] | None = None,
) -> Response:
    """Show an error: a notice for htmx requests, otherwise a page.

    Args:
        request: The request that failed.
        status_code: The response's status.
        message: What went wrong and what to do, for the user.
        title: The heading; by default one for the status.
        details: Technical details for a problem report, shown folded.
        headers: Extra response headers.

    Returns:
        The response.
    """
    title = title or ERROR_TITLES.get(status_code, "Error")
    context = {
        "status": status_code,
        "title": title,
        "message": message,
        "details": details,
    }
    if getattr(request.state, "session", None) is None:
        # Errors before a route ran (an unknown address): show the layout
        # all the same if the browser is logged in.
        session = request.app.state.auth.find(request.cookies.get(auth.COOKIE))
        if session is not None:
            request.state.session = session
    if is_htmx(request):
        return renderer(request).response(
            request,
            "components/error_notice.html",
            status_code=status_code,
            headers={**(headers or {}), NOTICE_HEADER: "error"},
            **context,
        )
    return renderer(request).response(
        request,
        "error.html",
        status_code=status_code,
        headers=headers,
        **context,
    )


def install_error_handlers(app: FastAPI) -> None:
    """Show the app's errors as notices or error pages.

    Args:
        app: The app.
    """

    @app.exception_handler(auth.LoginRequiredError)
    async def login_required(request: Request, error: Exception) -> Response:
        return auth.login_redirect(request)

    @app.exception_handler(auth.CsrfError)
    async def csrf_failed(request: Request, error: Exception) -> Response:
        return error_response(
            request,
            403,
            "The request lacked this page's security token. Reload the page "
            "and try again.",
        )

    @app.exception_handler(UploadError)
    async def upload_failed(request: Request, error: Exception) -> Response:
        status = 413 if isinstance(error, BodyTooLargeError) else 422
        return error_response(request, status, str(error))

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, error: Exception) -> Response:
        assert isinstance(error, HTTPException)
        message = error.detail if isinstance(error.detail, str) else ""
        if error.status_code == 404 and not message:
            message = "There is no page at this address."
        elif error.status_code == 405:
            message = "This address does not accept this kind of request."
        return error_response(
            request, error.status_code, message, headers=error.headers
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request: Request, error: Exception) -> Response:
        assert isinstance(error, RequestValidationError)
        problems = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'][1:])}: {item['msg']}"
            for item in error.errors()
        )
        return error_response(
            request,
            422,
            "Some of the values sent are not valid.",
            details=problems,
        )

    @app.exception_handler(Exception)
    async def unexpected(request: Request, error: Exception) -> Response:
        error_id = secrets.token_hex(4)
        LOGGER.error(
            "Error %s in %s %s",
            error_id,
            request.method,
            request.url.path,
            exc_info=error,
        )
        return error_response(
            request,
            500,
            "Text Lab could not complete this request. If it happens again, "
            "report it with the session's textlab.log and this error id: "
            f"{error_id}.",
            details=f"{type(error).__name__}: {error}",
        )


def create_pages(settings: WebSettings, runner: jobs.JobRunner) -> FastAPI:
    """Return the FastAPI app with the pages, served under the base path.

    Args:
        settings: The settings.
        runner: The job runner.

    Returns:
        The app. Its state holds ``settings``, ``auth``, ``runner`` and
        ``renderer``.
    """
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings = settings
    app.state.auth = auth.Auth(settings)
    app.state.runner = runner
    app.state.renderer = Renderer(settings)
    # The navigation, in every page's layout.
    app.state.renderer.environment.globals.update(
        tools=pages.TOOLS,
        tool_groups=pages.tool_groups(),
        page_path=pages.page_path,
        page_label=pages.page_label,
    )
    app.mount("/static", StaticAssets(directory=STATIC_DIR), name="static")
    app.include_router(auth.router)
    app.include_router(jobs.router)
    for router in pages.routers():
        app.include_router(router)
    install_error_handlers(app)
    return app


def create_app(
    settings: WebSettings, *, runner: jobs.JobRunner | None = None
) -> ASGIApp:
    """Build the web UI.

    Args:
        settings: The settings.
        runner: The job runner; a new one by default. Tests pass one with a
            fake GPU release and their own workspace.

    Returns:
        The ASGI app to serve.
    """
    runner = runner or jobs.JobRunner()
    pages_app = create_pages(settings, runner)
    base = settings.base_path

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        LOGGER.info(
            "Text Lab web UI ready under %s/ (process %d)",
            base,
            os.getpid(),
        )
        try:
            yield
        finally:
            runner.close()

    async def to_base(request: Request) -> Response:
        return RedirectResponse(f"{base}/", status_code=307)

    async def outside_base(request: Request, error: Exception) -> Response:
        return PlainTextResponse(
            f"Text Lab is served at {base}/.", status_code=404
        )

    if base:
        routes = [Route(base, to_base), Mount(base, app=pages_app)]
    else:
        routes = [Mount("", app=pages_app)]
    app = Starlette(
        routes=routes,
        lifespan=lifespan,
        exception_handlers={404: outside_base},
    )
    return SecurityHeaders(
        ProxyUserCheck(BodyLimit(app, settings.upload_limit), settings.owner)
    )
