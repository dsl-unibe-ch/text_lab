"""The Home page: what Text Lab is, its tools and the session's jobs."""

from __future__ import annotations

import time

from fastapi import Request
from fastapi.responses import HTMLResponse

from textlab.ui.web.auth import protected_router
from textlab.ui.web.render import renderer

router = protected_router()


@router.get("/")
def home(request: Request) -> HTMLResponse:
    """Show the Home page.

    Args:
        request: The current request.

    Returns:
        The page. Its list of jobs follows the job runner live, through the
        page's event stream (``jobs.events``).
    """
    return renderer(request).response(
        request,
        "pages/home.html",
        page_id="home",
        now=time.time(),
    )
