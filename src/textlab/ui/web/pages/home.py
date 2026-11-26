"""The Home page: what Text Lab offers and the session's recent jobs.

Until the pilot (phase 2 of the web UI plan) it says which pages exist in
the web UI so far.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import HTMLResponse

from textlab.ui.web.auth import protected_router
from textlab.ui.web.pages import TOOLS
from textlab.ui.web.render import renderer

router = protected_router()


@router.get("/")
def home(request: Request) -> HTMLResponse:
    """Show the Home page.

    Args:
        request: The current request.

    Returns:
        The page.
    """
    return renderer(request).response(
        request, "pages/home.html", page_id="home", tools=TOOLS
    )
