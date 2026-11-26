"""Placeholders for the tools whose pages are not in the web UI yet."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import Request
from fastapi.responses import HTMLResponse

from textlab.ui.web.auth import protected_router
from textlab.ui.web.pages import TOOLS, Tool
from textlab.ui.web.render import renderer

router = protected_router()


def _placeholder(tool: Tool) -> Callable[[Request], HTMLResponse]:
    """Return the route that shows ``tool``'s placeholder."""

    def show(request: Request) -> HTMLResponse:
        return renderer(request).response(
            request, "pages/placeholder.html", page_id=tool.id, tool=tool
        )

    show.__name__ = f"placeholder_{tool.id}"
    show.__doc__ = f"Say that the {tool.label} page is not built yet."
    return show


for _tool in TOOLS:
    if not _tool.ready:
        router.add_api_route(
            _tool.path,
            _placeholder(_tool),
            methods=["GET"],
            response_class=HTMLResponse,
        )
