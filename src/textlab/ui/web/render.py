"""Rendering templates: pages, fragments, icons and links.

Every link is built from the base path the app is mounted under
(:func:`root`), never from the host and scheme of the request, which behind
the Open OnDemand proxy are the compute node's. Static files get a version
in their link (``?v=`` and a hash of the file), so browsers can cache them
for good and still load a changed file at once.
"""

from __future__ import annotations

import functools
import hashlib
import re
import time
import tomllib
from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from markupsafe import Markup, escape

import textlab
from textlab.ui.web.settings import WebSettings

HERE = Path(__file__).resolve().parent
TEMPLATES_DIR = HERE / "templates"
STATIC_DIR = HERE / "static"
ICONS_DIR = STATIC_DIR / "vendor" / "lucide"

#: Response header that marks an error notice for ``static/js/app.js``.
NOTICE_HEADER = "X-Textlab-Notice"


def root(request: Request) -> str:
    """Return the base path the app is mounted under, without a final slash.

    Args:
        request: The current request.

    Returns:
        For example ``/node/gnode21.hpc.unibe.ch/31234``; ``""`` when the
        app is served at the root.
    """
    return request.scope.get("root_path", "").rstrip("/")


def is_htmx(request: Request) -> bool:
    """Return whether htmx sent the request (rather than a page load).

    Args:
        request: The current request.

    Returns:
        ``True`` for requests with the ``HX-Request`` header.
    """
    return request.headers.get("HX-Request") == "true"


def file_size(size: int | float | None) -> str:
    """Return a file size for people to read, in decimal units.

    Args:
        size: The size in bytes.

    Returns:
        For example ``"512 bytes"``, ``"48.2 MB"`` or ``"10 GB"``.
    """
    if size is None:
        return ""
    if size < 1000:
        return f"{int(size)} bytes"
    for unit in ("kB", "MB", "GB", "TB"):
        size /= 1000
        if size < 1000 or unit == "TB":
            return f"{size:.1f} {unit}".replace(".0 ", " ")
    raise AssertionError("unreachable")


def duration(seconds: int | float | None) -> str:
    """Return a duration for people to read.

    Args:
        seconds: The duration.

    Returns:
        For example ``"45 s"``, ``"2 min 05 s"`` or ``"1 h 02 min"``.
    """
    if seconds is None:
        return ""
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {seconds:02d} s"
    return f"{seconds} s"


def media_time(seconds: int | float | None) -> str:
    """Return a time in a recording, as players show it.

    Args:
        seconds: Seconds from the start.

    Returns:
        For example ``"00:14"``, ``"12:40"`` or ``"1:02:05"``; ``""`` for
        ``None``.
    """
    if seconds is None:
        return ""
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def time_left(seconds: int | float | None) -> str:
    """Return the time left in a session, to the minute.

    Args:
        seconds: The seconds left.

    Returns:
        For example ``"2 h 05 min"``, ``"14 min"`` or ``"under 1 min"``.
    """
    if seconds is None:
        return ""
    minutes = int(max(0, seconds) // 60)
    if minutes < 1:
        return "under 1 min"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min" if hours else f"{minutes} min"


def clock_time(timestamp: float | None) -> str:
    """Return the local time of day of a timestamp, such as ``"14:05"``.

    Args:
        timestamp: Seconds since the epoch.

    Returns:
        The time, or ``""`` for ``None``.
    """
    if timestamp is None:
        return ""
    return time.strftime("%H:%M", time.localtime(timestamp))


@functools.cache
def _icon_markup(name: str) -> str:
    """Return a vendored Lucide icon as one line of inline SVG."""
    text = (ICONS_DIR / f"{name}.svg").read_text(encoding="utf-8")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r'\s*class="[^"]*"', "", text, count=1)
    text = " ".join(text.split()).replace(" />", "/>").replace("> <", "><")
    return text.replace(
        "<svg", '<svg class="icon" aria-hidden="true" focusable="false"', 1
    )


def icon(name: str, label: str = "", size: str = "") -> Markup:
    """Return a Lucide icon from ``static/vendor/lucide/`` as inline SVG.

    Inline icons take the colour of the text around them and need no
    request of their own.

    Args:
        name: The icon's file name without ``.svg``, such as ``"upload"``.
        label: Text for screen readers when the icon stands alone; without
            it, the icon is hidden from them.
        size: ``"sm"`` or ``"lg"`` for a smaller or larger icon.

    Returns:
        The ``<svg>`` element.

    Raises:
        FileNotFoundError: If the icon is not vendored.
    """
    markup = _icon_markup(name)
    if size:
        markup = markup.replace('class="icon"', f'class="icon icon--{size}"')
    if label:
        markup = markup.replace(
            'aria-hidden="true"', f'role="img" aria-label="{escape(label)}"'
        )
    return Markup(markup)


@functools.cache
def textlab_version() -> str:
    """Return Text Lab's version, from ``pyproject.toml`` next to ``src/``.

    Returns:
        For example ``"3.0.1"``; ``""`` if the file is not there.
    """
    pyproject = Path(textlab.__file__).resolve().parents[2] / "pyproject.toml"
    try:
        with pyproject.open("rb") as handle:
            return str(tomllib.load(handle)["project"]["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return ""


class Renderer:
    """Renders the templates in ``templates/`` for one app.

    Every template gets, besides its own context:

    - ``root``: the base path, for links (``{{ root }}/transcription``);
    - ``static(path)``: the versioned link of a static file;
    - ``icon(name)``: an inline icon;
    - ``csrf_token``: the session's CSRF token, or ``""`` before login;
    - ``current_path``: the request's path below the base path;
    - ``session_jobs``: the session's jobs, for the job indicator;
    - ``session_left``: the seconds until the session ends, or ``None``;
    - ``settings``, ``version``, the navigation (``tools``,
      ``tool_groups``, ``page_path``, ``page_label``, set by the app) and
      the filters ``file_size``, ``duration``, ``media_time``,
      ``time_left`` and ``clock_time``.
    """

    def __init__(self, settings: WebSettings):
        """Load the templates.

        Args:
            settings: The app's settings, available to every template.
        """
        self.environment = Environment(
            loader=FileSystemLoader(TEMPLATES_DIR),
            autoescape=True,
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.environment.globals.update(
            icon=icon, settings=settings, version=textlab_version()
        )
        self.environment.filters.update(
            file_size=file_size,
            duration=duration,
            media_time=media_time,
            clock_time=clock_time,
            time_left=time_left,
        )

    @functools.cache  # noqa: B019 - one renderer per app; files are fixed.
    def static_version(self, path: str) -> str:
        """Return a short hash of a static file's content.

        Args:
            path: The file's path below ``static/``.

        Returns:
            Ten hexadecimal digits, or ``"0"`` if the file is missing.
        """
        try:
            content = (STATIC_DIR / path).read_bytes()
        except OSError:
            return "0"
        return hashlib.sha256(content).hexdigest()[:10]

    def context(self, request: Request) -> dict[str, Any]:
        """Return the values every template gets for ``request``.

        Args:
            request: The current request.

        Returns:
            The shared context; see the class docstring.
        """
        base = root(request)
        session = getattr(request.state, "session", None)
        path = request.url.path
        current = path[len(base) :] if path.startswith(base) else path

        def static(file: str) -> str:
            return f"{base}/static/{file}?v={self.static_version(file)}"

        runner = getattr(request.app.state, "runner", None)
        end = request.app.state.settings.session_end
        return {
            "request": request,
            "root": base,
            "static": static,
            "csrf_token": session.csrf_token if session else "",
            "current_path": current or "/",
            "session_jobs": runner.jobs() if runner is not None else [],
            "session_left": None if end is None else end - time.time(),
        }

    def render(self, request: Request, name: str, **context: Any) -> str:
        """Render a template to a string.

        Args:
            request: The current request.
            name: The template's path below ``templates/``.
            **context: The template's own values.

        Returns:
            The HTML.
        """
        template = self.environment.get_template(name)
        return template.render({**self.context(request), **context})

    def response(
        self,
        request: Request,
        name: str,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        **context: Any,
    ) -> HTMLResponse:
        """Render a template into a response.

        Args:
            request: The current request.
            name: The template's path below ``templates/``.
            status_code: The response's status.
            headers: Extra response headers.
            **context: The template's own values.

        Returns:
            The HTML response, a whole page or a fragment for htmx.
        """
        return HTMLResponse(
            self.render(request, name, **context),
            status_code=status_code,
            headers=headers,
        )


def renderer(request: Request) -> Renderer:
    """Return the renderer of the app that handles ``request``.

    Args:
        request: The current request.

    Returns:
        The app's :class:`Renderer`.
    """
    return request.app.state.renderer
