"""The interface every manual OCR engine provides, and shared steps.

An engine reads one PDF or image and returns its text per page
(:class:`EngineOutput`). Most engines read page images
(:class:`PageEngine`): PDFs are rendered with ``pdftoppm`` first, and each
page keeps the image it was read from, with the outline of every text line
the engine found on it, if it finds lines. Engines that read PDFs
themselves subclass :class:`Engine` directly.

Every engine stops when the run's ``cancel`` event is set: page engines
between pages, engines in a process of their own at once
(:func:`run_process`), by raising
:class:`~textlab.common.progress.CancelledError`.
"""

from __future__ import annotations

import abc
import dataclasses
import os
import re
import signal
import subprocess
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from textlab.common.progress import (
    CancelledError,
    Progress,
    ProgressCallback,
    check_cancelled,
    no_progress,
)

#: Seconds between checks for cancelling while an engine process runs.
CANCEL_POLL = 0.5

#: An HTML table in an engine's text, as GLM-OCR writes them.
_HTML_TABLE = re.compile(r"<table\b.*?</table\s*>", re.IGNORECASE | re.DOTALL)


@dataclasses.dataclass(frozen=True)
class EngineOptions:
    """Settings for one run of an engine.

    Attributes:
        language: The engine's code for the document language, for engines
            with a language choice (:attr:`Engine.languages`).
        mode: The kind of content to extract, for engines with a choice of
            modes (:attr:`Engine.modes`).
        previews: Draw preview images of the result.
    """

    language: str = "en"
    mode: str = ""
    previews: bool = True


@dataclasses.dataclass(frozen=True)
class TextLine:
    """A line of text an engine found, where it found it.

    PaddleOCR finds lines; EasyOCR, reading paragraphs, finds blocks of
    lines, which are kept the same way.

    Attributes:
        text: The line's text.
        points: The corners of its outline, in pixels of the page image
            it was read from (:attr:`PageText.image`).
    """

    text: str
    points: tuple[tuple[float, float], ...]

    @classmethod
    def of(cls, text: str, points: Any) -> TextLine:
        """Return a line from its text and its corners.

        Args:
            text: The line's text.
            points: The corners as ``(x, y)`` pairs, such as an ``(n, 2)``
                array from :func:`.payloads.polygon_points`.

        Returns:
            The line.
        """
        return cls(text, tuple((float(x), float(y)) for x, y in points))


@dataclasses.dataclass
class PageText:
    """The text an engine read on one page.

    Attributes:
        page: The page number, from 1.
        text: The page's text.
        raw: The engine's own result for the page, in JSON types or
            convertible by :func:`.payloads.make_json_serializable`.
        lines: The text lines found, with their outlines, for engines that
            find lines; their texts make up ``text``.
        image: The page image the engine read, as image file data, which
            the outlines of ``lines`` belong to; ``None`` for engines that
            read whole PDFs.
    """

    page: int
    text: str
    raw: Any = None
    lines: list[TextLine] = dataclasses.field(default_factory=list)
    image: bytes | None = None


@dataclasses.dataclass
class Preview:
    """Preview images of one page, as PNG data.

    Attributes:
        image: The page, with the detected text outlined if the engine
            draws that.
        layout: The recognized text drawn where it was found, for engines
            that draw it.
    """

    image: bytes
    layout: bytes | None = None


@dataclasses.dataclass
class EngineOutput:
    """What an engine read in one document.

    Attributes:
        pages: The text per page. An engine that reads the whole document
            at once returns it as one page.
        previews: Preview images, one per page, if requested.
        record: The engine's own result as one JSON line, for engines that
            write one (OlmOCR).
    """

    pages: list[PageText]
    previews: list[Preview] = dataclasses.field(default_factory=list)
    record: str | None = None

    @property
    def text(self) -> str:
        """The text of all pages, separated by blank lines."""
        return "\n\n".join(page.text for page in self.pages)


class EngineError(RuntimeError):
    """An engine failed.

    Attributes:
        details: More information for a developer, such as the engine
            process's output.
    """

    def __init__(self, message: str, details: str = "") -> None:
        """Create the error with a message for users and details."""
        super().__init__(message)
        self.details = details


class Engine(abc.ABC):
    """A manual OCR engine.

    Attributes:
        name: The name users choose the engine by.
        description: What the engine is good at, in a sentence, for users
            choosing one.
        languages: Document languages the user can choose, as label and
            engine code; empty when the engine has no language choice.
        default_language: The label preselected in ``languages``.
        modes: Kinds of content the user can choose to extract; empty when
            there is no choice.
        markdown: Whether the engine writes its text as Markdown.
        ollama_model: The Ollama model the engine runs on, which stays
            loaded when the GPU is released for it; ``None`` for engines
            that run on their own.
    """

    name: ClassVar[str]
    description: ClassVar[str] = ""
    languages: ClassVar[Mapping[str, str]] = {}
    default_language: ClassVar[str | None] = None
    modes: ClassVar[Sequence[str]] = ()
    markdown: ClassVar[bool] = False
    ollama_model: ClassVar[str | None] = None

    def problem(self) -> str | None:
        """Return why the engine cannot run now, or ``None`` if it can.

        Engines that need nothing beyond the image keep this default.

        Returns:
            A sentence for users, such as that a server is not running.
        """
        return None

    def prepare(  # noqa: B027 (an optional hook, empty by default)
        self, options: EngineOptions, on_progress: ProgressCallback
    ) -> None:
        """Get the engine's model ready, once before a run.

        Engines without a model to prepare keep this empty default.

        Args:
            options: The run's settings.
            on_progress: Receives progress updates, such as a download.
        """

    @abc.abstractmethod
    def recognize(
        self,
        input_path: Path,
        work_dir: Path,
        options: EngineOptions,
        on_progress: ProgressCallback = no_progress,
        cancel: threading.Event | None = None,
    ) -> EngineOutput:
        """Read one PDF or image.

        Args:
            input_path: The file.
            work_dir: A folder for intermediate files; the caller removes
                it.
            options: The run's settings.
            on_progress: Receives progress updates.
            cancel: Stops the engine when set.

        Returns:
            The text per page.

        Raises:
            EngineError: If the engine fails.
            CancelledError: If ``cancel`` was set.
        """


class PageEngine(Engine):
    """An engine that reads page images."""

    def recognize(
        self,
        input_path: Path,
        work_dir: Path,
        options: EngineOptions,
        on_progress: ProgressCallback = no_progress,
        cancel: threading.Event | None = None,
    ) -> EngineOutput:
        """Read one PDF or image, page by page (see :meth:`Engine.recognize`).

        Each page keeps the image it was read from, unless the engine read
        another one (PaddleOCR straightens pages first).

        Args:
            input_path: The file.
            work_dir: A folder for the page images.
            options: The run's settings.
            on_progress: Receives an update after each page.
            cancel: Stops the engine when set.

        Returns:
            The text per page.
        """
        images = page_images(input_path, work_dir / "images")
        check_cancelled(cancel)
        output = self.recognize_pages(images, options, on_progress, cancel)
        for page in output.pages:
            if page.image is None and 0 < page.page <= len(images):
                page.image = images[page.page - 1].read_bytes()
        return output

    @abc.abstractmethod
    def recognize_pages(
        self,
        images: list[Path],
        options: EngineOptions,
        on_progress: ProgressCallback,
        cancel: threading.Event | None = None,
    ) -> EngineOutput:
        """Read page images.

        Args:
            images: One image per page, in page order.
            options: The run's settings.
            on_progress: Receives an update after each page.
            cancel: Stops the engine when set, between pages at least.

        Returns:
            The text per page.
        """


def page_images(input_path: Path, out_dir: Path) -> list[Path]:
    """Return a file's pages as images.

    Args:
        input_path: A PDF, or an image, which is returned as it is.
        out_dir: Where to write the page images of a PDF.

    Returns:
        The page images, in page order.

    Raises:
        EngineError: If a PDF gives no pages.
    """
    if input_path.suffix.lower() != ".pdf":
        return [input_path]
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["pdftoppm", "-png", str(input_path), str(out_dir / "page")],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    # pdftoppm pads page numbers to the same width, so names sort in order.
    images = sorted(out_dir.glob("page-*.png"))
    if not images:
        raise EngineError("No images generated from PDF.")
    return images


def page_done(
    on_progress: ProgressCallback, engine: str, done: int, count: int
) -> None:
    """Report that an engine finished a page."""
    on_progress(
        Progress(f"Running {engine}... page {done}/{count}", done / count)
    )


def process_output(result: subprocess.CompletedProcess) -> str:
    """Return the end of a finished process's output, for error details."""
    return (
        f"stdout:\n{result.stdout[-4000:]}\n\nstderr:\n{result.stderr[-4000:]}"
    )


def contains_html_table(text: str) -> bool:
    """Return True if ``text`` holds an HTML table, as GLM-OCR writes."""
    return "<table>" in text or '<table class="' in text


def text_parts(text: str) -> list[tuple[str, str]]:
    """Split an engine's text into text and the HTML tables in it.

    Args:
        text: An engine's text output.

    Returns:
        ``("text", ...)`` and ``("table", ...)`` parts, in order, the
        tables as the engine wrote them (not sanitized); parts with only
        white space are left out.
    """
    parts = []
    position = 0
    for match in _HTML_TABLE.finditer(text):
        parts.append(("text", text[position : match.start()]))
        parts.append(("table", match.group(0)))
        position = match.end()
    parts.append(("text", text[position:]))
    return [(kind, value.strip()) for kind, value in parts if value.strip()]


def html_table(text: str) -> Any:
    """Return the first HTML table in ``text`` as a data frame.

    Args:
        text: An engine's text output.

    Returns:
        A ``pandas.DataFrame``, or ``None`` if there is no table or it
        cannot be parsed.
    """
    if not contains_html_table(text):
        return None
    from textlab.features.ocr.doc_ir import extract_html_table

    return extract_html_table(text)


def run_process(
    command: Sequence[str],
    env: Mapping[str, str] | None = None,
    *,
    cancel: threading.Event | None = None,
) -> subprocess.CompletedProcess:
    """Run an engine process to completion, capturing its output as text.

    The process gets a process group of its own, so that cancelling stops
    the programs it started too (OlmOCR starts a vLLM server).

    Args:
        command: The program and its arguments.
        env: Its environment; by default this process's.
        cancel: Kills the process and its group when set.

    Returns:
        The finished process, with its exit code and output.

    Raises:
        CancelledError: If ``cancel`` was set before the process ended.
    """
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=dict(env) if env is not None else dict(os.environ),
        start_new_session=True,
    )
    while True:
        try:
            # Waiting again after a timeout loses no output.
            stdout, stderr = process.communicate(timeout=CANCEL_POLL)
            break
        except subprocess.TimeoutExpired:
            if cancel is not None and cancel.is_set():
                _kill_group(process)
                raise CancelledError("The work was cancelled.") from None
    return subprocess.CompletedProcess(
        process.args, process.returncode, stdout, stderr
    )


def _kill_group(process: subprocess.Popen) -> None:
    """Kill a process and every process in its group, and reap it."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.communicate()
