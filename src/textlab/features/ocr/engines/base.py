"""The interface every manual OCR engine provides, and shared steps.

An engine reads one PDF or image and returns its text per page
(:class:`EngineOutput`). Most engines read page images
(:class:`PageEngine`): PDFs are rendered with ``pdftoppm`` first. Engines
that read PDFs themselves subclass :class:`Engine` directly.
"""

from __future__ import annotations

import abc
import dataclasses
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from textlab.common.progress import Progress, ProgressCallback, no_progress


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


@dataclasses.dataclass
class PageText:
    """The text an engine read on one page.

    Attributes:
        page: The page number, from 1.
        text: The page's text.
        raw: The engine's own result for the page, in JSON types or
            convertible by :func:`.payloads.make_json_serializable`.
    """

    page: int
    text: str
    raw: Any = None


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
        languages: Document languages the user can choose, as label and
            engine code; empty when the engine has no language choice.
        default_language: The label preselected in ``languages``.
        modes: Kinds of content the user can choose to extract; empty when
            there is no choice.
    """

    name: ClassVar[str]
    languages: ClassVar[Mapping[str, str]] = {}
    default_language: ClassVar[str | None] = None
    modes: ClassVar[Sequence[str]] = ()

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
    ) -> EngineOutput:
        """Read one PDF or image.

        Args:
            input_path: The file.
            work_dir: A folder for intermediate files; the caller removes
                it.
            options: The run's settings.
            on_progress: Receives progress updates.

        Returns:
            The text per page.

        Raises:
            EngineError: If the engine fails.
        """


class PageEngine(Engine):
    """An engine that reads page images."""

    def recognize(
        self,
        input_path: Path,
        work_dir: Path,
        options: EngineOptions,
        on_progress: ProgressCallback = no_progress,
    ) -> EngineOutput:
        """Read one PDF or image, page by page (see :meth:`Engine.recognize`).

        Args:
            input_path: The file.
            work_dir: A folder for the page images.
            options: The run's settings.
            on_progress: Receives an update after each page.

        Returns:
            The text per page.
        """
        images = page_images(input_path, work_dir / "images")
        return self.recognize_pages(images, options, on_progress)

    @abc.abstractmethod
    def recognize_pages(
        self,
        images: list[Path],
        options: EngineOptions,
        on_progress: ProgressCallback,
    ) -> EngineOutput:
        """Read page images.

        Args:
            images: One image per page, in page order.
            options: The run's settings.
            on_progress: Receives an update after each page.

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
    command: Sequence[str], env: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess:
    """Run an engine process to completion, capturing its output as text."""
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=dict(env) if env is not None else dict(os.environ),
    )
