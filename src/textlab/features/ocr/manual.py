"""Manual engine selection: one chosen engine on a document or a batch.

The engines (:mod:`.engines`) return plain text per page, without the
layout, tables and figures of the automatic pipeline. Each run works in a
job folder in the workspace (area ``ocr``), removed when the run ends;
results are returned in memory. Interfaces reach these names through
:mod:`.service`.

Every run writes the same files, per document: ``page_0001.txt`` and
``page_0001.json`` per page, ``page_0001_table_1.csv`` per HTML table on a
page (GLM-OCR writes them), and the whole text and result as
``<name>.txt`` and ``<name>.json``. OlmOCR, which reads a document at once,
writes ``<name>.txt`` and its own result record, ``<name>.jsonl``.

A run stops when its ``cancel`` event is set: between files and pages, and
an engine's own process at once, by raising
:class:`~textlab.common.progress.CancelledError`.
"""

from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
from pathlib import Path
from typing import BinaryIO

from textlab.common.progress import (
    CancelledError,
    Progress,
    ProgressCallback,
    check_cancelled,
    no_progress,
)
from textlab.common.storage import get_workspace, remove_tree
from textlab.common.upload_safety import (
    safe_upload_name,
    unique_output_directory,
)
from textlab.features.ocr.archives import extract_inputs, zip_folder
from textlab.features.ocr.doc_ir import extract_html_table
from textlab.features.ocr.engines.base import (
    Engine,
    EngineError,
    EngineOptions,
    EngineOutput,
    PageText,
    Preview,
    TextLine,
    contains_html_table,
    html_table,
    text_parts,
)
from textlab.features.ocr.engines.easy_ocr import EasyOcr
from textlab.features.ocr.engines.glm_ocr import GlmOcr
from textlab.features.ocr.engines.olm_ocr import OlmOcr
from textlab.features.ocr.engines.paddle_ocr import PaddleOcr
from textlab.features.ocr.engines.payloads import make_json_serializable

__all__ = [
    "ENGINES",
    "EngineBatch",
    "EngineBatchFile",
    "EngineError",
    "EngineOptions",
    "EngineRun",
    "PageText",
    "Preview",
    "TextLine",
    "contains_html_table",
    "html_table",
    "recognize_batch_with_engine",
    "recognize_with_engine",
    "text_parts",
]

#: The manual engines, by the name users choose them by.
ENGINES: dict[str, Engine] = {
    engine.name: engine
    for engine in (EasyOcr(), PaddleOcr(), OlmOcr(), GlmOcr())
}

#: Workspace area for job folders, shared with the automatic pipeline.
WORKSPACE_AREA = "ocr"


@dataclasses.dataclass
class EngineRun:
    """The outcome of :func:`recognize_with_engine`.

    Attributes:
        engine: The engine's name.
        text: The text of all pages.
        text_name: File name for the text download.
        json: The engine's result, as JSON (one line for OlmOCR).
        json_name: File name for the JSON download.
        previews: Preview images, one per page.
        zip_bytes: Every output file in one ZIP.
        pages: The text per page, with the image each was read from and
            the outlines of its lines, for engines that find lines.
        options: The settings it was read with.
    """

    engine: str
    text: str
    text_name: str
    json: str
    json_name: str
    previews: list[Preview]
    zip_bytes: bytes
    pages: list[PageText] = dataclasses.field(default_factory=list)
    options: EngineOptions = EngineOptions()

    @property
    def stem(self) -> str:
        """The base name of the downloads, after the document."""
        return self.text_name.removesuffix(".txt")


@dataclasses.dataclass(frozen=True)
class EngineBatchFile:
    """One file of a batch, as :func:`recognize_batch_with_engine` read it.

    Attributes:
        name: Its path in the archive.
        pages: How many pages the engine returned (one for OlmOCR, which
            reads a document at once).
        characters: How many characters of text it read.
        seconds: How long it took.
    """

    name: str
    pages: int
    characters: int
    seconds: float


@dataclasses.dataclass(frozen=True)
class EngineBatch:
    """The outcome of :func:`recognize_batch_with_engine`.

    Attributes:
        zip_bytes: The result ZIP, a folder of outputs per file.
        files: Each file's overview, in the archive's order.
        seconds: How long the batch took.
    """

    zip_bytes: bytes
    files: tuple[EngineBatchFile, ...]
    seconds: float


def recognize_with_engine(
    name: str,
    data: bytes,
    engine: str,
    options: EngineOptions,
    *,
    on_progress: ProgressCallback = no_progress,
    cancel: threading.Event | None = None,
) -> EngineRun:
    """Read one PDF or image with a chosen engine.

    Args:
        name: The uploaded file's name; only its last part is used.
        data: The file's content.
        engine: A key of :data:`ENGINES`.
        options: The engine's settings.
        on_progress: Receives progress updates.
        cancel: Stops the run when set.

    Returns:
        The text, the engine's result and the downloads.

    Raises:
        EngineError: If the engine fails.
        CancelledError: If ``cancel`` was set.
    """
    adapter = ENGINES[engine]
    upload_name = safe_upload_name(name)
    stem = Path(upload_name).stem or "document"
    with get_workspace().temp_dir(WORKSPACE_AREA, prefix="manual-") as job:
        input_path = job / "input" / upload_name
        input_path.parent.mkdir()
        input_path.write_bytes(data)
        check_cancelled(cancel)
        adapter.prepare(options, on_progress)
        check_cancelled(cancel)
        output = adapter.recognize(
            input_path, job / "work", options, on_progress, cancel
        )
        results_dir = job / "results"
        text_name, json_name = write_outputs(output, results_dir, stem)
        return EngineRun(
            engine=adapter.name,
            text=output.text,
            text_name=text_name,
            json=(results_dir / json_name).read_text(encoding="utf-8"),
            json_name=json_name,
            previews=output.previews,
            zip_bytes=zip_folder(results_dir),
            pages=output.pages,
            options=options,
        )


def recognize_batch_with_engine(
    archive: BinaryIO | str | os.PathLike,
    engine: str,
    options: EngineOptions,
    *,
    on_progress: ProgressCallback = no_progress,
    cancel: threading.Event | None = None,
) -> EngineBatch:
    """Read every PDF and image in a ZIP archive with a chosen engine.

    No previews are drawn. The result ZIP mirrors the archive's folders,
    with one folder of outputs per file.

    Args:
        archive: The ZIP archive, as a file object or path.
        engine: A key of :data:`ENGINES`.
        options: The engine's settings.
        on_progress: Receives an update per file.
        cancel: Stops the batch when set.

    Returns:
        The result ZIP and an overview of the files.

    Raises:
        ValueError: If the archive holds no supported file.
        EngineError: If the engine fails on a file; the message starts
            with the file's path in the archive.
        CancelledError: If ``cancel`` was set.
    """
    adapter = ENGINES[engine]
    options = dataclasses.replace(options, previews=False)
    started = time.monotonic()
    with get_workspace().temp_dir(WORKSPACE_AREA, prefix="manual-") as job:
        input_dir = job / "input"
        results_dir = job / "results"
        input_dir.mkdir()
        results_dir.mkdir()
        files = extract_inputs(archive, input_dir)
        if not files:
            raise ValueError("No valid documents or images found in the ZIP.")
        check_cancelled(cancel)
        adapter.prepare(options, on_progress)

        used_output_dirs: set[str] = set()
        overview = []
        for index, file_path in enumerate(files):
            check_cancelled(cancel)
            file_started = time.monotonic()
            rel_path = file_path.relative_to(input_dir)
            on_progress(
                Progress(
                    f"Processing ({index + 1}/{len(files)}): {rel_path}",
                    index / len(files),
                )
            )
            output_rel_dir = unique_output_directory(
                rel_path, used_output_dirs
            )
            work_dir = job / "work" / str(index)
            try:
                output = adapter.recognize(
                    file_path, work_dir, options, cancel=cancel
                )
            except CancelledError:
                raise
            except Exception as error:
                raise EngineError(
                    f"{rel_path}: {error}",
                    details=getattr(error, "details", ""),
                ) from error
            write_outputs(
                output, results_dir / output_rel_dir, output_rel_dir.name
            )
            # Page images are only needed while a file is read.
            remove_tree(work_dir)
            overview.append(
                EngineBatchFile(
                    name=rel_path.as_posix(),
                    pages=len(output.pages),
                    characters=len(output.text),
                    seconds=time.monotonic() - file_started,
                )
            )

        on_progress(Progress("Batch OCR complete! Zipping results...", 1.0))
        return EngineBatch(
            zip_bytes=zip_folder(results_dir),
            files=tuple(overview),
            seconds=time.monotonic() - started,
        )


def write_outputs(
    output: EngineOutput, folder: Path, stem: str
) -> tuple[str, str]:
    """Write an engine's output for one document.

    Args:
        output: What the engine read.
        folder: Where to write; created if needed.
        stem: Base name of the whole-document files.

    Returns:
        The names of the text file and of the result file.
    """
    folder.mkdir(parents=True, exist_ok=True)
    text_name = f"{stem}.txt"
    (folder / text_name).write_text(output.text, encoding="utf-8")
    if output.record is not None:
        json_name = f"{stem}.jsonl"
        (folder / json_name).write_text(output.record + "\n", encoding="utf-8")
        return text_name, json_name

    pages = []
    for page in output.pages:
        item = make_json_serializable(
            {"page": page.page, "text": page.text, "raw": page.raw}
        )
        pages.append(item)
        page_stem = f"page_{page.page:04d}"
        (folder / f"{page_stem}.txt").write_text(page.text, encoding="utf-8")
        (folder / f"{page_stem}.json").write_text(
            json.dumps(item, ensure_ascii=False), encoding="utf-8"
        )
        for number, table in enumerate(page_tables(page.text), start=1):
            table.to_csv(
                folder / f"{page_stem}_table_{number}.csv", index=False
            )
    json_name = f"{stem}.json"
    (folder / json_name).write_text(
        json.dumps(pages, ensure_ascii=False), encoding="utf-8"
    )
    return text_name, json_name


def page_tables(text: str) -> list:
    """Return the HTML tables in a page's text that can be read as tables.

    Args:
        text: The page's text.

    Returns:
        A ``pandas.DataFrame`` per table, in order; tables that cannot be
        parsed are left out.
    """
    tables = []
    for kind, value in text_parts(text):
        if kind == "table":
            table = extract_html_table(value)
            if table is not None:
                tables.append(table)
    return tables
