"""Recognizing documents and images: the OCR feature's API.

Interfaces call this module. :func:`recognize_document` runs the automatic
pipeline (:mod:`.pipeline`) on one uploaded PDF or image and prepares every
download; :func:`recognize_batch` does the same for a ZIP archive, keeping
the recognition model loaded for the whole batch, and optionally reads a
batch of filled-in questionnaires against the form they share
(:mod:`textlab.features.survey.service`). Manual engine selection
(:mod:`.manual`: EasyOCR, PaddleOCR, OlmOCR, GLM-OCR) is reached through
:func:`recognize_with_engine` and :func:`recognize_batch_with_engine`.

Uploads, page images and intermediate files go to a job folder in the
workspace (area ``ocr``), which is removed when the run ends; results are
returned in memory. :func:`save_document` and :func:`save_engine_run` write
a result's downloads, page images, tables and figures to a folder an
interface serves them from, and return what it needs to show the result.

Every recognition stops when its ``cancel`` event is set: between pages
and files, and a recognition worker or engine process at once, by raising
:class:`~textlab.common.progress.CancelledError`.

The few names interfaces need from other modules (Tesseract languages, the
layout preview, the manual engines) are re-exported here, so interfaces
import only this module and the result model :mod:`.doc_ir`.
"""

from __future__ import annotations

import base64
import dataclasses
import io
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

from textlab.common.html_safety import sanitize_table_html
from textlab.common.progress import (
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
from textlab.features.ocr import doc_ir, vision_enrich
from textlab.features.ocr.archives import (
    INPUT_EXTENSIONS,
    extract_inputs,
    zip_folder,
)
from textlab.features.ocr.layout_preview import (
    LAYOUT_TYPE_COLORS,
    render_layout_preview,
)
from textlab.features.ocr.manual import (
    ENGINES,
    EngineBatch,
    EngineBatchFile,
    EngineError,
    EngineOptions,
    EngineRun,
    contains_html_table,
    html_table,
    page_tables,
    recognize_batch_with_engine,
    recognize_with_engine,
    text_parts,
)
from textlab.features.ocr.pipeline import document_summary, process_document
from textlab.features.ocr.searchable_pdf import (
    DEFAULT_TESSERACT_LANG,
    TESSERACT_LANGUAGES,
)
from textlab.features.ocr.vl_session import VLWorkerSession
from textlab.features.survey.service import (
    QuestionnaireBatch,
    SameLayoutTemplate,
)

__all__ = [
    "INPUT_EXTENSIONS",
    "BatchFile",
    "BatchResult",
    "DocumentDownloads",
    "DocumentResult",
    "OcrOptions",
    "document_summary",
    "process_document",
    "recognize_batch",
    "recognize_document",
    # Results for interfaces
    "SavedDocument",
    "SavedFile",
    "SavedPage",
    "SavedRegion",
    "SavedTable",
    "save_document",
    # Re-exported for interfaces
    "DEFAULT_TESSERACT_LANG",
    "LAYOUT_TYPE_COLORS",
    "TESSERACT_LANGUAGES",
    "render_layout_preview",
    # Manual engine selection
    "ENGINES",
    "EngineBatch",
    "EngineBatchFile",
    "EngineError",
    "EngineOptions",
    "EngineRun",
    "SavedEnginePage",
    "SavedEngineRun",
    "SavedLine",
    "SavedPart",
    "contains_html_table",
    "html_table",
    "recognize_batch_with_engine",
    "recognize_with_engine",
    "save_engine_run",
]

#: Workspace area for job folders.
WORKSPACE_AREA = "ocr"


@dataclasses.dataclass(frozen=True)
class OcrOptions:
    """How to recognize documents.

    Attributes:
        native_fast_lane: Read PDF pages with a usable text layer directly;
            when False, every page goes through the vision model.
        describe_images: Add a generated description to each figure.
        extract_survey: Extract question-level survey responses
            (experimental; hidden in the app).
        same_template: With ``extract_survey`` in a batch, all files use
            the same questionnaire layout.
        survey_batch_mode: In a batch of one filled-in questionnaire, learn
            the blank form from the batch and read every file against it.
        searchable_pdf: Build a PDF with an invisible text layer.
        ocr_lang: Tesseract language for word positions in the searchable
            PDF, or ``"auto"`` to detect it per page.
    """

    native_fast_lane: bool = True
    describe_images: bool = False
    extract_survey: bool = False
    same_template: bool = False
    survey_batch_mode: bool = False
    searchable_pdf: bool = False
    ocr_lang: str = DEFAULT_TESSERACT_LANG


@dataclasses.dataclass
class DocumentDownloads:
    """Every download offered for one recognized document.

    Attributes:
        stem: Base name of the download files.
        markdown_zip: Markdown with an ``assets/`` folder of figures.
        text: Plain text.
        docx: A Word document, or ``None`` without python-docx.
        searchable_pdf: The searchable PDF, if it was requested and built.
        json: The full result with regions, boxes and confidences.
        tables_zip: One CSV per table, or ``None`` without tables.
        responses_csv: Extracted form responses, if there are any.
        full: All of the above in one ZIP.
    """

    stem: str
    markdown_zip: bytes | None
    text: bytes
    docx: bytes | None
    searchable_pdf: bytes | None
    json: bytes
    tables_zip: bytes | None
    responses_csv: bytes | None
    full: bytes | None

    @classmethod
    def build(cls, document: doc_ir.Document, stem: str) -> DocumentDownloads:
        """Build the downloads of a document.

        Args:
            document: The recognized document.
            stem: Base name of the download files.

        Returns:
            The downloads.
        """
        return cls(
            stem=stem,
            markdown_zip=doc_ir.build_markdown_zip(document, stem),
            text=doc_ir.to_text(document).encode("utf-8"),
            docx=doc_ir.build_docx(document, stem),
            searchable_pdf=document.searchable_pdf,
            json=doc_ir.to_json(document).encode("utf-8"),
            tables_zip=doc_ir.build_tables_csv_zip(document),
            responses_csv=doc_ir.build_form_responses_csv(document),
            full=doc_ir.build_full_bundle(document, stem),
        )

    def refresh_responses(self, document: doc_ir.Document) -> None:
        """Rebuild the downloads that contain form responses.

        Call this after a reviewer corrected responses in ``document``.

        Args:
            document: The corrected document.
        """
        self.json = doc_ir.to_json(document).encode("utf-8")
        self.responses_csv = doc_ir.build_form_responses_csv(document)
        self.full = doc_ir.build_full_bundle(document, self.stem)


@dataclasses.dataclass
class DocumentResult:
    """The outcome of :func:`recognize_document`.

    Attributes:
        document: The recognized document.
        summary: Counts for an overview (:func:`document_summary`).
        downloads: Every download for the document.
    """

    document: doc_ir.Document
    summary: dict[str, Any]
    downloads: DocumentDownloads


@dataclasses.dataclass(frozen=True)
class BatchFile:
    """One recognized file of a batch, for an overview.

    Attributes:
        name: Its path inside the archive.
        pages: How many pages it has.
        text_layer_pages: How many of them were read from their text layer;
            the others went through the recognition model.
        region_counts: How many regions of each type were found.
        seconds: How long it took.
    """

    name: str
    pages: int
    text_layer_pages: int
    region_counts: dict[str, int]
    seconds: float


@dataclasses.dataclass
class BatchResult:
    """The outcome of :func:`recognize_batch`.

    Attributes:
        zip_bytes: One folder of outputs per file, mirroring the archive.
        file_count: Number of files recognized.
        elapsed: Duration of the run in seconds.
        survey: The questionnaire results, in survey batch mode.
        files: Each file's overview, in order.
    """

    zip_bytes: bytes
    file_count: int
    elapsed: float
    survey: QuestionnaireBatch | None = None
    files: list[BatchFile] = dataclasses.field(default_factory=list)


def recognize_document(
    name: str,
    data: bytes,
    options: OcrOptions,
    *,
    on_progress: ProgressCallback = no_progress,
    cancel: threading.Event | None = None,
) -> DocumentResult:
    """Recognize one PDF or image.

    Args:
        name: The uploaded file's name; only its last part is used.
        data: The file's content.
        options: How to recognize it.
        on_progress: Receives progress updates.
        cancel: Stops the recognition when set.

    Returns:
        The document, its summary and its downloads.

    Raises:
        CancelledError: If ``cancel`` was set.
    """
    upload_name = safe_upload_name(name)
    with get_workspace().temp_dir(WORKSPACE_AREA, prefix="job-") as job_dir:
        input_dir = job_dir / "input"
        input_dir.mkdir()
        input_path = input_dir / upload_name
        input_path.write_bytes(data)
        document = process_document(
            input_path,
            job_dir / "workspace",
            native_fast_lane=options.native_fast_lane,
            progress=_progress_adapter(on_progress),
            source_name=upload_name,
            describe_images=options.describe_images,
            extract_survey=options.extract_survey,
            searchable_pdf=options.searchable_pdf,
            ocr_lang=options.ocr_lang,
            cancel=cancel,
        )
    stem = Path(upload_name).stem or "document"
    return DocumentResult(
        document=document,
        summary=document_summary(document),
        downloads=DocumentDownloads.build(document, stem),
    )


def recognize_batch(
    archive: BinaryIO | str | os.PathLike,
    options: OcrOptions,
    *,
    on_progress: ProgressCallback = no_progress,
    cancel: threading.Event | None = None,
) -> BatchResult:
    """Recognize every PDF and image in a ZIP archive.

    The result ZIP mirrors the archive's folders, with a ``document.md``,
    ``document.json``, ``tables/`` and ``assets/`` per file, and a
    ``models_used.txt`` for citation. In survey batch mode it also has a
    ``survey/`` folder with one row per respondent.

    Args:
        archive: The ZIP archive, as a file object or path.
        options: How to recognize the files.
        on_progress: Receives progress updates naming the file and step.
        cancel: Stops the batch when set.

    Returns:
        The result ZIP, an overview of each file, and the questionnaire
        results in survey batch mode.

    Raises:
        ValueError: If the archive holds no supported file.
        CancelledError: If ``cancel`` was set.
    """
    vision_client = (
        vision_enrich.OllamaVisionClient()
        if options.describe_images or options.extract_survey
        else None
    )
    same_layout_template = (
        SameLayoutTemplate()
        if options.extract_survey and options.same_template
        else None
    )
    with get_workspace().temp_dir(WORKSPACE_AREA, prefix="batch-") as job_dir:
        # Keep the recognition model loaded for the whole batch; the worker
        # stops before its job folder is removed.
        vl_session = VLWorkerSession()
        try:
            return _recognize_batch(
                archive,
                options,
                job_dir,
                on_progress,
                vision_client=vision_client,
                same_layout_template=same_layout_template,
                vl_session=vl_session,
                cancel=cancel,
            )
        finally:
            vl_session.close()
            if vision_client is not None:
                # Not unloaded: it expires after its keep-alive, and the
                # next OCR worker frees the GPU itself.
                vision_client.close()


def _recognize_batch(
    archive: BinaryIO | str | os.PathLike,
    options: OcrOptions,
    job_dir: Path,
    on_progress: ProgressCallback,
    *,
    vision_client: Any,
    same_layout_template: Any,
    vl_session: VLWorkerSession,
    cancel: threading.Event | None = None,
) -> BatchResult:
    """Run :func:`recognize_batch` inside its job folder."""
    input_dir = job_dir / "input"
    workspace_dir = job_dir / "workspace"
    results_dir = workspace_dir / "results"
    input_dir.mkdir()
    results_dir.mkdir(parents=True)

    files = extract_inputs(archive, input_dir)
    if not files:
        raise ValueError("No valid documents or images found in the ZIP.")
    n_files = len(files)

    # The consensus template needs all questionnaires up front.
    survey = None
    if options.survey_batch_mode:
        survey = QuestionnaireBatch.learn(
            files, on_progress=on_progress, vl_session=vl_session
        )

    batch_provenance = []
    batch_started = time.monotonic()
    file_times = []
    used_output_dirs: set[str] = set()
    overview: list[BatchFile] = []
    for index, file_path in enumerate(files):
        check_cancelled(cancel)
        rel_path = file_path.relative_to(input_dir)

        def file_progress(fraction, text, _index=index, _rel=rel_path):
            done = _index + max(0.0, min(1.0, fraction))
            on_progress(
                Progress(
                    f"File {_index + 1} of {n_files} · {_rel} — {text}",
                    min(1.0, done / n_files),
                )
            )

        file_progress(0.0, "starting...")
        file_started = time.monotonic()

        output_rel_dir = unique_output_directory(rel_path, used_output_dirs)
        file_output_dir = results_dir / output_rel_dir
        file_output_dir.mkdir(parents=True, exist_ok=True)
        file_workspace = workspace_dir / "tmp" / f"job_{index}"
        document = process_document(
            file_path,
            file_workspace,
            native_fast_lane=options.native_fast_lane,
            progress=file_progress,
            source_name=file_path.name,
            describe_images=options.describe_images,
            extract_survey=options.extract_survey,
            searchable_pdf=options.searchable_pdf,
            ocr_lang=options.ocr_lang,
            vision_client=vision_client,
            same_layout_template=same_layout_template,
            vl_session=vl_session,
            cancel=cancel,
        )

        skip_tables = None
        if survey is not None:
            skip_tables = survey.read(
                file_path, document, file_output_dir, output_rel_dir.as_posix()
            )

        doc_ir.write_document_outputs(
            document,
            file_output_dir,
            "document",
            provenance=False,
            skip_tables=skip_tables,
            form_responses=survey is None,
        )
        batch_provenance.append(doc_ir.model_provenance(document))
        if survey is not None:
            survey.keep_document(output_rel_dir.as_posix(), document)

        # Page images are only needed while a file is recognized.
        remove_tree(file_workspace)
        file_times.append(time.monotonic() - file_started)
        summary = document_summary(document)
        overview.append(
            BatchFile(
                name=rel_path.as_posix(),
                pages=summary["n_pages"],
                text_layer_pages=sum(
                    1 for page in document.pages if page.source == "native"
                ),
                region_counts=dict(summary["region_counts"]),
                seconds=file_times[-1],
            )
        )
        print(
            f"[OCR] file {index + 1}/{n_files} {rel_path}: "
            f"{file_times[-1]:.1f}s",
            flush=True,
        )

    if survey is not None:
        on_progress(Progress("Collecting questionnaire responses..."))
        survey.write_outputs(results_dir / "survey")

    elapsed = time.monotonic() - batch_started
    print(
        f"[OCR] {n_files} file(s) in {elapsed:.1f}s "
        f"(mean {elapsed / n_files:.1f}s/file; "
        f"first {file_times[0]:.1f}s, rest mean "
        f"{sum(file_times[1:]) / max(1, len(file_times) - 1):.1f}s)",
        flush=True,
    )
    on_progress(
        Progress(
            f"{n_files} file(s) parsed in {elapsed / 60:.1f} min — zipping "
            "results...",
            1.0,
        )
    )

    # One summary at the root, the union over files: a batch can mix lanes.
    merged_provenance = doc_ir.provenance_to_text(
        doc_ir.merge_provenance(batch_provenance)
    )
    if merged_provenance:
        (results_dir / "models_used.txt").write_text(
            merged_provenance + "\n", encoding="utf-8"
        )
    return BatchResult(
        zip_bytes=zip_folder(results_dir),
        file_count=n_files,
        elapsed=elapsed,
        survey=survey,
        files=overview,
    )


# ---------------------------------------------------------------------------
# Results for interfaces
# ---------------------------------------------------------------------------

#: What the citation summary's keys mean (:func:`doc_ir.model_provenance`).
PROVENANCE_LABELS = {
    "text_recognition": "Text recognition",
    "figure_descriptions": "Figure descriptions",
    "text_layer": "Word positions in the searchable PDF",
}

#: The media type of Word files.
_DOCX = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)


@dataclasses.dataclass(frozen=True)
class SavedFile:
    """A file :func:`save_document` wrote.

    Attributes:
        name: Its file name, unique in the folder.
        label: What it is, for a download list.
        media_type: Its media type.
        path: Where it is.
        listed: Whether it is a download of its own; page images, figures
            and single tables are shown or linked in the result instead.
    """

    name: str
    label: str
    media_type: str
    path: Path
    listed: bool = True


@dataclasses.dataclass(frozen=True)
class SavedRegion:
    """One region of a page, as an interface shows it.

    Attributes:
        id: The region's id, unique in the document.
        type: Its type, one of the ``doc_ir`` types (``"title"``, ...).
        box: ``(x, y, width, height)`` in the pixels of the page image, or
            ``None`` if it has no position.
        text: Its text; for a table the plain text, for a formula LaTeX.
        html: A table's HTML, sanitized (``common.html_safety``).
        latex: A formula's LaTeX.
        image: The file name of its picture (figures and seals).
        description: A generated description of a figure.
        visible_text: The text a vision model read inside a figure.
        described_by: The model that wrote the description.
        mark: For a check box, ``"checked"``, ``"unchecked"`` or
            ``"uncertain"``; empty for other regions.
    """

    id: str
    type: str
    box: tuple[float, float, float, float] | None
    text: str
    html: str = ""
    latex: str = ""
    image: str | None = None
    description: str = ""
    visible_text: str = ""
    described_by: str = ""
    mark: str = ""


@dataclasses.dataclass(frozen=True)
class SavedPage:
    """One page of a document, as an interface shows it.

    Attributes:
        number: The page number, from 1.
        text_layer: Whether the page was read from its text layer rather
            than by the recognition model.
        image: The file name of the page's image, if there is one.
        width: The image's width in pixels, the space of the region boxes.
        height: The image's height in pixels.
        regions: The page's regions, in reading order.
    """

    number: int
    text_layer: bool
    image: str | None
    width: int
    height: int
    regions: tuple[SavedRegion, ...]


@dataclasses.dataclass(frozen=True)
class SavedTable:
    """One table of a document.

    Attributes:
        region: The table's region.
        page: The page it is on.
        csv: The file name of its CSV, if it could be read as a table.
    """

    region: SavedRegion
    page: int
    csv: str | None


@dataclasses.dataclass(frozen=True)
class SavedDocument:
    """A recognized document as saved by :func:`save_document`.

    Attributes:
        name: The document's name.
        pages: Its pages.
        tables: Its tables, in reading order.
        figures: Its figures and seals, with the page each is on.
        files: Every file written, downloads first.
        summary: The counts of :func:`document_summary`.
        models: The models that made the result, as ``(what, which)``
            pairs, for citation.
    """

    name: str
    pages: tuple[SavedPage, ...]
    tables: tuple[SavedTable, ...]
    figures: tuple[tuple[int, SavedRegion], ...]
    files: tuple[SavedFile, ...]
    summary: dict[str, Any]
    models: tuple[tuple[str, str], ...]

    @property
    def text_layer_pages(self) -> int:
        """How many pages were read from their text layer."""
        return sum(1 for page in self.pages if page.text_layer)


def save_document(result: DocumentResult, folder: Path) -> SavedDocument:
    """Write a result's downloads, page images and figures into a folder.

    The downloads are the ones of :class:`DocumentDownloads`; besides them
    each page image, each figure and each table's CSV gets a file of its
    own, named after the document, so an interface can show and link them.
    The result's images and bytes are not needed afterwards.

    Args:
        result: The result of :func:`recognize_document`.
        folder: An existing folder, such as a job's outputs.

    Returns:
        What was written, and the pages, tables and figures to show.
    """
    document, downloads = result.document, result.downloads
    stem = downloads.stem
    files: list[SavedFile] = []

    def write(name: str, data: bytes, label: str, media: str, listed=True):
        path = folder / name
        path.write_bytes(data)
        files.append(SavedFile(name, label, media, path, listed))
        return name

    for suffix, data, label, media in (
        ("_bundle.zip", downloads.full, "Everything", "application/zip"),
        (".txt", downloads.text, "Plain text", "text/plain; charset=utf-8"),
        (".docx", downloads.docx, "Word", _DOCX),
        (
            "_markdown.zip",
            downloads.markdown_zip,
            "Markdown with figures",
            "application/zip",
        ),
        (
            "_searchable.pdf",
            downloads.searchable_pdf,
            "Searchable PDF",
            "application/pdf",
        ),
        (".json", downloads.json, "JSON with boxes", "application/json"),
        (
            "_tables.zip",
            downloads.tables_zip,
            "Tables, CSV",
            "application/zip",
        ),
    ):
        if data:
            write(f"{stem}{suffix}", data, label, media)

    csv_names = {}
    for entry in doc_ir.tables_to_dataframes(document):
        name = f"{stem}_table-{entry['region_id']}.csv"
        csv = entry["dataframe"].to_csv(index=False).encode("utf-8")
        csv_names[entry["region_id"]] = write(
            name, csv, "Table, CSV", "text/csv; charset=utf-8", listed=False
        )

    pages, tables, figures = [], [], []
    for page in document.pages:
        image, width, height = None, 0, 0
        picture = _decode(page.image_b64)
        if picture:
            width, height = _image_size(picture)
            image = write(
                f"{stem}_page-{page.page_number:04d}.png",
                picture,
                f"Page {page.page_number}",
                "image/png",
                listed=False,
            )
        regions = []
        for region in page.ordered_regions():
            saved = _saved_region(region, stem, write)
            regions.append(saved)
            if region.type == doc_ir.TABLE:
                csv = csv_names.get(region.id)
                tables.append(SavedTable(saved, page.page_number, csv))
            elif saved.image is not None:
                figures.append((page.page_number, saved))
        pages.append(
            SavedPage(
                number=page.page_number,
                text_layer=page.source == "native",
                image=image,
                width=width,
                height=height,
                regions=tuple(regions),
            )
        )

    models = []
    for key, value in doc_ir.model_provenance(document).items():
        if value:
            items = value if isinstance(value, list) else [value]
            joined = ", ".join(str(item) for item in items)
            models.append((PROVENANCE_LABELS.get(key, key), joined))
    return SavedDocument(
        name=document.source_name,
        pages=tuple(pages),
        tables=tuple(tables),
        figures=tuple(figures),
        files=tuple(files),
        summary=result.summary,
        models=tuple(models),
    )


@dataclasses.dataclass(frozen=True)
class SavedLine:
    """A text line a manual engine found, as an interface shows it.

    Attributes:
        id: Its id, unique in the run, such as ``"p1-l3"``.
        text: Its text.
        points: Its outline as SVG polygon points (``"x,y x,y ..."``), in
            pixels of the page image.
    """

    id: str
    text: str
    points: str


@dataclasses.dataclass(frozen=True)
class SavedPart:
    """A part of a page's text from a manual engine: text, or a table.

    Attributes:
        kind: ``"text"`` or ``"table"``.
        text: The text; for a table, the HTML the engine wrote.
        html: For a table, its sanitized HTML (table tags only), safe to
            show; empty for text.
        csv: For a table that could be read, the file name of its CSV.
    """

    kind: str
    text: str
    html: str = ""
    csv: str | None = None


@dataclasses.dataclass(frozen=True)
class SavedEnginePage:
    """One page a manual engine read, as an interface shows it.

    Attributes:
        number: The page number, from 1.
        text: The page's whole text, for copying.
        image: The file name of the page image the engine read, or
            ``None`` (OlmOCR reads whole PDFs).
        width: The image's width in pixels; 0 without an image.
        height: Its height in pixels; 0 without an image.
        lines: The text lines with their outlines, for engines that find
            lines; their texts make up ``text``.
        parts: The text split into text and tables, for engines that find
            no lines.
    """

    number: int
    text: str
    image: str | None
    width: int
    height: int
    lines: tuple[SavedLine, ...]
    parts: tuple[SavedPart, ...]


@dataclasses.dataclass(frozen=True)
class SavedEngineRun:
    """What :func:`save_engine_run` wrote, and what to show of a run.

    Attributes:
        name: The document's name.
        engine: The engine's name.
        markdown: Whether the engine writes its text as Markdown.
        pages: The pages, in order.
        files: Every file written, downloads first.
    """

    name: str
    engine: str
    markdown: bool
    pages: tuple[SavedEnginePage, ...]
    files: tuple[SavedFile, ...]

    @property
    def lines(self) -> int:
        """How many text lines were found, on all pages."""
        return sum(len(page.lines) for page in self.pages)

    @property
    def tables(self) -> int:
        """How many tables were found, on all pages."""
        return sum(
            1
            for page in self.pages
            for part in page.parts
            if part.kind == "table"
        )


def save_engine_run(run: EngineRun, name: str, folder: Path) -> SavedEngineRun:
    """Write a manual engine's downloads, page images and tables.

    The downloads are everything in one ZIP, the text and the engine's
    result; besides them each page image and each table's CSV gets a file
    of its own, named after the document, so an interface can show and
    link them. Page images an interface cannot show (TIFF, BMP) are saved
    as PNG.

    Args:
        run: The result of :func:`recognize_with_engine`.
        name: The document's name, as uploaded.
        folder: An existing folder, such as a job's outputs.

    Returns:
        What was written, and the pages to show.
    """
    stem = run.stem
    files: list[SavedFile] = []

    def write(
        file_name: str, data: bytes, label: str, media: str, listed=True
    ):
        path = folder / file_name
        path.write_bytes(data)
        files.append(SavedFile(file_name, label, media, path, listed))
        return file_name

    write(
        f"{stem}_outputs.zip", run.zip_bytes, "Everything", "application/zip"
    )
    write(
        run.text_name,
        run.text.encode("utf-8"),
        "Plain text",
        "text/plain; charset=utf-8",
    )
    record = run.json_name.endswith(".jsonl")
    write(
        run.json_name,
        run.json.encode("utf-8"),
        f"{run.engine}'s record, JSON Lines" if record else "JSON, per page",
        "application/x-ndjson" if record else "application/json",
    )

    pages = []
    for page in run.pages:
        prefix = f"{stem}_page-{page.page:04d}"
        image, width, height = None, 0, 0
        if page.image:
            picture, extension, (width, height) = _web_image(page.image)
            if picture:
                image = write(
                    f"{prefix}.{extension}",
                    picture,
                    f"Page {page.page}",
                    f"image/{'jpeg' if extension == 'jpg' else extension}",
                    listed=False,
                )
        lines = tuple(
            SavedLine(
                id=f"p{page.page}-l{index}",
                text=line.text,
                points=" ".join(f"{x:.1f},{y:.1f}" for x, y in line.points),
            )
            for index, line in enumerate(page.lines, start=1)
        )
        parts, tables = [], 0
        if not lines:
            for kind, value in text_parts(page.text):
                if kind != "table":
                    parts.append(SavedPart("text", value))
                    continue
                tables += 1
                csv = None
                found = page_tables(value)
                if found:
                    csv = write(
                        f"{prefix}_table-{tables}.csv",
                        found[0].to_csv(index=False).encode("utf-8"),
                        f"Table {tables}, page {page.page}, CSV",
                        "text/csv; charset=utf-8",
                        listed=False,
                    )
                parts.append(
                    SavedPart("table", value, sanitize_table_html(value), csv)
                )
        pages.append(
            SavedEnginePage(
                number=page.page,
                text=page.text,
                image=image,
                width=width,
                height=height,
                lines=lines,
                parts=tuple(parts),
            )
        )
    adapter = ENGINES.get(run.engine)
    return SavedEngineRun(
        name=name,
        engine=run.engine,
        markdown=bool(adapter and adapter.markdown),
        pages=tuple(pages),
        files=tuple(files),
    )


def _web_image(data: bytes) -> tuple[bytes | None, str, tuple[int, int]]:
    """Return an image as a browser shows it: PNG or JPEG data.

    Args:
        data: Image file data, in any format PIL reads.

    Returns:
        The data (converted to PNG unless it is PNG or JPEG), its file
        extension and its ``(width, height)``; ``None`` data if it cannot
        be read.
    """
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as picture:
            size = picture.size
            if picture.format == "PNG":
                return data, "png", size
            if picture.format == "JPEG":
                return data, "jpg", size
            buffer = io.BytesIO()
            picture.convert("RGB").save(buffer, format="PNG")
            return buffer.getvalue(), "png", size
    except Exception:
        return None, "png", (0, 0)


def _saved_region(region: doc_ir.Region, stem: str, write) -> SavedRegion:
    """Describe a region, writing its picture if it has one."""
    box = None
    if region.bbox and len(region.bbox) >= 4:
        x1, y1, x2, y2 = (float(value) for value in region.bbox[:4])
        if x2 > x1 and y2 > y1:
            box = (x1, y1, x2 - x1, y2 - y1)
    image = None
    picture = _decode((region.asset or {}).get("b64"))
    if picture and region.type in (doc_ir.FIGURE, doc_ir.SEAL):
        extension = (region.asset or {}).get("ext") or "png"
        image = write(
            f"{stem}_{region.id}.{extension}",
            picture,
            "Figure",
            f"image/{'jpeg' if extension == 'jpg' else extension}",
            listed=False,
        )
    described = region.visual_description
    mark = ""
    if region.type == doc_ir.CHECKBOX:
        mark = (region.markup or {}).get("state", "uncertain")
    return SavedRegion(
        id=region.id,
        type=region.type,
        box=box,
        text=region.text.strip(),
        html=sanitize_table_html(region.content.get("html", "").strip())
        if region.type == doc_ir.TABLE
        else "",
        latex=region.content.get("latex", "").strip()
        if region.type == doc_ir.FORMULA
        else "",
        image=image,
        description=described.description if described else "",
        visible_text=described.visible_text if described else "",
        described_by=described.model if described else "",
        mark=mark,
    )


def _decode(b64: str | None) -> bytes | None:
    """Return the bytes of a base64 text, or ``None``."""
    if not b64:
        return None
    try:
        return base64.b64decode(b64)
    except ValueError:
        return None


def _image_size(data: bytes) -> tuple[int, int]:
    """Return an image's ``(width, height)``, or ``(0, 0)``."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as picture:
            return picture.size
    except Exception:
        return 0, 0


def _progress_adapter(
    on_progress: ProgressCallback,
) -> Callable[[float, str], None]:
    """Turn the pipeline's ``(fraction, text)`` callbacks into updates."""

    def report(fraction: float, text: str) -> None:
        on_progress(Progress(text, fraction))

    return report
