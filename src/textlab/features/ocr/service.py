"""Recognizing documents and images: the OCR feature's API.

Interfaces call this module. :func:`recognize_document` runs the automatic
pipeline (:mod:`.pipeline`) on one uploaded PDF or image and prepares every
download; :func:`recognize_batch` does the same for a ZIP archive, keeping
the recognition model loaded for the whole batch, and optionally reads a
batch of filled-in questionnaires against the form they share
(:mod:`textlab.features.survey.survey_batch`).

Uploads, page images and intermediate files go to a job folder in the
workspace (area ``ocr``), which is removed when the run ends; results are
returned in memory.

The few names interfaces need from other modules (the result model
:mod:`.doc_ir`, Tesseract languages, the layout preview) are re-exported
here, so interfaces import only this module and :mod:`.doc_ir`.
"""

from __future__ import annotations

import dataclasses
import io
import os
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

from textlab.common.progress import Progress, ProgressCallback, no_progress
from textlab.common.storage import get_workspace, remove_tree
from textlab.common.upload_safety import (
    extract_zip_safely,
    safe_upload_name,
    unique_output_directory,
)
from textlab.features.ocr import doc_ir, vision_enrich
from textlab.features.ocr.layout_preview import (
    LAYOUT_TYPE_COLORS,
    render_layout_preview,
)
from textlab.features.ocr.pipeline import document_summary, process_document
from textlab.features.ocr.searchable_pdf import (
    DEFAULT_TESSERACT_LANG,
    TESSERACT_LANGUAGES,
)
from textlab.features.ocr.vl_session import VLWorkerSession
from textlab.features.survey import form_extract, survey_batch

__all__ = [
    "INPUT_EXTENSIONS",
    "BatchResult",
    "DocumentDownloads",
    "DocumentResult",
    "OcrOptions",
    "SurveyBatch",
    "document_summary",
    "process_document",
    "recognize_batch",
    "recognize_document",
    # Re-exported for interfaces
    "DEFAULT_TESSERACT_LANG",
    "LAYOUT_TYPE_COLORS",
    "TESSERACT_LANGUAGES",
    "render_layout_preview",
]

#: File types the automatic pipeline reads, alone or inside a ZIP archive.
INPUT_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")

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


@dataclasses.dataclass
class SurveyBatch:
    """A batch of questionnaires read against the form they share.

    Attributes:
        template: The blank form learned from the batch.
        readings: Each questionnaire's answers.
        documents: Each file's recognized document, slimmed, by its folder
            in the result ZIP; used to rebuild the exports after review.
        warning: A note for the user when the batch is too small to learn
            the form reliably.
    """

    template: Any
    readings: list[Any]
    documents: dict[str, Any]
    warning: str | None = None


@dataclasses.dataclass
class BatchResult:
    """The outcome of :func:`recognize_batch`.

    Attributes:
        zip_bytes: One folder of outputs per file, mirroring the archive.
        file_count: Number of files recognized.
        elapsed: Duration of the run in seconds.
        survey: The questionnaire results, in survey batch mode.
    """

    zip_bytes: bytes
    file_count: int
    elapsed: float
    survey: SurveyBatch | None = None


def recognize_document(
    name: str,
    data: bytes,
    options: OcrOptions,
    *,
    on_progress: ProgressCallback = no_progress,
) -> DocumentResult:
    """Recognize one PDF or image.

    Args:
        name: The uploaded file's name; only its last part is used.
        data: The file's content.
        options: How to recognize it.
        on_progress: Receives progress updates.

    Returns:
        The document, its summary and its downloads.
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

    Returns:
        The result ZIP, and the questionnaire results in survey batch mode.

    Raises:
        ValueError: If the archive holds no supported file.
    """
    vision_client = (
        vision_enrich.OllamaVisionClient()
        if options.describe_images or options.extract_survey
        else None
    )
    same_layout_template = (
        form_extract.SameLayoutTemplate()
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
) -> BatchResult:
    """Run :func:`recognize_batch` inside its job folder."""
    input_dir = job_dir / "input"
    workspace_dir = job_dir / "workspace"
    results_dir = workspace_dir / "results"
    input_dir.mkdir()
    results_dir.mkdir(parents=True)

    files = _extract_inputs(archive, input_dir)
    if not files:
        raise ValueError("No valid documents or images found in the ZIP.")
    n_files = len(files)

    # The consensus template needs all questionnaires up front.
    survey = None
    if options.survey_batch_mode:

        def template_progress(fraction: float, text: str) -> None:
            on_progress(
                Progress(
                    f"Questionnaire layout — {text}",
                    min(0.99, max(0.0, fraction)),
                )
            )

        template_progress(0.0, f"reading {n_files} file(s)...")
        template, _blanks = survey_batch.prepare_template(
            files,
            label=True,
            progress=template_progress,
            vl_session=vl_session,
        )
        on_progress(
            Progress(
                f"Questionnaire layout — {template.control_count} response "
                f"controls in {len(template.rules)} answers"
            )
        )
        survey = SurveyBatch(
            template=template,
            readings=[],
            documents={},
            warning=template.provenance.get("small_batch_warning"),
        )

    batch_provenance = []
    batch_started = time.monotonic()
    file_times = []
    used_output_dirs: set[str] = set()
    for index, file_path in enumerate(files):
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
        )

        skip_tables = None
        if survey is not None:
            reading = survey_batch.read_document(file_path, survey.template)
            reading.export_directory = output_rel_dir.as_posix()
            survey.readings.append(reading)
            survey_batch.safe_csv(
                survey_batch.answers_for_document(reading, survey.template),
                file_output_dir / "survey_answers.csv",
            )
            survey_batch.to_form_groups(reading, survey.template, document)
            skip_tables = survey_batch.survey_table_regions(
                document, survey.template
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
            folder = file_output_dir.relative_to(results_dir).as_posix()
            survey.documents[folder] = survey_batch.slim_document(document)

        # Page images are only needed while a file is recognized.
        remove_tree(file_workspace)
        file_times.append(time.monotonic() - file_started)
        print(
            f"[OCR] file {index + 1}/{n_files} {rel_path}: "
            f"{file_times[-1]:.1f}s",
            flush=True,
        )

    if survey is not None:
        on_progress(Progress("Collecting questionnaire responses..."))
        survey_batch.write_batch_outputs(
            survey.readings, survey.template, results_dir / "survey"
        )

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
        zip_bytes=_zip_folder(results_dir),
        file_count=n_files,
        elapsed=elapsed,
        survey=survey,
    )


def _extract_inputs(
    archive: BinaryIO | str | os.PathLike, input_dir: Path
) -> list[Path]:
    """Extract the supported files of a ZIP archive, in a stable order.

    macOS resource-fork files (``._name``) are left out.
    """
    with zipfile.ZipFile(archive, "r") as source:
        files = extract_zip_safely(
            source, input_dir, allowed_extensions=set(INPUT_EXTENSIONS)
        )
    files = [path for path in files if not path.name.startswith("._")]
    files.sort(key=lambda path: str(path.relative_to(input_dir)).casefold())
    return files


def _zip_folder(folder: Path) -> bytes:
    """Return a ZIP archive of a folder's files, with relative names."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for root, _dirs, names in os.walk(folder):
            for name in names:
                path = Path(root) / name
                archive.write(path, path.relative_to(folder))
    return buffer.getvalue()


def _progress_adapter(
    on_progress: ProgressCallback,
) -> Callable[[float, str], None]:
    """Turn the pipeline's ``(fraction, text)`` callbacks into updates."""

    def report(fraction: float, text: str) -> None:
        on_progress(Progress(text, fraction))

    return report
