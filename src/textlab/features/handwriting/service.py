"""Historic handwriting: the functions interfaces call.

A transcription runs in four steps, for one document or a whole batch:

1. **Pages** (:mod:`.documents`): PDFs are rendered at 300 DPI, every frame
   of an image file becomes a page.
2. **Line detection** (:mod:`.detect_worker`): kraken's baseline
   segmenter finds the text lines on every page, in reading order, in a
   worker in the image's ``kraken_backend`` environment. Each line is
   cropped (:mod:`.lines`).
3. **Recognition** (:mod:`.worker`): the chosen model reads every line
   crop, in a worker in the app's environment that loads the model once.
4. **Exports** (:mod:`.documents`): text, JSON and PAGE XML per document.

Both workers exit when they are done, so the GPU is free afterwards. Each
run works in a job folder in the workspace (area ``handwriting``), removed
when the run ends; results are returned in memory.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

from textlab.common import container, jobs
from textlab.common.progress import Progress, ProgressCallback, no_progress
from textlab.common.storage import get_workspace
from textlab.common.upload_safety import (
    safe_upload_name,
    unique_output_directory,
)
from textlab.features.handwriting import documents
from textlab.features.handwriting.documents import INPUT_EXTENSIONS
from textlab.features.handwriting.lines import line_boxes
from textlab.features.handwriting.models import (
    LINE_DETECTOR,
    BatchRun,
    DocumentTranscription,
    HandwritingError,
    HandwritingModel,
    HandwritingOptions,
    Line,
    PageTranscription,
    TranscriptionRun,
    available_models,
    get_model,
)

__all__ = [
    "INPUT_EXTENSIONS",
    "LINE_DETECTOR",
    "BatchRun",
    "HandwritingError",
    "HandwritingModel",
    "HandwritingOptions",
    "TranscriptionRun",
    "available_models",
    "transcribe_batch",
    "transcribe_document",
]

#: Workspace area for job folders.
WORKSPACE_AREA = "handwriting"
DETECT_WORKER = "textlab.features.handwriting.detect_worker"
RECOGNIZE_WORKER = "textlab.features.handwriting.worker"

#: Share of the progress bar for each step; recognition takes the rest.
_PAGES_END = 0.05
_DETECT_END = 0.25
_CROPS_END = 0.3


@dataclasses.dataclass
class _Page:
    """A page during a run: its image and, later, its line crops."""

    file_index: int
    number: int
    image: Path
    image_name: str
    include_image: bool
    width: int = 0
    height: int = 0
    lines: list[tuple[float, float, float, float]] = dataclasses.field(
        default_factory=list
    )
    crops: list[Path] = dataclasses.field(default_factory=list)


def transcribe_document(
    name: str,
    data: bytes,
    options: HandwritingOptions,
    *,
    on_progress: ProgressCallback = no_progress,
) -> TranscriptionRun:
    """Transcribe one PDF or image.

    Args:
        name: The uploaded file's name; only its last part is used.
        data: The file's content.
        options: The model and whether to draw previews.
        on_progress: Receives progress updates.

    Returns:
        The transcription and its downloads.

    Raises:
        HandwritingError: If the file cannot be read or the model is
            unknown.
        textlab.common.jobs.WorkerError: If a worker fails.
    """
    spec = get_model(options.model)
    upload_name = safe_upload_name(name)
    stem = Path(upload_name).stem or "document"
    with get_workspace().temp_dir(WORKSPACE_AREA, prefix="run-") as job:
        input_path = job / "input" / upload_name
        input_path.parent.mkdir()
        input_path.write_bytes(data)
        pages = _pages([input_path], job / "pages", on_progress)
        document = _transcribe([upload_name], pages, spec, job, on_progress)[0]
        previews = []
        if options.previews:
            previews = [
                documents.preview_png(page.image, result)
                for page, result in zip(pages, document.pages, strict=True)
            ]
        out_dir = job / "output"
        documents.write_outputs(document, out_dir, _included_images(pages))
        (out_dir / "models_used.txt").write_text(
            documents.models_used_text(document.models_used()),
            encoding="utf-8",
        )
        zip_bytes = documents.zip_folder(out_dir)
    on_progress(Progress("Done.", 1.0))
    return TranscriptionRun(
        document=document,
        text_name=f"{stem}.txt",
        json=documents.to_json(document),
        json_name=f"{stem}.json",
        previews=previews,
        zip_bytes=zip_bytes,
    )


def transcribe_batch(
    archive: BinaryIO | str | os.PathLike,
    options: HandwritingOptions,
    *,
    on_progress: ProgressCallback = no_progress,
) -> BatchRun:
    """Transcribe every PDF and image in a ZIP archive.

    All pages of all files go through one line-detection and one
    recognition worker, so each model is loaded once for the batch.

    Args:
        archive: The ZIP archive, as a file object or path.
        options: The model; previews are not drawn for a batch.
        on_progress: Receives progress updates.

    Returns:
        A ZIP with one folder per file, mirroring the archive.

    Raises:
        HandwritingError: If the archive holds no supported files, a file
            cannot be read or the model is unknown.
        textlab.common.jobs.WorkerError: If a worker fails.
    """
    spec = get_model(options.model)
    with get_workspace().temp_dir(WORKSPACE_AREA, prefix="batch-") as job:
        input_dir = (job / "input").resolve()
        files = documents.extract_inputs(archive, input_dir)
        if not files:
            raise HandwritingError(
                "The ZIP archive holds no supported files ("
                + ", ".join(INPUT_EXTENSIONS)
                + ")."
            )
        names = [path.relative_to(input_dir).as_posix() for path in files]
        pages = _pages(files, job / "pages", on_progress)
        results = _transcribe(names, pages, spec, job, on_progress)

        out_dir = job / "output"
        used: set[str] = set()
        for index, (name, document) in enumerate(
            zip(names, results, strict=True)
        ):
            folder = out_dir / unique_output_directory(name, used)
            file_pages = [page for page in pages if page.file_index == index]
            documents.write_outputs(
                document, folder, _included_images(file_pages)
            )
        models_used = results[0].models_used()
        (out_dir / "models_used.txt").write_text(
            documents.models_used_text(models_used), encoding="utf-8"
        )
        zip_bytes = documents.zip_folder(out_dir)
    on_progress(Progress("Done.", 1.0))
    return BatchRun(
        zip_bytes=zip_bytes,
        files=len(results),
        lines=sum(document.line_count for document in results),
        models_used=models_used,
    )


def _pages(
    files: list[Path], out_dir: Path, on_progress: ProgressCallback
) -> list[_Page]:
    """Return the pages of every file, as images."""
    pages = []
    for file_index, path in enumerate(files):
        on_progress(
            Progress(
                f"Preparing pages... {path.name}",
                _PAGES_END * file_index / len(files),
            )
        )
        images = documents.page_images(path, out_dir / f"{file_index:05d}")
        # A single image keeps its own pixels, so PAGE XML can refer to the
        # original file; rendered PDF pages and TIFF frames are included.
        single_image = path.suffix.lower() != ".pdf" and len(images) == 1
        for number, image in enumerate(images, start=1):
            name = path.name if single_image else f"page_{number:04d}.png"
            pages.append(
                _Page(file_index, number, image, name, not single_image)
            )
    return pages


def _transcribe(
    names: list[str],
    pages: list[_Page],
    spec: HandwritingModel,
    job: Path,
    on_progress: ProgressCallback,
) -> list[DocumentTranscription]:
    """Detect, crop and read the lines of all pages; one result per file."""
    polygons = detect_lines(
        [page.image for page in pages],
        on_progress=_scaled(on_progress, _PAGES_END, _DETECT_END),
    )
    crop_dir = job / "crops"
    crop_count = 0
    for index, (page, found) in enumerate(
        zip(pages, polygons, strict=True), start=1
    ):
        page.width, page.height = documents.image_size(page.image)
        page.lines = line_boxes(found)
        page.crops = documents.write_crops(
            page.image,
            documents.line_crops(page.lines, page.width, page.height),
            crop_dir,
            spec.max_pixels,
            first=crop_count,
        )
        crop_count += len(page.crops)
        fraction = _DETECT_END + (_CROPS_END - _DETECT_END) * (
            index / len(pages)
        )
        on_progress(Progress("Cropping lines...", fraction))

    crops = [crop for page in pages for crop in page.crops]
    texts = (
        recognize_lines(
            crops, spec, on_progress=_scaled(on_progress, _CROPS_END, 1.0)
        )
        if crops
        else []
    )

    results = [DocumentTranscription(name, spec, []) for name in names]
    remaining = iter(texts)
    for page in pages:
        lines = [
            Line(tuple(round(value) for value in box), next(remaining))
            for box in page.lines
        ]
        results[page.file_index].pages.append(
            PageTranscription(
                page=page.number,
                width=page.width,
                height=page.height,
                lines=lines,
                image_name=page.image_name,
            )
        )
    return results


def _included_images(pages: list[_Page]) -> list[Path | None]:
    """Return the page images to include next to the PAGE XML files."""
    return [page.image if page.include_image else None for page in pages]


def detect_lines(
    images: list[Path],
    *,
    on_progress: ProgressCallback = no_progress,
    backend_python: str | None = None,
) -> list[list[list[list[float]]]]:
    """Find the text lines on page images, in the line-detection worker.

    Args:
        images: The page images.
        on_progress: Receives the worker's progress.
        backend_python: The worker's interpreter; by default the one of the
            image's kraken environment, or ``KRAKEN_BACKEND_PYTHON``.

    Returns:
        Per image, the boundary polygon of every line in reading order, as
        lists of ``[x, y]`` points.
    """
    python = backend_python or container.env_python(
        container.KRAKEN_ENV, "KRAKEN_BACKEND_PYTHON"
    )
    on_progress(Progress("Starting line detection...", None))
    result = jobs.run_worker(
        DETECT_WORKER,
        {"images": [str(image) for image in images]},
        area=WORKSPACE_AREA,
        on_progress=on_progress,
        python=python,
    )
    pages = result["pages"]
    if len(pages) != len(images):
        raise HandwritingError(
            f"Line detection returned {len(pages)} pages for {len(images)}."
        )
    return [page["polygons"] for page in pages]


def recognize_lines(
    crops: list[Path],
    spec: HandwritingModel,
    *,
    on_progress: ProgressCallback = no_progress,
) -> list[str]:
    """Read line images with a model, in the recognition worker.

    Args:
        crops: The line images.
        spec: The recognition model.
        on_progress: Receives the worker's progress.

    Returns:
        One text per line image.
    """
    result = jobs.run_worker(
        RECOGNIZE_WORKER,
        {"model": spec.to_dict(), "crops": [str(crop) for crop in crops]},
        area=WORKSPACE_AREA,
        on_progress=on_progress,
    )
    texts = result["texts"]
    if len(texts) != len(crops):
        raise HandwritingError(
            f"Recognition returned {len(texts)} texts for {len(crops)} lines."
        )
    return texts


def _scaled(
    on_progress: ProgressCallback, start: float, end: float
) -> Callable[[Progress], None]:
    """Map a step's progress (0 to 1) onto ``start`` to ``end``."""

    def report(update: Progress) -> None:
        fraction = update.fraction
        if fraction is not None:
            fraction = start + (end - start) * fraction
        on_progress(Progress(update.message, fraction))

    return report
