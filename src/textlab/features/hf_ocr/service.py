"""Hugging Face OCR models: the functions interfaces call.

A run reads one document or a whole batch with a model the user names:

1. **Pages** (:mod:`textlab.common.pages`): PDFs are rendered at
   :data:`DPI`, every frame of an image file becomes a page.
2. **Reading** (:mod:`.worker`): the model reads every page, in a worker in
   the image's Transformers 5 environment that loads the model once. The
   model is downloaded into the user's own folder (:mod:`.store`) on first
   use.
3. **Exports**: text and JSON per document, and ``model_used.txt`` naming
   the model, its revision and the prompt.

The worker exits when it is done, so the GPU is free afterwards. Each run
works in a job folder in the workspace (area ``hf_ocr``), removed when the
run ends; results are returned in memory.
"""

from __future__ import annotations

import io
import os
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

from textlab.common import container, jobs, pages
from textlab.common.progress import Progress, ProgressCallback, no_progress
from textlab.common.storage import get_workspace
from textlab.common.upload_safety import (
    safe_upload_name,
    unique_output_directory,
)
from textlab.features.hf_ocr import store
from textlab.features.hf_ocr.models import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_PROMPT,
    MAX_NEW_TOKENS_LIMIT,
    BatchRun,
    DocumentText,
    HfOcrError,
    HfOcrOptions,
    ModelUsed,
    OcrRun,
    PageText,
    model_used_text,
    parse_repo_id,
    result_json,
)

__all__ = [
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_PROMPT",
    "INPUT_EXTENSIONS",
    "MAX_NEW_TOKENS_LIMIT",
    "BatchRun",
    "HfOcrError",
    "HfOcrOptions",
    "OcrRun",
    "parse_repo_id",
    "read_batch",
    "read_document",
]

#: File types read, alone or inside a ZIP archive.
INPUT_EXTENSIONS = pages.INPUT_EXTENSIONS
#: Resolution PDFs are rendered at. Vision-language models resize pages
#: to their own limits, which 200 DPI already exceeds for an A4 page.
DPI = 200
#: Longest side of a preview image, in pixels.
PREVIEW_SIDE = 1200
#: Workspace area for job folders.
WORKSPACE_AREA = "hf_ocr"
WORKER = "textlab.features.hf_ocr.worker"
#: Name of the worker's own error class, whose message is meant for users.
_MODEL_ERROR = "OcrModelError"

#: Share of the progress bar for preparing pages; reading takes the rest.
_PAGES_END = 0.05


def read_document(
    name: str,
    data: bytes,
    options: HfOcrOptions,
    *,
    on_progress: ProgressCallback = no_progress,
) -> OcrRun:
    """Read one PDF or image with the chosen model.

    Args:
        name: The uploaded file's name; only its last part is used.
        data: The file's content.
        options: The model, prompt and token limit.
        on_progress: Receives progress updates.

    Returns:
        The text and its downloads.

    Raises:
        HfOcrError: If the file cannot be read or the model cannot be
            used.
        textlab.common.jobs.WorkerError: If the worker fails otherwise.
    """
    upload_name = safe_upload_name(name)
    stem = Path(upload_name).stem or "document"
    with get_workspace().temp_dir(WORKSPACE_AREA, prefix="run-") as job:
        input_path = job / "input" / upload_name
        input_path.parent.mkdir()
        input_path.write_bytes(data)
        on_progress(Progress("Preparing pages...", 0.0))
        images = _page_images(input_path, job / "pages")
        texts, model = read_pages(
            images, options, on_progress=_scaled(on_progress, _PAGES_END, 1)
        )
        document = _document(upload_name, texts)
        out_dir = job / "output"
        _write_outputs(document, model, options.prompt, out_dir, stem)
        (out_dir / "model_used.txt").write_text(
            model_used_text(model, options.prompt), encoding="utf-8"
        )
        zip_bytes = pages.zip_folder(out_dir)
        previews = [_preview_png(image) for image in images]
    on_progress(Progress("Done.", 1.0))
    return OcrRun(
        document=document,
        model=model,
        prompt=options.prompt,
        stem=stem,
        zip_bytes=zip_bytes,
        previews=previews,
    )


def read_batch(
    archive: BinaryIO | str | os.PathLike,
    options: HfOcrOptions,
    *,
    on_progress: ProgressCallback = no_progress,
) -> BatchRun:
    """Read every PDF and image in a ZIP archive with the chosen model.

    All pages of all files go through one worker, so the model is loaded
    once for the batch.

    Args:
        archive: The ZIP archive, as a file object or path.
        options: The model, prompt and token limit.
        on_progress: Receives progress updates.

    Returns:
        A ZIP with one folder per file, mirroring the archive.

    Raises:
        HfOcrError: If the archive holds no supported files, a file cannot
            be read or the model cannot be used.
        textlab.common.jobs.WorkerError: If the worker fails otherwise.
    """
    with get_workspace().temp_dir(WORKSPACE_AREA, prefix="batch-") as job:
        input_dir = (job / "input").resolve()
        try:
            files = pages.extract_inputs(archive, input_dir)
        except pages.PageError as error:
            raise HfOcrError(str(error)) from error
        if not files:
            raise HfOcrError(
                "The ZIP archive holds no supported files ("
                + ", ".join(INPUT_EXTENSIONS)
                + ")."
            )
        images, counts = [], []
        for index, path in enumerate(files):
            on_progress(
                Progress(
                    f"Preparing pages... {path.name}",
                    _PAGES_END * index / len(files),
                )
            )
            file_images = _page_images(path, job / "pages" / f"{index:05d}")
            images.extend(file_images)
            counts.append(len(file_images))
        texts, model = read_pages(
            images, options, on_progress=_scaled(on_progress, _PAGES_END, 1)
        )

        out_dir = job / "output"
        used: set[str] = set()
        start = 0
        for path, count in zip(files, counts, strict=True):
            name = path.relative_to(input_dir).as_posix()
            document = _document(name, texts[start : start + count])
            start += count
            folder = out_dir / unique_output_directory(name, used)
            _write_outputs(
                document, model, options.prompt, folder, path.stem or "file"
            )
        (out_dir / "model_used.txt").write_text(
            model_used_text(model, options.prompt), encoding="utf-8"
        )
        zip_bytes = pages.zip_folder(out_dir)
    on_progress(Progress("Done.", 1.0))
    return BatchRun(
        zip_bytes=zip_bytes, files=len(files), pages=len(images), model=model
    )


def read_pages(
    images: list[Path],
    options: HfOcrOptions,
    *,
    on_progress: ProgressCallback = no_progress,
    backend_python: str | None = None,
    cache_dir: Path | None = None,
) -> tuple[list[str], ModelUsed]:
    """Read page images with a model, in the worker.

    Args:
        images: The page images.
        options: The model, prompt and token limit.
        on_progress: Receives the worker's progress.
        backend_python: The worker's interpreter; by default the one of the
            image's Hugging Face OCR environment, or
            ``HF_OCR_BACKEND_PYTHON``.
        cache_dir: Where the model is downloaded to; by default the user's
            folder (:func:`.store.models_dir`).

    Returns:
        One text per image, and the model that read them.

    Raises:
        HfOcrError: If the model cannot be used.
        textlab.common.jobs.WorkerError: If the worker fails otherwise.
    """
    python = backend_python or container.env_python(
        container.HF_OCR_ENV, "HF_OCR_BACKEND_PYTHON"
    )
    cache_dir = store.models_dir() if cache_dir is None else cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    on_progress(Progress("Starting the model worker...", None))
    request = {
        **options.to_dict(),
        "cache_dir": str(cache_dir),
        "images": [str(image) for image in images],
    }
    try:
        result = jobs.run_worker(
            WORKER,
            request,
            area=WORKSPACE_AREA,
            on_progress=on_progress,
            python=python,
        )
    except jobs.WorkerError as error:
        if error.error_type == _MODEL_ERROR:
            raise HfOcrError(error.error_message) from error
        raise
    texts = result["texts"]
    if len(texts) != len(images):
        raise HfOcrError(
            f"The model returned {len(texts)} texts for {len(images)} pages."
        )
    return texts, ModelUsed.from_dict(result["model"])


def _page_images(path: Path, out_dir: Path) -> list[Path]:
    """Return a file's pages as images, with errors for users."""
    try:
        return pages.page_images(path, out_dir, DPI)
    except pages.PageError as error:
        raise HfOcrError(str(error)) from error


def _document(name: str, texts: list[str]) -> DocumentText:
    """Return a document from its page texts."""
    return DocumentText(
        name,
        [PageText(number, text) for number, text in enumerate(texts, 1)],
    )


def _write_outputs(
    document: DocumentText,
    model: ModelUsed,
    prompt: str,
    folder: Path,
    stem: str,
) -> None:
    """Write a document's text and JSON into a folder."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}.txt").write_text(document.text, encoding="utf-8")
    (folder / f"{stem}.json").write_text(
        result_json(document, model, prompt), encoding="utf-8"
    )


def _preview_png(image: Path) -> bytes:
    """Return a page image scaled down for display, as PNG."""
    from PIL import Image

    with Image.open(image) as page:
        preview = page.convert("RGB")
    preview.thumbnail((PREVIEW_SIDE, PREVIEW_SIDE))
    buffer = io.BytesIO()
    preview.save(buffer, format="PNG")
    return buffer.getvalue()


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
