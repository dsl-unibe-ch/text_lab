"""Historic handwriting, a tool of the OCR page.

Historic German handwriting (Kurrent, medieval manuscripts) read line by
line (``features.handwriting.service``): kraken finds the lines on each
page, and a recognition model of the user's choice reads them one by one.
One PDF or image, or a ZIP read as a batch with each model loaded once.

The form is drawn by the OCR page (:mod:`.ocr`, with
:func:`form_context`); its jobs are the OCR page's. The result shows each
page beside its numbered lines, each outlined on the page and linked to
its text, and the models used, for citation.
"""

from __future__ import annotations

import dataclasses
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

from textlab.common import gpu_manager
from textlab.common.progress import Progress
from textlab.features.handwriting import service
from textlab.ui.web import uploads
from textlab.ui.web.auth import Checked, protected_router
from textlab.ui.web.jobs import Download, JobContext, render_card, runner_of

router = protected_router()

#: The page whose jobs these are: the OCR page.
PAGE = "ocr"

#: What the tool reads: documents and images, or a ZIP of them.
EXTENSIONS = (*service.INPUT_EXTENSIONS, ".zip")


@dataclasses.dataclass(frozen=True)
class ModelChoice:
    """A recognition model, as the form offers it.

    Attributes:
        key: Its key, the form's value for it.
        label: Its name.
        description: What it was trained on and how well it reads.
        recognition: The models that read the lines, for citation.
    """

    key: str
    label: str
    description: str
    recognition: str


def model_choices() -> tuple[list[ModelChoice], str | None]:
    """Return the recognition models to choose from, or why there are none.

    Returns:
        The models, in the configured order, and ``None``; or no models
        and a sentence for users.
    """
    try:
        models = service.available_models()
    except ValueError as error:
        return [], f"The handwriting models are not configured: {error}"
    if not models:
        return [], "No handwriting models are configured."
    return [
        ModelChoice(
            key=model.key,
            label=model.label,
            description=model.description,
            recognition=model.models_used()["recognition"],
        )
        for model in models
    ], None


def form_context() -> dict[str, Any]:
    """Return what the OCR page needs to draw this tool's form.

    Returns:
        The models (``handwriting_models``), why there are none
        (``handwriting_problem``), the line detector, and the files read.
    """
    models, problem = model_choices()
    return {
        "handwriting_models": models,
        "handwriting_problem": problem,
        "line_detector": service.LINE_DETECTOR,
        "handwriting_accept": ",".join(EXTENSIONS),
        "handwriting_formats": uploads.format_names(service.INPUT_EXTENSIONS)
        + ", or a ZIP of them",
    }


@router.post("/ocr/handwriting")
async def transcribe(request: Request, session: Checked) -> HTMLResponse:
    """Transcribe a document, an image or a ZIP of them.

    Args:
        request: The form: ``file`` and ``model``.
        session: The session, with its CSRF token checked.

    Returns:
        The new job's card.

    Raises:
        UploadError: If the file is missing or of another type, or the
            model is not one offered.
    """
    form = await uploads.read_form(request, max_files=1)
    upload = uploads.single_file(form, "file", suffixes=EXTENSIONS)
    models, _problem = model_choices()
    key = str(form.get("model") or "")
    if key not in {model.key for model in models}:
        raise uploads.UploadError("Choose a handwriting model from the list.")
    title = uploads.display_name(upload)
    batch = PurePosixPath(title).suffix.lower() == ".zip"
    runner = runner_of(request)
    job = runner.create(
        title=title,
        page=PAGE,
        gpu=gpu_manager.HANDWRITING,
        result_template="ocr/handwriting_batch_result.html"
        if batch
        else "ocr/handwriting_result.html",
    )
    try:
        path = await run_in_threadpool(uploads.save, upload, job.inputs)
    except BaseException:
        runner.discard(job)
        raise
    work = transcribe_archive if batch else transcribe_document
    options = service.HandwritingOptions(model=key, previews=False)
    runner.submit(job, partial(work, path, title, options))
    return HTMLResponse(render_card(request, job))


# ---------------------------------------------------------------------------
# The jobs' work
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class HandwritingArchive:
    """A ZIP transcribed: an overview of its files.

    Attributes:
        files: Each file's overview.
        models: The models that produced it, as role and model.
    """

    files: tuple[service.BatchFile, ...]
    models: tuple[tuple[str, str], ...]

    @property
    def pages(self) -> int:
        """How many pages the files have in all."""
        return sum(item.pages for item in self.files)

    @property
    def lines(self) -> int:
        """How many lines were transcribed in all."""
        return sum(item.lines for item in self.files)


def transcribe_document(
    path: Path,
    title: str,
    options: service.HandwritingOptions,
    context: JobContext,
) -> service.SavedTranscription:
    """Transcribe one PDF or image and offer the downloads.

    Args:
        path: The upload, in the job's inputs.
        title: Its name as uploaded.
        options: The model.
        context: The job's context.

    Returns:
        The transcription to show.
    """
    context.report(Progress("Starting", 0.0))
    run = service.transcribe_document(
        title,
        path.read_bytes(),
        options,
        on_progress=context.report,
        cancel=context.cancel,
    )
    context.report(Progress("Saving the results", 1.0))
    saved = service.save_transcription(run, context.outputs)
    for item in saved.files:
        context.offer(
            Download(
                item.name,
                item.label,
                item.media_type,
                path=item.path,
                listed=item.listed,
            )
        )
    return saved


def transcribe_archive(
    path: Path,
    title: str,
    options: service.HandwritingOptions,
    context: JobContext,
) -> HandwritingArchive:
    """Transcribe every document and image in a ZIP, and offer the ZIP.

    Args:
        path: The ZIP, in the job's inputs.
        title: Its name as uploaded.
        options: The model.
        context: The job's context.

    Returns:
        The overview of the files.
    """
    context.report(Progress("Unpacking the archive", 0.0))
    batch = service.transcribe_batch(
        path, options, on_progress=context.report, cancel=context.cancel
    )
    target = context.outputs / f"{PurePosixPath(title).stem}_handwriting.zip"
    target.write_bytes(batch.zip_bytes)
    context.offer(
        Download(target.name, "All results", "application/zip", path=target)
    )
    return HandwritingArchive(
        files=batch.items, models=service.cited_models(batch.models_used)
    )
