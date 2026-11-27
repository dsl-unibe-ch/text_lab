"""Hugging Face models, a tool of the OCR page.

Documents read by a vision-language model the user names on the Hugging
Face Hub (``features.hf_ocr.service``), with a prompt and a token limit of
their choice. The model reads each page as a whole; it is downloaded into
the user's own folder on first use, which the form lists with a delete
button per model, since it counts towards the user's quota. One PDF or
image, or a ZIP read as a batch with the model loaded once.

The form is drawn by the OCR page (:mod:`.ocr`, with
:func:`form_context`); its jobs are the OCR page's. The result shows each
page beside the text the model returned, as written, and the model,
revision and prompt, for citation.
"""

from __future__ import annotations

import dataclasses
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import FormData

from textlab.common import gpu_manager
from textlab.common.progress import Progress
from textlab.features.hf_ocr import service
from textlab.ui.web import uploads
from textlab.ui.web.auth import Checked, protected_router
from textlab.ui.web.jobs import (
    Download,
    JobContext,
    JobRunner,
    render_card,
    runner_of,
)
from textlab.ui.web.render import renderer

router = protected_router()

#: The page whose jobs these are: the OCR page.
PAGE = "ocr"

#: What the tool reads: documents and images, or a ZIP of them.
EXTENSIONS = (*service.INPUT_EXTENSIONS, ".zip")

#: A model to name as an example.
EXAMPLE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


@dataclasses.dataclass(frozen=True)
class StoredModels:
    """The user's downloaded models, as the form lists them.

    Attributes:
        models: Each model's name and size, by name.
        total: Their size in all, for people.
        folder: Where they are.
    """

    models: tuple[tuple[str, str], ...]
    total: str
    folder: str


def stored_models() -> StoredModels:
    """Return the user's downloaded models.

    Returns:
        The models with their sizes, and the folder they are in.
    """
    found = service.stored_models()
    return StoredModels(
        models=tuple(
            (model.repo_id, service.format_size(model.size_bytes))
            for model in found
        ),
        total=service.format_size(sum(model.size_bytes for model in found)),
        folder=str(service.models_dir()),
    )


def form_context() -> dict[str, Any]:
    """Return what the OCR page needs to draw this tool's form.

    Returns:
        The downloaded models (``hf_stored``), the defaults and limits,
        and the files read.
    """
    return {
        "hf_stored": stored_models(),
        "hf_example": EXAMPLE_MODEL,
        "hf_prompt": service.DEFAULT_PROMPT,
        "hf_tokens": service.DEFAULT_MAX_NEW_TOKENS,
        "hf_tokens_limit": service.MAX_NEW_TOKENS_LIMIT,
        "hf_accept": ",".join(EXTENSIONS),
        "hf_formats": uploads.format_names(service.INPUT_EXTENSIONS)
        + ", or a ZIP of them",
    }


def running(runner: JobRunner) -> bool:
    """Return whether one of the session's jobs uses a Hugging Face model.

    Args:
        runner: The session's job runner.

    Returns:
        True while such a job waits or runs.
    """
    return any(
        job.active and job.gpu == gpu_manager.HF_OCR
        for job in runner.jobs(PAGE)
    )


@router.post("/ocr/huggingface")
async def read(request: Request, session: Checked) -> HTMLResponse:
    """Read a document, an image or a ZIP of them with a named model.

    Args:
        request: The form: ``file``, ``repo``, ``prompt`` and
            ``max_new_tokens``.
        session: The session, with its CSRF token checked.

    Returns:
        The new job's card.

    Raises:
        UploadError: If the file is missing or of another type, or the
            model's name or the token limit is not valid.
    """
    form = await uploads.read_form(request, max_files=1)
    upload = uploads.single_file(form, "file", suffixes=EXTENSIONS)
    options = read_options(form)
    title = uploads.display_name(upload)
    batch = PurePosixPath(title).suffix.lower() == ".zip"
    runner = runner_of(request)
    job = runner.create(
        title=title,
        page=PAGE,
        gpu=gpu_manager.HF_OCR,
        result_template="ocr/huggingface_batch_result.html"
        if batch
        else "ocr/huggingface_result.html",
    )
    try:
        path = await run_in_threadpool(uploads.save, upload, job.inputs)
    except BaseException:
        runner.discard(job)
        raise
    work = read_archive if batch else read_document
    runner.submit(job, partial(work, path, title, options))
    return HTMLResponse(render_card(request, job))


def read_options(form: FormData) -> service.HfOcrOptions:
    """Read the model, prompt and token limit from the form.

    Args:
        form: The form.

    Returns:
        The options.

    Raises:
        UploadError: If the model's name or the token limit is not valid.
    """
    try:
        tokens = int(str(form.get("max_new_tokens") or "").strip())
    except ValueError:
        tokens = 0
    try:
        return service.HfOcrOptions(
            repo_id=str(form.get("repo") or ""),
            prompt=str(form.get("prompt") or "").strip(),
            max_new_tokens=tokens,
        )
    except service.HfOcrError as error:
        raise uploads.UploadError(str(error)) from error


@router.post("/ocr/huggingface/models/delete")
async def remove_model(request: Request, session: Checked) -> HTMLResponse:
    """Delete one of the user's downloaded models.

    Refused while a job of the session uses a Hugging Face model, which
    may be the one to delete.

    Args:
        request: The form: ``repo``, the model's name.
        session: The session, with its CSRF token checked.

    Returns:
        The list of downloaded models, without it.

    Raises:
        UploadError: If a job uses a model now.
    """
    form = await uploads.read_form(request, max_files=0, max_fields=1)
    if running(runner_of(request)):
        raise uploads.UploadError(
            "A model is in use. Delete it when its job has ended."
        )
    repo = str(form.get("repo") or "")
    names = {model.repo_id for model in service.stored_models()}
    if repo in names:
        await run_in_threadpool(service.delete_model, repo)
    return renderer(request).response(
        request, "ocr/hf_models.html", hf_stored=stored_models()
    )


# ---------------------------------------------------------------------------
# The jobs' work
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class HfArchive:
    """A ZIP read by a model: an overview of its files.

    Attributes:
        files: Each file's overview.
        model: The model that read them.
        prompt: The instruction sent with every page.
    """

    files: tuple[service.BatchFile, ...]
    model: service.ModelUsed
    prompt: str

    @property
    def pages(self) -> int:
        """How many pages the files have in all."""
        return sum(item.pages for item in self.files)


def read_document(
    path: Path,
    title: str,
    options: service.HfOcrOptions,
    context: JobContext,
) -> service.SavedHfRun:
    """Read one PDF or image with the model and offer the downloads.

    Args:
        path: The upload, in the job's inputs.
        title: Its name as uploaded.
        options: The model, prompt and token limit.
        context: The job's context.

    Returns:
        The run to show.
    """
    context.report(Progress("Starting", 0.0))
    run = service.read_document(
        title,
        path.read_bytes(),
        options,
        on_progress=context.report,
        cancel=context.cancel,
    )
    context.report(Progress("Saving the results", 1.0))
    saved = service.save_run(run, context.outputs)
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


def read_archive(
    path: Path,
    title: str,
    options: service.HfOcrOptions,
    context: JobContext,
) -> HfArchive:
    """Read every document and image in a ZIP, and offer the result ZIP.

    Args:
        path: The ZIP, in the job's inputs.
        title: Its name as uploaded.
        options: The model, prompt and token limit.
        context: The job's context.

    Returns:
        The overview of the files.
    """
    context.report(Progress("Unpacking the archive", 0.0))
    batch = service.read_batch(
        path, options, on_progress=context.report, cancel=context.cancel
    )
    target = context.outputs / f"{PurePosixPath(title).stem}_hf_ocr.zip"
    target.write_bytes(batch.zip_bytes)
    context.offer(
        Download(target.name, "All results", "application/zip", path=target)
    )
    return HfArchive(
        files=batch.items, model=batch.model, prompt=options.prompt
    )
