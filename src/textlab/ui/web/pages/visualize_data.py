"""The Visualize Data page: charts and statistics of a table, by LLM agents.

Choosing a table uploads it at once: it is read as the agents will read it
and described (its columns, as the agents will treat them, and its first
rows), so the user can choose the columns to analyze, and requests that
name its columns are suggested. The upload waits in a draft job
(``JobRunner.draft``) while the user writes the request; every analysis
links it into a job of its own, so the same table can be asked about
again.

An analysis is a job in the GPU queue, as the agents' model runs on the
GPU. Its card shows the agents' steps while they work (the job's preview),
then the summary, the statistical results and the charts, each with its
code. Interactive charts are drawn by app.js with the vendored plotly.js,
as in Chat. The downloads are a report and a ZIP of everything
(``service.save_result``).
"""

from __future__ import annotations

import dataclasses
from functools import partial
from pathlib import Path, PurePosixPath

from fastapi import Request
from fastapi.responses import HTMLResponse
from markupsafe import Markup
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import FormData

from textlab.common import gpu_manager
from textlab.common.progress import CancelledError, Progress
from textlab.features.visualization import service
from textlab.ui.web import uploads
from textlab.ui.web.auth import Checked, protected_router
from textlab.ui.web.jobs import (
    Download,
    Job,
    JobContext,
    render_card,
    runner_of,
)
from textlab.ui.web.render import renderer, shorten

router = protected_router()

#: The page's id, which also names its jobs.
PAGE = "visualize_data"

#: The table files the agents read.
EXTENSIONS = tuple(f".{kind}" for kind in service.INPUT_TYPES)

#: The templates of an analysis's card: its result, and its steps while
#: the agents work.
RESULT_TEMPLATE = "visualize_data/result.html"
STEPS_TEMPLATE = "visualize_data/steps.html"

#: Characters of the request in a job's title.
TITLE_CHARS = 60

#: Characters of an agent's step shown as the job's progress.
STEP_CHARS = 140

#: How many times starting an analysis checks that the model server
#: answers, half a second apart (it may still be starting).
SERVER_ATTEMPTS = 4

#: Most fields of the form that starts an analysis: a check box for each
#: column, and a few more.
MAX_FIELDS = 10_000

#: Most uploaded tables kept at a time. A table replaces the one its form
#: names; a page opened again names none, so the oldest go beyond this.
MAX_TABLES = 3


@dataclasses.dataclass(frozen=True)
class Table:
    """An uploaded table waiting in a draft job.

    Attributes:
        name: Its file name, as uploaded.
        file: Its file in the draft's inputs.
        size: Its size in bytes.
        preview: Its columns and first rows.
    """

    name: str
    file: str
    size: int
    preview: service.TablePreview


@dataclasses.dataclass(frozen=True)
class Analysis:
    """A finished analysis, as its card shows it.

    Attributes:
        request: What was asked.
        saved: The result, saved in the job's outputs.
        steps: The agents' steps, as ``(level, message)`` pairs.
    """

    request: service.AnalysisRequest
    saved: service.SavedAnalysis
    steps: tuple[tuple[str, str], ...]

    @property
    def r_note(self) -> str:
        """Why a chart has no R code, if R code was asked for; else empty."""
        return service.R_NOT_AVAILABLE if self.request.include_r_code else ""


def tables(request: Request) -> dict[str, Table]:
    """Return the session's uploaded tables, by draft job id.

    Args:
        request: The current request.

    Returns:
        The app's record of them; empty at first.
    """
    state = request.app.state
    found = getattr(state, "visualize_tables", None)
    if found is None:
        found = state.visualize_tables = {}
    return found


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/visualize-data")
def page(request: Request) -> HTMLResponse:
    """Show the form and the session's analyses.

    Args:
        request: The current request.

    Returns:
        The page.
    """
    jobs = runner_of(request).jobs(PAGE)
    options = service.model_options()
    return renderer(request).response(
        request,
        "visualize_data/page.html",
        page_id=PAGE,
        accept=",".join(EXTENSIONS),
        formats=uploads.format_names(EXTENSIONS),
        options=options,
        model=last_model(jobs, options),
        ready=service.server_ready(),
        max_rows=service.MAX_ROWS,
        minutes=service.ANALYSIS_TIMEOUT_SECONDS // 60,
        cards=Markup("".join(render_card(request, job) for job in jobs[::-1])),
    )


@router.post("/visualize-data/table")
async def upload_table(request: Request, session: Checked) -> HTMLResponse:
    """Read an uploaded table and describe its columns to choose from.

    The upload is kept in a draft job until the session ends or another
    table replaces it (``table``, the previous one's id, which stays if
    the new one cannot be read); of the others, the newest
    :data:`MAX_TABLES` are kept.

    Args:
        request: The form: ``file`` and, after a first upload, ``table``.
        session: The session, with its CSRF token checked.

    Returns:
        The table's description, with the columns to choose from, and the
        suggested requests out of band.

    Raises:
        UploadError: If the file is missing, of another type, or cannot be
            read as a table.
    """
    form = await uploads.read_form(request, max_files=1, max_fields=2)
    runner = runner_of(request)
    known = tables(request)
    upload = uploads.single_file(form, "file", suffixes=EXTENSIONS)
    name = uploads.display_name(upload)
    draft = runner.create(title=name, page=PAGE)
    try:
        path = await run_in_threadpool(uploads.save, upload, draft.inputs)
        preview = await run_in_threadpool(service.preview_table, name, path)
    except ValueError as error:
        runner.discard(draft)
        raise uploads.UploadError(str(error)) from error
    except BaseException:
        runner.discard(draft)
        raise
    previous = str(form.get("table") or "")
    if previous in known:
        forget_table(request, previous)
    table = known[draft.id] = Table(
        name, path.name, path.stat().st_size, preview
    )
    for old in list(known)[:-MAX_TABLES]:
        forget_table(request, old)
    return renderer(request).response(
        request,
        "visualize_data/table.html",
        table_id=draft.id,
        table=table,
        examples=service.example_requests(preview),
        max_rows=service.MAX_ROWS,
    )


@router.post("/visualize-data")
async def start_analysis(request: Request, session: Checked) -> HTMLResponse:
    """Start an analysis of the uploaded table.

    Args:
        request: The form: ``table``, the ticked ``columns``, the
            ``request``, the ``model`` and ``r_code``.
        session: The session, with its CSRF token checked.

    Returns:
        The new job's card.

    Raises:
        UploadError: If there is no table, a choice is not valid, or the
            model server does not answer.
    """
    form = await uploads.read_form(request, max_files=0, max_fields=MAX_FIELDS)
    runner = runner_of(request)
    key = str(form.get("table") or "")
    table = tables(request).get(key)
    draft = runner.draft(key) if table else None
    if table is None or draft is None:
        raise uploads.UploadError(
            "Choose the table first: a CSV, TSV, Excel or JSON file."
        )
    asked = read_request(form, table, service.model_options())
    if not await run_in_threadpool(service.server_ready, SERVER_ATTEMPTS):
        raise uploads.UploadError(
            "The language model server (Ollama) does not answer. Try again "
            "in a minute; if it keeps failing, send ollama.log from the "
            "session's folder to support."
        )
    job = runner.create(
        title=job_title(table.name, asked.prompt),
        page=PAGE,
        gpu=gpu_manager.LLM,
        ollama_model=asked.model,
        result_template=RESULT_TEMPLATE,
        preview_template=STEPS_TEMPLATE,
    )
    try:
        path = await run_in_threadpool(
            uploads.copy_into, draft.inputs / table.file, job.inputs
        )
    except BaseException:
        runner.discard(job)
        raise
    runner.submit(job, partial(analyze, asked, path))
    return HTMLResponse(render_card(request, job))


def forget_table(request: Request, key: str) -> None:
    """Drop an uploaded table with its draft job and file.

    Analyses of it go on: each has its own link to the file.

    Args:
        request: The current request.
        key: The table's draft job id.
    """
    tables(request).pop(key, None)
    runner = runner_of(request)
    draft = runner.draft(key)
    if draft is not None:
        runner.discard(draft)


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------


def read_request(
    form: FormData, table: Table, options: service.ModelOptions
) -> service.AnalysisRequest:
    """Read what to analyze from the form.

    Columns are sent ticked; all of them, or none, means the whole table,
    which the agents are then told nothing about.

    Args:
        form: The form.
        table: The uploaded table.
        options: The models offered.

    Returns:
        The request.

    Raises:
        UploadError: If no model is set up, another model or an unknown
            column was sent.
    """
    if not options.models:
        raise uploads.UploadError(
            "No language models are set up for Visualize Data. Please tell "
            "support."
        )
    model = str(form.get("model") or options.models[0])
    if model not in options.models:
        raise uploads.UploadError("Choose one of the models offered.")
    names = table.preview.names
    ticked = {str(value) for value in form.getlist("columns")}
    if not ticked <= set(names):
        raise uploads.UploadError("Choose the columns from the table's list.")
    columns = ()
    if ticked and ticked != set(names):
        columns = tuple(name for name in names if name in ticked)
    return service.AnalysisRequest(
        model=model,
        prompt=str(form.get("request") or "").strip(),
        columns=columns,
        include_r_code=bool(form.get("r_code")),
        file_name=table.name,
    )


def job_title(name: str, prompt: str) -> str:
    """Return an analysis's title: the table, and the request in short."""
    return f"{name}: {shorten(prompt or 'explore the table', TITLE_CHARS)}"


def last_model(jobs: list[Job], options: service.ModelOptions) -> str:
    """Return the model to offer first: the last analysis's, if offered."""
    for job in reversed(jobs):
        if job.ollama_model in options.models:
            return job.ollama_model
    return options.models[0] if options.models else ""


# ---------------------------------------------------------------------------
# The job's work
# ---------------------------------------------------------------------------


def analyze(
    asked: service.AnalysisRequest, path: Path, context: JobContext
) -> Analysis:
    """Run the agents on a table and offer the downloads.

    Args:
        asked: What to analyze.
        path: The table, in the job's inputs; the agents write their
            charts next to it, and the inputs go when the job ends.
        context: The job's context.

    Returns:
        The result to show.

    Raises:
        CancelledError: If the job was cancelled; what the agents made
            until then is not kept.
        RuntimeError: If the analysis failed or took too long.
    """
    steps: list[tuple[str, str]] = []

    def on_log(level: str, message: str) -> None:
        steps.append((level, message))
        line = message.strip().splitlines()[0] if message.strip() else ""
        context.report(Progress(shorten(line, STEP_CHARS)))
        context.preview(tuple(steps), every=0)

    run = service.analyze(
        asked,
        str(path),
        on_progress=context.report,
        on_log=on_log,
        cancel=context.cancel,
        run_prefix="viz",
    )
    if run.status == service.CANCELLED:
        raise CancelledError("The analysis was cancelled.")
    if run.status == service.TIMEOUT:
        minutes = service.ANALYSIS_TIMEOUT_SECONDS // 60
        raise RuntimeError(
            f"The analysis took longer than {minutes} minutes and was "
            "stopped. Ask for less at once, or choose fewer columns."
        )
    if run.status != service.DONE or run.result is None:
        raise RuntimeError(run.error or "The analysis failed.")
    context.report(Progress("Saving the results", 1.0))
    saved = service.save_result(
        run.result,
        asked,
        context.outputs,
        name=PurePosixPath(asked.file_name).stem or "data",
    )
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
    return Analysis(asked, saved, tuple(run.logs))
