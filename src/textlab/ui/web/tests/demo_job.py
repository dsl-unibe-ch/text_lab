"""A demo job for the tests: a job through the whole runner, without a model.

``conftest.py`` adds :data:`router` to the test app, under ``/test/demo``;
the app itself never serves it. The job reports progress, can be cancelled,
reads an upload and offers two downloads, so the tests of the job pages
need no feature.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

from textlab.common.progress import Progress
from textlab.ui.web import uploads
from textlab.ui.web.auth import Checked, protected_router
from textlab.ui.web.jobs import Download, JobContext, render_card, runner_of

#: Where the tests post the demo job.
PATH = "/test/demo"

#: Page id of the demo jobs.
PAGE = "demo"

#: Longest demo job, in seconds.
MAX_SECONDS = 600

router: APIRouter = protected_router(prefix=PATH)


@router.post("")
async def start(request: Request, session: Checked) -> HTMLResponse:
    """Start a demo job from the form; return its card.

    Args:
        request: The form: ``seconds``, ``outcome`` (``"done"`` or
            ``"failed"``), ``gpu`` (checkbox) and an optional ``file``.
        session: The session, with its CSRF token checked.

    Returns:
        The new job's card.

    Raises:
        UploadError: If the values are not valid.
    """
    form = await uploads.read_form(request, max_files=1)
    try:
        seconds = int(str(form.get("seconds", "20")))
    except ValueError:
        seconds = -1
    if not 1 <= seconds <= MAX_SECONDS:
        raise uploads.UploadError(
            f"Choose between 1 and {MAX_SECONDS} seconds."
        )
    fail = form.get("outcome") == "failed"
    files = uploads.files(form, "file")

    runner = runner_of(request)
    job = runner.create(
        title=f"Demo job, {seconds} s",
        page=PAGE,
        gpu="llm" if form.get("gpu") else None,
    )
    try:
        saved = (
            [await run_in_threadpool(uploads.save, files[0], job.inputs)]
            if files
            else []
        )
    except BaseException:
        runner.discard(job)
        raise
    runner.submit(job, work_of(seconds, fail, saved))
    return HTMLResponse(render_card(request, job))


def work_of(seconds: int, fail: bool, inputs: list[Path]):
    """Return the work of a demo job.

    Args:
        seconds: How long the job runs.
        fail: Whether it fails at the end, to show an error.
        inputs: The uploaded file, if any, in the job's inputs.

    Returns:
        The work: reports progress twice a second, can be cancelled, reads
        the upload and offers two downloads. Its result is the report's
        lines.
    """

    def work(context: JobContext) -> dict:
        steps = seconds * 2
        for step in range(steps):
            context.check_cancelled()
            context.report(
                Progress(f"Step {step + 1} of {steps}", step / steps)
            )
            time.sleep(0.5)
        if fail:
            raise RuntimeError("The demo job failed on purpose.")

        lines = [f"The demo job ran for {seconds} s."]
        for path in inputs:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(
                f"Upload: {path.name}, {path.stat().st_size} bytes, "
                f"SHA-256 {digest}"
            )
        report = context.outputs / "demo-report.txt"
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        context.offer(
            Download("demo-report.txt", "Report", "text/plain", path=report)
        )
        context.offer(
            Download(
                "Grüße aus Bern.txt",
                "Umlauts in the name",
                "text/plain; charset=utf-8",
                data="Grüße aus Bern\n".encode(),
            )
        )
        return {"lines": lines}

    return work
