"""Developer pages: the component gallery, the page mockups and a demo job.

Served under ``/dev`` only when ``TEXT_LAB_WEB_DEV`` is set (a sandbox's
``template/dev.env``), never in production. Mockups show a page's design
with sample data before the page is built, to agree on it; they are
removed once the page exists. The demo job runs a job through the whole
runner (queue, progress, cancelling, upload, downloads) without a model.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse
from markupsafe import Markup
from starlette.concurrency import run_in_threadpool

from textlab.common.progress import Progress
from textlab.features.transcription import service
from textlab.ui.web import uploads
from textlab.ui.web.auth import Checked, protected_router
from textlab.ui.web.jobs import (
    Download,
    Job,
    JobContext,
    JobState,
    render_card,
    runner_of,
)
from textlab.ui.web.pages import TOOLS
from textlab.ui.web.pages.transcription import Listening
from textlab.ui.web.render import renderer

router = protected_router(prefix="/dev")

#: Page id of the demo job's page.
DEMO_PAGE = "demo"

#: The mockups, by name (``templates/mockups/<name>.html``): their title.
#: Home and Transcription were agreed and built in phase 2 of the web UI
#: plan; the next pages get theirs here.
MOCKUPS: dict[str, str] = {}

#: Longest demo job, in seconds.
DEMO_MAX_SECONDS = 600


@router.get("")
def overview(request: Request) -> HTMLResponse:
    """List the developer pages.

    Args:
        request: The current request.

    Returns:
        The page.
    """
    return renderer(request).response(
        request, "dev/index.html", page_id="dev", mockups=MOCKUPS
    )


@router.get("/components")
def components(request: Request) -> HTMLResponse:
    """Show every shared component in its variants.

    Args:
        request: The current request.

    Returns:
        The gallery.
    """
    return renderer(request).response(
        request,
        "dev/components.html",
        page_id="dev",
        sample_jobs=sample_jobs(),
        now=time.time(),
    )


@router.get("/mockups/{name}")
def mockup(request: Request, name: str) -> HTMLResponse:
    """Show a page's mockup.

    Args:
        request: The current request.
        name: The mockup's name, a key of :data:`MOCKUPS`.

    Returns:
        The mockup.

    Raises:
        HTTPException: If there is no such mockup (404).
    """
    if name not in MOCKUPS:
        raise HTTPException(status_code=404, detail="No such mockup.")
    return renderer(request).response(
        request,
        f"mockups/{name}.html",
        page_id=f"mockup-{name}",
        tools=TOOLS,
        sample_jobs=sample_jobs(),
        now=time.time(),
    )


@router.get("/demo")
def demo_page(request: Request) -> HTMLResponse:
    """Show the demo job's form and its jobs.

    Args:
        request: The current request.

    Returns:
        The page.
    """
    jobs = runner_of(request).jobs(DEMO_PAGE)
    return renderer(request).response(
        request,
        "dev/demo.html",
        page_id=DEMO_PAGE,
        cards=Markup("".join(render_card(request, job) for job in jobs[::-1])),
        max_seconds=DEMO_MAX_SECONDS,
    )


@router.post("/demo")
async def start_demo(request: Request, session: Checked) -> HTMLResponse:
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
    if not 1 <= seconds <= DEMO_MAX_SECONDS:
        raise uploads.UploadError(
            f"Choose between 1 and {DEMO_MAX_SECONDS} seconds."
        )
    fail = form.get("outcome") == "failed"
    files = uploads.files(form, "file")

    runner = runner_of(request)
    title = f"Demo job, {seconds} s"
    if files:
        title += f", {uploads.display_name(files[0])}"
    job = runner.create(
        title=title,
        page=DEMO_PAGE,
        gpu="llm" if form.get("gpu") else None,
        result_template="dev/demo_result.html",
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
    runner.submit(job, demo_work(seconds, fail, saved))
    return HTMLResponse(render_card(request, job))


def demo_work(seconds: int, fail: bool, inputs: list[Path]):
    """Return the work of a demo job.

    Args:
        seconds: How long the job runs.
        fail: Whether it fails at the end, to show an error.
        inputs: The uploaded file, if any, in the job's inputs.

    Returns:
        The work: reports progress twice a second, can be cancelled, reads
        the upload and offers two downloads.
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
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            lines.append(
                f"Upload: {path.name}, {path.stat().st_size} bytes, "
                f"SHA-256 {digest.hexdigest()}"
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


#: Passages of the sample transcript: speaker, start, text.
SAMPLE_TURNS = (
    (
        1,
        0.0,
        "Guten Morgen zusammen und willkommen zur ersten Sitzung des "
        "Seminars. Heute geht es darum, wie wir Interviews systematisch "
        "auswerten.",
    ),
    (
        2,
        14.0,
        "Danke. Darf ich gleich fragen, ob wir die Transkripte selbst "
        "korrigieren müssen?",
    ),
    (
        1,
        21.0,
        "Ja, unbedingt. Die automatische Transkription ist ein Entwurf: "
        "Namen und Fachbegriffe sollten Sie immer prüfen.",
    ),
    (
        3,
        33.0,
        "Und wie gehen wir mit Dialekt um, zum Beispiel mit Berndeutsch?",
    ),
    (
        1,
        39.0,
        "Dafür gibt es eigene Modelle für Schweizerdeutsch. Die wählen "
        "Sie bei der Sprache aus.",
    ),
)


def sample_listening() -> Listening:
    """Return the result of a transcription with sample data.

    Returns:
        A transcript of five passages by three speakers, with word times;
        its recording does not exist, so the player does not play.
    """
    turns = []
    for number, start, text in SAMPLE_TURNS:
        words = tuple(
            service.TimedWord(
                word, start + 0.4 * index, start + 0.4 * index + 0.3
            )
            for index, word in enumerate(text.split())
        )
        turns.append(
            service.Turn(
                start=start,
                end=words[-1].end or start,
                text=text,
                words=words,
                speaker=f"SPEAKER_{number - 1:02d}",
                speaker_number=number,
            )
        )
    return Listening(
        audio="lecture_intro.mp3",
        duration=760.0,
        levels=tuple(
            (index * 37 + (index // 7) * 11) % 8 + 1 for index in range(120)
        ),
        turns=turns,
        language="German",
        detected=True,
        models=[
            ("Transcription", "large-v3-turbo"),
            ("Word timing", "WhisperX alignment model for German"),
            ("Speakers", service.DIARIZATION_MODEL),
            ("Silence", "not skipped"),
        ],
    )


def sample_jobs() -> dict[str, Job]:
    """Return jobs in every state with sample data, for the mockups.

    Returns:
        The jobs by name: ``waiting``, ``running``, ``done``, ``failed``,
        ``cancelled`` and ``demo_done``. They belong to no runner and have
        no folder.
    """
    now = time.time()
    nowhere = Path("/nonexistent")
    transcript_files = [
        ("lecture_intro_text.txt", "Text", 4_210),
        ("lecture_intro.srt", "Subtitles", 7_388),
        ("lecture_intro.vtt", "Subtitles", 6_912),
        ("lecture_intro_transcription.csv", "Passages", 9_034),
        ("lecture_intro_words.csv", "Words", 51_877),
        ("lecture_intro_ELAN_compatible.tsv", "ELAN", 8_120),
        ("lecture_intro_outputs.zip", "All files", 4_811_502),
    ]
    return {
        "waiting": Job(
            id="w1",
            title="focus_group_sessions.zip (6 recordings)",
            page="transcription",
            folder=nowhere,
            submitted=now - 25,
        ),
        "running": Job(
            id="r1",
            title="interview_03.m4a",
            page="transcription",
            folder=nowhere,
            state=JobState.RUNNING,
            progress=Progress("Aligning the words to the audio", 0.42),
            submitted=now - 80,
            started=now - 72,
        ),
        "done": Job(
            id="d1",
            title="lecture_intro.mp3",
            page="transcription",
            folder=nowhere,
            state=JobState.DONE,
            result_template="transcription/result.html",
            result=sample_listening(),
            submitted=now - 640,
            started=now - 600,
            finished=now - 475,
            downloads=[
                Download(name, label, "text/plain", data=bytes(size))
                for name, label, size in transcript_files
            ],
        ),
        "failed": Job(
            id="f1",
            title="street_noise.ogg",
            page="transcription",
            folder=nowhere,
            state=JobState.FAILED,
            submitted=now - 900,
            started=now - 890,
            finished=now - 884,
            error=(
                "The recording could not be read: it holds no audio "
                "stream that ffmpeg can decode."
            ),
            error_details=(
                "WorkerError: ffmpeg exited with status 1\n"
                "[ogg @ 0x55d0c] Failed to find a valid audio stream\n"
                "street_noise.ogg: Invalid data found when processing input"
            ),
        ),
        "cancelled": Job(
            id="c1",
            title="seminar_recording.wav",
            page="transcription",
            folder=nowhere,
            state=JobState.CANCELLED,
            submitted=now - 1500,
            started=now - 1490,
            finished=now - 1430,
        ),
    }
