"""Background jobs: one at a time, with progress, cancelling and downloads.

A *job* is one run of a feature: a recording to transcribe, a batch of
scans to read. A page creates it, puts its uploads into the job's
``inputs`` folder and submits it with its *work*, a function that calls the
feature's service. The :class:`JobRunner` runs the work in its own thread,
one job at a time, so jobs never take the GPU from each other: it releases
the GPU for a job's feature (``gpu_manager.prepare_gpu``) when the job
starts, not when it is submitted.

Users can leave the page or reload it and find their jobs again, with
their results, until they remove them or the session ends. A page shows
each of its jobs as a *job card* (``templates/jobs/card.html``) and follows
them over server-sent events (:func:`events`): one stream per page, open
only while a job waits or runs, because browsers open at most six
connections to the portal for all its tabs (see the web UI plan). Every
event carries the changed job cards and the job indicator as out-of-band
fragments, which ``static/js/app.js`` swaps in with htmx.

Each job has a private folder in the workspace: ``inputs/`` for uploads,
removed when the job ends, and ``outputs/`` for files to download, removed
with the job, or when it ends without finishing (failed or cancelled).
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import dataclasses
import enum
import logging
import secrets
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from sse_starlette import EventSourceResponse, ServerSentEvent

from textlab.common.progress import CancelledError, Progress
from textlab.common.storage import Workspace, get_workspace, remove_tree
from textlab.ui.web.auth import Checked, protected_router
from textlab.ui.web.render import renderer

LOGGER = logging.getLogger(__name__)

#: Workspace area that holds the jobs' folders.
WORKSPACE_AREA = "web-jobs"

#: Seconds between keep-alive comments on an event stream. The proxy keeps
#: streams open through quiet periods with them (phase 0 of the plan).
PING_SECONDS = 15

#: Least seconds between two previews of a job's output (``JobContext.
#: preview``), so a model writing many words a second sends few updates.
PREVIEW_SECONDS = 1.0

#: Characters of a preview the card shows: the end of the output so far.
PREVIEW_CHARS = 1500

#: The clock previews are timed with; tests replace it.
_preview_clock = time.monotonic


class JobState(enum.StrEnum):
    """Where a job is in its life."""

    WAITING = "waiting"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def label(self) -> str:
        """The state as shown on job cards, such as ``"Running"``."""
        return self.value.capitalize()

    @property
    def active(self) -> bool:
        """Whether the job still waits or runs."""
        return self in (JobState.WAITING, JobState.RUNNING)


@dataclasses.dataclass(frozen=True)
class Download:
    """A file a finished job offers for download.

    Attributes:
        name: The file name the browser saves it under.
        label: Text of the download button, such as ``"Subtitles (.srt)"``.
        media_type: The file's media type.
        path: The file on disk, usually in the job's ``outputs`` folder.
        data: The file's content, for small files made in memory.
        listed: Whether the card lists it among the downloads. Files the
            result uses otherwise, such as the recording an audio player
            plays, are not listed but can be fetched the same way.
    """

    name: str
    label: str
    media_type: str = "application/octet-stream"
    path: Path | None = None
    data: bytes | None = dataclasses.field(default=None, repr=False)
    listed: bool = True

    @property
    def size(self) -> int | None:
        """The file's size in bytes, if known."""
        if self.data is not None:
            return len(self.data)
        if self.path is not None and self.path.exists():
            return self.path.stat().st_size
        return None


@dataclasses.dataclass(eq=False)
class Job:
    """One run of a feature, from submission until the user removes it.

    The runner changes a job only while holding its lock; pages read it.

    Attributes:
        id: Random id, used in links.
        title: What the job works on, such as the uploaded file's name.
        page: Id of the page that submitted the job; its card is shown
            there.
        folder: The job's private folder in the workspace.
        gpu: The ``gpu_manager`` owner of the job's feature, or ``None``
            for work that does not use the GPU.
        ollama_model: The Ollama model the job uses, which stays loaded
            when the GPU is released for it; ``None`` for other jobs.
        cancellable: Whether a running job stops when cancelled. Waiting
            jobs can always be cancelled.
        result_template: Template that shows a finished job's result on
            its card, or ``None`` for the downloads only.
        card_template: Template of the job's card. Pages whose jobs are
            not records in a ledger, such as a translation shown in an
            editor's pane, give their own; it must keep the card's id,
            ``job-<id>``, and ``data-job-active`` while the job is active.
        listed: Whether the job is listed in the job indicator and on
            Home. Short requests a page shows in place, such as a
            translation in the editor, are not.
        state: Where the job is in its life.
        progress: The work's latest progress update.
        preview: The work's output so far, such as a summary being
            written, shown while it runs.
        submitted: When the job was submitted (seconds since the epoch).
        started: When the work started.
        finished: When the work ended.
        result: What the work returned.
        error: Why the job failed, for the user.
        error_details: The traceback or other details, for support.
        downloads: The files the job offers.
        version: Increases with every change, so streams send only what
            changed.
        cancel: Set to ask running work to stop.
    """

    id: str
    title: str
    page: str
    folder: Path
    gpu: str | None = None
    ollama_model: str | None = None
    cancellable: bool = True
    result_template: str | None = None
    card_template: str = "jobs/card.html"
    listed: bool = True
    state: JobState = JobState.WAITING
    progress: Progress | None = None
    preview: str = ""
    submitted: float | None = None
    started: float | None = None
    finished: float | None = None
    result: Any = None
    error: str = ""
    error_details: str = ""
    downloads: list[Download] = dataclasses.field(default_factory=list)
    version: int = 0
    cancel: threading.Event = dataclasses.field(
        default_factory=threading.Event, repr=False
    )

    @property
    def inputs(self) -> Path:
        """Folder for the job's uploads, removed when the job ends."""
        return self.folder / "inputs"

    @property
    def outputs(self) -> Path:
        """Folder for files to download, removed with the job."""
        return self.folder / "outputs"

    @property
    def active(self) -> bool:
        """Whether the job still waits or runs."""
        return self.state.active

    @property
    def cancelling(self) -> bool:
        """Whether the job runs on although it was asked to stop."""
        return self.state is JobState.RUNNING and self.cancel.is_set()

    def elapsed(self, now: float | None = None) -> float | None:
        """Return the seconds the work has run, or ran in all.

        Args:
            now: The current time; defaults to :func:`time.time`.

        Returns:
            The seconds, or ``None`` if the work has not started.
        """
        if self.started is None:
            return None
        end = self.finished or (time.time() if now is None else now)
        return end - self.started

    @property
    def preview_tail(self) -> str:
        """The end of the preview the card shows, at most PREVIEW_CHARS."""
        if len(self.preview) <= PREVIEW_CHARS:
            return self.preview
        return "\u2026" + self.preview[-PREVIEW_CHARS:]

    @property
    def listed_downloads(self) -> list[Download]:
        """The downloads the card lists."""
        return [item for item in self.downloads if item.listed]

    def download(self, name: str) -> Download | None:
        """Return the download called ``name``, if the job offers one.

        Args:
            name: The download's file name.

        Returns:
            The download, or ``None``.
        """
        return next(
            (item for item in self.downloads if item.name == name), None
        )


class JobContext:
    """What a job's work receives: its folders, progress and cancelling."""

    def __init__(self, runner: JobRunner, job: Job):
        """Wrap a job for its work.

        Args:
            runner: The runner that runs the job.
            job: The job.
        """
        self._runner = runner
        self._job = job
        self._previewed = float("-inf")

    @property
    def inputs(self) -> Path:
        """The job's uploads."""
        return self._job.inputs

    @property
    def outputs(self) -> Path:
        """The folder for files to download."""
        return self._job.outputs

    @property
    def cancel(self) -> threading.Event:
        """Set when the user cancels; pass it to the service."""
        return self._job.cancel

    def report(self, progress: Progress) -> None:
        """Show a progress update on the job's card.

        Pass this as the service's ``on_progress`` callback.

        Args:
            progress: The update.
        """
        self._runner._update(self._job, progress=progress)

    def preview(self, text: str) -> None:
        """Show the work's output so far on the job's card.

        For output that grows while the work runs, such as a summary a
        model writes. Updates reach the card at most every
        :data:`PREVIEW_SECONDS`; the result replaces the preview when the
        job ends.

        Args:
            text: The whole output so far.
        """
        now = _preview_clock()
        if now - self._previewed < PREVIEW_SECONDS:
            return
        self._previewed = now
        self._runner._update(self._job, preview=text)

    def use_gpu(self, feature: str, ollama_model: str | None = None) -> None:
        """Release the GPU for another step of the work.

        A job's GPU is released for its own feature when it starts. Work
        with two steps on the GPU, such as transcribing and then
        summarizing, releases it again before the second. The job holds
        the GPU throughout, so nothing else runs in between.

        Args:
            feature: The ``gpu_manager`` owner of the next step.
            ollama_model: The Ollama model it uses, which stays loaded.

        Raises:
            RuntimeError: If the job was not created to use the GPU.
        """
        if not self._job.gpu:
            raise RuntimeError("This job was not created to use the GPU.")
        self._runner._prepare_gpu(feature, ollama_model)

    def check_cancelled(self) -> None:
        """Stop the work here if the user cancelled the job.

        Raises:
            CancelledError: If the job was cancelled.
        """
        if self._job.cancel.is_set():
            raise CancelledError("The job was cancelled.")

    def offer(self, download: Download) -> None:
        """Offer a file for download on the job's card.

        Args:
            download: The file.
        """
        self._runner._update(
            self._job, downloads=[*self._job.downloads, download]
        )


#: A job's work: gets the job's context and returns its result.
Work = Callable[[JobContext], Any]


class GpuBusyError(RuntimeError):
    """A short GPU request came while a job holds the GPU.

    Attributes:
        job: The running job, or ``None`` if another short request holds
            it.
    """

    def __init__(self, job: Job | None):
        """Name the job that holds the GPU.

        Args:
            job: The running job.
        """
        name = f'"{job.title}"' if job else "another request"
        super().__init__(f"The GPU is busy with {name}.")
        self.job = job


def _prepare_gpu(feature: str, ollama_model: str | None = None) -> None:
    """Release the GPU for a feature (``gpu_manager.prepare_gpu``)."""
    from textlab.common import gpu_manager

    gpu_manager.prepare_gpu(feature, ollama_model=ollama_model)


class JobRunner:
    """Runs submitted jobs one after the other in a thread of its own.

    Short GPU requests outside jobs, such as a chat message, use
    :meth:`exclusive`, which takes the GPU only while no job runs.
    """

    def __init__(
        self,
        *,
        prepare_gpu: Callable[[str, str | None], Any] = _prepare_gpu,
        workspace: Workspace | None = None,
    ):
        """Set up an empty runner; its thread starts with the first job.

        Args:
            prepare_gpu: Releases the GPU for a feature when a job of it
                starts, keeping an Ollama model loaded if one is named;
                tests pass a fake.
            workspace: Where the jobs' folders go; defaults to the session's
                workspace.
        """
        self._prepare_gpu = prepare_gpu
        self._workspace = workspace
        self._lock = threading.Lock()
        self._wakeup = threading.Condition(self._lock)
        self._gpu = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._drafts: dict[str, Job] = {}
        self._work: dict[str, Work] = {}
        self._queue: collections.deque[str] = collections.deque()
        self._running: Job | None = None
        self._listeners: set[Callable[[], None]] = set()
        self._thread: threading.Thread | None = None
        self._beside: list[threading.Thread] = []
        self._closed = False
        self.version = 0

    # -- Submitting ---------------------------------------------------------

    def create(
        self,
        *,
        title: str,
        page: str,
        gpu: str | None = None,
        ollama_model: str | None = None,
        cancellable: bool = True,
        result_template: str | None = None,
        card_template: str | None = None,
        listed: bool = True,
    ) -> Job:
        """Create a job and its folders, to be submitted with :meth:`submit`.

        Put the job's uploads into ``job.inputs`` before submitting it; if
        that fails, call :meth:`discard`.

        Args:
            title: What the job works on, shown on its card.
            page: Id of the page the job belongs to.
            gpu: The ``gpu_manager`` owner of the job's feature, or
                ``None``.
            ollama_model: The Ollama model the job uses, if any; it stays
                loaded when the GPU is released for the job.
            cancellable: Whether the work stops when the job is cancelled.
            result_template: Template for the finished job's result.
            card_template: Template of the job's card; the ledger record
                (``jobs/card.html``) by default.
            listed: Whether the job indicator and Home list the job.

        Returns:
            The job, not yet listed or queued.
        """
        workspace = self._workspace or get_workspace()
        job_id = secrets.token_hex(6)
        folder = workspace.dir(WORKSPACE_AREA, job_id)
        workspace.dir(WORKSPACE_AREA, job_id, "inputs")
        workspace.dir(WORKSPACE_AREA, job_id, "outputs")
        job = Job(
            id=job_id,
            title=title,
            page=page,
            folder=folder,
            gpu=gpu,
            ollama_model=ollama_model,
            cancellable=cancellable,
            result_template=result_template,
            card_template=card_template or "jobs/card.html",
            listed=listed,
        )
        with self._lock:
            self._drafts[job.id] = job
        return job

    def submit(self, job: Job, work: Work, *, at_once: bool = False) -> Job:
        """Queue a job created with :meth:`create`, or start it at once.

        Args:
            job: The job.
            work: The work to run for it.
            at_once: Start the work now in a thread of its own, beside the
                queue, instead of after the jobs before it. Only for short
                work without the GPU, such as preparing a recording to
                review, which should not wait for an hour-long
                transcription.

        Returns:
            The job, now waiting (or running, with ``at_once``).

        Raises:
            RuntimeError: If the runner is closed.
            KeyError: If the job was not created by this runner or was
                submitted already.
            ValueError: If ``at_once`` is set for a job that uses the GPU.
        """
        if at_once and job.gpu:
            raise ValueError("Jobs that use the GPU wait in the queue.")
        beside = None
        with self._wakeup:
            if self._closed:
                raise RuntimeError("The job runner is closed.")
            self._drafts.pop(job.id)
            job.submitted = time.time()
            self._jobs[job.id] = job
            if at_once:
                job.state = JobState.RUNNING
                job.started = job.submitted
                beside = threading.Thread(
                    target=self._run,
                    args=(job, work),
                    name=f"textlab-job-{job.id}",
                    daemon=True,
                )
                self._beside = [
                    thread for thread in self._beside if thread.is_alive()
                ]
                self._beside.append(beside)
            else:
                self._work[job.id] = work
                self._queue.append(job.id)
                if self._thread is None:
                    self._thread = threading.Thread(
                        target=self._loop, name="textlab-jobs", daemon=True
                    )
                    self._thread.start()
                self._wakeup.notify()
            self._touch(job)
        if beside is not None:
            beside.start()
        self._notify()
        return job

    def start(self, work: Work, **job_options: Any) -> Job:
        """Create and submit a job that has no uploads.

        Args:
            work: The work to run.
            **job_options: The options of :meth:`create`.

        Returns:
            The job, now waiting.
        """
        return self.submit(self.create(**job_options), work)

    def discard(self, job: Job) -> None:
        """Drop a job that was created but not submitted, with its folder.

        Args:
            job: The job.
        """
        with self._lock:
            self._drafts.pop(job.id, None)
        remove_tree(job.folder)

    # -- Reading ------------------------------------------------------------

    def get(self, job_id: str) -> Job | None:
        """Return a submitted job.

        Args:
            job_id: The job's id.

        Returns:
            The job, or ``None`` if there is none with that id.
        """
        with self._lock:
            return self._jobs.get(job_id)

    def draft(self, job_id: str) -> Job | None:
        """Return a job that was created but not submitted yet.

        A page can create a job when the user uploads a file, keep the
        upload in its ``inputs`` while the user chooses settings, and
        submit it when they start the work.

        Args:
            job_id: The job's id.

        Returns:
            The job, or ``None`` if no draft has that id.
        """
        with self._lock:
            return self._drafts.get(job_id)

    def jobs(
        self, page: str | None = None, *, listed: bool | None = None
    ) -> list[Job]:
        """Return the submitted jobs, oldest first.

        Args:
            page: Only the jobs of this page; all if ``None``.
            listed: Only the listed jobs (``True``), only the others
                (``False``), or all (``None``).

        Returns:
            The jobs.
        """
        with self._lock:
            return [
                job
                for job in self._jobs.values()
                if (page is None or job.page == page)
                and (listed is None or job.listed == listed)
            ]

    def ahead(self, job: Job) -> int:
        """Return how many jobs run or wait before a waiting job.

        Args:
            job: The job.

        Returns:
            The number of jobs ahead of it; 0 if it does not wait.
        """
        with self._lock:
            if job.id not in self._queue:
                return 0
            return self._queue.index(job.id) + (self._running is not None)

    @property
    def running(self) -> Job | None:
        """The job whose work runs now, if any."""
        with self._lock:
            return self._running

    # -- Changing -----------------------------------------------------------

    def cancel(self, job_id: str) -> Job | None:
        """Cancel a job: drop it if it waits, ask its work to stop if it runs.

        Work that cannot be cancelled runs on until it ends.

        Args:
            job_id: The job's id.

        Returns:
            The job, or ``None`` if there is none with that id.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or not job.active:
                return job
            if job.state is JobState.WAITING:
                self._queue.remove(job.id)
                self._work.pop(job.id, None)
                job.state = JobState.CANCELLED
                job.finished = time.time()
                remove_tree(job.inputs)
            elif job.cancellable:
                job.cancel.set()
            self._touch(job)
        self._notify()
        return job

    def remove(self, job_id: str) -> bool:
        """Remove a finished job, with its result and files.

        Args:
            job_id: The job's id.

        Returns:
            ``True`` if it was removed; ``False`` if there is no such job or
            it still waits or runs.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.active:
                return False
            del self._jobs[job_id]
            self.version += 1
        remove_tree(job.folder)
        self._notify()
        return True

    @contextlib.contextmanager
    def exclusive(self) -> Iterator[None]:
        """Hold the GPU for a short request outside a job.

        Use it for requests that need the GPU for seconds, such as a chat
        message. Inside, call ``gpu_manager.prepare_gpu`` for the feature as
        usual. Jobs wait until the block ends.

        Yields:
            Nothing; the GPU is held until the block ends.

        Raises:
            GpuBusyError: If a job or another request holds the GPU.
        """
        if not self._gpu.acquire(blocking=False):
            raise GpuBusyError(self.running)
        try:
            yield
        finally:
            self._gpu.release()

    def close(self) -> None:
        """Stop the runner: cancel the jobs and remove their folders.

        Called when the server stops. Waits a few seconds for running work
        to notice that it was cancelled.
        """
        with self._wakeup:
            self._closed = True
            jobs = [*self._jobs.values(), *self._drafts.values()]
            for job in jobs:
                job.cancel.set()
            self._wakeup.notify_all()
            threads = [self._thread, *self._beside]
        deadline = time.monotonic() + 10
        for thread in threads:
            if thread is not None:
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
        for job in jobs:
            remove_tree(job.folder)

    # -- Following changes --------------------------------------------------

    def subscribe(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Call ``listener`` after every change of a job.

        It is called in the thread that made the change, so it must be
        quick and thread-safe, such as setting an event.

        Args:
            listener: The function to call.

        Returns:
            A function that ends the subscription.
        """
        with self._lock:
            self._listeners.add(listener)

        def unsubscribe() -> None:
            with self._lock:
                self._listeners.discard(listener)

        return unsubscribe

    # -- Internals ----------------------------------------------------------

    def _touch(self, job: Job) -> None:
        """Count a change of ``job``; the caller holds the lock."""
        job.version += 1
        self.version += 1

    def _notify(self) -> None:
        """Tell the listeners that something changed."""
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener()
            except Exception:
                LOGGER.exception("A job listener failed")

    def _update(self, job: Job, **changes: Any) -> None:
        """Change a job's fields and tell the listeners."""
        with self._lock:
            for name, value in changes.items():
                setattr(job, name, value)
            self._touch(job)
        self._notify()

    def _next(self) -> tuple[Job, Work] | None:
        """Take the next waiting job and the GPU for it.

        Returns:
            The job, now running, and its work; ``None`` once the runner is
            closed. The caller releases the GPU when the work ends.
        """
        while True:
            with self._wakeup:
                while not self._queue and not self._closed:
                    self._wakeup.wait()
                if self._closed:
                    return None
            # Short GPU requests (exclusive) end before the job starts.
            self._gpu.acquire()
            with self._lock:
                if self._closed or not self._queue:
                    # Closed, or the job was cancelled while waiting.
                    self._gpu.release()
                    if self._closed:
                        return None
                    continue
                job = self._jobs[self._queue.popleft()]
                work = self._work.pop(job.id)
                self._running = job
                job.state = JobState.RUNNING
                job.started = time.time()
                self._touch(job)
                return job, work

    def _loop(self) -> None:
        """Run the queued jobs one after the other, until closed."""
        while (taken := self._next()) is not None:
            job, work = taken
            try:
                self._notify()
                self._run(job, work)
            finally:
                with self._lock:
                    self._running = None
                self._gpu.release()
                self._notify()

    def _run(self, job: Job, work: Work) -> None:
        """Run one job's work and record how it ended."""
        changes: dict[str, Any] = {"preview": ""}
        try:
            if job.gpu:
                self._prepare_gpu(job.gpu, job.ollama_model)
            changes["result"] = work(JobContext(self, job))
            changes["state"] = JobState.DONE
        except CancelledError:
            changes["state"] = JobState.CANCELLED
        except Exception as error:
            LOGGER.exception("Job %s (%s) failed", job.id, job.page)
            changes["state"] = JobState.FAILED
            changes["error"] = str(error) or type(error).__name__
            changes["error_details"] = getattr(
                error, "details", ""
            ) or traceback.format_exc(limit=20)
        finally:
            remove_tree(job.inputs)
        if changes["state"] is not JobState.DONE:
            # Files written before the work stopped are not offered.
            remove_tree(job.outputs)
            changes["downloads"] = []
        changes["finished"] = time.time()
        self._update(job, **changes)


# ---------------------------------------------------------------------------
# Rendering and routes
# ---------------------------------------------------------------------------


def runner_of(request: Request) -> JobRunner:
    """Return the job runner of the app that handles ``request``.

    Args:
        request: The current request.

    Returns:
        The app's runner.
    """
    return request.app.state.runner


def render_card(request: Request, job: Job, *, oob: bool = False) -> str:
    """Render a job's card, with its own card template.

    Args:
        request: The current request.
        job: The job.
        oob: Whether to mark the card for an out-of-band swap, as in
            event streams.

    Returns:
        The card's HTML. The template gets ``job``, ``ahead`` (the jobs
        before a waiting one), ``running`` (the job that runs now, if any),
        ``now`` and ``oob``.
    """
    runner = runner_of(request)
    return renderer(request).render(
        request,
        job.card_template,
        job=job,
        ahead=runner.ahead(job),
        running=runner.running,
        now=time.time(),
        oob=oob,
    )


def render_indicator(
    request: Request, *, oob: bool = False, page: str = ""
) -> str:
    """Render the job indicator's count and list.

    Args:
        request: The current request.
        oob: Whether to mark them for an out-of-band swap.
        page: Id of the page they go to; Home also gets its list of
            recent jobs.

    Returns:
        The HTML of the parts.
    """
    return renderer(request).render(
        request,
        "jobs/indicator_parts.html",
        oob=oob,
        page=page,
        now=time.time(),
    )


def download_url(request_root: str, job: Job, download: Download) -> str:
    """Return the link of a job's download.

    Args:
        request_root: The base path, from :func:`render.root`.
        job: The job.
        download: One of its downloads.

    Returns:
        The link.
    """
    return f"{request_root}/jobs/{job.id}/files/{quote(download.name)}"


def content_disposition(name: str) -> str:
    """Return a ``Content-Disposition`` header that saves under ``name``.

    Args:
        name: The file name, with any characters.

    Returns:
        The header, with an ASCII fallback and the exact name in
        ``filename*`` (RFC 6266).
    """
    fallback = "".join(
        char
        if char.isascii() and char.isprintable() and char not in '"\\'
        else "_"
        for char in name
    )
    return (
        f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(name)}"
    )


router = protected_router()


def _job_or_404(request: Request, job_id: str) -> Job:
    """Return a job, or answer 404."""
    job = runner_of(request).get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail="This job no longer exists. It may have been removed.",
        )
    return job


@router.get("/jobs/{job_id}")
def job_card(request: Request, job_id: str) -> HTMLResponse:
    """Return a job's card.

    Args:
        request: The current request.
        job_id: The job's id.

    Returns:
        The card.
    """
    return HTMLResponse(render_card(request, _job_or_404(request, job_id)))


@router.post("/jobs/{job_id}/cancel")
def cancel_job(
    request: Request, job_id: str, session: Checked
) -> HTMLResponse:
    """Cancel a job and return its card.

    Args:
        request: The current request.
        job_id: The job's id.
        session: The session, with its CSRF token checked.

    Returns:
        The job's card, and the job indicator out of band.
    """
    job = runner_of(request).cancel(job_id) or _job_or_404(request, job_id)
    return HTMLResponse(
        render_card(request, job) + render_indicator(request, oob=True)
    )


@router.post("/jobs/{job_id}/remove")
def remove_job(
    request: Request, job_id: str, session: Checked
) -> HTMLResponse:
    """Remove a finished job with its files.

    Args:
        request: The current request.
        job_id: The job's id.
        session: The session, with its CSRF token checked.

    Returns:
        Nothing in place of the card (htmx removes it), and the job
        indicator out of band.

    Raises:
        HTTPException: If the job still waits or runs (409).
    """
    job = _job_or_404(request, job_id)
    if not runner_of(request).remove(job.id):
        raise HTTPException(
            status_code=409,
            detail="Cancel the job before removing it.",
        )
    return HTMLResponse(render_indicator(request, oob=True))


@router.get("/jobs/{job_id}/files/{name}")
def download(request: Request, job_id: str, name: str) -> Response:
    """Send a file a finished job offers.

    Args:
        request: The current request.
        job_id: The job's id.
        name: The download's file name.

    Returns:
        The file, as an attachment.

    Raises:
        HTTPException: If the job or the file does not exist (404).
    """
    item = _job_or_404(request, job_id).download(name)
    if item is None or (item.path is None and item.data is None):
        raise HTTPException(status_code=404, detail="No such file.")
    headers = {"Content-Disposition": content_disposition(item.name)}
    if item.path is not None:
        return FileResponse(
            item.path, media_type=item.media_type, headers=headers
        )
    return Response(item.data, media_type=item.media_type, headers=headers)


@router.get("/events")
async def events(request: Request, page: str = "") -> EventSourceResponse:
    """Stream changes of the jobs to a page, until no job waits or runs.

    Each ``update`` event holds out-of-band fragments: the cards of the
    page's jobs that changed, an empty element in place of removed ones,
    and the job indicator (on Home also the list of recent jobs). The
    ``idle`` event tells the page to close the stream; it opens a new one
    when it gets an active job again.

    Args:
        request: The current request.
        page: Id of the page whose job cards to send.

    Returns:
        The event stream.
    """
    runner = runner_of(request)

    async def stream():
        loop = asyncio.get_running_loop()
        changed = asyncio.Event()
        unsubscribe = runner.subscribe(
            lambda: loop.call_soon_threadsafe(changed.set)
        )
        sent: dict[str, int] = {}
        indicator_version = None
        try:
            while True:
                changed.clear()
                jobs = runner.jobs()
                parts = []
                for job in jobs:
                    # The version is read before rendering, so a change
                    # during rendering is sent again.
                    version = job.version
                    if job.page == page and sent.get(job.id) != version:
                        sent[job.id] = version
                        parts.append(render_card(request, job, oob=True))
                listed = {job.id for job in jobs}
                for job_id in [key for key in sent if key not in listed]:
                    del sent[job_id]
                    parts.append(
                        f'<div id="job-{job_id}" hx-swap-oob="delete"></div>'
                    )
                version = runner.version
                if version != indicator_version:
                    indicator_version = version
                    parts.append(
                        render_indicator(request, oob=True, page=page)
                    )
                if parts:
                    yield ServerSentEvent(data="".join(parts), event="update")
                current = runner.version == indicator_version and all(
                    sent.get(job.id) == job.version
                    for job in jobs
                    if job.page == page
                )
                if not current:
                    continue
                if not any(job.active for job in jobs):
                    yield ServerSentEvent(data="idle", event="idle")
                    return
                await changed.wait()
        finally:
            unsubscribe()

    return EventSourceResponse(
        stream(), ping=PING_SECONDS, headers={"Cache-Control": "no-store"}
    )
