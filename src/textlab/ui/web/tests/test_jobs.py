"""The job runner: one job at a time, GPU release, cancelling, files."""

import threading

import pytest

from textlab.common.jobs import WorkerError
from textlab.common.progress import Progress
from textlab.ui.web.jobs import (
    Download,
    GpuBusyError,
    JobRunner,
    JobState,
    content_disposition,
)
from textlab.ui.web.tests.conftest import wait_for


class Gate:
    """Work that waits until the test opens the gate, then returns."""

    def __init__(self, result="done"):
        self.started = threading.Event()
        self.opened = threading.Event()
        self.result = result
        self.inputs_seen = None

    def __call__(self, context):
        self.inputs_seen = sorted(p.name for p in context.inputs.iterdir())
        self.started.set()
        while not self.opened.wait(0.01):
            context.check_cancelled()
        return self.result


def start(runner, work, **options):
    options.setdefault("title", "a job")
    options.setdefault("page", "test")
    return runner.start(work, **options)


def test_jobs_run_one_at_a_time_in_order(runner):
    first, second = Gate("first"), Gate("second")
    job1 = start(runner, first)
    job2 = start(runner, second)
    assert first.started.wait(5)
    assert job1.state is JobState.RUNNING
    assert job2.state is JobState.WAITING
    assert not second.started.is_set()
    assert runner.ahead(job2) == 1

    first.opened.set()
    assert second.started.wait(5)
    second.opened.set()
    wait_for(lambda: job2.state is JobState.DONE)
    assert (job1.result, job2.result) == ("first", "second")
    assert job1.finished <= job2.started


def test_the_gpu_is_released_when_a_job_starts_not_before(runner, released):
    first, second = Gate(), Gate()
    start(runner, first, gpu="transcribe")
    start(runner, second, gpu="ocr")
    assert first.started.wait(5)
    assert released == ["transcribe"]
    first.opened.set()
    assert second.started.wait(5)
    assert released == ["transcribe", "ocr"]
    second.opened.set()


def test_jobs_without_a_gpu_release_nothing(runner, released):
    work = Gate()
    work.opened.set()
    job = start(runner, work)
    wait_for(lambda: job.state is JobState.DONE)
    assert released == []


def test_a_waiting_job_is_dropped_when_cancelled(runner):
    first, second = Gate(), Gate()
    start(runner, first)
    job = runner.create(title="queued", page="test")
    (job.inputs / "upload.wav").write_bytes(b"audio")
    runner.submit(job, second)
    assert first.started.wait(5)

    assert runner.cancel(job.id) is job
    assert job.state is JobState.CANCELLED
    assert not job.inputs.exists()
    first.opened.set()
    wait_for(lambda: runner.running is None)
    assert not second.started.is_set()


def test_a_running_job_stops_when_cancelled(runner):
    work = Gate()
    job = start(runner, work)
    assert work.started.wait(5)
    runner.cancel(job.id)
    assert job.cancel.is_set()
    wait_for(lambda: job.state is JobState.CANCELLED)


def test_work_that_cannot_stop_runs_on(runner):
    work = Gate()
    job = start(runner, work, cancellable=False)
    assert work.started.wait(5)
    runner.cancel(job.id)
    assert not job.cancel.is_set()
    assert job.state is JobState.RUNNING
    work.opened.set()
    wait_for(lambda: job.state is JobState.DONE)


def test_a_failure_is_kept_with_its_details(runner):
    def fail(context):
        raise WorkerError(
            "The worker failed.", "Traceback: line 1", error_type="OSError"
        )

    job = start(runner, fail)
    wait_for(lambda: job.state is JobState.FAILED)
    assert job.error == "The worker failed."
    assert job.error_details == "Traceback: line 1"


def test_other_failures_keep_their_traceback(runner):
    def fail(context):
        raise ValueError("bad value")

    job = start(runner, fail)
    wait_for(lambda: job.state is JobState.FAILED)
    assert job.error == "bad value"
    assert "ValueError: bad value" in job.error_details


def test_progress_and_downloads_reach_the_job(runner):
    reached = threading.Event()

    def work(context):
        context.report(Progress("Halfway", 0.5))
        reached.set()
        path = context.outputs / "result.txt"
        path.write_text("text")
        context.offer(Download("result.txt", "Text", "text/plain", path=path))
        return {"words": 1}

    seen = []
    runner.subscribe(lambda: seen.append(runner.version))
    job = start(runner, work)
    wait_for(lambda: job.state is JobState.DONE)
    assert reached.is_set()
    assert job.result == {"words": 1}
    assert job.download("result.txt").size == 4
    assert job.download("other.txt") is None
    assert seen and seen == sorted(seen)


def test_uploads_are_deleted_when_the_job_ends(runner):
    work = Gate()
    job = runner.create(title="with upload", page="test")
    (job.inputs / "recording.wav").write_bytes(b"audio")
    runner.submit(job, work)
    assert work.started.wait(5)
    assert work.inputs_seen == ["recording.wav"]
    work.opened.set()
    wait_for(lambda: job.state is JobState.DONE)
    assert not job.inputs.exists()
    assert job.outputs.exists()


def test_removing_a_job_deletes_its_files(runner):
    work = Gate()
    job = start(runner, work)
    assert work.started.wait(5)
    assert not runner.remove(job.id)
    work.opened.set()
    wait_for(lambda: job.state is JobState.DONE)
    assert runner.remove(job.id)
    assert runner.get(job.id) is None
    assert not job.folder.exists()
    assert not runner.remove(job.id)


def test_a_discarded_draft_leaves_nothing(runner):
    job = runner.create(title="draft", page="test")
    assert job.folder.exists()
    runner.discard(job)
    assert not job.folder.exists()
    assert runner.jobs() == []


def test_jobs_are_listed_by_page(runner):
    for page in ("a", "b", "a"):
        work = Gate()
        work.opened.set()
        start(runner, work, page=page)
    assert len(runner.jobs("a")) == 2
    assert len(runner.jobs()) == 3


def test_jobs_can_be_left_off_the_lists(runner):
    for listed in (True, False):
        work = Gate()
        work.opened.set()
        start(runner, work, page="a", listed=listed)
    assert [job.listed for job in runner.jobs("a", listed=True)] == [True]
    assert [job.listed for job in runner.jobs("a", listed=False)] == [False]
    assert len(runner.jobs("a")) == 2


def test_a_job_keeps_its_card_template(runner):
    job = runner.create(title="t", page="test")
    assert job.card_template == "jobs/card.html"
    runner.discard(job)
    job = runner.create(
        title="t", page="test", card_template="translation/editor.html"
    )
    assert job.card_template == "translation/editor.html"
    runner.discard(job)


def test_short_gpu_requests_wait_for_no_job(runner):
    with runner.exclusive():
        work = Gate()
        work.opened.set()
        job = start(runner, work)
        assert not work.started.wait(0.3)
        assert job.state is JobState.WAITING
    wait_for(lambda: job.state is JobState.DONE)


def test_short_gpu_requests_are_refused_while_a_job_runs(runner):
    work = Gate()
    job = start(runner, work, title="interview.m4a")
    assert work.started.wait(5)
    with pytest.raises(GpuBusyError, match="interview.m4a") as raised:
        with runner.exclusive():
            pass
    assert raised.value.job is job
    work.opened.set()


def test_closing_cancels_jobs_and_removes_their_folders(released, workspace):
    runner = JobRunner(prepare_gpu=released, workspace=workspace)
    work = Gate()
    job = start(runner, work)
    assert work.started.wait(5)
    runner.close()
    assert job.state is JobState.CANCELLED
    assert not job.folder.exists()
    with pytest.raises(RuntimeError):
        start(runner, Gate())


def test_file_names_with_any_characters_download_under_their_name():
    header = content_disposition('Grüße "neu".txt')
    assert header.startswith('attachment; filename="Gr__e _neu_.txt"')
    assert "filename*=UTF-8''Gr%C3%BC%C3%9Fe%20%22neu%22.txt" in header


def test_work_without_the_gpu_can_start_beside_the_queue(runner, released):
    long = Gate("long")
    queued = start(runner, long, gpu="transcribe")
    assert long.started.wait(5)

    quick = runner.create(title="quick", page="test")
    runner.submit(quick, lambda context: "quick", at_once=True)
    wait_for(lambda: quick.state is JobState.DONE)
    assert quick.result == "quick"
    assert queued.state is JobState.RUNNING
    assert released == ["transcribe"]
    long.opened.set()
    wait_for(lambda: queued.state is JobState.DONE)


def test_work_beside_the_queue_can_be_cancelled(runner):
    work = Gate()
    job = runner.create(title="quick", page="test")
    runner.submit(job, work, at_once=True)
    assert work.started.wait(5)
    assert job.state is JobState.RUNNING
    runner.cancel(job.id)
    wait_for(lambda: job.state is JobState.CANCELLED)


def test_work_with_the_gpu_always_waits_in_the_queue(runner):
    job = runner.create(title="gpu", page="test", gpu="transcribe")
    with pytest.raises(ValueError, match="queue"):
        runner.submit(job, lambda context: None, at_once=True)
    runner.discard(job)


def test_closing_waits_for_work_beside_the_queue(released, workspace):
    runner = JobRunner(prepare_gpu=released, workspace=workspace)
    work = Gate()
    job = runner.create(title="quick", page="test")
    runner.submit(job, work, at_once=True)
    assert work.started.wait(5)
    runner.close()
    assert job.state is JobState.CANCELLED
    assert not job.folder.exists()


def test_files_a_result_uses_can_stay_off_the_list(runner):
    def work(context):
        context.offer(Download("a.txt", "Text", data=b"a"))
        context.offer(Download("a.mp3", "Recording", data=b"b", listed=False))

    job = start(runner, work)
    wait_for(lambda: job.state is JobState.DONE)
    assert [item.name for item in job.listed_downloads] == ["a.txt"]
    assert job.download("a.mp3").data == b"b"


def test_a_job_that_stops_early_keeps_no_files(runner):
    def work(context):
        path = context.outputs / "half.txt"
        path.write_text("half")
        context.offer(Download("half.txt", "Text", path=path))
        raise RuntimeError("broke halfway")

    job = start(runner, work)
    wait_for(lambda: job.state is JobState.FAILED)
    assert job.downloads == []
    assert not job.outputs.exists()


def test_a_jobs_language_model_stays_loaded(runner, released):
    job = start(runner, lambda context: None, gpu="llm", ollama_model="m:7b")
    wait_for(lambda: job.state is JobState.DONE)
    assert released == ["llm:m:7b"]


def test_work_can_switch_the_gpu_between_its_steps(runner, released):
    def work(context):
        context.use_gpu("llm", "m:7b")

    job = start(runner, work, gpu="transcribe")
    wait_for(lambda: job.state is JobState.DONE)
    assert released == ["transcribe", "llm:m:7b"]


def test_work_without_the_gpu_cannot_take_it(runner, released):
    job = start(runner, lambda context: context.use_gpu("llm"))
    wait_for(lambda: job.state is JobState.FAILED)
    assert "not created to use the GPU" in job.error
    assert released == []


def test_a_preview_reaches_the_card_at_most_once_a_second(runner, monkeypatch):
    from textlab.ui.web import jobs

    clock = iter([100.0, 100.4, 101.2, 101.3])
    monkeypatch.setattr(jobs, "_preview_clock", lambda: next(clock))
    seen = []

    def work(context):
        for text in ("a", "ab", "abc", "abcd"):
            context.preview(text)
            seen.append(job.preview)

    job = runner.create(title="t", page="test")
    runner.submit(job, work)
    wait_for(lambda: job.state is JobState.DONE)
    assert seen == ["a", "a", "abc", "abc"]
    assert job.preview == ""


def test_a_long_preview_shows_its_end(runner):
    from textlab.ui.web.jobs import PREVIEW_CHARS, Job

    job = Job(id="x", title="t", page="p", folder=None)
    job.preview = "a" * 10 + "b" * PREVIEW_CHARS
    assert job.preview_tail == "…" + "b" * PREVIEW_CHARS
