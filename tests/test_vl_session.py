"""The resident PaddleOCR-VL worker: the weights load once for a whole batch.

Exercised against a stub worker that speaks the same protocol, so the contract
is tested on any machine -- the real worker needs a GPU and 17 s of loading,
which is the very cost these tests exist to keep paid only once.
"""

import conftest_path  # noqa: F401

import json
import pathlib
import sys
import tempfile

import pytest

from core import auto_ocr

STUB = '''
import argparse, json, os, sys

RESULT = "TEXTLAB_PADDLEVL_RESULT_JSON="
PROGRESS = "TEXTLAB_PADDLEVL_PROGRESS="

parser = argparse.ArgumentParser()
parser.add_argument("images", nargs="*")
parser.add_argument("--extra-labels", default="")
parser.add_argument("--crop-margin", type=float, default=0.06)
parser.add_argument("--serve", action="store_true")
args = parser.parse_args()

def load():
    """Stand in for building the pipeline, and count how often it happens."""
    counter = os.environ["STUB_LOAD_COUNTER"]
    with open(counter, "a") as fh:
        fh.write("load\\n")

def answer(images, serving):
    for index, image in enumerate(images, start=1):
        print(f"{PROGRESS}{index}/{len(images)}", flush=True)
    if serving and os.environ.get("STUB_DIE_ON") in set(images):
        sys.exit(9)
    if os.environ.get("STUB_ERROR_ON") in set(images):
        return {"pages": [], "error": "stub was asked to fail"}
    return {"pages": [{"page_number": i, "image": p} for i, p in enumerate(images, 1)]}

if args.serve:
    load()
    print("TEXTLAB_PADDLEVL_READY=1", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        images = request["images"]
        print(f"{PROGRESS}0/{len(images)}", flush=True)
        print(RESULT + json.dumps(answer(images, True)), flush=True)
else:
    load()
    print(RESULT + json.dumps(answer(args.images, False)), flush=True)
'''


@pytest.fixture
def stub(monkeypatch):
    """A worker stub plus the file it records every model load in."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = pathlib.Path(tmp)
        worker = folder / "stub_worker.py"
        worker.write_text(STUB)
        counter = folder / "loads.txt"
        counter.write_text("")
        monkeypatch.setenv("STUB_LOAD_COUNTER", str(counter))
        yield worker, counter


def _loads(counter) -> int:
    return len(counter.read_text().split())


def test_a_batch_loads_the_weights_once(stub):
    """One resident worker serves every document in the batch."""
    worker, counter = stub
    with auto_ocr.VLWorkerSession(
        backend_python=sys.executable, worker_path=worker
    ) as session:
        first = session.run([pathlib.Path("a.png")])
        second = session.run([pathlib.Path("b.png"), pathlib.Path("c.png")])

    assert [p["image"] for p in first] == ["a.png"]
    assert [p["image"] for p in second] == ["b.png", "c.png"]
    assert _loads(counter) == 1, "the resident worker reloaded its weights"


def test_page_progress_still_arrives_per_document(stub):
    worker, _ = stub
    seen = []
    with auto_ocr.VLWorkerSession(
        backend_python=sys.executable, worker_path=worker
    ) as session:
        session.run([pathlib.Path("a.png"), pathlib.Path("b.png")],
                    on_page=lambda done, total: seen.append((done, total)))
    assert (0, 2) in seen and (2, 2) in seen


def test_a_dead_worker_falls_back_instead_of_failing_the_batch(stub, monkeypatch):
    """A crash costs the file its speed, not its result."""
    worker, counter = stub
    monkeypatch.setenv("STUB_DIE_ON", "boom.png")
    session = auto_ocr.VLWorkerSession(
        backend_python=sys.executable, worker_path=worker
    )
    try:
        pages = auto_ocr.run_vl_worker(
            [pathlib.Path("boom.png")], backend_python=sys.executable,
            worker_path=worker, session=session,
        )
    finally:
        session.close()
    assert [p["image"] for p in pages] == ["boom.png"], "the fallback lost the document"
    assert _loads(counter) == 2, "the fallback worker never started"


def test_the_session_recovers_for_the_next_document(stub, monkeypatch):
    worker, counter = stub
    monkeypatch.setenv("STUB_DIE_ON", "boom.png")
    session = auto_ocr.VLWorkerSession(
        backend_python=sys.executable, worker_path=worker
    )
    try:
        with pytest.raises(RuntimeError):
            session.run([pathlib.Path("boom.png")])
        pages = session.run([pathlib.Path("fine.png")])
    finally:
        session.close()
    assert [p["image"] for p in pages] == ["fine.png"]
    assert _loads(counter) == 2, "the worker was restarted exactly once"


def test_a_reported_error_is_raised_not_returned_as_no_pages(stub, monkeypatch):
    worker, _ = stub
    monkeypatch.setenv("STUB_ERROR_ON", "bad.png")
    session = auto_ocr.VLWorkerSession(
        backend_python=sys.executable, worker_path=worker
    )
    try:
        with pytest.raises(RuntimeError, match="stub was asked to fail"):
            session.run([pathlib.Path("bad.png")])
    finally:
        session.close()


def test_the_one_shot_worker_still_works_without_a_session(stub):
    """The single-document path has no batch to amortise a session over."""
    worker, counter = stub
    pages = auto_ocr.run_vl_worker(
        [pathlib.Path("a.png")], backend_python=sys.executable, worker_path=worker
    )
    assert [p["image"] for p in pages] == ["a.png"]
    assert _loads(counter) == 1


def test_no_images_never_starts_a_worker(stub):
    worker, counter = stub
    assert auto_ocr.run_vl_worker([], backend_python=sys.executable, worker_path=worker) == []
    assert _loads(counter) == 0


def test_close_reaps_a_worker_after_forced_termination():
    class Stream:
        def close(self):
            pass

    class Process:
        def __init__(self):
            self.stdin = Stream()
            self.stdout = Stream()
            self.stderr = Stream()
            self.wait_calls = 0
            self.killed = False
            self.reaped = False

        def poll(self):
            return 0 if self.reaped else None

        def wait(self, timeout):
            self.wait_calls += 1
            if not self.killed:
                raise TimeoutError("worker did not stop")
            self.reaped = True
            return 0

        def kill(self):
            self.killed = True

    process = Process()
    session = auto_ocr.VLWorkerSession()
    session._proc = process

    session.close()

    assert process.killed
    assert process.reaped
    assert process.wait_calls == 2
    assert session._proc is None
