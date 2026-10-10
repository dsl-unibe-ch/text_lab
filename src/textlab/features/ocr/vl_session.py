"""The PaddleOCR-VL worker: a separate process in its own environment.

PaddleOCR-VL needs dependencies that conflict with the app's, so it runs in
the image's ``paddle_vl_backend`` environment as
``python -m textlab.features.ocr.paddle_vl_worker``. This module starts it,
sends it page images and reads its answers: a :class:`VLWorkerSession`
keeps one worker running for a whole batch, :func:`run_vl_worker` runs one
request, on a session or on a worker started for it alone.

The worker reports on standard output with marker lines: one per finished
page, one when it is ready (serving mode) and one with the result as JSON.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import weakref
from collections.abc import Callable
from pathlib import Path

from textlab.common import container
from textlab.common.jobs import worker_environment
from textlab.common.ollama import release_models

#: The worker module, run with ``python -m`` in the PaddleOCR-VL environment.
WORKER_MODULE = "textlab.features.ocr.paddle_vl_worker"

#: Marker lines; they mirror the constants in ``paddle_vl_worker``, which
#: runs in another environment and cannot be imported here.
RESULT_MARKER = "TEXTLAB_PADDLEVL_RESULT_JSON="
PROGRESS_MARKER = "TEXTLAB_PADDLEVL_PROGRESS="
READY_MARKER = "TEXTLAB_PADDLEVL_READY="


def _default_backend_python() -> str:
    """Return the PaddleOCR-VL environment's interpreter."""
    # The VL parser has its own environment so the PaddleOCR 2 engine stays
    # pinned; PADDLE_BACKEND_PYTHON is honored only if the VL one is unset.
    return container.env_python(
        container.PADDLE_VL_ENV,
        "PADDLE_VL_BACKEND_PYTHON",
        "PADDLE_BACKEND_PYTHON",
    )


def _worker_command(
    backend_python: str, worker_path: Path | None
) -> list[str]:
    """Return the command that starts the worker.

    ``worker_path`` runs a script instead of the worker module; tests use it
    to start stub workers that speak the same protocol.
    """
    if worker_path is not None:
        return [backend_python, str(worker_path)]
    return [backend_python, "-m", WORKER_MODULE]


#: Kill the worker after this long with no output at all. A stall budget, not a
#: total one: a 40-page scan legitimately runs for half an hour, but the worker
#: reports every finished page, so silence this long means it is wedged.
VL_STALL_TIMEOUT = float(os.environ.get("TEXTLAB_VL_STALL_TIMEOUT", "900"))


def _worker_env(backend_python: str) -> dict:
    """Return the worker's environment, with Paddle's model checks off."""
    env = worker_environment(backend_python)
    env.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
    env.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    return env


def _free_gpu_for_worker() -> None:
    """Unload Ollama models before the worker starts.

    The worker allocates about 8.4 GiB, which does not fit beside a resident
    20 GiB vision model on a 23 GiB card, and Ollama cannot account for a
    consumer outside Ollama.
    """
    try:
        evicted = release_models()
        if evicted:
            print(
                "[ocr] freed GPU for the OCR worker: " + ", ".join(evicted),
                flush=True,
            )
    except Exception:
        pass


def _pages_from_result(stdout: str, stderr: str) -> list[dict]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            payload = json.loads(line[len(RESULT_MARKER) :])
            if payload.get("error"):
                raise RuntimeError(
                    f"PaddleOCR-VL backend failed: {payload['error']}"
                )
            return payload.get("pages", [])
    raise RuntimeError(
        "PaddleOCR-VL backend did not return JSON.\n"
        f"stdout:\n{stdout[-4000:]}\n\nstderr:\n{stderr[-4000:]}"
    )


#: Open worker sessions, so another feature can stop them to free the GPU.
_LIVE_SESSIONS: weakref.WeakSet[VLWorkerSession] = weakref.WeakSet()


class VLWorkerSession:
    """A PaddleOCR-VL worker kept alive across many documents.

    Loading the weights costs ~17 s and the first prediction another ~7 s of
    warm-up. A batch that starts a worker per file pays that for every file,
    which on short questionnaires is most of the wall clock; one session pays
    it once. Not thread-safe: one request at a time, which is also all a single
    GPU would do with them.
    """

    def __init__(
        self,
        *,
        backend_python: str | None = None,
        worker_path: Path | None = None,
        stall_timeout: float | None = None,
    ):
        """Set up a session; the worker starts with the first request.

        Args:
            backend_python: The PaddleOCR-VL environment's interpreter, if not
                the image's.
            worker_path: A worker script to run instead of the worker module;
                for tests.
            stall_timeout: Seconds without output before the worker counts as
                stuck; defaults to ``VL_STALL_TIMEOUT``.
        """
        self.backend_python = backend_python or _default_backend_python()
        self.worker_path = worker_path
        self.stall_timeout = (
            stall_timeout if stall_timeout is not None else VL_STALL_TIMEOUT
        )
        self._proc = None
        self._lines = None
        self._stderr: list[str] = []
        self._started_at = 0.0
        self.failed = False
        _LIVE_SESSIONS.add(self)
        #: Documents this session has answered, so a log reader can tell a
        #: resident worker from one that keeps dying and being restarted.
        self.documents = 0

    # -- lifecycle ------------------------------------------------------------
    def _start(self):
        import queue
        import threading

        _free_gpu_for_worker()
        self._started_at = time.monotonic()
        self._proc = subprocess.Popen(
            [
                *_worker_command(self.backend_python, self.worker_path),
                "--serve",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_worker_env(self.backend_python),
            bufsize=1,
        )
        self._lines = queue.Queue()

        def drain_stdout(stream, sink):
            for line in stream:
                sink.put(line)
            sink.put(None)  # the worker is gone; unblock whoever is waiting

        def drain_stderr(stream, sink):
            for line in stream:
                sink.append(line)
                del sink[:-400]  # a tail is enough to explain a crash

        for target, args in (
            (drain_stdout, (self._proc.stdout, self._lines)),
            (drain_stderr, (self._proc.stderr, self._stderr)),
        ):
            threading.Thread(target=target, args=args, daemon=True).start()

    def close(self):
        """Stop the worker, killing it if it does not exit in time."""
        if self._proc is None:
            return
        proc = self._proc
        self._proc = None
        try:
            if proc.poll() is None:
                proc.stdin.close()
                proc.wait(timeout=10)
        except Exception:
            pass
        finally:
            if proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except Exception:
                    pass
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    stream.close()
                except Exception:
                    pass

    def __enter__(self):
        """Use the session in a ``with`` block."""
        return self

    def __exit__(self, *exc):
        """Stop the worker."""
        self.close()

    # -- requests -------------------------------------------------------------
    def run(
        self,
        image_paths: list[Path],
        *,
        extra_labels: str = "",
        on_page: Callable[[int, int], None] | None = None,
    ) -> list[dict]:
        """Recognise one document's pages on the resident worker."""
        import queue

        if self._proc is None or self._proc.poll() is not None:
            self.close()
            self._start()

        started = time.monotonic()
        request = {"images": [str(p) for p in image_paths]}
        if extra_labels:
            request["extra_labels"] = extra_labels
        try:
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()
        except Exception as exc:
            self.failed = True
            raise RuntimeError(
                f"PaddleOCR-VL worker is not accepting work: {exc}"
            ) from exc

        out_lines: list[str] = []
        while True:
            try:
                line = self._lines.get(timeout=self.stall_timeout)
            except queue.Empty:
                self.failed = True
                self.close()
                raise RuntimeError(
                    f"PaddleOCR-VL backend produced no output for "
                    f"{self.stall_timeout:.0f}s and was stopped. Re-run the "
                    "document; if it happens again, the page count or GPU "
                    "memory may be the cause.\n"
                    f"stderr:\n{''.join(self._stderr)[-4000:]}"
                ) from None
            if line is None:
                self.failed = True
                self.close()
                raise RuntimeError(
                    "PaddleOCR-VL worker exited mid-document.\n"
                    f"stdout:\n{''.join(out_lines)[-4000:]}\n\n"
                    f"stderr:\n{''.join(self._stderr)[-4000:]}"
                )
            out_lines.append(line)
            if line.startswith(READY_MARKER):
                print(
                    f"[ocr] VL worker ready in "
                    f"{time.monotonic() - self._started_at:.1f}s",
                    flush=True,
                )
            if line.startswith(PROGRESS_MARKER):
                _report_progress(line, on_page)
            if line.startswith(RESULT_MARKER):
                pages = _pages_from_result(
                    "".join(out_lines), "".join(self._stderr)
                )
                self.documents += 1
                print(
                    f"[ocr] recognised {len(image_paths)} page(s) in "
                    f"{time.monotonic() - started:.1f}s "
                    f"(resident worker, document {self.documents})",
                    flush=True,
                )
                return pages


def _report_progress(
    line: str, on_page: Callable[[int, int], None] | None
) -> None:
    if on_page is None:
        return
    try:
        done, total = line[len(PROGRESS_MARKER) :].strip().split("/")
        on_page(int(done), int(total))
    except Exception:
        pass


def run_vl_worker(
    image_paths: list[Path],
    *,
    backend_python: str | None = None,
    worker_path: Path | None = None,
    extra_labels: str = "",
    on_page: Callable[[int, int], None] | None = None,
    stall_timeout: float | None = None,
    session: VLWorkerSession | None = None,
) -> list[dict]:
    """Invoke the PaddleOCR-VL worker on *image_paths*; return per-page dicts.

    ``on_page(done, total)`` fires as each page is recognised. A failed
    resident session falls back to a one-shot worker.
    """
    if not image_paths:
        return []
    if session is not None:
        try:
            return session.run(
                image_paths, extra_labels=extra_labels, on_page=on_page
            )
        except Exception as exc:
            print(
                f"[ocr] resident VL worker failed ({exc}); "
                "falling back to a one-shot worker",
                flush=True,
            )

    backend_python = backend_python or _default_backend_python()
    env = _worker_env(backend_python)

    cmd = [
        *_worker_command(backend_python, worker_path),
        *[str(p) for p in image_paths],
    ]
    if extra_labels:
        cmd += ["--extra-labels", extra_labels]

    _free_gpu_for_worker()

    started = time.monotonic()
    returncode, stdout, stderr = _run_streaming(
        cmd,
        env,
        on_page,
        stall_timeout if stall_timeout is not None else VL_STALL_TIMEOUT,
    )
    if returncode != 0:
        raise RuntimeError(
            "PaddleOCR-VL backend failed.\n"
            f"stdout:\n{stdout[-4000:]}\n\nstderr:\n{stderr[-4000:]}"
        )
    pages = _pages_from_result(stdout, stderr)
    print(
        f"[ocr] recognised {len(image_paths)} page(s) in "
        f"{time.monotonic() - started:.1f}s (one-shot worker: model loaded "
        "for this document alone)",
        flush=True,
    )
    return pages


def _run_streaming(
    cmd: list[str],
    env: dict,
    on_page: Callable[[int, int], None] | None,
    stall_timeout: float,
) -> tuple[int, str, str]:
    """Run *cmd*, forwarding page-progress markers, and guard against a stall.

    Both pipes need reader threads: a single-pipe read deadlocks once the
    other's buffer fills.
    """
    import threading

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    out_lines: list[str] = []
    err_lines: list[str] = []
    last_output = [time.monotonic()]

    def drain(stream, sink, watch_progress):
        for line in stream:
            last_output[0] = time.monotonic()
            sink.append(line)
            if watch_progress and line.startswith(PROGRESS_MARKER):
                _report_progress(line, on_page)
        stream.close()

    threads = [
        threading.Thread(
            target=drain, args=(proc.stdout, out_lines, True), daemon=True
        ),
        threading.Thread(
            target=drain, args=(proc.stderr, err_lines, False), daemon=True
        ),
    ]
    for thread in threads:
        thread.start()

    while proc.poll() is None:
        if stall_timeout and time.monotonic() - last_output[0] > stall_timeout:
            proc.kill()
            proc.wait()
            for thread in threads:
                thread.join(timeout=5)
            raise RuntimeError(
                f"PaddleOCR-VL backend produced no output for "
                f"{stall_timeout:.0f}s "
                "and was stopped. Re-run the document; if it happens again, "
                "the "
                "page count or GPU memory may be the cause.\n"
                f"stdout:\n{''.join(out_lines)[-4000:]}\n\n"
                f"stderr:\n{''.join(err_lines)[-4000:]}"
            )
        time.sleep(0.5)
    for thread in threads:
        thread.join(timeout=10)
    return proc.returncode, "".join(out_lines), "".join(err_lines)


def _release_for_other_feature() -> bool:
    """GPU-manager hook: stop resident PaddleOCR-VL workers."""
    running = [
        session
        for session in list(_LIVE_SESSIONS)
        if session._proc is not None and session._proc.poll() is None
    ]
    for session in running:
        session.close()
    return bool(running)


try:
    from textlab.common import gpu_manager as _gpu_manager
except ImportError:  # pragma: no cover - standalone imports
    _gpu_manager = None
if _gpu_manager is not None:
    _gpu_manager.register(
        _gpu_manager.OCR,
        "Stopped OCR worker",
        _release_for_other_feature,
    )
