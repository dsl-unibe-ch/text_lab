"""Running heavy backend work in a separate worker process.

GPU-heavy steps (WhisperX, OCR models) run in a subprocess instead of the
app's own process. When the subprocess exits, every byte of GPU memory it
used is released, and a crash or an out-of-memory error in it cannot take
down the app.

The parent calls :func:`run_worker` with the worker's module name and a
JSON-serializable request. The worker module calls :func:`worker_main` with
a handler function. They communicate through files in a private job folder
inside the workspace (see :mod:`textlab.common.storage`):

- ``request.json``: the request, written by the parent.
- ``progress.json``: the latest :class:`~textlab.common.progress.Progress`,
  rewritten by the worker and polled by the parent.
- ``result.json``: ``{"ok": true, "result": ...}`` or
  ``{"ok": false, "error": ..., "traceback": ...}``, written by the worker.

The job folder is deleted when the run ends. The worker's standard output
and error go to the app's log.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import textlab
from textlab.common.progress import (
    CancelledError,
    Progress,
    ProgressCallback,
    no_progress,
)
from textlab.common.storage import get_workspace

REQUEST_FILE = "request.json"
PROGRESS_FILE = "progress.json"
RESULT_FILE = "result.json"

#: Seconds a worker gets to exit after SIGTERM before it is killed.
STOP_TIMEOUT = 10.0

#: Signature of a worker's handler: request and progress callback in, a
#: JSON-serializable result out.
WorkerHandler = Callable[[Mapping[str, Any], ProgressCallback], Any]


class WorkerError(RuntimeError):
    """The worker failed or stopped without a result.

    Attributes:
        details: The worker's traceback, or other diagnostic text.
    """

    def __init__(self, message: str, details: str = ""):
        """Store the message and the diagnostic details.

        Args:
            message: Short description of the failure.
            details: The worker's traceback, or other diagnostic text.
        """
        super().__init__(message)
        self.details = details


def run_worker(
    module: str,
    request: Mapping[str, Any],
    *,
    area: str = "jobs",
    on_progress: ProgressCallback = no_progress,
    cancel: threading.Event | None = None,
    poll_interval: float = 0.5,
    python: str | None = None,
) -> Any:
    """Run ``python -m module`` on a request and return its result.

    Args:
        module: Dotted name of the worker module, which calls
            :func:`worker_main`.
        request: JSON-serializable input for the worker's handler.
        area: Workspace area for the job folder, usually the feature name.
        on_progress: Receives the worker's progress updates. It is called in
            the caller's thread; if it raises, the worker is stopped and the
            exception propagates.
        cancel: Stops the worker when set.
        poll_interval: Seconds between checks for progress and exit.
        python: The interpreter to run the worker with, for workers that
            need another environment of the image (see
            :mod:`textlab.common.container`); defaults to this process's.

    Returns:
        The value the worker's handler returned.

    Raises:
        WorkerError: If the worker raised an exception or stopped without
            writing a result.
        CancelledError: If ``cancel`` was set.
    """
    with get_workspace().temp_dir(area, prefix="job-") as job_dir:
        _write_json(job_dir / REQUEST_FILE, request)
        process = subprocess.Popen(
            [python or sys.executable, "-m", module, str(job_dir)],
            env=worker_environment(python),
        )
        try:
            last_update = None
            while True:
                exit_code = process.poll()
                update = _read_progress(job_dir)
                if update is not None and update != last_update:
                    last_update = update
                    on_progress(update)
                if exit_code is not None:
                    return _read_result(job_dir, exit_code)
                if cancel is not None and cancel.is_set():
                    raise CancelledError("The work was cancelled.")
                time.sleep(poll_interval)
        finally:
            _stop(process)


def worker_main(handler: WorkerHandler) -> None:
    """Run a worker's handler on the request in the job folder.

    Call this from the worker module's ``if __name__ == "__main__":`` block.
    The job folder is the first command-line argument. The process exits
    with status 0 on success and 1 on failure.

    Args:
        handler: Receives the request and a progress callback, and returns a
            JSON-serializable result.
    """
    job_dir = Path(sys.argv[1])
    request = json.loads((job_dir / REQUEST_FILE).read_text(encoding="utf-8"))

    def report(progress: Progress) -> None:
        _write_json(job_dir / PROGRESS_FILE, progress.to_dict())

    try:
        payload = {"ok": True, "result": handler(request, report)}
        exit_code = 0
    except Exception as exc:
        payload = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        exit_code = 1
    _write_json(job_dir / RESULT_FILE, payload)
    sys.exit(exit_code)


def worker_environment(python: str | None = None) -> dict[str, str]:
    """Return the environment for a worker process.

    ``textlab`` is importable in it, so workers can be started with
    ``python -m``. A worker that runs with another environment's
    interpreter also gets that environment's programs and libraries first
    on ``PATH`` and ``LD_LIBRARY_PATH``, as conda activation would set them.

    Args:
        python: The worker's interpreter, if it is not this process's.

    Returns:
        A copy of this process's environment with those changes.
    """
    env = os.environ.copy()
    src_root = str(Path(textlab.__file__).resolve().parents[1])
    _prepend(env, "PYTHONPATH", src_root)
    if python is not None:
        bin_dir = Path(python).resolve().parent
        _prepend(env, "PATH", str(bin_dir))
        _prepend(env, "LD_LIBRARY_PATH", str(bin_dir.parent / "lib"))
    return env


def _prepend(env: dict[str, str], name: str, path: str) -> None:
    """Put ``path`` first in a path-list variable, without duplicates."""
    existing = env.get(name, "").split(os.pathsep)
    paths = [path, *filter(None, existing)]
    env[name] = os.pathsep.join(dict.fromkeys(paths))


def _read_progress(job_dir: Path) -> Progress | None:
    """Return the worker's latest progress update, if there is one."""
    try:
        data = json.loads(
            (job_dir / PROGRESS_FILE).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    return Progress.from_dict(data)


def _read_result(job_dir: Path, exit_code: int) -> Any:
    """Return the worker's result, or raise its error."""
    try:
        payload = json.loads(
            (job_dir / RESULT_FILE).read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise WorkerError(
            f"The worker stopped without a result (exit code {exit_code}). "
            "The session log has details."
        ) from exc
    if payload.get("ok"):
        return payload.get("result")
    raise WorkerError(
        payload.get("error", "The worker failed."),
        payload.get("traceback", ""),
    )


def _stop(process: subprocess.Popen) -> None:
    """Terminate a worker that is still running, killing it if needed."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=STOP_TIMEOUT)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _write_json(path: Path, data: Any) -> None:
    """Write JSON atomically, so a reader never sees a half-written file."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    """Convert NumPy scalars and arrays, which ``json`` cannot encode."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")
