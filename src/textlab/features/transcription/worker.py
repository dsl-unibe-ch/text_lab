"""Worker process for transcription; started by ``run_transcription``.

Runs :func:`textlab.features.transcription.service.transcribe_files` in its
own process, so that all GPU memory WhisperX used is released when it
exits::

    python -m textlab.features.transcription.worker JOB_DIR

``JOB_DIR`` holds the request; see :mod:`textlab.common.jobs` for the
protocol. Not meant to be run by hand.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from textlab.common.jobs import worker_main
from textlab.common.progress import ProgressCallback
from textlab.features.transcription.models import (
    AudioFile,
    TranscriptionOptions,
)
from textlab.features.transcription.service import transcribe_files


def handle(request: Mapping[str, Any], report: ProgressCallback) -> dict:
    """Transcribe the recordings named in a request.

    Args:
        request: ``{"files": [{"path", "name"}, ...], "options": {...}}``.
        report: Receives progress updates.

    Returns:
        The :class:`~textlab.features.transcription.models.TranscriptionResult`
        as a dictionary.
    """
    files = [AudioFile(Path(f["path"]), f["name"]) for f in request["files"]]
    options = TranscriptionOptions.from_dict(request["options"])
    return transcribe_files(files, options, on_progress=report).to_dict()


if __name__ == "__main__":
    worker_main(handle)
