"""Recognition worker: read line images with a handwriting model.

Started by :func:`.service.recognize_lines` through
:mod:`textlab.common.jobs`, in the app's own environment::

    python -m textlab.features.handwriting.worker JOB_DIR

Running in its own process means all GPU memory the model used is released
when it exits. The request is ``{"model": {...}, "crops": [path, ...]}``
with the model as :meth:`.models.HandwritingModel.to_dict` gives it; the
result is ``{"texts": [...]}``, one text per crop.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from textlab.common.jobs import worker_main
from textlab.common.progress import Progress, ProgressCallback
from textlab.features.handwriting.models import HandwritingModel
from textlab.features.handwriting.recognizer import load_recognizer

#: Line images opened at a time, so a large batch does not fill the memory.
CHUNK = 64


def handle(request: Mapping[str, Any], report: ProgressCallback) -> dict:
    """Read the line images named in a request.

    Args:
        request: ``{"model": {...}, "crops": [...]}``.
        report: Receives progress updates.

    Returns:
        ``{"texts": [...]}``.
    """
    from PIL import Image

    spec = HandwritingModel.from_dict(request["model"])
    crops = list(request["crops"])
    report(Progress(f"Loading the model '{spec.label}'...", 0.0))
    recognizer = load_recognizer(spec)

    texts: list[str] = []
    total = len(crops)
    for start in range(0, total, CHUNK):
        images = []
        for path in crops[start : start + CHUNK]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))

        def done(count: int, offset: int = start) -> None:
            read = offset + count
            report(Progress(f"Reading lines... {read}/{total}", read / total))

        texts.extend(recognizer.read(images, on_done=done))
    return {"texts": texts}


if __name__ == "__main__":
    worker_main(handle)
