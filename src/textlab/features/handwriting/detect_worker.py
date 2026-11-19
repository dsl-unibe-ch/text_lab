"""Line-detection worker: find the text on page images with PaddleOCR.

Runs in the image's PaddleOCR environment (``common.container.PADDLE_ENV``),
started by :func:`.service.detect_lines` through :mod:`textlab.common.jobs`::

    python -m textlab.features.handwriting.detect_worker JOB_DIR

The request is ``{"images": [path, ...], "model": name}``; the result is
``{"pages": [{"image": path, "polygons": [[[x, y], ...], ...]}, ...]}``,
one entry per image with the quadrilateral of every text piece found.

The app's packages are not installed in this environment: only the
standard library, its own packages and the standard-library-only modules
``textlab.common.jobs`` and ``textlab.common.progress`` may be imported. A
test checks this.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from textlab.common.jobs import worker_main
from textlab.common.progress import Progress, ProgressCallback

# Paddle crashes in OpenBLAS with more than one thread, and must not try to
# reach the internet for model checks.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def polygons(result: Any) -> list[list[list[float]]]:
    """Return the text polygons of one PaddleOCR detection result.

    Args:
        result: A result of ``TextDetection.predict``, which can be read
            like a dictionary; older versions only offer ``.json``.

    Returns:
        The polygons, as lists of ``[x, y]`` points.
    """
    try:
        polys = result["dt_polys"]
    except (KeyError, TypeError):
        polys = (getattr(result, "json", None) or {}).get("res", {})
        polys = polys.get("dt_polys", [])
    if hasattr(polys, "tolist"):
        polys = polys.tolist()
    return [
        [[float(x), float(y)] for x, y in polygon] for polygon in polys or []
    ]


def detect(request: Mapping[str, Any], report: ProgressCallback) -> dict:
    """Find the text polygons on the requested page images.

    Args:
        request: ``{"images": [...], "model": ...}``.
        report: Receives an update after each page.

    Returns:
        ``{"pages": [{"image", "polygons"}, ...]}``.
    """
    import paddle
    from paddleocr import TextDetection

    images = list(request["images"])
    use_gpu = paddle.device.is_compiled_with_cuda() and (
        paddle.device.cuda.device_count() > 0
    )
    detector = TextDetection(
        model_name=request["model"], device="gpu" if use_gpu else "cpu"
    )
    pages = []
    for index, image in enumerate(images, start=1):
        found = []
        for result in detector.predict(image, batch_size=1):
            found.extend(polygons(result))
        pages.append({"image": image, "polygons": found})
        report(
            Progress(
                f"Finding lines... page {index}/{len(images)}",
                index / len(images),
            )
        )
    return {"pages": pages}


if __name__ == "__main__":
    worker_main(detect)
