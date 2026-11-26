"""Line-detection worker: find the text lines on page images with kraken.

Runs in the image's kraken environment (``common.container.KRAKEN_ENV``),
started by :func:`.service.detect_lines` through :mod:`textlab.common.jobs`::

    python -m textlab.features.handwriting.detect_worker JOB_DIR

The request is ``{"images": [path, ...]}``; the result is
``{"pages": [{"image": path, "polygons": [[[x, y], ...], ...]}, ...]}``,
one entry per image with the boundary polygon of every line, in reading
order.

kraken's baseline segmenter (``blla``) was trained on historical
manuscripts. It finds whole lines, also slanted or curved ones, and sorts
them into reading order, columns and marginal notes included. Its default
model ships with the kraken package, so nothing is downloaded.

The app's packages are not installed in this environment: only the
standard library, its own packages and the standard-library-only modules
``textlab.common.jobs`` and ``textlab.common.progress`` may be imported. A
test checks this.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from textlab.common.jobs import worker_main
from textlab.common.progress import Progress, ProgressCallback


def boundaries(lines: Iterable[Any]) -> list[list[list[float]]]:
    """Return the boundary polygons of kraken's line records.

    Lines without a boundary (kraken could not compute one) are left out:
    without it, there is nothing to crop.

    Args:
        lines: The ``lines`` of a kraken ``Segmentation``, each with a
            ``boundary`` of ``(x, y)`` points.

    Returns:
        The polygons, in the order given, as lists of ``[x, y]`` points.
    """
    polygons = []
    for line in lines:
        boundary: Sequence[Sequence[float]] | None = getattr(
            line, "boundary", None
        )
        if boundary:
            polygons.append([[float(x), float(y)] for x, y in boundary])
    return polygons


def detect(request: Mapping[str, Any], report: ProgressCallback) -> dict:
    """Find the text lines on the requested page images.

    Args:
        request: ``{"images": [...]}``.
        report: Receives an update after each page.

    Returns:
        ``{"pages": [{"image", "polygons"}, ...]}``.
    """
    import torch
    from kraken.configs import SegmentationInferenceConfig
    from kraken.tasks import SegmentationTaskModel
    from PIL import Image

    images = list(request["images"])
    # A session has one GPU; kraken's default would claim every visible one.
    config = SegmentationInferenceConfig(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        device=1,
    )
    segmenter = SegmentationTaskModel.load_model()
    pages = []
    for index, image in enumerate(images, start=1):
        with Image.open(image) as page:
            segmentation = segmenter.predict(page, config)
        pages.append(
            {"image": image, "polygons": boundaries(segmentation.lines)}
        )
        report(
            Progress(
                f"Finding lines... page {index}/{len(images)}",
                index / len(images),
            )
        )
    return {"pages": pages}


if __name__ == "__main__":
    worker_main(detect)
