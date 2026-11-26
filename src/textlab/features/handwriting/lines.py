"""From detected text lines to line crops.

The line detector (:mod:`.detect_worker`) returns one boundary polygon per
text line, already in reading order. Here each polygon becomes a box, boxes
too small to hold writing are dropped, and the crop of each line on the
page is computed. The recognition models were trained on whole lines,
cropped this way.

Only plain Python is used here; cropping itself is in :mod:`.documents`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

#: A box as ``(left, top, right, bottom)``, in pixels.
Box = tuple[float, float, float, float]

#: Boxes smaller than this, in pixels, are detector noise and dropped.
MIN_SIDE = 6
#: Margin added around a line when it is cropped, as a share of its height.
CROP_MARGIN = 0.1


def polygon_box(points: Iterable[Sequence[float]]) -> Box:
    """Return the bounding box of a polygon given as ``(x, y)`` points.

    Raises:
        ValueError: If there are no points.
    """
    xs, ys = [], []
    for x, y in points:
        xs.append(float(x))
        ys.append(float(y))
    if not xs:
        raise ValueError("A polygon needs at least one point.")
    return (min(xs), min(ys), max(xs), max(ys))


def _height(box: Box) -> float:
    return box[3] - box[1]


def line_boxes(
    polygons: Iterable[Iterable[Sequence[float]]],
    *,
    min_side: float = MIN_SIDE,
) -> list[Box]:
    """Return the box of every line, keeping the detector's reading order.

    Args:
        polygons: One boundary polygon per line, in reading order.
        min_side: See :data:`MIN_SIDE`.

    Returns:
        The boxes of the lines at least ``min_side`` wide and high.
    """
    boxes = []
    for polygon in polygons:
        points = list(polygon)
        if not points:
            continue
        box = polygon_box(points)
        if box[2] - box[0] >= min_side and _height(box) >= min_side:
            boxes.append(box)
    return boxes


def crop_box(
    box: Box, width: int, height: int, margin: float = CROP_MARGIN
) -> tuple[int, int, int, int]:
    """Return a line's crop on the page: the box with a margin, in pixels.

    Args:
        box: The line.
        width: Width of the page image.
        height: Height of the page image.
        margin: The margin on every side, as a share of the line's height.

    Returns:
        Integer ``(left, top, right, bottom)`` within the page, at least
        one pixel wide and high.
    """
    pad = margin * _height(box)
    left = max(0, int(box[0] - pad))
    top = max(0, int(box[1] - pad))
    right = min(width, int(box[2] + pad + 0.999))
    bottom = min(height, int(box[3] + pad + 0.999))
    return (left, top, max(right, left + 1), max(bottom, top + 1))


def fit_pixels(width: int, height: int, max_pixels: int | None) -> tuple:
    """Return a size with at most ``max_pixels`` pixels, same aspect.

    Sizes within the limit, or with no limit, are returned unchanged.
    """
    if not max_pixels or width * height <= max_pixels:
        return (width, height)
    scale = (max_pixels / (width * height)) ** 0.5
    return (max(1, int(width * scale)), max(1, int(height * scale)))
