"""From detected text pieces to text lines in reading order.

The text detector returns a quadrilateral per piece of text it finds, which
is a whole line or, where the writing has wide gaps, a part of one. The
recognition models were trained on whole lines, so pieces that sit side by
side at the same height are joined into one line here, and the lines are
put into reading order: top to bottom, and left to right for lines next to
each other (a marginal note beside the body text, for example).

Only plain Python is used here; cropping is in :mod:`.documents`.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence

#: A box as ``(left, top, right, bottom)``, in pixels.
Box = tuple[float, float, float, float]

#: Pieces join a line when their vertical spans overlap by at least this
#: share of the shorter one.
MIN_VERTICAL_OVERLAP = 0.5
#: ... and the horizontal gap between them is at most this many median
#: line heights.
MAX_GAP_LINE_HEIGHTS = 2.0
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


def _vertical_overlap(a: Box, b: Box) -> float:
    """Overlap of the vertical spans, as a share of the shorter span."""
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    shorter = min(_height(a), _height(b))
    return overlap / shorter if shorter > 0 else 0.0


def _union(boxes: Sequence[Box]) -> Box:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def group_lines(
    boxes: Iterable[Box],
    *,
    min_overlap: float = MIN_VERTICAL_OVERLAP,
    max_gap: float = MAX_GAP_LINE_HEIGHTS,
    min_side: float = MIN_SIDE,
) -> list[Box]:
    """Join text pieces into lines and return the lines in reading order.

    Pieces are taken from left to right. Each joins the line whose
    rightmost piece it overlaps most vertically, if that overlap is at
    least ``min_overlap`` and the gap to that piece is at most ``max_gap``
    median heights; otherwise it starts a new line. Comparing with the
    rightmost piece rather than the whole line lets a slanted line join
    up, while a line above or below does not.

    Args:
        boxes: The detected pieces.
        min_overlap: See :data:`MIN_VERTICAL_OVERLAP`.
        max_gap: See :data:`MAX_GAP_LINE_HEIGHTS`.
        min_side: See :data:`MIN_SIDE`.

    Returns:
        One box per line, in reading order (:func:`reading_order`).
    """
    pieces = [
        box
        for box in boxes
        if box[2] - box[0] >= min_side and _height(box) >= min_side
    ]
    if not pieces:
        return []
    gap_limit = max_gap * statistics.median(map(_height, pieces))

    lines: list[list[Box]] = []
    for piece in sorted(pieces, key=lambda box: (box[0], box[1])):
        best, best_overlap = None, min_overlap
        for line in lines:
            last = line[-1]
            if piece[0] - last[2] > gap_limit:
                continue
            overlap = _vertical_overlap(piece, last)
            if overlap >= best_overlap:
                best, best_overlap = line, overlap
        if best is None:
            lines.append([piece])
        else:
            best.append(piece)
    return reading_order([_union(line) for line in lines], min_overlap)


def reading_order(
    lines: Iterable[Box], min_overlap: float = MIN_VERTICAL_OVERLAP
) -> list[Box]:
    """Sort lines top to bottom, and side-by-side lines left to right.

    Lines are sorted by their vertical centre. Consecutive lines whose
    vertical spans overlap by at least ``min_overlap`` form a row, and each
    row is sorted from left to right.
    """
    ordered = sorted(lines, key=lambda box: (box[1] + box[3]) / 2)
    result: list[Box] = []
    row: list[Box] = []
    for line in ordered:
        if row and _vertical_overlap(line, row[-1]) < min_overlap:
            result.extend(sorted(row, key=lambda box: box[0]))
            row = []
        row.append(line)
    result.extend(sorted(row, key=lambda box: box[0]))
    return result


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
