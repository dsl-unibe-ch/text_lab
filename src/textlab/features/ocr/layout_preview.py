"""A page image with the recognized regions outlined, for checking layout."""

from __future__ import annotations

import cv2
import numpy as np

#: Per-region-type overlay colors (BGR) for the automatic pipeline's layout
#: tab.
LAYOUT_TYPE_COLORS = {
    "text": (150, 150, 150),
    "title": (0, 140, 255),
    "table": (0, 0, 220),
    "figure": (220, 120, 0),
    "formula": (200, 0, 200),
    "footnote": (120, 120, 120),
    "header": (150, 150, 0),
    "footer": (150, 150, 0),
    "seal": (0, 170, 170),
    "checkbox": (0, 200, 0),
    "list": (200, 180, 0),
    "reference": (140, 140, 140),
    "other": (90, 90, 90),
}


def render_layout_preview(image_bytes, regions):
    """Draw colored, labeled boxes per region type onto a page raster.

    ``regions`` is a list of dicts with ``bbox`` ([x1,y1,x2,y2] in the image's
    pixel space) and ``type`` (one of :data:`LAYOUT_TYPE_COLORS`). Returns PNG
    bytes, or ``None`` if the image cannot be decoded.
    """
    if not image_bytes:
        return None
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    overlay = img.copy()
    for region in regions:
        bbox = region.get("bbox") or []
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in bbox[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        rtype = str(region.get("type", "other"))
        color = LAYOUT_TYPE_COLORS.get(rtype, LAYOUT_TYPE_COLORS["other"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            overlay,
            rtype,
            (x1 + 2, max(14, y1 + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    blended = cv2.addWeighted(overlay, 0.8, img, 0.2, 0)
    ok, encoded = cv2.imencode(".png", blended)
    return encoded.tobytes() if ok else None
