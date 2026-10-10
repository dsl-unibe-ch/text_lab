"""Page rasters: decoding, encoding, cropping and preview downscaling."""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from textlab.features.ocr import doc_ir

#: Longest side of the page image kept for previews (keeps results light).
MAX_PREVIEW_DIM = 1600


def downscale_png_b64(
    png_bytes: bytes,
    regions: list[doc_ir.Region],
    max_dim: int = MAX_PREVIEW_DIM,
    form_groups: list[doc_ir.FormGroup] | None = None,
) -> str | None:
    """Encode a page as base64 PNG, capping its size and scaling boxes."""
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(png_bytes))
        img.load()
    except Exception:
        return base64.b64encode(png_bytes).decode("ascii")
    w, h = img.size
    scale = min(1.0, max_dim / max(w, h)) if max(w, h) > 0 else 1.0
    if scale < 1.0:
        img = img.convert("RGB").resize(
            (max(1, int(w * scale)), max(1, int(h * scale)))
        )
        for region in regions:
            if region.bbox:
                region.bbox = [round(c * scale, 2) for c in region.bbox]
        for group in form_groups or []:
            if group.bbox:
                group.bbox = [round(c * scale, 2) for c in group.bbox]
            for row in group.rows:
                for option in row.options:
                    if option.bbox:
                        option.bbox = [
                            round(c * scale, 2) for c in option.bbox
                        ]
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def crop_region(page_bgr, bbox):
    """Return the part of a page image inside ``bbox``, or ``None``.

    Crops smaller than 4 pixels on a side are ``None``.
    """
    h, w = page_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return page_bgr[y1:y2, x1:x2]


def encode_png_b64(img_bgr) -> str | None:
    """Encode an image as base64 PNG, or return ``None`` on failure."""
    try:
        import cv2

        ok, enc = cv2.imencode(".png", img_bgr)
        return base64.b64encode(enc.tobytes()).decode("ascii") if ok else None
    except Exception:
        return None


def decode_bgr(png_bytes: bytes | None):
    """Decode PNG bytes into a BGR image, or return ``None``."""
    if not png_bytes:
        return None
    try:
        import cv2
        import numpy as np

        arr = np.frombuffer(png_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None
