"""Preview images for the manual engines: boxes on the page, text layout.

EasyOCR gets two images per page: the page with the detected boxes, and a
white page with each text drawn into its box. PaddleOCR draws its own result
image; when it does not, the page with its boxes is drawn here.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from textlab.features.ocr.engines.payloads import (
    extract_polygons,
    extract_texts,
    paddle_core,
    png_bytes,
    polygon_points,
    prediction_dict,
    rendered_png,
)

#: Fonts for texts OpenCV cannot draw (non-ASCII), first found is used.
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)
#: Font scales tried, largest first, until a text fits its box.
_FONT_SCALES = (0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BOX_COLOR = (0, 200, 0)
_OUTLINE_COLOR = (210, 210, 210)

TextItem = tuple[np.ndarray, str]


def render_easyocr_preview(
    image_path: str | os.PathLike, page_res: Sequence[Any]
) -> tuple[bytes | None, bytes | None]:
    """Draw EasyOCR's result for one page.

    Args:
        image_path: The page image.
        page_res: EasyOCR's ``readtext`` result: ``(box, text, ...)``
            items.

    Returns:
        PNG data of the page with the boxes, and of the text layout; both
        ``None`` if the image cannot be read.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return None, None
    text_items = []
    for item in page_res:
        if not isinstance(item, list | tuple) or len(item) < 2:
            continue
        pts = polygon_points(item[0])
        if pts is None:
            continue
        text = str(item[1]).strip()
        text_items.append((pts, text))
        cv2.polylines(
            image, [pts], isClosed=True, color=_BOX_COLOR, thickness=2
        )
    text_canvas = _draw_text_canvas(image.shape, text_items)
    return png_bytes(image), png_bytes(text_canvas)


def render_paddle_preview(
    image_path: str | os.PathLike, page_preds: Sequence[Any]
) -> tuple[bytes | None, bytes | None]:
    """Draw PaddleOCR's result for one page.

    Args:
        image_path: The page image.
        page_preds: The page's predictions, as objects or in their compact
            form.

    Returns:
        PNG data of PaddleOCR's own result image, or of the page with the
        boxes, and of the text layout; both ``None`` if the image cannot be
        read.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return None, None
    left_png = None
    text_items = []
    for pred in page_preds:
        if left_png is None:
            left_png = rendered_png(pred)
        core = paddle_core(prediction_dict(pred))
        texts = core.get("rec_texts")
        if not isinstance(texts, list):
            texts = extract_texts(core)
        polys_src = (
            core.get("rec_polys")
            or core.get("dt_polys")
            or core.get("rec_boxes")
            or core.get("boxes")
        )
        if polys_src is None:
            polys = extract_polygons(core)
        else:
            polys = extract_polygons(polys_src)
        for i, poly in enumerate(polys):
            pts = polygon_points(poly)
            if pts is None:
                continue
            text = str(texts[i]).strip() if i < len(texts) else ""
            text_items.append((pts, text))
    text_canvas = _draw_text_canvas(image.shape, text_items)
    if left_png is None:
        for pts, _ in text_items:
            cv2.polylines(
                image, [pts], isClosed=True, color=_BOX_COLOR, thickness=2
            )
        left_png = png_bytes(image)
    return left_png, png_bytes(text_canvas)


def _draw_text_canvas(
    image_shape: tuple[int, ...], items: Sequence[TextItem]
) -> np.ndarray:
    """Draw each text into its box on a white page of the image's size."""
    h, w = image_shape[:2]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    pil_img = None
    pil_draw = None
    fonts = _FontCache()

    for pts, text in items:
        if pts is None:
            continue
        cv2.polylines(
            canvas, [pts], isClosed=True, color=_OUTLINE_COLOR, thickness=1
        )
        if not text:
            continue
        x, y, bw, bh = cv2.boundingRect(pts)
        lines, font_scale, thickness, line_h = _fit_text_to_box(text, bw, bh)
        if not lines:
            continue
        y_cursor = max(12, y + line_h)
        y_limit = min(h - 2, y + bh - 2)
        for line in lines:
            if y_cursor > y_limit:
                break
            px = max(0, x + 2)
            py = min(h - 4, y_cursor)
            if any(ord(ch) > 127 for ch in line):
                # OpenCV's fonts are ASCII only; PIL draws the rest.
                if pil_img is None:
                    pil_img = Image.fromarray(
                        cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                    )
                    pil_draw = ImageDraw.Draw(pil_img)
                pil_draw.text(
                    (px, max(0, py - line_h + 4)),
                    line,
                    fill=(0, 0, 0),
                    font=fonts.get(26 * font_scale),
                )
            else:
                cv2.putText(
                    canvas,
                    line,
                    (px, py),
                    _FONT,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                    cv2.LINE_AA,
                )
            y_cursor += line_h
    if pil_img is not None:
        canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return canvas


class _FontCache:
    """TrueType fonts by pixel size, from the first usable candidate."""

    def __init__(self) -> None:
        self._paths = [path for path in FONT_CANDIDATES if Path(path).exists()]
        self._fonts: dict[int, Any] = {}

    def get(self, px_size: float) -> Any:
        """Return a font of at least 12 pixels, or PIL's default font."""
        size = max(12, int(px_size))
        if size not in self._fonts:
            self._fonts[size] = self._load(size)
        return self._fonts[size]

    def _load(self, size: int) -> Any:
        for path in self._paths:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
        return ImageFont.load_default()


def _wrap_line_to_width(
    text: str, max_width: int, font_scale: float, thickness: int
) -> list[str]:
    """Split a line at spaces into lines no wider than ``max_width``."""
    words = text.split(" ")
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        candidate_w = cv2.getTextSize(candidate, _FONT, font_scale, thickness)[
            0
        ][0]
        if candidate_w <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _wrap(text: str, max_width: int, font_scale: float, thickness: int):
    """Wrap every line of ``text``; empty lines stay empty."""
    wrapped = []
    for raw_line in text.split("\n"):
        raw_line = raw_line.strip()
        if not raw_line:
            wrapped.append("")
            continue
        wrapped.extend(
            _wrap_line_to_width(raw_line, max_width, font_scale, thickness)
        )
    return wrapped


def _line_height(font_scale: float, thickness: int) -> int:
    """Return the height of one line of text, with spacing."""
    return cv2.getTextSize("Ag", _FONT, font_scale, thickness)[0][1] + 4


def _fit_text_to_box(
    text: str, bw: int, bh: int
) -> tuple[list[str], float, int, int]:
    """Choose the largest font scale at which ``text`` fits a box.

    Returns:
        The wrapped lines, font scale, thickness and line height. If the
        text does not fit even at the smallest scale, the lines that fit
        are returned and the last one ends with "...".
    """
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return [], 0.4, 1, 12
    thickness = 1
    max_width = max(8, bw - 4)
    for font_scale in _FONT_SCALES:
        line_h = _line_height(font_scale, thickness)
        max_lines = max(1, bh // max(1, line_h))
        wrapped = _wrap(text, max_width, font_scale, thickness)
        if len(wrapped) <= max_lines:
            return wrapped, font_scale, thickness, line_h
    font_scale = _FONT_SCALES[-1]
    line_h = _line_height(font_scale, thickness)
    max_lines = max(1, bh // max(1, line_h))
    wrapped = _wrap(text, max_width, font_scale, thickness)[:max_lines]
    if wrapped:
        last = wrapped[-1]
        while (
            last
            and cv2.getTextSize(last + "...", _FONT, font_scale, thickness)[0][
                0
            ]
            > max_width
        ):
            last = last[:-1]
        wrapped[-1] = (last + "...") if last else "..."
    return wrapped, font_scale, thickness, line_h
