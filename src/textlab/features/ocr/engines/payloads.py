"""Turning engine output into JSON: texts, polygons and plain values.

EasyOCR and PaddleOCR return nested structures with NumPy values, whose
shape differs between engine versions. These helpers find the texts and
polygons wherever they are, and convert values for ``json.dumps``.

The PaddleOCR worker imports this module in its own environment, so at
module level it may only use the standard library and NumPy; OpenCV, which
both environments have, is imported where an image is encoded.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np

#: Keys of a PaddleOCR prediction kept in the compact result.
PADDLE_KEYS = (
    "input_path",
    "page_index",
    "model_settings",
    "rec_texts",
    "rec_scores",
    "rec_polys",
    "dt_polys",
    "rec_boxes",
    "boxes",
)
#: Keys that may also sit in a prediction's ``res`` part.
PADDLE_RESULT_KEYS = (
    "rec_texts",
    "rec_scores",
    "rec_polys",
    "dt_polys",
    "rec_boxes",
    "boxes",
)
_POLYGON_KEYS = {"rec_polys", "dt_polys", "polys", "boxes", "rec_boxes"}


def make_json_serializable(value: Any) -> Any:
    """Return ``value`` with NumPy and other values turned into JSON types.

    Args:
        value: Any value, possibly nested in lists, tuples, sets and
            dictionaries.

    Returns:
        The same structure built from JSON types only; values without a
        JSON form become strings.
    """
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: make_json_serializable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [make_json_serializable(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return make_json_serializable(value.tolist())
        except Exception:
            pass
    return str(value)


def extract_texts(payload: Any) -> list[str]:
    """Return every recognized text in an engine result, in order.

    Texts are the values of ``rec_texts``, ``texts`` and ``text`` keys at
    any depth.

    Args:
        payload: A result converted by :func:`make_json_serializable`.

    Returns:
        The non-empty texts, stripped.
    """
    texts = []
    if isinstance(payload, str):
        return texts
    if isinstance(payload, dict):
        for key, val in payload.items():
            key_l = str(key).lower()
            if key_l in {"rec_texts", "texts"}:
                if isinstance(val, list):
                    texts.extend(
                        [str(x).strip() for x in val if str(x).strip()]
                    )
                elif isinstance(val, str) and val.strip():
                    texts.append(val.strip())
                else:
                    texts.extend(extract_texts(val))
            elif key_l == "text":
                if isinstance(val, str) and val.strip():
                    texts.append(val.strip())
                else:
                    texts.extend(extract_texts(val))
            else:
                texts.extend(extract_texts(val))
        return texts
    if isinstance(payload, list | tuple):
        for item in payload:
            texts.extend(extract_texts(item))
        return texts
    return texts


def is_number_list(seq: Any) -> bool:
    """Return True for a non-empty list or tuple of numbers only."""
    if not isinstance(seq, list | tuple) or not seq:
        return False
    return all(
        isinstance(v, int | float | np.integer | np.floating) for v in seq
    )


def polygon_points(poly: Any) -> np.ndarray | None:
    """Return a polygon as an ``(n, 2)`` array of integer points.

    Args:
        poly: A flat or nested sequence of coordinates; four numbers are a
            box ``(x1, y1, x2, y2)``.

    Returns:
        The points, or ``None`` if there are fewer than three.
    """
    arr = np.array(poly)
    if arr.size == 0:
        return None
    arr = arr.astype(np.float32).reshape(-1)
    if arr.size == 4:
        x1, y1, x2, y2 = arr.tolist()
        arr = np.array([x1, y1, x2, y1, x2, y2, x1, y2], dtype=np.float32)
    if arr.size < 8:
        return None
    if arr.size % 2 != 0:
        arr = arr[:-1]
    pts = arr.reshape(-1, 2).astype(np.int32)
    if pts.shape[0] < 3:
        return None
    return pts


def extract_polygons(payload: Any) -> list[np.ndarray]:
    """Return every polygon in an engine result, in order.

    Args:
        payload: A result converted by :func:`make_json_serializable`, or
            the value of one of its polygon keys.

    Returns:
        The polygons as returned by :func:`polygon_points`.
    """
    polys = []
    if isinstance(payload, dict):
        for key, val in payload.items():
            key_l = str(key).lower()
            if key_l in _POLYGON_KEYS:
                polys.extend(extract_polygons(val))
                continue
            polys.extend(extract_polygons(val))
        return polys
    if isinstance(payload, list | tuple):
        if is_number_list(payload):
            pts = polygon_points(payload)
            if pts is not None:
                polys.append(pts)
            return polys
        if payload and isinstance(payload[0], list | tuple):
            if all(
                isinstance(p, list | tuple) and len(p) >= 2 for p in payload
            ):
                pts = polygon_points(payload)
                if pts is not None:
                    polys.append(pts)
                return polys
        for item in payload:
            polys.extend(extract_polygons(item))
        return polys
    return polys


def prediction_dict(pred: Any) -> Any:
    """Return a PaddleOCR prediction as JSON types.

    Args:
        pred: A prediction object, or an already plain value.

    Returns:
        Its ``json`` or ``to_dict()`` form, converted with
        :func:`make_json_serializable`.
    """
    if hasattr(pred, "json"):
        raw = pred.json
    elif hasattr(pred, "to_dict"):
        raw = pred.to_dict()
    else:
        raw = pred
    return make_json_serializable(raw)


def paddle_core(raw: Any) -> dict:
    """Return the part of a PaddleOCR prediction that holds the results."""
    if isinstance(raw, dict) and isinstance(raw.get("res"), dict):
        return raw["res"]
    return raw if isinstance(raw, dict) else {}


def compact_paddle_prediction(pred: Any) -> dict:
    """Return the useful part of a PaddleOCR prediction.

    Args:
        pred: A prediction object from ``PaddleOCR.predict``.

    Returns:
        The keys in :data:`PADDLE_KEYS`, taken from the prediction or its
        ``res`` part, with ``rec_texts`` holding every text once.
    """
    raw = prediction_dict(pred)
    compact = {}
    if isinstance(raw, dict):
        for key in PADDLE_KEYS:
            if key in raw:
                compact[key] = raw[key]
        if isinstance(raw.get("res"), dict):
            for key in PADDLE_RESULT_KEYS:
                if key in raw["res"] and key not in compact:
                    compact[key] = raw["res"][key]
    else:
        compact["raw"] = raw
    rec_texts = extract_texts(compact if compact else raw)
    compact["rec_texts"] = list(dict.fromkeys(rec_texts))
    return compact


def png_bytes(image_like: Any) -> bytes | None:
    """Return an image as PNG data.

    Args:
        image_like: PNG data, a BGR NumPy array or an object with a PIL-style
            ``save`` method.

    Returns:
        The PNG data, or ``None`` if the image cannot be encoded.
    """
    if image_like is None:
        return None
    try:
        if isinstance(image_like, bytes | bytearray):
            return bytes(image_like)
        if isinstance(image_like, np.ndarray):
            import cv2

            ok, encoded = cv2.imencode(".png", image_like)
            return encoded.tobytes() if ok else None
        if hasattr(image_like, "save"):
            buffer = io.BytesIO()
            image_like.save(buffer, format="PNG")
            return buffer.getvalue()
    except Exception:
        return None
    return None


def rendered_png(pred: Any) -> bytes | None:
    """Return the result image PaddleOCR drew for a prediction, as PNG.

    Args:
        pred: A prediction object from ``PaddleOCR.predict``.

    Returns:
        The OCR result image if there is one, else the first image the
        prediction carries, or ``None``.
    """
    payload = getattr(pred, "img", None)
    if payload is None:
        return None
    if isinstance(payload, dict):
        for key in ("ocr_res_img", "overall_ocr_res", "layout_det_res"):
            if key in payload:
                out = png_bytes(payload[key])
                if out is not None:
                    return out
        for val in payload.values():
            out = png_bytes(val)
            if out is not None:
                return out
        return None
    return png_bytes(payload)
