"""Checkbox / survey-mark state detection.

Three cooperating mechanisms:

1. **VL glyphs** — the PaddleOCR-VL recogniser usually transcribes survey marks
   as glyphs *inside* table or text content (``☒``/``☐``/``✗``/``○``).
   :func:`extract_mark_glyphs` pulls those out of any region content so answers
   embedded in tables/text are surfaced, not just ``checkbox``-typed regions.

2. **Geometry** — OpenCV analysis of the region crop. Two layers:
   :func:`detect_markup_geometric` classifies a single mark crop by interior
   ink *and* stroke lines (a thin pen cross has a tiny fill ratio, so a pure
   fill threshold misses it — the Hough stroke rule catches it), and
   :func:`find_marks` locates the individual circles/boxes inside a larger
   region crop so each one can be classified.

3. **Reconciliation** — :func:`reconcile_marks` compares both signals. The VL
   model frequently transcribes a *crossed* circle as a plain ``○``; geometry
   can flag that disagreement for review, but the production pipeline does not
   rewrite the OCR result. The old override remains available only as a frozen
   benchmark baseline via ``allow_override=True``.

Every verdict is ``{"state": "checked"|"unchecked"|"uncertain", ...}``;
``uncertain`` results are never dropped — the UI surfaces them for review.
"""

from __future__ import annotations

import base64
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---- geometric decision thresholds ------------------------------------------
CHECKED_FILL = 0.18      # >= this fraction of interior inked -> checked
UNCHECKED_FILL = 0.045   # <= this fraction inked (and no stroke) -> unchecked
STRIKE_CHECKED = 0.45    # stroke line >= this fraction of interior diagonal -> checked
STRIKE_AMBIGUOUS = 0.30  # stroke above this blocks a confident "unchecked"
_INTERIOR_MARGIN = 0.24  # fraction of the crop trimmed on each side as border
_SATURATION_MIN = 80     # HSV saturation above which a pixel counts as a colored mark

# ---- glyphs the VL recogniser might emit ------------------------------------
_CHECKED_GLYPHS = set("☑☒✓✔✗✘√●■▣▪◼✕⨯×")
_UNCHECKED_GLYPHS = set("☐□○◯▢▭▯❍◌◦")

#: Glyph written into content when geometry overrides a missed mark.
OVERRIDE_GLYPH = "☒"


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ==========================================
#        1. VL GLYPH PARSING
# ==========================================


def detect_markup_from_vl(block_content: Optional[str]) -> Optional[Dict[str, Any]]:
    """Infer a single-checkbox state from recognised text, or ``None``."""
    if not block_content:
        return None
    s = str(block_content)

    # Markdown-style checkboxes: [x], [X], [ ], []
    if re.search(r"\[\s*[xX×✓✔✗✘]\s*\]", s):
        return {"state": "checked", "method": "vl", "score": 0.9}
    if re.search(r"\[\s*\]", s):
        return {"state": "unchecked", "method": "vl", "score": 0.9}

    has_checked = any(g in s for g in _CHECKED_GLYPHS)
    has_unchecked = any(g in s for g in _UNCHECKED_GLYPHS)
    if has_checked and not has_unchecked:
        return {"state": "checked", "method": "vl", "score": 0.8}
    if has_unchecked and not has_checked:
        return {"state": "unchecked", "method": "vl", "score": 0.8}

    # Ambiguous or no recognised glyph -> let geometry decide.
    return None


def extract_mark_glyphs(content: Optional[str]) -> List[Dict[str, Any]]:
    """All mark glyphs in *content*, in order, with their transcribed state."""
    items: List[Dict[str, Any]] = []
    if not content:
        return items
    for i, ch in enumerate(str(content)):
        if ch in _CHECKED_GLYPHS:
            items.append({"index": i, "glyph": ch, "state": "checked", "method": "vl"})
        elif ch in _UNCHECKED_GLYPHS:
            items.append({"index": i, "glyph": ch, "state": "unchecked", "method": "vl"})
    return items


def apply_states_to_content(content: str, states: List[str]) -> Tuple[str, bool]:
    """Rewrite mark glyphs in *content* to match reconciled *states*.

    Only upgrades: a glyph transcribed as unchecked whose reconciled state is
    ``checked`` becomes :data:`OVERRIDE_GLYPH`. Returns ``(new_content,
    changed)``; if the glyph count does not match ``states`` the content is
    returned untouched.
    """
    items = extract_mark_glyphs(content)
    if len(items) != len(states):
        return content, False
    chars = list(content)
    changed = False
    for item, state in zip(items, states):
        if state == "checked" and item["state"] == "unchecked":
            chars[item["index"]] = OVERRIDE_GLYPH
            changed = True
    return "".join(chars), changed


# ==========================================
#        2. GEOMETRIC ANALYSIS
# ==========================================


def decode_crop_b64(b64: Optional[str]):
    """Decode a base64 PNG crop into a BGR ``np.ndarray`` (or ``None``)."""
    if not b64:
        return None
    try:
        import cv2

        data = base64.b64decode(b64)
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _ink_mask(crop_bgr):
    """Binary mask of ink (dark or strongly colored pixels).

    The dark-ink threshold is anchored to the estimated paper background
    (high percentile of gray) instead of Otsu: on a sparse, noisy scan crop
    Otsu bisects the paper texture itself and reports phantom ink everywhere,
    which made every empty survey circle look filled.
    """
    import cv2

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    background = float(np.percentile(gray, 88))
    threshold = max(25.0, background - max(45.0, 0.22 * background))
    dark = ((gray < threshold) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    colored = (
        ((hsv[:, :, 1] > _SATURATION_MIN) & (hsv[:, :, 2] > 60)) * 255
    ).astype(np.uint8)
    return cv2.bitwise_or(dark, colored)


def _interior(crop_bgr):
    """Interior ink mask (border/outline trimmed) and its fill ratio."""
    if crop_bgr is None:
        return None, None
    h, w = crop_bgr.shape[:2]
    if h < 6 or w < 6:
        return None, None
    ink = _ink_mask(crop_bgr)
    my = max(1, int(h * _INTERIOR_MARGIN))
    mx = max(1, int(w * _INTERIOR_MARGIN))
    interior = ink[my : h - my, mx : w - mx]
    if interior.size == 0:
        return None, None
    n_ink = int(np.count_nonzero(interior))
    # A handful of stray dark pixels (scanner noise, dust) is not a mark.
    if n_ink < max(8, int(interior.size * 0.002)):
        interior = np.zeros_like(interior)
        n_ink = 0
    fill = n_ink / float(interior.size)
    return interior, fill


def _line_strike_score(interior) -> float:
    """Longest central stroke in the interior, as a fraction of its diagonal.

    Catches thin pen crosses/checks whose fill ratio is tiny. Segments whose
    midpoint lies near the edge are ignored so a residual box outline or the
    arc of an empty circle does not count as a strike.
    """
    import cv2

    h, w = interior.shape[:2]
    if min(h, w) < 8:
        return 0.0
    diag = math.hypot(h, w)
    min_len = max(6, int(min(h, w) * 0.45))
    lines = cv2.HoughLinesP(
        interior, 1, np.pi / 180,
        threshold=max(8, int(min_len * 0.7)),
        minLineLength=min_len, maxLineGap=3,
    )
    if lines is None:
        return 0.0
    best = 0.0
    cx_lo, cx_hi = w * 0.18, w * 0.82
    cy_lo, cy_hi = h * 0.18, h * 0.82
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if not (cx_lo <= mx <= cx_hi and cy_lo <= my <= cy_hi):
            continue
        best = max(best, math.hypot(x2 - x1, y2 - y1))
    return best / max(1.0, diag)


def detect_markup_geometric(crop_bgr) -> Dict[str, Any]:
    """Classify a single checkbox/mark crop by interior ink and strokes."""
    interior, fill = _interior(crop_bgr)
    if fill is None:
        return {"state": "uncertain", "method": "geometric", "score": 0.0,
                "fill_ratio": None, "strike": None}

    strike = _line_strike_score(interior)

    if fill >= CHECKED_FILL:
        score = _clamp(0.6 + (fill - CHECKED_FILL) * 2.0)
        state = "checked"
    elif strike >= STRIKE_CHECKED:
        score = _clamp(0.55 + (strike - STRIKE_CHECKED) * 0.8)
        state = "checked"
    elif fill <= UNCHECKED_FILL and strike < STRIKE_AMBIGUOUS:
        score = _clamp(0.6 + (UNCHECKED_FILL - fill) * 8.0)
        state = "unchecked"
    else:
        span = CHECKED_FILL - UNCHECKED_FILL
        nearest = min(CHECKED_FILL - fill, max(0.0, fill - UNCHECKED_FILL)) / span
        score = _clamp(0.5 - nearest, 0.0, 0.5)
        state = "uncertain"

    return {"state": state, "method": "geometric", "score": round(score, 3),
            "fill_ratio": round(fill, 4), "strike": round(strike, 3)}


# ==========================================
#        3. MARK FINDER (region crops)
# ==========================================


def find_marks(region_bgr, n_expected: Optional[int] = None) -> List[Dict[str, Any]]:
    """Locate candidate survey marks (circles/boxes) in a region crop.

    Returns marks in reading order, each ``{"bbox": [x1,y1,x2,y2], "state": ..,
    "score": .., ...}``. When *n_expected* is given (from the glyph count) and
    more candidates are found, the most size-consistent subset of that length
    is kept — this filters out letter shapes ('o', 'a', ...) that sneak past
    the geometric filters.
    """
    import cv2

    if region_bgr is None:
        return []
    H, W = region_bgr.shape[:2]
    if H < 12 or W < 12:
        return []

    ink = _ink_mask(region_bgr)
    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cands = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 9 or h < 9:
            continue
        if h > 0.98 * H or w > 0.6 * W:
            continue
        aspect = w / float(h)
        if not (0.55 <= aspect <= 1.8):
            continue
        # closed shapes (circle/box outlines) enclose most of their bbox;
        # thin letter strokes like J/l/i do not.
        if cv2.contourArea(cnt) < 0.3 * w * h:
            continue
        # Survey marks are closed outline shapes. Solid letter shapes (n, h,
        # m, ...) pass the area test but are neither box-like (ink along all
        # four bbox sides) nor circle-like (contour on the enclosing circle).
        if not (_looks_like_box(ink, x, y, w, h) or _looks_like_circle(cnt)):
            continue
        if not _is_isolated(ink, x, y, w, h, W):
            continue
        cands.append(_make_candidate(region_bgr, x, y, w, h))

    # Survey marks are laid out in regular patterns (an option column, a scale
    # row); stray letter shapes that survive the filters are scattered. Keep
    # only the pattern, and recover a mark whose outline was destroyed by the
    # pen cross from the gap it leaves in the pattern.
    if n_expected == 1:
        # A single mark ("☐ label ...") sits at the start of its region.
        cands = [c for c in cands if (c["bbox"][0] + c["bbox"][2]) / 2.0 <= 0.28 * W]
        cands = sorted(cands, key=lambda c: c["bbox"][0])[:1]
    elif n_expected and n_expected >= 2:
        cands = _dominant_alignment_group(cands)
        if len(cands) < 2:
            cands = []
        elif len(cands) == n_expected - 1:
            filled = _fill_pattern_gap(cands, region_bgr)
            if filled is not None:
                cands.append(filled)

    if n_expected and len(cands) > n_expected:
        cands = _most_uniform_subset(cands, n_expected)

    # reading order: cluster into rows by y, then sort by x
    if not cands:
        return []
    med_h = float(np.median([c["h"] for c in cands]))
    cands.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))
    rows: List[List[dict]] = []
    for c in cands:
        cy = (c["bbox"][1] + c["bbox"][3]) / 2.0
        if rows and abs(cy - rows[-1][0]) <= max(6.0, med_h * 0.7):
            rows[-1][1].append(c)
        else:
            rows.append((cy, [c]))
        rows[-1] = (
            float(np.mean([(m["bbox"][1] + m["bbox"][3]) / 2.0 for m in rows[-1][1]])),
            rows[-1][1],
        )
    out = []
    for _, row in rows:
        out.extend(sorted(row, key=lambda c: c["bbox"][0]))
    for c in out:
        c.pop("h", None)
    return out


def _make_candidate(region_bgr, x, y, w, h) -> Dict[str, Any]:
    H, W = region_bgr.shape[:2]
    pad_x, pad_y = max(2, w // 8), max(2, h // 8)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(W, x + w + pad_x), min(H, y + h + pad_y)
    verdict = detect_markup_geometric(region_bgr[y1:y2, x1:x2])
    verdict["bbox"] = [x1, y1, x2, y2]
    verdict["h"] = h
    return verdict


def _dominant_alignment_group(cands: List[dict]) -> List[dict]:
    """Largest subset sharing an x-column or a y-row (regular survey layout)."""
    if len(cands) < 2:
        return cands
    heights = [c["h"] for c in cands]
    tol = max(6.0, float(np.median(heights)))
    best: List[dict] = []
    for axis in (0, 1):  # 0: x-centers (column), 1: y-centers (row)
        centers = [
            ((c["bbox"][axis] + c["bbox"][axis + 2]) / 2.0, c) for c in cands
        ]
        for seed, _ in centers:
            group = [c for center, c in centers if abs(center - seed) <= tol]
            if len(group) > len(best):
                best = group
    return best


def _fill_pattern_gap(group: List[dict], region_bgr) -> Optional[dict]:
    """Synthesize the one mark missing from a regular row/column pattern.

    A heavy pen cross destroys the printed outline, so the marked option is
    the one most often *not* detected. With n-1 marks found in a regular
    pattern, an inner spacing gap of ~2x the median pitch pinpoints it; the
    synthesized slot is then classified from the image like any other mark.
    """
    if len(group) < 2:
        return None
    # choose the axis with the larger spread (the pattern axis)
    xs = [(c["bbox"][0] + c["bbox"][2]) / 2.0 for c in group]
    ys = [(c["bbox"][1] + c["bbox"][3]) / 2.0 for c in group]
    axis = 0 if (max(xs) - min(xs)) >= (max(ys) - min(ys)) else 1
    coords = sorted(xs if axis == 0 else ys)
    if len(coords) < 2:
        return None
    diffs = [b - a for a, b in zip(coords, coords[1:])]
    med = float(np.median(diffs))
    if med <= 4:
        return None
    gaps = [i for i, d in enumerate(diffs) if 1.6 * med <= d <= 2.6 * med]
    if len(gaps) != 1:
        return None
    center = coords[gaps[0]] + diffs[gaps[0]] / 2.0
    med_w = float(np.median([c["bbox"][2] - c["bbox"][0] for c in group]))
    med_h = float(np.median([c["bbox"][3] - c["bbox"][1] for c in group]))
    cross_center = float(np.median(ys if axis == 0 else xs))
    cx, cy = (center, cross_center) if axis == 0 else (cross_center, center)
    H, W = region_bgr.shape[:2]
    x1 = max(0, int(cx - med_w / 2))
    y1 = max(0, int(cy - med_h / 2))
    x2 = min(W, int(cx + med_w / 2))
    y2 = min(H, int(cy + med_h / 2))
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    verdict = detect_markup_geometric(region_bgr[y1:y2, x1:x2])
    verdict["bbox"] = [x1, y1, x2, y2]
    verdict["h"] = y2 - y1
    verdict["recovered_from_gap"] = True
    return verdict


def _looks_like_box(ink, x, y, w, h) -> bool:
    """Ink runs along all four sides of the bbox (a printed checkbox)."""
    band = max(2, min(w, h) // 8)
    top = ink[y : y + band, x : x + w]
    bottom = ink[y + h - band : y + h, x : x + w]
    left = ink[y : y + h, x : x + band]
    right = ink[y : y + h, x + w - band : x + w]
    cov_h = lambda strip: float(np.mean(np.any(strip > 0, axis=0)))  # noqa: E731
    cov_v = lambda strip: float(np.mean(np.any(strip > 0, axis=1)))  # noqa: E731
    return (
        min(cov_h(top), cov_h(bottom)) >= 0.55
        and min(cov_v(left), cov_v(right)) >= 0.55
    )


def _looks_like_circle(cnt) -> bool:
    """Most contour points lie on the min-enclosing circle (a printed ○)."""
    import cv2

    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    if r < 4:
        return False
    pts = cnt.reshape(-1, 2).astype(float)
    dist = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    on_ring = np.abs(dist - r) <= max(2.0, 0.18 * r)
    return float(np.mean(on_ring)) >= 0.65


def _is_isolated(ink, x, y, w, h, W) -> bool:
    """Reject shapes with ink immediately left/right (letters inside words).

    At 200 DPI letters are ~1-3 px from their neighbours while survey marks
    have >=6 px clearance to their label, so the strip must be wide enough to
    reach a neighbouring letter.
    """
    pad = max(5, w // 2)
    for x1, x2 in ((max(0, x - pad), x), (x + w, min(W, x + w + pad))):
        if x2 <= x1:
            continue
        strip = ink[y : y + h, x1:x2]
        if strip.size and np.count_nonzero(strip) / strip.size > 0.10:
            return False
    return True


def _most_uniform_subset(cands: List[dict], n: int) -> List[dict]:
    """The n candidates with the smallest height spread (marks are uniform)."""
    by_h = sorted(cands, key=lambda c: c["h"])
    best, best_spread = by_h[:n], float("inf")
    for i in range(len(by_h) - n + 1):
        window = by_h[i : i + n]
        spread = window[-1]["h"] - window[0]["h"]
        if spread < best_spread:
            best, best_spread = window, spread
    return list(best)


# ==========================================
#        4. RECONCILIATION
# ==========================================


def reconcile_marks(
    glyph_items: List[Dict[str, Any]],
    geo_marks: List[Dict[str, Any]],
    *,
    allow_override: bool = False,
) -> Tuple[List[Dict[str, Any]], str]:
    """Compare VL glyph states with geometric verdicts (position-aligned).

    Returns ``(items, status)`` where status is ``"matched"``,
    ``"count_mismatch"``, ``"no-geometry"`` or ``"geometry_saturated"``. Only
    By default geometry is an observation/review trigger only: it never changes
    the OCR state. ``allow_override=True`` preserves the old experimental
    unchecked->checked behaviour for the frozen benchmark baseline, but must
    not be used by the production pipeline.

    Safety guard: if geometry claims *every* mark in a multi-mark region is
    inked, the signal is non-discriminative (bad crop, shading, systematic
    binarization failure) — real survey rows are never all-checked. The VL
    glyph states are then kept untouched.
    """
    items = [dict(g) for g in glyph_items]
    if not geo_marks:
        return items, "no-geometry"
    if len(geo_marks) != len(items):
        return items, "count_mismatch"
    if len(items) >= 4 and all(m["state"] == "checked" for m in geo_marks):
        return items, "geometry_saturated"
    disagreements = 0
    for item, mark in zip(items, geo_marks):
        if mark.get("bbox"):
            item["geo_bbox"] = mark["bbox"]
        item["geometry"] = {
            k: mark.get(k) for k in ("state", "score", "fill_ratio", "strike")
        }
        if (
            item["state"] == "unchecked"
            and mark["state"] == "checked"
            and (mark.get("score") or 0.0) >= 0.55
        ):
            disagreements += 1
            item["needs_review"] = True
            if allow_override:
                item.update(
                    state="checked",
                    method="geometric-override",
                    score=mark.get("score"),
                )
        elif (
            item["state"] == "checked"
            and mark["state"] == "unchecked"
            and (mark.get("score") or 0.0) >= 0.55
        ):
            disagreements += 1
            item["needs_review"] = True
    return items, "geometry_disagreement" if disagreements else "matched"


def summarize_items(items: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {
        "n_checked": 0,
        "n_unchecked": 0,
        "n_uncertain": 0,
        "n_overridden": 0,
        "n_disagreements": 0,
    }
    for item in items:
        state = item.get("state")
        if state == "checked":
            counts["n_checked"] += 1
        elif state == "unchecked":
            counts["n_unchecked"] += 1
        else:
            counts["n_uncertain"] += 1
        if item.get("method") == "geometric-override":
            counts["n_overridden"] += 1
        if item.get("needs_review"):
            counts["n_disagreements"] += 1
    return counts


# ==========================================
#        5. COMBINED SINGLE-CHECKBOX CLASSIFIER
# ==========================================


def classify_checkbox(block_content: Optional[str] = None, crop_bgr=None) -> Dict[str, Any]:
    """Record VL and geometric observations without geometric correction."""
    vl = detect_markup_from_vl(block_content)
    geo = detect_markup_geometric(crop_bgr) if crop_bgr is not None else None
    observations = []
    if vl:
        observations.append({"source": "paddleocr-vl", **vl})
    if geo:
        observations.append({"source": "geometric", **geo})

    state = vl["state"] if vl else "uncertain"
    method = "vl" if vl else "unverified"
    score = vl.get("score", 0.0) if vl else 0.0
    disagreement = bool(
        vl
        and geo
        and geo.get("state") not in ("uncertain", vl.get("state"))
        and (geo.get("score") or 0.0) >= 0.55
    )
    return {
        "state": state,
        "method": method,
        "score": score,
        "status": "geometry_disagreement" if disagreement else ("observed" if vl else "needs_review"),
        "observations": observations,
    }


def classify_from_b64(block_content: Optional[str], crop_b64: Optional[str]) -> Dict[str, Any]:
    """Convenience wrapper: decode a base64 crop, then :func:`classify_checkbox`."""
    return classify_checkbox(block_content, decode_crop_b64(crop_b64))
