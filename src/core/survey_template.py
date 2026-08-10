"""Template-first survey extraction: the printed form, learned once per batch.

A batch of questionnaires is N copies of one printed form. Registering the
copies and taking a per-pixel median cancels the respondent ink (it lands
somewhere different in every copy) and leaves the printed form, so a blank
template can be synthesized without ever being given one.

Locating the response controls on that blank is the whole point: from then on
no model has to infer the structure of the form, and reading a respondent is
"is this known box marked?" instead of "what questions does this page have?".
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core import markup_detect

DEFAULT_DPI = 300  # matches auto_ocr.SURVEY_DPI
MIN_REGISTRATION_QUALITY = 0.30
CIRCLE_MAX_EXTENT = 0.85  # bbox fill below this is a ring, above it a square
_ORB_FEATURES = 8000
_ORB_SCALE = 0.25
_MEDIAN_STRIP = 400  # rows per pass, so a 4960x3507 stack stays off the heap


# ==========================================
#              1. RASTERIZATION
# ==========================================


def page_count(pdf_path) -> int:
    import fitz

    with fitz.open(str(pdf_path)) as doc:
        return doc.page_count


def render_gray(pdf_path, page_index: int, dpi: int = DEFAULT_DPI):
    """Render one PDF page to a grayscale array."""
    import cv2
    import fitz

    with fitz.open(str(pdf_path)) as doc:
        pix = doc.load_page(page_index).get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        if pix.n == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img.reshape(pix.height, pix.width).copy()


# ==========================================
#              2. REGISTRATION
# ==========================================


def register(moving_gray, reference_gray, scale: float = _ORB_SCALE):
    """Homography mapping *moving_gray* onto *reference_gray*.

    Returns ``(H, quality)`` where quality is the RANSAC inlier ratio; ``H`` is
    None when the pages could not be matched. Features are found on downscaled
    copies (the forms are near-identical, so this is both faster and steadier)
    and the transform is lifted back to full resolution.
    """
    import cv2

    m_small = cv2.resize(moving_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    r_small = cv2.resize(reference_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    orb = cv2.ORB_create(nfeatures=_ORB_FEATURES)
    kp_m, des_m = orb.detectAndCompute(m_small, None)
    kp_r, des_r = orb.detectAndCompute(r_small, None)
    if des_m is None or des_r is None:
        return None, 0.0

    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_m, des_r, k=2)
    good = [a for a, b in (p for p in pairs if len(p) == 2) if a.distance < 0.75 * b.distance]
    if len(good) < 12:
        return None, 0.0

    src = np.float32([kp_m[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_r[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
    H_small, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H_small is None:
        return None, 0.0

    quality = float(mask.sum()) / len(good)
    S = np.diag([scale, scale, 1.0]).astype(np.float64)
    return np.linalg.inv(S) @ H_small @ S, quality


def warp_to_reference(moving_gray, H, shape: Tuple[int, int]):
    """Warp *moving_gray* into the reference frame, padding with paper white."""
    import cv2

    height, width = shape
    return cv2.warpPerspective(
        moving_gray, H, (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


# ==========================================
#            3. CONSENSUS BLANK
# ==========================================


def consensus_blank(images: Sequence[np.ndarray]):
    """Per-pixel median of registered copies: ink cancels, the form survives.

    Median rather than max because it tolerates a pixel of registration error
    without eroding thin printed rules.
    """
    if not images:
        raise ValueError("consensus_blank needs at least one image")
    height, width = images[0].shape[:2]
    out = np.empty((height, width), np.uint8)
    for y in range(0, height, _MEDIAN_STRIP):
        y2 = min(height, y + _MEDIAN_STRIP)
        block = np.stack([im[y:y2] for im in images])
        out[y:y2] = np.median(block, axis=0).astype(np.uint8)
    return out


@dataclass
class BlankPage:
    """A synthesized blank plus the registration record that produced it."""

    page_index: int
    image: Any
    reference: str
    contributors: List[Tuple[str, float]] = field(default_factory=list)
    failures: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.image.shape[:2]


def build_blank(
    pdf_paths: Sequence[Any],
    page_index: int,
    *,
    dpi: int = DEFAULT_DPI,
    min_quality: float = MIN_REGISTRATION_QUALITY,
) -> BlankPage:
    """Register every copy of one page onto the first and median them."""
    import cv2

    paths = [pathlib.Path(p) for p in pdf_paths]
    if not paths:
        raise ValueError("build_blank needs at least one document")

    reference = render_gray(paths[0], page_index, dpi)
    stack = [reference]
    contributors: List[Tuple[str, float]] = [(paths[0].name, 1.0)]
    failures: List[Tuple[str, float]] = []

    for path in paths[1:]:
        moving = render_gray(path, page_index, dpi)
        if moving.shape != reference.shape:
            moving = cv2.resize(moving, (reference.shape[1], reference.shape[0]))
        H, quality = register(moving, reference)
        if H is None or quality < min_quality:
            failures.append((path.name, quality))
            continue
        stack.append(warp_to_reference(moving, H, reference.shape))
        contributors.append((path.name, quality))

    return BlankPage(
        page_index=page_index,
        image=consensus_blank(stack),
        reference=paths[0].name,
        contributors=contributors,
        failures=failures,
    )


# ==========================================
#           4. CONTROL DETECTION
# ==========================================


def _modal_width(widths: Sequence[int], tolerance: float = 0.22) -> Optional[Tuple[int, int]]:
    """Width band of the dominant candidate size cluster.

    Printed controls on one form are all the same size, so they form a tight
    mode; leftover text glyphs are smaller and more scattered. Deriving the
    band from the data keeps this DPI- and form-independent.
    """
    if not widths:
        return None
    values = np.asarray(sorted(widths), dtype=float)
    best, best_count = None, 0
    for centre in values:
        lo, hi = centre * (1 - tolerance), centre * (1 + tolerance)
        count = int(np.count_nonzero((values >= lo) & (values <= hi)))
        if count > best_count:
            best, best_count = centre, count
    if best is None:
        return None
    return int(round(best * (1 - tolerance))), int(round(best * (1 + tolerance)))


def find_controls(
    gray,
    *,
    min_side: int = 14,
    max_side: int = 60,
) -> List[Dict[str, Any]]:
    """Locate every empty response control on a blank form.

    Candidates must be square-ish, hollow, isolated from surrounding text, and
    fall in the dominant size cluster. Returns reading-order dicts with pixel
    ``bbox`` and ``shape``.
    """
    import cv2

    ink = markup_detect._ink_mask(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_width = gray.shape[1]

    candidates: List[Dict[str, Any]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if not (min_side <= w <= max_side and min_side <= h <= max_side):
            continue
        if not (0.72 <= w / float(h) <= 1.39):
            continue
        extent = cv2.contourArea(cnt) / float(w * h)
        if extent < 0.55:
            continue
        interior = ink[y + h // 4: y + 3 * h // 4, x + w // 4: x + 3 * w // 4]
        if interior.size == 0 or interior.mean() > 40:
            continue
        if not markup_detect._is_isolated(ink, x, y, w, h, page_width):
            continue
        if not (
            markup_detect._looks_like_circle(cnt)
            or markup_detect._looks_like_box(ink, x, y, w, h)
        ):
            continue
        # Shape decides single-choice (○) from multi-select (□) downstream, and
        # the two separate cleanly on how much of the bbox the outline encloses:
        # a ring reaches pi/4, a square outline nearly 1.
        candidates.append({
            "bbox": [int(x), int(y), int(x + w), int(y + h)],
            "shape": "circle" if extent < CIRCLE_MAX_EXTENT else "box",
            "w": int(w),
        })

    band = _modal_width([c["w"] for c in candidates])
    if band is not None:
        lo, hi = band
        candidates = [c for c in candidates if lo <= c["w"] <= hi]

    for candidate in candidates:
        candidate.pop("w", None)
    candidates.sort(key=lambda c: (c["bbox"][1] // 20, c["bbox"][0]))
    return candidates


# ==========================================
#         5. TEMPLATE DIFFERENCING
# ==========================================

RESIDUAL_INK = 40         # gray levels darker than the blank to count as new ink
RESIDUAL_CHECKED = 0.08   # >= this fraction of the interior inked -> checked
RESIDUAL_EMPTY = 0.025    # <= this fraction -> unchecked
_RESIDUAL_TOLERANCE = 9   # px of registration slack absorbed before differencing
_RESIDUAL_MARGIN = 0.26   # fraction trimmed per side, clearing the printed outline


def residual_ink(blank_gray, registered_gray):
    """Ink the respondent added, with the printed form removed.

    Eroding the blank first takes the darkest pixel in a small neighbourhood,
    so a couple of pixels of registration error cannot leave a halo of the
    printed rules behind and be mistaken for a mark.
    """
    import cv2

    kernel = np.ones((_RESIDUAL_TOLERANCE, _RESIDUAL_TOLERANCE), np.uint8)
    return cv2.subtract(cv2.erode(blank_gray, kernel), registered_gray)


def classify_residual(residual_crop) -> Dict[str, Any]:
    """Mark state from the residual inside one control.

    Takes the residual over the control's full bbox and measures its interior.
    Differencing removes the printed outline and the interior margin absorbs
    what registration slack leaves of it, so on this batch the fill ratio is
    sharply bimodal: unmarked controls sit at exactly 0.0 and marked ones above
    0.2, with 1.6% of cells in between.
    """
    if residual_crop is None or residual_crop.size == 0:
        return {"state": "uncertain", "method": "residual", "score": 0.0, "fill_ratio": None}

    height, width = residual_crop.shape[:2]
    my, mx = int(height * _RESIDUAL_MARGIN), int(width * _RESIDUAL_MARGIN)
    interior = residual_crop[my: height - my, mx: width - mx]
    if interior.size == 0:
        return {"state": "uncertain", "method": "residual", "score": 0.0, "fill_ratio": None}

    fill = float(np.count_nonzero(interior > RESIDUAL_INK)) / interior.size

    if fill >= RESIDUAL_CHECKED:
        state = "checked"
        score = min(1.0, 0.6 + (fill - RESIDUAL_CHECKED) * 4.0)
    elif fill <= RESIDUAL_EMPTY:
        state = "unchecked"
        score = min(1.0, 0.6 + (RESIDUAL_EMPTY - fill) * 20.0)
    else:
        state = "uncertain"
        span = RESIDUAL_CHECKED - RESIDUAL_EMPTY
        score = max(0.0, 0.5 - min(RESIDUAL_CHECKED - fill, fill - RESIDUAL_EMPTY) / span)

    return {
        "state": state,
        "method": "residual",
        "score": round(score, 3),
        "fill_ratio": round(fill, 4),
    }


def overlay(gray, controls: Sequence[Dict[str, Any]], states: Optional[Sequence[str]] = None):
    """Audit image: every control outlined, coloured by state when given."""
    import cv2

    colors = {
        "checked": (0, 170, 0),
        "unchecked": (170, 170, 170),
        "uncertain": (0, 140, 255),
    }
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for index, control in enumerate(controls):
        x1, y1, x2, y2 = [int(v) for v in control["bbox"]]
        if states is not None and index < len(states):
            color = colors.get(states[index], (0, 0, 255))
        else:
            color = (0, 160, 0) if control.get("shape") == "circle" else (220, 60, 0)
        cv2.rectangle(vis, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), color, 2)
    return vis


# ==========================================
#             5. THE TEMPLATE
# ==========================================


@dataclass
class TemplateControl:
    """One response control, in page-normalized coordinates."""

    id: str
    bbox: List[float]  # [x1, y1, x2, y2] as fractions of page width/height
    shape: str = "circle"
    label: str = ""
    question_id: str = ""
    row_id: str = ""  # the answer group this control competes in

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "bbox": [round(float(v), 6) for v in self.bbox],
            "shape": self.shape,
            "label": self.label,
            "question_id": self.question_id,
            "row_id": self.row_id,
        }

    def pixel_bbox(self, width: int, height: int) -> List[int]:
        x1, y1, x2, y2 = self.bbox
        return [
            max(0, int(round(x1 * width))),
            max(0, int(round(y1 * height))),
            min(width, int(round(x2 * width))),
            min(height, int(round(y2 * height))),
        ]


@dataclass
class TemplatePage:
    page_index: int
    width: int
    height: int
    controls: List[TemplateControl] = field(default_factory=list)
    blank_png_b64: Optional[str] = None

    def blank_filename(self, stem: str) -> str:
        return f"{stem}_blank_page{self.page_index + 1}.png"

    def to_dict(self, stem: str = "") -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "page_index": self.page_index,
            "width": self.width,
            "height": self.height,
            "controls": [c.to_dict() for c in self.controls],
        }
        if self.blank_png_b64 and stem:
            data["blank_image"] = self.blank_filename(stem)
        return data


@dataclass
class SurveyTemplate:
    """The printed form: where every response control is, in every page."""

    pages: List[TemplatePage] = field(default_factory=list)
    dpi: int = DEFAULT_DPI
    provenance: Dict[str, Any] = field(default_factory=dict)
    rules: Dict[str, str] = field(default_factory=dict)  # row_id -> single|multiple
    row_labels: Dict[str, str] = field(default_factory=dict)  # row_id -> printed stem

    @property
    def control_count(self) -> int:
        return sum(len(page.controls) for page in self.pages)

    def to_dict(self, stem: str = "") -> Dict[str, Any]:
        return {
            "dpi": self.dpi,
            "provenance": dict(self.provenance),
            "rules": dict(self.rules),
            "row_labels": dict(self.row_labels),
            "pages": [page.to_dict(stem) for page in self.pages],
        }

    def save(self, path) -> None:
        """Write the template JSON plus one blank raster per page beside it.

        The blanks are the registration reference every later read needs, so
        they travel with the template rather than being embedded in the JSON
        (they are megabytes each, and useful to open on their own).
        """
        import base64

        path = pathlib.Path(path)
        stem = path.stem
        path.write_text(
            json.dumps(self.to_dict(stem), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        for page in self.pages:
            if page.blank_png_b64:
                (path.parent / page.blank_filename(stem)).write_bytes(
                    base64.b64decode(page.blank_png_b64)
                )

    @classmethod
    def load(cls, path) -> "SurveyTemplate":
        import base64

        path = pathlib.Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        template = cls(
            dpi=int(raw.get("dpi", DEFAULT_DPI)),
            provenance=dict(raw.get("provenance") or {}),
            rules=dict(raw.get("rules") or {}),
            row_labels=dict(raw.get("row_labels") or {}),
            pages=[
                TemplatePage(
                    page_index=int(page["page_index"]),
                    width=int(page["width"]),
                    height=int(page["height"]),
                    controls=[
                        TemplateControl(
                            id=str(control["id"]),
                            bbox=[float(v) for v in control["bbox"]],
                            shape=str(control.get("shape") or "circle"),
                            label=str(control.get("label") or ""),
                            question_id=str(control.get("question_id") or ""),
                            row_id=str(control.get("row_id") or ""),
                        )
                        for control in page.get("controls") or []
                    ],
                )
                for page in raw.get("pages") or []
            ],
        )

        for page, page_raw in zip(template.pages, raw.get("pages") or []):
            blank = page_raw.get("blank_image")
            if not blank:
                continue
            blank_path = path.parent / blank
            if blank_path.exists():
                page.blank_png_b64 = base64.b64encode(
                    blank_path.read_bytes()
                ).decode("ascii")
        return template


def reading_order(controls) -> List[Any]:
    """Printed order within one answer group.

    Left-to-right for a row, top-to-bottom for a vertical list. The vertical
    key is quantized by control height: controls printed on one line still
    differ by a few pixels, and a finer quantum lets y outrank x and scrambles
    the row.
    """
    if not controls:
        return []
    heights = sorted(c.bbox[3] - c.bbox[1] for c in controls)
    quantum = max(1e-6, heights[len(heights) // 2])
    return sorted(
        controls,
        key=lambda c: (round(((c.bbox[1] + c.bbox[3]) / 2) / quantum), c.bbox[0]),
    )


def _rows_of(controls, width: int, height: int) -> List[List[Any]]:
    """Split controls into left-to-right runs sharing a baseline."""
    rows: Dict[int, List[Any]] = {}
    for control in controls:
        x1, y1, x2, y2 = control.pixel_bbox(width, height)
        tolerance = max(6, (y2 - y1) // 2)
        centre = (y1 + y2) // 2
        key = next((k for k in rows if abs(k - centre) <= tolerance), centre)
        rows.setdefault(key, []).append(control)
    return [
        sorted(members, key=lambda c: c.pixel_bbox(width, height)[0])
        for _, members in sorted(rows.items())
    ]


def _answer_groups(controls, width: int, height: int) -> List[List[Any]]:
    """Partition one question's controls into the sets that compete together.

    A row holding several controls is one answer (a matrix row, a rating scale,
    a "Ja / Nein" pair). A run of consecutive rows holding one control each is a
    vertical option list, which is also one answer.
    """
    groups: List[List[Any]] = []
    pending: List[Any] = []
    for row in _rows_of(controls, width, height):
        if len(row) >= 2:
            if pending:
                groups.append(pending)
                pending = []
            groups.append(row)
        else:
            pending.append(row[0])
    if pending:
        groups.append(pending)
    return groups


def infer_structure(template: "SurveyTemplate") -> Dict[str, str]:
    """Work out what each control competes with, and under which rule.

    Controls are split by printed shape before anything else: this form, like
    most, draws single choice as ○ and multi-select as □, and the two are often
    laid out side by side within one question. Each resulting answer group then
    takes its rule from that shape.
    """
    template.rules = {}
    for page in template.pages:
        by_question: Dict[str, List[TemplateControl]] = {}
        for control in page.controls:
            key = (control.question_id or f"p{page.page_index + 1}", control.shape)
            by_question.setdefault(key, []).append(control)

        counters: Dict[str, int] = {}
        for (question_id, shape), controls in by_question.items():
            for group in _answer_groups(controls, page.width, page.height):
                counters[question_id] = counters.get(question_id, 0) + 1
                row_id = question_id
                if len(by_question) > 1 or counters[question_id] > 1:
                    row_id = f"{question_id}_r{counters[question_id]:02d}"
                for control in group:
                    control.row_id = row_id
                template.rules[row_id] = "multiple" if shape == "box" else "single"
    return template.rules


def build_template(
    pdf_paths: Sequence[Any],
    *,
    dpi: int = DEFAULT_DPI,
    keep_blanks: bool = True,
) -> Tuple[SurveyTemplate, List[BlankPage]]:
    """Synthesize the blank form from a batch and locate its controls."""
    import base64

    import cv2

    paths = [pathlib.Path(p) for p in pdf_paths]
    if not paths:
        raise ValueError("build_template needs at least one document")

    n_pages = page_count(paths[0])
    template = SurveyTemplate(dpi=dpi)
    blanks: List[BlankPage] = []

    for page_index in range(n_pages):
        blank = build_blank(paths, page_index, dpi=dpi)
        blanks.append(blank)
        height, width = blank.shape
        controls = find_controls(blank.image)

        page = TemplatePage(page_index=page_index, width=width, height=height)
        for order, control in enumerate(controls, start=1):
            x1, y1, x2, y2 = control["bbox"]
            page.controls.append(
                TemplateControl(
                    id=f"p{page_index + 1}_c{order:03d}",
                    bbox=[x1 / width, y1 / height, x2 / width, y2 / height],
                    shape=control["shape"],
                )
            )
        if keep_blanks:
            ok, encoded = cv2.imencode(".png", blank.image)
            if ok:
                page.blank_png_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        template.pages.append(page)

    template.provenance = {
        "documents": [p.name for p in paths],
        "dpi": dpi,
        "registration": {
            f"page_{b.page_index + 1}": {
                "reference": b.reference,
                "contributors": len(b.contributors),
                "failures": [name for name, _ in b.failures],
                "min_quality": round(min((q for _, q in b.contributors), default=0.0), 3),
            }
            for b in blanks
        },
    }
    return template, blanks
