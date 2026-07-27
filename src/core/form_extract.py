"""Question-level survey/form response extraction from high-resolution crops.

This module is intentionally activated only when the user requests survey
analysis. Paddle regions provide printed-text/schema hints; a targeted VLM reads
the actual marks. The result is stored as :mod:`doc_ir` form annotations and
never mutates the OCR transcription.
"""

from __future__ import annotations

import base64
import copy
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser
import json
import math
import os
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from core import doc_ir, markup_detect
except ImportError:  # pragma: no cover - standalone imports
    import doc_ir  # type: ignore
    import markup_detect  # type: ignore


_QUESTION_START = re.compile(r"^\s*(\d{1,3})\s*([).:])\s*", re.UNICODE)
_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")
_PADDLE_MARK_NOTATION = re.compile(
    r"(?:\\(?:bigotimes|boxtimes|square|Box|bigcirc)\b|[⊗⊠])"
)
_BINARY_OPTION_PAIRS = (
    ("ja", "nein"),
    ("yes", "no"),
    ("oui", "non"),
    ("sí", "no"),
)
MAX_FORM_SECTIONS_PER_PAGE = 30
SCHEMA_FREE_CONTRACT = "schema-free-v2"
PADDLE_ID_CONTRACT = "paddle-id-v1"
DEFAULT_SURVEY_CONTRACT = SCHEMA_FREE_CONTRACT
_VALID_STATES = {"selected", "cancelled", "ambiguous"}
_VALID_VISUAL_MARKS = {"x", "tick", "filled", "scribbled", "other", "uncertain"}
_VALID_RESPONSE_TYPES = {"single", "multiple", "rating", "matrix", "unknown"}
_VALID_SELECTION_RULES = {"zero_or_one", "zero_or_more", "one_per_row", "unknown"}


def _model_has_release_approval(client) -> bool:
    approved = {
        name.strip()
        for name in os.environ.get("TEXTLAB_APPROVED_SURVEY_MODELS", "").split(",")
        if name.strip()
    }
    return str(getattr(client, "model", "")) in approved


@dataclass
class SameLayoutTemplate:
    """In-job crop template learned from the first batch document.

    Coordinates are normalized to page dimensions. This is intentionally a
    conservative first step toward registered template differencing: it reuses
    question locations, but the VLM still reconstructs and reads every response
    image and no answer state is copied between documents. Switching later
    respondents to a frozen ID-only structure remains gated on drift validation.
    """

    pages: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)
    learned_pages: int = 0
    reused_pages: int = 0

    def get(self, page_number: int, width: int, height: int) -> List[Dict[str, Any]]:
        specs = self.pages.get(page_number) or []
        if not specs:
            return []
        self.reused_pages += 1
        jobs = []
        for spec in specs:
            nx1, ny1, nx2, ny2 = spec["normalized_bbox"]
            jobs.append(
                {
                    "bbox": [
                        max(0, int(round(nx1 * width))),
                        max(0, int(round(ny1 * height))),
                        min(width, int(round(nx2 * width))),
                        min(height, int(round(ny2 * height))),
                    ],
                    "group_id": spec["group_id"],
                    "section_id": spec.get("section_id", spec["group_id"]),
                    "schema_hint": copy.deepcopy(spec["schema_hint"]),
                    "candidate_evidence": list(spec.get("candidate_evidence") or []),
                    "validation_text": str(spec.get("validation_text") or ""),
                    "question_number": spec.get("question_number"),
                    "template_reused": True,
                }
            )
        return jobs

    def learn(self, page_number: int, width: int, height: int, jobs: List[Dict[str, Any]]):
        if page_number in self.pages or width <= 0 or height <= 0:
            return
        specs = []
        for job in jobs:
            x1, y1, x2, y2 = job["bbox"]
            specs.append(
                {
                    "normalized_bbox": [x1 / width, y1 / height, x2 / width, y2 / height],
                    "group_id": job["group_id"],
                    "section_id": job.get("section_id", job["group_id"]),
                    "schema_hint": copy.deepcopy(job["schema_hint"]),
                    "candidate_evidence": list(job.get("candidate_evidence") or []),
                    "validation_text": str(job.get("validation_text") or ""),
                    "question_number": job.get("question_number"),
                }
            )
        if specs:
            self.pages[page_number] = specs
            self.learned_pages += 1

def _plain(value: str) -> str:
    return _SPACE.sub(" ", unescape(_TAG.sub(" ", str(value or "")))).strip()


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _normalized_label(value: str) -> str:
    """Conservative comparison form for a model's positional text echo."""
    value = unicodedata.normalize("NFKC", str(value or ""))
    value = _SPACE.sub(" ", value).strip()
    value = value.strip(" \t\r\n.,;:!?()[]{}'\"")
    return value.casefold()


def _clean_question_text(value: str) -> str:
    """Keep response glyphs and their answer labels out of Paddle prompts."""
    text = _plain(value)
    glyphs = markup_detect.extract_mark_glyphs(text)
    if glyphs:
        text = text[: glyphs[0]["index"]]
    return text.strip(" \t|,.;:-")


def _survey_contract(value: Optional[str]) -> str:
    contract = value or os.environ.get("TEXTLAB_SURVEY_CONTRACT", DEFAULT_SURVEY_CONTRACT)
    contract = contract.strip().casefold()
    aliases = {
        # v1 remains accepted as a CLI/config spelling, but resolves to the
        # current contract so an old launch command cannot silently retain the
        # weaker validator.
        "schema-free-v1": SCHEMA_FREE_CONTRACT,
        "schema-free": SCHEMA_FREE_CONTRACT,
        "universal": SCHEMA_FREE_CONTRACT,
        "paddle": PADDLE_ID_CONTRACT,
        "id-only": PADDLE_ID_CONTRACT,
    }
    contract = aliases.get(contract, contract)
    if contract not in {SCHEMA_FREE_CONTRACT, PADDLE_ID_CONTRACT}:
        raise ValueError(
            f"Unknown survey contract {contract!r}; expected "
            f"{SCHEMA_FREE_CONTRACT!r} or {PADDLE_ID_CONTRACT!r}"
        )
    return contract


def _slug(value: str, fallback: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "")).strip("_").lower()
    return slug[:64] or fallback


def _label_is_resolved(value: str) -> bool:
    text = _plain(value)
    if not text or _key(text) in {"na", "none", "unknown"}:
        return False
    without_marks = re.sub(r"[○◯◉●☐☑☒✓✔✗✘×]", "", text).strip(" |,.;:-")
    return bool(without_marks)


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: List[List[str]] = []
        self._row: Optional[List[str]] = None
        self._cell: Optional[List[str]] = None

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append(_SPACE.sub(" ", "".join(self._cell)).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def _table_rows(html: str) -> List[List[str]]:
    parser = _TableParser()
    try:
        parser.feed(html or "")
    except Exception:
        return []
    return parser.rows


def _question_number(text: str) -> Optional[str]:
    match = _QUESTION_START.match(_plain(text))
    return match.group(1) if match else None


def _is_numbered_question_anchor(region: "doc_ir.Region", text: str) -> bool:
    """Distinguish question numbers from numbered list items.

    Parenthesis/colon numbering is normally form structure. A period is more
    ambiguous, so require Paddle to classify it as a title or require direct
    question/mark evidence. This prevents prize lists such as ``1. voucher``
    from becoming giant form sections while retaining common ``1. Question?``
    forms.
    """
    match = _QUESTION_START.match(text)
    if not match:
        return False
    punctuation = match.group(2)
    return (
        punctuation in "):"
        or region.type == doc_ir.TITLE
        or "?" in text
        or bool(markup_detect.extract_mark_glyphs(text))
    )


def _question_sections(page: "doc_ir.Page") -> List[Dict[str, Any]]:
    """Build high-recall question sections from Paddle's reading order.

    Numbered questions are the strongest boundary, but forms are not required
    to be numbered. Question-like text anchors and otherwise-unclaimed tables,
    mark-bearing text, and dedicated checkbox regions become fallback sections.
    This runs only after the user explicitly requested survey extraction.
    """
    ordered = page.ordered_regions()
    sections: List[Dict[str, Any]] = []
    numbered_anchors = []
    question_anchors = []
    for index, region in enumerate(ordered):
        text = _plain(region.text)
        number = _question_number(text)
        entry = (index, number, text, region)
        if number is not None and _is_numbered_question_anchor(region, text):
            numbered_anchors.append(entry)
        elif "?" in text and 2 <= len(text) <= 700:
            question_anchors.append(entry)

    # A wrapped numbered question often puts its question mark in the next
    # Paddle block. Mixing that continuation into the anchor list splits the
    # actual question in two. Prefer numbered anchors whenever the page has
    # them; use question-mark anchors for genuinely unnumbered forms.
    anchors = numbered_anchors or question_anchors

    def _top(entry):
        region = entry[3]
        return float(region.bbox[1]) if len(region.bbox) >= 4 else float(entry[0]) * 1000

    def _center_x(region):
        return (float(region.bbox[0]) + float(region.bbox[2])) / 2 if len(region.bbox) >= 4 else 0.0

    page_width = float(page.width or max(
        (region.bbox[2] for region in ordered if len(region.bbox) >= 4),
        default=1,
    ))
    centres = sorted(_center_x(entry[3]) for entry in anchors if len(entry[3].bbox) >= 4)
    column_bounds = [(0.0, page_width)]
    if len(centres) >= 4 and page_width > 0:
        gaps = [(centres[i + 1] - centres[i], i) for i in range(len(centres) - 1)]
        largest_gap, gap_pos = max(gaps, default=(0.0, 0))
        if largest_gap >= page_width * 0.18:
            boundary = (centres[gap_pos] + centres[gap_pos + 1]) / 2
            column_bounds = [(0.0, boundary), (boundary, page_width)]

    def _column_for(region):
        centre = _center_x(region)
        for column, (left, right) in enumerate(column_bounds):
            if left <= centre <= right:
                return column
        return 0

    anchors.sort(
        key=lambda entry: (
            _column_for(entry[3]),
            _top(entry),
            entry[3].bbox[0] if entry[3].bbox else 0,
        )
    )

    covered = set()
    for anchor_pos, (start, number, anchor_text, anchor_region) in enumerate(anchors):
        crop_limits = None
        if len(anchor_region.bbox) >= 4:
            y_start = float(anchor_region.bbox[1])
            y_end = float("inf")
            anchor_column = _column_for(anchor_region)
            column_left, column_right = column_bounds[anchor_column]
            # Paddle title blocks are reliable visual section boundaries. Stop
            # at either the next question anchor or the next non-question title
            # in the same panel, whichever comes first.
            for candidate in ordered:
                if candidate is anchor_region or len(candidate.bbox) < 4:
                    continue
                if _column_for(candidate) != anchor_column:
                    continue
                next_top = float(candidate.bbox[1])
                is_anchor = any(candidate is entry[3] for entry in anchors)
                if not is_anchor and candidate.type != doc_ir.TITLE:
                    continue
                if next_top > y_start + 5:
                    y_end = min(y_end, next_top)
            tolerance = max(4.0, (anchor_region.bbox[3] - y_start) * 0.2)
            regions = [
                region
                for region in ordered
                if len(region.bbox) >= 4
                and float(region.bbox[1]) >= y_start - tolerance
                and float(region.bbox[1]) < y_end
                and column_left <= _center_x(region) <= column_right
            ]
            crop_limits = [column_left, y_start, column_right, y_end]
        else:
            next_index = anchors[anchor_pos + 1][0] if anchor_pos + 1 < len(anchors) else len(ordered)
            regions = ordered[start:next_index]
        if anchor_region not in regions:
            regions.insert(0, anchor_region)
        regions.sort(
            key=lambda region: (
                region.bbox[1] if len(region.bbox) >= 4 else region.reading_order,
                region.bbox[0] if len(region.bbox) >= 4 else 0,
            )
        )
        if not regions:
            continue
        sections.append(
            {
                "number": number,
                "regions": regions,
                "anchor": anchor_text,
                "crop_limits": crop_limits,
            }
        )
        covered.update(region.id for region in regions)

    # Fallback for unnumbered forms and for Paddle layouts where the question
    # and options were merged into a table/text block without a separate anchor.
    for index, region in enumerate(ordered):
        if region.id in covered:
            continue
        text = _plain(region.text)
        is_candidate = (
            region.type in (doc_ir.TABLE, doc_ir.CHECKBOX)
            or bool(markup_detect.extract_mark_glyphs(text))
        )
        if not is_candidate:
            continue
        regions = []
        if index > 0:
            previous = ordered[index - 1]
            if (
                previous.id not in covered
                and previous.type in (doc_ir.TEXT, doc_ir.TITLE)
                and _plain(previous.text)
                and _column_for(previous) == _column_for(region)
            ):
                regions.append(previous)
        regions.append(region)
        sections.append(
            {
                "number": None,
                "regions": regions,
                "anchor": _plain(regions[0].text) or text,
                "crop_limits": None,
            }
        )
        covered.update(item.id for item in regions)

    return sections[:MAX_FORM_SECTIONS_PER_PAGE]


def _section_bbox(
    regions: Iterable["doc_ir.Region"],
    width: int,
    height: int,
    crop_limits: Optional[Iterable[float]] = None,
) -> List[int]:
    boxes = [r.bbox for r in regions if len(r.bbox) >= 4]
    if not boxes:
        return []
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    mx = max(12, int((x2 - x1) * 0.035))
    my = max(12, int((y2 - y1) * 0.06))
    bbox = [
        max(0, int(x1) - mx), max(0, int(y1) - my),
        min(width, int(x2) + mx), min(height, int(y2) + my),
    ]
    if crop_limits is not None:
        limits = list(crop_limits)
        if len(limits) >= 4:
            left, top, right, bottom = limits[:4]
            # Keep a tiny allowance for Paddle's bounding-box uncertainty, but
            # never let generic crop padding pull in the previous/next question.
            bbox[0] = max(bbox[0], max(0, int(math.floor(left))))
            bbox[1] = max(bbox[1], max(0, int(math.floor(top)) - 3))
            bbox[2] = min(bbox[2], min(width, int(math.ceil(right))))
            if math.isfinite(bottom):
                bbox[3] = min(bbox[3], max(bbox[1] + 8, int(math.floor(bottom)) - 3))
    return bbox


def _crop(image, bbox):
    if image is None or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return image[y1:y2, x1:x2]


def _has_geometry_disagreement(page: "doc_ir.Page", bbox: List[int]) -> bool:
    if len(bbox) < 4:
        return False
    x1, y1, x2, y2 = bbox[:4]
    for region in page.regions:
        status = (region.markup or {}).get("status")
        if status not in ("geometry_disagreement", "count_mismatch"):
            continue
        if len(region.bbox) < 4:
            continue
        rx1, ry1, rx2, ry2 = region.bbox[:4]
        if min(x2, rx2) > max(x1, rx1) and min(y2, ry2) > max(y1, ry1):
            return True
    return False


def _png_bytes(image) -> bytes:
    import cv2

    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode form crop")
    return encoded.tobytes()


def _complete_section_view(crop) -> Tuple[int, int, int, int, Any]:
    """Return one unfragmented question image.

    Earlier versions split wide and tall sections into independent VLM calls.
    Those snippets could omit the question, row label, or the mark-to-option
    relationship. The vision backends already resize variable-resolution
    images, so the default policy is one complete Paddle-bounded question per
    request. If a future benchmark proves that a very large matrix needs zoom
    images, they must accompany this overview in the *same* request rather
    than replacing it with context-free calls.
    """
    height, width = crop.shape[:2]
    return 0, 0, width, height, crop


def _table_schema(region: "doc_ir.Region") -> List[dict]:
    rows = _table_rows(region.content.get("html", ""))
    hints = []
    for row_idx, cells in enumerate(rows):
        marked_cols = [i for i, cell in enumerate(cells) if markup_detect.extract_mark_glyphs(cell)]
        if len(marked_cols) < 2:
            continue
        header = []
        for previous in range(row_idx - 1, -1, -1):
            if len(rows[previous]) > max(marked_cols):
                candidate = rows[previous]
                resolved = sum(
                    _label_is_resolved(candidate[i]) for i in marked_cols
                )
                # A preceding response row contains non-empty circle glyphs,
                # but those are not column labels. Require at least two real
                # labels before accepting a row as a matrix header.
                if resolved >= min(2, len(marked_cols)):
                    header = candidate
                    break
        first_mark = min(marked_cols)
        label = " ".join(c for c in cells[:first_mark] if c).strip()
        options = []
        for option_pos, col_idx in enumerate(marked_cols, start=1):
            glyphs = markup_detect.extract_mark_glyphs(cells[col_idx])
            paddle_state = glyphs[0]["state"] if glyphs else "unknown"
            option_label = header[col_idx].strip() if header and col_idx < len(header) else ""
            if not _label_is_resolved(option_label):
                option_label = f"option {option_pos}"
            options.append(
                {
                    "option_id": f"{region.id}_r{row_idx}_o{option_pos}",
                    "label": option_label,
                    "paddle_state": paddle_state,
                }
            )
        hints.append(
            {
                "row_id": f"{region.id}_r{row_idx}",
                "label": label,
                "options": options,
            }
        )
    return hints


def _text_schema(region: "doc_ir.Region") -> List[dict]:
    text = _plain(region.text)
    glyphs = markup_detect.extract_mark_glyphs(text)
    if not glyphs:
        return []
    options = []
    for idx, item in enumerate(glyphs):
        start = item["index"] + 1
        end = glyphs[idx + 1]["index"] if idx + 1 < len(glyphs) else len(text)
        label = text[start:end].strip(" \t|:;,-") or f"option {idx + 1}"
        options.append(
            {
                "option_id": f"{region.id}_o{idx + 1}",
                "label": label,
                "paddle_state": item["state"],
            }
        )
    prefix = text[: glyphs[0]["index"]].strip(" \t|:;,-")
    return [{"row_id": region.id, "label": prefix, "options": options}]


def _schema_hint(section: dict, group_id: str) -> dict:
    rows = []
    seen = set()
    for region in section["regions"]:
        candidates = _table_schema(region) if region.type == doc_ir.TABLE else _text_schema(region)
        for row in candidates:
            key = row["row_id"]
            if key not in seen:
                seen.add(key)
                rows.append(row)
    return {
        "question_id": group_id,
        "question_text": _clean_question_text(section.get("anchor", ""))[:600],
        "rows": rows,
    }


def _has_binary_option_pair(section: dict) -> bool:
    """Recognise a compact yes/no option row when Paddle dropped its circles."""
    for region in section["regions"]:
        text = _plain(region.text).casefold()
        # Apply the lexical fallback only to a compact region. This avoids
        # treating prose such as participation conditions as a response row.
        if not text or len(text) > 260:
            continue
        words = set(re.findall(r"[^\W\d_]+", text, flags=re.UNICODE))
        if any(left in words and right in words for left, right in _BINARY_OPTION_PAIRS):
            return True
    return False


def _form_candidate_evidence(section: dict, crop) -> List[str]:
    """Return conservative reasons why a Paddle question needs mark reading.

    Paddle's transcription/layout is authoritative for proposal. Geometry is
    retained only as a fallback when it finds a repeated aligned mark pattern;
    one circle-like contour is never enough to submit a section to the VLM.
    Generic tables are deliberately not evidence because ordinary document
    tables caused false survey candidates in the original audit.
    """
    regions = section["regions"]
    text = " ".join(_plain(region.text) for region in regions)
    evidence = []
    if markup_detect.extract_mark_glyphs(text):
        evidence.append("paddle-mark-glyph")
    if _PADDLE_MARK_NOTATION.search(text):
        evidence.append("paddle-mark-notation")
    if any(region.type == doc_ir.CHECKBOX for region in regions):
        evidence.append("paddle-checkbox-region")
    if _has_binary_option_pair(section):
        evidence.append("paddle-binary-options")
    if evidence:
        return evidence

    # Recover cases such as Q11 where Paddle read the option labels but omitted
    # every printed circle. Requiring an aligned pair sharply narrows the old
    # single-contour fallback while leaving the geometry out of the VLM prompt.
    try:
        if len(markup_detect.find_marks(crop, n_expected=2)) >= 2:
            return ["aligned-geometric-mark-pattern"]
    except Exception:
        pass
    return []


def _section_validation_text(section: dict) -> str:
    """Independent printed-text cues used only for deterministic review rules."""
    return _SPACE.sub(
        " ", " ".join(_plain(region.text) for region in section.get("regions") or [])
    ).strip()


def _looks_like_form_section(section: dict, crop) -> bool:
    """Compatibility predicate used by tests and callers outside this module."""
    return bool(_form_candidate_evidence(section, crop))


def _compact_prompt_schema(schema_hint: dict) -> dict:
    """Remove repeated matrix labels and non-authoritative Paddle states.

    The previous raw hint could exceed the 4k context and was sliced mid-JSON.
    This representation always remains valid JSON and gives the model exactly
    the stable IDs it is allowed to return.
    """
    source_rows = schema_hint.get("rows") or []
    label_vectors = [
        [str(option.get("label") or "")[:160] for option in row.get("options") or []]
        for row in source_rows
    ]
    is_matrix = (
        len(source_rows) >= 2
        and label_vectors
        and len(label_vectors[0]) >= 2
        and all(vector == label_vectors[0] for vector in label_vectors[1:])
    )
    compact = {
        "question_id": schema_hint.get("question_id"),
        "question_text": str(schema_hint.get("question_text") or "")[:600],
    }
    if is_matrix:
        compact["columns"] = [
            {"position": index + 1, "label": label}
            for index, label in enumerate(label_vectors[0])
        ]
        compact["rows"] = [
            {
                "row_id": row.get("row_id"),
                "label": str(row.get("label") or "")[:220],
                "option_ids_by_column": [
                    option.get("option_id") for option in row.get("options") or []
                ],
            }
            for row in source_rows
        ]
    else:
        compact["rows"] = [
            {
                "row_id": row.get("row_id"),
                "label": str(row.get("label") or "")[:220],
                "options": [
                    {
                        "option_id": option.get("option_id"),
                        "label": str(option.get("label") or "")[:160],
                    }
                    for option in row.get("options") or []
                ],
            }
            for row in source_rows
        ]
    return compact


MARK_ONLY_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "visible_row_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "marks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "option_id": {"type": "string"},
                    "state": {
                        "type": "string",
                        "enum": ["selected", "cancelled", "ambiguous"],
                    },
                    "visual_mark": {
                        "type": "string",
                        "enum": [
                            "x", "tick", "filled", "scribbled", "other", "uncertain"
                        ],
                    },
                },
                "required": ["option_id", "state", "visual_mark"],
                "additionalProperties": False,
            },
        },
        "unmapped_marks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "row_label": {"type": "string"},
                    "option_label": {"type": "string"},
                    "state": {
                        "type": "string",
                        "enum": ["selected", "cancelled", "ambiguous"],
                    },
                    "visual_mark": {
                        "type": "string",
                        "enum": [
                            "x", "tick", "filled", "scribbled", "other", "uncertain"
                        ],
                    },
                },
                "required": ["row_label", "option_label", "state", "visual_mark"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["visible_row_ids", "marks", "unmapped_marks"],
    "additionalProperties": False,
}


UNIVERSAL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        # Every array here is bounded. Constrained decoding never forces an
        # unbounded array to close, and the non-thinking Qwen3-VL build reliably
        # exploits that: after emitting the real answer it pads with empty
        # question objects (63 on a two-option "Geschlecht" question) or empty
        # rows (490 on a single-choice question) until num_predict truncates the
        # JSON mid-string and the whole section is lost. The caps are generous
        # relative to any real paper form, so they only ever bite on runaway
        # generation.
        "questions": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "question_text": {"type": "string"},
                    "response_type": {
                        "type": "string",
                        "enum": sorted(_VALID_RESPONSE_TYPES),
                    },
                    # This is a model observation only. TextLab derives the
                    # rule used for validation independently after parsing.
                    "selection_rule": {
                        "type": "string",
                        "enum": sorted(_VALID_SELECTION_RULES),
                    },
                    "parent_question_index": {"type": "integer", "minimum": 0},
                    "condition_text": {"type": "string"},
                    "choices": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 24,
                        "items": {
                            "type": "object",
                            "properties": {"choice_text": {"type": "string"}},
                            "required": ["choice_text"],
                            "additionalProperties": False,
                        },
                    },
                    "rows": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 24,
                        "items": {
                            "type": "object",
                            "properties": {
                                "row_text": {"type": "string"},
                                "extra_choices": {
                                    "type": "array",
                                    "maxItems": 8,
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "choice_text": {"type": "string"}
                                        },
                                        "required": ["choice_text"],
                                        "additionalProperties": False,
                                    },
                                },
                                "marked_answers": {
                                    "type": "array",
                                    "maxItems": 24,
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "choice_position": {
                                                "type": "integer",
                                                "minimum": 1,
                                            },
                                            # Echoing the position's text is a
                                            # cheap independent consistency
                                            # check for off-by-one failures.
                                            "choice_text": {"type": "string"},
                                            "state": {
                                                "type": "string",
                                                "enum": sorted(_VALID_STATES),
                                            },
                                            "visual_mark": {
                                                "type": "string",
                                                "enum": sorted(_VALID_VISUAL_MARKS),
                                            },
                                            "associated_text": {"type": "string"},
                                        },
                                        "required": [
                                            "choice_position",
                                            "choice_text",
                                            "state",
                                            "visual_mark",
                                            "associated_text",
                                        ],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": ["row_text", "extra_choices", "marked_answers"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": [
                    "question_text",
                    "response_type",
                    "selection_rule",
                    "parent_question_index",
                    "condition_text",
                    "choices",
                    "rows",
                ],
                "additionalProperties": False,
            },
        },
        "unmapped_marks": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "nearby_text": {"type": "string"},
                    "state": {"type": "string", "enum": sorted(_VALID_STATES)},
                    "visual_mark": {
                        "type": "string",
                        "enum": sorted(_VALID_VISUAL_MARKS),
                    },
                },
                "required": ["nearby_text", "state", "visual_mark"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["questions", "unmapped_marks"],
    "additionalProperties": False,
}


def _infer_question_contract(schema_hint: dict) -> Tuple[str, str]:
    rows = schema_hint.get("rows") or []
    option_counts = [len(row.get("options") or []) for row in rows]
    question = _plain(schema_hint.get("question_text") or "").casefold()
    multiple_hints = (
        "mehrere antwort" in question
        or "multiple" in question
        or "select all" in question
        or "mehrfach" in question
    )
    if len(rows) >= 2 and sum(count >= 2 for count in option_counts) >= 2:
        return "matrix", "one_per_row"
    if len(rows) == 1 and option_counts and option_counts[0] >= 2:
        return ("multiple", "zero_or_more") if multiple_hints else ("single", "zero_or_one")
    if rows:
        return "multiple", "zero_or_more"
    return "unknown", "zero_or_more"


def _normalize_mark_only_result(result: dict, schema_hint: dict) -> List[dict]:
    """Convert the minimal model response to the internal rich merge shape."""
    if not isinstance(result, dict):
        raise ValueError("mark-only response is not an object")
    visible = result.get("visible_row_ids")
    marks = result.get("marks")
    unmapped = result.get("unmapped_marks")
    if not isinstance(visible, list) or not isinstance(marks, list) or not isinstance(unmapped, list):
        raise ValueError("mark-only response is missing required arrays")

    hint_rows = {str(row.get("row_id")): row for row in schema_hint.get("rows") or []}
    option_map = {}
    for row_id, row in hint_rows.items():
        for option in row.get("options") or []:
            option_map[str(option.get("option_id"))] = (row_id, option)

    rows_out: Dict[str, dict] = {}
    for row_id in visible:
        row_id = str(row_id)
        hint = hint_rows.get(row_id)
        if hint is not None:
            rows_out[row_id] = {
                "row_id": row_id,
                "label": str(hint.get("label") or ""),
                "marks": [],
            }

    valid_states = {"selected", "cancelled", "ambiguous"}
    valid_visuals = {"x", "tick", "filled", "scribbled", "other", "uncertain"}
    for mark in marks:
        if not isinstance(mark, dict):
            continue
        option_id = str(mark.get("option_id") or "")
        mapped = option_map.get(option_id)
        state, visual = mark.get("state"), mark.get("visual_mark")
        if state not in valid_states or visual not in valid_visuals:
            continue
        if mapped is None:
            row_id = f"unknown_option_{_slug(option_id, 'id')}"
            rows_out[row_id] = {
                "row_id": row_id,
                "label": "",
                "marks": [
                    {
                        "option_id": option_id,
                        "label": option_id or "unknown option ID",
                        "state": state,
                        "visual_mark": visual,
                    }
                ],
            }
            continue
        row_id, option = mapped
        hint = hint_rows[row_id]
        row = rows_out.setdefault(
            row_id,
            {"row_id": row_id, "label": str(hint.get("label") or ""), "marks": []},
        )
        row["marks"].append(
            {
                "option_id": option_id,
                "label": str(option.get("label") or option_id),
                "state": state,
                "visual_mark": visual,
            }
        )

    for index, mark in enumerate(unmapped, start=1):
        if not isinstance(mark, dict):
            continue
        state, visual = mark.get("state"), mark.get("visual_mark")
        if state not in valid_states or visual not in valid_visuals:
            continue
        row_id = "unmapped_" + _slug(
            f"{mark.get('row_label', '')}_{mark.get('option_label', '')}",
            str(index),
        )
        rows_out[row_id] = {
            "row_id": row_id,
            "label": str(mark.get("row_label") or ""),
            "marks": [
                {
                    "option_id": "",
                    "label": str(mark.get("option_label") or "unmapped option"),
                    "state": state,
                    "visual_mark": visual,
                }
            ],
        }

    question_type, selection_rule = _infer_question_contract(schema_hint)
    return [
        {
            "question_id": schema_hint.get("question_id"),
            "question": schema_hint.get("question_text", ""),
            "question_type": question_type,
            "selection_rule": selection_rule,
            "rows": list(rows_out.values()),
        }
    ]


def _prompt(schema_hint: dict) -> str:
    output_instruction = (
        "Do not repeat or rewrite questions, rows, labels, or option text. "
        "Return only: (1) the supplied row IDs actually visible in this section, "
        "(2) visibly marked supplied option IDs with state/mark type, and "
        "(3) unmapped visible marks only when no supplied option ID matches. "
    )
    return (
        "Read the actual handwritten response marks in this high-resolution paper "
        "survey crop. Do not rely on OCR glyphs and never infer an answer from the "
        "wording. A normal handwritten X, tick, or fill over a printed circle/box "
        "is selected. A clean printed outline is unselected and must be omitted. "
        "A printed circle with a white/empty centre is NOT a response. Before "
        "returning any mark, verify that separate handwritten ink visibly crosses "
        "or fills that option's centre; never list all printed options merely "
        "because their outlines are visible. "
        "A dense scribble or multiple cancellation strokes may be cancelled; if "
        "intent is unclear use ambiguous. Return only visibly marked options. "
        + output_instruction
        + "The image is one complete Paddle-bounded question section, including "
        "its question, response labels, and marks. Judge each mark together with "
        "its visible option; handwriting in a free-text line is not a checkbox or "
        "radio response. "
        "Paddle schema hints follow:\n"
        + json.dumps(
            _compact_prompt_schema(schema_hint),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\nReturn only the requested JSON schema."
    )


def _universal_prompt(section_id: str) -> str:
    """Provider-neutral schema-free survey prompt (one complete section/call)."""
    return (
        f"Contract version: textlab-survey-{SCHEMA_FREE_CONTRACT}. "
        f"Audit section reference: {section_id}. "
        "Read this high-resolution scan of one complete paper-form question section. "
        "Ignore OCR transcriptions and use only the visible pixels. Reconstruct every "
        "logical question or conditional subquestion, every printed choice, and every "
        "visible row in visual reading order. Keep text concise: question_text is only "
        "the actual question, and choice_text is only a printed response label. Exclude "
        "examples, prize/legal prose, handwriting, and free-text-only prompts from those "
        "fields. Questions with different choice sets are separate objects. For single, "
        "multiple, or rating return exactly one row with empty row_text, even when choices "
        "are vertical. Only a matrix has multiple rows: use shared choices and one row per "
        "statement; never duplicate choices as rows. Put a row-only N/A choice in "
        "extra_choices. For a conditional subquestion set its one-based "
        "parent_question_index and condition_text; otherwise use 0 and an empty string. "
        "Do not create a question for a free-text-only follow-up. If its handwriting is "
        "conditioned on a marked choice, put it in that mark's associated_text. "
        "choice_position is one-based over shared choices then extra_choices. Return every "
        "printed choice and matrix row, including unselected choices and unanswered rows. "
        "In marked_answers return every choice with respondent-added ink, including "
        "selected, cancelled, and ambiguous marks. A normal "
        "handwritten X, tick, or fill over a printed circle or box is selected. A clean "
        "printed outline is unselected and must be omitted. A printed circle with a "
        "white/empty centre is not a response. Verify separate ink at each reported "
        "control; never list printed outlines. A dense overwrite may be cancelled; if "
        "intent is unclear use state ambiguous with visual_mark uncertain. Do not pair "
        "visual_mark uncertain with state selected. Never infer an answer from wording. "
        "For every marked answer, copy the exact visible printed choice into choice_text "
        "as a positional echo. associated_text is only respondent text visibly linked "
        "to that marked choice; otherwise it is empty. selection_rule is only your "
        "structural observation; TextLab validates it independently. unmapped_marks is "
        "only stray response ink outside every logical question. "
        "Ranking boxes containing written numbers and continuous mark-anywhere lines are "
        "out of scope: use response_type unknown, retain any visible labelled targets as "
        "choices, and do not interpret written ranks or line positions as choice marks. "
        "Do not create identifiers. Return only JSON matching the supplied schema."
    )


def _derived_selection_rule(
    printed_validation_text: str,
    question_text: str,
    condition_text: str,
    rows: List[dict],
    model_type: str = "unknown",
) -> str:
    """Derive review constraints from question-local, corroborated print cues."""
    paddle_text = _plain(printed_validation_text).casefold()
    model_scope = _plain(f"{question_text} {condition_text}").casefold()
    multiple_cues = (
        "mehrere antwort",
        "mehrfach",
        "multiple",
        "select all",
        "all that apply",
        "plusieurs réponses",
        "plusieurs reponses",
        "sélectionnez toutes",
        "seleziona tutte",
        "varias respuestas",
    )
    # Paddle provides independent evidence that the instruction is printed,
    # while the reconstructed question scopes it to the current subquestion.
    # A later "multiple answers" follow-up must not weaken an earlier yes/no.
    if any(
        cue in paddle_text and (cue in model_scope or model_type == "multiple")
        for cue in multiple_cues
    ):
        return "zero_or_more"
    if len(rows) > 1:
        return "one_per_row"
    # Conservative default: multiple marks in a one-row question should be
    # reviewed unless the printed wording explicitly permits them.
    return "zero_or_one"


def _derived_question_type(model_type: str, rows: List[dict], rule: str) -> str:
    if len(rows) > 1:
        return "matrix"
    if rule == "zero_or_more":
        return "multiple"
    return model_type if model_type in _VALID_RESPONSE_TYPES else "unknown"


def _region_overlaps_bbox(region: "doc_ir.Region", bbox: List[int]) -> bool:
    if len(region.bbox) < 4 or len(bbox) < 4:
        return False
    x1, y1, x2, y2 = bbox[:4]
    rx1, ry1, rx2, ry2 = region.bbox[:4]
    return min(x2, rx2) > max(x1, rx1) and min(y2, ry2) > max(y1, ry1)


def _paddle_mark_presence(page: "doc_ir.Page", bbox: List[int]) -> bool:
    """Weak section-level signal used only to strengthen under-selection review."""
    for region in page.regions:
        if not _region_overlaps_bbox(region, bbox):
            continue
        markup = region.markup or {}
        if markup.get("state") in {"checked", "uncertain"}:
            return True
        for item in markup.get("items") or []:
            if item.get("state") in {"checked", "uncertain"} or item.get("needs_review"):
                return True
        source_text = (
            region.content.get("html")
            if region.type == doc_ir.TABLE
            else (region.content.get("text") or region.content.get("markdown") or "")
        ) or ""
        if any(item["state"] == "checked" for item in markup_detect.extract_mark_glyphs(source_text)):
            return True
    return False


def _choice_texts(items: Any, *, field_name: str) -> List[str]:
    if not isinstance(items, list):
        raise ValueError(f"schema-free response {field_name} is not an array")
    output = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("choice_text"), str):
            raise ValueError(f"schema-free response has an invalid {field_name} item")
        value = _SPACE.sub(" ", item["choice_text"]).strip()
        if len(value) > 240:
            raise ValueError(f"schema-free response {field_name} item is too long")
        output.append(value)
    return output


def _suspicious_choice_label(value: str) -> bool:
    """Detect model-format leakage that can still satisfy the JSON grammar."""
    text = str(value or "")
    if markup_detect.extract_mark_glyphs(text):
        return True
    lowered = text.casefold()
    if any(
        token in lowered
        for token in ("associated_text", "choice_text", "marked_answers")
    ):
        return True
    return bool(re.search(r"(?:\{\s*['\"]|['\"],\s*$|\}\s*,?)", text))


def _prune_section_groups(
    groups: List["doc_ir.FormGroup"],
) -> List["doc_ir.FormGroup"]:
    """Drop stray empty-question groups and exact duplicates within a section.

    qwen-class reasoning models occasionally emit the same conditional twice,
    duplicate an entire matrix, or turn a stray glyph / page marker into an
    extra question whose text never resolves. None of these are valid review
    items, so remove them here. Safety rules: never drop a group another group
    depends on as its conditional parent, and never prune a section to nothing.
    """
    if len(groups) <= 1:
        return groups
    referenced = {g.parent_question_id for g in groups if g.parent_question_id}

    def signature(group: "doc_ir.FormGroup"):
        return (
            _normalized_label(group.question_text),
            tuple(
                (row.label, tuple((opt.label, opt.state) for opt in row.options))
                for row in group.rows
            ),
        )

    seen: set = set()
    kept: List["doc_ir.FormGroup"] = []
    for group in groups:
        is_parent = group.id in referenced
        sig = signature(group)
        if not is_parent and (sig in seen or not group.question_text.strip()):
            continue
        seen.add(sig)
        kept.append(group)
    return kept or groups


def _universal_to_form_groups(
    result: dict,
    *,
    section_id: str,
    bbox: List[int],
    crop_b64: str,
    client,
    template_reused: bool,
    candidate_evidence: Optional[List[str]],
    paddle_mark_present: bool,
    printed_validation_text: str,
    expected_question_number: Optional[str] = None,
    geometric_control_count: int = 0,
) -> List["doc_ir.FormGroup"]:
    """Validate the universal wire format and materialize the existing row IR."""
    if not isinstance(result, dict):
        raise ValueError("schema-free response is not an object")
    raw_questions = result.get("questions")
    unmapped = result.get("unmapped_marks")
    if not isinstance(raw_questions, list) or not isinstance(unmapped, list):
        raise ValueError("schema-free response is missing required arrays")
    if not raw_questions:
        raise ValueError("schema-free response returned no logical questions")
    for raw_unmapped in unmapped:
        if (
            not isinstance(raw_unmapped, dict)
            or not isinstance(raw_unmapped.get("nearby_text"), str)
            or raw_unmapped.get("state") not in _VALID_STATES
            or raw_unmapped.get("visual_mark") not in _VALID_VISUAL_MARKS
        ):
            raise ValueError("schema-free response contains an invalid unmapped mark")
        if len(raw_unmapped["nearby_text"]) > 240:
            raise ValueError("schema-free response unmapped nearby text is too long")

    # Padding and imperfect layout boxes can expose a sliver of the neighboring
    # question. The crop anchor is independent Paddle evidence, so explicitly
    # numbered model questions that disagree with it are boundary spill, not a
    # second response group. Unnumbered conditional subquestions remain valid.
    kept_questions: List[Tuple[int, dict]] = []
    excluded_numbers: List[str] = []
    expected_number = str(expected_question_number or "").strip()
    for original_index, raw_question in enumerate(raw_questions, start=1):
        if not isinstance(raw_question, dict):
            raise ValueError("schema-free response contains a non-object question")
        visible_number = _question_number(str(raw_question.get("question_text") or ""))
        if expected_number and visible_number and visible_number != expected_number:
            excluded_numbers.append(visible_number)
            continue
        kept_questions.append((original_index, raw_question))

    question_ids = [
        f"{section_id}_q{index}" for index in range(1, len(kept_questions) + 1)
    ]
    id_by_original_index = {
        original_index: question_ids[new_index - 1]
        for new_index, (original_index, _) in enumerate(kept_questions, start=1)
    }
    groups: List[doc_ir.FormGroup] = []
    total_marked = 0

    for question_index, (original_index, raw_question) in enumerate(
        kept_questions, start=1
    ):
        question_text = _SPACE.sub(" ", str(raw_question.get("question_text") or "")).strip()
        condition_text = _SPACE.sub(" ", str(raw_question.get("condition_text") or "")).strip()
        if len(question_text) > 800 or len(condition_text) > 240:
            raise ValueError("schema-free response contains an overlong question field")
        model_type = str(raw_question.get("response_type") or "unknown")
        model_rule = str(raw_question.get("selection_rule") or "unknown")
        if model_type not in _VALID_RESPONSE_TYPES or model_rule not in _VALID_SELECTION_RULES:
            raise ValueError("schema-free response contains an invalid question enum")
        choices = _choice_texts(raw_question.get("choices"), field_name="choices")
        raw_rows = raw_question.get("rows")
        if not choices or not isinstance(raw_rows, list) or not raw_rows:
            raise ValueError("schema-free question must contain choices and rows")

        group_id = question_ids[question_index - 1]
        group_warnings: List[str] = []
        parent_id = ""
        parent_index = raw_question.get("parent_question_index", 0)
        if not isinstance(parent_index, int) or parent_index < 0:
            raise ValueError("schema-free response has an invalid parent_question_index")
        if parent_index:
            if parent_index >= original_index:
                group_warnings.append("conditional parent is not a preceding question")
            elif parent_index not in id_by_original_index:
                group_warnings.append("conditional parent was excluded as boundary spill")
            else:
                parent_id = id_by_original_index[parent_index]
        if condition_text and not parent_id:
            group_warnings.append("condition text has no valid parent question")
        if not question_text:
            group_warnings.append("visible question text was not resolved")
        if any(not choice for choice in choices):
            group_warnings.append("one or more printed choice labels were not resolved")
        suspicious_choices = [
            index
            for index, choice in enumerate(choices, start=1)
            if _suspicious_choice_label(choice)
        ]
        normalized_choices = [_normalized_label(choice) for choice in choices]
        duplicate_choices = bool(
            normalized_choices
            and len(set(normalized_choices)) < len(normalized_choices)
        )
        if suspicious_choices:
            group_warnings.append(
                "one or more choice labels contain response glyphs or format leakage"
            )
        if duplicate_choices:
            group_warnings.append("model returned duplicate printed choice labels")
        if (
            len(kept_questions) == 1
            and model_type in {"single", "multiple", "rating"}
            and geometric_control_count > len(choices)
        ):
            group_warnings.append(
                "model returned fewer choices than independently detected controls"
            )

        structural_rows_invalid = False
        if model_type in {"single", "multiple", "rating"} and len(raw_rows) != 1:
            structural_rows_invalid = True
            group_warnings.append(
                f"non-matrix response type {model_type!r} must contain exactly one row"
            )
        elif model_type == "matrix" and len(raw_rows) < 2:
            structural_rows_invalid = True
            group_warnings.append("matrix response contains fewer than two visible rows")
        elif model_type == "unknown":
            group_warnings.append("response type could not be resolved")

        final_rows: List[doc_ir.FormRow] = []
        row_contracts: List[dict] = []
        for row_index, raw_row in enumerate(raw_rows, start=1):
            if not isinstance(raw_row, dict):
                raise ValueError("schema-free response contains a non-object row")
            row_text = _SPACE.sub(" ", str(raw_row.get("row_text") or "")).strip()
            if len(row_text) > 400:
                raise ValueError("schema-free response row text is too long")
            extra_choices = _choice_texts(
                raw_row.get("extra_choices"), field_name="extra_choices"
            )
            effective_choices = choices + extra_choices
            row_suspicious_choices = suspicious_choices + [
                len(choices) + index
                for index, choice in enumerate(extra_choices, start=1)
                if _suspicious_choice_label(choice)
            ]
            raw_marks = raw_row.get("marked_answers")
            if not isinstance(raw_marks, list):
                raise ValueError("schema-free row marked_answers is not an array")
            row_contracts.append({"choice_count": len(effective_choices)})
            row_id = f"{group_id}_r{row_index}"
            options = [
                doc_ir.FormOption(
                    id=f"{row_id}_c{choice_index}",
                    label=choice_text,
                )
                for choice_index, choice_text in enumerate(effective_choices, start=1)
            ]
            row_warnings: List[str] = []
            seen_positions: Dict[int, Tuple[str, str]] = {}
            row_needs_review = (
                structural_rows_invalid
                or bool(row_suspicious_choices)
                or duplicate_choices
                or any(not choice for choice in effective_choices)
            )
            if model_type in {"single", "multiple", "rating"} and row_text:
                row_needs_review = True
                row_warnings.append(
                    "non-matrix response must use one row with an empty row label"
                )
            if row_needs_review:
                row_warnings.append("one or more row choice labels were not resolved")
                for option in options:
                    if not option.label:
                        option.warnings.append("printed choice label was not resolved")
            for option_index in row_suspicious_choices:
                if option_index <= len(options):
                    options[option_index - 1].warnings.append(
                        "choice label contains response glyphs or model-format leakage"
                    )
            if duplicate_choices:
                for option in options[: len(choices)]:
                    option.warnings.append("choice label is duplicated in this question")

            for raw_mark in raw_marks:
                if not isinstance(raw_mark, dict):
                    raise ValueError("schema-free response contains a non-object mark")
                position = raw_mark.get("choice_position")
                state = raw_mark.get("state")
                visual = raw_mark.get("visual_mark")
                echo = str(raw_mark.get("choice_text") or "")
                associated_text = _SPACE.sub(
                    " ", str(raw_mark.get("associated_text") or "")
                ).strip()
                if len(echo) > 240 or len(associated_text) > 800:
                    raise ValueError("schema-free response contains an overlong mark field")
                if (
                    not isinstance(position, int)
                    or state not in _VALID_STATES
                    or visual not in _VALID_VISUAL_MARKS
                ):
                    raise ValueError("schema-free response contains an invalid marked answer")
                if position < 1 or position > len(options):
                    row_warnings.append(
                        f"marked choice position {position} is outside the visible choice list"
                    )
                    row_needs_review = True
                    continue
                option = options[position - 1]
                if _normalized_label(echo) != _normalized_label(option.label):
                    option.warnings.append(
                        "choice-text echo does not match the reported choice position"
                    )
                    row_needs_review = True
                previous = seen_positions.get(position)
                if previous is not None:
                    option.state = "ambiguous"
                    option.visual_mark = "uncertain"
                    option.warnings.append("model returned this choice position more than once")
                    row_needs_review = True
                else:
                    option.state = state
                    option.visual_mark = visual
                    seen_positions[position] = (state, visual)
                option.associated_text = associated_text
                option.observations.append(
                    doc_ir.Observation(
                        source=client.provider,
                        value=state,
                        method="schema-free-complete-question-section",
                        raw={
                            "model": client.model,
                            "choice_position": position,
                            "choice_text_echo": echo,
                        },
                    )
                )
                total_marked += 1
                if state in {"cancelled", "ambiguous"}:
                    option.warnings.append(
                        "changed or ambiguous response requires human review"
                    )
                    row_needs_review = True
                if visual == "uncertain":
                    option.warnings.append("visual mark type is uncertain")
                    row_needs_review = True

            final_rows.append(
                doc_ir.FormRow(
                    id=row_id,
                    label=row_text,
                    options=options,
                    status="needs_review" if row_needs_review else "accepted",
                    warnings=row_warnings,
                )
            )

        derived_rule = _derived_selection_rule(
            printed_validation_text,
            question_text,
            condition_text,
            row_contracts,
            model_type,
        )
        rule_disagreement = model_rule != derived_rule
        # A permissive model-only rule could hide an over-selection and remains
        # a review warning. Other rule wording disagreements are diagnostic;
        # the independently derived rule is authoritative and avoids flooding
        # correct one-row answers with Gemma's common `one_per_row` mistake.
        if model_rule == "zero_or_more" and derived_rule == "zero_or_one":
            group_warnings.append(
                f"model selection rule {model_rule!r} disagrees with TextLab rule {derived_rule!r}"
            )
        for row in final_rows:
            selected_count = sum(option.state == "selected" for option in row.options)
            marked_count = sum(option.state != "unselected" for option in row.options)
            if derived_rule in {"zero_or_one", "one_per_row"} and selected_count > 1:
                row.status = "needs_review"
                row.warnings.append("multiple visible marks violate the derived selection rule")
                for option in row.options:
                    if option.state == "selected":
                        option.warnings.append("multiple marks in a single-choice row")
            if derived_rule == "one_per_row" and marked_count == 0:
                row.status = "needs_review"
                row.warnings.append("matrix row has no visible response mark")

        group_needs_review = bool(group_warnings) or any(
            row.status == "needs_review" for row in final_rows
        )
        groups.append(
            doc_ir.FormGroup(
                id=group_id,
                bbox=list(bbox),
                question_text=question_text,
                question_type=_derived_question_type(model_type, row_contracts, derived_rule),
                selection_rule=derived_rule,
                rows=final_rows,
                status="needs_review" if group_needs_review else "accepted",
                warnings=group_warnings,
                parent_question_id=parent_id,
                condition_text=condition_text,
                provenance={
                    "provider": client.provider,
                    "model": client.model,
                    "method": "schema-free-complete-question-section",
                    "contract_version": SCHEMA_FREE_CONTRACT,
                    "model_response_type": model_type,
                    "model_selection_rule": model_rule,
                    "selection_rule_disagreement": rule_disagreement,
                    "selection_rule_source": (
                        "textlab-structure-and-question-local-paddle-printed-cues"
                    ),
                    "candidate_evidence": list(candidate_evidence or []),
                    "same_layout_template": template_reused,
                    "unmapped_marks": copy.deepcopy(unmapped),
                    "excluded_boundary_question_numbers": list(excluded_numbers),
                    "geometric_control_count": geometric_control_count,
                },
                source_crop_b64=crop_b64,
            )
        )

    if not groups:
        groups.append(
            doc_ir.FormGroup(
                id=f"{section_id}_q1",
                bbox=list(bbox),
                status="needs_review",
                warnings=["model returned no logical questions for this candidate section"],
                provenance={
                    "provider": client.provider,
                    "model": client.model,
                    "method": "schema-free-complete-question-section",
                    "contract_version": SCHEMA_FREE_CONTRACT,
                    "candidate_evidence": list(candidate_evidence or []),
                    "same_layout_template": template_reused,
                    "unmapped_marks": copy.deepcopy(unmapped),
                    "excluded_boundary_question_numbers": list(excluded_numbers),
                    "geometric_control_count": geometric_control_count,
                },
                source_crop_b64=crop_b64,
            )
        )

    groups = _prune_section_groups(groups)

    if excluded_numbers:
        excluded = ", ".join(dict.fromkeys(excluded_numbers))
        for group in groups:
            group.status = "needs_review"
            group.warnings.append(
                f"numbered question(s) {excluded} were excluded as crop-boundary spill"
            )

    if unmapped:
        for group in groups:
            group.status = "needs_review"
            group.warnings.append("visible response ink was not mapped to a logical choice")
    if total_marked == 0:
        warning = "section returned zero marked answers and requires under-selection review"
        if paddle_mark_present:
            warning += "; Paddle has weak evidence that response ink is present"
        for group in groups:
            group.status = "needs_review"
            group.warnings.append(warning)
    return groups


def _hint_rows(schema_hint: dict) -> Dict[str, dict]:
    return {str(row["row_id"]): row for row in schema_hint.get("rows") or []}


def _merge_results(
    results: List[List[dict]],
    schema_hint: dict,
    bbox: List[int],
    crop_b64: str,
    page_number: int,
    section_index: int,
    client,
    *,
    template_reused: bool = False,
    candidate_evidence: Optional[List[str]] = None,
) -> List["doc_ir.FormGroup"]:
    """Reconcile a complete-section response with Paddle's schema hints."""
    merged: Dict[str, dict] = {}
    for response_groups in results:
        for raw_group in response_groups:
            # One crop represents one proposed question. Keep the stable local
            # ID even when the model echoes/hallucinates a different ID.
            group_key = schema_hint["question_id"]
            group = merged.setdefault(
                group_key,
                {
                    "raw": raw_group,
                    "rows": {},
                    "warnings": [],
                },
            )
            returned_group_id = str(raw_group.get("question_id") or "").strip()
            if returned_group_id and returned_group_id != schema_hint["question_id"]:
                warning = (
                    f"targeted VLM returned unknown question ID {returned_group_id!r}"
                )
                if warning not in group["warnings"]:
                    group["warnings"].append(warning)
            for raw_row in raw_group.get("rows") or []:
                row_key = _key(raw_row.get("row_id") or raw_row.get("label")) or "row"
                row = group["rows"].setdefault(
                    row_key,
                    {"raw": raw_row, "marks": {}},
                )
                for mark in raw_row.get("marks") or []:
                    mark_key = _key(mark.get("option_id") or mark.get("label")) or "option"
                    entry = row["marks"].setdefault(mark_key, {"values": [], "raw": mark})
                    entry["values"].append((mark.get("state"), mark.get("visual_mark")))

    output = []
    hinted_rows = _hint_rows(schema_hint)
    if not merged:
        merged[schema_hint["question_id"]] = {
            "raw": {
                "question_id": schema_hint["question_id"],
                "question": schema_hint.get("question_text", ""),
                "question_type": "unknown",
                "selection_rule": "zero_or_more",
            },
            "rows": {},
            "warnings": [],
        }

    for group_pos, group_data in enumerate(merged.values(), start=1):
        raw_group = group_data["raw"]
        final_rows: List[doc_ir.FormRow] = []
        used_hint_ids = set()
        group_needs_review = bool(group_data.get("warnings"))

        for row_pos, row_data in enumerate(group_data["rows"].values(), start=1):
            raw_row = row_data["raw"]
            row_id = str(raw_row.get("row_id") or f"row_{row_pos}")
            hint = hinted_rows.get(row_id)
            if hint is None and hinted_rows:
                label_key = _key(raw_row.get("label"))
                matched_hint = next(
                    (
                        (candidate_id, candidate)
                        for candidate_id, candidate in hinted_rows.items()
                        if label_key and _key(candidate.get("label")) == label_key
                    ),
                    None,
                )
                if matched_hint:
                    row_id, hint = matched_hint
            if hint:
                used_hint_ids.add(row_id)
            option_hints = {
                str(o["option_id"]): o for o in (hint or {}).get("options") or []
            }
            options: Dict[str, doc_ir.FormOption] = {}
            for option_id, option_hint in option_hints.items():
                paddle_state = option_hint.get("paddle_state", "unknown")
                option = doc_ir.FormOption(
                    id=option_id,
                    label=str(option_hint.get("label") or option_id),
                    observations=[
                        doc_ir.Observation(
                            source="paddleocr-vl",
                            value=paddle_state,
                            method="glyph-transcription",
                        )
                    ],
                )
                if not _label_is_resolved(option.label) or option.label.startswith("option "):
                    option.warnings.append("printed option label was not resolved")
                options[option_id] = option

            row_needs_review = bool(hinted_rows and hint is None)
            row_warnings = []
            if row_needs_review:
                row_warnings.append("targeted VLM returned an unknown row ID")
            for mark_data in row_data["marks"].values():
                raw_mark = mark_data["raw"]
                option_id = str(raw_mark.get("option_id") or "").strip()
                label = str(raw_mark.get("label") or "").strip()
                if option_id not in options:
                    matched = next(
                        (oid for oid, opt in options.items() if _key(opt.label) == _key(label)),
                        None,
                    )
                    option_id = matched or option_id or f"discovered_{len(options) + 1}"
                option = options.get(option_id)
                if option is None:
                    option = doc_ir.FormOption(id=option_id, label=label or option_id)
                    option.warnings.append("option was not present in Paddle schema")
                    options[option_id] = option
                    row_needs_review = True

                states = {value[0] for value in mark_data["values"]}
                visuals = {value[1] for value in mark_data["values"]}
                if len(states) > 1:
                    option.state = "ambiguous"
                    option.visual_mark = "uncertain"
                    option.warnings.append("overlapping VLM crops disagree")
                    row_needs_review = True
                else:
                    option.state = next(iter(states))
                    option.visual_mark = next(iter(visuals)) if len(visuals) == 1 else "uncertain"
                if option.state in ("cancelled", "ambiguous"):
                    option.warnings.append(
                        "changed or ambiguous response requires human review"
                    )
                    row_needs_review = True
                if option.state == "selected" and any(
                    "label was not resolved" in warning for warning in option.warnings
                ):
                    row_needs_review = True
                option.observations.append(
                    doc_ir.Observation(
                        source=client.provider,
                        value=option.state,
                        method="complete-question-section",
                        raw={"model": client.model, "votes": mark_data["values"]},
                    )
                )
                paddle_values = {
                    o.value for o in option.observations if o.source == "paddleocr-vl"
                }
                if option.state == "selected" and "unchecked" in paddle_values:
                    option.warnings.append("VLM selected mark disagrees with Paddle glyph")
                    row_needs_review = True

            selected_count = sum(o.state == "selected" for o in options.values())
            selection_rule = str(raw_group.get("selection_rule") or "zero_or_more")
            if selection_rule in ("zero_or_one", "exactly_one", "one_per_row") and selected_count > 1:
                row_needs_review = True
                for option in options.values():
                    if option.state == "selected":
                        option.warnings.append("multiple visible marks in a single-choice row")

            # A checked Paddle glyph omitted by a VLM response is a disagreement,
            # but only when this row was returned as visible in the section.
            for option in options.values():
                paddle_checked = any(
                    o.source == "paddleocr-vl" and o.value == "checked"
                    for o in option.observations
                )
                has_vlm = any(o.source == client.provider for o in option.observations)
                if paddle_checked and not has_vlm:
                    option.warnings.append("Paddle reports selected but targeted VLM omitted it")
                    row_needs_review = True

            row_status = "needs_review" if row_needs_review else "accepted"
            group_needs_review = group_needs_review or row_needs_review
            final_rows.append(
                doc_ir.FormRow(
                    id=row_id,
                    label=str((hint or {}).get("label") or raw_row.get("label") or ""),
                    options=list(options.values()),
                    status=row_status,
                    warnings=row_warnings,
                )
            )

        # The prompt requires every visible row, including unanswered rows. A
        # hinted row omitted from the complete-section response is therefore
        # not safe to silently call blank: retain its Paddle schema, but flag it
        # for review instead of inventing a VLM observation.
        for row_id, hint in hinted_rows.items():
            if row_id in used_hint_ids:
                continue
            options = [
                doc_ir.FormOption(
                    id=str(option["option_id"]),
                    label=str(option.get("label") or option["option_id"]),
                    observations=[
                        doc_ir.Observation(
                            source="paddleocr-vl",
                            value=str(option.get("paddle_state") or "unknown"),
                            method="glyph-transcription",
                        )
                    ],
                )
                for option in hint.get("options") or []
            ]
            final_rows.append(
                doc_ir.FormRow(
                    id=row_id,
                    label=str(hint.get("label") or ""),
                    options=options,
                    status="needs_review",
                    warnings=["targeted VLM did not return this visible schema row"],
                )
            )
            group_needs_review = True

        if not final_rows:
            group_needs_review = True

        group_id = schema_hint["question_id"]
        group_warnings = []
        if not final_rows:
            group_warnings.append("no response rows could be resolved")
        group_warnings.extend(group_data.get("warnings") or [])
        output.append(
            doc_ir.FormGroup(
                id=group_id,
                bbox=list(bbox),
                question_text=str(
                    raw_group.get("question") or schema_hint.get("question_text") or ""
                ),
                question_type=str(raw_group.get("question_type") or "unknown"),
                selection_rule=str(raw_group.get("selection_rule") or "zero_or_more"),
                rows=final_rows,
                status="needs_review" if group_needs_review else "accepted",
                warnings=group_warnings,
                provenance={
                    "provider": client.provider,
                    "model": client.model,
                    "method": "complete-question-section",
                    "candidate_evidence": list(candidate_evidence or []),
                    "same_layout_template": template_reused,
                },
                source_crop_b64=crop_b64,
            )
        )
    return output


def extract_page_forms(
    page: "doc_ir.Page",
    page_bgr,
    client,
    *,
    same_layout_template: Optional[SameLayoutTemplate] = None,
    contract: Optional[str] = None,
) -> List["doc_ir.FormGroup"]:
    """Extract question-level responses from one full-resolution page raster."""
    if page_bgr is None:
        return []
    contract_name = _survey_contract(contract)
    height, width = page_bgr.shape[:2]
    groups: List[doc_ir.FormGroup] = []
    jobs = (
        same_layout_template.get(page.page_number, width, height)
        if same_layout_template is not None
        else []
    )

    if not jobs:
        used_group_ids = set()
        for section_index, section in enumerate(_question_sections(page), start=1):
            bbox = _section_bbox(
                section["regions"],
                width,
                height,
                crop_limits=section.get("crop_limits"),
            )
            crop = _crop(page_bgr, bbox)
            if crop is None:
                continue
            candidate_evidence = _form_candidate_evidence(section, crop)
            if not candidate_evidence:
                continue
            number = section.get("number") or str(section_index)
            group_id = f"p{page.page_number}_q{number}"
            if group_id in used_group_ids:
                suffix = 2
                while f"{group_id}_{suffix}" in used_group_ids:
                    suffix += 1
                group_id = f"{group_id}_{suffix}"
            used_group_ids.add(group_id)
            jobs.append(
                {
                    "bbox": bbox,
                    "group_id": group_id,
                    "section_id": f"p{page.page_number}_s{section_index}",
                    "schema_hint": _schema_hint(section, group_id),
                    "candidate_evidence": candidate_evidence,
                    "validation_text": _section_validation_text(section),
                    "question_number": section.get("number"),
                    "template_reused": False,
                }
            )
    for section_index, job in enumerate(jobs, start=1):
        bbox = job["bbox"]
        crop = _crop(page_bgr, bbox)
        if crop is None:
            continue
        schema_hint = job["schema_hint"]
        section_id = job.get("section_id") or f"p{page.page_number}_s{section_index}"
        crop_bytes = _png_bytes(crop)
        crop_b64 = base64.b64encode(crop_bytes).decode("ascii")
        errors = []
        _, _, _, _, complete_section = _complete_section_view(crop)
        if contract_name == PADDLE_ID_CONTRACT:
            section_results: List[List[dict]] = []
            try:
                result = client.analyze(
                    _png_bytes(complete_section),
                    _prompt(schema_hint),
                    MARK_ONLY_RESPONSE_SCHEMA,
                )
                section_results.append(_normalize_mark_only_result(result, schema_hint))
                job["structure_valid"] = True
            except Exception as exc:
                job["structure_valid"] = False
                errors.append(f"complete question section: {exc}")
            extracted = _merge_results(
                section_results,
                schema_hint,
                bbox,
                crop_b64,
                page.page_number,
                section_index,
                client,
                template_reused=bool(job.get("template_reused")),
                candidate_evidence=job.get("candidate_evidence"),
            )
            for group in extracted:
                group.provenance["contract_version"] = PADDLE_ID_CONTRACT
        else:
            try:
                try:
                    geometric_control_count = len(
                        markup_detect.find_marks(complete_section)
                    )
                except Exception:
                    geometric_control_count = 0
                result = client.analyze(
                    _png_bytes(complete_section),
                    _universal_prompt(section_id),
                    UNIVERSAL_RESPONSE_SCHEMA,
                )
                extracted = _universal_to_form_groups(
                    result,
                    section_id=section_id,
                    bbox=bbox,
                    crop_b64=crop_b64,
                    client=client,
                    template_reused=bool(job.get("template_reused")),
                    candidate_evidence=job.get("candidate_evidence"),
                    paddle_mark_present=_paddle_mark_presence(page, bbox),
                    printed_validation_text=str(job.get("validation_text") or ""),
                    expected_question_number=job.get("question_number"),
                    geometric_control_count=geometric_control_count,
                )
                job["structure_valid"] = True
            except Exception as exc:
                job["structure_valid"] = False
                errors.append(f"complete question section: {exc}")
                extracted = [
                    doc_ir.FormGroup(
                        id=f"{section_id}_q1",
                        bbox=list(bbox),
                        question_text=str(schema_hint.get("question_text") or ""),
                        status="needs_review",
                        warnings=["schema-free response could not be validated"],
                        provenance={
                            "provider": client.provider,
                            "model": client.model,
                            "method": "schema-free-complete-question-section",
                            "contract_version": SCHEMA_FREE_CONTRACT,
                            "candidate_evidence": list(job.get("candidate_evidence") or []),
                            "same_layout_template": bool(job.get("template_reused")),
                        },
                        source_crop_b64=crop_b64,
                    )
                ]
        if not _model_has_release_approval(client):
            for group in extracted:
                group.status = "needs_review"
                group.warnings.append(
                    "survey model has not yet passed TextLab's release benchmark"
                )
        if _has_geometry_disagreement(page, bbox):
            for group in extracted:
                group.status = "needs_review"
                group.warnings.append(
                    "geometric ink evidence disagrees with the OCR transcription"
                )
        if errors:
            for group in extracted:
                group.status = "needs_review"
                group.warnings.extend(errors)
        groups.extend(extracted)
    if (
        same_layout_template is not None
        and jobs
        and not any(job.get("template_reused") for job in jobs)
        and all(job.get("structure_valid") for job in jobs)
    ):
        same_layout_template.learn(page.page_number, width, height, jobs)
    page.form_groups.extend(groups)
    return groups
