"""Name the controls on a synthesized blank, using Paddle's layout of it.

Running layout+OCR on the blank rather than on a filled copy is the whole
advantage here: the printed text has nothing written over it, so this is the
easiest page Paddle will ever be given, and the labels it produces are reused
for every respondent in the batch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import doc_ir, form_extract, survey_template

MAX_LABEL_GAP = 0.16   # fraction of page width a label may sit from its control
_BAND_TOLERANCE = 0.6  # share of control height a label must vertically overlap
MAX_LABEL_CHARS = 70   # labels become spreadsheet headers, so keep them short


def sanitize(text: str) -> str:
    """One tidy line: no glyphs, no newlines, short enough for a CSV header."""
    import re

    text = re.sub(r"\s+", " ", str(text or "")).strip()
    text = text.lstrip("○☐□☑☒●◯O0 \t.-–—|")
    text = text.strip(" \t|,.;:-–—")
    if len(text) > MAX_LABEL_CHARS:
        text = text[:MAX_LABEL_CHARS].rsplit(" ", 1)[0] + "…"
    return text


def _blocks(page: "doc_ir.Page") -> List[Tuple[List[float], str]]:
    out = []
    for region in page.ordered_regions():
        text = form_extract._plain(region.text)
        if text and len(region.bbox) >= 4:
            out.append(([float(v) for v in region.bbox[:4]], text))
    return out


def _label_for(control_bbox: Sequence[float], blocks, max_gap: float) -> str:
    """Nearest text starting to the right of the control, on its line."""
    cx1, cy1, cx2, cy2 = control_bbox
    centre_y = (cy1 + cy2) / 2.0
    height = max(1.0, cy2 - cy1)

    best, best_gap, best_box = "", None, None
    for box, text in blocks:
        bx1, by1, bx2, by2 = box
        if by2 < centre_y - height * _BAND_TOLERANCE:
            continue
        if by1 > centre_y + height * _BAND_TOLERANCE:
            continue
        # Paddle usually swallows the control into the block that holds its
        # option text, so a block overlapping the control counts as gap zero
        # rather than being rejected for starting to its left.
        if bx2 <= cx1 or bx1 > cx2 + max_gap:
            continue
        gap = max(0.0, bx1 - cx2)
        if best_gap is None or gap < best_gap:
            best, best_gap, best_box = text, gap, box

    # Paddle often merges a whole option list into one block. Pick the line at
    # the control's own height within it.
    lines = [line for line in best.splitlines() if line.strip()]
    if len(lines) > 1 and best_box is not None:
        top, bottom = best_box[1], best_box[3]
        share = (centre_y - top) / max(1.0, bottom - top)
        best = lines[min(len(lines) - 1, max(0, int(share * len(lines))))]
    elif lines:
        best = lines[0]
    return sanitize(best)


def _sections(page: "doc_ir.Page", width: int, height: int) -> List[Dict[str, Any]]:
    sections = []
    for index, section in enumerate(form_extract._question_sections(page), start=1):
        bbox = form_extract._section_bbox(
            section["regions"], width, height, crop_limits=section.get("crop_limits")
        )
        if not bbox:
            continue
        text = form_extract._clean_question_text(
            form_extract._plain(section["regions"][0].text)
        ) if section.get("regions") else ""
        sections.append({
            "id": f"q{section.get('number') or index}",
            "bbox": bbox,
            "text": text,
        })
    return sections


def _section_for(control_bbox: Sequence[float], sections) -> Dict[str, Any]:
    cx = (control_bbox[0] + control_bbox[2]) / 2.0
    cy = (control_bbox[1] + control_bbox[3]) / 2.0
    for section in sections:
        x1, y1, x2, y2 = section["bbox"]
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return section
    return {}


_SCALE_TOKEN = None  # compiled lazily; a rating point like "3", "4,5" or "2.5"


def _horizontal_runs(controls, width: int, height: int) -> List[List[int]]:
    """Group control indices into left-to-right runs sharing a baseline."""
    rows: Dict[int, List[int]] = {}
    for index, control in enumerate(controls):
        x1, y1, x2, y2 = control.pixel_bbox(width, height)
        tolerance = max(6, (y2 - y1) // 2)
        key = next(
            (k for k in rows if abs(k - (y1 + y2) // 2) <= tolerance),
            (y1 + y2) // 2,
        )
        rows.setdefault(key, []).append(index)
    return [
        sorted(members, key=lambda i: controls[i].pixel_bbox(width, height)[0])
        for members in rows.values()
        if len(members) > 1
    ]


def _scale_tokens(text: str) -> List[str]:
    global _SCALE_TOKEN
    if _SCALE_TOKEN is None:
        import re

        _SCALE_TOKEN = re.compile(r"\d+(?:[.,]\d+)?")
    return _SCALE_TOKEN.findall(text or "")


def _label_horizontal_runs(controls, width: int, height: int) -> None:
    """Split a shared header across a row of controls.

    A rating scale or a matrix column band gives every control in the row the
    same merged Paddle block. When that block carries exactly one numeric
    point per control the scale can be recovered positionally; otherwise the
    control keeps an index, which the template review pass can correct.
    """
    for run in _horizontal_runs(controls, width, height):
        labels = [controls[i].label for i in run]
        if len(set(labels)) == len(run):
            continue  # each control already has its own text
        # Some controls in the run may have latched onto a neighbouring block;
        # the shared majority label is the one holding the scale.
        shared = max(set(labels), key=labels.count)
        tokens = _scale_tokens(shared)
        for position, index in enumerate(run):
            control = controls[index]
            control.label = (
                tokens[position] if len(tokens) == len(run) else f"option {position + 1}"
            )


def label_page(
    template_page: survey_template.TemplatePage,
    page_json: Dict[str, Any],
    *,
    max_gap: Optional[float] = None,
) -> int:
    """Attach option labels and question ids to one template page in place."""
    page = doc_ir.from_paddle_vl(page_json)
    page.page_number = template_page.page_index + 1
    width, height = template_page.width, template_page.height
    blocks = _blocks(page)
    sections = _sections(page, width, height)
    gap = (max_gap if max_gap is not None else MAX_LABEL_GAP) * width

    labelled = 0
    for control in template_page.controls:
        bbox = control.pixel_bbox(width, height)
        control.label = _label_for(bbox, blocks, gap)
        section = _section_for(bbox, sections)
        control.question_id = (
            f"p{template_page.page_index + 1}_{section['id']}" if section else ""
        )
        if control.label:
            labelled += 1

    # Per question, not per page: an A3 spread holds two columns of questions
    # whose rows share a baseline, and grouping across them merges unrelated
    # controls into one run.
    by_question: Dict[str, List[Any]] = {}
    for control in template_page.controls:
        by_question.setdefault(control.question_id, []).append(control)
    for group in by_question.values():
        _label_horizontal_runs(group, width, height)
    return labelled


def label_template(
    template: survey_template.SurveyTemplate,
    page_jsons: Sequence[Dict[str, Any]],
) -> int:
    """Label every page of *template* from the matching Paddle page result."""
    return sum(
        label_page(page, page_json)
        for page, page_json in zip(template.pages, page_jsons)
    )
