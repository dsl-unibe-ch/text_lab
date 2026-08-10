"""Name the controls on a synthesized blank, using Paddle's layout of it.

Running layout+OCR on the blank rather than on a filled copy is the whole
advantage here: the printed text has nothing written over it, so this is the
easiest page Paddle will ever be given, and the labels it produces are reused
for every respondent in the batch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import doc_ir, form_extract, markup_detect, survey_template

MAX_LABEL_GAP = 0.16   # fraction of page width a label may sit from its control
_BAND_TOLERANCE = 0.6  # share of control height a label must vertically overlap
MAX_LABEL_CHARS = 70   # labels become spreadsheet headers, so keep them short


def sanitize(text: str) -> str:
    """One tidy line: no glyphs, no newlines, short enough for a CSV header."""
    import re

    text = re.sub(r"\s+", " ", str(text or "")).strip()
    # Drop a leading control glyph. "O" and "0" are only glyphs when they stand
    # alone -- stripping them unconditionally turns "Oey" into "ey".
    text = re.sub(r"^(?:[○☐□☑☒●◯]\s*)+", "", text)
    text = re.sub(r"^[O0](?=\s)\s*", "", text)
    text = text.strip(" \t|,.;:-–—")
    if len(text) > MAX_LABEL_CHARS:
        text = text[:MAX_LABEL_CHARS].rsplit(" ", 1)[0] + "…"
    return text


def _plain_lines(value: str) -> str:
    """Strip markup but keep line breaks.

    ``form_extract._plain`` collapses newlines into spaces, which would merge a
    block like "Ja\\nFalls ja, E-Mail oder Telefonnummer" into one label and
    leave nothing for the per-line pick below to work with.
    """
    import re
    from html import unescape

    text = unescape(form_extract._TAG.sub(" ", str(value or "")))
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _blocks(page: "doc_ir.Page") -> List[Tuple[List[float], str]]:
    out = []
    for region in page.ordered_regions():
        text = _plain_lines(region.text)
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


FOOTER_BAND = 0.94  # crop below this fraction of the page to find "1 / 4"


def assign_sheet_pages(template) -> int:
    """Tag each control with its content column and printed page number.

    A two-up scan puts two questionnaire pages on one scan page, so "page 1"
    is useless to a human labeller. The form prints its own page number in the
    footer of each side, which is what the questionnaire actually says.
    """
    import base64
    import re

    import cv2
    import numpy as np

    tagged = 0
    for page in template.pages:
        if not page.blank_png_b64:
            continue
        blank = cv2.imdecode(
            np.frombuffer(base64.b64decode(page.blank_png_b64), np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        columns = _content_columns(markup_detect._ink_mask(
            cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR)
        ))
        if not columns:
            continue

        footers = []
        for x1, x2 in columns:
            strip = blank[int(page.height * FOOTER_BAND):page.height, x1:x2]
            text = _ocr_plain(strip)
            match = re.search(r"(\d+)\s*/\s*(\d+)", text)
            footers.append(f"{match.group(1)}/{match.group(2)}" if match else "")

        for control in page.controls:
            centre = (control.bbox[0] + control.bbox[2]) / 2 * page.width
            index = next(
                (i for i, (x1, x2) in enumerate(columns) if x1 <= centre <= x2),
                0,
            )
            control.column = index
            control.sheet_page = footers[index] if index < len(footers) else ""
            if control.sheet_page:
                tagged += 1
    return tagged


def _ocr_plain(crop) -> str:
    try:
        import pytesseract
    except Exception:
        return ""
    try:
        return pytesseract.image_to_string(crop, lang=OCR_LANG, config="--psm 7").strip()
    except Exception:
        return ""


def name_options(template) -> int:
    """Read each option's own text off the blank, where Paddle merged them.

    Mirror of the row-stem crop: the label sits to the right of its control,
    bounded by the next control on the same line or the edge of the content
    column. Only used for controls left with a positional placeholder.
    """
    import base64
    import re

    import cv2
    import numpy as np

    placeholder = re.compile(r"^option \d+$")
    named = 0
    for page in template.pages:
        if not page.blank_png_b64:
            continue
        blank = cv2.imdecode(
            np.frombuffer(base64.b64decode(page.blank_png_b64), np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        columns = _content_columns(markup_detect._ink_mask(
            cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR)
        ))
        boxes = {
            control.id: control.pixel_bbox(page.width, page.height)
            for control in page.controls
        }
        for control in page.controls:
            if not placeholder.match(control.label or ""):
                continue
            x1, y1, x2, y2 = boxes[control.id]
            centre = (y1 + y2) / 2

            right = next(
                (c[1] for c in columns if c[0] <= (x1 + x2) / 2 <= c[1]), page.width
            )
            for other in page.controls:
                ox1, oy1, ox2, oy2 = boxes[other.id]
                if other.id == control.id or ox1 <= x2:
                    continue
                if oy2 < centre or oy1 > centre:
                    continue
                right = min(right, ox1)

            pad = (y2 - y1) // 3
            crop = blank[max(0, y1 - pad):y2 + pad, x2 + 2:right]
            if crop.size == 0 or crop.shape[1] < 20:
                continue
            text = _ocr(crop, OCR_LANG)
            if text and not placeholder.match(text):
                control.label = text
                named += 1
    return named


def disambiguate_labels(template) -> int:
    """Force option labels to be distinct within each answer.

    Where Paddle merged an option list into one block every control in it
    inherits the same text. Identical labels make duplicate spreadsheet
    headers and give a human nothing to label against, so they are replaced
    by their printed position. Requires ``infer_structure`` to have run.
    """
    rows: Dict[str, List[Any]] = {}
    for page in template.pages:
        for control in page.controls:
            rows.setdefault(control.row_id or control.id, []).append((page, control))

    fixed = 0
    for members in rows.values():
        labels = [c.label for _, c in members]
        if len(set(labels)) == len(labels):
            continue
        ordered = survey_template.reading_order([c for _, c in members])
        for position, control in enumerate(ordered, start=1):
            control.label = f"option {position}"
        fixed += 1
    return fixed


# ==========================================
#            ROW STEM NAMING
# ==========================================

MIN_GUTTER = 120       # px of blank columns that separate content columns
MIN_STEM_GAP = 20      # px of blank columns between a stem and its controls
MIN_TEXT_HEIGHT = 6    # px of ink in a column before it counts as text, not a rule
MIN_OCR_CONFIDENCE = 65  # mean Tesseract word confidence below which a stem is dropped
OCR_LANG = "deu"


def _content_columns(ink, min_gutter: int = MIN_GUTTER) -> List[Tuple[int, int]]:
    """Column bands separated by full-height vertical whitespace.

    A two-up A3 scan holds two pages side by side; without this bound a row
    stem search runs off the edge of one page and reads the other one.
    """
    empty = (ink > 0).sum(axis=0) == 0
    gutters, start = [], None
    for x, is_empty in enumerate(empty):
        if is_empty and start is None:
            start = x
        elif not is_empty and start is not None:
            if x - start >= min_gutter:
                gutters.append((start, x))
            start = None
    if start is not None and len(empty) - start >= min_gutter:
        gutters.append((start, len(empty)))

    columns, left = [], 0
    for a, b in gutters:
        if a > left:
            columns.append((left, a))
        left = b
    if left < len(empty):
        columns.append((left, len(empty)))
    return columns


def _stem_span(ink, columns, x_limit: int, y1: int, y2: int) -> Optional[Tuple[int, int]]:
    """Horizontal span of the text block immediately left of a row's controls.

    Walks left from the controls: skip the whitespace separating them from the
    stem, take the ink that follows, and stop at the next wide gap.
    """
    column = next((c for c in columns if c[0] <= x_limit <= c[1]), None)
    if column is None:
        return None
    left_bound = column[0]
    band = ink[max(0, y1):y2, left_bound:x_limit]
    if band.size == 0:
        return None
    # Count ink height per column rather than presence: a table's horizontal
    # row rule spans the full width and would otherwise leave no gap anywhere.
    has_ink = (band > 0).sum(axis=0) >= MIN_TEXT_HEIGHT

    x = len(has_ink) - 1
    while x >= 0 and not has_ink[x]:          # gap between stem and controls
        x -= 1
    if x < 0 or (len(has_ink) - 1 - x) < MIN_STEM_GAP:
        return None
    end = x
    gap = 0
    while x >= 0:
        if has_ink[x]:
            gap = 0
        else:
            gap += 1
            if gap >= MIN_GUTTER:
                break
        x -= 1
    return left_bound + x + gap + 1, left_bound + end + 1


def _row_bands(groups, width: int, height: int) -> Dict[str, Tuple[int, int]]:
    """Vertical extent to read for each row: up to its neighbours' midpoints.

    A matrix stem often wraps onto a second line, so reading only the control's
    own height would clip it.
    """
    centres = []
    for row_id, controls in groups.items():
        boxes = [c.pixel_bbox(width, height) for c in controls]
        centres.append((sum((b[1] + b[3]) / 2 for b in boxes) / len(boxes), row_id, boxes))
    centres.sort()

    bands = {}
    for index, (centre, row_id, boxes) in enumerate(centres):
        above = centres[index - 1][0] if index else centre - (centre - min(b[1] for b in boxes)) * 4
        below = centres[index + 1][0] if index + 1 < len(centres) else (
            centre + (max(b[3] for b in boxes) - centre) * 4
        )
        bands[row_id] = (
            int(max(0, (above + centre) / 2 + 2)),
            int(min(height, (centre + below) / 2 - 2)),
        )
    return bands


def _ocr(crop, lang: str) -> str:
    """Read a stem strip, refusing the read when Tesseract is not confident.

    A garbled stem is worse than none: it would go straight into a column
    header and read as though the form said that.
    """
    try:
        import pytesseract
    except Exception:
        return ""
    try:
        data = pytesseract.image_to_data(
            crop, lang=lang, config="--psm 6",
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return ""

    words, confidences = [], []
    for text, confidence in zip(data.get("text", []), data.get("conf", [])):
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            continue
        if confidence < 0 or not str(text).strip():
            continue
        words.append(str(text).strip())
        confidences.append(confidence)

    if not words or sum(confidences) / len(confidences) < MIN_OCR_CONFIDENCE:
        return ""
    return sanitize(" ".join(words))


def name_answer_rows(template, *, lang: str = OCR_LANG) -> int:
    """Read the row stem beside each multi-control answer row.

    Paddle merges a matrix's row stems into one block, so they are recovered by
    cropping the strip left of each row and running Tesseract on that alone.
    Requires ``infer_structure`` to have assigned row ids.
    """
    import base64

    import cv2
    import numpy as np

    named = 0
    for page in template.pages:
        if not page.blank_png_b64:
            continue
        blank = cv2.imdecode(
            np.frombuffer(base64.b64decode(page.blank_png_b64), np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        ink = markup_detect._ink_mask(cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR))
        columns = _content_columns(ink)

        by_question: Dict[str, Dict[str, List[Any]]] = {}
        for control in page.controls:
            by_question.setdefault(control.question_id, {}).setdefault(
                control.row_id, []
            ).append(control)

        for groups in by_question.values():
            bands = _row_bands(groups, page.width, page.height)
            for row_id, controls in groups.items():
                if len(controls) < 2:
                    continue
                boxes = [c.pixel_bbox(page.width, page.height) for c in controls]
                # Only a horizontal run has a stem beside it; a vertical option
                # list is already named by the text against each option.
                if max(b[1] for b in boxes) >= min(b[3] for b in boxes):
                    continue
                y1, y2 = bands[row_id]
                span = _stem_span(ink, columns, min(b[0] for b in boxes), y1, y2)
                if span is None:
                    continue
                stem = _ocr(blank[y1:y2, span[0]:span[1]], lang)
                if stem:
                    template.row_labels[row_id] = stem
                    named += 1
    return named


def label_template(
    template: survey_template.SurveyTemplate,
    page_jsons: Sequence[Dict[str, Any]],
) -> int:
    """Label every page of *template* from the matching Paddle page result."""
    return sum(
        label_page(page, page_json)
        for page, page_json in zip(template.pages, page_jsons)
    )
