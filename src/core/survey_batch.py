"""Read a batch of questionnaires against a known template.

Structure comes from the template, so nothing here infers what the form
contains: every document answers the same fixed list of controls, which is
what makes one row per respondent well defined. Mark state is read by
``markup_detect.detect_markup_geometric`` on the registered crop.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import markup_detect, survey_template

UNCERTAIN = "uncertain"
REGISTRATION_FAILED = "registration_failed"


@dataclass
class ControlReading:
    control_id: str
    page_index: int
    state: str = UNCERTAIN
    score: float = 0.0
    fill_ratio: Optional[float] = None
    strike: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "control_id": self.control_id,
            "page": self.page_index + 1,
            "state": self.state,
            "score": self.score,
            "fill_ratio": self.fill_ratio,
            "strike": self.strike,
        }


@dataclass
class DocumentReading:
    document: str
    readings: List[ControlReading] = field(default_factory=list)
    registration: Dict[int, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def checked(self) -> int:
        return sum(1 for r in self.readings if r.state == "checked")

    @property
    def uncertain(self) -> int:
        return sum(1 for r in self.readings if r.state == UNCERTAIN)


def _blank_gray(page: survey_template.TemplatePage):
    import base64

    import cv2
    import numpy as np

    if not page.blank_png_b64:
        return None
    buf = np.frombuffer(base64.b64decode(page.blank_png_b64), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def read_document(
    pdf_path,
    template: survey_template.SurveyTemplate,
    *,
    dpi: Optional[int] = None,
    min_quality: float = survey_template.MIN_REGISTRATION_QUALITY,
    debug_dir=None,
) -> DocumentReading:
    """Read every template control on one questionnaire."""
    import cv2

    path = pathlib.Path(pdf_path)
    dpi = dpi or template.dpi
    result = DocumentReading(document=path.name)

    if not any(page.blank_png_b64 for page in template.pages):
        # Without a blank there is nothing to register against, and every
        # control would silently come back unread.
        raise ValueError(
            "Template carries no blank raster; rebuild it with build_template "
            "(the blanks are saved beside the template JSON)"
        )

    for page in template.pages:
        reference = _blank_gray(page)
        if reference is None:
            result.warnings.append(
                f"page {page.page_index + 1}: template carries no blank raster"
            )
            continue

        gray = survey_template.render_gray(path, page.page_index, dpi)
        if gray.shape != reference.shape:
            gray = cv2.resize(gray, (reference.shape[1], reference.shape[0]))

        H, quality = survey_template.register(gray, reference)
        result.registration[page.page_index] = round(quality, 3)
        if H is None or quality < min_quality:
            result.warnings.append(
                f"page {page.page_index + 1}: {REGISTRATION_FAILED} (quality={quality:.2f})"
            )
            result.readings.extend(
                ControlReading(control.id, page.page_index, state=REGISTRATION_FAILED)
                for control in page.controls
            )
            continue

        warped = survey_template.warp_to_reference(gray, H, reference.shape)
        residual = survey_template.residual_ink(reference, warped)
        states: List[str] = []
        for control in page.controls:
            x1, y1, x2, y2 = control.pixel_bbox(page.width, page.height)
            verdict = survey_template.classify_residual(
                residual[y1:y2, x1:x2],
                survey_template.halo_crop(residual, (x1, y1, x2, y2)),
            )
            # Stroke geometry is not needed to decide, but it is cheap evidence
            # for a human adjudicating an uncertain cell.
            strike = None
            if verdict["state"] == UNCERTAIN:
                crop = warped[y1:y2, x1:x2]
                if crop.size:
                    strike = markup_detect.detect_markup_geometric(
                        cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
                    ).get("strike")
            result.readings.append(
                ControlReading(
                    control_id=control.id,
                    page_index=page.page_index,
                    state=str(verdict.get("state") or UNCERTAIN),
                    score=float(verdict.get("score") or 0.0),
                    fill_ratio=verdict.get("fill_ratio"),
                    strike=strike,
                )
            )
            states.append(result.readings[-1].state)

        _resolve_shared_marks(result, page, template, residual)

        if debug_dir is not None:
            out = pathlib.Path(debug_dir)
            out.mkdir(parents=True, exist_ok=True)
            vis = survey_template.overlay(
                warped,
                [{"bbox": c.pixel_bbox(page.width, page.height), "shape": c.shape}
                 for c in page.controls],
                states=states,
            )
            cv2.imwrite(str(out / f"{path.stem}_page{page.page_index + 1}.png"), vis)

    return result


SHARED_MARK_SCORE = 0.5  # answered, but the contest stays visible in the export


def _resolve_shared_marks(result, page, template, residual) -> None:
    """Undo a double read caused by one stroke covering two controls.

    Only single-choice answers that came back with more than one mark are
    touched, so this can turn a flagged answer into a definite one but cannot
    change an answer that was already unambiguous.
    """
    by_id = {reading.control_id: reading for reading in result.readings}
    groups: Dict[str, List[survey_template.TemplateControl]] = {}
    for control in page.controls:
        groups.setdefault(control.row_id or control.id, []).append(control)

    for row_id, controls in groups.items():
        if template.rules.get(row_id, "single") != "single":
            continue
        checked = [
            control for control in controls
            if control.id in by_id and by_id[control.id].state == "checked"
        ]
        if len(checked) < 2:
            continue

        boxes = [c.pixel_bbox(page.width, page.height) for c in checked]
        winner = survey_template.dominant_control(residual, boxes)
        if winner is None:
            continue
        for index, control in enumerate(checked):
            reading = by_id[control.id]
            reading.state = "checked" if index == winner else "unchecked"
            reading.score = SHARED_MARK_SCORE


def read_batch(
    pdf_paths: Sequence[Any],
    template: survey_template.SurveyTemplate,
    *,
    dpi: Optional[int] = None,
    debug_dir=None,
    progress=None,
) -> List[DocumentReading]:
    results = []
    paths = [pathlib.Path(p) for p in pdf_paths]
    for index, path in enumerate(paths, start=1):
        if progress is not None:
            progress(index / len(paths), f"Reading {path.name} ({index}/{len(paths)})")
        results.append(read_document(path, template, dpi=dpi, debug_dir=debug_dir))
    return results


# ==========================================
#               AGGREGATION
# ==========================================


def _column_names(template: Optional[survey_template.SurveyTemplate]) -> Dict[str, str]:
    """Readable, unique column name per control id."""
    if template is None:
        return {}
    names: Dict[str, str] = {}
    used: Dict[str, int] = {}
    for page in template.pages:
        for control in page.controls:
            parts = [part for part in (control.question_id, control.label) if part]
            if not parts:
                continue
            name = " | ".join(parts)
            used[name] = used.get(name, 0) + 1
            if used[name] > 1:
                name = f"{name} ({used[name]})"
            names[control.id] = name
    return names


def readings_in(result: DocumentReading,
                template: Optional[survey_template.SurveyTemplate]):
    """A document's readings for the controls the template still has.

    The readings are taken once, against the template as it was learned. When a
    control is dropped afterwards -- printed text that fooled the detector --
    its reading is still in hand and must not reach an export: the template is
    what says which controls exist.
    """
    if template is None:
        return result.readings
    known = {control.id for page in template.pages for control in page.controls}
    return [reading for reading in result.readings if reading.control_id in known]


def to_wide(results: Sequence[DocumentReading],
            template: Optional[survey_template.SurveyTemplate] = None):
    """One row per respondent, one column per control."""
    import pandas as pd

    names = _column_names(template)
    records = []
    for result in results:
        row: Dict[str, Any] = {"document": result.document}
        row.update({
            names.get(r.control_id, r.control_id): r.state
            for r in readings_in(result, template)
        })
        records.append(row)
    return pd.DataFrame(records)


def to_long(results: Sequence[DocumentReading], template: survey_template.SurveyTemplate):
    """One row per respondent-control, carrying the geometric evidence."""
    import pandas as pd

    labels = {
        control.id: (control.question_id, control.row_id, control.label)
        for page in template.pages
        for control in page.controls
    }
    records = []
    for result in results:
        for reading in readings_in(result, template):
            question_id, row_id, label = labels.get(reading.control_id, ("", "", ""))
            records.append({
                "document": result.document,
                "control_id": reading.control_id,
                "question_id": question_id,
                "row_id": row_id,
                "row_label": template.row_labels.get(row_id, ""),
                "label": label,
                "page": reading.page_index + 1,
                "state": reading.state,
                "score": reading.score,
                "fill_ratio": reading.fill_ratio,
                "strike": reading.strike,
            })
    return pd.DataFrame(records)


CERTAINTY_SUFFIX = " [certainty]"
MULTIPLE = "MULTIPLE"


def resolve_single(states) -> Tuple[str, float]:
    """Collapse one single-choice answer's controls into a value.

    *states* is ``[(control, state, score), ...]``. Returns the chosen option's
    label, or ``""`` when the respondent left the answer blank, or a marker
    when the answer cannot be resolved. Shared by the export and the scorer so
    the two can never disagree about what the pipeline answered.
    """
    chosen = [control for control, state, _ in states if state == "checked"]
    unread = [s for _, state, s in states if state == UNCERTAIN]
    weakest = min((score for _, _, score in states), default=0.0)

    if unread or len(chosen) > 1:
        # Contradictory or unreadable: say so rather than pick one.
        return (MULTIPLE if len(chosen) > 1 else UNCERTAIN.upper()), 0.0
    if len(chosen) == 1:
        return (chosen[0].label or chosen[0].id), weakest
    return "", weakest


def _unique(name: str, used: Dict[str, int]) -> str:
    used[name] = used.get(name, 0) + 1
    return name if used[name] == 1 else f"{name} ({used[name]})"


def _row_plan(template: survey_template.SurveyTemplate):
    """Ordered (row_id, rule, [(control, column name)]) for the whole form."""
    rows: Dict[str, List[survey_template.TemplateControl]] = {}
    for page in template.pages:
        for control in page.controls:
            rows.setdefault(control.row_id or control.question_id or "ungrouped", []).append(
                control
            )

    plan, used = [], {}
    for row_id, controls in rows.items():
        name = template.display_name(row_id)
        controls = survey_template.reading_order(controls)
        columns = [
            (control, _unique(f"{name} | {control.label or control.id}", used))
            for control in controls
        ]
        plan.append((row_id, name, template.rules.get(row_id, "single"), columns))
    return plan


def to_checkbox_table(
    results: Sequence[DocumentReading],
    template: survey_template.SurveyTemplate,
):
    """One row per respondent in the requested export shape.

    Every checkbox gets a TRUE/FALSE column, and every single-choice answer
    additionally gets a column naming the option that was chosen. Each data
    column is followed by its certainty, so a reviewer can sort on it and only
    open the questionnaires that need a human.

    Certainty is the geometric margin of the read (how far the measured ink sits
    from the decision thresholds), not a validated probability. 1.0 means the
    control was unambiguously empty or unambiguously marked.
    """
    import pandas as pd

    plan = _row_plan(template)
    records = []
    for result in results:
        readings = {r.control_id: r for r in result.readings}
        record: Dict[str, Any] = {
            "document": result.document,
            "registration": round(min(result.registration.values(), default=0.0), 3),
        }

        for _row_id, row_name, rule, columns in plan:
            states = []
            for control, column in columns:
                reading = readings.get(control.id)
                state = reading.state if reading else UNCERTAIN
                states.append((control, state, reading.score if reading else 0.0))
                record[column] = {"checked": True, "unchecked": False}.get(state, "")
                record[column + CERTAINTY_SUFFIX] = round(reading.score if reading else 0.0, 3)

            if rule != "single":
                continue

            value, certainty = resolve_single(states)
            record[row_name] = value
            record[row_name + CERTAINTY_SUFFIX] = round(float(certainty), 3)

        records.append(record)

    frame = pd.DataFrame(records)
    # value column first, then its checkboxes, each followed by its certainty
    order = ["document", "registration"]
    for _row_id, name, rule, columns in plan:
        if rule == "single" and name in frame.columns:
            order += [name, name + CERTAINTY_SUFFIX]
        for _, name in columns:
            order += [name, name + CERTAINTY_SUFFIX]
    return frame[[c for c in order if c in frame.columns]]


def review_queue(results: Sequence[DocumentReading], template: survey_template.SurveyTemplate):
    """Every cell a human should look at, most doubtful first."""
    long = to_long(results, template)
    if long.empty:
        return long
    flagged = long[long["state"].isin({UNCERTAIN, REGISTRATION_FAILED})]
    return flagged.sort_values(["score", "document", "control_id"]).reset_index(drop=True)


def unused_controls(
    results: Sequence[DocumentReading],
    template: survey_template.SurveyTemplate,
):
    """Controls no respondent ever marked.

    A single unmarked control is usually just an unpopular option. A whole
    answer row that nobody touched is the interesting case: it is normally a
    detection false positive -- printed text that passed the shape filters --
    so ``whole_row_unused`` is the column to sort on in the template pass.
    """
    import pandas as pd

    marked = {
        reading.control_id
        for result in results for reading in result.readings
        if reading.state == "checked"
    }
    rows_marked = {
        control.row_id
        for page in template.pages for control in page.controls
        if control.id in marked
    }
    records = [
        {
            "control_id": control.id,
            "page": page.page_index + 1,
            "row_id": control.row_id,
            "row_label": template.row_labels.get(control.row_id, ""),
            "label": control.label,
            "shape": control.shape,
            "whole_row_unused": control.row_id not in rows_marked,
        }
        for page in template.pages for control in page.controls
        if control.id not in marked
    ]
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame = frame.sort_values(
            ["whole_row_unused", "page", "row_id"], ascending=[False, True, True]
        ).reset_index(drop=True)
    return frame


def summarize(
    results: Sequence[DocumentReading],
    template: Optional[survey_template.SurveyTemplate] = None,
) -> Dict[str, Any]:
    kept = [readings_in(result, template) for result in results]
    total = sum(len(readings) for readings in kept)
    checked = sum(1 for readings in kept for r in readings if r.state == "checked")
    uncertain = sum(1 for readings in kept for r in readings if r.state == UNCERTAIN)
    return {
        "documents": len(results),
        "controls_per_document": total // max(1, len(results)),
        "checked": checked,
        "uncertain": uncertain,
        "uncertain_rate": round(uncertain / max(1, total), 4),
        "documents_with_warnings": sum(1 for r in results if r.warnings),
        "worst_registration": round(
            min((min(r.registration.values(), default=0.0) for r in results), default=0.0), 3
        ),
    }


# ==========================================
#          GROUND TRUTH AND SCORING
# ==========================================

BLANK_MARK = ""          # respondent left the answer empty
AMBIGUOUS_MARK = "?"     # a human could not tell either
MULTI_SEPARATOR = ";"


def answer_sheet(template: survey_template.SurveyTemplate, documents: Sequence[str]):
    """A blank sheet for a human to fill in, one line per answer.

    Deliberately blank rather than pre-filled with the pipeline's reads: a
    corrected copy of the output would measure whether a reviewer notices
    mistakes, not whether the pipeline makes them.
    """
    import pandas as pd

    scan_page = {
        control.row_id: page.page_index
        for page in template.pages for control in page.controls
    }
    rows = template.rows()

    # Reading order across the whole form: scan page, then the column of the
    # two-up spread, then down the page.
    ordered = sorted(
        rows.items(),
        key=lambda item: (
            scan_page.get(item[0], 0), item[1][0].column, item[1][0].bbox[1]
        ),
    )

    # Where a question has several answers and no printed row name, say which
    # one this is: otherwise two rows of "option 1..4" are indistinguishable.
    siblings: Dict[str, int] = {}
    for row_id, controls in ordered:
        siblings[controls[0].question_id] = siblings.get(controls[0].question_id, 0) + 1
    position: Dict[str, int] = {}

    records = []
    for document in documents:
        position.clear()
        for row_id, controls in ordered:
            question = controls[0].question_id
            position[question] = position.get(question, 0) + 1
            row = template.row_labels.get(row_id, "")
            if not row and siblings[question] > 1:
                row = f"row {position[question]} of {siblings[question]} (top to bottom)"
            number = question.rsplit("_q", 1)[-1]
            records.append({
                "document": document,
                "answer_id": row_id,
                "sheet_page": controls[0].sheet_page or "?",
                "question": f"Q{number}" if number else "",
                "row": row,
                "type": template.rules.get(row_id, "single"),
                "options": " | ".join(c.label or c.id for c in controls),
                "answer": "",
            })
    return pd.DataFrame(records)


def _pipeline_answers(
    results: Sequence[DocumentReading],
    template: survey_template.SurveyTemplate,
) -> Dict[Tuple[str, str], Tuple[str, float]]:
    """(document, answer_id) -> (value, certainty), matching the export."""
    rows: Dict[str, List[survey_template.TemplateControl]] = {}
    for page in template.pages:
        for control in page.controls:
            rows.setdefault(control.row_id, []).append(control)

    answers = {}
    for result in results:
        readings = {r.control_id: r for r in result.readings}
        for row_id, controls in rows.items():
            controls = survey_template.reading_order(controls)
            states = [
                (
                    control,
                    readings[control.id].state if control.id in readings else UNCERTAIN,
                    readings[control.id].score if control.id in readings else 0.0,
                )
                for control in controls
            ]
            if template.rules.get(row_id, "single") == "single":
                answers[(result.document, row_id)] = resolve_single(states)
            else:
                chosen = sorted(
                    (c.label or c.id) for c, state, _ in states if state == "checked"
                )
                unread = any(state == UNCERTAIN for _, state, _ in states)
                answers[(result.document, row_id)] = (
                    UNCERTAIN.upper() if unread else MULTI_SEPARATOR.join(chosen),
                    min((score for _, _, score in states), default=0.0),
                )
    return answers


def _norm_text(value: str) -> str:
    import re

    value = re.sub(r"\s+", " ", str(value or "")).strip()
    return re.sub(r"[^\w\s]", "", value, flags=re.UNICODE).casefold().strip()


def match_option(text: str, options: Sequence[str]) -> Optional[str]:
    """Resolve a hand-written answer to one of the printed options.

    A labeller copying a long option shortens it ("Ja Falls ja" for "Ja Falls
    ja, E-Mail oder Telefonnummer"), and scoring that as a pipeline error would
    be measuring transcription, not extraction. Returns None when the text
    cannot be pinned to exactly one option, so it can be reported rather than
    silently counted against either side.
    """
    wanted = _norm_text(text)
    if not wanted:
        return ""
    lookup = {_norm_text(option): option for option in options}
    if wanted in lookup:
        return lookup[wanted]
    for test in (
        lambda a, b: a.startswith(b) or b.startswith(a),
        lambda a, b: a in b or b in a,
    ):
        hits = [option for norm, option in lookup.items() if test(norm, wanted)]
        if len(hits) == 1:
            return hits[0]
    return None


def _resolve_answer(value: str, options: Sequence[str]):
    """(canonical answer, unresolved parts) for a possibly multi-part answer."""
    parts = [p.strip() for p in str(value or "").split(MULTI_SEPARATOR) if p.strip()]
    if not parts:
        return "", []
    resolved, unresolved = [], []
    for part in parts:
        match = match_option(part, options)
        if match is None:
            unresolved.append(part)
        elif match:
            resolved.append(match)
    return MULTI_SEPARATOR.join(sorted(resolved)).casefold(), unresolved


def score_sheet(
    sheet,
    results: Sequence[DocumentReading],
    template: survey_template.SurveyTemplate,
):
    """Compare a filled answer sheet with what the pipeline read.

    Returns ``(per_answer, summary)``. Rows the human marked ambiguous are
    excluded from accuracy: there is no ground truth to score against.
    """
    import pandas as pd

    predicted = _pipeline_answers(results, template)
    options = {
        row_id: [c.label or c.id for c in controls]
        for row_id, controls in template.rows().items()
    }
    records = []
    for row in sheet.to_dict("records"):
        key = (str(row["document"]), str(row["answer_id"]))
        if key not in predicted:
            continue
        value, certainty = predicted[key]
        truth = str(row.get("answer") or "").strip()
        flagged = value in {MULTIPLE, UNCERTAIN.upper()}
        choices = options.get(key[1], [])
        truth_canonical, unresolved = _resolve_answer(truth, choices)
        value_canonical, _ = _resolve_answer(value, choices)
        records.append({
            "document": key[0],
            "answer_id": key[1],
            "row": row.get("row", ""),
            "truth": truth,
            "predicted": value,
            "certainty": certainty,
            "flagged": flagged,
            "correct": (not flagged) and not unresolved
            and truth_canonical == value_canonical,
            "human_unsure": truth == AMBIGUOUS_MARK,
            "unmatched_label": MULTI_SEPARATOR.join(unresolved),
        })

    per_answer = pd.DataFrame(records)
    if per_answer.empty:
        return per_answer, {"scored": 0}

    scorable = per_answer[
        (~per_answer["human_unsure"]) & (per_answer["unmatched_label"] == "")
    ]
    auto = scorable[~scorable["flagged"]]
    wrong = auto[~auto["correct"]]
    summary = {
        "answers_scored": int(len(scorable)),
        "flagged_for_review": int(scorable["flagged"].sum()),
        "auto_accepted": int(len(auto)),
        "auto_accepted_correct": int(auto["correct"].sum()),
        "auto_accepted_accuracy": round(auto["correct"].mean(), 4) if len(auto) else None,
        "silent_errors": int(len(wrong)),
        "human_unsure": int(per_answer["human_unsure"].sum()),
        "labels_not_matched_to_an_option": int((per_answer["unmatched_label"] != "").sum()),
    }
    if len(wrong):
        # The number that matters: does certainty actually rank the mistakes?
        summary["max_certainty_of_a_silent_error"] = round(wrong["certainty"].max(), 3)
        summary["median_certainty_of_a_silent_error"] = round(wrong["certainty"].median(), 3)
    return per_answer, summary


# ==========================================
#           BATCH ORCHESTRATION
# ==========================================


def prepare_template(
    paths: Sequence[Any],
    *,
    label: bool = True,
    dpi: Optional[int] = None,
    progress=None,
    vl_session=None,
):
    """Learn the questionnaire from a batch: blank, controls, structure, names.

    Shared by the CLI and the TextLab batch page so the two cannot drift.
    *label* runs PaddleOCR-VL over the synthesized blanks to recover question
    grouping and option text; everything else needs only OpenCV and Tesseract.
    *vl_session* is a resident worker to run that on, so the batch loads the
    recognition weights once rather than once here and once per file.
    """
    from core import survey_label

    def _say(fraction, text):
        if progress is not None:
            progress(fraction, text)

    template, blanks = survey_template.build_template(
        paths, dpi=dpi or survey_template.DEFAULT_DPI,
        progress=lambda f, t: _say(0.1 + 0.5 * f, t),
    )

    if label:
        _say(0.65, "Reading the questionnaire's own wording...")
        try:
            template_page_jsons = _blank_layout(template, blanks, vl_session)
            survey_label.label_template(template, template_page_jsons)
        except Exception as exc:  # layout is an enrichment, not a prerequisite
            template.provenance["labelling_error"] = str(exc)

    # With few copies a popular option is marked by a majority and survives
    # the median, leaving ghost ink in the "blank" that every read then
    # subtracts away. Say so rather than quietly returning a worse template.
    stacked = min((len(b.contributors) for b in blanks), default=0)
    if stacked < survey_template.MIN_BLANK_DOCUMENTS:
        template.provenance["small_batch_warning"] = (
            f"The blank form was averaged from only {stacked} questionnaire(s). "
            f"Below {survey_template.MIN_BLANK_DOCUMENTS} an option that most "
            "respondents chose can survive as a faint mark on the blank and be "
            "subtracted from every answer. Results are usable but check the "
            "template overlay."
        )

    _say(0.85, "Working out which answers belong together...")
    survey_template.infer_structure(template)
    survey_label.disambiguate_labels(template)
    survey_label.assign_sheet_pages(template)
    survey_label.name_options(template)
    survey_label.disambiguate_labels(template)
    survey_template.infer_structure(template)
    survey_label.name_answer_rows(template)
    return template, blanks


def _blank_layout(template, blanks, vl_session=None):
    """Run the layout/OCR worker over the synthesized blanks."""
    import tempfile

    import cv2

    from core import auto_ocr

    with tempfile.TemporaryDirectory() as tmp:
        images = []
        for page, blank in zip(template.pages, blanks):
            path = pathlib.Path(tmp) / f"blank_page{page.page_index + 1}.png"
            cv2.imwrite(str(path), blank.image)
            images.append(path)
        return auto_ocr.run_vl_worker(images, session=vl_session)


def template_overlays(
    template: survey_template.SurveyTemplate, *, tag_answers: bool = True
) -> Dict[str, bytes]:
    """Audit PNG per page: the synthesized blank with every control outlined.

    Each answer group is tagged with the name it carries in the exports, so the
    column called "p3/4 Q10 Biodiversitaet" can be found on the paper without
    counting rows. Drawn from the blank the template already carries, so a
    refined template can be re-rendered without the original questionnaires.
    """
    import cv2

    images = {}
    for page in template.pages:
        blank = _blank_gray(page)
        if blank is None:
            continue
        vis = survey_template.overlay(
            blank,
            [{"bbox": c.pixel_bbox(page.width, page.height), "shape": c.shape}
             for c in page.controls],
        )
        if tag_answers:
            _tag_answers(vis, page, template)
        ok, encoded = cv2.imencode(".png", vis)
        if ok:
            images[f"template_page{page.page_index + 1}.png"] = encoded.tobytes()
    return images


def question_tag(control: survey_template.TemplateControl) -> str:
    """Where a question sits on the paper, e.g. ``p3/4 Q10``.

    The same anchor on the overlay image, in the CSV column names and in
    ``document.md``, so one name finds a question in all three.
    """
    number = (
        control.question_id.rsplit("_q", 1)[-1] if "_q" in control.question_id else ""
    )
    parts = [
        f"p{control.sheet_page}" if control.sheet_page else "",
        f"Q{number}" if number else "",
    ]
    return " ".join(part for part in parts if part).strip() or (
        control.question_id or control.row_id
    )


def _tag_answers(image, page, template) -> None:
    """Box each question and name it once.

    Per-answer tags drown a matrix: sixteen labels stacked down one grid say
    less than a single box round the question. The row names are printed on the
    form itself, so the question is the level worth marking.
    """
    import cv2

    scale = max(0.7, page.width / 3600.0)
    thickness = max(1, int(round(scale * 1.8)))
    pad = int(14 * scale)
    colour = (0, 110, 220)

    questions = {}
    for control in page.controls:
        questions.setdefault(control.question_id or control.row_id, []).append(control)

    for question_id, controls in questions.items():
        boxes = [c.pixel_bbox(page.width, page.height) for c in controls]
        x1 = max(0, min(b[0] for b in boxes) - pad)
        y1 = max(0, min(b[1] for b in boxes) - pad)
        x2 = min(image.shape[1] - 1, max(b[2] for b in boxes) + pad)
        y2 = min(image.shape[0] - 1, max(b[3] for b in boxes) + pad)
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness)

        first = survey_template.reading_order(controls)[0]
        text = question_tag(first)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)

        tx = min(x1, max(0, image.shape[1] - tw - 4))
        ty = y1 - int(6 * scale)
        if ty - th < 0:
            ty = y2 + th + int(6 * scale)
        cv2.rectangle(image, (tx - 3, ty - th - 5), (tx + tw + 3, ty + 4), (255, 255, 255), -1)
        cv2.putText(image, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, colour, thickness, cv2.LINE_AA)


def drop_controls(
    template: survey_template.SurveyTemplate, control_ids: Sequence[str]
) -> int:
    """Remove controls and re-derive the structure that depended on them."""
    from core import survey_label

    drop = {str(c) for c in control_ids}
    before = template.control_count
    for page in template.pages:
        page.controls = [c for c in page.controls if c.id not in drop]
    survey_template.infer_structure(template)
    survey_label.disambiguate_labels(template)
    survey_label.name_answer_rows(template)
    return before - template.control_count


def answer_overview(
    results: Sequence[DocumentReading],
    template: survey_template.SurveyTemplate,
):
    """One line per answer group: what it is, and how often anyone marked it.

    The review table for the batch page. An answer nobody in the batch touched
    is either a question they all skipped or a detection false positive, and
    only a human looking at the overlay can tell which.
    """
    import pandas as pd

    marked = {
        reading.control_id
        for result in results for reading in result.readings
        if reading.state == "checked"
    }
    records = []
    for row_id, controls in template.rows().items():
        respondents = {
            result.document
            for result in results for reading in result.readings
            if reading.state == "checked"
            and reading.control_id in {c.id for c in controls}
        }
        records.append({
            "answer": template.display_name(row_id),
            "answer_id": row_id,
            "sheet_page": controls[0].sheet_page,
            "type": template.rules.get(row_id, "single"),
            "options": len(controls),
            "answered_by": len(respondents),
            "never_marked": not any(c.id in marked for c in controls),
            "control_ids": ",".join(c.id for c in controls),
        })
    return pd.DataFrame(records)


SURVEY_README = """# Questionnaire responses — what these files are

TextLab read a batch of the **same paper questionnaire**, filled in by different
people. It learned the blank form from the batch itself, then read every
questionnaire against it. No text-recognition model decides the answers: each
response box is compared against the same box on the blank form, so what is
reported is whether *that* box was marked.

Alongside these files, each questionnaire has its own folder with the ordinary
text extraction (`document.md`, `.txt`, `.docx`, `.json`). The answers are
written into those too, each one under the question it belongs to, so a single
questionnaire can be read as a document instead of as a table.

## Start here

| File | What it is |
|---|---|
| `responses_checkboxes.csv` | **The main table.** One row per questionnaire, one column per checkbox. |
| `<file>/survey_answers.csv` | One questionnaire's answers, one line per question. Easier to read for a single respondent. |
| `review_queue.csv` | Only the answers a human should check. Start your review here. |
| `template_page*.png` | The blank form with every detected response box outlined and each question boxed and named. **This is the key between a column name and a place on the paper.** |

## Reading `responses_checkboxes.csv`

Column names look like `p3/4 Q10 Biodiversität | 4`:

* `p3/4` — the questionnaire's own printed page number, from its footer. A scan
  page may hold two of these side by side.
* `Q10` — the question number printed on the form.
* `Biodiversität` — the row, for a question with several rows.
* `4` — the option.

For each question you get:

* **one column per checkbox**, holding `True` (marked) or `False` (not marked);
* **for single-choice questions, an extra column** naming the option chosen —
  empty when the respondent answered nothing;
* **a `[certainty]` column beside every one of those.**

Two values need care in an answer column:

* `MULTIPLE` — more than one option was marked where only one was allowed. The
  answer is not guessed; look at the questionnaire.
* `UNCERTAIN` — at least one box in that question could not be read confidently.

`registration` is how well the scan lined up with the blank form (1.0 is
perfect). A low value on one questionnaire usually explains why it has more
uncertain answers than the rest.

## Certainty, and what it is not

Certainty is how far the measured ink sat from the decision threshold: 1.0
means the box was unmistakably empty or unmistakably marked. It is a way of
**ranking what to check**, not a probability that the answer is right.

Measured against 11 questionnaires answered by hand for comparison: of 338
answers the pipeline accepted on its own, 338 were correct, and about one
answer in ten was flagged for a human instead of being answered. Those numbers
come from one questionnaire design; treat them as encouraging, not as a
guarantee for a different form.

## The other files

| File | What it is |
|---|---|
| `responses_long.csv` | One row per questionnaire *and* checkbox, with the raw ink measurements. For analysis or debugging. |
| `responses_matrix.csv` | Every checkbox's state as text (`checked`/`unchecked`/`uncertain`). |
| `answers_overview.csv` | One line per question: its type, how many respondents answered it. |
| `unused_controls.csv` | Boxes nobody marked. Where `whole_row_unused` is true, the whole answer was untouched — usually printed text mistaken for a checkbox, occasionally a question everyone skipped. |
| `survey_template.json` + `survey_template_blank_page*.png` | The learned form. Keep these to read more copies of the same questionnaire later. |

## Before trusting the numbers

1. Open `template_page*.png` and check the outlined boxes really are the
   response boxes. If printed text was mistaken for one, remove it in TextLab
   and rebuild, then download the ZIP again — the answers are not read again,
   only regrouped, and the rebuild rewrites every file that carries them: these
   tables, each `<file>/survey_answers.csv`, and each `<file>/document.*`.
2. Work through `review_queue.csv`.
3. Check any questionnaire with a low `registration`.
"""


def slim_document(document):
    """Strip a parsed document down to what re-exporting it needs.

    The preview raster and the searchable PDF are the bulk of a parsed page and
    both are already written out; keeping a whole batch of them in memory so a
    template can be refined afterwards would not fit. The preview's pixel size
    is kept, because the region boxes were scaled to it.
    """
    for page in document.pages:
        if page.image_b64 and not page.preview_size:
            page.preview_size = preview_size(page)
        page.image_b64 = None
    document.searchable_pdf = None
    return document


def _document_exports(template, readings, documents) -> Dict[str, bytes]:
    """Fresh per-file exports for a refined template: ``{zip path: bytes}``.

    The answers live inside ``document.md`` as well as in the tables, so a
    dropped control has to be taken out of both. Nothing is parsed again: the
    documents already in hand are re-rendered against the pruned template.
    """
    import tempfile

    from core import doc_ir

    by_stem = {
        pathlib.PurePosixPath(reading.document).stem: reading for reading in readings
    }
    out: Dict[str, bytes] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for rel_dir, document in documents.items():
            reading = by_stem.get(pathlib.PurePosixPath(rel_dir).name)
            if reading is None:
                continue
            to_form_groups(reading, template, document)
            target = pathlib.Path(tmp) / rel_dir
            names = doc_ir.write_document_outputs(
                document, target, "document", provenance=False,
                skip_tables=survey_table_regions(document, template),
                form_responses=False,
            )
            for name in names:
                out[f"{rel_dir}/{pathlib.PurePosixPath(name).as_posix()}"] = (
                    (target / name).read_bytes()
                )
    return out


def rebuild_exports(zip_bytes, template, readings, documents=None):
    """Rewrite every answer-carrying file in a finished result ZIP.

    Dropping a control changes only how answers are grouped and exported, not
    what was read off the page, so the questionnaires are not parsed again: the
    batch tables, each file's own answers and each file's document exports are
    re-rendered from the readings already in hand. Returns the new ZIP and the
    batch summary.
    """
    import io
    import tempfile
    import zipfile

    documents = documents or {}
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "survey"
        summary = write_batch_outputs(readings, template, out)
        replacements = {
            f"survey/{path.name}": path.read_bytes() for path in out.iterdir()
        }

    buffer = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as source:
        # Same writer as the first pass, so a rebuild cannot quietly change a
        # file's own answers back to the wide batch shape.
        per_file = {
            pathlib.PurePosixPath(reading.document).stem: reading
            for reading in readings
        }
        for item in source.namelist():
            if item.endswith("/survey_answers.csv"):
                # Folder names are the file stems, so match on the stem:
                # "split_1_2" must not claim "split_1_20".
                reading = per_file.get(pathlib.PurePosixPath(item).parent.name)
                if reading is not None:
                    replacements[item] = answers_for_document(
                        reading, template
                    ).to_csv(index=False).encode("utf-8")
        replacements.update(_document_exports(template, readings, documents))
        # Which tables count as questionnaire grids moves with the template, so
        # a re-exported file's table CSVs replace the old set rather than join it.
        stale = tuple(f"{rel_dir}/tables/" for rel_dir in documents)

        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as target:
            for item in source.namelist():
                if item.startswith("survey/") or item in replacements:
                    continue
                if stale and item.startswith(stale):
                    continue
                target.writestr(item, source.read(item))
            for name, data in replacements.items():
                target.writestr(name, data)
    return buffer.getvalue(), summary


def write_batch_outputs(
    results: Sequence[DocumentReading],
    template: survey_template.SurveyTemplate,
    out_dir,
) -> Dict[str, Any]:
    """Write the questionnaire tables and audit artifacts for one batch."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    to_checkbox_table(results, template).to_csv(
        out / "responses_checkboxes.csv", index=False
    )
    to_wide(results, template).to_csv(out / "responses_matrix.csv", index=False)
    to_long(results, template).to_csv(out / "responses_long.csv", index=False)
    review_queue(results, template).to_csv(out / "review_queue.csv", index=False)
    unused = unused_controls(results, template)
    unused.to_csv(out / "unused_controls.csv", index=False)
    answer_overview(results, template).to_csv(out / "answers_overview.csv", index=False)
    (out / "README.md").write_text(SURVEY_README, encoding="utf-8")
    template.save(out / "survey_template.json")

    for name, data in template_overlays(template).items():
        (out / name).write_bytes(data)

    summary = summarize(results, template)
    summary["controls"] = template.control_count
    summary["answer_groups"] = len(template.rules)
    summary["unused_controls"] = int(len(unused))
    return summary


def answers_for_document(
    result: DocumentReading, template: survey_template.SurveyTemplate
):
    """One respondent's answers, one row per question.

    The batch table is wide by necessity -- a row per respondent means a column
    per checkbox. A single file's own folder is better served the other way up:
    ~35 readable lines instead of one 400-column line.
    """
    import pandas as pd

    readings = {r.control_id: r for r in result.readings}
    scan_page = {
        control.row_id: page.page_index
        for page in template.pages for control in page.controls
    }
    records = []
    for row_id, controls in sorted(
        template.rows().items(),
        key=lambda item: (
            scan_page.get(item[0], 0), item[1][0].column, item[1][0].bbox[1]
        ),
    ):
        states = [
            (
                control,
                readings[control.id].state if control.id in readings else UNCERTAIN,
                readings[control.id].score if control.id in readings else 0.0,
            )
            for control in controls
        ]
        rule = template.rules.get(row_id, "single")
        if rule == "single":
            answer, certainty = resolve_single(states)
        else:
            chosen = [c.label or c.id for c, state, _ in states if state == "checked"]
            unread = any(state == UNCERTAIN for _, state, _ in states)
            answer = UNCERTAIN.upper() if unread else MULTI_SEPARATOR.join(sorted(chosen))
            certainty = min((score for _, _, score in states), default=0.0)

        number = controls[0].question_id.rsplit("_q", 1)[-1]
        records.append({
            "document": result.document,
            "sheet_page": controls[0].sheet_page,
            "question": f"Q{number}" if number else "",
            "row": template.row_labels.get(row_id, ""),
            "type": rule,
            "answer": answer,
            "certainty": round(float(certainty), 3),
            "marked": MULTI_SEPARATOR.join(
                (c.label or c.id) for c, state, _ in states if state == "checked"
            ),
            "options": " | ".join(c.label or c.id for c in controls),
            "answer_id": row_id,
        })
    return pd.DataFrame(records)


def preview_size(page) -> Optional[Tuple[int, int]]:
    """Pixel size of the raster a parsed page's region boxes are measured in.

    The preview raster, when the page carries one: region boxes were scaled to
    it. Falls back to the size a caller recorded before dropping that raster,
    then to the page's own pixel size.
    """
    import base64
    import io as _io

    if page.preview_size:
        return tuple(page.preview_size)
    if page.image_b64:
        try:
            from PIL import Image

            with Image.open(_io.BytesIO(base64.b64decode(page.image_b64))) as preview:
                return preview.size
        except Exception:
            pass
    if page.width and page.height:
        return page.width, page.height
    return None


def survey_table_regions(document, template) -> set:
    """Table regions that are really questionnaire grids, not data tables.

    A rating scale or a matrix reads as a table to the layout model, and
    exporting it as a CSV of empty cells is noise next to the response tables.
    Region boxes are in preview pixels and template controls in page fractions,
    so both are normalized before they are compared.
    """
    from core import doc_ir

    grids = set()
    for page, template_page in zip(document.pages, template.pages):
        size = preview_size(page)
        if not size:
            continue
        width, height = size

        centres = [
            ((c.bbox[0] + c.bbox[2]) / 2, (c.bbox[1] + c.bbox[3]) / 2)
            for c in template_page.controls
        ]
        for region in page.regions:
            if region.type != doc_ir.TABLE or len(region.bbox) < 4:
                continue
            x1, y1, x2, y2 = (
                region.bbox[0] / width, region.bbox[1] / height,
                region.bbox[2] / width, region.bbox[3] / height,
            )
            inside = sum(1 for cx, cy in centres if x1 <= cx <= x2 and y1 <= cy <= y2)
            if inside >= 2:
                grids.add(region.id)
    return grids


#: Marks the form groups this module attaches, so a rebuild replaces its own
#: work instead of stacking a second copy of every answer on the page.
FORM_GROUP_METHOD = "template-omr"

_FORM_STATE = {"checked": "selected", "unchecked": "unselected"}
_FORM_MARK = {"checked": "filled", "unchecked": "none"}


def to_form_groups(
    result: DocumentReading,
    template: survey_template.SurveyTemplate,
    document,
) -> int:
    """Attach one respondent's answers to their parsed document.

    The batch tables answer "what did everyone say"; ``document.md`` has to
    answer "what does this sheet say", and until now it could not -- the marks
    are ink the layout model reads as an empty grid. The readings are hung on
    the page they came from as :class:`doc_ir.FormGroup`, which every export
    renders in place. Returns the number of groups attached.
    """
    from core import doc_ir

    readings = {r.control_id: r for r in result.readings}
    attached = 0
    for page, template_page in zip(document.pages, template.pages):
        page.form_groups = [
            group for group in page.form_groups
            if group.provenance.get("method") != FORM_GROUP_METHOD
        ]
        size = preview_size(page)
        if not size:
            continue
        width, height = size

        by_question: Dict[str, List[survey_template.TemplateControl]] = {}
        for control in survey_template.reading_order(template_page.controls):
            by_question.setdefault(control.question_id or control.row_id, []).append(control)

        groups, centres, labelled = [], {}, {}
        for question_id, controls in by_question.items():
            by_row: Dict[str, List[survey_template.TemplateControl]] = {}
            for control in controls:
                by_row.setdefault(control.row_id or control.id, []).append(control)

            rows, boxes = [], []
            for row_id, row_controls in by_row.items():
                options, unread = [], False
                for control in row_controls:
                    reading = readings.get(control.id)
                    state = reading.state if reading else UNCERTAIN
                    unread = unread or state not in _FORM_STATE
                    box = control.pixel_bbox(width, height)
                    boxes.append(box)
                    options.append(doc_ir.FormOption(
                        id=control.id,
                        label=control.label or control.id,
                        state=_FORM_STATE.get(state, "ambiguous"),
                        visual_mark=_FORM_MARK.get(state, "uncertain"),
                        bbox=[float(v) for v in box],
                        observations=[doc_ir.Observation(
                            source="textlab-omr",
                            value=state,
                            method=FORM_GROUP_METHOD,
                            score=round(float(reading.score), 3) if reading else None,
                        )],
                    ))
                rows.append(doc_ir.FormRow(
                    id=row_id,
                    label=template.row_labels.get(row_id, ""),
                    options=options,
                    status="needs_review" if unread else "accepted",
                    review_reasons=["mark could not be read"] if unread else [],
                ))

            rules = {template.rules.get(row.id, "single") for row in rows}
            multiple = rules == {"multiple"}
            group = doc_ir.FormGroup(
                id=question_id,
                bbox=[
                    float(min(b[0] for b in boxes)), float(min(b[1] for b in boxes)),
                    float(max(b[2] for b in boxes)), float(max(b[3] for b in boxes)),
                ],
                question_text=question_tag(controls[0]),
                question_type=("matrix" if len(rows) > 1 else
                               ("multiple" if multiple else "single")),
                selection_rule="zero_or_more" if multiple else "zero_or_one",
                rows=rows,
                status=("needs_review" if any(r.status == "needs_review" for r in rows)
                        else "accepted"),
                provenance={
                    "method": FORM_GROUP_METHOD,
                    "template_controls": len(controls),
                    "registration": result.registration.get(template_page.page_index),
                },
            )
            groups.append(group)
            centres[group.id] = [
                ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes
            ]
            labelled[group.id] = all(row.label.strip() for row in rows)

        # The printed grid is what the layout model saw: empty cells whose row
        # stems the answers above already carry. Standing in for it is only safe
        # when every row of every question on it was named, or the document
        # would silently lose printed text.
        for region in page.regions:
            if region.type != doc_ir.TABLE or len(region.bbox) < 4:
                continue
            x1, y1, x2, y2 = region.bbox
            owners = [
                group for group in groups
                if sum(1 for cx, cy in centres[group.id]
                       if x1 <= cx <= x2 and y1 <= cy <= y2) >= 2
            ]
            if owners and all(labelled[group.id] for group in owners):
                owners[0].covered_region_ids.append(region.id)

        page.form_groups.extend(groups)
        attached += len(groups)
    return attached
