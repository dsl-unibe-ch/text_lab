"""Read a batch of questionnaires against a known template.

Structure comes from the template, so nothing here infers what the form
contains: every document answers the same fixed list of controls, which is
what makes one row per respondent well defined. Mark state is read by
``markup_detect.detect_markup_geometric`` on the registered crop.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

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
            verdict = survey_template.classify_residual(residual[y1:y2, x1:x2])
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


def to_wide(results: Sequence[DocumentReading],
            template: Optional[survey_template.SurveyTemplate] = None):
    """One row per respondent, one column per control."""
    import pandas as pd

    names = _column_names(template)
    records = []
    for result in results:
        row: Dict[str, Any] = {"document": result.document}
        row.update({names.get(r.control_id, r.control_id): r.state for r in result.readings})
        records.append(row)
    return pd.DataFrame(records)


def to_long(results: Sequence[DocumentReading], template: survey_template.SurveyTemplate):
    """One row per respondent-control, carrying the geometric evidence."""
    import pandas as pd

    labels = {
        control.id: (control.question_id, control.label)
        for page in template.pages
        for control in page.controls
    }
    records = []
    for result in results:
        for reading in result.readings:
            question_id, label = labels.get(reading.control_id, ("", ""))
            records.append({
                "document": result.document,
                "control_id": reading.control_id,
                "question_id": question_id,
                "label": label,
                "page": reading.page_index + 1,
                "state": reading.state,
                "score": reading.score,
                "fill_ratio": reading.fill_ratio,
                "strike": reading.strike,
            })
    return pd.DataFrame(records)


CERTAINTY_SUFFIX = " [certainty]"


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
        # Left-to-right, then top-to-bottom, so a scale reads in printed order.
        controls = sorted(controls, key=lambda c: (round(c.bbox[1], 3), c.bbox[0]))
        columns = [
            (control, _unique(f"{row_id} | {control.label or control.id}", used))
            for control in controls
        ]
        plan.append((row_id, template.rules.get(row_id, "single"), columns))
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
        record: Dict[str, Any] = {"document": result.document}

        for row_id, rule, columns in plan:
            states = []
            for control, name in columns:
                reading = readings.get(control.id)
                state = reading.state if reading else UNCERTAIN
                states.append((control, state, reading.score if reading else 0.0))
                record[name] = {"checked": True, "unchecked": False}.get(state, "")
                record[name + CERTAINTY_SUFFIX] = round(reading.score if reading else 0.0, 3)

            if rule != "single":
                continue

            chosen = [(c, score) for c, state, score in states if state == "checked"]
            unread = [s for _, state, s in states if state == UNCERTAIN]
            if len(chosen) == 1 and not unread:
                value, certainty = chosen[0][0].label or chosen[0][0].id, min(
                    score for _, _, score in states
                )
            elif not chosen and not unread:
                value, certainty = "", min(score for _, _, score in states)
            else:
                # Contradictory or unreadable: say so rather than pick one.
                value = "MULTIPLE" if len(chosen) > 1 else "UNCERTAIN"
                certainty = 0.0
            record[row_id] = value
            record[row_id + CERTAINTY_SUFFIX] = round(float(certainty), 3)

        records.append(record)

    frame = pd.DataFrame(records)
    # value column first, then its checkboxes, each followed by its certainty
    order = ["document"]
    for row_id, rule, columns in plan:
        if rule == "single" and row_id in frame.columns:
            order += [row_id, row_id + CERTAINTY_SUFFIX]
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


def summarize(results: Sequence[DocumentReading]) -> Dict[str, Any]:
    total = sum(len(r.readings) for r in results)
    checked = sum(r.checked for r in results)
    uncertain = sum(r.uncertain for r in results)
    return {
        "documents": len(results),
        "controls_per_document": total // max(1, len(results)),
        "checked": checked,
        "uncertain": uncertain,
        "uncertain_rate": round(uncertain / max(1, total), 4),
        "documents_with_warnings": sum(1 for r in results if r.warnings),
    }
