"""Template-first survey extraction: registration, consensus blank, mark reads."""

import conftest_path  # noqa: F401

import pathlib
import tempfile

import cv2
import numpy as np
import pytest

from core import survey_batch as sb
from core import survey_label as sl
from core import survey_template as st

W, H = 1800, 1400
RADIUS = 17
CENTRES = [(200 + col * 300, 380 + row * 220) for row in range(4) for col in range(5)]

# Registration keys on printed detail, so the fixture carries a page worth of
# it — a near-empty synthetic page would fail here for reasons a real scan
# never hits.
_BODY = [
    "Wie gut kennst du die Ziele des Naturparks in seiner taeglichen Arbeit?",
    "Profitieren die Bewohnenden deiner Meinung nach vom Label Naturpark",
    "oder ist der Naturpark fuer die Bevoelkerung eher ein Nachteil?",
    "Inwieweit befuerwortest du die nachfolgenden Aufgaben des Naturparks?",
    "Fuer welche Themen soll sich der Naturpark in Zukunft engagieren?",
]


def blank_form():
    """A printed form: body text plus a grid of empty response circles."""
    img = np.full((H, W), 255, np.uint8)
    cv2.putText(img, "Bevoelkerungsbefragung", (120, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3)
    for row, line in enumerate(_BODY):
        cv2.putText(img, line, (120, 150 + row * 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, 0, 1)
        cv2.putText(img, line[::-1], (120, 1200 + row * 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, 0, 1)
    for index, (x, y) in enumerate(CENTRES):
        cv2.circle(img, (x, y), RADIUS, 0, 2)
        cv2.putText(img, f"option {index}", (x + 30, y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 1)
    return img


def filled_form(marked, *, angle=0.0, shift=(0, 0)):
    """One respondent: the form, crosses in *marked*, and a scan-like offset."""
    img = blank_form()
    for index in marked:
        x, y = CENTRES[index]
        d = int(RADIUS * 0.7)
        cv2.line(img, (x - d, y - d), (x + d, y + d), 0, 3)
        cv2.line(img, (x + d, y - d), (x - d, y + d), 0, 3)
    matrix = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    matrix[0, 2] += shift[0]
    matrix[1, 2] += shift[1]
    return cv2.warpAffine(img, matrix, (W, H), borderValue=255)


RESPONDENTS = [
    ([0, 6, 12, 18], 0.0, (0, 0)),
    ([1, 7, 13, 19], 0.8, (6, -4)),
    ([2, 8, 14, 15], -0.6, (-5, 7)),
    ([3, 9, 10, 16], 0.4, (3, 3)),
    ([4, 5, 11, 17], -0.3, (-2, -6)),
]


def registered_stack():
    reference = filled_form(*RESPONDENTS[0][0:1], angle=RESPONDENTS[0][1],
                            shift=RESPONDENTS[0][2])
    stack = [reference]
    for marked, angle, shift in RESPONDENTS[1:]:
        moving = filled_form(marked, angle=angle, shift=shift)
        matrix, quality = st.register(moving, reference)
        assert matrix is not None and quality > 0.3, f"registration failed ({quality})"
        stack.append(st.warp_to_reference(moving, matrix, reference.shape))
    return reference, stack


def test_register_recovers_a_known_offset():
    reference = blank_form()
    moving = filled_form([], angle=0.0, shift=(12, -9))
    matrix, quality = st.register(moving, reference)
    assert matrix is not None
    assert quality > 0.5
    # the homography should undo the shift it was given
    corner = matrix @ np.array([12.0, -9.0, 1.0])
    corner /= corner[2]
    assert abs(corner[0] - 0.0) < 3.0
    assert abs(corner[1] - 0.0) < 3.0


def test_register_reports_failure_instead_of_guessing():
    matrix, quality = st.register(np.full((H, W), 255, np.uint8), blank_form())
    assert matrix is None
    assert quality == 0.0


def test_consensus_blank_cancels_respondent_ink():
    reference, stack = registered_stack()
    blank = st.consensus_blank(stack)
    truth = blank_form()
    # every respondent's crosses are gone: the interiors are paper again
    for x, y in CENTRES:
        interior = blank[y - 6: y + 6, x - 6: x + 6]
        assert interior.min() > 200, f"ink survived the median at {(x, y)}"
    # ...while the printed circles and text survive
    assert np.count_nonzero(truth < 128) > 0
    assert abs(np.count_nonzero(blank < 128) - np.count_nonzero(truth < 128)) < (
        0.25 * np.count_nonzero(truth < 128)
    )


def test_find_controls_locates_every_circle_and_no_text():
    controls = st.find_controls(blank_form())
    assert len(controls) == len(CENTRES)
    found = sorted(
        ((c["bbox"][0] + c["bbox"][2]) // 2, (c["bbox"][1] + c["bbox"][3]) // 2)
        for c in controls
    )
    for (fx, fy), (ex, ey) in zip(found, sorted(CENTRES)):
        assert abs(fx - ex) <= 3 and abs(fy - ey) <= 3
    assert {c["shape"] for c in controls} == {"circle"}


def test_residual_read_separates_marked_from_empty():
    blank = blank_form()
    marked = [2, 9, 14]
    respondent = filled_form(marked)
    residual = st.residual_ink(blank, respondent)
    for index, (x, y) in enumerate(CENTRES):
        crop = residual[y - RADIUS: y + RADIUS, x - RADIUS: x + RADIUS]
        verdict = st.classify_residual(crop)
        expected = "checked" if index in marked else "unchecked"
        assert verdict["state"] == expected, (
            f"control {index}: {verdict} (expected {expected})"
        )


def test_residual_read_survives_registration_slack():
    """A couple of pixels of misalignment must not read as a mark."""
    blank = blank_form()
    respondent = filled_form([7], angle=0.35, shift=(2, -2))
    matrix, _ = st.register(respondent, blank)
    residual = st.residual_ink(blank, st.warp_to_reference(respondent, matrix, blank.shape))
    states = []
    for x, y in CENTRES:
        crop = residual[y - RADIUS: y + RADIUS, x - RADIUS: x + RADIUS]
        states.append(st.classify_residual(crop)["state"])
    assert states.count("checked") == 1
    assert states[7] == "checked"


def test_template_round_trips_through_json():
    import base64

    template = st.SurveyTemplate(dpi=300)
    page = st.TemplatePage(page_index=0, width=W, height=H)
    ok, encoded = cv2.imencode(".png", blank_form())
    assert ok
    page.blank_png_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    for index, control in enumerate(st.find_controls(blank_form()), start=1):
        x1, y1, x2, y2 = control["bbox"]
        page.controls.append(
            st.TemplateControl(
                id=f"p1_c{index:03d}",
                bbox=[x1 / W, y1 / H, x2 / W, y2 / H],
                shape=control["shape"],
                label=f"option {index}",
            )
        )
    template.pages.append(page)

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "template.json"
        template.save(path)
        restored = st.SurveyTemplate.load(path)

    assert restored.control_count == template.control_count
    assert restored.dpi == 300
    # the blank travels with the template: without it no batch can be read
    assert restored.pages[0].blank_png_b64 == page.blank_png_b64
    original = template.pages[0].controls[3]
    copy = restored.pages[0].controls[3]
    assert copy.id == original.id
    assert copy.label == original.label
    # normalized coordinates survive the trip back to pixels
    assert copy.pixel_bbox(W, H) == original.pixel_bbox(W, H)


# ==========================================
#     STRUCTURE INFERENCE AND CSV EXPORT
# ==========================================


def _template_with(controls, rules=None):
    """A one-page template from (x, y, shape, label, question) tuples."""
    template = st.SurveyTemplate(dpi=300)
    page = st.TemplatePage(page_index=0, width=W, height=H)
    for index, (x, y, shape, label, question) in enumerate(controls, start=1):
        page.controls.append(
            st.TemplateControl(
                id=f"p1_c{index:03d}",
                bbox=[(x - RADIUS) / W, (y - RADIUS) / H,
                      (x + RADIUS) / W, (y + RADIUS) / H],
                shape=shape, label=label, question_id=question,
            )
        )
    template.pages.append(page)
    if rules is None:
        st.infer_structure(template)
    else:
        template.rules = rules
    return template


def test_matrix_rows_become_one_single_choice_answer_each():
    grid = [
        (200 + col * 300, 380 + row * 220, "circle", str(col), "q1")
        for row in range(3) for col in range(4)
    ]
    template = _template_with(grid)
    rows = {c.row_id for c in template.pages[0].controls}
    assert len(rows) == 3, f"expected one answer group per matrix row, got {rows}"
    assert set(template.rules.values()) == {"single"}


def test_vertical_checkbox_list_is_one_multi_select_answer():
    column = [(200, 380 + i * 120, "box", f"option {i}", "q2") for i in range(5)]
    template = _template_with(column)
    rows = {c.row_id for c in template.pages[0].controls}
    assert len(rows) == 1
    assert set(template.rules.values()) == {"multiple"}


def test_shape_splits_a_question_holding_both_kinds():
    """A Ja/Nein pair beside a checkbox list must not become one answer."""
    controls = [
        (200, 380, "circle", "Ja", "q3"), (400, 380, "circle", "Nein", "q3"),
        (200, 560, "box", "a", "q3"), (200, 680, "box", "b", "q3"),
    ]
    template = _template_with(controls)
    by_row = {}
    for control in template.pages[0].controls:
        by_row.setdefault(control.row_id, []).append(control.shape)
    assert len(by_row) == 2
    assert {"circle"} in [set(v) for v in by_row.values()]
    assert {"box"} in [set(v) for v in by_row.values()]
    assert sorted(template.rules.values()) == ["multiple", "single"]


def _reading(document, states, template):
    controls = [c for page in template.pages for c in page.controls]
    return sb.DocumentReading(
        document=document,
        readings=[
            sb.ControlReading(control.id, 0, state=state, score=1.0 if state != "uncertain" else 0.1)
            for control, state in zip(controls, states)
        ],
    )


def test_checkbox_export_has_a_certainty_beside_every_data_column():
    template = _template_with(
        [(200 + i * 300, 380, "circle", str(i), "q1") for i in range(4)]
    )
    row_id = template.pages[0].controls[0].row_id
    results = [
        _reading("a.pdf", ["unchecked", "checked", "unchecked", "unchecked"], template),
        _reading("b.pdf", ["checked", "unchecked", "unchecked", "checked"], template),
        _reading("c.pdf", ["unchecked", "unchecked", "uncertain", "unchecked"], template),
    ]
    frame = sb.to_checkbox_table(results, template)

    # "registration" is scan metadata, not an answer, so it has no certainty
    data_columns = [c for c in frame.columns
                    if c not in ("document", "registration")
                    and not c.endswith(sb.CERTAINTY_SUFFIX)]
    for column in data_columns:
        assert column + sb.CERTAINTY_SUFFIX in frame.columns, f"{column} has no certainty"

    # single choice: a value column naming the chosen option
    assert frame.loc[0, row_id] == "1"
    assert frame.loc[0, row_id + sb.CERTAINTY_SUFFIX] == 1.0
    # two marks is a contradiction, reported as such rather than resolved
    assert frame.loc[1, row_id] == "MULTIPLE"
    assert frame.loc[1, row_id + sb.CERTAINTY_SUFFIX] == 0.0
    # an unreadable control makes the whole answer unreadable
    assert frame.loc[2, row_id] == "UNCERTAIN"
    # every checkbox is TRUE/FALSE
    assert bool(frame.loc[0, f"{row_id} | 1"]) is True
    assert bool(frame.loc[0, f"{row_id} | 0"]) is False


def test_checkbox_export_orders_a_scale_left_to_right():
    template = _template_with(
        [(200 + i * 300, 380, "circle", str(i), "q1") for i in range(4)]
    )
    row_id = template.pages[0].controls[0].row_id
    frame = sb.to_checkbox_table(
        [_reading("a.pdf", ["checked"] + ["unchecked"] * 3, template)], template
    )
    checkboxes = [c for c in frame.columns
                  if c.startswith(f"{row_id} | ") and not c.endswith(sb.CERTAINTY_SUFFIX)]
    assert checkboxes == [f"{row_id} | {i}" for i in range(4)]


# ==========================================
#      ROW STEM NAMING AND AUDIT OUTPUT
# ==========================================


def two_column_page():
    """Two content columns separated by a gutter, as in a two-up A3 scan."""
    img = np.full((900, 2000), 255, np.uint8)
    for x0 in (100, 1200):
        for row in range(6):
            cv2.putText(img, "Naturpark Diemtigtal Umfrage", (x0, 120 + row * 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    return img


def test_content_columns_finds_the_gutter():
    ink = md_ink(two_column_page())
    columns = sl._content_columns(ink)
    assert len(columns) == 2, f"expected two content columns, got {columns}"
    left, right = columns
    assert left[1] < 1200 <= right[0]


def md_ink(gray):
    from core import markup_detect

    return markup_detect._ink_mask(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))


def test_stem_span_ignores_a_table_rule():
    """A full-width row rule must not hide the gap before the controls."""
    img = np.full((200, 1400), 255, np.uint8)
    cv2.putText(img, "Nachhaltige Landwirtschaft", (60, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    cv2.line(img, (0, 170), (1399, 170), 0, 2)          # the table rule
    ink = md_ink(img)
    columns = [(0, 1400)]
    span = sl._stem_span(ink, columns, x_limit=1200, y1=40, y2=190)
    assert span is not None, "the rule swallowed the gap"
    x1, x2 = span
    assert 40 <= x1 <= 80, span
    assert x2 < 1200


def test_stem_span_stays_inside_its_content_column():
    ink = md_ink(two_column_page())
    columns = sl._content_columns(ink)
    # controls in the right column must not pick up the left column's text
    span = sl._stem_span(ink, columns, x_limit=1190, y1=80, y2=140)
    assert span is None or span[0] >= columns[1][0]


def test_row_naming_is_skipped_without_tesseract_rather_than_crashing():
    template = _template_with(
        [(200 + i * 300, 380, "circle", str(i), "q1") for i in range(4)]
    )
    template.pages[0].blank_png_b64 = None
    assert sl.name_answer_rows(template) == 0


def test_unused_controls_separates_dead_rows_from_unpopular_options():
    template = _template_with(
        [(200 + i * 300, 380, "circle", str(i), "q1") for i in range(3)]
        + [(200 + i * 300, 700, "circle", str(i), "q2") for i in range(3)]
    )
    controls = [c for page in template.pages for c in page.controls]
    # first row: option 0 chosen every time; second row: nobody marked anything
    results = [
        _reading("a.pdf", ["checked", "unchecked", "unchecked",
                           "unchecked", "unchecked", "unchecked"], template),
    ]
    unused = sb.unused_controls(results, template)
    dead = unused[unused["whole_row_unused"]]
    assert set(dead["row_id"]) == {controls[3].row_id}
    assert len(dead) == 3
    # the unpopular options in the answered row are listed but not flagged
    assert set(unused[~unused["whole_row_unused"]]["control_id"]) == {
        controls[1].id, controls[2].id
    }


def test_export_carries_registration_quality():
    template = _template_with(
        [(200 + i * 300, 380, "circle", str(i), "q1") for i in range(3)]
    )
    result = _reading("a.pdf", ["checked", "unchecked", "unchecked"], template)
    result.registration = {0: 0.91, 1: 0.44}
    frame = sb.to_checkbox_table([result], template)
    # the worst page is what a reviewer needs to see
    assert frame.loc[0, "registration"] == 0.44


# ==========================================
#      GROUND-TRUTH SHEET AND SCORING
# ==========================================


def test_reading_order_keeps_a_row_left_to_right_despite_jitter():
    """Scanned controls on one line differ by a few pixels vertically."""
    template = _template_with([
        (200, 380, "circle", "0", "q1"), (500, 383, "circle", "1", "q1"),
        (800, 377, "circle", "2", "q1"), (1100, 381, "circle", "3", "q1"),
    ])
    ordered = st.reading_order(template.pages[0].controls)
    assert [c.label for c in ordered] == ["0", "1", "2", "3"]


def test_reading_order_keeps_a_vertical_list_top_to_bottom():
    template = _template_with(
        [(200, 380 + i * 120, "circle", str(i), "q1") for i in range(4)]
    )
    ordered = st.reading_order(template.pages[0].controls)
    assert [c.label for c in ordered] == ["0", "1", "2", "3"]


def test_answer_sheet_is_one_blank_line_per_answer():
    template = _template_with(
        [(200 + i * 300, 380, "circle", str(i), "q1") for i in range(4)]
        + [(200, 700 + i * 120, "box", f"box {i}", "q2") for i in range(3)]
    )
    sheet = sb.answer_sheet(template, ["a.pdf", "b.pdf"])
    rows = {c.row_id for c in template.pages[0].controls}
    assert len(sheet) == 2 * len(rows)
    assert (sheet["answer"] == "").all(), "the sheet must not be pre-filled"
    single = sheet[sheet["type"] == "single"].iloc[0]
    assert single["options"] == "0 | 1 | 2 | 3"
    assert set(sheet["document"]) == {"a.pdf", "b.pdf"}


def test_scoring_finds_a_wrong_answer_and_excludes_the_unsure_ones():
    template = _template_with(
        [(200 + i * 300, 380, "circle", str(i), "q1") for i in range(4)]
    )
    row_id = template.pages[0].controls[0].row_id
    results = [
        _reading("a.pdf", ["unchecked", "checked", "unchecked", "unchecked"], template),
        _reading("b.pdf", ["checked", "unchecked", "unchecked", "unchecked"], template),
        _reading("c.pdf", ["checked", "unchecked", "unchecked", "checked"], template),
    ]
    sheet = sb.answer_sheet(template, ["a.pdf", "b.pdf", "c.pdf"])
    truth = {"a.pdf": "1", "b.pdf": "2", "c.pdf": sb.AMBIGUOUS_MARK}
    sheet["answer"] = [truth[d] for d in sheet["document"]]

    per_answer, summary = sb.score_sheet(sheet, results, template)
    assert summary["answers_scored"] == 2      # the "?" row is not scorable
    assert summary["human_unsure"] == 1
    assert summary["auto_accepted"] == 2
    assert summary["silent_errors"] == 1       # b.pdf: read 0, truth 2
    assert summary["auto_accepted_accuracy"] == 0.5

    wrong = per_answer[(~per_answer["correct"]) & (~per_answer["human_unsure"])]
    assert list(wrong["document"]) == ["b.pdf"]
    assert wrong.iloc[0]["predicted"] == "0"
    assert wrong.iloc[0]["truth"] == "2"
    assert row_id in set(per_answer["answer_id"])


def test_scoring_counts_a_flagged_answer_separately_from_an_error():
    """A MULTIPLE the pipeline refused to resolve is not a silent error."""
    template = _template_with(
        [(200 + i * 300, 380, "circle", str(i), "q1") for i in range(3)]
    )
    results = [_reading("a.pdf", ["checked", "checked", "unchecked"], template)]
    sheet = sb.answer_sheet(template, ["a.pdf"])
    sheet["answer"] = "0;1"

    _per_answer, summary = sb.score_sheet(sheet, results, template)
    assert summary["flagged_for_review"] == 1
    assert summary["auto_accepted"] == 0
    assert summary["silent_errors"] == 0


def test_answer_sheet_locates_each_answer_on_the_printed_page():
    """A two-up scan makes the PDF page useless; the sheet must say more."""
    template = _template_with(
        [(200 + i * 300, 380, "circle", str(i), "p1_q4") for i in range(3)]
        + [(200 + i * 300, 700, "circle", str(i), "p1_q7") for i in range(3)]
        + [(200 + i * 300, 1000, "circle", str(i), "p1_q7") for i in range(3)]
    )
    for control in template.pages[0].controls:
        control.sheet_page = "2/4"
    sheet = sb.answer_sheet(template, ["a.pdf"])

    assert set(sheet["sheet_page"]) == {"2/4"}
    assert set(sheet["question"]) == {"Q4", "Q7"}
    # the single-answer question needs no row hint; the two-answer one does
    q4 = sheet[sheet["question"] == "Q4"]
    q7 = sheet[sheet["question"] == "Q7"]
    assert list(q4["row"]) == [""]
    assert sorted(q7["row"]) == [
        "row 1 of 2 (top to bottom)", "row 2 of 2 (top to bottom)"
    ]


def test_answer_sheet_is_ordered_down_the_printed_page():
    template = _template_with(
        [(2000, 300, "circle", "a", "p1_q9")]      # right column, top
        + [(200, 900, "circle", "b", "p1_q2")]     # left column, lower
        + [(200, 300, "circle", "c", "p1_q1")]     # left column, top
    )
    controls = {c.label: c for c in template.pages[0].controls}
    controls["a"].column = 1
    for name in ("b", "c"):
        controls[name].column = 0
    sheet = sb.answer_sheet(template, ["a.pdf"])
    assert list(sheet["question"]) == ["Q1", "Q2", "Q9"]


def test_sanitize_keeps_a_leading_letter_o():
    assert sl.sanitize("Oey") == "Oey"
    assert sl.sanitize("○ Ja") == "Ja"
    assert sl.sanitize("O Nein") == "Nein"


def test_a_stroke_clipping_the_control_is_doubted_not_called_empty():
    """The failure mode found by hand-labelling: a mark that misses the box.

    Respondents sometimes strike across a circle's edge rather than through
    it, leaving the interior clean. Reading that as a confident "unchecked" is
    a silent error, so the halo around the control has to raise doubt.
    """
    blank = blank_form()
    x, y = CENTRES[6]
    marked = blank.copy()
    # An X the respondent centred beside the control instead of on it, so it
    # clips the edge and lands mostly outside. Sized to reproduce the real
    # case found by hand-labelling: interior fill 0.0, halo ratio ~0.07.
    cx, d = x + RADIUS + 8, int(RADIUS * 1.8)
    cv2.line(marked, (cx - d, y - d), (cx + d, y + d), 0, 5)
    cv2.line(marked, (cx + d, y - d), (cx - d, y + d), 0, 5)

    residual = st.residual_ink(blank, marked)
    box = (x - RADIUS, y - RADIUS, x + RADIUS, y + RADIUS)
    crop = residual[box[1]:box[3], box[0]:box[2]]

    assert st.classify_residual(crop)["state"] == "unchecked", (
        "the interior really is empty; the halo is what carries the signal"
    )
    verdict = st.classify_residual(crop, st.halo_crop(residual, box))
    assert verdict["state"] == "uncertain"
    assert verdict["halo_ratio"] >= st.HALO_INK


def test_an_untouched_control_stays_confidently_empty():
    blank = blank_form()
    residual = st.residual_ink(blank, filled_form([6]))
    for index, (x, y) in enumerate(CENTRES):
        if index == 6:
            continue
        box = (x - RADIUS, y - RADIUS, x + RADIUS, y + RADIUS)
        verdict = st.classify_residual(
            residual[box[1]:box[3], box[0]:box[2]], st.halo_crop(residual, box)
        )
        assert verdict["state"] == "unchecked", f"control {index}: {verdict}"


def test_block_text_keeps_its_line_breaks():
    """Paddle returns an option and a nearby caption as two lines of one block.

    Collapsing them into one string merged "Ja" with the caption printed above
    and right of it, and gave the per-line pick nothing to work with.
    """
    assert sl._plain_lines("Ja\nFalls ja, E-Mail oder Telefonnummer") == (
        "Ja\nFalls ja, E-Mail oder Telefonnummer"
    )
    assert sl._plain_lines("  Ja \n\n <b>Nein</b>  ") == "Ja\nNein"
    assert sl._plain_lines("one   two") == "one two"


def test_one_stroke_across_two_controls_belongs_to_the_one_holding_it():
    """A long X tail reaching into the neighbour is not a second answer."""
    blank = blank_form()
    x0, y = CENTRES[0]
    x1 = CENTRES[1][0]
    marked = blank.copy()
    # X centred on the first control, tail sweeping up-right into the second
    d = RADIUS
    cv2.line(marked, (x0 - d, y - d), (x0 + d, y + d), 0, 5)
    cv2.line(marked, (x0 - d, y + d), (x1 + d, y - d - 10), 0, 5)

    residual = st.residual_ink(blank, marked)
    boxes = [
        (cx - RADIUS, y - RADIUS, cx + RADIUS, y + RADIUS) for cx in (x0, x1)
    ]
    assert st.dominant_control(residual, boxes) == 0


def test_two_separate_marks_are_not_resolved_away():
    """A real double answer must stay flagged, not be silently reduced to one."""
    blank = blank_form()
    marked = blank.copy()
    for index in (0, 1):
        cx, cy = CENTRES[index]
        d = int(RADIUS * 0.7)
        cv2.line(marked, (cx - d, cy - d), (cx + d, cy + d), 0, 4)
        cv2.line(marked, (cx + d, cy - d), (cx - d, cy + d), 0, 4)

    residual = st.residual_ink(blank, marked)
    boxes = [
        (CENTRES[i][0] - RADIUS, CENTRES[i][1] - RADIUS,
         CENTRES[i][0] + RADIUS, CENTRES[i][1] + RADIUS)
        for i in (0, 1)
    ]
    assert st.dominant_control(residual, boxes) is None


# ==========================================
#          BATCH ORCHESTRATION
# ==========================================


# The fixture PNGs carry no DPI metadata, so PyMuPDF gives the PDF a 96 DPI
# page box; rendering it back at that DPI reproduces the original pixels.
FIXTURE_DPI = 96


def _write_batch(folder, count=6):
    """A folder of one-page 'scans', each a different respondent."""
    import fitz

    paths = []
    for index in range(count):
        marked = [index % len(CENTRES), (index * 7 + 3) % len(CENTRES)]
        image = filled_form(marked, angle=0.3 * (index % 3 - 1), shift=(index, -index))
        png = folder / f"respondent_{index:02d}.png"
        cv2.imwrite(str(png), image)
        pdf = folder / f"respondent_{index:02d}.pdf"
        with fitz.open(str(png)) as doc:
            pdf.write_bytes(doc.convert_to_pdf())
        png.unlink()
        paths.append(pdf)
    return paths


def test_batch_orchestration_produces_the_survey_tables():
    """The path the TextLab batch page runs: learn, read, aggregate."""
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmp:
        folder = pathlib.Path(tmp)
        paths = _write_batch(folder)

        # label=False keeps this off the PaddleOCR-VL backend; the geometry,
        # grouping and export are what this exercises.
        template, _blanks = sb.prepare_template(paths, label=False, dpi=FIXTURE_DPI)
        assert template.control_count == len(CENTRES)
        assert template.rules, "structure inference produced no answer groups"

        results = sb.read_batch(paths, template)
        assert len(results) == len(paths)

        out = folder / "survey"
        summary = sb.write_batch_outputs(results, template, out)
        assert summary["documents"] == len(paths)
        assert summary["controls"] == len(CENTRES)

        for name in (
            "responses_checkboxes.csv", "responses_matrix.csv",
            "responses_long.csv", "review_queue.csv", "unused_controls.csv",
            "answers_overview.csv", "survey_template.json",
            "template_page1.png",
        ):
            assert (out / name).exists(), f"{name} was not written"

        table = pd.read_csv(out / "responses_checkboxes.csv")
        assert len(table) == len(paths), "one row per respondent"
        assert "registration" in table.columns
        # each respondent's two marks come back
        counts = [
            sum(1 for value in row if str(value) == "True")
            for row in table.drop(columns=["document"]).values
        ]
        assert all(c == 2 for c in counts), f"marks per respondent: {counts}"


def test_single_document_answers_are_one_row_per_question():
    """The batch table is wide; a single file's own answers read better long."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = pathlib.Path(tmp)
        paths = _write_batch(folder, count=5)
        template, _ = sb.prepare_template(paths, label=False, dpi=FIXTURE_DPI)
        reading = sb.read_document(paths[0], template)
        frame = sb.answers_for_document(reading, template)

        assert len(frame) == len(template.rules), "one line per answer"
        assert set(frame["document"]) == {paths[0].name}
        assert {"question", "answer", "certainty", "options", "answer_id"} <= set(
            frame.columns
        )
        # the respondent's marks show up as answers
        assert (frame["answer"].astype(str).str.strip() != "").any()


def test_the_blank_is_built_from_a_bounded_sample():
    """A large batch must not hold every page at 300 DPI at once."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = pathlib.Path(tmp)
        paths = _write_batch(folder, count=8)
        blank = st.build_blank(paths, 0, max_documents=3)
        assert len(blank.contributors) <= 3


def test_documents_with_a_different_page_count_are_left_out():
    import fitz

    with tempfile.TemporaryDirectory() as tmp:
        folder = pathlib.Path(tmp)
        paths = _write_batch(folder, count=5)
        odd = folder / "two_pager.pdf"
        with fitz.open(str(paths[0])) as source, fitz.open() as doubled:
            doubled.insert_pdf(source)
            doubled.insert_pdf(source)
            doubled.save(str(odd))

        template, _ = sb.prepare_template(paths + [odd], label=False, dpi=FIXTURE_DPI)
        assert len(template.pages) == 1, "the majority page count wins"
        assert odd.name in template.provenance["skipped_wrong_page_count"]


def test_a_small_batch_says_the_blank_may_be_unreliable():
    """Below a handful of copies the median stops cancelling popular marks."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = pathlib.Path(tmp)
        few = _write_batch(folder, count=3)
        template, _ = sb.prepare_template(few, label=False, dpi=FIXTURE_DPI)
        assert "small_batch_warning" in template.provenance

    with tempfile.TemporaryDirectory() as tmp:
        folder = pathlib.Path(tmp)
        many = _write_batch(folder, count=st.MIN_BLANK_DOCUMENTS + 1)
        template, _ = sb.prepare_template(many, label=False, dpi=FIXTURE_DPI)
        assert "small_batch_warning" not in template.provenance


def test_dropping_a_control_regroups_without_rereading():
    """The batch page's refine step: fix the form, keep the readings."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = pathlib.Path(tmp)
        paths = _write_batch(folder, count=6)
        template, _ = sb.prepare_template(paths, label=False, dpi=FIXTURE_DPI)
        results = sb.read_batch(paths, template)

        overview = sb.answer_overview(results, template)
        assert len(overview) == len(template.rules)
        assert {"answer", "answered_by", "never_marked", "control_ids"} <= set(
            overview.columns
        )

        victim = template.pages[0].controls[0].id
        before = template.control_count
        removed = sb.drop_controls(template, [victim])
        assert removed == 1
        assert template.control_count == before - 1
        assert victim not in {c.id for p in template.pages for c in p.controls}

        # the export rebuilds from the readings already taken
        table = sb.to_checkbox_table(results, template)
        assert len(table) == len(paths)
        assert all(victim not in str(c) for c in table.columns)


def test_answer_overview_counts_who_answered_each_question():
    with tempfile.TemporaryDirectory() as tmp:
        folder = pathlib.Path(tmp)
        paths = _write_batch(folder, count=6)
        template, _ = sb.prepare_template(paths, label=False, dpi=FIXTURE_DPI)
        results = sb.read_batch(paths, template)
        overview = sb.answer_overview(results, template)
        assert overview["answered_by"].max() >= 1
        assert overview["answered_by"].max() <= len(paths)
        # never_marked and answered_by must agree
        for _, row in overview.iterrows():
            assert row["never_marked"] == (row["answered_by"] == 0)


def test_dropping_a_row_does_not_leave_a_stale_name_on_another_row():
    """Row ids are positional, so removing one renumbers the rest.

    A leftover row_labels entry would then attach one row's printed name to a
    different row -- a wrong label on real data, not a missing one.
    """
    template = _template_with(
        [(200 + i * 300, 380 + r * 220, "circle", str(i), "p1_q9")
         for r in range(3) for i in range(4)]
    )
    rows = sorted(template.rows())
    assert len(rows) == 3
    template.row_labels = {
        rows[0]: "first row", rows[1]: "second row", rows[2]: "third row",
    }
    # no blanks on this synthetic template, so naming cannot re-derive anything
    victim = [c.id for c in template.rows()[rows[0]]]
    sb.drop_controls(template, victim)

    assert len(template.rows()) == 2
    for row_id in template.rows():
        assert template.row_labels.get(row_id, "") == "", (
            f"{row_id} kept a name from the old numbering: "
            f"{template.row_labels.get(row_id)!r}"
        )


def test_template_overlay_tags_each_answer_with_its_export_name():
    """The overlay is the key between a CSV column and a spot on the paper."""
    import base64

    template = _template_with(
        [(200 + i * 300, 380, "circle", str(i), "p1_q4") for i in range(3)]
    )
    page = template.pages[0]
    ok, encoded = cv2.imencode(".png", blank_form())
    assert ok
    page.blank_png_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")

    plain = sb.template_overlays(template, tag_answers=False)
    tagged = sb.template_overlays(template, tag_answers=True)
    assert set(plain) == set(tagged) == {"template_page1.png"}
    assert plain["template_page1.png"] != tagged["template_page1.png"], (
        "tagging changed nothing"
    )
    # the tag is drawn, so the tagged image differs near the first control
    import numpy as np

    a = cv2.imdecode(np.frombuffer(plain["template_page1.png"], np.uint8), cv2.IMREAD_COLOR)
    b = cv2.imdecode(np.frombuffer(tagged["template_page1.png"], np.uint8), cv2.IMREAD_COLOR)
    x, y = CENTRES[0]
    band_a = a[max(0, y - 60):y, max(0, x - 40):x + 400]
    band_b = b[max(0, y - 60):y, max(0, x - 40):x + 400]
    assert not np.array_equal(band_a, band_b), "no tag above the first control"


def test_rebuilding_keeps_a_file_s_own_answers_one_row_per_question():
    """The refine step must not quietly restore the wide batch shape.

    The batch page wrote each file's answers long, then rebuilding after a
    control was removed re-wrote them from the wide table -- 468 columns on one
    line again.
    """
    import io
    import zipfile

    import pandas as pd

    streamlit = pytest.importorskip("streamlit")

    class _State(dict):
        __getattr__ = dict.get

        def __setattr__(self, key, value):
            self[key] = value

    streamlit.session_state = _State(ocr_running=False)
    ocr = pytest.importorskip("pages.OCR")

    with tempfile.TemporaryDirectory() as tmp:
        folder = pathlib.Path(tmp)
        paths = _write_batch(folder, count=6)
        template, _ = sb.prepare_template(paths, label=False, dpi=FIXTURE_DPI)
        results = sb.read_batch(paths, template)

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for result in results:
                stem = pathlib.Path(result.document).stem
                archive.writestr(
                    f"{stem}/survey_answers.csv",
                    sb.answers_for_document(result, template).to_csv(index=False),
                )
        original = buffer.getvalue()

        row_id = template.pages[0].controls[0].row_id
        before = len(template.rules)
        dropped = [c.id for c in template.rows()[row_id]]
        sb.drop_controls(template, dropped)
        assert len(template.rules) == before - 1
        rebuilt, _summary = ocr._rebuild_survey_zip(original, template, results)

        stem = pathlib.Path(results[0].document).stem
        frame = pd.read_csv(
            io.BytesIO(zipfile.ZipFile(io.BytesIO(rebuilt)).read(
                f"{stem}/survey_answers.csv"
            ))
        )
        assert len(frame) == len(template.rules), "one line per answer, not one wide row"
        assert len(frame) == before - 1, "the removed answer is still there"
        assert "answer" in frame.columns and "certainty" in frame.columns
        # ids are positional and get recycled after a drop, so the controls are
        # what must be gone, not the id
        remaining = {c.id for page in template.pages for c in page.controls}
        assert not (set(dropped) & remaining)
