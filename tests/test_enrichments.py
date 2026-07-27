"""Deterministic tests for opt-in figure and form enrichment."""

import conftest_path  # noqa: F401

import base64
import io
import json
import pathlib
import tempfile
import zipfile

import cv2
import numpy as np

from core import doc_ir, form_extract, vision_enrich


def _png_bytes(width=640, height=420):
    image = np.full((height, width, 3), 255, np.uint8)
    cv2.putText(
        image,
        "1. Preferred option",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
    )
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return image, encoded.tobytes()


class FakeVisionClient:
    provider = "fake-local"
    model = "fake-vl"

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def analyze(self, image_bytes, prompt, schema):
        self.calls.append((image_bytes, prompt, schema))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def test_figure_description_is_separate_from_ocr():
    _, png = _png_bytes()
    original_caption = "Printed caption"
    region = doc_ir.Region(
        id="p1_r0",
        type=doc_ir.FIGURE,
        bbox=[0, 0, 640, 420],
        reading_order=0,
        content={"text": original_caption},
        asset={"b64": base64.b64encode(png).decode("ascii"), "ext": "png"},
    )
    page = doc_ir.Page(page_number=1, regions=[region])
    client = FakeVisionClient(
        [{"description": "A simple document figure.", "visible_text": "Figure 1"}]
    )

    vision_enrich.describe_page_figures(page, client)

    assert region.content["text"] == original_caption
    assert region.visual_description.description == "A simple document figure."
    assert region.visual_description.visible_text == "Figure 1"
    assert region.visual_description.source == "fake-local"
    assert len(client.calls) == 1


def _survey_page():
    title = doc_ir.Region(
        id="p1_r0",
        type=doc_ir.TEXT,
        bbox=[20, 20, 600, 90],
        reading_order=0,
        content={"text": "1. Which option do you prefer?"},
    )
    table = doc_ir.Region(
        id="p1_r1",
        type=doc_ir.TABLE,
        bbox=[20, 100, 600, 260],
        reading_order=1,
        content={
            "html": (
                "<table><tr><th></th><th>Alpha</th><th>Beta</th></tr>"
                "<tr><td>Answer</td><td>○</td><td>○</td></tr></table>"
            )
        },
    )
    return doc_ir.Page(
        page_number=1,
        width=640,
        height=420,
        regions=[title, table],
        source="paddleocr-vl-1.6",
    )


def _universal_result(*, marks=None, selection_rule="zero_or_one"):
    return {
        "questions": [
            {
                "question_text": "Which option do you prefer?",
                "response_type": "single",
                "selection_rule": selection_rule,
                "parent_question_index": 0,
                "condition_text": "",
                "choices": [
                    {"choice_text": "Alpha"},
                    {"choice_text": "Beta"},
                ],
                "rows": [
                    {
                        "row_text": "",
                        "extra_choices": [],
                        "marked_answers": list(marks or []),
                    }
                ],
            }
        ],
        "unmapped_marks": [],
    }


def test_schema_free_is_default_and_assigns_ids_after_echo_validation():
    image, _ = _png_bytes()
    page = _survey_page()
    client = FakeVisionClient(
        [
            _universal_result(
                marks=[
                    {
                        "choice_position": 2,
                        "choice_text": "Beta",
                        "state": "selected",
                        "visual_mark": "x",
                        "associated_text": "",
                    }
                ]
            )
        ]
    )

    groups = form_extract.extract_page_forms(page, image, client)

    assert len(groups) == 1
    group = groups[0]
    assert group.id == "p1_s1_q1"
    assert group.selection_rule == "zero_or_one"
    assert group.provenance["contract_version"] == "schema-free-v2"
    assert group.rows[0].options[1].id == "p1_s1_q1_r1_c2"
    assert group.rows[0].options[1].state == "selected"
    assert {o.source for o in group.rows[0].options[1].observations} == {"fake-local"}
    assert "questions" in client.calls[0][2]["properties"]
    assert "Paddle schema hints" not in client.calls[0][1]
    assert "choice_text" in json.dumps(client.calls[0][2])
    assert any("release benchmark" in warning for warning in group.warnings)


def test_schema_free_echo_and_model_rule_disagreement_force_review():
    image, _ = _png_bytes()
    page = _survey_page()
    result = _universal_result(
        selection_rule="zero_or_more",
        marks=[
            {
                "choice_position": 2,
                "choice_text": "Alpha",
                "state": "selected",
                "visual_mark": "tick",
                "associated_text": "",
            }
        ],
    )
    # Even a model-added permissive cue cannot weaken the independently
    # derived rule; the Paddle-visible section has no multiple-answer cue.
    result["questions"][0]["question_text"] = "Select all options"

    group = form_extract.extract_page_forms(page, image, FakeVisionClient([result]))[0]

    assert group.selection_rule == "zero_or_one"
    assert group.status == group.rows[0].status == "needs_review"
    assert "disagrees" in " ".join(group.warnings)
    assert "echo does not match" in " ".join(group.rows[0].options[1].warnings)


def test_schema_free_zero_marks_is_never_silently_accepted():
    image, _ = _png_bytes()
    group = form_extract.extract_page_forms(
        _survey_page(), image, FakeVisionClient([_universal_result()])
    )[0]

    assert group.status == "needs_review"
    assert "zero marked answers" in " ".join(group.warnings)


def test_schema_free_preserves_conditional_parent_and_row_extra_choice():
    image, _ = _png_bytes()
    page = _survey_page()
    result = _universal_result(
        marks=[
            {
                "choice_position": 1,
                "choice_text": "Alpha",
                "state": "selected",
                "visual_mark": "x",
                "associated_text": "",
            }
        ]
    )
    result["questions"].append(
        {
            "question_text": "If Alpha, why?",
            "response_type": "single",
            "selection_rule": "zero_or_one",
            "parent_question_index": 1,
            "condition_text": "If Alpha",
            "choices": [{"choice_text": "Reason A"}, {"choice_text": "Reason B"}],
            "rows": [
                {
                    "row_text": "",
                    "extra_choices": [{"choice_text": "Not applicable"}],
                    "marked_answers": [
                        {
                            "choice_position": 3,
                            "choice_text": "Not applicable",
                            "state": "selected",
                            "visual_mark": "filled",
                            "associated_text": "handwritten note",
                        }
                    ],
                }
            ],
        }
    )

    groups = form_extract.extract_page_forms(page, image, FakeVisionClient([result]))

    assert len(groups) == 2
    assert groups[1].parent_question_id == groups[0].id
    assert groups[1].condition_text == "If Alpha"
    extra = groups[1].rows[0].options[2]
    assert extra.label == "Not applicable" and extra.associated_text == "handwritten note"


def test_multiple_answer_cue_is_scoped_to_the_current_subquestion():
    paddle_section = (
        "2) Did this happen? Ja Nein. If yes, why? "
        "Mehrere Antworten möglich"
    )

    primary = form_extract._derived_selection_rule(
        paddle_section, "Did this happen?", "", [{"choice_count": 2}]
    )
    follow_up = form_extract._derived_selection_rule(
        paddle_section,
        "If yes, why?",
        "If yes",
        [{"choice_count": 4}],
        "multiple",
    )

    assert primary == "zero_or_one"
    assert follow_up == "zero_or_more"


def test_non_matrix_multirow_shape_is_flagged():
    image, _ = _png_bytes()
    result = _universal_result(
        marks=[
            {
                "choice_position": 1,
                "choice_text": "Alpha",
                "state": "selected",
                "visual_mark": "x",
                "associated_text": "",
            }
        ]
    )
    result["questions"][0]["rows"].append(
        {"row_text": "Beta", "extra_choices": [], "marked_answers": []}
    )

    group = form_extract.extract_page_forms(
        _survey_page(), image, FakeVisionClient([result])
    )[0]

    assert "must contain exactly one row" in " ".join(group.warnings)
    assert all(row.status == "needs_review" for row in group.rows)


def test_matrix_empty_row_and_uncertain_visual_mark_are_flagged():
    image, _ = _png_bytes()
    result = _universal_result()
    question = result["questions"][0]
    question["response_type"] = "matrix"
    question["selection_rule"] = "one_per_row"
    question["rows"] = [
        {
            "row_text": "First statement",
            "extra_choices": [],
            "marked_answers": [
                {
                    "choice_position": 1,
                    "choice_text": "Alpha",
                    "state": "selected",
                    "visual_mark": "uncertain",
                    "associated_text": "",
                }
            ],
        },
        {
            "row_text": "Second statement",
            "extra_choices": [],
            "marked_answers": [],
        },
    ]

    group = form_extract.extract_page_forms(
        _survey_page(), image, FakeVisionClient([result])
    )[0]

    assert group.selection_rule == "one_per_row"
    assert "visual mark type is uncertain" in " ".join(
        group.rows[0].options[0].warnings
    )
    assert group.rows[0].status == "needs_review"
    assert "no visible response mark" in " ".join(group.rows[1].warnings)
    assert group.rows[1].status == "needs_review"


def test_differently_numbered_question_is_excluded_as_boundary_spill():
    image, _ = _png_bytes()
    result = _universal_result(
        marks=[
            {
                "choice_position": 2,
                "choice_text": "Beta",
                "state": "selected",
                "visual_mark": "x",
                "associated_text": "",
            }
        ]
    )
    adjacent = json.loads(json.dumps(result["questions"][0]))
    adjacent["question_text"] = "2) Adjacent question?"
    result["questions"].append(adjacent)

    groups = form_extract.extract_page_forms(
        _survey_page(), image, FakeVisionClient([result])
    )

    assert len(groups) == 1
    assert groups[0].provenance["excluded_boundary_question_numbers"] == ["2"]
    assert "crop-boundary spill" in " ".join(groups[0].warnings)


def test_simple_question_choice_truncation_is_checked_against_geometry():
    result = _universal_result(
        marks=[
            {
                "choice_position": 1,
                "choice_text": "Alpha",
                "state": "selected",
                "visual_mark": "x",
                "associated_text": "",
            }
        ]
    )
    result["questions"][0]["choices"] = [{"choice_text": "Alpha"}]

    group = form_extract._universal_to_form_groups(
        result,
        section_id="p1_s1",
        bbox=[0, 0, 100, 100],
        crop_b64="",
        client=FakeVisionClient([]),
        template_reused=False,
        candidate_evidence=[],
        paddle_mark_present=False,
        printed_validation_text="1) Question Alpha Beta Gamma",
        expected_question_number="1",
        geometric_control_count=3,
    )[0]

    assert group.status == "needs_review"
    assert "fewer choices" in " ".join(group.warnings)


def test_choice_label_format_leakage_and_duplicates_are_rejected():
    result = _universal_result(
        marks=[
            {
                "choice_position": 1,
                "choice_text": "Alpha ○",
                "state": "selected",
                "visual_mark": "x",
                "associated_text": "",
            }
        ]
    )
    result["questions"][0]["choices"] = [
        {"choice_text": "Alpha ○"},
        {"choice_text": "Alpha ○"},
    ]

    group = form_extract._universal_to_form_groups(
        result,
        section_id="p1_s1",
        bbox=[0, 0, 100, 100],
        crop_b64="",
        client=FakeVisionClient([]),
        template_reused=False,
        candidate_evidence=[],
        paddle_mark_present=False,
        printed_validation_text="1) Question",
    )[0]

    assert group.rows[0].status == "needs_review"
    assert "format leakage" in " ".join(group.warnings)
    assert "duplicate" in " ".join(group.warnings)


def test_repaired_paddle_schema_strips_answer_leakage_and_skips_response_headers():
    title = doc_ir.Region(
        id="q7", type=doc_ir.TEXT, bbox=[0, 0, 500, 50], reading_order=0,
        content={"text": "7) Who benefits? ✗ Population has disadvantages"},
    )
    table = doc_ir.Region(
        id="matrix", type=doc_ir.TABLE, bbox=[0, 60, 500, 240], reading_order=1,
        content={
            "html": (
                "<table><tr><th></th><th>Good</th><th>Bad</th></tr>"
                "<tr><td>First</td><td>○</td><td>○</td></tr>"
                "<tr><td>Second</td><td>○</td><td>○</td></tr></table>"
            )
        },
    )
    section = {"anchor": title.text, "regions": [title, table]}

    schema = form_extract._schema_hint(section, "p1_q7")

    assert schema["question_text"] == "7) Who benefits?"
    matrix_rows = [row for row in schema["rows"] if row["row_id"].startswith("matrix_")]
    assert [[option["label"] for option in row["options"]] for row in matrix_rows] == [
        ["Good", "Bad"],
        ["Good", "Bad"],
    ]
def test_form_extraction_builds_question_ir_without_mutating_ocr():
    image, _ = _png_bytes()
    page = _survey_page()
    original_html = page.regions[1].content["html"]
    selected_id = "p1_r1_r1_o2"
    client = FakeVisionClient(
        [
            {
                "visible_row_ids": ["p1_r1_r1"],
                "marks": [
                    {
                        "option_id": selected_id,
                        "state": "selected",
                        "visual_mark": "x",
                    }
                ],
                "unmapped_marks": [],
            }
        ]
    )

    groups = form_extract.extract_page_forms(
        page, image, client, contract=form_extract.PADDLE_ID_CONTRACT
    )

    assert page.regions[1].content["html"] == original_html
    assert len(groups) == len(page.form_groups) == 1
    assert "visible_row_ids" in client.calls[0][2]["properties"]
    assert "Do not repeat or rewrite" in client.calls[0][1]
    group = groups[0]
    assert group.id == "p1_q1" and group.question_type == "single"
    row = group.rows[0]
    selected = next(option for option in row.options if option.id == selected_id)
    assert selected.state == "selected" and selected.visual_mark == "x"
    assert {o.source for o in selected.observations} == {"paddleocr-vl", "fake-local"}
    # Paddle transcribed an empty circle while the image reader saw an X: the
    # disagreement is explicit, and the raw OCR table stays unchanged.
    assert group.status == row.status == "needs_review"
    assert any("release benchmark" in warning for warning in group.warnings)

    document = doc_ir.Document(pages=[page], source_name="survey.pdf")
    csv_bytes = doc_ir.build_form_responses_csv(document)
    assert csv_bytes and b"Beta" in csv_bytes
    bundle = zipfile.ZipFile(io.BytesIO(doc_ir.build_full_bundle(document)))
    assert "responses/form_responses.csv" in bundle.namelist()
    assert any(name.startswith("assets/form_p1_q1") for name in bundle.namelist())


def test_missing_vlm_row_is_not_silently_accepted_as_blank():
    image, _ = _png_bytes()
    page = _survey_page()
    client = FakeVisionClient(
        [
            {
                "visible_row_ids": [],
                "marks": [],
                "unmapped_marks": [],
            }
        ]
    )

    group = form_extract.extract_page_forms(
        page, image, client, contract=form_extract.PADDLE_ID_CONTRACT
    )[0]

    assert group.status == "needs_review"
    assert len(group.rows) == 1
    assert group.rows[0].status == "needs_review"
    assert "did not return" in group.rows[0].warnings[0]


def test_unnumbered_question_is_a_form_section():
    image, _ = _png_bytes()
    page = _survey_page()
    page.regions[0].content["text"] = "Which option do you prefer?"
    client = FakeVisionClient(
        [
            {
                "visible_row_ids": ["p1_r1_r1"],
                "marks": [],
                "unmapped_marks": [],
            }
        ]
    )

    groups = form_extract.extract_page_forms(
        page, image, client, contract=form_extract.PADDLE_ID_CONTRACT
    )

    assert len(groups) == 1
    assert groups[0].question_text == "Which option do you prefer?"
    assert len(client.calls) == 1


def test_question_grouping_uses_page_geometry_not_paddle_order():
    q10 = doc_ir.Region(
        id="p1_r0", type=doc_ir.TEXT, bbox=[20, 20, 600, 70], reading_order=0,
        content={"text": "10) Matrix question"},
    )
    q11 = doc_ir.Region(
        id="p1_r1", type=doc_ir.TEXT, bbox=[20, 300, 600, 350], reading_order=1,
        content={"text": "11) Gender"},
    )
    # Paddle may emit a large table after the next title in reading order even
    # though the table is visibly above that title.
    q10_table = doc_ir.Region(
        id="p1_r2", type=doc_ir.TABLE, bbox=[20, 80, 600, 270], reading_order=2,
        content={"html": "<table><tr><td>row</td><td>○</td><td>○</td></tr></table>"},
    )
    q11_answers = doc_ir.Region(
        id="p1_r3", type=doc_ir.TEXT, bbox=[20, 360, 600, 400], reading_order=3,
        content={"text": "○ female ○ male"},
    )
    page = doc_ir.Page(page_number=1, regions=[q10, q11, q10_table, q11_answers])

    sections = form_extract._question_sections(page)

    by_number = {section["number"]: {r.id for r in section["regions"]} for section in sections}
    assert "p1_r2" in by_number["10"] and "p1_r2" not in by_number["11"]
    assert "p1_r3" in by_number["11"]

    by_section = {section["number"]: section for section in sections}
    q10_bbox = form_extract._section_bbox(
        by_section["10"]["regions"],
        640,
        420,
        crop_limits=by_section["10"]["crop_limits"],
    )
    q11_bbox = form_extract._section_bbox(
        by_section["11"]["regions"],
        640,
        420,
        crop_limits=by_section["11"]["crop_limits"],
    )
    # Three pixels of tolerance remain for Paddle's bbox uncertainty, but
    # generic padding cannot include the adjacent numbered question.
    assert q10_bbox[3] <= 297
    assert q11_bbox[1] >= 297


def test_numbered_prize_list_is_not_a_question_and_title_ends_section():
    regions = [
        doc_ir.Region(
            id="q15", type=doc_ir.TITLE, bbox=[20, 20, 700, 60], reading_order=0,
            content={"text": "15) Participate in the draw?"},
        ),
        doc_ir.Region(
            id="prize1", type=doc_ir.TEXT, bbox=[40, 80, 600, 110], reading_order=1,
            content={"text": "1. Voucher worth 100"},
        ),
        doc_ir.Region(
            id="prize2", type=doc_ir.TEXT, bbox=[40, 115, 600, 145], reading_order=2,
            content={"text": "2. Voucher worth 50"},
        ),
        doc_ir.Region(
            id="answers", type=doc_ir.TEXT, bbox=[40, 170, 400, 210], reading_order=3,
            content={"text": "○ Yes ○ No"},
        ),
        doc_ir.Region(
            id="privacy", type=doc_ir.TITLE, bbox=[20, 250, 700, 290], reading_order=4,
            content={"text": "Data processing"},
        ),
        doc_ir.Region(
            id="privacy_text", type=doc_ir.TEXT, bbox=[20, 300, 700, 390], reading_order=5,
            content={"text": "Personal data is deleted after the draw."},
        ),
    ]
    page = doc_ir.Page(page_number=1, width=800, height=450, regions=regions)

    sections = form_extract._question_sections(page)
    numbered = {section["number"]: section for section in sections if section["number"]}

    assert set(numbered) == {"15"}
    q15_ids = {region.id for region in numbered["15"]["regions"]}
    assert {"q15", "prize1", "prize2", "answers"} <= q15_ids
    assert "privacy" not in q15_ids and "privacy_text" not in q15_ids


def test_same_layout_template_reuses_crop_and_schema_only():
    image, _ = _png_bytes()
    template = form_extract.SameLayoutTemplate()
    result = {
        "visible_row_ids": ["p1_r1_r1"],
        "marks": [],
        "unmapped_marks": [],
    }
    first = _survey_page()
    form_extract.extract_page_forms(
        first,
        image,
        FakeVisionClient([result]),
        same_layout_template=template,
        contract=form_extract.PADDLE_ID_CONTRACT,
    )
    # Simulate a later parse whose layout/schema regions were missed. The
    # learned normalized crop still drives an independent image-model call.
    second = doc_ir.Page(page_number=1, width=640, height=420, regions=[])
    second_client = FakeVisionClient([result])
    groups = form_extract.extract_page_forms(
        second,
        image,
        second_client,
        same_layout_template=template,
        contract=form_extract.PADDLE_ID_CONTRACT,
    )

    assert template.learned_pages == 1 and template.reused_pages == 1
    assert len(groups) == len(second_client.calls) == 1
    assert groups[0].provenance["same_layout_template"] is True


def test_same_layout_template_does_not_learn_an_invalid_first_response():
    image, _ = _png_bytes()
    template = form_extract.SameLayoutTemplate()

    groups = form_extract.extract_page_forms(
        _survey_page(),
        image,
        FakeVisionClient([{"questions": [], "unmapped_marks": []}]),
        same_layout_template=template,
    )

    assert groups[0].status == "needs_review"
    assert template.learned_pages == 0 and template.pages == {}


def test_wide_rating_row_stays_complete():
    crop = np.full((220, 2400, 3), 255, np.uint8)
    x1, y1, x2, y2, view = form_extract._complete_section_view(crop)

    assert (x1, y1, x2, y2) == (0, 0, 2400, 220)
    assert view is crop


def test_candidate_filter_rejects_generic_table_and_single_shape():
    generic_table = doc_ir.Region(
        id="p1_r0",
        type=doc_ir.TABLE,
        bbox=[0, 0, 500, 180],
        reading_order=0,
        content={"html": "<table><tr><td>Price</td><td>100</td></tr></table>"},
    )
    section = {"regions": [generic_table], "anchor": "Price list"}
    blank = np.full((180, 500, 3), 255, np.uint8)

    assert form_extract._form_candidate_evidence(section, blank) == []

    cv2.circle(blank, (60, 90), 16, (0, 0, 0), 2)
    assert form_extract._form_candidate_evidence(section, blank) == []


def test_candidate_filter_accepts_paddle_or_aligned_mark_evidence():
    binary = doc_ir.Region(
        id="p1_r0",
        type=doc_ir.TEXT,
        bbox=[0, 0, 500, 80],
        reading_order=0,
        content={"text": "Ja     Nein"},
    )
    blank = np.full((180, 500, 3), 255, np.uint8)
    evidence = form_extract._form_candidate_evidence(
        {"regions": [binary], "anchor": "Participate?"}, blank
    )
    assert evidence == ["paddle-binary-options"]

    visual_only = doc_ir.Region(
        id="p1_r1",
        type=doc_ir.TEXT,
        bbox=[0, 0, 500, 180],
        reading_order=1,
        content={"text": "Female Male No answer"},
    )
    cv2.circle(blank, (60, 90), 16, (0, 0, 0), 2)
    cv2.circle(blank, (180, 90), 16, (0, 0, 0), 2)
    evidence = form_extract._form_candidate_evidence(
        {"regions": [visual_only], "anchor": "Gender"}, blank
    )
    assert evidence == ["aligned-geometric-mark-pattern"]

    notation = doc_ir.Region(
        id="p1_r2",
        type=doc_ir.OTHER,
        bbox=[0, 0, 80, 80],
        reading_order=2,
        content={"latex": r"\\bigotimes"},
    )
    evidence = form_extract._form_candidate_evidence(
        {"regions": [notation], "anchor": "Rating"}, blank
    )
    assert evidence == ["paddle-mark-notation"]


def test_extraction_sends_one_complete_question_image():
    image = np.full((300, 1600, 3), 255, np.uint8)
    title = doc_ir.Region(
        id="p1_r0",
        type=doc_ir.TEXT,
        bbox=[20, 20, 1500, 80],
        reading_order=0,
        content={"text": "1) Wide question?"},
    )
    answers = doc_ir.Region(
        id="p1_r1",
        type=doc_ir.TEXT,
        bbox=[40, 100, 1550, 180],
        reading_order=1,
        content={"text": "○ First ○ Second ○ Third ○ Fourth"},
    )
    page = doc_ir.Page(
        page_number=1, width=1600, height=300, regions=[title, answers]
    )
    client = FakeVisionClient(
        [{"visible_row_ids": ["p1_r1"], "marks": [], "unmapped_marks": []}]
    )

    groups = form_extract.extract_page_forms(
        page, image, client, contract=form_extract.PADDLE_ID_CONTRACT
    )

    assert len(groups) == len(client.calls) == 1
    encoded_image, prompt, _ = client.calls[0]
    decoded = cv2.imdecode(np.frombuffer(encoded_image, np.uint8), cv2.IMREAD_COLOR)
    section = form_extract._question_sections(page)[0]
    expected_bbox = form_extract._section_bbox(
        section["regions"],
        1600,
        300,
        crop_limits=section["crop_limits"],
    )
    assert decoded.shape[1] == expected_bbox[2] - expected_bbox[0]
    assert decoded.shape[0] == expected_bbox[3] - expected_bbox[1]
    assert "complete Paddle-bounded question section" in prompt
    assert "tile" not in prompt.casefold()
    assert groups[0].provenance["method"] == "complete-question-section"
    assert groups[0].provenance["candidate_evidence"] == ["paddle-mark-glyph"]


def test_two_page_spread_does_not_mix_question_columns():
    regions = [
        doc_ir.Region(
            id="left_q14", type=doc_ir.TEXT, bbox=[20, 20, 900, 70], reading_order=0,
            content={"text": "14) Left-column question?"},
        ),
        doc_ir.Region(
            id="right_q1", type=doc_ir.TEXT, bbox=[1320, 20, 2200, 70], reading_order=1,
            content={"text": "1) Right-column question?"},
        ),
        doc_ir.Region(
            id="left_answers", type=doc_ir.TEXT, bbox=[20, 90, 900, 140], reading_order=2,
            content={"text": "○ left A ○ left B"},
        ),
        doc_ir.Region(
            id="right_answers", type=doc_ir.TEXT, bbox=[1320, 90, 2200, 140], reading_order=3,
            content={"text": "○ right A ○ right B"},
        ),
        doc_ir.Region(
            id="left_q15", type=doc_ir.TEXT, bbox=[20, 200, 900, 250], reading_order=4,
            content={"text": "15) Another left question?"},
        ),
        doc_ir.Region(
            id="right_q2", type=doc_ir.TEXT, bbox=[1320, 200, 2200, 250], reading_order=5,
            content={"text": "2) Another right question?"},
        ),
    ]
    page = doc_ir.Page(page_number=1, width=2240, height=400, regions=regions)

    sections = form_extract._question_sections(page)
    first_by_number = {}
    for section in sections:
        first_by_number.setdefault(section["number"], section)
    q14_ids = {region.id for region in first_by_number["14"]["regions"]}
    q1_ids = {region.id for region in first_by_number["1"]["regions"]}

    assert "left_answers" in q14_ids and "right_answers" not in q14_ids
    assert "right_answers" in q1_ids and "left_answers" not in q1_ids


def test_matrix_prompt_schema_is_compact_and_keeps_ids():
    rows = []
    for row_index in range(20):
        rows.append(
            {
                "row_id": f"row_{row_index}",
                "label": f"A fairly long matrix row label {row_index}",
                "options": [
                    {
                        "option_id": f"row_{row_index}_option_{option_index}",
                        "label": f"Rating column {option_index}",
                        "paddle_state": "unchecked",
                    }
                    for option_index in range(6)
                ],
            }
        )
    raw = {"question_id": "q10", "question_text": "Matrix", "rows": rows}

    compact = form_extract._compact_prompt_schema(raw)
    encoded = json.dumps(compact)

    assert "paddle_state" not in encoded
    assert "row_19_option_5" in encoded
    assert len(encoded) < len(json.dumps(raw)) * 0.7


def test_no_schema_uses_same_contract_and_normalizes_unmapped_mark():
    schema_hint = {"question_id": "p1_q4", "question_text": "Question?", "rows": []}
    result = {
        "visible_row_ids": [],
        "marks": [],
        "unmapped_marks": [
            {
                "row_label": "Answer",
                "option_label": "Mittelmässig",
                "state": "selected",
                "visual_mark": "x",
            }
        ],
    }

    normalized = form_extract._normalize_mark_only_result(result, schema_hint)

    assert normalized[0]["question_id"] == "p1_q4"
    assert normalized[0]["rows"][0]["marks"][0]["label"] == "Mittelmässig"
    assert set(form_extract.MARK_ONLY_RESPONSE_SCHEMA["properties"]) == {
        "visible_row_ids", "marks", "unmapped_marks"
    }


def test_ollama_audit_saves_exact_request_and_raw_answer():
    audit_dir = pathlib.Path(tempfile.mkdtemp(prefix="textlab_vlm_audit_test_"))
    client = vision_enrich.OllamaVisionClient(
        model="fake-vl", base_url="http://127.0.0.1:1", audit_dir=audit_dir
    )
    client._prepared = True
    raw_result = {"groups": []}
    client._request = lambda path, payload=None: {
        "message": {"content": json.dumps(raw_result)},
        "eval_count": 12,
    }

    result = client.analyze(
        b"exact-image-bytes",
        'schema {"question_id":"p1_q2"} Complete question section.',
        {"type": "object"},
    )

    call_dir = next(audit_dir.glob("call_*_p1_q2"))
    assert result == raw_result
    assert (call_dir / "input.png").read_bytes() == b"exact-image-bytes"
    assert json.loads((call_dir / "parsed_response.json").read_text()) == raw_result
    assert "eval_count" in (call_dir / "raw_response.json").read_text()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK {name}")
    print("ALL ENRICHMENT TESTS PASSED")
