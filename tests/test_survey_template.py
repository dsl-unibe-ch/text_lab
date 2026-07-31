"""Template-first survey extraction: registration, consensus blank, mark reads."""

import conftest_path  # noqa: F401

import pathlib
import tempfile

import cv2
import numpy as np

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
