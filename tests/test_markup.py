"""Markup detection: stroke-aware observations without OCR mutation."""

import conftest_path  # noqa: F401

import base64

import numpy as np
import cv2

from core import markup_detect as md
from core import auto_ocr, doc_ir


def b64(img):
    ok, enc = cv2.imencode(".png", img)
    return base64.b64encode(enc.tobytes()).decode()


def box_crop(box_px, stroke, kind):
    img = np.full((box_px, box_px, 3), 255, np.uint8)
    m = max(2, box_px // 20)
    cv2.rectangle(img, (m, m), (box_px - m, box_px - m), (0, 0, 0), 1)
    a, b = int(box_px * 0.25), int(box_px * 0.75)
    if kind == "X":
        cv2.line(img, (a, a), (b, b), (0, 0, 0), stroke)
        cv2.line(img, (b, a), (a, b), (0, 0, 0), stroke)
    elif kind == "check":
        cv2.line(img, (a, int(box_px * 0.55)), (int(box_px * 0.45), b), (0, 0, 0), stroke)
        cv2.line(img, (int(box_px * 0.45), b), (b, a), (0, 0, 0), stroke)
    return img


def survey_row(n=4, crossed=(1,), rad=15, y=30, W=420, H=60, label_words=None):
    img = np.full((H, W, 3), 255, np.uint8)
    xs = [40 + i * 100 for i in range(n)]
    for i, x in enumerate(xs):
        cv2.circle(img, (x, y), rad, (0, 0, 0), 1)
        if i in crossed:
            d = int(rad * 0.75)
            cv2.line(img, (x - d, y - d), (x + d, y + d), (0, 0, 0), 2)
            cv2.line(img, (x + d, y - d), (x - d, y + d), (0, 0, 0), 2)
    if label_words:
        for x, word in label_words:
            cv2.putText(img, word, (x, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def test_stroke_aware_classifier():
    for box in (40, 60, 90, 130):
        for stroke in (1, 2, 3):
            for kind in ("X", "check"):
                r = md.detect_markup_geometric(box_crop(box, stroke, kind))
                assert r["state"] == "checked", (box, stroke, kind, r)
    for box in (40, 90, 130):
        img = np.full((box, box, 3), 255, np.uint8)
        m = max(2, box // 20)
        cv2.rectangle(img, (m, m), (box - m, box - m), (0, 0, 0), 1)
        assert md.detect_markup_geometric(img)["state"] == "unchecked", box
    for rad in (14, 22, 35):
        sz = rad * 2 + 8
        img = np.full((sz, sz, 3), 255, np.uint8)
        cv2.circle(img, (sz // 2, sz // 2), rad, (0, 0, 0), 1)
        assert md.detect_markup_geometric(img)["state"] == "unchecked", rad


def test_glyph_extraction():
    assert [i["state"] for i in md.extract_mark_glyphs("○ Ja ○ Nein")] == ["unchecked", "unchecked"]
    assert [i["state"] for i in md.extract_mark_glyphs("<td>☒</td><td>☐</td>")] == ["checked", "unchecked"]
    assert [i["state"] for i in md.extract_mark_glyphs("○\t✗\t○")] == ["unchecked", "checked", "unchecked"]
    assert md.extract_mark_glyphs("plain text") == []


def test_find_marks_and_letter_filtering():
    marks = md.find_marks(survey_row(n=4, crossed=(1,)), n_expected=4)
    assert [m["state"] for m in marks] == ["unchecked", "checked", "unchecked", "unchecked"]
    marks2 = md.find_marks(
        survey_row(n=2, crossed=(0,), label_words=[(70, "Ja"), (170, "Nein")]),
        n_expected=2,
    )
    assert [m["state"] for m in marks2] == ["checked", "unchecked"]


def test_reconcile_is_non_mutating_by_default():
    content = "○\t○\t○\t○"
    marks = md.find_marks(survey_row(n=4, crossed=(1,)), n_expected=4)
    items, status = md.reconcile_marks(md.extract_mark_glyphs(content), marks)
    states = [i["state"] for i in items]
    assert states == ["unchecked", "unchecked", "unchecked", "unchecked"]
    assert status == "geometry_disagreement"
    assert items[1]["needs_review"] is True
    assert items[1]["geometry"]["state"] == "checked"

    # The old behaviour remains callable only as a frozen benchmark baseline.
    baseline, _ = md.reconcile_marks(
        md.extract_mark_glyphs(content), marks, allow_override=True
    )
    baseline_states = [i["state"] for i in baseline]
    assert baseline_states == ["unchecked", "checked", "unchecked", "unchecked"]
    assert baseline[1]["method"] == "geometric-override"
    new_content, changed = md.apply_states_to_content(content, baseline_states)
    assert changed and new_content == "○\t☒\t○\t○"
    # transcribed checked never downgraded
    items2, _ = md.reconcile_marks(
        [{"index": 0, "glyph": "☒", "state": "checked", "method": "vl"}],
        [{"state": "unchecked", "score": 0.9}],
    )
    assert items2[0]["state"] == "checked"
    # count mismatch -> untouched
    items3, status3 = md.reconcile_marks(md.extract_mark_glyphs(content), marks[:3])
    assert status3 == "count_mismatch"
    assert [i["state"] for i in items3] == ["unchecked"] * 4


def test_apply_markup_end_to_end():
    page_img = np.full((800, 600, 3), 255, np.uint8)
    page_img[100:160, 90:510] = survey_row(n=4, crossed=(2,))
    region = doc_ir.Region(
        id="p1_r0", type=doc_ir.TABLE, bbox=[90, 100, 510, 160], reading_order=0,
        content={"html": "<table><tr><td>○</td><td>○</td><td>○</td><td>○</td></tr></table>"},
    )
    cb = doc_ir.Region(
        id="p1_r1", type=doc_ir.CHECKBOX, bbox=[50, 300, 140, 390], reading_order=1,
        content={"text": ""}, asset={"b64": b64(box_crop(90, 2, "X")), "ext": "png"},
    )
    page = doc_ir.Page(page_number=1, regions=[region, cb], source="paddleocr-vl-1.6")
    auto_ocr._apply_markup(page, page_img)
    mk = region.markup
    assert mk["status"] == "geometry_disagreement"
    assert mk["n_checked"] == 0 and mk["n_overridden"] == 0
    assert mk["n_disagreements"] == 1
    assert "☒" not in region.content["html"]
    assert cb.markup["state"] == "uncertain"
    assert cb.markup["observations"][0]["source"] == "geometric"
    summ = auto_ocr.document_summary(doc_ir.Document(pages=[page]))
    assert summ["n_marks"] == 5 and summ["n_overridden_marks"] == 0
    assert summ["n_markup_disagreements"] == 1
    disputed = [i for i in mk["items"] if i.get("needs_review")]
    assert disputed and disputed[0].get("crop_b64")


def test_math_routing():
    class FakePage:
        def __init__(self, text, fonts=()):
            self._t, self._f = text, fonts

        def get_text(self, mode):
            if mode == "text":
                return self._t
            if mode == "dict":
                return {"blocks": [{"lines": [{"spans": [{"font": f, "text": "x"} for f in self._f]}]}]}
            return []

    assert auto_ocr._page_has_math(FakePage("∑ ∫ α β γ ≤ ≥ ± √ formulas")) is True
    assert auto_ocr._page_has_math(FakePage("Ganz normaler deutscher Text über Tourismus.")) is False
    assert auto_ocr._page_has_math(FakePage("short", fonts=("CMMI10", "CMSY7"))) is True
    assert auto_ocr._page_has_math(FakePage("footnote²  ", fonts=("Arial",))) is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK {name}")
    print("ALL MARKUP TESTS PASSED")
