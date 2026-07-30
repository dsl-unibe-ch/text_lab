"""Searchable-PDF text-layer checks.

The alignment is tested with a stubbed word provider so these run without a
Tesseract binary; only ``test_live_tesseract_words`` needs the real engine and
skips itself when it is unavailable.
"""

import conftest_path  # noqa: F401

import io
import pathlib

import numpy as np

from core import doc_ir, searchable_pdf as sp


def words(*specs):
    """``("Die", 10, 20, 50, 44)`` -> the provider's word-box dict."""
    return [
        {"text": t, "bbox": [float(a), float(b), float(c), float(d)], "conf": 96.0}
        for t, a, b, c, d in specs
    ]


def test_exact_alignment_uses_word_boxes():
    tess = words(("Die", 10, 20, 50, 44), ("Befragung", 60, 20, 190, 44))
    placed = sp.align_tokens(["Die", "Befragung"], tess)
    assert [p["text"] for p in placed] == ["Die", "Befragung"]
    assert all(p["exact"] for p in placed)
    assert placed[1]["bbox"] == [60.0, 20.0, 190.0, 44.0]


def test_vl_text_wins_when_transcriptions_disagree():
    # Tesseract misreads the word; the VL text must still be what is indexed.
    tess = words(("Die", 10, 20, 50, 44), ("Befragurg", 60, 20, 190, 44))
    placed = sp.align_tokens(["Die", "Befragung"], tess)
    texts = " ".join(p["text"] for p in placed)
    assert "Befragung" in texts and "Befragurg" not in texts
    # ...and it lands on the ink Tesseract did find.
    bad = [p for p in placed if p["text"] == "Befragung"][0]
    assert bad["bbox"] == [60.0, 20.0, 190.0, 44.0]
    assert bad["exact"] is False


def test_tokens_without_ink_fall_back_to_a_real_box():
    tess = words(("Die", 10, 20, 50, 44))
    placed = sp.align_tokens(["Die", "Befragung"], tess, fallback_bbox=[0, 0, 500, 60])
    assert "Befragung" in " ".join(p["text"] for p in placed)
    for entry in placed:
        assert entry["bbox"][2] > entry["bbox"][0]
        assert entry["bbox"][3] > entry["bbox"][1]


def test_extra_ink_is_not_invented_into_the_layer():
    # Tesseract sees a word the VL lane did not transcribe: VL is authoritative,
    # so that token must not appear in the searchable text.
    tess = words(("Die", 10, 20, 50, 44), ("Randnotiz", 60, 20, 190, 44))
    placed = sp.align_tokens(["Die"], tess)
    assert "Randnotiz" not in " ".join(p["text"] for p in placed)


def test_no_words_falls_back_to_region_box():
    placed = sp.align_tokens(["Ganzer", "Satz"], [], fallback_bbox=[5, 5, 300, 40])
    assert len(placed) == 1
    assert placed[0]["text"] == "Ganzer Satz"
    assert placed[0]["bbox"] == [5.0, 5.0, 300.0, 40.0]
    assert sp.align_tokens([], []) == []
    assert sp.align_tokens(["x"], []) == []


def test_form_glyphs_are_not_indexed():
    # The VL lane transcribes box/circle glyphs inline on questionnaires. They
    # are not words, and the base-14 PDF fonts cannot encode them — writing one
    # made PyMuPDF drop the whole entry, box included.
    assert sp.tokenize("□ Ja ○ Nein") == ["Ja", "Nein"]
    assert sp.tokenize("☒Zutreffend") == ["Zutreffend"]
    assert sp.tokenize("□ ○ ☐") == []
    assert sp.tokenize("") == []


def test_non_latin1_text_is_folded_not_dropped():
    # A single unencodable character must not cost the whole word its place.
    assert sp._encodable("Frühjahr") == "Frühjahr"
    assert sp._encodable("„Antwort“") == '"Antwort"'
    assert sp._encodable("2024–2025") == "2024-2025"
    assert sp._encodable("□") == ""


def test_pdf_writer_survives_unencodable_entries():
    import fitz

    ok, enc = __import__("cv2").imencode("(.png)"[1:5], np.full((300, 500, 3), 255, np.uint8))
    layers = {1: [
        {"text": "□", "bbox": [10, 10, 40, 40], "exact": True},
        {"text": "Frühjahr", "bbox": [50, 10, 200, 40], "exact": True},
    ]}
    blob = sp.build_searchable_pdf(layers, rasters={1: enc.tobytes()})
    doc = fitz.open("pdf", blob)
    # The unencodable entry is skipped; its neighbour is unaffected.
    assert doc.load_page(0).search_for("Frühjahr")
    doc.close()


def test_table_cells_are_indexed_but_not_the_html():
    region = doc_ir.Region(
        "r1", doc_ir.TABLE, [0, 0, 100, 100], 0,
        {"html": "<table><tr><th>Jahr</th><th>Total</th></tr>"
                 "<tr><td>2024</td><td>17</td></tr></table>"},
    )
    text = sp.region_layer_text(region)
    assert "Jahr" in text and "Total" in text and "2024" in text and "17" in text
    # The markup itself must never reach the searchable layer.
    assert "<" not in text and "table" not in text.lower()
    # Unparseable table -> nothing indexed, rather than raw HTML.
    assert sp.region_layer_text(
        doc_ir.Region("r2", doc_ir.TABLE, [0, 0, 1, 1], 0, {"html": "not html"})
    ) == ""


def test_unprinted_table_headers_are_not_indexed():
    """Regression: tables landed misplaced because of invented header tokens.

    Without ``<th>`` pandas names the columns ``0, 1, ...``. Emitting those put
    tokens in the layer that appear nowhere on the page, and since alignment is
    positional every following cell was pushed onto the wrong box.
    """
    plain = doc_ir.Region(
        "r", doc_ir.TABLE, [0, 0, 1, 1], 0,
        {"html": "<table><tr><td>2024</td><td>17</td></tr>"
                 "<tr><td>2025</td><td>23</td></tr></table>"},
    )
    tokens = sp.tokenize(sp.region_layer_text(plain))
    assert tokens == ["2024", "17", "2025", "23"], tokens

    # A real header row is still indexed, because it really is printed.
    titled = doc_ir.Region(
        "r2", doc_ir.TABLE, [0, 0, 1, 1], 0,
        {"html": "<table><tr><th>Jahr</th><th>Total</th></tr>"
                 "<tr><td>2024</td><td>17</td></tr></table>"},
    )
    assert sp.tokenize(sp.region_layer_text(titled)) == ["Jahr", "Total", "2024", "17"]


def test_page_layer_calls_the_engine_once_for_the_whole_page():
    page = doc_ir.Page(page_number=1, regions=[
        doc_ir.Region("r1", doc_ir.TEXT, [100, 200, 400, 240], 0, {"text": "Hallo Welt"}),
        # Formulas stay out: exported LaTeX is not what is printed.
        doc_ir.Region("r3", doc_ir.FORMULA, [100, 400, 400, 440], 2, {"latex": "E = mc^2"}),
    ])
    seen = []

    def provider(image, lang):
        seen.append(image.shape)
        return words(("Hallo", 100, 200, 160, 230), ("Welt", 170, 200, 230, 230))

    entries = sp.page_text_layer(page, np.full((600, 500, 3), 255, np.uint8),
                                 word_provider=provider)
    # One call, given the whole page — not one call per region.
    assert len(seen) == 1 and seen[0] == (600, 500, 3)
    assert [e["text"] for e in entries] == ["Hallo", "Welt"]
    # Engine output is already in page space, so boxes pass through unchanged.
    assert entries[0]["bbox"] == [100.0, 200.0, 160.0, 230.0]
    assert entries[1]["bbox"] == [170.0, 200.0, 230.0, 230.0]


def test_words_are_bucketed_into_the_region_that_holds_them():
    page = doc_ir.Page(page_number=1, regions=[
        doc_ir.Region("left", doc_ir.TEXT, [0, 0, 200, 100], 0, {"text": "links oben"}),
        doc_ir.Region("right", doc_ir.TEXT, [300, 0, 500, 100], 1, {"text": "rechts oben"}),
    ])

    def provider(image, lang):
        return words(
            ("links", 10, 10, 80, 40), ("oben", 90, 10, 160, 40),
            ("rechts", 310, 10, 390, 40), ("oben", 400, 10, 470, 40),
            ("ausserhalb", 600, 600, 700, 640),   # in no region at all
        )

    entries = sp.page_text_layer(page, np.full((800, 800, 3), 255, np.uint8),
                                 word_provider=provider)
    # Each column keeps its own words; the stray one is not indexed, because the
    # VL lane never transcribed that area.
    assert [e["text"] for e in entries] == ["links", "oben", "rechts", "oben"]
    assert all(e["exact"] for e in entries)
    assert entries[2]["bbox"][0] == 310.0
    assert "ausserhalb" not in " ".join(e["text"] for e in entries)


def test_a_word_is_never_indexed_by_two_overlapping_regions():
    page = doc_ir.Page(page_number=1, regions=[
        doc_ir.Region("outer", doc_ir.TEXT, [0, 0, 400, 200], 0, {"text": "Wort"}),
        doc_ir.Region("inner", doc_ir.TEXT, [50, 50, 300, 150], 1, {"text": "Wort"}),
    ])

    def provider(image, lang):
        return words(("Wort", 100, 80, 180, 110))

    entries = sp.page_text_layer(page, np.full((400, 500, 3), 255, np.uint8),
                                 word_provider=provider)
    # First region in reading order claims it; the second falls back to its box.
    exact = [e for e in entries if e["exact"]]
    assert len(exact) == 1, [(e["text"], e["exact"]) for e in entries]


def test_provider_failure_degrades_to_region_box():
    page = doc_ir.Page(page_number=1, regions=[
        doc_ir.Region("r1", doc_ir.TEXT, [10, 10, 300, 50], 0, {"text": "Hallo Welt"}),
    ])

    def boom(crop, lang):
        raise RuntimeError("tesseract exploded")

    entries = sp.page_text_layer(page, np.full((600, 500, 3), 255, np.uint8),
                                 word_provider=boom)
    assert len(entries) == 1 and entries[0]["text"] == "Hallo Welt"
    assert entries[0]["bbox"] == [10.0, 10.0, 300.0, 50.0]


GERMAN = (
    "Die Befragung wurde im Frühjahr durchgeführt und die Antworten wurden anonym "
    "erfasst. Bitte kreuzen Sie an, wie zufrieden Sie mit den Angeboten sind. Wenn "
    "Sie nicht sicher sind, lassen Sie die Frage bitte offen."
)
ENGLISH = (
    "The survey was carried out in the spring and all responses were recorded "
    "anonymously. Please indicate how satisfied you are with the services provided. "
    "If you are not sure, please leave the question blank."
)
FRENCH = (
    "L'enquête a été réalisée au printemps et toutes les réponses ont été "
    "enregistrées de manière anonyme. Veuillez indiquer dans quelle mesure vous "
    "êtes satisfait des services. Si vous n'êtes pas sûr, laissez la question."
)


def test_language_detection_picks_the_installed_pack():
    assert sp.detect_language(GERMAN) == "deu"
    assert sp.detect_language(ENGLISH) == "eng"
    assert sp.detect_language(FRENCH) == "fra"


def test_language_detection_falls_back_rather_than_guessing():
    # Too little evidence, or no prose at all: the default beats a coin flip.
    assert sp.detect_language("") == "eng"
    assert sp.detect_language("Ja Nein Total") == "eng"
    assert sp.detect_language(" ".join(str(n) for n in range(40))) == "eng"
    assert sp.detect_language("", default="deu") == "deu"


def test_language_detection_is_deterministic():
    # langdetect is seeded, so a close call must not flip between runs.
    assert len({sp.detect_language(GERMAN) for _ in range(5)}) == 1


def test_page_language_reads_only_text_regions():
    page = doc_ir.Page(page_number=1, regions=[
        doc_ir.Region("r1", doc_ir.TEXT, [0, 0, 10, 10], 0, {"text": GERMAN}),
        # A table full of English must not sway the page's language.
        doc_ir.Region("r2", doc_ir.TABLE, [0, 20, 10, 30], 1, {"html": ENGLISH}),
    ])
    assert sp.page_language(page) == "deu"
    assert sp.page_language(doc_ir.Page(page_number=1, regions=[])) == "eng"


def test_pdf_is_searchable_and_text_is_invisible():
    import fitz

    raster = np.full((400, 600, 3), 255, np.uint8)
    ok, enc = __import__("cv2").imencode(".png", raster)
    layers = {1: [
        {"text": "Befragung", "bbox": [60, 100, 190, 130], "exact": True},
        {"text": "Frühjahr", "bbox": [200, 100, 320, 130], "exact": True},
    ]}
    blob = sp.build_searchable_pdf(layers, rasters={1: enc.tobytes()})
    assert blob and blob[:5] == b"%PDF-"

    doc = fitz.open("pdf", blob)
    page = doc.load_page(0)
    assert page.search_for("Befragung"), "word is not findable in the PDF"
    hit = page.search_for("Frühjahr")
    assert hit, "accented word is not findable"
    # The box must sit where the ink was, not span the whole page.
    assert hit[0].width < page.rect.width / 2
    # Invisible: rendered with text-render-mode 3.
    assert "3 Tr" in page.read_contents().decode("latin-1", "ignore")
    doc.close()


def test_png_dpi_tag_does_not_rescale_the_layer():
    """Regression: a screenshot PNG tagged 96 dpi shifted every word.

    PyMuPDF sizes a page from an embedded dpi tag (72/96 = 0.75x the pixels),
    so writing the layer at raw pixel coordinates pushed each word 1.333x too
    far right and down — worsening toward the bottom-right, with words falling
    off the page edge entirely.
    """
    import io

    import fitz
    from PIL import Image

    W, H = 1216, 1696
    marker = {"text": "Marker", "bbox": [1000.0, 1500.0, 1150.0, 1530.0], "exact": True}
    for dpi_tag in (None, (96, 96), (144, 144)):
        buf = io.BytesIO()
        Image.new("RGB", (W, H), "white").save(
            buf, format="PNG", **({"dpi": dpi_tag} if dpi_tag else {})
        )
        blob = sp.build_searchable_pdf(
            {1: [dict(marker)]}, rasters={1: buf.getvalue()}, page_sizes={1: (W, H)}
        )
        doc = fitz.open("pdf", blob)
        page = doc.load_page(0)
        assert (page.rect.width, page.rect.height) == (W, H), (dpi_tag, page.rect)
        hit = page.search_for("Marker")
        assert hit, dpi_tag
        assert page.rect.contains(hit[0]), (dpi_tag, hit[0])
        # Horizontally exact, vertically centred on the ink it was given.
        assert abs(hit[0].x0 - marker["bbox"][0]) < 2, (dpi_tag, hit[0])
        centre = (hit[0].y0 + hit[0].y1) / 2
        assert abs(centre - 1515.0) < 8, (dpi_tag, centre)
        doc.close()


def test_line_metrics_are_shared_across_a_line():
    """Regression: font size came from each word's own ink box.

    "we" has neither ascender nor descender, so its box is far shorter than
    "preliminary" on the same line — which gave it a smaller font and a lower
    baseline, making the highlight jitter word to word.
    """
    line = (0, 0, 1)
    tess = [
        {"text": "preliminary", "bbox": [10.0, 100.0, 120.0, 130.0], "conf": 96.0, "line": line},
        {"text": "we", "bbox": [130.0, 110.0, 160.0, 122.0], "conf": 96.0, "line": line},
        {"text": "Frühjahr", "bbox": [170.0, 100.0, 260.0, 130.0], "conf": 96.0, "line": line},
    ]
    placed = sp.align_tokens(["preliminary", "we", "Frühjahr"], tess)
    sp._apply_line_metrics(placed, tess)

    sizes = {round(p["fontsize"], 6) for p in placed}
    baselines = {round(p["baseline"], 6) for p in placed}
    assert len(sizes) == 1, f"font size varies within a line: {sizes}"
    assert len(baselines) == 1, f"baseline varies within a line: {baselines}"
    # Derived from the line's full ink extent (100..130), not the short word.
    assert abs(placed[0]["fontsize"] - 30.0 / (sp._ASCENT + sp._DESCENT)) < 0.01


def test_separate_lines_keep_separate_baselines():
    tess = [
        {"text": "first", "bbox": [10.0, 100.0, 80.0, 130.0], "conf": 96.0, "line": (0, 0, 1)},
        {"text": "second", "bbox": [10.0, 140.0, 90.0, 170.0], "conf": 96.0, "line": (0, 0, 2)},
    ]
    placed = sp.align_tokens(["first", "second"], tess)
    sp._apply_line_metrics(placed, tess)
    assert placed[0]["baseline"] < placed[1]["baseline"]


def test_live_ruled_table_rows_stay_separate():
    """Regression: a ruled table collapsed onto the header separator line.

    ``--psm 6`` treats a crop as one uniform text block, so on a ruled table it
    read the rules themselves as "|" words and merged every data row into a
    couple of bogus lines. All the cells then shared one tiny baseline.
    """
    if not sp.tesseract_available():
        print("   (skipped: tesseract not installed)")
        return
    from PIL import Image, ImageDraw, ImageFont

    # A real TrueType face, not cv2.putText: the Hershey stroke fonts are not
    # OCR-able, so a fixture drawn with them tests nothing.
    font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
    if not pathlib.Path(font_path).exists():
        print("   (skipped: Times New Roman not installed)")
        return
    font = ImageFont.truetype(font_path, 26)

    img = Image.new("RGB", (620, 280), "white")
    draw = ImageDraw.Draw(img)
    rows = [("MHA 1L", "0.32", "0.53"), ("MuRS 1L", "0.33", "0.54"),
            ("MHA 2L", "0.31", "0.52"), ("MuRS 2L", "0.30", "0.51")]
    draw.rectangle([10, 10, 610, 270], outline="black", width=2)
    for index, (label, a, b) in enumerate(rows):
        y = 30 + index * 60
        draw.line([10, y - 8, 610, y - 8], fill="black", width=2)   # row rule
        for x, text in ((25, label), (300, a), (450, b)):
            draw.text((x, y), text, fill="black", font=font)
    for x in (280, 430):                                            # column rules
        draw.line([x, 10, x, 270], fill="black", width=2)

    found = sp.tesseract_words(np.array(img)[:, :, ::-1], lang="eng")
    assert found, "no words read from the ruled table"
    # Rules must never enter the layer as anchors.
    for word in found:
        assert sp._normalise(word["text"]), f"rule glyph kept as a word: {word['text']!r}"
    # Rows must not all land on one line.
    lines = {word["line"] for word in found}
    assert len(lines) >= 3, f"table rows collapsed into {len(lines)} line(s)"
    # And the per-line ink boxes must be text-height, not hairlines.
    heights = []
    for key in lines:
        boxes = [w["bbox"] for w in found if w["line"] == key]
        heights.append(max(b[3] for b in boxes) - min(b[1] for b in boxes))
    assert max(heights) > 10, f"line boxes are hairlines: {heights}"


def test_live_tesseract_words():
    if not sp.tesseract_available():
        print("   (skipped: tesseract not installed in this container)")
        return
    import cv2

    img = np.full((120, 700, 3), 255, np.uint8)
    cv2.putText(img, "Hallo Welt", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    found = sp.tesseract_words(img, lang="deu")
    assert found, "tesseract returned no words"
    for word in found:
        x1, y1, x2, y2 = word["bbox"]
        assert x2 > x1 and y2 > y1
        assert word["conf"] >= sp.MIN_WORD_CONFIDENCE
    joined = " ".join(w["text"] for w in found)
    assert "Hallo" in joined, joined


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK {name}")
    print("ALL SEARCHABLE-PDF TESTS PASSED")
