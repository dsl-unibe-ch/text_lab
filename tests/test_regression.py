"""Core IR / adapter / worker-protocol / native-lane regression checks."""

import conftest_path  # noqa: F401

import base64
import io
import json
import pathlib
import sys
import tempfile
import time
import zipfile

import numpy as np
import cv2

from core import doc_ir, auto_ocr, markup_detect as md

WORK = pathlib.Path(tempfile.mkdtemp(prefix="textlab_test_"))


def crop_b64(kind):
    img = np.full((40, 40, 3), 255, np.uint8)
    cv2.rectangle(img, (2, 2), (37, 37), (0, 0, 0), 1)
    if kind == "checked":
        cv2.line(img, (8, 8), (31, 31), (0, 0, 0), 3)
        cv2.line(img, (31, 8), (8, 31), (0, 0, 0), 3)
    ok, enc = cv2.imencode(".png", img)
    return base64.b64encode(enc.tobytes()).decode()


PAGE_JSON = {
    "page_number": 1, "width": 800, "height": 1000,
    "parsing_res_list": [
        {"block_label": "title", "block_content": "Survey Form", "block_bbox": [10, 10, 400, 40], "block_order": 0},
        {"block_label": "checkbox", "block_content": "", "block_bbox": [10, 90, 50, 130], "block_order": 1},
        {"block_label": "table", "block_content": "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>", "block_bbox": [10, 200, 400, 300], "block_order": 2},
        {"block_label": "formula", "block_content": "E = mc^2", "block_bbox": [10, 320, 200, 360], "block_order": 3},
    ],
    "layout_det_res": [{"label": "checkbox", "score": 0.85, "coordinate": [10, 90, 50, 130]}],
    "assets": {"1": {"b64": crop_b64("checked"), "ext": "png"}},
    "markdown": "# Survey",
}


def test_adapter_and_exports():
    page = doc_ir.from_paddle_vl(PAGE_JSON)
    assert [r.type for r in page.regions] == ["title", "checkbox", "table", "formula"]
    doc = doc_ir.Document(pages=[page], source_name="s.pdf")
    tables = doc_ir.tables_to_dataframes(doc)
    assert len(tables) == 1 and list(tables[0]["dataframe"].columns) == ["A", "B"]
    mdtxt = doc_ir.to_markdown(doc)
    assert "## Survey Form" in mdtxt and "$$" in mdtxt
    names = zipfile.ZipFile(io.BytesIO(doc_ir.build_full_bundle(doc))).namelist()
    assert any(n.startswith("tables/") for n in names)
    assert any(n.startswith("assets/") for n in names)
    json.loads(doc_ir.to_json(doc))


def test_text_export():
    doc = doc_ir.Document(pages=[doc_ir.from_paddle_vl(PAGE_JSON)], source_name="s.pdf")
    txt = doc_ir.to_text(doc)
    assert "Survey Form" in txt and "E = mc^2" in txt
    # Tables become readable columns, never raw markup.
    assert "A" in txt and "1" in txt
    assert "<table>" not in txt and "<td>" not in txt and "##" not in txt
    # A checkbox keeps its state; the crop is named rather than embedded.
    assert "[?]" in txt or "[x]" in txt or "[ ]" in txt

    two = doc_ir.Document(pages=[
        doc_ir.from_paddle_vl(PAGE_JSON), doc_ir.from_paddle_vl(PAGE_JSON)
    ])
    two.pages[1].page_number = 2
    assert "--- page 2 ---" in doc_ir.to_text(two)
    assert doc_ir.to_text(doc_ir.Document(pages=[])).strip() == ""


def test_docx_export():
    doc = doc_ir.Document(pages=[doc_ir.from_paddle_vl(PAGE_JSON)], source_name="s.pdf")
    blob = doc_ir.build_docx(doc, "s")
    if blob is None:  # python-docx absent: the caller hides the download
        return
    assert "word/document.xml" in zipfile.ZipFile(io.BytesIO(blob)).namelist()

    # A figure carries its crop into the document; a checkbox stays a text
    # marker, so embedding is checked on a page that actually has a figure.
    with_figure = doc_ir.Document(pages=[doc_ir.Page(page_number=1, regions=[
        doc_ir.Region("r1", doc_ir.FIGURE, [0, 0, 40, 40], 0, {"text": "Fig 1"},
                      asset={"b64": crop_b64("checked"), "ext": "png"}),
    ])])
    fig_blob = doc_ir.build_docx(with_figure, "f")
    fig_names = zipfile.ZipFile(io.BytesIO(fig_blob)).namelist()
    assert any(n.startswith("word/media/") for n in fig_names), "figure crop not embedded"

    import docx as _docx

    # The caption is italic on its run. Setting ``Paragraph.italic`` instead is
    # accepted silently by python-docx and formats nothing.
    caption_runs = [
        run
        for paragraph in _docx.Document(io.BytesIO(fig_blob)).paragraphs
        for run in paragraph.runs
        if run.text == "Fig 1"
    ]
    assert caption_runs, "figure caption missing"
    assert all(run.italic for run in caption_runs), "figure caption is not italic"

    parsed = _docx.Document(io.BytesIO(blob))
    assert parsed.paragraphs[0].style.name.startswith("Heading")
    assert parsed.paragraphs[0].text == "Survey Form"
    # The table must be a real Word table, not a pasted string.
    assert len(parsed.tables) == 1
    table = parsed.tables[0]
    assert (len(table.rows), len(table.columns)) == (2, 2)
    assert [c.text for c in table.rows[0].cells] == ["A", "B"]
    assert [c.text for c in table.rows[1].cells] == ["1", "2"]
    assert doc_ir.build_docx(doc_ir.Document(pages=[]), "empty") is not None


def test_full_bundle_carries_every_format():
    doc = doc_ir.Document(pages=[doc_ir.from_paddle_vl(PAGE_JSON)], source_name="s.pdf")
    names = zipfile.ZipFile(io.BytesIO(doc_ir.build_full_bundle(doc, "s"))).namelist()
    for expected in ("s.md", "s.txt", "s.json"):
        assert expected in names, f"{expected} missing from bundle: {names}"
    if doc_ir.build_docx(doc, "s") is not None:
        assert "s.docx" in names


def test_finalize_vl_page():
    raster = WORK / "p1.png"
    cv2.imwrite(str(raster), np.full((1000, 800, 3), 255, np.uint8))
    fp = auto_ocr._finalize_vl_page(PAGE_JSON, 1, raster)
    assert fp.image_b64
    cb = [r for r in fp.regions if r.type == doc_ir.CHECKBOX][0]
    # Ordinary OCR is immutable and does not run form interpretation implicitly.
    assert cb.markup is None

    # Geometry can still be requested as review evidence, but it is never
    # allowed to rewrite the OCR token.
    auto_ocr._apply_markup(fp, cv2.imread(str(raster)))
    assert cb.markup and cb.markup["state"] == "uncertain"
    geometry = next(
        item for item in cb.markup["observations"] if item["source"] == "geometric"
    )
    assert geometry["state"] in {"checked", "unchecked", "uncertain"}


def test_worker_protocol():
    stub = WORK / "stub.py"
    stub.write_text(
        "import json\nprint('TEXTLAB_PADDLEVL_RESULT_JSON=' + "
        "json.dumps({'pages': [{'page_number':1,'parsing_res_list':[],'layout_det_res':[],'assets':{}}]}))\n"
    )
    pages = auto_ocr.run_vl_worker([pathlib.Path("x.png")], backend_python=sys.executable, worker_path=stub)
    assert len(pages) == 1
    bad = WORK / "bad.py"
    bad.write_text("import sys; sys.exit(3)\n")
    try:
        auto_ocr.run_vl_worker([pathlib.Path("x.png")], backend_python=sys.executable, worker_path=bad)
        raise AssertionError("should raise")
    except RuntimeError:
        pass


def test_native_lane():
    _img = np.full((20, 20, 3), 128, np.uint8)
    _ok, _enc = cv2.imencode(".png", _img)
    PNG = _enc.tobytes()

    class FR:
        width = 595.0
        height = 842.0

    class FPix:
        def tobytes(self, fmt="png"):
            return PNG

    class FPage:
        rect = FR()

        def get_text(self, mode):
            if mode == "text":
                return "This is a born digital page with plenty of words in it."
            if mode == "words":
                return [("w",)] * 11
            if mode == "dict":
                return {"blocks": [
                    {"type": 0, "bbox": [72, 72, 500, 96], "lines": [{"spans": [{"text": "Hello world", "font": "Arial"}]}]},
                    {"type": 1, "bbox": [72, 120, 300, 300], "image": PNG, "ext": "png"},
                ]}
            return []

        def get_pixmap(self, dpi=150):
            return FPix()

    assert auto_ocr._page_has_text_layer(FPage()) is True
    assert auto_ocr._page_has_math(FPage()) is False
    np_page = auto_ocr._native_page(FPage(), 3)
    assert [r.type for r in np_page.regions] == ["text", "figure"]
    assert np_page.regions[0].text == "Hello world"
    assert np_page.image_b64


def test_api_compat():
    assert md.classify_from_b64("☑", None)["state"] == "checked"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK {name}")
    print("ALL REGRESSION TESTS PASSED")
