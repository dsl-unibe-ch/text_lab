"""Page images, line crops, previews and exports."""

import io
import json
import zipfile
from xml.etree import ElementTree

import pytest
from PIL import Image

from textlab.features.handwriting import documents
from textlab.features.handwriting.models import (
    DocumentTranscription,
    HandwritingError,
    HandwritingModel,
    Line,
    PageTranscription,
)

SPEC = HandwritingModel("k", "Kurrent", "", "org/base", None, "Read.")
NS = {"p": documents.PAGE_NAMESPACE}


def test_every_frame_of_an_image_becomes_a_page(tmp_path):
    source = tmp_path / "scan.tif"
    frames = [Image.new("L", (40, 30), shade) for shade in (0, 255)]
    frames[0].save(source, save_all=True, append_images=frames[1:])
    pages = documents.page_images(source, tmp_path / "pages")
    assert [page.name for page in pages] == ["page-0001.png", "page-0002.png"]
    with Image.open(pages[1]) as image:
        assert image.mode == "RGB" and image.size == (40, 30)


def test_an_unreadable_image_is_an_error(tmp_path):
    source = tmp_path / "broken.png"
    source.write_bytes(b"not an image")
    with pytest.raises(HandwritingError, match="broken.png"):
        documents.page_images(source, tmp_path / "pages")


def test_crops_are_numbered_across_pages_and_scaled(tmp_path):
    page = tmp_path / "page.png"
    Image.new("RGB", (400, 100), "white").save(page)
    first = documents.write_crops(page, [(0, 0, 400, 50)], tmp_path / "c", 0)
    second = documents.write_crops(
        page,
        [(0, 0, 400, 100), (0, 0, 10, 10)],
        tmp_path / "c",
        10_000,
        first=1,
    )
    assert [path.name for path in first + second] == [
        "line-000000.png",
        "line-000001.png",
        "line-000002.png",
    ]
    with Image.open(second[0]) as image:
        assert image.width * image.height <= 10_000
        assert image.width == 4 * image.height


def test_the_preview_is_a_scaled_png(tmp_path):
    path = tmp_path / "page.png"
    Image.new("RGB", (3200, 1000), "white").save(path)
    page = PageTranscription(1, 3200, 1000, [Line((10, 10, 3000, 90), "a")])
    with Image.open(io.BytesIO(documents.preview_png(path, page))) as image:
        assert image.size == (documents.PREVIEW_SIDE, 500)


def test_page_xml_holds_every_line_with_its_box():
    page = PageTranscription(
        2,
        800,
        600,
        [Line((10, 20, 300, 60), "Erste <Zeile>"), Line((10, 70, 200, 110))],
        image_name="scan.png",
    )
    root = ElementTree.fromstring(documents.page_xml(page, "Kurrent"))
    element = root.find("p:Page", NS)
    assert element.get("imageFilename") == "scan.png"
    assert (element.get("imageWidth"), element.get("imageHeight")) == (
        "800",
        "600",
    )
    region = element.find("p:TextRegion", NS)
    assert region.find("p:Coords", NS).get("points") == (
        "10,20 300,20 300,110 10,110"
    )
    text_lines = region.findall("p:TextLine", NS)
    assert [line.get("id") for line in text_lines] == ["r1l1", "r1l2"]
    assert (
        text_lines[0].find("p:TextEquiv/p:Unicode", NS).text == "Erste <Zeile>"
    )


def test_a_page_without_lines_has_no_region():
    page = PageTranscription(1, 10, 10, [])
    root = ElementTree.fromstring(documents.page_xml(page, "Kurrent"))
    assert root.find("p:Page/p:TextRegion", NS) is None


def test_outputs_include_page_images_only_when_asked(tmp_path):
    image = tmp_path / "rendered.png"
    Image.new("RGB", (10, 10)).save(image)
    pages = [
        PageTranscription(
            1, 10, 10, [Line((0, 0, 5, 5), "a")], "page_0001.png"
        ),
        PageTranscription(2, 10, 10, [], "page_0002.png"),
    ]
    document = DocumentTranscription("letter.pdf", SPEC, pages)
    documents.write_outputs(document, tmp_path / "out", [image, None])
    out = tmp_path / "out"
    assert (out / "letter.txt").read_text(encoding="utf-8") == "a\n\n"
    assert json.loads((out / "letter.json").read_text())["file"] == (
        "letter.pdf"
    )
    assert sorted(path.name for path in (out / "page_xml").iterdir()) == [
        "page_0001.png",
        "page_0001.xml",
        "page_0002.xml",
    ]


def test_zip_inputs_are_filtered_and_sorted(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("b/Two.png", b"x")
        archive.writestr("a.pdf", b"x")
        archive.writestr("notes.txt", b"x")
        archive.writestr("__MACOSX/._a.pdf", b"x")
    buffer.seek(0)
    input_dir = (tmp_path / "in").resolve()
    files = documents.extract_inputs(buffer, input_dir)
    assert [path.relative_to(input_dir).as_posix() for path in files] == [
        "a.pdf",
        "b/Two.png",
    ]


def test_a_broken_zip_is_an_error(tmp_path):
    with pytest.raises(HandwritingError, match="ZIP"):
        documents.extract_inputs(io.BytesIO(b"nope"), tmp_path)


def test_models_used_text_names_both_models():
    text = documents.models_used_text(
        {"line_detection": "Detector", "recognition": "Reader"}
    )
    assert text == "Line detection: Detector\nText recognition: Reader\n"
