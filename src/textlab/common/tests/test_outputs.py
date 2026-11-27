"""Writing a result's files: the noted files and images for browsers."""

import io

from PIL import Image

from textlab.common.outputs import (
    OutputFolder,
    SavedFile,
    image_media_type,
    web_image,
)


def image(format_, size=(40, 20)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, format=format_)
    return buffer.getvalue()


def test_every_file_written_is_noted_in_order(tmp_path):
    output = OutputFolder(tmp_path)
    assert output.write("a.txt", b"a", "Text", "text/plain") == "a.txt"
    output.write("p.png", b"p", "Page 1", "image/png", listed=False)
    assert output.files == [
        SavedFile("a.txt", "Text", "text/plain", tmp_path / "a.txt"),
        SavedFile("p.png", "Page 1", "image/png", tmp_path / "p.png", False),
    ]
    assert (tmp_path / "a.txt").read_bytes() == b"a"


def test_png_and_jpeg_are_kept_as_they_are():
    for format_, extension in (("PNG", "png"), ("JPEG", "jpg")):
        data = image(format_)
        assert web_image(data) == (data, extension, (40, 20))


def test_other_formats_become_png():
    data, extension, size = web_image(image("TIFF"))
    assert (extension, size) == ("png", (40, 20))
    assert Image.open(io.BytesIO(data)).format == "PNG"


def test_a_large_image_is_scaled_down_with_its_original_size():
    data, extension, size = web_image(image("PNG", (4000, 1000)), 1600)
    assert (extension, size) == ("jpg", (4000, 1000))
    assert Image.open(io.BytesIO(data)).size == (1600, 400)


def test_an_unreadable_image_gives_nothing():
    assert web_image(b"not an image") == (None, "png", (0, 0))


def test_image_media_types():
    assert image_media_type("jpg") == "image/jpeg"
    assert image_media_type("png") == "image/png"
