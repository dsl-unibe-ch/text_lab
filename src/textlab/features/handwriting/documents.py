"""Files in and out: page images, line crops, previews and exports.

Every page is turned into a PNG image first: a PDF is rendered with
``pdftoppm`` at :data:`DPI`, and every frame of an image file (a TIFF can
have several) becomes a page. Lines are cropped from those images, and the
results are written as plain text, JSON and PAGE XML, the format
eScriptorium and Transkribus import for correction.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO
from xml.etree import ElementTree

from textlab.common.upload_safety import extract_zip_safely
from textlab.features.handwriting.lines import crop_box, fit_pixels
from textlab.features.handwriting.models import (
    DocumentTranscription,
    HandwritingError,
    PageTranscription,
)

#: File types read, alone or inside a ZIP archive.
INPUT_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
#: Resolution PDFs are rendered at. Handwriting needs more detail than
#: print, and 300 DPI is the usual resolution of archive scans.
DPI = 300
#: Longest side of a preview image, in pixels.
PREVIEW_SIDE = 1600
#: PAGE XML namespace (the 2019 schema, read by eScriptorium and
#: Transkribus).
PAGE_NAMESPACE = (
    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
)


def page_images(input_path: Path, out_dir: Path) -> list[Path]:
    """Return a file's pages as PNG images.

    Args:
        input_path: A PDF or an image file.
        out_dir: Where to write the images.

    Returns:
        The page images, in page order.

    Raises:
        HandwritingError: If the file cannot be read or has no pages.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if input_path.suffix.lower() == ".pdf":
        result = subprocess.run(
            [
                "pdftoppm",
                "-r",
                str(DPI),
                "-png",
                str(input_path),
                str(out_dir / "page"),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise HandwritingError(
                f"Could not read the PDF {input_path.name}: "
                f"{result.stderr.strip()[-500:]}"
            )
        # pdftoppm pads page numbers to the same width, so names sort.
        images = sorted(out_dir.glob("page-*.png"))
    else:
        images = _image_frames(input_path, out_dir)
    if not images:
        raise HandwritingError(f"{input_path.name} has no pages.")
    return images


def _image_frames(input_path: Path, out_dir: Path) -> list[Path]:
    """Write every frame of an image file as an RGB PNG."""
    from PIL import Image, ImageSequence, UnidentifiedImageError

    try:
        with Image.open(input_path) as image:
            images = []
            for index, frame in enumerate(ImageSequence.Iterator(image), 1):
                path = out_dir / f"page-{index:04d}.png"
                frame.convert("RGB").save(path)
                images.append(path)
            return images
    except (OSError, UnidentifiedImageError) as error:
        raise HandwritingError(
            f"Could not read the image {input_path.name}: {error}"
        ) from error


def image_size(path: Path) -> tuple[int, int]:
    """Return the width and height of an image file."""
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def write_crops(
    image_path: Path,
    boxes: list[tuple[int, int, int, int]],
    out_dir: Path,
    max_pixels: int | None,
    first: int = 0,
) -> list[Path]:
    """Crop lines from a page image and save them as PNG files.

    Args:
        image_path: The page image.
        boxes: The crops (:func:`.lines.crop_box`), in reading order.
        out_dir: Where to write them.
        max_pixels: Crops larger than this are scaled down to it.
        first: Number of the first crop, so all pages of a run can share
            the folder.

    Returns:
        The crop files, in the order of ``boxes``.
    """
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with Image.open(image_path) as page:
        page = page.convert("RGB")
        for index, box in enumerate(boxes, start=first):
            crop = page.crop(box)
            size = fit_pixels(*crop.size, max_pixels)
            if size != crop.size:
                crop = crop.resize(size, Image.Resampling.LANCZOS)
            path = out_dir / f"line-{index:06d}.png"
            crop.save(path)
            paths.append(path)
    return paths


def line_crops(
    lines: list[tuple[float, float, float, float]], width: int, height: int
) -> list[tuple[int, int, int, int]]:
    """Return the crop of each line on a page of this size."""
    return [crop_box(line, width, height) for line in lines]


def preview_png(image_path: Path, page: PageTranscription) -> bytes:
    """Return the page image with its lines outlined and numbered, as PNG.

    The image is scaled down to :data:`PREVIEW_SIDE` first.
    """
    from PIL import Image, ImageDraw

    with Image.open(image_path) as image:
        image = image.convert("RGB")
    scale = min(1.0, PREVIEW_SIDE / max(image.size))
    if scale < 1.0:
        size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    for number, line in enumerate(page.lines, start=1):
        box = [round(value * scale) for value in line.box]
        draw.rectangle(box, outline=(220, 40, 40), width=2)
        draw.text((box[0] + 3, box[1] + 1), str(number), fill=(220, 40, 40))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def to_json(document: DocumentTranscription) -> str:
    """Return the transcription as indented JSON."""
    return json.dumps(document.to_dict(), ensure_ascii=False, indent=2)


def models_used_text(models: dict[str, str]) -> str:
    """Return the models used as text, for ``models_used.txt``."""
    return "".join(
        f"{label}: {models[key]}\n"
        for key, label in (
            ("line_detection", "Line detection"),
            ("recognition", "Text recognition"),
        )
    )


def page_xml(page: PageTranscription, model_name: str) -> bytes:
    """Return one page as PAGE XML, with one region holding all lines.

    Args:
        page: The page.
        model_name: The recognition model, recorded as the creator.

    Returns:
        The XML document, UTF-8 encoded.
    """
    ns = PAGE_NAMESPACE
    ElementTree.register_namespace("", ns)
    root = ElementTree.Element(f"{{{ns}}}PcGts")
    metadata = ElementTree.SubElement(root, f"{{{ns}}}Metadata")
    ElementTree.SubElement(
        metadata, f"{{{ns}}}Creator"
    ).text = f"Text Lab ({model_name})"
    element = ElementTree.SubElement(
        root,
        f"{{{ns}}}Page",
        imageFilename=page.image_name or f"page_{page.page:04d}.png",
        imageWidth=str(page.width),
        imageHeight=str(page.height),
    )
    if page.lines:
        region = ElementTree.SubElement(
            element, f"{{{ns}}}TextRegion", id="r1"
        )
        ElementTree.SubElement(
            region,
            f"{{{ns}}}Coords",
            points=_points(_union_box([line.box for line in page.lines])),
        )
        for number, line in enumerate(page.lines, start=1):
            text_line = ElementTree.SubElement(
                region, f"{{{ns}}}TextLine", id=f"r1l{number}"
            )
            ElementTree.SubElement(
                text_line, f"{{{ns}}}Coords", points=_points(line.box)
            )
            equiv = ElementTree.SubElement(text_line, f"{{{ns}}}TextEquiv")
            ElementTree.SubElement(equiv, f"{{{ns}}}Unicode").text = line.text
    ElementTree.indent(root)
    return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)


def _union_box(boxes: list[tuple[int, int, int, int]]) -> tuple:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _points(box: tuple[int, int, int, int]) -> str:
    """Return a box as the corner points of a PAGE XML ``Coords``."""
    left, top, right, bottom = box
    return f"{left},{top} {right},{top} {right},{bottom} {left},{bottom}"


def write_outputs(
    document: DocumentTranscription,
    folder: Path,
    images: Sequence[Path | None] = (),
) -> None:
    """Write a document's text, JSON and PAGE XML files into a folder.

    The files are ``<name>.txt``, ``<name>.json`` and
    ``page_xml/page_0001.xml`` per page. The PAGE XML coordinates are on
    the page image the lines were found on, so that image is copied next
    to the XML as ``page.image_name``, unless it is the uploaded file
    itself (a single image), given as ``None``.

    Args:
        document: The transcription.
        folder: Where to write the files.
        images: Per page, the page image to include, or ``None``.
    """
    stem = Path(document.name).stem or "document"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}.txt").write_text(document.text, encoding="utf-8")
    (folder / f"{stem}.json").write_text(to_json(document), encoding="utf-8")
    xml_dir = folder / "page_xml"
    xml_dir.mkdir(exist_ok=True)
    for index, page in enumerate(document.pages):
        (xml_dir / f"page_{page.page:04d}.xml").write_bytes(
            page_xml(page, document.model.label)
        )
        image = images[index] if index < len(images) else None
        if image is not None:
            shutil.copyfile(image, xml_dir / page.image_name)


def extract_inputs(
    archive: BinaryIO | str | os.PathLike, input_dir: Path
) -> list[Path]:
    """Extract the supported files of a ZIP archive, in a stable order.

    macOS resource-fork files (``._name``) are left out.

    Raises:
        HandwritingError: If the archive cannot be read.
    """
    try:
        with zipfile.ZipFile(archive, "r") as source:
            files = extract_zip_safely(
                source, input_dir, allowed_extensions=set(INPUT_EXTENSIONS)
            )
    except zipfile.BadZipFile as error:
        raise HandwritingError(f"Not a valid ZIP archive: {error}") from error
    files = [path for path in files if not path.name.startswith("._")]
    files.sort(key=lambda path: str(path.relative_to(input_dir)).casefold())
    return files


def zip_folder(folder: Path) -> bytes:
    """Return a ZIP archive of a folder's files, with relative names."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for root, _dirs, names in sorted(os.walk(folder)):
            for name in sorted(names):
                path = Path(root) / name
                archive.write(path, path.relative_to(folder))
    return buffer.getvalue()
