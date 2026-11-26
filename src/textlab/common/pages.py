"""Documents as page images, and ZIP archives in and out.

Tools that read a page at a time (historic handwriting, Hugging Face OCR
models) first turn every document into PNG page images: a PDF is rendered
with ``pdftoppm``, and every frame of an image file (a TIFF can have
several) becomes a page. Batches come as ZIP archives and go back as ZIP
archives.

Errors are :class:`PageError`, with a message for users; features re-raise
them as their own error type where they need one.
"""

from __future__ import annotations

import io
import os
import subprocess
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import BinaryIO

from textlab.common.upload_safety import extract_zip_safely

#: File types read, alone or inside a ZIP archive.
INPUT_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


class PageError(ValueError):
    """A document or archive could not be read, with a message for users."""


def page_images(input_path: Path, out_dir: Path, dpi: int) -> list[Path]:
    """Return a file's pages as PNG images.

    Args:
        input_path: A PDF or an image file.
        out_dir: Where to write the images; created if needed.
        dpi: Resolution a PDF is rendered at. Image files keep their own
            pixels.

    Returns:
        The page images, in page order.

    Raises:
        PageError: If the file cannot be read or has no pages.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if input_path.suffix.lower() == ".pdf":
        result = subprocess.run(
            [
                "pdftoppm",
                "-r",
                str(dpi),
                "-png",
                str(input_path),
                str(out_dir / "page"),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise PageError(
                f"Could not read the PDF {input_path.name}: "
                f"{result.stderr.strip()[-500:]}"
            )
        # pdftoppm pads page numbers to the same width, so names sort.
        images = sorted(out_dir.glob("page-*.png"))
    else:
        images = _image_frames(input_path, out_dir)
    if not images:
        raise PageError(f"{input_path.name} has no pages.")
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
        raise PageError(
            f"Could not read the image {input_path.name}: {error}"
        ) from error


def extract_inputs(
    archive: BinaryIO | str | os.PathLike,
    input_dir: Path,
    extensions: Iterable[str] = INPUT_EXTENSIONS,
) -> list[Path]:
    """Extract the supported files of a ZIP archive, in a stable order.

    macOS resource-fork files (``._name``) are left out.

    Args:
        archive: The ZIP archive, as a file object or path.
        input_dir: Where to extract to.
        extensions: The file types to extract, with their dot.

    Returns:
        The extracted files, sorted by their path in the archive.

    Raises:
        PageError: If the archive cannot be read.
    """
    try:
        with zipfile.ZipFile(archive, "r") as source:
            files = extract_zip_safely(
                source, input_dir, allowed_extensions=set(extensions)
            )
    except zipfile.BadZipFile as error:
        raise PageError(f"Not a valid ZIP archive: {error}") from error
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
