"""ZIP archives in and out: the files of a batch, and its results."""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from typing import BinaryIO

from textlab.common.upload_safety import extract_zip_safely

#: File types OCR reads, alone or inside a ZIP archive.
INPUT_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")


def extract_inputs(
    archive: BinaryIO | str | os.PathLike, input_dir: Path
) -> list[Path]:
    """Extract the supported files of a ZIP archive, in a stable order.

    macOS resource-fork files (``._name``) are left out.

    Args:
        archive: The ZIP archive, as a file object or path.
        input_dir: Where to extract the files.

    Returns:
        The extracted files, sorted by their path in the archive.
    """
    with zipfile.ZipFile(archive, "r") as source:
        files = extract_zip_safely(
            source, input_dir, allowed_extensions=set(INPUT_EXTENSIONS)
        )
    files = [path for path in files if not path.name.startswith("._")]
    files.sort(key=lambda path: str(path.relative_to(input_dir)).casefold())
    return files


def zip_folder(folder: Path) -> bytes:
    """Return a ZIP archive of a folder's files, with relative names."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for root, _dirs, names in os.walk(folder):
            for name in names:
                path = Path(root) / name
                archive.write(path, path.relative_to(folder))
    return buffer.getvalue()
